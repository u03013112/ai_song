"""Voice conversion using RVC models and instrumental pitch shifting."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)


def _patch_torch_load() -> None:
    """Patch torch.load for PyTorch 2.6+ where weights_only defaults to True.

    fairseq and rvc-python use torch.load without weights_only=False,
    which breaks on PyTorch >=2.6. This patches the default back to False.
    """
    import functools

    import torch

    _original_load = torch.load

    @functools.wraps(_original_load)
    def _patched_load(*args: object, **kwargs: object) -> object:
        if "weights_only" not in kwargs:
            kwargs["weights_only"] = False
        return _original_load(*args, **kwargs)

    torch.load = _patched_load


_patch_torch_load()

DEFAULT_OUTPUT_DIR = Path("output/converted")


class ConversionError(Exception):
    """Failed to convert voice."""


def _detect_rvc_version(model_path: Path) -> str:
    """Detect RVC model version (v1 or v2) from checkpoint weights.

    v1 models use 256-dim embedding, v2 models use 768-dim.
    """
    import torch

    cpt = torch.load(str(model_path), map_location="cpu", weights_only=False)
    weight = cpt.get("weight", {})
    emb_key = "enc_p.emb_phone.weight"
    if emb_key in weight:
        dim = weight[emb_key].shape[1]
        return "v1" if dim == 256 else "v2"
    return "v2"


@dataclass
class ConvertConfig:
    """Configuration for voice conversion.

    Attributes:
        model_path: Path to the RVC model (.pth file).
        index_path: Path to the feature index (.index file), optional.
        transpose: Vocal pitch shift in semitones (-12 to +12).
        f0_method: Pitch extraction method (rmvpe, harvest, crepe).
        index_rate: Feature retrieval ratio (0.0-1.0). Higher = closer
            to training voice timbre.
        filter_radius: Median filter radius for pitch (0-7).
        rms_mix_rate: Volume envelope mix ratio (0.0-1.0).
            0 = use converted voice envelope, 1 = use original envelope.
        protect: Consonant protection ratio (0.0-0.5).
            Higher = more protection for breath/plosive sounds.
        instrumental_shift: Instrumental pitch shift in semitones.
            Safe range: -2 to +2.
        device: Inference device (mps, cpu, cuda).
    """

    model_path: Path = field(default_factory=lambda: Path("model.pth"))
    index_path: Path | None = None
    transpose: int = 0
    f0_method: str = "rmvpe"
    index_rate: float = 0.7
    filter_radius: int = 3
    rms_mix_rate: float = 1.0
    protect: float = 0.33
    instrumental_shift: int = 0
    device: str = "mps"


def convert_vocals(
    input_path: Path,
    output_path: Path,
    config: ConvertConfig,
) -> Path:
    """Convert vocals using an RVC model.

    Args:
        input_path: Path to the input vocal audio (WAV).
        output_path: Path to save converted vocal audio.
        config: Voice conversion configuration.

    Returns:
        Path to the converted audio file.

    Raises:
        ConversionError: If conversion fails.
    """
    if not input_path.exists():
        raise ConversionError(f"Input file not found: {input_path}")

    model_path = Path(config.model_path)
    if not model_path.exists():
        raise ConversionError(f"Model file not found: {model_path}")

    index_str = ""
    if config.index_path is not None:
        index_p = Path(config.index_path)
        if not index_p.exists():
            logger.warning("Index file not found: %s (proceeding without)", index_p)
        else:
            index_str = str(index_p)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        from rvc_python.infer import RVCInference

        logger.info("Loading RVC model: %s (device=%s)", model_path.name, config.device)
        start = time.time()

        rvc = RVCInference(device=config.device)

        version = _detect_rvc_version(model_path)
        logger.info("Detected RVC %s model", version)
        rvc.load_model(str(model_path), version=version, index_path=index_str)
        rvc.set_params(
            f0up_key=config.transpose,
            f0method=config.f0_method,
            index_rate=config.index_rate,
            filter_radius=config.filter_radius,
            rms_mix_rate=config.rms_mix_rate,
            protect=config.protect,
        )

        logger.info(
            "Converting: transpose=%+d, f0_method=%s",
            config.transpose,
            config.f0_method,
        )

        model_info = rvc.models[rvc.current_model]
        file_index = model_info.get("index", "")

        audio_opt = rvc.vc.vc_single(
            sid=0,
            input_audio_path=str(input_path),
            f0_up_key=config.transpose,
            f0_method=config.f0_method,
            file_index=file_index,
            index_rate=config.index_rate,
            filter_radius=config.filter_radius,
            resample_sr=0,
            rms_mix_rate=config.rms_mix_rate,
            protect=config.protect,
            f0_file="",
            file_index2="",
        )

        if isinstance(audio_opt, tuple):
            error_msg = audio_opt[0] if audio_opt else "Unknown error"
            raise ConversionError(f"RVC inference failed: {error_msg}")

        sf.write(str(output_path), audio_opt, rvc.vc.tgt_sr)

        elapsed = time.time() - start
        logger.info("Conversion complete in %.1fs: %s", elapsed, output_path)
        return output_path

    except ImportError as e:
        raise ConversionError(
            "rvc-python not installed. Run: pip install rvc-python"
        ) from e
    except Exception as e:
        raise ConversionError(f"Voice conversion failed: {e}") from e


def shift_instrumental(
    input_path: Path,
    output_path: Path,
    semitones: int,
) -> Path:
    """Shift instrumental pitch using pedalboard.

    Args:
        input_path: Path to the instrumental audio (WAV).
        output_path: Path to save pitch-shifted audio.
        semitones: Number of semitones to shift (-12 to +12).
            Safe range is -2 to +2 for minimal artifacts.

    Returns:
        Path to the pitch-shifted audio file.

    Raises:
        ConversionError: If pitch shifting fails.
    """
    if not input_path.exists():
        raise ConversionError(f"Input file not found: {input_path}")

    if semitones == 0:
        logger.info("Instrumental shift is 0, copying as-is: %s", input_path)
        import shutil

        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(input_path, output_path)
        return output_path

    if abs(semitones) > 2:
        logger.warning(
            "Instrumental shift %+d exceeds safe range (±2). "
            "Expect some quality degradation.",
            semitones,
        )

    try:
        from pedalboard import PitchShift
        from pedalboard.io import AudioFile

        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info("Shifting instrumental by %+d semitones: %s", semitones, input_path)
        start = time.time()

        with AudioFile(str(input_path)) as f:
            audio = f.read(f.frames)
            sample_rate = f.samplerate

        board = PitchShift(semitones=semitones)
        shifted = board(audio, sample_rate)

        sf.write(str(output_path), shifted.T, sample_rate)

        elapsed = time.time() - start
        logger.info("Instrumental shift complete in %.1fs: %s", elapsed, output_path)
        return output_path

    except ImportError as e:
        raise ConversionError(
            "pedalboard not installed. Run: pip install pedalboard"
        ) from e
    except Exception as e:
        raise ConversionError(f"Instrumental pitch shift failed: {e}") from e


def convert_with_strategy(
    vocals_path: Path,
    instrumental_path: Path,
    output_dir: Path,
    config: ConvertConfig,
) -> tuple[Path, Path]:
    """Convert vocals and optionally shift instrumental using the hybrid strategy.

    When total pitch difference is large, splits the shift between
    vocal transpose and instrumental pitch shift for best quality:
    - Vocal transpose: safe within ±4 semitones
    - Instrumental shift: safe within ±2 semitones

    Args:
        vocals_path: Path to separated vocals (WAV).
        instrumental_path: Path to separated instrumental (WAV).
        output_dir: Directory for output files.
        config: Conversion configuration. Uses transpose for total desired
            shift and instrumental_shift for explicit instrumental shift.

    Returns:
        Tuple of (converted_vocals_path, shifted_instrumental_path).

    Raises:
        ConversionError: If conversion fails.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = vocals_path.stem.replace("(Vocals)", "").strip("_ ")
    converted_vocals_path = output_dir / f"{stem}_converted.wav"
    shifted_instrumental_path = output_dir / f"{stem}_instrumental.wav"

    logger.info(
        "Vocal transpose: %+d semitones, Instrumental shift: %+d semitones",
        config.transpose,
        config.instrumental_shift,
    )

    convert_vocals(vocals_path, converted_vocals_path, config)

    shift_instrumental(
        instrumental_path,
        shifted_instrumental_path,
        config.instrumental_shift,
    )

    return converted_vocals_path, shifted_instrumental_path


def main() -> None:
    """CLI entry point for voice conversion."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Voice conversion with RVC and instrumental pitch shifting"
    )
    parser.add_argument("input", type=Path, help="Input vocal audio (WAV)")
    parser.add_argument(
        "--model", type=Path, required=True, help="RVC model path (.pth)"
    )
    parser.add_argument(
        "--index", type=Path, default=None, help="Feature index path (.index)"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--transpose",
        type=int,
        default=0,
        help="Vocal pitch shift in semitones, e.g. -4 for lower (default: 0)",
    )
    parser.add_argument(
        "--f0-method",
        choices=["rmvpe", "harvest", "crepe"],
        default="rmvpe",
        help="Pitch extraction method (default: rmvpe)",
    )
    parser.add_argument(
        "--index-rate",
        type=float,
        default=0.7,
        help="Feature retrieval ratio 0.0-1.0 (default: 0.7)",
    )
    parser.add_argument(
        "--protect",
        type=float,
        default=0.33,
        help="Consonant protection 0.0-0.5 (default: 0.33)",
    )
    parser.add_argument(
        "--instrumental",
        type=Path,
        default=None,
        help="Instrumental audio for pitch shifting (WAV)",
    )
    parser.add_argument(
        "--instrumental-shift",
        type=int,
        default=0,
        help="Instrumental pitch shift in semitones (default: 0)",
    )
    parser.add_argument(
        "--device",
        default="mps",
        help="Inference device: mps, cpu, cuda (default: mps)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    config = ConvertConfig(
        model_path=args.model,
        index_path=args.index,
        transpose=args.transpose,
        f0_method=args.f0_method,
        index_rate=args.index_rate,
        protect=args.protect,
        instrumental_shift=args.instrumental_shift,
        device=args.device,
    )

    if args.instrumental:
        converted, shifted = convert_with_strategy(
            args.input, args.instrumental, args.output_dir, config
        )
        print(f"Converted vocals: {converted}")
        print(f"Shifted instrumental: {shifted}")
    else:
        output_path = args.output_dir / f"{args.input.stem}_converted.wav"
        convert_vocals(args.input, output_path, config)
        print(f"Converted: {output_path}")


if __name__ == "__main__":
    main()
