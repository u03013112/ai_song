"""Voice conversion using Applio RVC backend and instrumental pitch shifting."""

from __future__ import annotations

import logging
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

APPLIO_DIR = Path(__file__).resolve().parent.parent / "third_party" / "Applio"

DEFAULT_OUTPUT_DIR = Path("output/converted")


class ConversionError(Exception):
    """Failed to convert voice."""


@dataclass
class ConvertConfig:
    """Configuration for voice conversion.

    Attributes:
        model_path: Path to the RVC model (.pth file).
        index_path: Path to the feature index (.index file), optional.
        transpose: Vocal pitch shift in semitones (-12 to +12).
        f0_method: Pitch extraction method (fcpe, rmvpe, crepe).
        index_rate: Feature retrieval ratio (0.0-1.0). Higher = closer
            to training voice timbre.
        protect: Consonant protection ratio (0.0-0.5).
            Higher = more protection for breath/plosive sounds.
        volume_envelope: RMS mix rate (0.0-1.0).
            0 = use converted voice envelope, 1 = use original envelope.
        instrumental_shift: Instrumental pitch shift in semitones.
            Safe range: -2 to +2.
        f0_autotune: Enable F0 autotune (snap to nearest note).
        f0_autotune_strength: Autotune correction strength (0.0-1.0).
        split_audio: Split long audio into chunks for processing.
        clean_audio: Apply noise reduction after conversion.
        clean_strength: Noise reduction strength (0.0-1.0).
        embedder_model: Feature embedder (contentvec, chinese-hubert-base, etc.).
    """

    model_path: Path = field(default_factory=lambda: Path("model.pth"))
    index_path: Path | None = None
    transpose: int = 0
    f0_method: str = "fcpe"
    index_rate: float = 0.0
    protect: float = 0.33
    volume_envelope: float = 1.0
    instrumental_shift: int = 0
    f0_autotune: bool = False
    f0_autotune_strength: float = 1.0
    split_audio: bool = False
    clean_audio: bool = False
    clean_strength: float = 0.5
    embedder_model: str = "contentvec"


def _get_voice_converter():
    """Lazily import and return Applio VoiceConverter singleton."""
    original_dir = os.getcwd()
    applio_str = str(APPLIO_DIR)

    if applio_str not in sys.path:
        sys.path.insert(0, applio_str)

    os.chdir(applio_str)
    try:
        from rvc.infer.infer import VoiceConverter
        vc = VoiceConverter()
        logger.info("Applio RVC engine loaded (device=%s)", vc.config.device)
        return vc
    finally:
        os.chdir(original_dir)


_vc_instance = None


def _ensure_vc():
    """Get or create the global VoiceConverter instance."""
    global _vc_instance
    if _vc_instance is None:
        _vc_instance = _get_voice_converter()
    return _vc_instance


def convert_vocals(
    input_path: Path,
    output_path: Path,
    config: ConvertConfig,
) -> Path:
    """Convert vocals using an RVC model via Applio backend.

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
        logger.info(
            "Converting: model=%s, transpose=%+d, f0=%s, index_rate=%.2f, protect=%.2f",
            model_path.name,
            config.transpose,
            config.f0_method,
            config.index_rate,
            config.protect,
        )
        start = time.time()

        vc = _ensure_vc()

        original_dir = os.getcwd()
        os.chdir(str(APPLIO_DIR))
        try:
            vc.convert_audio(
                audio_input_path=str(input_path.resolve()),
                audio_output_path=str(output_path.resolve()),
                model_path=str(model_path.resolve()),
                index_path=index_str,
                pitch=config.transpose,
                f0_method=config.f0_method,
                index_rate=config.index_rate,
                volume_envelope=config.volume_envelope,
                protect=config.protect,
                split_audio=config.split_audio,
                f0_autotune=config.f0_autotune,
                f0_autotune_strength=config.f0_autotune_strength,
                proposed_pitch=False,
                proposed_pitch_threshold=155.0,
                embedder_model=config.embedder_model,
                clean_audio=config.clean_audio,
                clean_strength=config.clean_strength,
                export_format="WAV",
            )
        finally:
            os.chdir(original_dir)

        if not output_path.exists():
            raise ConversionError("Applio produced no output file")

        elapsed = time.time() - start
        info = sf.info(str(output_path))
        logger.info(
            "Conversion complete in %.1fs (%.1fs audio @ %dHz): %s",
            elapsed,
            info.duration,
            info.samplerate,
            output_path,
        )
        return output_path

    except ConversionError:
        raise
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
        description="Voice conversion with Applio RVC and instrumental pitch shifting"
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
        choices=["fcpe", "rmvpe", "crepe"],
        default="fcpe",
        help="Pitch extraction method (default: fcpe)",
    )
    parser.add_argument(
        "--index-rate",
        type=float,
        default=0.0,
        help="Feature retrieval ratio 0.0-1.0 (default: 0.0)",
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
        "--f0-autotune",
        action="store_true",
        help="Enable F0 autotune (snap to nearest note)",
    )
    parser.add_argument(
        "--split-audio",
        action="store_true",
        help="Split long audio into chunks for processing",
    )
    parser.add_argument(
        "--clean-audio",
        action="store_true",
        help="Apply noise reduction after conversion",
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
        f0_autotune=args.f0_autotune,
        split_audio=args.split_audio,
        clean_audio=args.clean_audio,
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
