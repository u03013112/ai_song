"""Vocal and instrumental separation."""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "model_bs_roformer_ep_368_sdr_12.9628.ckpt"
KARAOKE_MODEL = "mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt"
DEFAULT_OUTPUT_DIR = Path("output/separated")


class SeparationError(Exception):
    """Failed to separate audio stems."""


def separate_vocals(
    input_path: Path,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    model_name: str = DEFAULT_MODEL,
) -> tuple[Path, Path]:
    """Separate vocals from instrumental track.

    Args:
        input_path: Path to the input audio file (WAV format).
        output_dir: Directory to write separated tracks.
        model_name: Model filename for audio-separator.

    Returns:
        Tuple of (vocals_path, instrumental_path).

    Raises:
        SeparationError: If separation fails.
    """
    if not input_path.exists():
        raise SeparationError(f"Input file not found: {input_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        from audio_separator.separator import Separator

        separator = Separator(
            output_dir=str(output_dir),
            output_format="WAV",
        )
        separator.load_model(model_filename=model_name)
        output_files = separator.separate(str(input_path))

        logger.info("Separation complete: %s", output_files)

        # audio-separator returns filenames relative to output_dir
        if len(output_files) < 2:
            raise SeparationError(
                f"Expected 2 output files, got {len(output_files)}: {output_files}"
            )

        vocals_path = None
        instrumental_path = None
        for f in output_files:
            fp = output_dir / Path(f).name
            name_lower = fp.name.lower()
            if "vocal" in name_lower:
                vocals_path = fp
            elif "instrument" in name_lower:
                instrumental_path = fp

        if vocals_path is None:
            vocals_path = output_dir / Path(output_files[0]).name
        if instrumental_path is None:
            instrumental_path = output_dir / Path(output_files[1]).name

        return vocals_path, instrumental_path

    except ImportError as e:
        raise SeparationError(
            "audio-separator not installed. Run: pip install audio-separator"
        ) from e
    except Exception as e:
        raise SeparationError(f"Separation failed: {e}") from e


def separate_karaoke(
    vocals_path: Path,
    output_dir: Path,
    model_name: str = KARAOKE_MODEL,
) -> tuple[Path, Path]:
    """Separate vocals into lead and backing vocals.

    Args:
        vocals_path: Path to the input vocal audio file (WAV format).
        output_dir: Directory to write separated karaoke stems.
        model_name: Karaoke model filename for audio-separator.

    Returns:
        Tuple of (lead_vocals_path, backing_vocals_path).

    Raises:
        SeparationError: If karaoke separation fails.
    """
    if not vocals_path.exists():
        raise SeparationError(f"Input file not found: {vocals_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    try:
        from audio_separator.separator import Separator

        separator = Separator(
            output_dir=str(output_dir),
            output_format="WAV",
        )
        separator.load_model(model_filename=model_name)
        output_files = separator.separate(str(vocals_path))

        logger.info("Karaoke separation complete: %s", output_files)

        if len(output_files) < 2:
            raise SeparationError(
                f"Expected 2 output files, got {len(output_files)}: {output_files}"
            )

        lead_vocals_path = None
        backing_vocals_path = None
        for f in output_files:
            fp = output_dir / Path(f).name
            name_lower = fp.name.lower()
            # Karaoke model labels: (Vocals) = lead, (Instrumental) = backing
            if "instrument" in name_lower or "back" in name_lower:
                backing_vocals_path = fp
            elif "vocal" in name_lower or "lead" in name_lower:
                lead_vocals_path = fp

        if lead_vocals_path is None:
            lead_vocals_path = output_dir / Path(output_files[0]).name
        if backing_vocals_path is None:
            backing_vocals_path = output_dir / Path(output_files[1]).name

        return lead_vocals_path, backing_vocals_path

    except ImportError as e:
        raise SeparationError(
            "audio-separator not installed. Run: pip install audio-separator"
        ) from e
    except Exception as e:
        raise SeparationError(f"Karaoke separation failed: {e}") from e


def main() -> None:
    """CLI entry point for vocal separation."""
    import argparse

    parser = argparse.ArgumentParser(description="Separate vocals from instrumental")
    parser.add_argument("input", type=Path, help="Input audio file (WAV)")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"Model name (default: {DEFAULT_MODEL})",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    vocals, instrumental = separate_vocals(args.input, args.output_dir, args.model)
    print(f"Vocals: {vocals}")
    print(f"Instrumental: {instrumental}")


if __name__ == "__main__":
    main()
