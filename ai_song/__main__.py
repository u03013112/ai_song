"""Full AI song pipeline: download → separate → convert → mix."""

from __future__ import annotations

import logging
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

ICLOUD_OUTPUT = Path.home() / "Library/Mobile Documents/com~apple~CloudDocs/ai_song_output"


@dataclass
class PipelineConfig:
    """End-to-end pipeline configuration.

    Attributes:
        model_path: Path to the RVC voice model (.pth).
        index_path: Optional path to the feature index (.index).
        transpose: Vocal pitch shift in semitones.
        instrumental_shift: Instrumental pitch shift in semitones.
        f0_method: Pitch extraction method (fcpe, rmvpe, crepe).
        index_rate: Feature retrieval ratio 0.0-1.0 (0 disables faiss).
        output_dir: Base output directory.
        icloud_copy: Copy final output to iCloud Drive.
        output_name: Optional custom name for the final output file.
    """

    model_path: Path = field(default_factory=lambda: Path("model.pth"))
    index_path: Path | None = None
    transpose: int = 0
    instrumental_shift: int = 0
    f0_method: str = "fcpe"
    index_rate: float = 0.0
    output_dir: Path = field(default_factory=lambda: Path("output"))
    icloud_copy: bool = True
    output_name: str | None = None


def run_pipeline(url: str, config: PipelineConfig) -> Path:
    """Run the full pipeline: download → separate → convert → mix.

    Args:
        url: Video/audio URL (Bilibili, YouTube, etc.).
        config: Pipeline configuration.

    Returns:
        Path to the final mixed audio file.
    """
    from ai_song.convert import ConvertConfig, convert_vocals
    from ai_song.download import download_audio
    from ai_song.mix import MixConfig, mix_tracks
    from ai_song.separate import separate_vocals

    pipeline_start = time.time()

    logger.info("=" * 60)
    logger.info("STAGE 1: Download")
    logger.info("=" * 60)
    wav_path = download_audio(url, config.output_dir / "downloads")
    song_stem = wav_path.stem
    logger.info("Downloaded: %s", wav_path)

    logger.info("=" * 60)
    logger.info("STAGE 2: Vocal Separation")
    logger.info("=" * 60)
    vocals_path, instrumental_path = separate_vocals(
        wav_path, config.output_dir / "separated"
    )
    logger.info("Vocals: %s", vocals_path)
    logger.info("Instrumental: %s", instrumental_path)

    logger.info("=" * 60)
    logger.info("STAGE 3: Voice Conversion")
    logger.info("=" * 60)
    convert_config = ConvertConfig(
        model_path=config.model_path,
        index_path=config.index_path,
        transpose=config.transpose,
        f0_method=config.f0_method,
        index_rate=config.index_rate,
        instrumental_shift=config.instrumental_shift,
    )
    converted_path = config.output_dir / "converted" / f"{song_stem}_converted.wav"
    convert_vocals(vocals_path, converted_path, convert_config)
    logger.info("Converted: %s", converted_path)

    logger.info("=" * 60)
    logger.info("STAGE 4: Mixing")
    logger.info("=" * 60)
    final_name = config.output_name or f"{song_stem}_final.wav"
    if not final_name.endswith(".wav"):
        final_name += ".wav"
    mixed_path = config.output_dir / "mixed" / final_name
    mix_tracks(converted_path, instrumental_path, mixed_path, MixConfig())
    logger.info("Mixed: %s", mixed_path)

    if config.icloud_copy:
        ICLOUD_OUTPUT.mkdir(parents=True, exist_ok=True)
        icloud_path = ICLOUD_OUTPUT / final_name
        shutil.copy2(mixed_path, icloud_path)
        logger.info("Copied to iCloud: %s", icloud_path)

    elapsed = time.time() - pipeline_start
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE in %.1fs", elapsed)
    logger.info("Output: %s", mixed_path)
    logger.info("=" * 60)

    return mixed_path


def main() -> None:
    """CLI entry point for the full pipeline."""
    import argparse

    parser = argparse.ArgumentParser(
        description="AI Song: download → separate → convert → mix"
    )
    parser.add_argument("url", help="Video/audio URL (Bilibili, YouTube, etc.)")
    parser.add_argument(
        "--model", type=Path, required=True, help="RVC model path (.pth)"
    )
    parser.add_argument(
        "--index", type=Path, default=None, help="Feature index path (.index)"
    )
    parser.add_argument(
        "--transpose",
        type=int,
        default=0,
        help="Vocal pitch shift in semitones (default: 0)",
    )
    parser.add_argument(
        "--instrumental-shift",
        type=int,
        default=0,
        help="Instrumental pitch shift in semitones (default: 0)",
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
        help="Feature retrieval ratio 0.0-1.0, 0 disables faiss (default: 0)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Base output directory (default: output/)",
    )
    parser.add_argument(
        "--name",
        default=None,
        help="Custom name for final output file",
    )
    parser.add_argument(
        "--no-icloud",
        action="store_true",
        help="Do not copy output to iCloud Drive",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    config = PipelineConfig(
        model_path=args.model,
        index_path=args.index,
        transpose=args.transpose,
        instrumental_shift=args.instrumental_shift,
        f0_method=args.f0_method,
        index_rate=args.index_rate,
        output_dir=args.output_dir,
        icloud_copy=not args.no_icloud,
        output_name=args.name,
    )

    result = run_pipeline(args.url, config)
    print(f"\n✓ Final output: {result}")


if __name__ == "__main__":
    main()
