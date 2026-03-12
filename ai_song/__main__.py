"""Full AI song pipeline: download → separate → convert → mix."""

from __future__ import annotations

import logging
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

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
        enable_backing: Enable backing vocal separation and conversion.
        auto_transpose: Analyze F0 and auto-recommend transpose values.
        evaluate: Run quality evaluation after mixing.
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
    enable_backing: bool = True
    auto_transpose: bool = False
    evaluate: bool = False


def _has_backing_vocals(path: Path, threshold_dbfs: float = -40.0) -> bool:
    """Check if backing vocals file has significant audio content."""
    import soundfile as sf

    audio, _ = sf.read(str(path), dtype="float32")
    rms = np.sqrt(np.mean(audio ** 2))
    dbfs = 20 * np.log10(rms + 1e-10)
    return dbfs > threshold_dbfs


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
    from ai_song.separate import separate_karaoke, separate_vocals

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

    lead_vocals_path = vocals_path
    backing_vocals_path: Path | None = None
    if config.enable_backing:
        logger.info("=" * 60)
        logger.info("STAGE 2.5: Karaoke Separation")
        logger.info("=" * 60)
        lead_vocals_path, backing_vocals_path = separate_karaoke(
            vocals_path,
            config.output_dir / "separated_karaoke",
        )
        logger.info("Lead vocals: %s", lead_vocals_path)
        logger.info("Backing vocals: %s", backing_vocals_path)
        if not _has_backing_vocals(backing_vocals_path):
            logger.info(
                "Backing vocals below threshold, skipping backing conversion: %s",
                backing_vocals_path,
            )
            backing_vocals_path = None

    transpose = config.transpose
    instrumental_shift = config.instrumental_shift
    if config.auto_transpose and transpose == 0:
        from ai_song.transpose import analyze_f0, recommend_transpose

        logger.info("=" * 60)
        logger.info("STAGE 2.8: F0 Analysis & Transpose Recommendation")
        logger.info("=" * 60)
        f0_analysis = analyze_f0(lead_vocals_path, method=config.f0_method)
        recommendation = recommend_transpose(f0_analysis)
        logger.info(
            "F0 median: %.1f Hz, range: %.1f semitones (P5-P95: %.1f-%.1f Hz)",
            f0_analysis.median_hz,
            f0_analysis.range_semitones,
            f0_analysis.min_hz,
            f0_analysis.max_hz,
        )
        logger.info(
            "Recommendation: vocal=%+d, instrumental=%+d (confidence=%.2f)",
            recommendation.vocal_transpose,
            recommendation.instrumental_shift,
            recommendation.confidence,
        )
        logger.info("Reason: %s", recommendation.reason)
        transpose = recommendation.vocal_transpose
        instrumental_shift = recommendation.instrumental_shift

    logger.info("=" * 60)
    logger.info("STAGE 3: Voice Conversion")
    logger.info("=" * 60)
    convert_config = ConvertConfig(
        model_path=config.model_path,
        index_path=config.index_path,
        transpose=transpose,
        f0_method=config.f0_method,
        index_rate=config.index_rate,
        instrumental_shift=instrumental_shift,
    )
    converted_path = config.output_dir / "converted" / f"{song_stem}_converted.wav"
    convert_vocals(lead_vocals_path, converted_path, convert_config)
    logger.info("Converted: %s", converted_path)

    converted_backing_path: Path | None = None
    if backing_vocals_path is not None:
        converted_backing_path = (
            config.output_dir / "converted" / f"{song_stem}_backing_converted.wav"
        )
        convert_vocals(backing_vocals_path, converted_backing_path, convert_config)
        logger.info("Converted backing: %s", converted_backing_path)

    logger.info("=" * 60)
    logger.info("STAGE 4: Mixing")
    logger.info("=" * 60)
    final_name = config.output_name or f"{song_stem}_final.wav"
    if not final_name.endswith(".wav"):
        final_name += ".wav"
    mixed_path = config.output_dir / "mixed" / final_name
    mix_tracks(
        converted_path,
        instrumental_path,
        mixed_path,
        backing_vocals_path=converted_backing_path,
        config=MixConfig(),
    )
    logger.info("Mixed: %s", mixed_path)

    if config.icloud_copy:
        ICLOUD_OUTPUT.mkdir(parents=True, exist_ok=True)
        icloud_path = ICLOUD_OUTPUT / final_name
        shutil.copy2(mixed_path, icloud_path)
        logger.info("Copied to iCloud: %s", icloud_path)

    if config.evaluate:
        from ai_song.evaluate import evaluate_all

        logger.info("=" * 60)
        logger.info("STAGE 5: Quality Evaluation")
        logger.info("=" * 60)
        report = evaluate_all(
            converted_path=converted_path,
            reference_path=lead_vocals_path,
            f0_method=config.f0_method,
            transpose=transpose,
        )
        logger.info(
            "UTMOSv2: %.2f / 5.0 (%s)",
            report.naturalness.utmos_score,
            report.naturalness.quality_label,
        )
        if report.pitch is not None:
            logger.info(
                "Pitch accuracy: RPA=%.1f%%, RCA=%.1f%%, mean deviation=%.1f cents",
                report.pitch.rpa * 100,
                report.pitch.rca * 100,
                report.pitch.mean_deviation_cents,
            )
        logger.info("Composite score: %.1f / 100", report.composite_score)

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
    parser.add_argument(
        "--no-backing",
        action="store_true",
        help="Skip backing vocal separation and conversion",
    )
    parser.add_argument(
        "--auto-transpose",
        action="store_true",
        help="Analyze F0 and auto-recommend transpose (ignored if --transpose is set)",
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Run quality evaluation after mixing (UTMOSv2 + pitch accuracy)",
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
        enable_backing=not args.no_backing,
        auto_transpose=args.auto_transpose,
        evaluate=args.evaluate,
    )

    result = run_pipeline(args.url, config)
    print(f"\n✓ Final output: {result}")


if __name__ == "__main__":
    main()
