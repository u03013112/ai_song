"""Automatic quality evaluation for voice-converted singing audio."""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ai_song.transpose import analyze_f0

logger = logging.getLogger(__name__)

_utmos_model = None


class EvaluationError(Exception):
    """Failed to evaluate converted singing quality."""


@dataclass
class PitchAccuracyResult:
    """Pitch accuracy evaluation results.

    Attributes:
        rpa: Raw Pitch Accuracy (fraction of frames within 50 cents).
        rca: Raw Chroma Accuracy (RPA ignoring octave errors).
        mean_deviation_cents: Mean absolute pitch deviation in cents.
        median_deviation_cents: Median absolute pitch deviation in cents.
        max_deviation_cents: Maximum absolute pitch deviation in cents.
        pitch_drift_std_cents: Standard deviation of signed pitch deviation.
        voiced_ratio_ref: Fraction of voiced frames in reference.
        voiced_ratio_est: Fraction of voiced frames in estimate.
    """

    rpa: float
    rca: float
    mean_deviation_cents: float
    median_deviation_cents: float
    max_deviation_cents: float
    pitch_drift_std_cents: float
    voiced_ratio_ref: float
    voiced_ratio_est: float


@dataclass
class NaturalnessResult:
    """UTMOSv2 naturalness evaluation results.

    Attributes:
        utmos_score: UTMOSv2 MOS prediction (1.0-5.0 scale).
        quality_label: Human-readable quality label.
    """

    utmos_score: float
    quality_label: str


@dataclass
class EvaluationReport:
    """Complete evaluation report.

    Attributes:
        pitch: Pitch accuracy results (None if no reference provided).
        naturalness: UTMOSv2 naturalness results.
        composite_score: Weighted composite score (0-100).
    """

    pitch: PitchAccuracyResult | None
    naturalness: NaturalnessResult
    composite_score: float


def _zero_pitch_result() -> PitchAccuracyResult:
    return PitchAccuracyResult(
        rpa=0.0,
        rca=0.0,
        mean_deviation_cents=0.0,
        median_deviation_cents=0.0,
        max_deviation_cents=0.0,
        pitch_drift_std_cents=0.0,
        voiced_ratio_ref=0.0,
        voiced_ratio_est=0.0,
    )


def _patch_autocast_for_cpu() -> None:
    import contextlib
    import importlib

    import torch.cuda.amp as cuda_amp

    if not hasattr(cuda_amp, "_original_autocast"):
        cuda_amp._original_autocast = cuda_amp.autocast  # type: ignore[attr-defined]
        cuda_amp.autocast = contextlib.nullcontext  # type: ignore[attr-defined]

    for mod_name in ("utmosv2._core.model._common", "utmosv2.runner._inference"):
        try:
            mod = importlib.import_module(mod_name)
            if hasattr(mod, "autocast"):
                mod.autocast = contextlib.nullcontext  # type: ignore[attr-defined]
        except ImportError:
            pass


def _get_utmos_model():
    global _utmos_model
    if _utmos_model is None:
        import torch

        if not torch.cuda.is_available():
            _patch_autocast_for_cpu()

        import utmosv2

        _utmos_model = utmosv2.create_model(pretrained=True)
        logger.info("UTMOSv2 model loaded")
    return _utmos_model


def _quality_label(score: float) -> str:
    if score >= 4.0:
        return "Excellent"
    if score >= 3.5:
        return "Good"
    if score >= 3.0:
        return "Fair"
    if score >= 2.5:
        return "Poor"
    return "Bad"


def evaluate_pitch_accuracy(
    reference_path: Path,
    converted_path: Path,
    f0_method: str = "fcpe",
    transpose: int = 0,
) -> PitchAccuracyResult:
    """Evaluate F0 pitch accuracy between reference and converted singing."""
    try:
        from mir_eval import melody
        from scipy.interpolate import interp1d

        ref_analysis = analyze_f0(reference_path, method=f0_method)
        est_analysis = analyze_f0(converted_path, method=f0_method)

        ref_f0 = ref_analysis.f0_hz.astype(np.float64, copy=False)
        est_f0 = est_analysis.f0_hz.astype(np.float64, copy=False)

        if transpose != 0:
            shift_ratio = 2.0 ** (transpose / 12.0)
            voiced_mask = ref_f0 > 0.0
            ref_f0 = ref_f0.copy()
            ref_f0[voiced_mask] *= shift_ratio

        if len(ref_f0) == 0 or len(est_f0) == 0:
            raise EvaluationError("Empty F0 curve")

        if len(est_f0) != len(ref_f0):
            src_idx = np.arange(len(est_f0), dtype=np.float64)
            dst_idx = np.linspace(
                0.0,
                len(est_f0) - 1,
                num=len(ref_f0),
                dtype=np.float64,
            )
            interp = interp1d(
                src_idx,
                est_f0,
                kind="linear",
                bounds_error=False,
                fill_value=(float(est_f0[0]), float(est_f0[-1])),
            )
            est_f0 = interp(dst_idx).astype(np.float64, copy=False)

        hop_sec = 0.01
        ref_time = np.arange(len(ref_f0), dtype=np.float64) * hop_sec
        est_time = np.arange(len(est_f0), dtype=np.float64) * hop_sec

        metrics = melody.evaluate(ref_time, ref_f0, est_time, est_f0)
        rpa = float(metrics.get("Raw Pitch Accuracy", 0.0))
        rca = float(metrics.get("Raw Chroma Accuracy", 0.0))

        voiced_ref = ref_f0 > 0.0
        voiced_est = est_f0 > 0.0
        both_voiced = voiced_ref & voiced_est

        voiced_ratio_ref = float(np.mean(voiced_ref))
        voiced_ratio_est = float(np.mean(voiced_est))

        if not np.any(both_voiced):
            raise EvaluationError("No overlapping voiced frames")

        ref_voiced = ref_f0[both_voiced]
        est_voiced = est_f0[both_voiced]
        deviation_cents = 1200.0 * np.log2(est_voiced / ref_voiced)
        abs_dev = np.abs(deviation_cents)

        return PitchAccuracyResult(
            rpa=rpa,
            rca=rca,
            mean_deviation_cents=float(np.mean(abs_dev)),
            median_deviation_cents=float(np.median(abs_dev)),
            max_deviation_cents=float(np.max(abs_dev)),
            pitch_drift_std_cents=float(np.std(deviation_cents)),
            voiced_ratio_ref=voiced_ratio_ref,
            voiced_ratio_est=voiced_ratio_est,
        )
    except Exception as e:
        logger.warning("Pitch evaluation failed: %s", e)
        return _zero_pitch_result()


def evaluate_utmos(audio_path: Path) -> NaturalnessResult:
    """Evaluate naturalness with UTMOSv2."""
    try:
        import torch

        device = "cpu" if not torch.cuda.is_available() else "cuda:0"
        model = _get_utmos_model()
        score = float(
            model.predict(
                input_path=str(audio_path),
                device=device,
                verbose=False,
                num_workers=0,
            )
        )
        return NaturalnessResult(utmos_score=score, quality_label=_quality_label(score))
    except Exception as e:
        logger.warning("UTMOSv2 evaluation failed: %s", e)
        return NaturalnessResult(utmos_score=0.0, quality_label="Error")


def evaluate_all(
    converted_path: Path,
    reference_path: Path | None = None,
    f0_method: str = "fcpe",
    transpose: int = 0,
) -> EvaluationReport:
    """Run full evaluation and compute a weighted composite score."""
    naturalness = evaluate_utmos(converted_path)
    utmos_normalized = (naturalness.utmos_score - 1.0) / 4.0 * 100.0
    utmos_normalized = float(np.clip(utmos_normalized, 0.0, 100.0))

    pitch: PitchAccuracyResult | None = None
    composite_score = utmos_normalized

    if reference_path is not None:
        pitch = evaluate_pitch_accuracy(
            reference_path,
            converted_path,
            f0_method=f0_method,
            transpose=transpose,
        )
        rpa_score = pitch.rpa * 100.0
        mean_deviation_normalized = min(pitch.mean_deviation_cents / 100.0, 1.0)
        composite_score = (
            0.4 * utmos_normalized
            + 0.3 * rpa_score
            + 0.3 * (1.0 - mean_deviation_normalized) * 100.0
        )

    return EvaluationReport(
        pitch=pitch,
        naturalness=naturalness,
        composite_score=float(np.clip(composite_score, 0.0, 100.0)),
    )


def main() -> None:
    """CLI entry point for quality evaluation."""
    parser = argparse.ArgumentParser(
        description="Automatic quality evaluation for voice-converted singing"
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Converted singing audio path (WAV)",
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=None,
        help="Reference original singing audio path (WAV)",
    )
    parser.add_argument(
        "--f0-method",
        type=str,
        default="fcpe",
        choices=["fcpe", "crepe"],
        help="F0 extraction method (default: fcpe)",
    )
    parser.add_argument(
        "--transpose",
        type=int,
        default=0,
        help="Transpose applied during conversion (compensates pitch comparison)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    report = evaluate_all(
        converted_path=args.input,
        reference_path=args.reference,
        f0_method=args.f0_method,
        transpose=args.transpose,
    )

    print("=== Quality Evaluation ===")
    print(
        "UTMOSv2 Score: "
        f"{report.naturalness.utmos_score:.2f} / 5.0 "
        f"({report.naturalness.quality_label})"
    )

    if report.pitch is not None:
        print(f"Pitch Accuracy (RPA): {report.pitch.rpa * 100:.1f}%")
        print(f"Pitch Accuracy (RCA): {report.pitch.rca * 100:.1f}%")
        print(f"Mean Pitch Deviation: {report.pitch.mean_deviation_cents:.1f} cents")
        print(f"Max Pitch Deviation: {report.pitch.max_deviation_cents:.1f} cents")
        print(f"Pitch Drift (std): {report.pitch.pitch_drift_std_cents:.1f} cents")

    print(f"Composite Score: {report.composite_score:.1f} / 100")


if __name__ == "__main__":
    main()
