"""Intelligent pitch transposition for voice conversion."""

from __future__ import annotations

import argparse
import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)


class TransposeError(Exception):
    pass


@dataclass
class F0Analysis:
    """F0 analysis results for a vocal track.

    Attributes:
        f0_hz: Raw F0 values in Hz (0 = unvoiced).
        times: Time stamps for each F0 value in seconds.
        sample_rate: Audio sample rate.
        voiced_f0: F0 values only for voiced segments (excludes 0s).
        median_hz: Median F0 of voiced segments.
        mean_hz: Mean F0 of voiced segments.
        min_hz: 5th percentile F0 (ignoring outliers).
        max_hz: 95th percentile F0 (ignoring outliers).
        range_semitones: Range in semitones between min and max.
    """

    f0_hz: np.ndarray
    times: np.ndarray
    sample_rate: int
    voiced_f0: np.ndarray
    median_hz: float
    mean_hz: float
    min_hz: float
    max_hz: float
    range_semitones: float


@dataclass
class TransposeRecommendation:
    """Recommended transpose settings.

    Attributes:
        vocal_transpose: Recommended vocal transpose in semitones.
        instrumental_shift: Recommended instrumental shift in semitones.
        confidence: Confidence score 0-1 (higher = more benefit from transposing).
        reason: Human-readable explanation.
        source_median_hz: Source vocal median F0.
        target_median_hz: Target after transposition.
    """

    vocal_transpose: int
    instrumental_shift: int
    confidence: float
    reason: str
    source_median_hz: float
    target_median_hz: float


def hz_to_midi(hz: float) -> float:
    """Convert Hz to MIDI note number."""
    if hz <= 0:
        raise ValueError("Frequency must be positive for MIDI conversion")
    return 69 + 12 * np.log2(hz / 440.0)


def midi_to_hz(midi: float) -> float:
    """Convert MIDI note number to Hz."""
    return 440.0 * 2 ** ((midi - 69) / 12)


def semitones_between(f1_hz: float, f2_hz: float) -> float:
    """Calculate interval in semitones between two frequencies."""
    if f1_hz <= 0 or f2_hz <= 0:
        raise ValueError("Frequencies must be positive for semitone calculation")
    return 12 * np.log2(f2_hz / f1_hz)


def _to_mono_float32(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        mono = audio
    else:
        mono = audio.mean(axis=1)
    return mono.astype(np.float32, copy=False)


def _safe_f0_stats(voiced_f0: np.ndarray) -> tuple[float, float, float, float, float]:
    if len(voiced_f0) == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0

    median_hz = float(np.median(voiced_f0))
    mean_hz = float(np.mean(voiced_f0))
    min_hz = float(np.percentile(voiced_f0, 5.0))
    max_hz = float(np.percentile(voiced_f0, 95.0))
    range_st = float(semitones_between(min_hz, max_hz)) if min_hz > 0 else 0.0
    return median_hz, mean_hz, min_hz, max_hz, range_st


def analyze_f0(audio_path: Path, method: str = "fcpe") -> F0Analysis:
    """Extract and analyze F0 distribution from vocals.

    Args:
        audio_path: Input vocal file path.
        method: F0 extraction method ("fcpe" or "crepe").

    Returns:
        F0 analysis object.

    Raises:
        TransposeError: If F0 extraction fails.
    """
    if not audio_path.exists():
        raise TransposeError(f"Audio file not found: {audio_path}")

    audio, sample_rate = sf.read(str(audio_path), dtype="float32")
    mono_audio = _to_mono_float32(audio)

    try:
        import librosa
    except ImportError as e:
        raise TransposeError("librosa is required for F0 analysis") from e

    target_sr = 16000
    hop_length = 160
    audio_16k = librosa.resample(mono_audio, orig_sr=sample_rate, target_sr=target_sr)

    f0_hz: np.ndarray
    method_l = method.lower()

    if method_l not in {"fcpe", "crepe"}:
        raise TransposeError(f"Unsupported F0 method: {method}")

    if method_l == "fcpe":
        try:
            import torch
            from torchfcpe import spawn_bundled_infer_model

            audio_tensor = torch.from_numpy(audio_16k).float().unsqueeze(0).unsqueeze(-1)
            device = "mps" if torch.backends.mps.is_available() else "cpu"
            model = spawn_bundled_infer_model(device=device)
            f0 = model.infer(
                audio_tensor.to(device),
                sr=target_sr,
                decoder_mode="local_argmax",
                threshold=0.003,
                f0_min=50,
                f0_max=1100,
                interp_uv=False,
                output_interp_target_length=None,
            )
            f0_hz = f0.squeeze().detach().cpu().numpy().astype(np.float32)
            logger.info("F0 extracted with torchfcpe (%d frames)", len(f0_hz))
        except Exception as e:
            logger.warning("torchfcpe failed (%s), fallback to librosa.pyin", e)
            method_l = "crepe"

    if method_l == "crepe":
        try:
            f0_py, _, _ = librosa.pyin(
                audio_16k,
                fmin=50,
                fmax=1100,
                sr=target_sr,
                hop_length=hop_length,
            )
            f0_hz = np.nan_to_num(f0_py, nan=0.0).astype(np.float32)
            logger.info("F0 extracted with librosa.pyin (%d frames)", len(f0_hz))
        except Exception as e:
            raise TransposeError(f"F0 extraction failed with method={method}: {e}") from e

    times = np.arange(len(f0_hz), dtype=np.float32) * (hop_length / target_sr)
    voiced_f0 = f0_hz[f0_hz > 0.0]
    median_hz, mean_hz, min_hz, max_hz, range_st = _safe_f0_stats(voiced_f0)

    return F0Analysis(
        f0_hz=f0_hz,
        times=times,
        sample_rate=sample_rate,
        voiced_f0=voiced_f0,
        median_hz=median_hz,
        mean_hz=mean_hz,
        min_hz=min_hz,
        max_hz=max_hz,
        range_semitones=range_st,
    )


def recommend_transpose(
    analysis: F0Analysis,
    model_sweet_spot: tuple[float, float] = (200.0, 500.0),
) -> TransposeRecommendation:
    """Recommend transpose values from F0 distribution and model sweet spot."""
    low, high = model_sweet_spot
    if low <= 0 or high <= low:
        raise ValueError("model_sweet_spot must be positive and increasing")

    if analysis.median_hz <= 0:
        return TransposeRecommendation(
            vocal_transpose=0,
            instrumental_shift=0,
            confidence=0.0,
            reason="No voiced F0 detected; keep transpose at 0.",
            source_median_hz=analysis.median_hz,
            target_median_hz=analysis.median_hz,
        )

    center_hz = midi_to_hz((hz_to_midi(low) + hz_to_midi(high)) / 2.0)
    desired_shift_float = semitones_between(analysis.median_hz, center_hz)
    total_shift = int(np.rint(desired_shift_float))

    abs_total = abs(total_shift)
    if abs_total <= 4:
        vocal_transpose = total_shift
        instrumental_shift = 0
        strategy = "all_vocal"
    elif abs_total <= 8:
        vocal_transpose = int(np.clip(total_shift, -4, 4))
        instrumental_shift = int(np.clip(total_shift - vocal_transpose, -2, 2))
        strategy = "hybrid"
    else:
        vocal_transpose = int(np.clip(total_shift, -4, 4))
        instrumental_shift = int(np.clip(total_shift - vocal_transpose, -2, 2))
        strategy = "extreme"

    target_median_hz = analysis.median_hz * (2 ** (vocal_transpose / 12.0))

    voiced = analysis.voiced_f0
    if len(voiced) == 0:
        out_ratio = 0.0
    else:
        out_ratio = float(np.mean((voiced < low) | (voiced > high)))
    distance_norm = float(min(abs(desired_shift_float) / 8.0, 1.0))
    confidence = float(np.clip(0.5 * out_ratio + 0.5 * distance_norm, 0.0, 1.0))

    if strategy == "all_vocal":
        reason = (
            f"Median F0 {analysis.median_hz:.1f}Hz -> center {center_hz:.1f}Hz; "
            f"use vocal transpose {vocal_transpose:+d}."
        )
    elif strategy == "hybrid":
        reason = (
            f"Need {total_shift:+d} semitones; split as vocal {vocal_transpose:+d} "
            f"and instrumental {instrumental_shift:+d} to stay in safer ranges."
        )
    else:
        residual = total_shift - vocal_transpose - instrumental_shift
        reason = (
            f"Large shift required ({total_shift:+d} st). Using vocal {vocal_transpose:+d} "
            f"+ instrumental {instrumental_shift:+d}; residual {residual:+d} may degrade quality."
        )

    return TransposeRecommendation(
        vocal_transpose=vocal_transpose,
        instrumental_shift=instrumental_shift,
        confidence=confidence,
        reason=reason,
        source_median_hz=analysis.median_hz,
        target_median_hz=target_median_hz,
    )


def _find_out_of_range_segments(
    analysis: F0Analysis,
    sweet_spot: tuple[float, float],
    duration_sec: float,
    min_gap_sec: float = 0.2,
    min_len_sec: float = 0.5,
    padding_sec: float = 0.05,
) -> list[tuple[float, float]]:
    low, high = sweet_spot
    voiced = analysis.f0_hz > 0
    out = voiced & ((analysis.f0_hz < low) | (analysis.f0_hz > high))

    if len(out) == 0 or not np.any(out):
        return []

    frame_dt = float(np.median(np.diff(analysis.times))) if len(analysis.times) > 1 else 0.01
    idx = np.where(out)[0]
    segments: list[tuple[float, float]] = []
    seg_start = idx[0]
    prev = idx[0]

    for current in idx[1:]:
        if current != prev + 1:
            start_t = max(0.0, analysis.times[seg_start] - frame_dt / 2)
            end_t = min(duration_sec, analysis.times[prev] + frame_dt / 2)
            segments.append((start_t, end_t))
            seg_start = current
        prev = current

    start_t = max(0.0, analysis.times[seg_start] - frame_dt / 2)
    end_t = min(duration_sec, analysis.times[prev] + frame_dt / 2)
    segments.append((start_t, end_t))

    merged: list[tuple[float, float]] = []
    for start, end in segments:
        if not merged:
            merged.append((start, end))
            continue
        last_start, last_end = merged[-1]
        if start - last_end < min_gap_sec:
            merged[-1] = (last_start, end)
        else:
            merged.append((start, end))

    filtered = [(s, e) for s, e in merged if (e - s) >= min_len_sec]
    padded = [
        (max(0.0, s - padding_sec), min(duration_sec, e + padding_sec))
        for s, e in filtered
    ]
    return padded


def _build_base_segments(
    out_segments: list[tuple[float, float]], duration_sec: float
) -> list[tuple[float, float, bool]]:
    result: list[tuple[float, float, bool]] = []
    cursor = 0.0

    for start, end in out_segments:
        if start > cursor:
            result.append((cursor, start, False))
        result.append((start, end, True))
        cursor = end

    if cursor < duration_sec:
        result.append((cursor, duration_sec, False))

    return [(s, e, o) for s, e, o in result if e > s]


def _apply_pitch_shift(audio: np.ndarray, sample_rate: int, semitones: int) -> np.ndarray:
    if semitones == 0:
        return audio

    from pedalboard import PitchShift

    board = PitchShift(semitones=semitones)
    if audio.ndim == 1:
        shifted = board(audio[np.newaxis, :], sample_rate)
        return shifted.squeeze(0)

    shifted = board(audio.T, sample_rate)
    return shifted.T


def _local_pre_shift(
    analysis: F0Analysis,
    seg_start_sec: float,
    seg_end_sec: float,
    sweet_spot: tuple[float, float],
) -> int:
    low, high = sweet_spot
    center_hz = midi_to_hz((hz_to_midi(low) + hz_to_midi(high)) / 2.0)

    mask = (analysis.times >= seg_start_sec) & (analysis.times < seg_end_sec)
    local_f0 = analysis.f0_hz[mask]
    local_voiced = local_f0[local_f0 > 0]
    if len(local_voiced) == 0:
        return 0

    local_median = float(np.median(local_voiced))
    if low <= local_median <= high:
        return 0

    shift = int(np.rint(semitones_between(local_median, center_hz)))
    return int(np.clip(shift, -12, 12))


def _run_external_convert(
    convert_fn: Callable[..., Any],
    input_path: Path,
    output_path: Path,
    model_path: Path,
    convert_kwargs: dict[str, Any],
) -> Path:
    kwargs = dict(convert_kwargs)
    kwargs.setdefault("transpose", 0)

    result = convert_fn(
        input_path=input_path,
        output_path=output_path,
        model_path=model_path,
        **kwargs,
    )

    if isinstance(result, Path):
        return result
    return output_path


def bounce_back_convert(
    input_path: Path,
    output_path: Path,
    analysis: F0Analysis,
    model_path: Path,
    sweet_spot: tuple[float, float] = (200.0, 500.0),
    **convert_kwargs: Any,
) -> Path:
    """Run bounce-back local transposition conversion.

    This function needs a conversion callable from caller integration:
    `convert_fn(input_path=..., output_path=..., model_path=..., transpose=0, ...)`

    Args:
        input_path: Input vocals path.
        output_path: Output vocals path.
        analysis: F0 analysis from analyze_f0().
        model_path: RVC model path.
        sweet_spot: Preferred model F0 range in Hz.
        **convert_kwargs: Extra kwargs forwarded to convert_fn.

    Returns:
        Output path.
    """
    if not input_path.exists():
        raise TransposeError(f"Input file not found: {input_path}")
    if not model_path.exists():
        raise TransposeError(f"Model file not found: {model_path}")

    convert_fn = convert_kwargs.pop("convert_fn", None)
    if convert_fn is None or not callable(convert_fn):
        raise TransposeError(
            "bounce_back_convert requires callable convert_fn in convert_kwargs"
        )

    audio, sample_rate = sf.read(str(input_path), dtype="float32")
    duration_sec = len(audio) / sample_rate

    out_segments = _find_out_of_range_segments(
        analysis=analysis,
        sweet_spot=sweet_spot,
        duration_sec=duration_sec,
        min_gap_sec=0.2,
        min_len_sec=0.5,
        padding_sec=0.05,
    )
    base_segments = _build_base_segments(out_segments, duration_sec)

    crossfade_ms = 10.0
    crossfade_samples = int(sample_rate * (crossfade_ms / 1000.0))

    logger.info(
        "Bounce-back segments: total=%d, out-of-range=%d",
        len(base_segments),
        len(out_segments),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)

    if audio.ndim == 1:
        mixed = np.zeros_like(audio)
        weights = np.zeros(len(audio), dtype=np.float32)
    else:
        mixed = np.zeros_like(audio)
        weights = np.zeros((len(audio), 1), dtype=np.float32)

    with tempfile.TemporaryDirectory(prefix="ai_song_bounce_") as tmp_dir:
        tmp = Path(tmp_dir)

        for i, (start_sec, end_sec, is_out) in enumerate(base_segments):
            start = int(np.floor(start_sec * sample_rate))
            end = int(np.ceil(end_sec * sample_rate))

            ext_start = max(0, start - crossfade_samples)
            ext_end = min(len(audio), end + crossfade_samples)
            ext_audio = audio[ext_start:ext_end]

            pre_shift = 0
            if is_out:
                pre_shift = _local_pre_shift(
                    analysis=analysis,
                    seg_start_sec=start_sec,
                    seg_end_sec=end_sec,
                    sweet_spot=sweet_spot,
                )

            stage_audio = ext_audio
            if pre_shift != 0:
                stage_audio = _apply_pitch_shift(stage_audio, sample_rate, pre_shift)

            seg_in = tmp / f"seg_{i:04d}_in.wav"
            seg_conv = tmp / f"seg_{i:04d}_conv.wav"
            sf.write(str(seg_in), stage_audio, sample_rate)

            _run_external_convert(
                convert_fn=convert_fn,
                input_path=seg_in,
                output_path=seg_conv,
                model_path=model_path,
                convert_kwargs=convert_kwargs,
            )

            seg_audio, seg_sr = sf.read(str(seg_conv), dtype="float32")
            if seg_sr != sample_rate:
                raise TransposeError(
                    f"Converted segment sr mismatch: {seg_sr} != {sample_rate}"
                )

            if pre_shift != 0:
                seg_audio = _apply_pitch_shift(seg_audio, sample_rate, -pre_shift)

            target_len = ext_end - ext_start
            if len(seg_audio) < target_len:
                if seg_audio.ndim == 1:
                    pad = np.zeros(target_len - len(seg_audio), dtype=seg_audio.dtype)
                else:
                    pad = np.zeros(
                        (target_len - len(seg_audio), seg_audio.shape[1]),
                        dtype=seg_audio.dtype,
                    )
                seg_audio = np.concatenate([seg_audio, pad], axis=0)
            elif len(seg_audio) > target_len:
                seg_audio = seg_audio[:target_len]

            if ext_start == 0:
                fade_in = 0
            else:
                fade_in = min(crossfade_samples, target_len)
            if ext_end == len(audio):
                fade_out = 0
            else:
                fade_out = min(crossfade_samples, target_len)

            window = np.ones(target_len, dtype=np.float32)
            if fade_in > 0:
                window[:fade_in] = np.linspace(0.0, 1.0, fade_in, endpoint=False)
            if fade_out > 0:
                window[-fade_out:] = np.linspace(1.0, 0.0, fade_out, endpoint=False)

            if seg_audio.ndim == 1:
                mixed[ext_start:ext_end] += seg_audio * window
                weights[ext_start:ext_end] += window
            else:
                mixed[ext_start:ext_end] += seg_audio * window[:, np.newaxis]
                weights[ext_start:ext_end] += window[:, np.newaxis]

            logger.info(
                "Processed segment %d/%d: %.2fs-%.2fs (%s, pre_shift=%+d)",
                i + 1,
                len(base_segments),
                start_sec,
                end_sec,
                "out" if is_out else "in",
                pre_shift,
            )

    weights_safe = np.maximum(weights, 1e-8)
    mixed = mixed / weights_safe

    sf.write(str(output_path), mixed, sample_rate)
    logger.info("Bounce-back conversion complete: %s", output_path)
    return output_path


def _ascii_histogram(values: np.ndarray, bins: int = 16, width: int = 32) -> list[str]:
    if len(values) == 0:
        return ["(no voiced F0)"]

    hist, edges = np.histogram(values, bins=bins)
    peak = hist.max() if len(hist) else 1
    lines: list[str] = []

    for i, count in enumerate(hist):
        left = edges[i]
        right = edges[i + 1]
        bar_len = int(np.rint((count / peak) * width)) if peak > 0 else 0
        bar = "#" * bar_len
        lines.append(f"{left:6.1f}-{right:6.1f} Hz | {bar}")

    return lines


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze vocal F0 and recommend transpose settings"
    )
    parser.add_argument("--input", type=Path, required=True, help="Input vocal WAV path")
    parser.add_argument(
        "--method",
        type=str,
        default="fcpe",
        choices=["fcpe", "crepe"],
        help="F0 extraction method",
    )
    parser.add_argument("--sweet-low", type=float, default=200.0, help="Sweet spot low Hz")
    parser.add_argument(
        "--sweet-high", type=float, default=500.0, help="Sweet spot high Hz"
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    analysis = analyze_f0(args.input, method=args.method)
    recommendation = recommend_transpose(
        analysis=analysis,
        model_sweet_spot=(args.sweet_low, args.sweet_high),
    )

    logger.info("F0 analysis: %s", args.input)
    logger.info("Voiced frames: %d / %d", len(analysis.voiced_f0), len(analysis.f0_hz))
    logger.info("Median: %.2f Hz", analysis.median_hz)
    logger.info("Mean: %.2f Hz", analysis.mean_hz)
    logger.info("P5-P95: %.2f - %.2f Hz", analysis.min_hz, analysis.max_hz)
    logger.info("Range: %.2f semitones", analysis.range_semitones)

    logger.info(
        "Recommended transpose: vocal=%+d, instrumental=%+d, confidence=%.2f",
        recommendation.vocal_transpose,
        recommendation.instrumental_shift,
        recommendation.confidence,
    )
    logger.info("Reason: %s", recommendation.reason)

    logger.info("F0 histogram (voiced):")
    for line in _ascii_histogram(analysis.voiced_f0):
        logger.info("%s", line)


if __name__ == "__main__":
    main()
