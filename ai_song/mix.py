"""Mix converted vocals with instrumental tracks."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import soundfile as sf

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_DIR = Path("output/mixed")


class MixError(Exception):
    """Failed to mix audio tracks."""


@dataclass
class MixConfig:
    """Configuration for audio mixing.

    Attributes:
        vocal_lufs: Target loudness for vocals in LUFS (default: -16).
            Typical pop vocals sit around -14 to -18 LUFS.
        instrumental_lufs: Target loudness for instrumental in LUFS (default: -18).
            Slightly quieter than vocals so voice sits on top.
        vocal_reverb: Whether to add reverb to vocals (default: True).
        reverb_room_size: Reverb room size 0.0-1.0 (default: 0.3).
            0.1 = small room, 0.5 = hall, 0.9 = cathedral.
        reverb_wet: Reverb wet/dry mix 0.0-1.0 (default: 0.15).
            Lower values = subtler reverb. 0.1-0.2 is typical for pop.
        reverb_damping: High-frequency damping 0.0-1.0 (default: 0.7).
            Higher = warmer/darker reverb.
        reverb_width: Stereo width 0.0-1.0 (default: 1.0).
        vocal_gain_db: Additional vocal gain in dB after LUFS normalization
            (default: 0.0). Use +1 to +3 to push vocals forward.
        high_pass_freq: High-pass filter cutoff for vocals in Hz (default: 80).
            Removes low-end rumble from vocal track.
        sample_rate: Output sample rate in Hz (default: 44100).
    """

    vocal_lufs: float = -16.0
    instrumental_lufs: float = -18.0
    vocal_reverb: bool = True
    reverb_room_size: float = 0.3
    reverb_wet: float = 0.15
    reverb_damping: float = 0.7
    reverb_width: float = 1.0
    vocal_gain_db: float = 0.0
    high_pass_freq: float = 80.0
    sample_rate: int = 44100


def _measure_lufs(audio: np.ndarray, sample_rate: int) -> float:
    """Measure integrated loudness in LUFS.

    Args:
        audio: Audio data as numpy array, shape (samples,) or (samples, channels).
        sample_rate: Sample rate in Hz.

    Returns:
        Integrated loudness in LUFS.
    """
    import pyloudnorm as pyln

    meter = pyln.Meter(sample_rate)

    # pyloudnorm expects (samples, channels)
    if audio.ndim == 1:
        audio = audio[:, np.newaxis]

    return meter.integrated_loudness(audio)


def _normalize_lufs(
    audio: np.ndarray, sample_rate: int, target_lufs: float
) -> np.ndarray:
    """Normalize audio to a target LUFS level.

    Args:
        audio: Audio data as numpy array.
        sample_rate: Sample rate in Hz.
        target_lufs: Target loudness in LUFS.

    Returns:
        Loudness-normalized audio.
    """
    import pyloudnorm as pyln

    meter = pyln.Meter(sample_rate)

    # pyloudnorm expects (samples, channels)
    was_mono = audio.ndim == 1
    if was_mono:
        audio = audio[:, np.newaxis]

    current_lufs = meter.integrated_loudness(audio)
    logger.info("Current LUFS: %.1f, target: %.1f", current_lufs, target_lufs)

    if np.isinf(current_lufs):
        logger.warning("Audio is silent (LUFS=-inf), skipping normalization")
        return audio.squeeze() if was_mono else audio

    normalized = pyln.normalize.loudness(audio, current_lufs, target_lufs)

    if was_mono:
        normalized = normalized.squeeze()

    return normalized


def _apply_vocal_effects(
    audio: np.ndarray, sample_rate: int, config: MixConfig
) -> np.ndarray:
    """Apply reverb and EQ effects to vocals.

    Args:
        audio: Vocal audio data.
        sample_rate: Sample rate in Hz.
        config: Mix configuration.

    Returns:
        Processed vocal audio.
    """
    from pedalboard import (
        HighpassFilter,
        Pedalboard,
        Reverb,
    )

    effects: list[HighpassFilter | Reverb] = []

    # High-pass filter to remove low-end rumble
    if config.high_pass_freq > 0:
        effects.append(HighpassFilter(cutoff_frequency_hz=config.high_pass_freq))
        logger.info("Applying high-pass filter: %.0f Hz", config.high_pass_freq)

    # Reverb to help vocals sit in the mix
    if config.vocal_reverb:
        effects.append(
            Reverb(
                room_size=config.reverb_room_size,
                wet_level=config.reverb_wet,
                damping=config.reverb_damping,
                width=config.reverb_width,
            )
        )
        logger.info(
            "Applying reverb: room=%.2f, wet=%.2f, damping=%.2f",
            config.reverb_room_size,
            config.reverb_wet,
            config.reverb_damping,
        )

    if not effects:
        return audio

    board = Pedalboard(effects)

    # pedalboard expects (channels, samples)
    if audio.ndim == 1:
        audio_2d = audio[np.newaxis, :]
    else:
        audio_2d = audio.T

    processed = board(audio_2d, sample_rate)

    # Convert back to (samples,) or (samples, channels)
    if audio.ndim == 1:
        return processed.squeeze()
    return processed.T


def _apply_gain_db(audio: np.ndarray, gain_db: float) -> np.ndarray:
    """Apply gain in dB to audio.

    Args:
        audio: Audio data.
        gain_db: Gain in decibels.

    Returns:
        Gained audio.
    """
    if gain_db == 0.0:
        return audio
    factor = 10.0 ** (gain_db / 20.0)
    return audio * factor


def _match_length_and_channels(
    vocals: np.ndarray, instrumental: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Ensure vocals and instrumental have matching length and channel count.

    Pads the shorter track with silence and converts mono to stereo if needed.

    Args:
        vocals: Vocal audio array.
        instrumental: Instrumental audio array.

    Returns:
        Tuple of (vocals, instrumental) with matched dimensions.
    """
    # Match channels: if one is stereo and other is mono, convert mono to stereo
    v_channels = 1 if vocals.ndim == 1 else vocals.shape[1]
    i_channels = 1 if instrumental.ndim == 1 else instrumental.shape[1]

    if v_channels != i_channels:
        target_channels = max(v_channels, i_channels)
        if v_channels == 1 and target_channels == 2:
            vocals = np.column_stack([vocals, vocals])
            logger.info("Converted vocals from mono to stereo")
        if i_channels == 1 and target_channels == 2:
            instrumental = np.column_stack([instrumental, instrumental])
            logger.info("Converted instrumental from mono to stereo")

    # Match length: pad shorter track with silence
    v_len = len(vocals)
    i_len = len(instrumental)
    if v_len != i_len:
        target_len = max(v_len, i_len)
        if v_len < target_len:
            pad_shape = (target_len - v_len,) + vocals.shape[1:]
            vocals = np.concatenate([vocals, np.zeros(pad_shape, dtype=vocals.dtype)])
            logger.info(
                "Padded vocals with %.1fs silence",
                (target_len - v_len) / 44100,
            )
        if i_len < target_len:
            pad_shape = (target_len - i_len,) + instrumental.shape[1:]
            instrumental = np.concatenate(
                [instrumental, np.zeros(pad_shape, dtype=instrumental.dtype)]
            )
            logger.info(
                "Padded instrumental with %.1fs silence",
                (target_len - i_len) / 44100,
            )

    return vocals, instrumental


def mix_tracks(
    vocals_path: Path,
    instrumental_path: Path,
    output_path: Path,
    config: MixConfig | None = None,
) -> Path:
    """Mix converted vocals with instrumental track.

    Pipeline:
    1. Load both tracks
    2. Apply high-pass filter and reverb to vocals
    3. Normalize both tracks to target LUFS levels
    4. Apply optional vocal gain boost
    5. Match lengths and channels
    6. Sum and clip to [-1, 1]
    7. Write output

    Args:
        vocals_path: Path to converted vocal audio (WAV).
        instrumental_path: Path to instrumental audio (WAV).
        output_path: Path to save mixed output.
        config: Mix configuration. Uses defaults if None.

    Returns:
        Path to the mixed audio file.

    Raises:
        MixError: If mixing fails.
    """
    if config is None:
        config = MixConfig()

    if not vocals_path.exists():
        raise MixError(f"Vocals file not found: {vocals_path}")
    if not instrumental_path.exists():
        raise MixError(f"Instrumental file not found: {instrumental_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        start = time.time()

        # Load audio
        vocals, v_sr = sf.read(str(vocals_path), dtype="float32")
        instrumental, i_sr = sf.read(str(instrumental_path), dtype="float32")

        logger.info(
            "Loaded vocals: %.1fs @ %d Hz, instrumental: %.1fs @ %d Hz",
            len(vocals) / v_sr,
            v_sr,
            len(instrumental) / i_sr,
            i_sr,
        )

        sample_rate = i_sr
        if v_sr != sample_rate:
            import scipy.signal

            logger.info(
                "Resampling vocals: %d Hz -> %d Hz", v_sr, sample_rate
            )
            num_samples = int(len(vocals) * sample_rate / v_sr)
            if vocals.ndim == 1:
                vocals = scipy.signal.resample(vocals, num_samples).astype(
                    np.float32
                )
            else:
                vocals = scipy.signal.resample(vocals, num_samples, axis=0).astype(
                    np.float32
                )

        logger.info("Applying vocal effects...")
        vocals = _apply_vocal_effects(vocals, sample_rate, config)

        # Step 2: LUFS normalization
        logger.info("Normalizing loudness...")
        vocals = _normalize_lufs(vocals, sample_rate, config.vocal_lufs)
        instrumental = _normalize_lufs(instrumental, sample_rate, config.instrumental_lufs)

        # Step 3: Optional vocal gain boost
        if config.vocal_gain_db != 0.0:
            logger.info("Applying vocal gain: %+.1f dB", config.vocal_gain_db)
            vocals = _apply_gain_db(vocals, config.vocal_gain_db)

        # Step 4: Match dimensions
        vocals, instrumental = _match_length_and_channels(vocals, instrumental)

        # Step 5: Mix (sum) and clip
        mixed = vocals + instrumental
        peak = np.abs(mixed).max()
        if peak > 1.0:
            logger.warning(
                "Clipping detected (peak=%.2f). Normalizing to prevent distortion.",
                peak,
            )
            mixed = mixed / peak * 0.99

        # Step 6: Write output
        sf.write(str(output_path), mixed, sample_rate, subtype="PCM_16")

        elapsed = time.time() - start
        output_lufs = _measure_lufs(mixed, sample_rate)
        logger.info(
            "Mix complete in %.1fs: %s (%.1f LUFS)",
            elapsed,
            output_path,
            output_lufs,
        )
        return output_path

    except ImportError as e:
        raise MixError(
            "Missing dependency. Run: pip install pyloudnorm pedalboard"
        ) from e
    except Exception as e:
        raise MixError(f"Mixing failed: {e}") from e


def main() -> None:
    """CLI entry point for audio mixing."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Mix converted vocals with instrumental"
    )
    parser.add_argument("vocals", type=Path, help="Converted vocal audio (WAV)")
    parser.add_argument(
        "instrumental", type=Path, help="Instrumental audio (WAV)"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output file path (default: output/mixed/<stem>_mixed.wav)",
    )
    parser.add_argument(
        "--vocal-lufs",
        type=float,
        default=-16.0,
        help="Target vocal loudness in LUFS (default: -16)",
    )
    parser.add_argument(
        "--instrumental-lufs",
        type=float,
        default=-18.0,
        help="Target instrumental loudness in LUFS (default: -18)",
    )
    parser.add_argument(
        "--no-reverb",
        action="store_true",
        help="Disable vocal reverb",
    )
    parser.add_argument(
        "--reverb-room-size",
        type=float,
        default=0.3,
        help="Reverb room size 0.0-1.0 (default: 0.3)",
    )
    parser.add_argument(
        "--reverb-wet",
        type=float,
        default=0.15,
        help="Reverb wet/dry mix 0.0-1.0 (default: 0.15)",
    )
    parser.add_argument(
        "--vocal-gain",
        type=float,
        default=0.0,
        help="Additional vocal gain in dB (default: 0)",
    )
    parser.add_argument(
        "--high-pass",
        type=float,
        default=80.0,
        help="High-pass filter cutoff in Hz, 0 to disable (default: 80)",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    config = MixConfig(
        vocal_lufs=args.vocal_lufs,
        instrumental_lufs=args.instrumental_lufs,
        vocal_reverb=not args.no_reverb,
        reverb_room_size=args.reverb_room_size,
        reverb_wet=args.reverb_wet,
        vocal_gain_db=args.vocal_gain,
        high_pass_freq=args.high_pass,
    )

    if args.output is not None:
        output_path = args.output
    else:
        stem = args.vocals.stem.replace("_converted", "").replace("_vocals", "")
        output_path = DEFAULT_OUTPUT_DIR / f"{stem}_mixed.wav"

    result = mix_tracks(args.vocals, args.instrumental, output_path, config)
    print(f"Mixed: {result}")


if __name__ == "__main__":
    main()
