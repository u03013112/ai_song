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

    All defaults tuned to "J version" — the best-performing vocal chain
    from V1.2 A/B testing (pre-gain attenuation + gentle compression +
    moderate EQ + low-drive saturation + limiter + spatial effects).

    Attributes:
        vocal_lufs: Target loudness for vocals in LUFS.
        instrumental_lufs: Target loudness for instrumental in LUFS.
        backing_vocal_lufs: Target loudness for backing vocals in LUFS.
        pre_gain_db: Pre-attenuation before effects chain to prevent
            hot RVC output from overloading downstream processors.
        high_pass_freq: High-pass filter cutoff for vocals in Hz.
        vocal_compression: Whether to apply compression.
        compressor_threshold_db: Compressor threshold in dB.
        compressor_ratio: Compressor ratio (e.g. 2.5 means 2.5:1).
        compressor_attack_ms: Compressor attack time in milliseconds.
        compressor_release_ms: Compressor release time in milliseconds.
        makeup_gain_db: Gain applied after compressor to compensate
            for compression-induced level loss.
        vocal_eq: Whether to apply body/presence/air EQ.
        eq_body_gain_db: Low-shelf boost for body at 250 Hz.
        eq_presence_gain_db: Peak boost for presence at 3 kHz.
        eq_air_gain_db: High-shelf boost for air at 10 kHz.
        vocal_warmth: Whether to apply warmth/saturation.
        warmth_drive_db: Soft saturation drive amount. Keep ≤2 dB
            to avoid high-note overload.
        warmth_compensation_db: Gain reduction after distortion to
            compensate for drive-induced level increase.
        limiter_enabled: Whether to apply limiter after saturation.
            Critical for preventing high-note distortion.
        limiter_threshold_db: Limiter ceiling in dBFS.
        limiter_release_ms: Limiter release time in milliseconds.
        vocal_delay: Whether to apply subtle delay.
        delay_seconds: Delay time in seconds.
        delay_mix: Delay wet/dry mix.
        delay_feedback: Delay feedback amount (0.0-1.0).
        vocal_reverb: Whether to add reverb to vocals.
        reverb_room_size: Reverb room size 0.0-1.0.
        reverb_wet: Reverb wet/dry mix 0.0-1.0.
        reverb_damping: High-frequency damping 0.0-1.0.
        reverb_width: Stereo width 0.0-1.0.
        vocal_gain_db: Additional vocal gain in dB after LUFS normalization.
        sample_rate: Output sample rate in Hz.
        backing_compression: Whether to apply backing vocal compression.
        backing_compressor_threshold_db: Backing compressor threshold in dB.
        backing_compressor_ratio: Backing compressor ratio.
        backing_compressor_attack_ms: Backing compressor attack in milliseconds.
        backing_compressor_release_ms: Backing compressor release in milliseconds.
        backing_makeup_gain_db: Backing post-compression makeup gain in dB.
        backing_eq: Whether to apply backing vocal EQ.
        backing_eq_body_gain_db: Backing low-shelf gain at 250 Hz.
        backing_eq_presence_gain_db: Backing peak gain at 3 kHz.
        backing_eq_air_gain_db: Backing high-shelf gain at 10 kHz.
        backing_reverb: Whether to apply backing vocal reverb.
        backing_reverb_room_size: Backing reverb room size 0.0-1.0.
        backing_reverb_wet: Backing reverb wet/dry mix 0.0-1.0.
        backing_reverb_damping: Backing reverb damping 0.0-1.0.
        backing_reverb_width: Backing reverb stereo width 0.0-1.0.
        backing_limiter_threshold_db: Backing limiter ceiling in dBFS.
        bus_reverb: Whether to apply a shared bus reverb to the final
            mix. This places all tracks (vocals, backing, instrumental)
            into the same acoustic space, preventing the "KTV" effect
            where vocals sound like they're in a different room.
        bus_reverb_room_size: Bus reverb room size 0.0-1.0.
        bus_reverb_wet: Bus reverb wet/dry mix 0.0-1.0. Keep low
            (0.08-0.15) since this affects the entire mix.
        bus_reverb_damping: Bus reverb high-frequency damping 0.0-1.0.
        bus_reverb_width: Bus reverb stereo width 0.0-1.0.
    """

    # --- Loudness targets ---
    vocal_lufs: float = -17.0
    instrumental_lufs: float = -18.0
    backing_vocal_lufs: float = -22.0

    # --- Pre-gain ---
    pre_gain_db: float = -3.0

    # --- High-pass filter ---
    high_pass_freq: float = 80.0

    # --- Compressor ---
    vocal_compression: bool = True
    compressor_threshold_db: float = -16.0
    compressor_ratio: float = 2.5
    compressor_attack_ms: float = 15.0
    compressor_release_ms: float = 120.0
    makeup_gain_db: float = 1.5

    # --- EQ ---
    vocal_eq: bool = True
    eq_body_gain_db: float = 2.0
    eq_presence_gain_db: float = 1.0
    eq_air_gain_db: float = 2.0

    # --- Warmth / Saturation ---
    vocal_warmth: bool = True
    warmth_drive_db: float = 1.5
    warmth_compensation_db: float = -0.9

    # --- Limiter ---
    limiter_enabled: bool = True
    limiter_threshold_db: float = -3.0
    limiter_release_ms: float = 50.0

    # --- Delay ---
    vocal_delay: bool = True
    delay_seconds: float = 0.08
    delay_mix: float = 0.08
    delay_feedback: float = 0.15

    # --- Vocal track reverb (individual) ---
    # Keep low — most spatial cohesion comes from the bus reverb.
    vocal_reverb: bool = True
    reverb_room_size: float = 0.45
    reverb_wet: float = 0.10
    reverb_damping: float = 0.6
    reverb_width: float = 1.0

    # --- Post-mix ---
    vocal_gain_db: float = 0.0
    sample_rate: int = 44100

    # --- Backing vocal effects ---
    backing_compression: bool = True
    backing_compressor_threshold_db: float = -20.0
    backing_compressor_ratio: float = 2.0
    backing_compressor_attack_ms: float = 20.0
    backing_compressor_release_ms: float = 150.0
    backing_makeup_gain_db: float = 1.0
    backing_eq: bool = True
    backing_eq_body_gain_db: float = 1.0
    backing_eq_presence_gain_db: float = 0.5
    backing_eq_air_gain_db: float = 1.0
    backing_reverb: bool = True
    backing_reverb_room_size: float = 0.55
    backing_reverb_wet: float = 0.15
    backing_reverb_damping: float = 0.55
    backing_reverb_width: float = 1.0
    backing_limiter_threshold_db: float = -6.0

    # --- Bus reverb (shared across all tracks) ---
    # Applied to the final mix to unify acoustic space.
    bus_reverb: bool = True
    bus_reverb_room_size: float = 0.35
    bus_reverb_wet: float = 0.10
    bus_reverb_damping: float = 0.65
    bus_reverb_width: float = 1.0


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
    """Apply vocal chain: PreGain → HPF → Compressor → EQ → Warmth → Limiter → Delay → Reverb.

    Args:
        audio: Vocal audio data.
        sample_rate: Sample rate in Hz.
        config: Mix configuration.

    Returns:
        Processed vocal audio.
    """
    from pedalboard import (
        Compressor,
        Delay,
        Distortion,
        Gain,
        HighShelfFilter,
        HighpassFilter,
        Limiter,
        LowShelfFilter,
        PeakFilter,
        Pedalboard,
        Reverb,
    )

    effects: list = []

    if config.pre_gain_db != 0.0:
        effects.append(Gain(gain_db=config.pre_gain_db))
        logger.info("Pre-gain: %+.1f dB", config.pre_gain_db)

    if config.high_pass_freq > 0:
        effects.append(HighpassFilter(cutoff_frequency_hz=config.high_pass_freq))
        logger.info("HPF: %.0f Hz", config.high_pass_freq)

    if config.vocal_compression:
        effects.append(
            Compressor(
                threshold_db=config.compressor_threshold_db,
                ratio=config.compressor_ratio,
                attack_ms=config.compressor_attack_ms,
                release_ms=config.compressor_release_ms,
            )
        )
        effects.append(Gain(gain_db=config.makeup_gain_db))
        logger.info(
            "Compressor: threshold=%.0fdB, ratio=%.1f, attack=%.0fms, "
            "release=%.0fms + %.1fdB makeup",
            config.compressor_threshold_db,
            config.compressor_ratio,
            config.compressor_attack_ms,
            config.compressor_release_ms,
            config.makeup_gain_db,
        )

    if config.vocal_eq:
        effects.append(LowShelfFilter(
            cutoff_frequency_hz=250.0, gain_db=config.eq_body_gain_db
        ))
        effects.append(PeakFilter(
            cutoff_frequency_hz=3000.0, gain_db=config.eq_presence_gain_db, q=1.0
        ))
        effects.append(HighShelfFilter(
            cutoff_frequency_hz=10000.0, gain_db=config.eq_air_gain_db
        ))
        logger.info(
            "EQ: body=+%.1fdB@250Hz, presence=+%.1fdB@3kHz, air=+%.1fdB@10kHz",
            config.eq_body_gain_db,
            config.eq_presence_gain_db,
            config.eq_air_gain_db,
        )

    if config.vocal_warmth:
        effects.append(Distortion(drive_db=config.warmth_drive_db))
        effects.append(Gain(gain_db=config.warmth_compensation_db))
        logger.info(
            "Warmth: drive=%.1fdB, compensation=%+.1fdB",
            config.warmth_drive_db,
            config.warmth_compensation_db,
        )

    if config.limiter_enabled:
        effects.append(
            Limiter(
                threshold_db=config.limiter_threshold_db,
                release_ms=config.limiter_release_ms,
            )
        )
        logger.info(
            "Limiter: threshold=%.1fdB, release=%.0fms",
            config.limiter_threshold_db,
            config.limiter_release_ms,
        )

    if config.vocal_delay:
        effects.append(
            Delay(
                delay_seconds=config.delay_seconds,
                mix=config.delay_mix,
                feedback=config.delay_feedback,
            )
        )
        logger.info(
            "Delay: %.0fms, mix=%.0f%%, feedback=%.0f%%",
            config.delay_seconds * 1000,
            config.delay_mix * 100,
            config.delay_feedback * 100,
        )

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
            "Reverb: room=%.2f, wet=%.2f, damping=%.2f",
            config.reverb_room_size,
            config.reverb_wet,
            config.reverb_damping,
        )

    if not effects:
        return audio

    board = Pedalboard(effects)

    if audio.ndim == 1:
        audio_2d = audio[np.newaxis, :]
    else:
        audio_2d = audio.T

    processed = board(audio_2d, sample_rate)

    if audio.ndim == 1:
        return processed.squeeze()
    return processed.T


def _apply_backing_vocal_effects(
    audio: np.ndarray, sample_rate: int, config: MixConfig
) -> np.ndarray:
    """Apply backing vocal chain: HPF → Compressor → EQ → Limiter → Reverb."""
    from pedalboard import (
        Compressor,
        Gain,
        HighShelfFilter,
        HighpassFilter,
        Limiter,
        LowShelfFilter,
        PeakFilter,
        Pedalboard,
        Reverb,
    )

    effects: list = []

    if config.high_pass_freq > 0:
        effects.append(HighpassFilter(cutoff_frequency_hz=config.high_pass_freq))
        logger.info("Backing HPF: %.0f Hz", config.high_pass_freq)

    if config.backing_compression:
        effects.append(
            Compressor(
                threshold_db=config.backing_compressor_threshold_db,
                ratio=config.backing_compressor_ratio,
                attack_ms=config.backing_compressor_attack_ms,
                release_ms=config.backing_compressor_release_ms,
            )
        )
        effects.append(Gain(gain_db=config.backing_makeup_gain_db))
        logger.info(
            "Backing compressor: threshold=%.0fdB, ratio=%.1f, attack=%.0fms, "
            "release=%.0fms + %.1fdB makeup",
            config.backing_compressor_threshold_db,
            config.backing_compressor_ratio,
            config.backing_compressor_attack_ms,
            config.backing_compressor_release_ms,
            config.backing_makeup_gain_db,
        )

    if config.backing_eq:
        effects.append(LowShelfFilter(
            cutoff_frequency_hz=250.0, gain_db=config.backing_eq_body_gain_db
        ))
        effects.append(PeakFilter(
            cutoff_frequency_hz=3000.0,
            gain_db=config.backing_eq_presence_gain_db,
            q=1.0,
        ))
        effects.append(HighShelfFilter(
            cutoff_frequency_hz=10000.0, gain_db=config.backing_eq_air_gain_db
        ))
        logger.info(
            "Backing EQ: body=+%.1fdB@250Hz, presence=+%.1fdB@3kHz, "
            "air=+%.1fdB@10kHz",
            config.backing_eq_body_gain_db,
            config.backing_eq_presence_gain_db,
            config.backing_eq_air_gain_db,
        )

    effects.append(
        Limiter(
            threshold_db=config.backing_limiter_threshold_db,
            release_ms=config.limiter_release_ms,
        )
    )
    logger.info(
        "Backing limiter: threshold=%.1fdB, release=%.0fms",
        config.backing_limiter_threshold_db,
        config.limiter_release_ms,
    )

    if config.backing_reverb:
        effects.append(
            Reverb(
                room_size=config.backing_reverb_room_size,
                wet_level=config.backing_reverb_wet,
                damping=config.backing_reverb_damping,
                width=config.backing_reverb_width,
            )
        )
        logger.info(
            "Backing reverb: room=%.2f, wet=%.2f, damping=%.2f",
            config.backing_reverb_room_size,
            config.backing_reverb_wet,
            config.backing_reverb_damping,
        )

    board = Pedalboard(effects)

    if audio.ndim == 1:
        audio_2d = audio[np.newaxis, :]
    else:
        audio_2d = audio.T

    processed = board(audio_2d, sample_rate)

    if audio.ndim == 1:
        return processed.squeeze()
    return processed.T


def _apply_bus_reverb(
    audio: np.ndarray, sample_rate: int, config: MixConfig
) -> np.ndarray:
    """Apply a shared bus reverb to the final mix.

    This places all tracks into the same acoustic space, preventing
    the effect where individually-reverbed vocals sound disconnected
    from a dry instrumental.

    Args:
        audio: Mixed audio data (all tracks already summed).
        sample_rate: Sample rate in Hz.
        config: Mix configuration with bus reverb parameters.

    Returns:
        Audio with bus reverb applied.
    """
    if not config.bus_reverb:
        return audio

    from pedalboard import Pedalboard, Reverb

    board = Pedalboard([
        Reverb(
            room_size=config.bus_reverb_room_size,
            wet_level=config.bus_reverb_wet,
            damping=config.bus_reverb_damping,
            width=config.bus_reverb_width,
        ),
    ])

    logger.info(
        "Bus reverb: room=%.2f, wet=%.2f, damping=%.2f, width=%.2f",
        config.bus_reverb_room_size,
        config.bus_reverb_wet,
        config.bus_reverb_damping,
        config.bus_reverb_width,
    )

    if audio.ndim == 1:
        audio_2d = audio[np.newaxis, :]
    else:
        audio_2d = audio.T

    processed = board(audio_2d, sample_rate)

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
    backing_vocals_path: Path | None = None,
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
        backing_vocals_path: Optional path to converted backing vocal audio (WAV).
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
        backing_vocals = None
        b_sr = i_sr

        if backing_vocals_path is not None and backing_vocals_path.exists():
            backing_vocals, b_sr = sf.read(str(backing_vocals_path), dtype="float32")

        logger.info(
            "Loaded vocals: %.1fs @ %d Hz, instrumental: %.1fs @ %d Hz",
            len(vocals) / v_sr,
            v_sr,
            len(instrumental) / i_sr,
            i_sr,
        )

        sample_rate = i_sr
        if v_sr != sample_rate:
            import soxr

            logger.info(
                "Resampling vocals: %d Hz -> %d Hz (SoXR VHQ)", v_sr, sample_rate
            )
            vocals = soxr.resample(vocals, v_sr, sample_rate, quality="VHQ")

        if backing_vocals is not None and b_sr != sample_rate:
            import soxr

            logger.info(
                "Resampling backing vocals: %d Hz -> %d Hz (SoXR VHQ)",
                b_sr,
                sample_rate,
            )
            backing_vocals = soxr.resample(
                backing_vocals,
                b_sr,
                sample_rate,
                quality="VHQ",
            )

        logger.info("Applying vocal effects...")
        vocals = _apply_vocal_effects(vocals, sample_rate, config)
        if backing_vocals is not None:
            logger.info("Applying backing vocal effects...")
            backing_vocals = _apply_backing_vocal_effects(
                backing_vocals,
                sample_rate,
                config,
            )

        # Step 2: LUFS normalization
        logger.info("Normalizing loudness...")
        vocals = _normalize_lufs(vocals, sample_rate, config.vocal_lufs)
        instrumental = _normalize_lufs(instrumental, sample_rate, config.instrumental_lufs)
        if backing_vocals is not None:
            backing_vocals = _normalize_lufs(
                backing_vocals,
                sample_rate,
                config.backing_vocal_lufs,
            )

        # Step 3: Optional vocal gain boost
        if config.vocal_gain_db != 0.0:
            logger.info("Applying vocal gain: %+.1f dB", config.vocal_gain_db)
            vocals = _apply_gain_db(vocals, config.vocal_gain_db)

        # Step 4: Match dimensions
        vocals, instrumental = _match_length_and_channels(vocals, instrumental)
        if backing_vocals is not None:
            backing_vocals, vocals = _match_length_and_channels(backing_vocals, vocals)
            instrumental, vocals = _match_length_and_channels(instrumental, vocals)

        # Step 5: Mix (sum) and clip
        mixed = vocals + instrumental
        if backing_vocals is not None:
            mixed = mixed + backing_vocals

        # Step 5.5: Bus reverb — unify acoustic space across all tracks
        mixed = _apply_bus_reverb(mixed, sample_rate, config)

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

    result = mix_tracks(
        args.vocals,
        args.instrumental,
        output_path,
        config=config,
    )
    print(f"Mixed: {result}")


if __name__ == "__main__":
    main()
