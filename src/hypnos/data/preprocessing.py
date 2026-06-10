"""Signal preprocessing functions for physiological signals.

Provides the filtering, normalization, and amplitude-compression steps applied before
tokenization. The causal variants (forward-only filters + rolling z-score) match the
preprocessing the released causal tokenizers were trained with.

Pipeline: bandpass/highpass → notch → winsorized z-score → amplitude compression.

The winsorized z-score computes mean/std on quantile-clipped data (q01/q99),
making it robust to transient artifacts while producing identical results on
clean recordings.
"""

from math import gcd

import numpy as np
from scipy.signal import butter, iirnotch, lfilter, lfilter_zi, resample_poly, sosfilt, sosfilt_zi, sosfiltfilt, tf2sos


def resample_signal(signal: np.ndarray, original_fs: int, target_fs: int) -> np.ndarray:
    """Resample signal to target sampling rate using polyphase filtering.

    Args:
        signal: Input signal, shape (n_samples,) or (n_samples, n_channels).
        original_fs: Original sampling rate in Hz.
        target_fs: Target sampling rate in Hz.

    Returns:
        Resampled signal as float32.
    """
    if original_fs == target_fs:
        return signal

    g = gcd(target_fs, original_fs)
    up = target_fs // g
    down = original_fs // g

    if signal.ndim == 1:
        resampled = resample_poly(signal, up, down)
    else:
        resampled = resample_poly(signal, up, down, axis=0)

    return resampled.astype(np.float32)


def bandpass_filter(
    signal: np.ndarray,
    low: float,
    high: float,
    fs: float,
    order: int = 4,
) -> np.ndarray:
    """Apply zero-phase Butterworth bandpass filter.

    Args:
        signal: Input signal, shape (n_samples,) or (n_samples, n_channels).
        low: Low cutoff frequency in Hz.
        high: High cutoff frequency in Hz.
        fs: Sampling rate in Hz.
        order: Filter order.

    Returns:
        Filtered signal with same shape as input.
    """
    sos = butter(order, [low, high], btype="bandpass", fs=fs, output="sos")
    axis = 0
    return sosfiltfilt(sos, signal, axis=axis).astype(signal.dtype)


def highpass_filter(
    signal: np.ndarray,
    cutoff: float,
    fs: float,
    order: int = 4,
) -> np.ndarray:
    """Apply zero-phase Butterworth high-pass filter.

    Args:
        signal: Input signal, shape (n_samples,) or (n_samples, n_channels).
        cutoff: Cutoff frequency in Hz.
        fs: Sampling rate in Hz.
        order: Filter order.

    Returns:
        Filtered signal with same shape as input.
    """
    sos = butter(order, cutoff, btype="highpass", fs=fs, output="sos")
    axis = 0
    return sosfiltfilt(sos, signal, axis=axis).astype(signal.dtype)


def notch_filter(
    signal: np.ndarray,
    freq: float,
    fs: float,
    quality: float = 30.0,
) -> np.ndarray:
    """Apply zero-phase notch filter for powerline noise removal.

    Args:
        signal: Input signal, shape (n_samples,) or (n_samples, n_channels).
        freq: Notch frequency in Hz (typically 50 or 60).
        fs: Sampling rate in Hz.
        quality: Quality factor (higher = narrower notch).

    Returns:
        Filtered signal with same shape as input.
    """
    b, a = iirnotch(freq, quality, fs)
    sos = tf2sos(b, a)
    axis = 0
    return sosfiltfilt(sos, signal, axis=axis).astype(signal.dtype)


def robust_normalize(
    signal: np.ndarray,
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Winsorized z-score normalization along the time axis.

    Computes mean and std on quantile-clipped data, then normalizes the
    original (unclipped) signal. This is robust to transient artifacts:
    a few large spikes won't inflate std and squish normal morphology.

    On clean data, produces near-identical results to standard z-score.

    Args:
        signal: Input signal, shape (n_samples,) or (n_samples, n_channels).
        lower_quantile: Lower quantile for winsorization (default 0.01).
        upper_quantile: Upper quantile for winsorization (default 0.99).

    Returns:
        Tuple of (normalized_signal, mean, std) where mean and std are the
        winsorized statistics used for normalization.
    """
    q_low, q_high = np.nanquantile(signal, [lower_quantile, upper_quantile], axis=0)
    clipped = np.clip(signal, q_low, q_high)
    mean = np.mean(clipped, axis=0)
    std = np.std(clipped, axis=0)
    std = np.where(std > 1e-8, std, 1e-8)
    normalized = (signal - mean) / std
    return normalized.astype(np.float32), mean.astype(np.float32), std.astype(np.float32)


def compress_amplitude(signal: np.ndarray, knee: float = 8.0, scale: float = 2.0) -> np.ndarray:
    """Compress extreme amplitudes: identity for |x| <= knee, log beyond.

    Uses log1p compression beyond the knee to smoothly attenuate extremes
    without introducing a hard boundary. Unbounded output avoids giving
    GAN discriminators a trivial real-vs-fake signal. C1 continuous at knee.

    Args:
        signal: Input signal (should be z-score normalized).
        knee: Amplitude beyond which compression starts.
        scale: Controls compression rate (larger = gentler compression).
    """
    out = signal.copy()
    mask_pos = signal > knee
    mask_neg = signal < -knee
    out[mask_pos] = knee + scale * np.log1p((signal[mask_pos] - knee) / scale)
    out[mask_neg] = -knee - scale * np.log1p((-signal[mask_neg] - knee) / scale)
    return out.astype(signal.dtype)


# Per-modality preprocessing configurations
MODALITY_CONFIGS = {
    "eeg": {"highpass": 0.5, "lowpass": 45.0, "notch": True},
    "ecg": {"highpass": 0.05, "lowpass": 45.0, "notch": True},
    "emg": {"highpass": 0.5, "lowpass": 45.0, "notch": True},
    "ppg": {"highpass": 0.1, "lowpass": 30.0, "notch": False},
    "respiratory": {"highpass": 0.05, "lowpass": None, "notch": False},
}


# -----------------------------------------------------------------------------
# Causal preprocessing
# -----------------------------------------------------------------------------
# The functions above use `sosfiltfilt` (zero-phase) and full-recording z-score
# statistics, both of which violate causality. The functions below provide
# strictly causal replacements: forward-only IIR filtering and an exponential
# moving average normaliser. These are appropriate when downstream models must
# operate in a streaming / real-time setting, or to verify that a model is not
# relying on access to future samples.


def causal_bandpass_filter(
    signal: np.ndarray,
    low: float,
    high: float,
    fs: float,
    order: int = 4,
    warmup: bool = True,
) -> np.ndarray:
    """Forward-only Butterworth bandpass filter.

    Unlike `bandpass_filter` (which uses `sosfiltfilt` for zero-phase response),
    this introduces filter group delay but is strictly causal. The filter state
    is initialised from the first sample to reduce a startup transient when
    `warmup=True` (steady-state init), otherwise zero state is used.

    Args:
        signal: Input signal, shape (n_samples,) or (n_samples, n_channels).
        low: Low cutoff frequency in Hz.
        high: High cutoff frequency in Hz.
        fs: Sampling rate in Hz.
        order: Filter order (per band edge; total causal order is 2*order).
        warmup: If True, initialise filter state to steady-state value for the
            first sample, avoiding a startup ringing transient.

    Returns:
        Filtered signal with same shape as input.
    """
    sos = butter(order, [low, high], btype="bandpass", fs=fs, output="sos")
    return _apply_causal_sos(sos, signal, warmup=warmup)


def causal_highpass_filter(
    signal: np.ndarray,
    cutoff: float,
    fs: float,
    order: int = 4,
    warmup: bool = True,
) -> np.ndarray:
    """Forward-only Butterworth high-pass filter. See `causal_bandpass_filter`."""
    sos = butter(order, cutoff, btype="highpass", fs=fs, output="sos")
    return _apply_causal_sos(sos, signal, warmup=warmup)


def causal_notch_filter(
    signal: np.ndarray,
    freq: float,
    fs: float,
    quality: float = 30.0,
    warmup: bool = True,
) -> np.ndarray:
    """Forward-only IIR notch filter for powerline noise removal.

    Causal counterpart of `notch_filter`. The notch is narrow, so causal
    filtering produces only a small phase shift near the rejected frequency.
    """
    b, a = iirnotch(freq, quality, fs)
    sos = tf2sos(b, a)
    return _apply_causal_sos(sos, signal, warmup=warmup)


def _apply_causal_sos(sos: np.ndarray, signal: np.ndarray, warmup: bool) -> np.ndarray:
    """Apply an SOS filter causally along axis 0, optionally with steady-state init."""
    axis = 0
    if warmup:
        zi = sosfilt_zi(sos)
        if signal.ndim == 1:
            init = zi * signal[0]
            filtered, _ = sosfilt(sos, signal, axis=axis, zi=init)
        else:
            # Broadcast zi (n_sections, 2) against each channel
            init = zi[:, :, None] * signal[0][None, None, :]
            filtered, _ = sosfilt(sos, signal, axis=axis, zi=init)
    else:
        filtered = sosfilt(sos, signal, axis=axis)
    return filtered.astype(signal.dtype)


def rolling_normalize(
    signal: np.ndarray,
    fs: float,
    tau_seconds: float = 60.0,
    clip_sigma: float | None = 6.0,
    eps: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Causal rolling z-score via exponential moving average of mean and variance.

    Maintains running estimates with timescale ``tau_seconds``:

        mu_t  = a * mu_{t-1}  + (1 - a) * x_t
        var_t = a * var_{t-1} + (1 - a) * (x_t - mu_t)^2

    where ``a = exp(-1 / (tau_seconds * fs))``. The output at time t uses only
    samples up to and including t, so the operation is strictly causal.

    Robustness to transient artifacts (analogous to the winsorisation in
    ``robust_normalize``) is achieved in two causal passes:
      1. Compute an unclipped EMA variance ``var0`` (one-pole IIR via lfilter).
      2. Clip the per-sample squared deviation to ``clip_sigma**2 * var0`` (which
         depends only on past samples) and re-run the EMA on the clipped values.
    Set ``clip_sigma=None`` to skip the second pass.

    State is initialised from the variance of the first ``tau_seconds`` of
    signal so the startup transient is bounded and uses no future samples.

    Args:
        signal: Input signal, shape (n_samples,) or (n_samples, n_channels).
        fs: Sampling rate in Hz.
        tau_seconds: EMA timescale in seconds. Longer = smoother (closer to
            per-recording stats), shorter = more reactive to local drift.
        clip_sigma: Causal clip on the squared deviation contribution. ``None``
            disables; otherwise extremes beyond ``clip_sigma`` past-rolling-σ
            do not inflate the running variance.
        eps: Numerical floor on variance.

    Returns:
        Tuple ``(normalized, mu_track, std_track)``. ``mu_track`` and
        ``std_track`` have the same shape as ``signal``.
    """
    x = np.asarray(signal, dtype=np.float64)
    one_d = x.ndim == 1
    if one_d:
        x = x[:, None]

    n_samples = x.shape[0]
    alpha = float(np.exp(-1.0 / max(tau_seconds * fs, 1.0)))

    # First-order IIR: y[n] = alpha*y[n-1] + (1-alpha)*x[n]
    b = np.array([1.0 - alpha], dtype=np.float64)
    a = np.array([1.0, -alpha], dtype=np.float64)
    zi_unit = lfilter_zi(b, a)  # shape (1,), value 1.0 — scaled per-channel

    init_window = max(int(tau_seconds * fs), 1)
    init_window = min(init_window, n_samples)

    # Hold state on NaNs by substituting the column mean; rare in practice.
    init_mu = np.nanmean(x[:init_window], axis=0)
    init_var = np.maximum(np.nanvar(x[:init_window], axis=0), eps)
    x_safe = np.where(np.isnan(x), init_mu, x)

    # Pass 1 — causal mean
    zi_mu = zi_unit[:, None] * init_mu[None, :]
    mu_track, _ = lfilter(b, a, x_safe, axis=0, zi=zi_mu)

    sq_dev = (x_safe - mu_track) ** 2

    # Pass 2 — causal variance (unclipped)
    zi_v = zi_unit[:, None] * init_var[None, :]
    var0, _ = lfilter(b, a, sq_dev, axis=0, zi=zi_v)
    var0 = np.maximum(var0, eps)

    if clip_sigma is not None:
        # Clip sq_dev to clip_sigma^2 * past-causal-var (here we use var0 itself,
        # which depends only on samples <= t, so still causal).
        max_sq = (float(clip_sigma) ** 2) * var0
        sq_dev_clipped = np.minimum(sq_dev, max_sq)
        var_track, _ = lfilter(b, a, sq_dev_clipped, axis=0, zi=zi_v)
        var_track = np.maximum(var_track, eps)
    else:
        var_track = var0

    std_track = np.sqrt(var_track)
    normalized = (x - mu_track) / std_track

    if one_d:
        normalized = normalized[:, 0]
        mu_track = mu_track[:, 0]
        std_track = std_track[:, 0]

    return (
        normalized.astype(np.float32),
        mu_track.astype(np.float32),
        std_track.astype(np.float32),
    )


def causal_preprocess_signal(
    signal: np.ndarray,
    fs: float,
    modality: str,
    notch_freq: float = 50.0,
    tau_seconds: float = 60.0,
    knee: float = 8.0,
    compress_scale: float = 2.0,
    filter_order: int = 4,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fully causal modality-aware preprocessing.

    Pipeline (each step causal):
        causal bandpass / highpass  ->  causal notch (optional)
        ->  rolling EMA z-score  ->  log amplitude compression.

    The only behavioural differences from `preprocess_signal` are
    (a) filter group delay (each IIR section contributes a few samples) and
    (b) the normalisation tracks slow drift rather than using one global
    mean/std. The amplitude compression stage is identical (it's pointwise).

    Args:
        signal: Input signal, shape (n_samples,) or (n_samples, n_channels).
        fs: Sampling rate in Hz.
        modality: One of the keys of `MODALITY_CONFIGS`.
        notch_freq: Powerline frequency (50 / 60 Hz).
        tau_seconds: EMA timescale for the rolling normaliser.
        knee: Log-compression knee (matches `preprocess_signal`).
        compress_scale: Log-compression scale.
        filter_order: Butterworth order per band edge.

    Returns:
        Tuple `(processed, mu_track, std_track)`.
    """
    if modality not in MODALITY_CONFIGS:
        raise ValueError(f"Unknown modality '{modality}'. Must be one of {list(MODALITY_CONFIGS.keys())}")

    config = MODALITY_CONFIGS[modality]
    processed = signal.astype(np.float64, copy=True)

    if config["lowpass"] is not None:
        processed = causal_bandpass_filter(processed, config["highpass"], config["lowpass"], fs, order=filter_order)
    else:
        processed = causal_highpass_filter(processed, config["highpass"], fs, order=filter_order)

    if config["notch"]:
        processed = causal_notch_filter(processed, notch_freq, fs)

    processed, mu_track, std_track = rolling_normalize(processed, fs, tau_seconds=tau_seconds)

    processed = compress_amplitude(processed, knee, compress_scale)

    return processed, mu_track, std_track


def preprocess_signal(
    signal: np.ndarray,
    fs: float,
    modality: str,
    notch_freq: float = 50.0,
    knee: float = 8.0,
    compress_scale: float = 2.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply full modality-appropriate preprocessing pipeline.

    Pipeline: bandpass/highpass → notch (optional) → winsorized z-score → log compression.

    Args:
        signal: Input signal, shape (n_samples,) or (n_samples, n_channels).
        fs: Sampling rate in Hz.
        modality: One of 'eeg', 'ecg', 'ppg', 'respiratory'.
        notch_freq: Powerline frequency in Hz (50 for EU, 60 for US).
        knee: Amplitude beyond which log-compression starts.
        compress_scale: Controls compression rate (larger = gentler).

    Returns:
        Tuple of (preprocessed_signal, mean, std) where mean/std are the
        per-recording normalization statistics (for potential scale recovery).
    """
    if modality not in MODALITY_CONFIGS:
        raise ValueError(f"Unknown modality '{modality}'. Must be one of {list(MODALITY_CONFIGS.keys())}")

    config = MODALITY_CONFIGS[modality]
    processed = signal.copy()

    # Filtering
    if config["lowpass"] is not None:
        processed = bandpass_filter(processed, config["highpass"], config["lowpass"], fs)
    else:
        processed = highpass_filter(processed, config["highpass"], fs)

    # Notch filter for powerline noise
    if config["notch"]:
        processed = notch_filter(processed, notch_freq, fs)

    # Per-recording winsorized z-score normalization
    processed, mean, std = robust_normalize(processed)

    # Amplitude compression (log beyond knee)
    processed = compress_amplitude(processed, knee, compress_scale)

    return processed, mean, std
