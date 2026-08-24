"""Small signal-processing helpers, numpy only.

scipy is deliberately not a dependency: the analytic signal is five lines of
FFT, and the cluster search is written by hand either way.
"""

from __future__ import annotations

import numpy as np

WINDOWS = ("hann", "hamming", "none")


def window(name: str, n: int) -> np.ndarray:
    """Periodic window of length *n*, normalised to unit peak."""
    if n <= 0:
        raise ValueError("window length must be positive")
    if name == "none":
        return np.ones(n, dtype=np.float64)
    k = np.arange(n, dtype=np.float64)
    if name == "hann":
        return 0.5 - 0.5 * np.cos(2.0 * np.pi * k / n)
    if name == "hamming":
        return 0.54 - 0.46 * np.cos(2.0 * np.pi * k / n)
    raise ValueError(f"unknown window {name!r}; expected one of {WINDOWS}")


def analytic_signal(x: np.ndarray) -> np.ndarray:
    """Hilbert analytic signal: keep the positive frequencies, double them.

    Equivalent to ``scipy.signal.hilbert`` for real input.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if n == 0:
        return np.zeros(0, dtype=np.complex128)
    spectrum = np.fft.fft(x)
    weights = np.zeros(n, dtype=np.float64)
    weights[0] = 1.0
    if n % 2 == 0:
        weights[1 : n // 2] = 2.0
        weights[n // 2] = 1.0
    else:
        weights[1 : (n + 1) // 2] = 2.0
    return np.fft.ifft(spectrum * weights)


def bandpass_by_bins(x: np.ndarray, low_bin: int, high_bin: int) -> np.ndarray:
    """Keep only rfft bins ``[low_bin, high_bin]`` (inclusive) and transform back."""
    x = np.asarray(x, dtype=np.float64)
    spectrum = np.fft.rfft(x)
    keep = np.zeros_like(spectrum)
    low = max(0, low_bin)
    high = min(spectrum.size - 1, high_bin)
    if high >= low:
        keep[low : high + 1] = spectrum[low : high + 1]
    return np.fft.irfft(keep, n=x.size)


def pearson(a: np.ndarray, b: np.ndarray) -> float:
    """Pearson correlation, signed. NaN when either side is constant."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size != b.size or a.size < 2:
        raise ValueError("pearson needs two equal-length series of at least 2 points")
    a = a - a.mean()
    b = b - b.mean()
    denominator = float(np.sqrt(np.dot(a, a) * np.dot(b, b)))
    if denominator == 0.0:
        return float("nan")
    return float(np.dot(a, b) / denominator)


def goertzel(x: np.ndarray, frequency: float, weights: np.ndarray | None = None) -> complex:
    """Single-bin DFT of *x* at *frequency* in cycles/sample.

    Used instead of a per-line FFT: one scan line is short, and its full
    periodogram is noisy, but the projection onto a frequency we already know
    is well conditioned.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    if n == 0:
        return 0j
    w = np.ones(n, dtype=np.float64) if weights is None else np.asarray(weights, dtype=np.float64)
    phasor = np.exp(-2j * np.pi * frequency * np.arange(n))
    return complex(np.sum(x * w * phasor))


def circular_mean_deg(angles_rad: np.ndarray, weights: np.ndarray | None = None) -> float:
    """Weighted circular mean, in degrees within (-180, 180]."""
    angles_rad = np.asarray(angles_rad, dtype=np.float64)
    if angles_rad.size == 0:
        return float("nan")
    w = np.ones_like(angles_rad) if weights is None else np.asarray(weights, dtype=np.float64)
    vector = np.sum(w * np.exp(1j * angles_rad))
    return float(np.degrees(np.angle(vector)))


def wrap_deg(degrees: float) -> float:
    """Fold an angle into (-180, 180]."""
    wrapped = (float(degrees) + 180.0) % 360.0 - 180.0
    return 180.0 if wrapped == -180.0 else wrapped


def mad(x: np.ndarray, centre: float | None = None) -> float:
    """Median absolute deviation (unscaled)."""
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return float("nan")
    middle = float(np.median(x)) if centre is None else centre
    return float(np.median(np.abs(x - middle)))


#: MAD × this ≈ the standard deviation of a Gaussian.
MAD_TO_SIGMA = 1.4826
