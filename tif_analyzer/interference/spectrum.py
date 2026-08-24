"""Operation 4 — spectral estimate of the fixed-pattern profile.

Windowed FFT, dominant cluster, power-weighted centroid. The cluster matters
more than the peak bin: real interference is never exactly on a bin, so its
energy spreads over neighbours, and the centroid of that group is a better
estimate of the true frequency than the tallest bin alone.

Nothing is called *detected* unless the peak clears the noise floor: white noise
also has a tallest bin, and reporting its period as a finding would be worse
than reporting nothing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .dsp import window
from .params import InterferenceParams


@dataclass(frozen=True)
class SpectrumResult:
    detected: bool
    frequencies: np.ndarray  # cycles per pixel
    power: np.ndarray
    peak_bin: int
    peak_frequency: float
    centroid_frequency: float
    period_px: float
    cluster_low_bin: int
    cluster_high_bin: int
    cluster_width_cycles: float
    power_fraction: float
    noise_floor: float
    peak_to_floor: float
    required_ratio: float
    reason: str | None = None

    @property
    def cluster_bins(self) -> tuple[int, int]:
        return self.cluster_low_bin, self.cluster_high_bin

    @property
    def cluster_frequency_range(self) -> tuple[float, float]:
        return (
            float(self.frequencies[self.cluster_low_bin]),
            float(self.frequencies[self.cluster_high_bin]),
        )


def analyse(profile: np.ndarray, params: InterferenceParams | None = None) -> SpectrumResult:
    """Locate the dominant periodic component of *profile*."""
    params = params or InterferenceParams()
    values = np.asarray(profile, dtype=np.float64)
    n = values.size
    if n < 8:
        raise ValueError(f"profile too short for a spectral estimate: {n} points")

    centred = values - values.mean()
    weights = window(params.window, n)
    spectrum = np.fft.rfft(centred * weights)
    power = np.abs(spectrum) ** 2
    frequencies = np.fft.rfftfreq(n, d=1.0)

    low = max(1, params.exclude_low_bins)
    if low >= power.size:
        raise ValueError("exclude_low_bins leaves no bins to search")

    searchable = power[low:]
    peak_offset = int(np.argmax(searchable))
    peak_bin = low + peak_offset
    peak_power = float(power[peak_bin])

    # Grow the cluster outwards while the spectrum is still *descending* and
    # still above the floor. The descent condition is what stops the cluster
    # from swallowing a second, independent peak; the floor is what stops it
    # from crawling off into the noise. Both are needed: a plain power-drop cut
    # would leave the window's own side lobes outside the cluster, and a clean
    # single tone would then report only two thirds of its power as clustered.
    preliminary_floor = float(np.median(searchable))
    floor_level = max(params.cluster_power_drop * peak_power, 3.0 * preliminary_floor)
    lo = peak_bin
    while lo - 1 >= low and power[lo - 1] <= power[lo] and power[lo - 1] >= floor_level:
        lo -= 1
    hi = peak_bin
    while hi + 1 < power.size and power[hi + 1] <= power[hi] and power[hi + 1] >= floor_level:
        hi += 1

    cluster = power[lo : hi + 1]
    cluster_freqs = frequencies[lo : hi + 1]
    total = float(np.sum(power[low:]))
    cluster_power = float(np.sum(cluster))
    centroid = float(np.sum(cluster_freqs * cluster) / cluster_power) if cluster_power > 0 else 0.0

    outside = np.concatenate([power[low:lo], power[hi + 1 :]])
    noise_floor = float(np.median(outside)) if outside.size else 0.0
    peak_to_floor = peak_power / noise_floor if noise_floor > 0 else float("inf")

    bin_width = float(frequencies[1] - frequencies[0])
    width = float(cluster_freqs[-1] - cluster_freqs[0]) + bin_width

    # False-alarm-controlled threshold: for exponentially distributed noise
    # power, P(max of N bins > x * median) ~ N * 2^-x, so requiring
    # x >= log2(N / alpha) keeps the chance of calling noise a detection at alpha.
    searched_bins = max(1, power.size - low)
    statistical = float(np.log2(searched_bins / params.detection_false_alarm))
    required = max(params.detection_ratio, statistical)

    detected = bool(peak_to_floor >= required and centroid > 0.0)
    reason = None
    if not detected:
        reason = (
            f"no dominant component: the tallest bin is only {peak_to_floor:.1f}x the noise "
            f"floor, and {required:.1f}x is what {searched_bins} bins of pure noise would "
            f"already produce {params.detection_false_alarm:.0%} of the time — this profile "
            "looks like broadband noise, not a fixed pattern"
        )

    return SpectrumResult(
        detected=detected,
        frequencies=frequencies,
        power=power,
        peak_bin=peak_bin,
        peak_frequency=float(frequencies[peak_bin]),
        centroid_frequency=centroid,
        period_px=(1.0 / centroid) if centroid > 0 else float("inf"),
        cluster_low_bin=lo,
        cluster_high_bin=hi,
        cluster_width_cycles=width,
        power_fraction=(cluster_power / total) if total > 0 else float("nan"),
        noise_floor=noise_floor,
        peak_to_floor=peak_to_floor,
        required_ratio=required,
        reason=reason,
    )
