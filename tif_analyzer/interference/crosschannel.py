"""Operation 7 — does the reference channel carry the same interference?

Two correlations are reported, and the difference between them is the point:

* **raw** — the whole profiles, stripes and broadband noise together. This is
  what plain subtraction would act on.
* **band-restricted** — both profiles filtered to the stripe cluster. This is
  how much of the *interference* the two channels actually share.

A high band correlation with a low raw one means the stripes match but the rest
does not, so subtracting raw would cancel the stripes and inject the reference's
own noise in their place.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .dsp import analytic_signal, bandpass_by_bins, circular_mean_deg, pearson
from .spectrum import SpectrumResult


@dataclass(frozen=True)
class CrossChannelResult:
    period_signal_px: float
    period_reference_px: float
    period_relative_difference: float
    amplitude_ratio: float
    correlation_raw: float
    correlation_band: float
    #: Phase of the signal minus phase of the reference: positive means the
    #: signal's stripes sit ahead of the reference's along the line.
    phase_offset_deg: float
    phase_offset_px: float
    band_low_bin: int
    band_high_bin: int
    reference_detected: bool
    signal_band_rms: float
    reference_band_rms: float

    @property
    def correlation_gap(self) -> float:
        """How much of the agreement is stripes rather than everything else."""
        return self.correlation_band - abs(self.correlation_raw)


def compare(
    signal_profile: np.ndarray,
    reference_profile: np.ndarray,
    signal_spectrum: SpectrumResult,
    reference_spectrum: SpectrumResult,
) -> CrossChannelResult:
    """Compare the two fixed-pattern profiles."""
    signal_profile = np.asarray(signal_profile, dtype=np.float64)
    reference_profile = np.asarray(reference_profile, dtype=np.float64)
    if signal_profile.size != reference_profile.size:
        raise ValueError("profiles must have the same length")

    # The band spans both clusters: if the two channels disagree slightly on
    # frequency, restricting to only one channel's cluster would understate the
    # agreement.
    low = min(signal_spectrum.cluster_low_bin, reference_spectrum.cluster_low_bin)
    high = max(signal_spectrum.cluster_high_bin, reference_spectrum.cluster_high_bin)

    signal_band = bandpass_by_bins(signal_profile, low, high)
    reference_band = bandpass_by_bins(reference_profile, low, high)

    signal_rms = float(np.sqrt(np.mean(signal_band**2)))
    reference_rms = float(np.sqrt(np.mean(reference_band**2)))

    period_signal = signal_spectrum.period_px
    period_reference = reference_spectrum.period_px
    mean_period = 0.5 * (period_signal + period_reference)
    relative = (
        abs(period_signal - period_reference) / mean_period if np.isfinite(mean_period) and mean_period > 0 else float("nan")
    )

    analytic_sig = analytic_signal(signal_band)
    analytic_ref = analytic_signal(reference_band)
    product = analytic_sig * np.conj(analytic_ref)
    offset_deg = circular_mean_deg(np.angle(product), np.abs(product))
    offset_px = (
        offset_deg / 360.0 * period_signal if np.isfinite(period_signal) else float("nan")
    )

    return CrossChannelResult(
        period_signal_px=period_signal,
        period_reference_px=period_reference,
        period_relative_difference=relative,
        amplitude_ratio=(signal_rms / reference_rms) if reference_rms > 0 else float("inf"),
        correlation_raw=pearson(signal_profile, reference_profile),
        correlation_band=pearson(signal_band, reference_band),
        phase_offset_deg=offset_deg,
        phase_offset_px=offset_px,
        band_low_bin=low,
        band_high_bin=high,
        reference_detected=reference_spectrum.detected,
        signal_band_rms=signal_rms,
        reference_band_rms=reference_rms,
    )
