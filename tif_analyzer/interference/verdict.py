"""The decision, kept apart from the measurements.

Thresholds are the thing most likely to change once real frames arrive, so they
live in :mod:`.params` and are applied only here. Every check carries its value
*and* the threshold it was compared against, so the report can show both and the
reader can disagree with the verdict without reading any code.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .crosschannel import CrossChannelResult
from .params import InterferenceParams
from .spectrum import SpectrumResult
from .stability import StabilityResult


@dataclass(frozen=True)
class Check:
    name: str
    value: float
    threshold: float
    comparison: str  # "<=" or ">="
    passed: bool
    detail: str = ""

    @property
    def summary(self) -> str:
        mark = "pass" if self.passed else "FAIL"
        return f"{self.name}: {self.value:.4g} {self.comparison} {self.threshold:.4g}  [{mark}]"


@dataclass(frozen=True)
class Verdict:
    #: None when there is nothing to decide (no reference, or nothing detected).
    subtraction_viable: bool | None
    headline: str
    checks: tuple[Check, ...] = ()
    warnings: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()


def decide(
    spectrum: SpectrumResult,
    stability: StabilityResult,
    cross: CrossChannelResult | None,
    params: InterferenceParams,
) -> Verdict:
    """Apply the thresholds. Nothing here computes a measurement."""
    warnings: list[str] = []
    notes: list[str] = []

    if not spectrum.detected:
        return Verdict(
            subtraction_viable=None,
            headline="INCONCLUSIVE — no dominant fixed pattern found in the signal frame",
            notes=(spectrum.reason or "",),
        )

    if stability.available:
        if stability.locked_to_scan:
            notes.append(
                f"the pattern is locked to the line trigger: drift "
                f"{stability.phase_drift_deg_per_line:+.4f}°/line is within 3 standard errors "
                f"({3 * stability.phase_drift_stderr_deg:.4f}°) of zero, so it repeats frame to "
                "frame and a stored pattern could also be subtracted"
            )
        else:
            notes.append(
                f"the pattern drifts {stability.phase_drift_deg_per_line:+.4f}°/line "
                f"({stability.phase_span_deg:+.1f}° across the frame), so it is not locked to the "
                "scan and will move between frames — only a simultaneously acquired reference can "
                "cancel it"
            )
        if np.isfinite(stability.amplitude_cv) and stability.amplitude_cv > 0.5:
            warnings.append(
                f"stripe amplitude varies a lot line to line (CV {stability.amplitude_cv:.2f}); "
                "a single profile describes this frame only loosely"
            )

    if cross is None:
        return Verdict(
            subtraction_viable=None,
            headline="CHARACTERISED — no reference frame supplied, so subtraction was not assessed",
            notes=tuple(notes),
            warnings=tuple(warnings),
        )

    if not cross.reference_detected:
        warnings.append(
            "no dominant component was detected in the reference frame — it may not carry the "
            "interference at all"
        )

    period_check = Check(
        name="period agreement (relative)",
        value=cross.period_relative_difference,
        threshold=params.period_tolerance,
        comparison="<=",
        passed=bool(cross.period_relative_difference <= params.period_tolerance),
        detail=f"signal {cross.period_signal_px:.3f} px vs reference {cross.period_reference_px:.3f} px",
    )
    correlation_check = Check(
        name="band-restricted correlation",
        value=cross.correlation_band,
        threshold=params.band_corr_threshold,
        comparison=">=",
        passed=bool(cross.correlation_band >= params.band_corr_threshold),
        detail="how much of the interference itself the two channels share",
    )
    checks = (period_check, correlation_check)

    if cross.correlation_band <= -params.band_corr_threshold:
        warnings.append(
            f"band correlation is strongly negative ({cross.correlation_band:+.3f}): the reference "
            "carries the same pattern inverted, so it must be ADDED, not subtracted"
        )

    if cross.correlation_gap >= params.raw_band_gap_threshold:
        warnings.append(
            f"raw correlation ({cross.correlation_raw:+.3f}) is far below the band-restricted one "
            f"({cross.correlation_band:+.3f}), a gap of {cross.correlation_gap:.3f} ≥ "
            f"{params.raw_band_gap_threshold:.2f}: the channels share the stripes but not the rest, "
            "so subtracting the raw reference would cancel the pattern and inject the reference's "
            "own broadband noise in its place — filter the reference to the stripe band first"
        )

    viable = all(check.passed for check in checks)
    if viable:
        headline = (
            f"SUBTRACTION VIABLE — periods agree to "
            f"{cross.period_relative_difference:.2%} and band correlation is "
            f"{cross.correlation_band:+.3f}"
        )
    else:
        failed = ", ".join(check.name for check in checks if not check.passed)
        headline = f"SUBTRACTION NOT SUPPORTED — failed: {failed}"

    notes.append(
        f"reference amplitude is {1.0 / cross.amplitude_ratio:.3f}× the signal's in the stripe band"
        if cross.amplitude_ratio not in (0.0, float("inf")) and np.isfinite(cross.amplitude_ratio)
        else "reference stripe amplitude could not be compared"
    )
    direction = "ahead of" if cross.phase_offset_deg >= 0 else "behind"
    notes.append(
        f"the signal's stripes sit {abs(cross.phase_offset_deg):.1f}° "
        f"({abs(cross.phase_offset_px):.2f} px) {direction} the reference's — "
        "shift before subtracting"
    )

    return Verdict(
        subtraction_viable=viable,
        headline=headline,
        checks=checks,
        warnings=tuple(warnings),
        notes=tuple(notes),
    )
