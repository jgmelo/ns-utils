"""Operation 5 — cycles per pixel to Hz, and the alias family.

The sampling is what it is: one sample per dwell time, so anything above the
Nyquist limit folds down and lands somewhere in the band looking exactly like a
low-frequency signal. Reporting a single Hz value would be a lie of omission,
so the whole family of candidates is emitted and the Nyquist limit is stated.

Each candidate also carries the per-line phase drift it *would* produce. That
prediction is what lets :mod:`.stability` pick the family member that is really
there — neither measurement can do it alone.
"""

from __future__ import annotations

from dataclasses import dataclass

from .dsp import wrap_deg
from .params import Acquisition, InterferenceParams


@dataclass(frozen=True)
class AliasCandidate:
    """One frequency consistent with what was observed."""

    order: int  # k in |k·f_s ± f_obs|
    sign: int  # +1 or -1
    frequency_hz: float
    predicted_drift_deg_per_line: float | None

    @property
    def label(self) -> str:
        if self.order == 0:
            return "f_obs"
        return f"{self.order}·f_s {'+' if self.sign > 0 else '−'} f_obs"

    @property
    def is_baseband(self) -> bool:
        return self.order == 0


@dataclass(frozen=True)
class PhysicalResult:
    available: bool
    reason: str | None = None
    dwell_s: float | None = None
    sampling_hz: float | None = None
    nyquist_hz: float | None = None
    line_period_s: float | None = None
    observed_hz: float | None = None
    pixels_per_line: int | None = None
    frame_includes_flyback: bool = False
    aliases: tuple[AliasCandidate, ...] = ()


def _predicted_drift(frequency_hz: float, line_period_s: float | None) -> float | None:
    """Phase advance per line for a tone at *frequency_hz*.

    Line *r* starts at ``r·T_line``, so the component's phase at the start of
    each line advances by ``2π·f·T_line``; what is observable is that value
    wrapped into one turn.
    """
    if line_period_s is None:
        return None
    return wrap_deg(360.0 * frequency_hz * line_period_s)


def resolve(
    centroid_cycles_px: float,
    n_columns: int,
    acquisition: Acquisition | None,
    params: InterferenceParams | None = None,
) -> PhysicalResult:
    """Convert the centroid to Hz and enumerate the alias family."""
    params = params or InterferenceParams()
    if acquisition is None or not acquisition.complete:
        return PhysicalResult(
            available=False,
            reason="not computed: no line rate supplied, so cycles/pixel cannot become Hz",
        )

    dwell = acquisition.dwell_s(n_columns)
    assert dwell is not None
    sampling = 1.0 / dwell
    nyquist = sampling / 2.0
    line_period = acquisition.line_period_s
    observed = centroid_cycles_px * sampling

    candidates: list[AliasCandidate] = [
        AliasCandidate(
            order=0,
            sign=1,
            frequency_hz=observed,
            predicted_drift_deg_per_line=_predicted_drift(observed, line_period),
        )
    ]
    for order in range(1, params.alias_orders + 1):
        for sign in (-1, 1):
            frequency = order * sampling + sign * observed
            if frequency <= 0:
                continue
            candidates.append(
                AliasCandidate(
                    order=order,
                    sign=sign,
                    frequency_hz=frequency,
                    predicted_drift_deg_per_line=_predicted_drift(frequency, line_period),
                )
            )

    return PhysicalResult(
        available=True,
        dwell_s=dwell,
        sampling_hz=sampling,
        nyquist_hz=nyquist,
        line_period_s=line_period,
        observed_hz=observed,
        pixels_per_line=acquisition.resolution or n_columns,
        frame_includes_flyback=acquisition.frame_includes_flyback,
        aliases=tuple(candidates),
    )


@dataclass(frozen=True)
class AliasMatch:
    """Whether the measured per-line drift can single out one alias member.

    Often it cannot, and that has to be said rather than papered over. The
    predicted drift is ``360·f·T_line`` *wrapped into one turn*, so a small
    error in the centroid becomes a large error in the prediction: the
    uncertainty is ``360·σ_f·T_line``, and once that reaches 180° the
    prediction carries no information at all.
    """

    candidate: AliasCandidate | None
    residual_deg: float
    drift_uncertainty_deg: float
    #: The centroid is sharp enough for the comparison to mean anything.
    informative: bool
    #: The best candidate's prediction actually agrees with the measurement.
    consistent: bool
    #: No other candidate is equally close.
    unambiguous: bool
    runner_up_residual_deg: float | None


def match_alias(
    result: PhysicalResult,
    measured_drift_deg: float,
    frequency_uncertainty_hz: float,
    separation_factor: float = 2.0,
) -> AliasMatch | None:
    """Rank the alias family against the observed drift per line."""
    usable = [c for c in result.aliases if c.predicted_drift_deg_per_line is not None]
    if not usable or result.line_period_s is None or measured_drift_deg != measured_drift_deg:
        return None

    uncertainty = abs(360.0 * frequency_uncertainty_hz * result.line_period_s)
    informative = uncertainty < 180.0

    scored = sorted(
        usable,
        key=lambda c: abs(wrap_deg(c.predicted_drift_deg_per_line - measured_drift_deg)),  # type: ignore[operator]
    )
    best = scored[0]
    best_residual = abs(wrap_deg(best.predicted_drift_deg_per_line - measured_drift_deg))  # type: ignore[operator]
    runner_up = None
    if len(scored) > 1:
        runner_up = abs(
            wrap_deg(scored[1].predicted_drift_deg_per_line - measured_drift_deg)  # type: ignore[operator]
        )

    tolerance = max(uncertainty, 5.0)
    consistent = informative and best_residual <= tolerance
    unambiguous = (
        consistent
        and runner_up is not None
        and runner_up >= separation_factor * max(best_residual, tolerance)
    )

    return AliasMatch(
        candidate=best if consistent else None,
        residual_deg=best_residual,
        drift_uncertainty_deg=uncertainty,
        informative=informative,
        consistent=consistent,
        unambiguous=unambiguous,
        runner_up_residual_deg=runner_up,
    )
