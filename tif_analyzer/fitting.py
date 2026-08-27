"""Power-law fitting, and how much the fitted exponent is worth.

``y = a · x^b`` is one line in log-log space, so the whole thing is ordinary
least squares on ``log y`` against ``log x`` — no solver, no scipy. What takes
the space here is not the fit but the honesty around it: with four or five
points, an exponent of 0.62 can be perfectly consistent with 0.5 *and* with 1.0
at the same time, and a tool that answers "0.62, so shot-noise limited" has
answered a question the data did not settle.

Three things decide whether the number is worth anything, and all three are
reported:

* **The standard error of the exponent**, ``σ_residual / √Σ(log xᵢ − log x̄)²``.
  The denominator is the lever arm — how far the points sit from the centre of
  the x axis, in log. It is not the scatter of the points; scatter is the
  numerator. Spreading the x values apart shrinks the error on the slope even
  when every point is noisy.
* **The lever arm itself**, in decades. Powers within a factor of three of each
  other cannot separate 0.5 from 1.0 no matter how clean the measurement is,
  and the fit says so instead of picking one.
* **Which targets the interval contains.** A verdict is only issued when the
  interval contains one candidate exponent and excludes the others. Containing
  both is a real answer — *indeterminate* — and so is containing neither.

Log-log least squares weights *relative* deviations equally, which is the right
choice for a quantity measured across a decade of excitation, and it is the
assumption to remember when reading the residuals: they are in log units.

Nothing here prints, and nothing here knows any physics; :mod:`noise_regime`
supplies the meaning of a particular exponent.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping, Sequence

import numpy as np

#: Below this the fit is reported but the verdict is withheld: a factor of three
#: in x is about 0.48 decades, and less lever arm than that cannot separate
#: exponents half a unit apart from any realistic measurement.
MIN_DECADES = 0.45

#: Multiplier on the standard error for the reported interval. Three, matching
#: the "is it locked?" test on the interference phase drift: the point of the
#: interval is to refuse a verdict, so it should be generous.
DEFAULT_K_SIGMA = 3.0


class FitError(Exception):
    """The fit cannot be computed as asked."""


@dataclass(frozen=True)
class LockedModel:
    """The same power law with the exponent held fixed and only ``a`` fitted.

    Its ``residual_rms`` is directly comparable to the free fit's: same points,
    same log units, one fitted parameter instead of two.
    """

    exponent: float
    amplitude: float
    residual_rms: float


@dataclass(frozen=True)
class PowerLawFit:
    """``y = amplitude · x^exponent``, fitted in log-log space."""

    exponent: float
    exponent_stderr: float
    amplitude: float
    residual_rms: float  # in log10 units
    r_squared: float
    n_points: int
    n_dropped: int
    decades: float
    #: Fixed-exponent comparisons, keyed by exponent.
    locked: dict[float, LockedModel] = field(default_factory=dict)
    warnings: tuple[str, ...] = field(default_factory=tuple)

    def interval(self, k_sigma: float = DEFAULT_K_SIGMA) -> tuple[float, float]:
        if not math.isfinite(self.exponent_stderr):
            return (-math.inf, math.inf)
        margin = k_sigma * self.exponent_stderr
        return (self.exponent - margin, self.exponent + margin)

    def contains(self, target: float, k_sigma: float = DEFAULT_K_SIGMA) -> bool:
        low, high = self.interval(k_sigma)
        return low <= target <= high

    @property
    def has_lever_arm(self) -> bool:
        return self.decades >= MIN_DECADES

    def predict(self, xs) -> np.ndarray:
        xs = np.asarray(xs, dtype=np.float64)
        return self.amplitude * np.power(xs, self.exponent)

    def predict_locked(self, xs, exponent: float) -> np.ndarray:
        """The fixed-exponent model's curve, amplitude refitted for it."""
        model = self.locked.get(exponent)
        if model is None:
            raise FitError(f"no locked model at exponent {exponent}")
        xs = np.asarray(xs, dtype=np.float64)
        return model.amplitude * np.power(xs, model.exponent)


def _clean(xs: Sequence[float], ys: Sequence[float]) -> tuple[np.ndarray, np.ndarray, int]:
    """Keep the points a log can be taken of, and count what was dropped.

    Signal-to-noise goes negative when the signal area is dimmer than the noise
    area, and ``log`` of that does not exist. Dropping such a point silently
    would leave a fit with fewer points than the plot shows.
    """
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    if x.shape != y.shape:
        raise FitError(f"x has {x.size} points and y has {y.size}")
    keep = np.isfinite(x) & np.isfinite(y) & (x > 0.0) & (y > 0.0)
    return x[keep], y[keep], int((~keep).sum())


def locked_fit(log_x: np.ndarray, log_y: np.ndarray, exponent: float) -> tuple[float, float]:
    """Fit only the amplitude with the exponent held at *exponent*.

    With ``b`` fixed, the least-squares intercept is just the mean offset —
    ``d/dc Σ(log y − b log x − c)² = 0`` gives ``c = mean(log y − b log x)``.
    Returns ``(log10 amplitude, residual rms)``.
    """
    intercept = float(np.mean(log_y - exponent * log_x))
    residuals = log_y - (exponent * log_x + intercept)
    return intercept, float(np.sqrt(np.mean(residuals**2)))


def fit_power_law(
    xs: Sequence[float],
    ys: Sequence[float],
    locked: Sequence[float] = (),
) -> PowerLawFit:
    """Fit ``y = a·x^b`` by least squares on the logarithms.

    Raises :class:`FitError` when there is nothing meaningful to fit: fewer than
    three usable points (two points define a line exactly, leaving no residual
    and no error bar), or every x at the same value.
    """
    x, y, dropped = _clean(xs, ys)
    n = x.size
    if n < 3:
        raise FitError(
            f"a power-law fit needs at least 3 points with positive x and y; "
            f"{n} usable, {dropped} dropped"
        )

    log_x = np.log10(x)
    log_y = np.log10(y)
    centred = log_x - log_x.mean()
    lever = float(np.sqrt(np.sum(centred**2)))
    if lever == 0.0:
        raise FitError("every x value is the same — there is no slope to fit")

    exponent, intercept = (float(v) for v in np.polyfit(log_x, log_y, 1))
    residuals = log_y - (exponent * log_x + intercept)
    ss_res = float(np.sum(residuals**2))
    ss_tot = float(np.sum((log_y - log_y.mean()) ** 2))

    # n − 2: the two fitted parameters have already been spent bringing the line
    # close to the points, so the raw residuals understate the scatter.
    sigma_residual = math.sqrt(ss_res / (n - 2))
    stderr = sigma_residual / lever
    decades = float(np.log10(x.max() / x.min()))

    locked_models: dict[float, LockedModel] = {}
    for value in locked:
        log_amplitude, rms = locked_fit(log_x, log_y, float(value))
        locked_models[float(value)] = LockedModel(
            exponent=float(value), amplitude=10.0**log_amplitude, residual_rms=rms
        )

    warnings: list[str] = []
    if dropped:
        warnings.append(
            f"{dropped} point(s) dropped: a logarithm needs x and y both positive "
            "(a negative SNR means the signal area is dimmer than the noise area)"
        )
    if n == 3:
        warnings.append(
            "3 points leaves one degree of freedom — the error bar on the exponent "
            "is wide and barely constrained itself"
        )
    if decades < MIN_DECADES:
        warnings.append(
            f"the x values span {decades:.2f} decades (a factor of {10**decades:.1f}) — "
            "too short a lever arm to separate exponents half a unit apart"
        )

    return PowerLawFit(
        exponent=exponent,
        exponent_stderr=stderr,
        amplitude=10.0**intercept,
        residual_rms=float(np.sqrt(ss_res / n)),
        r_squared=(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        n_points=n,
        n_dropped=dropped,
        decades=decades,
        locked=locked_models,
        warnings=tuple(warnings),
    )


# --------------------------------------------------------------------------- #
# turning an exponent into an answer
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Verdict:
    """A named candidate exponent, or an explicit refusal to choose."""

    conclusion: str
    detail: str
    decided: bool

    def __str__(self) -> str:
        return self.conclusion


def classify(
    fit: PowerLawFit,
    targets: Mapping[float, str],
    k_sigma: float = DEFAULT_K_SIGMA,
) -> Verdict:
    """Name the candidate exponent the interval singles out, if any.

    Deciding requires the interval to contain exactly one candidate. Containing
    two of them is not a tie to be broken by whichever is nearer — it is the
    finding that this series cannot tell them apart.
    """
    low, high = fit.interval(k_sigma)
    span = f"b ± {k_sigma:g}·SE = [{low:.2f}, {high:.2f}]"
    inside = {value: name for value, name in targets.items() if low <= value <= high}
    names = ", ".join(f"{name} (b={value:g})" for value, name in sorted(targets.items()))

    if not fit.has_lever_arm:
        return Verdict(
            conclusion="indeterminate",
            detail=(
                f"only {fit.decades:.2f} decades of x — the lever arm cannot separate "
                f"{names} whatever the fit says"
            ),
            decided=False,
        )
    if len(inside) == 1:
        value, name = next(iter(inside.items()))
        excluded = ", ".join(
            f"{other_name} (b={other:g})"
            for other, other_name in sorted(targets.items())
            if other != value
        )
        detail = f"{span} contains {value:g}"
        if excluded:
            detail += f" and excludes {excluded}"
        return Verdict(conclusion=name, detail=detail, decided=True)
    if len(inside) > 1:
        both = " and ".join(f"{value:g}" for value in sorted(inside))
        return Verdict(
            conclusion="indeterminate",
            detail=f"{span} contains both {both} — this series cannot separate them",
            decided=False,
        )
    return Verdict(
        conclusion="neither",
        detail=f"{span} excludes every candidate ({names})",
        decided=False,
    )
