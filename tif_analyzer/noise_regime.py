"""Which noise dominates a series: shot noise, or the detector.

The physics is one line. The signal is proportional to the excitation power,
``S ∝ P``. Photons arrive as a Poisson process, where the variance *is* the
mean, so photon noise grows as ``√S``. Detector noise — dark current, the
amplifier, digitisation — is there with the laser off and does not scale with
the signal at all. So::

    shot-noise limited      σ ∝ S^0.5   →   SNR = S/σ ∝ P^0.5
    detector-noise limited  σ ∝ S^0.0   →   SNR = S/σ ∝ P^1.0

**But which sigma?** This is the part that decides whether the whole diagnostic
works. ``snr`` divides by ``noise_std``, measured in the *noise* area — an area
chosen precisely because the sample is not in it. That sigma holds the
detector's noise and the background's own photons, and none of the sample's. It
therefore cannot grow when the sample gets brighter, no matter how shot-limited
the image is, and ``SNR = S/σ_background ∝ P`` comes out linear by construction.
Verified on synthetic Poisson frames: sigma in the noise area sat at 31 counts
across a 16× range of power while sigma inside the signal area went 36 → 75.

So ``SNR`` vs power is *not* the regime diagnostic. It is the scaling of the
figure of merit, worth knowing and worth plotting, but its exponent is mostly a
statement about where the noise box was drawn.

The diagnostic is built from ``signal_std`` — the spread *inside* the signal
area, the only sigma here that carries the sample's photon statistics — and it
is plotted the way detector people plot it, as a **photon transfer curve**:
variance against mean, both measured in the same area.

Variance against mean, and not sigma against signal-above-background, for a
reason worth stating. Poisson says *variance = mean*, so::

    shot-limited      var ∝ mean^1.0
    detector-limited  var ∝ mean^0.0   (var is σ_read², fixed)

Sigma against signal-above-background looks equivalent and is not: the signal
area contains the background's photons too, so ``σ = √(background + signal)``,
which is not a power law in the signal at all. Fitted anyway it returns
something like 0.27 for a perfectly Poisson series and the verdict comes back
*neither* — measured, on synthetic Poisson frames, before this was changed to
the transfer curve, which returned 0.99 on the same frames.

Three fits, and the third is the one that decides:

1. ``snr`` vs parameter — how the figure of merit scales. Context.
2. ``signal above background`` vs parameter — the precondition. Worth 1.0 if the
   response is linear; saturation, photobleaching or a power scale that is not
   what reached the sample all show up here.
3. ``signal variance`` vs ``signal mean`` — **the regime**. Worth 1.0 (shot) or
   0.0 (detector), and it never touches the parameter axis, so a wrong power
   reading cannot corrupt it.

Two caveats travel with the third fit. The spread inside the signal area is only
noise if that area is uniform: drawn across structure, the fit measures the
sample rather than the photon statistics. And the mean is raw counts, so an
electronic offset (a bias level, distinct from the background's photons) pushes
the exponent below 1 — separating that would need a dark frame, which the tool
does not ask for. Where the transfer curve *is* clean, the amplitude of the
locked ``b = 1`` model is the system gain in counts per electron.

Nothing here prints, and nothing here reads a file: it works from a finished
:class:`~tif_analyzer.analysis.RunReport`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

from .analysis import RunReport
from .fitting import DEFAULT_K_SIGMA, FitError, PowerLawFit, Verdict, classify, fit_power_law

SHOT = 0.5
DETECTOR = 1.0
#: Photon transfer curve: Poisson makes the variance equal the mean, so a
#: shot-limited area has ``var ∝ mean``. Detector noise is a fixed variance.
SHOT_VS_MEAN = 1.0
DETECTOR_VS_MEAN = 0.0
LINEAR = 1.0

SNR_TARGETS = {SHOT: "shot-noise limited", DETECTOR: "detector-noise limited"}
TRANSFER_TARGETS = {
    SHOT_VS_MEAN: "shot-noise limited",
    DETECTOR_VS_MEAN: "detector-noise limited",
}

#: The metric whose outputs the cross-checks need.
SNR_METRIC = "snr"


@dataclass(frozen=True)
class Fitted:
    """One fit and its reading, or the reason there is no fit."""

    label: str
    x_label: str
    y_label: str
    xs: tuple[float, ...] = ()
    ys: tuple[float, ...] = ()
    fit: PowerLawFit | None = None
    verdict: Verdict | None = None
    error: str | None = None

    @property
    def available(self) -> bool:
        return self.fit is not None


@dataclass(frozen=True)
class RegimeReport:
    """The three fits and what they add up to.

    ``noise`` is the one that decides; ``snr`` is context. See the module
    docstring for why it is that way round.
    """

    snr: Fitted
    response: Fitted | None  # signal above background vs parameter
    noise: Fitted | None  # signal_std vs signal above background
    conclusion: str
    reasons: tuple[str, ...]

    @property
    def panels(self) -> list[Fitted]:
        return [f for f in (self.snr, self.response, self.noise) if f is not None]


def _fit_one(
    label: str,
    xs: Sequence[float],
    ys: Sequence[float],
    x_label: str,
    y_label: str,
    targets: dict[float, str],
    k_sigma: float,
) -> Fitted:
    try:
        fit = fit_power_law(xs, ys, locked=tuple(targets))
    except FitError as exc:
        return Fitted(label=label, x_label=x_label, y_label=y_label, error=str(exc))
    return Fitted(
        label=label,
        x_label=x_label,
        y_label=y_label,
        xs=tuple(float(v) for v in xs),
        ys=tuple(float(v) for v in ys),
        fit=fit,
        verdict=classify(fit, targets, k_sigma),
    )


def analyse(
    report: RunReport,
    series_key: str,
    param_label: str = "parameter",
    k_sigma: float = DEFAULT_K_SIGMA,
) -> RegimeReport:
    """Fit *series_key* against the parameter, with the cross-checks when the
    metric's other outputs make them possible."""
    xs, ys = report.series(series_key)
    headline = _fit_one(
        f"{series_key} vs {param_label}",
        xs,
        ys,
        param_label,
        series_key,
        SNR_TARGETS,
        k_sigma,
    )

    response = noise = None
    if _has_outputs(report, ("signal_mean", "noise_mean", "signal_std")):
        # Above background, like the noise fit below. The raw signal mean carries
        # the detector's offset, which is a constant the excitation never
        # touches: at low power it dominates and drags the exponent towards zero,
        # so a perfectly linear response reads as sub-linear.
        response_x, response_y = _series_of(report, _signal_above_background)
        response = _fit_one(
            f"signal above background vs {param_label}",
            response_x,
            response_y,
            param_label,
            "signal_mean − noise_mean (counts)",
            {LINEAR: "linear in the parameter"},
            k_sigma,
        )
        mean, variance = _transfer_curve(report)
        noise = _fit_one(
            "photon transfer curve: signal variance vs signal mean",
            mean,
            variance,
            "signal_mean (counts)",
            "variance inside the signal area (counts²)",
            TRANSFER_TARGETS,
            k_sigma,
        )

    conclusion, reasons = _combine(headline, response, noise)
    return RegimeReport(
        snr=headline, response=response, noise=noise, conclusion=conclusion, reasons=reasons
    )


def _has_outputs(report: RunReport, keys: Sequence[str]) -> bool:
    if not report.results:
        return False
    values = report.results[0].values
    return all(key in values for key in keys)


def _signal_above_background(result) -> float:
    """The photons from the sample, with the detector's offset taken off.

    Every fit here uses this rather than ``signal_mean``: the offset scales with
    nothing, so leaving it in bends both the response exponent and the noise
    exponent towards zero.
    """
    return result.values["signal_mean"] - result.values["noise_mean"]


def _series_of(report: RunReport, value_of) -> tuple[list[float], list[float]]:
    """``(parameter, value)`` pairs, sorted by parameter."""
    points = sorted(
        ((result.record.param_value, value_of(result)) for result in report.results),
        key=lambda pair: pair[0],
    )
    return [p[0] for p in points], [p[1] for p in points]


def _transfer_curve(report: RunReport) -> tuple[list[float], list[float]]:
    """``(mean, variance)`` inside the signal area, one pair per file.

    The mean is raw and total — the background's photons shot-noise exactly like
    the sample's, and taking them off the mean while leaving their variance in
    would break the equality the curve is built on. The parameter axis does not
    appear at all, which is what makes this fit immune to a wrong power scale.

    ``signal_std``, not ``noise_std``: the noise area holds no photons from the
    sample, so its variance is flat in the sample's brightness by construction
    and would report every series as detector-limited.
    """
    pairs = [
        (result.values["signal_mean"], result.values["signal_std"] ** 2)
        for result in report.results
    ]
    pairs.sort(key=lambda pair: pair[0])
    return [pair[0] for pair in pairs], [pair[1] for pair in pairs]


def _combine(
    snr: Fitted, response: Fitted | None, noise: Fitted | None
) -> tuple[str, tuple[str, ...]]:
    """Read the three fits together.

    The regime comes from *noise* — the only fit whose sigma contains photons
    from the sample. The SNR exponent is reported as context, and where the two
    differ the difference is explained rather than split.
    """
    reasons: list[str] = []

    if noise is not None and noise.available and noise.verdict is not None:
        conclusion = noise.verdict.conclusion
        reasons.append(f"{noise.label}: {conclusion} — {noise.verdict.detail}")
        reasons.append(
            "this is the fit that decides: the spread inside the signal area is the "
            "only sigma here that contains the sample's own photons"
        )
        gain = noise.fit.locked.get(SHOT_VS_MEAN)
        if gain is not None and conclusion == TRANSFER_TARGETS[SHOT_VS_MEAN]:
            reasons.append(
                f"the b = 1 model's amplitude is the system gain: {gain.amplitude:.3g} "
                "counts per electron, valid only if there is no electronic offset in "
                "these counts"
            )
    elif snr.available and snr.verdict is not None:
        conclusion = snr.verdict.conclusion
        reasons.append(
            "no signal-area sigma in this metric, so the regime is being read off the "
            "SNR exponent alone — see the caveat below"
        )
    elif noise is not None and noise.error:
        return "no fit", (f"{noise.label}: {noise.error}",)
    else:
        return "no fit", (snr.error or "nothing could be fitted",)

    if snr.available and snr.verdict is not None:
        reasons.append(f"{snr.label}: b = {snr.fit.exponent:.2f} ± {snr.fit.exponent_stderr:.2f}")
        if noise is not None and noise.available and noise.verdict is not None:
            if snr.verdict.decided and snr.verdict.conclusion != conclusion:
                # The two directions of disagreement have different causes, and
                # naming the wrong one sends you looking in the wrong place.
                if conclusion == SNR_TARGETS[SHOT]:
                    reasons.append(
                        "the SNR exponent points the other way, which is expected rather "
                        "than alarming when the noise area sits on plain background: its "
                        "sigma holds none of the sample's photons, so SNR = S/σ_background "
                        "comes out linear in the parameter whatever the sample's noise is "
                        "doing. Drawn there, this line will keep saying b ≈ 1"
                    )
                else:
                    reasons.append(
                        "the SNR exponent looks shot-limited while the sample's own noise "
                        "is not — that is what a sub-linear response does to it "
                        "(photobleaching, saturation, or a power axis that is not what "
                        "reached the sample). Check the response fit"
                    )
            elif snr.verdict.decided:
                reasons.append("the SNR exponent agrees, by a route that could have disagreed")

    if response is not None and response.available and response.fit is not None:
        exponent = f"P^{response.fit.exponent:.2f} ± {response.fit.exponent_stderr:.2f}"
        if response.verdict is not None and response.verdict.decided:
            reasons.append(f"signal ∝ {exponent} — the response is linear in the parameter")
        else:
            reasons.append(
                f"signal ∝ {exponent} — the response is not demonstrably linear "
                "(saturation, photobleaching, or a power axis that is not what reached the "
                "sample). The regime above still stands: it never uses the parameter"
            )

    if noise is not None and noise.available:
        reasons.append(
            "assumes the signal area is uniform — spread across real structure is not noise"
        )
    return conclusion, tuple(reasons)
