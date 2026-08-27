"""Shot noise or detector noise, and the cross-check that catches the fakes."""

from __future__ import annotations

import math

import pytest

from tif_analyzer import noise_regime
from tif_analyzer.analysis import MetricResult, RunReport
from tif_analyzer.discovery import FileRecord

POWERS = [5.0, 10.0, 20.0, 40.0, 80.0]
BACKGROUND = 1000.0


def report_from(values_by_power, directory="/data") -> RunReport:
    """A finished RunReport with the values handed in — no files, no metric."""
    report = RunReport(config_snapshot={"directory": directory, "metric": "snr"})
    for power, values in values_by_power.items():
        record = FileRecord(
            __import__("pathlib").Path(directory) / f"sample_{power:g}mW.tif",
            "power",
            float(power),
            f"{power:g}",
        )
        report.results.append(
            MetricResult(record=record, values=dict(values), duration_s=0.0, from_cache=False)
        )
    return report


BACKGROUND_SIGMA = 31.0


def measurement(signal: float, signal_sigma: float) -> dict[str, float]:
    """One file's outputs.

    ``noise_std`` is the background area's sigma and stays put: that area holds
    no photons from the sample. ``signal_std`` is the spread inside the signal
    area, which is where the regime is visible.
    """
    return {
        "snr": signal / BACKGROUND_SIGMA,
        "signal_mean": BACKGROUND + signal,
        "noise_mean": BACKGROUND,
        "noise_std": BACKGROUND_SIGMA,
        "signal_std": signal_sigma,
    }


def shot(signal: float) -> float:
    """Poisson sigma for the signal area: its *total* mean, background included.

    The background's photons shot-noise exactly like the sample's, and they are
    inside the same box.
    """
    return math.sqrt(BACKGROUND + signal)


def shot_limited(gain=50.0) -> RunReport:
    """Signal linear in power, the signal area Poisson throughout."""
    return report_from(
        {power: measurement(gain * power, shot(gain * power)) for power in POWERS}
    )


def detector_limited(gain=50.0, sigma=12.0) -> RunReport:
    """Signal linear in power, every sigma fixed by the electronics."""
    return report_from({power: measurement(gain * power, sigma) for power in POWERS})


# --------------------------------------------------------------------------- #
# the two clean regimes
# --------------------------------------------------------------------------- #


def test_shot_limited_data_is_named_shot_limited():
    regime = noise_regime.analyse(shot_limited(), "snr", "power (mW)")
    assert regime.conclusion == "shot-noise limited"
    assert regime.noise.fit.exponent == pytest.approx(1.0, abs=1e-9)


def test_detector_limited_data_is_named_detector_limited():
    regime = noise_regime.analyse(detector_limited(), "snr", "power (mW)")
    assert regime.conclusion == "detector-noise limited"
    assert regime.noise.fit.exponent == pytest.approx(0.0, abs=1e-9)


def test_the_verdict_comes_from_the_signal_area_not_the_snr_exponent():
    """The whole reason the cross-check exists. This series is shot-limited and
    its SNR is dead linear in power, because the sigma the ratio divides by was
    measured where the sample is not."""
    regime = noise_regime.analyse(shot_limited(), "snr", "power (mW)")
    assert regime.snr.fit.exponent == pytest.approx(1.0, abs=1e-9)
    assert regime.snr.verdict.conclusion == "detector-noise limited"
    assert regime.conclusion == "shot-noise limited"
    assert any("none of the sample's photons" in reason for reason in regime.reasons)


def test_the_transfer_curve_uses_the_raw_mean_and_the_variance():
    """Total mean, not signal above background: the background's photons carry
    their own shot noise and sit inside the same box."""
    regime = noise_regime.analyse(shot_limited(), "snr", "power (mW)")
    assert min(regime.noise.xs) == pytest.approx(BACKGROUND + 50.0 * POWERS[0])
    assert min(regime.noise.ys) == pytest.approx(BACKGROUND + 50.0 * POWERS[0])


def test_sigma_against_signal_would_have_answered_neither():
    """Why the fit is variance-against-mean. The same Poisson series, fitted the
    other way, lands nowhere near 0.5 because sigma = √(background + signal) is
    not a power law in the signal."""
    from tif_analyzer.fitting import fit_power_law

    signals = [50.0 * power for power in POWERS]
    sigmas = [shot(s) for s in signals]
    assert fit_power_law(signals, sigmas).exponent < 0.35


def test_the_uniformity_assumption_is_stated():
    regime = noise_regime.analyse(shot_limited(), "snr", "power (mW)")
    assert any("uniform" in reason for reason in regime.reasons)


def test_the_response_fit_ignores_the_detector_offset():
    """The background is a constant the excitation never touches. Fitting the
    raw signal_mean, which sits on ~1000 counts of offset, reads a perfectly
    linear response as P^0.5 and then blames the sample."""
    regime = noise_regime.analyse(shot_limited(), "snr", "power (mW)")
    assert regime.response.fit.exponent == pytest.approx(1.0, abs=1e-9)
    assert min(regime.response.ys) == pytest.approx(50.0 * POWERS[0])


def test_the_linear_response_is_confirmed():
    regime = noise_regime.analyse(shot_limited(), "snr", "power (mW)")
    assert regime.response.verdict.decided
    assert any("the response is linear" in reason for reason in regime.reasons)


def test_agreement_by_an_independent_route_is_said_out_loud():
    """A detector-limited series is the case where the SNR exponent could have
    disagreed and does not."""
    regime = noise_regime.analyse(detector_limited(), "snr", "power (mW)")
    assert regime.snr.verdict.conclusion == "detector-noise limited"
    assert any("could have disagreed" in reason for reason in regime.reasons)


# --------------------------------------------------------------------------- #
# the case the cross-check exists for
# --------------------------------------------------------------------------- #


def test_a_bleaching_series_is_caught_and_explained():
    """Detector-limited throughout, but the sample bleaches, so the signal grows
    as P^0.5 and drags the SNR exponent down to 0.5 with it. Read alone, that
    exponent says shot-limited; the sample's own noise has not moved a count."""
    values = {
        power: measurement(2000.0 * power**0.5, 12.0) for power in POWERS
    }
    regime = noise_regime.analyse(report_from(values), "snr", "power (mW)")

    assert regime.snr.fit.exponent == pytest.approx(0.5, abs=1e-9)
    assert regime.snr.verdict.conclusion == "shot-noise limited"  # the trap
    assert regime.conclusion == "detector-noise limited"  # the truth
    assert any("sub-linear response" in reason for reason in regime.reasons)
    assert any("not demonstrably linear" in reason for reason in regime.reasons)


def test_a_non_linear_response_is_flagged_without_unseating_the_regime():
    """The regime fit never touches the parameter axis, so a bent response
    does not invalidate it."""
    values = {
        power: measurement(2000.0 * power**0.6, shot(2000.0 * power**0.6))
        for power in POWERS
    }
    regime = noise_regime.analyse(report_from(values), "snr", "power (mW)")
    assert not regime.response.verdict.decided
    assert regime.conclusion == "shot-noise limited"
    assert any("still stands" in reason for reason in regime.reasons)


def test_a_dim_series_on_a_bright_background_has_no_lever_arm():
    """The mean axis only spans what the sample adds on top of the background.
    With 1000 counts of background and a few hundred of signal, the transfer
    curve has nothing to work with — and says so instead of guessing."""
    values = {
        power: measurement(5.0 * power, shot(5.0 * power)) for power in POWERS
    }
    regime = noise_regime.analyse(report_from(values), "snr", "power (mW)")
    assert regime.noise.fit.decades < 0.45
    assert regime.conclusion == "indeterminate"
    assert any("lever arm" in warning for warning in regime.noise.fit.warnings)


def test_the_regime_survives_powers_too_close_to_fit():
    """Only 0.11 decades of power, so nothing can be read off the parameter
    axis — but a steep response still spreads the signal axis enough."""
    values = {
        power: measurement(50.0 * power**4, shot(50.0 * power**4))
        for power in (20.0, 22.0, 24.0, 26.0)
    }
    regime = noise_regime.analyse(report_from(values), "snr", "power (mW)")
    assert not regime.snr.verdict.decided
    assert regime.noise.verdict.decided
    assert regime.conclusion == "shot-noise limited"


# --------------------------------------------------------------------------- #
# metrics without the extra outputs
# --------------------------------------------------------------------------- #


def test_a_metric_without_the_snr_outputs_gets_the_headline_only():
    values = {power: {"std": 10.0 * power**0.5} for power in POWERS}
    regime = noise_regime.analyse(report_from(values), "std", "power (mW)")
    assert regime.snr.available
    assert regime.response is None and regime.noise is None
    assert regime.panels == [regime.snr]


def test_an_unfittable_headline_says_no_fit():
    values = {power: {"std": 10.0} for power in POWERS[:2]}
    regime = noise_regime.analyse(report_from(values), "std", "power (mW)")
    assert regime.conclusion == "no fit"
    assert not regime.snr.available
    assert "at least 3" in regime.snr.error


def test_negative_snr_points_are_dropped_not_fatal():
    """A signal area dimmer than the noise area gives a negative ratio, which
    has no logarithm. The other four points still make a fit."""
    values = {
        power: measurement(50.0 * power, shot(50.0 * power)) for power in POWERS
    }
    values[POWERS[0]]["snr"] = -2.0
    regime = noise_regime.analyse(report_from(values), "snr", "power (mW)")
    assert regime.snr.fit.n_dropped == 1
    assert regime.snr.fit.n_points == 4
    assert regime.noise.fit.n_points == 5  # the regime fit is untouched by it
