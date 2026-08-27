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


# --------------------------------------------------------------------------- #
# the parameter cutoff
# --------------------------------------------------------------------------- #


def saturating(knee=20.0, gain=2000.0) -> RunReport:
    """Linear up to *knee*, flat above it — a series that saturates at the top."""
    values = {}
    for power in POWERS:
        signal = gain * min(power, knee)
        values[power] = measurement(signal, shot(signal))
    return report_from(values)


def test_the_cutoff_drops_the_files_above_it():
    regime = noise_regime.analyse(shot_limited(), "snr", "power (mW)", max_param=20.0)
    assert regime.max_param == 20.0
    assert regime.n_excluded == 2
    assert regime.snr.fit.n_points == 3
    assert max(regime.snr.xs) == 20.0


def test_the_cutoff_drops_whole_files_from_every_fit():
    """One measurement feeds all three panels; a cutoff that reached only some
    of them would be two analyses reported as one."""
    regime = noise_regime.analyse(shot_limited(), "snr", "power (mW)", max_param=20.0)
    assert regime.response.fit.n_points == 3
    assert regime.noise.fit.n_points == 3


def test_the_excluded_points_are_kept_for_the_figure():
    regime = noise_regime.analyse(shot_limited(), "snr", "power (mW)", max_param=20.0)
    for panel in regime.panels:
        assert len(panel.excluded_xs) == 2
        assert len(panel.excluded_ys) == 2
    assert sorted(regime.snr.excluded_xs) == [40.0, 80.0]


def test_the_cutoff_is_reported_not_silent():
    regime = noise_regime.analyse(shot_limited(), "snr", "power (mW)", max_param=20.0)
    assert any("above 20 left out" in reason for reason in regime.reasons)


def test_no_cutoff_leaves_everything_in():
    regime = noise_regime.analyse(shot_limited(), "snr", "power (mW)")
    assert regime.max_param is None
    assert regime.n_excluded == 0
    assert all(panel.excluded_xs == () for panel in regime.panels)
    assert regime.snr.fit.n_points == len(POWERS)


def test_the_cutoff_rescues_a_saturating_series():
    """The reason the knob exists: above the knee the response flattens, the
    response fit reads sub-linear, and cutting the top restores it."""
    whole = noise_regime.analyse(saturating(), "snr", "power (mW)")
    assert not whole.response.verdict.decided
    assert whole.response.fit.exponent < 0.9

    trimmed = noise_regime.analyse(saturating(), "snr", "power (mW)", max_param=20.0)
    assert trimmed.response.verdict.decided
    assert trimmed.response.fit.exponent == pytest.approx(1.0, abs=1e-9)


def test_a_cutoff_that_leaves_too_few_points_says_so():
    regime = noise_regime.analyse(shot_limited(), "snr", "power (mW)", max_param=10.0)
    assert regime.conclusion == "no fit"
    assert "at least 3" in regime.noise.error
    assert len(regime.noise.excluded_xs) == 3  # still drawable


def test_a_cutoff_below_everything_is_not_a_crash():
    regime = noise_regime.analyse(shot_limited(), "snr", "power (mW)", max_param=0.1)
    assert regime.conclusion == "no fit"
    assert regime.n_excluded == len(POWERS)


def test_splitting_keeps_the_boundary_value():
    """``max_param`` is inclusive: a cutoff of 20 keeps the 20 mW file."""
    kept, dropped = noise_regime.split_at(shot_limited(), 20.0)
    assert [r.record.param_value for r in kept.results] == [5.0, 10.0, 20.0]
    assert [r.record.param_value for r in dropped.results] == [40.0, 80.0]


def test_an_undecided_regime_is_not_narrated_as_a_disagreement():
    """The SNR exponent can only disagree with a regime that was decided.
    Against an indeterminate one it is context, and claiming a disagreement
    would invent a finding out of a refusal to answer."""
    values = {
        power: measurement(5.0 * power, shot(5.0 * power)) for power in POWERS
    }
    regime = noise_regime.analyse(report_from(values), "snr", "power (mW)")
    assert not regime.noise.verdict.decided  # dim sample, no lever arm
    assert regime.snr.verdict.decided
    assert not any("points the other way" in reason for reason in regime.reasons)
    assert not any("sub-linear response" in reason for reason in regime.reasons)
    assert any("snr vs power (mW): b =" in reason for reason in regime.reasons)
