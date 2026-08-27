"""Power-law fitting: the exponent, the error bar, and the refusals."""

from __future__ import annotations

import math

import numpy as np
import pytest

from tif_analyzer.fitting import (
    DEFAULT_K_SIGMA,
    MIN_DECADES,
    FitError,
    classify,
    fit_power_law,
    locked_fit,
)

POWERS = [5.0, 10.0, 20.0, 40.0, 80.0]
TARGETS = {0.5: "shot-noise limited", 1.0: "detector-noise limited"}


def curve(exponent, amplitude=3.0, xs=None):
    xs = POWERS if xs is None else xs
    return list(xs), [amplitude * x**exponent for x in xs]


# --------------------------------------------------------------------------- #
# recovering what was planted
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("exponent", [0.5, 1.0, 0.72, -0.4])
def test_a_planted_exponent_comes_back(exponent):
    xs, ys = curve(exponent)
    fit = fit_power_law(xs, ys)
    assert fit.exponent == pytest.approx(exponent, abs=1e-9)
    assert fit.amplitude == pytest.approx(3.0, rel=1e-9)
    assert fit.r_squared == pytest.approx(1.0)


def test_a_perfect_fit_has_no_uncertainty():
    fit = fit_power_law(*curve(0.5))
    assert fit.exponent_stderr == pytest.approx(0.0, abs=1e-12)
    assert fit.residual_rms == pytest.approx(0.0, abs=1e-12)


def test_a_noisy_exponent_lands_within_its_own_error_bar():
    rng = np.random.default_rng(0)
    xs, clean = curve(0.5)
    ys = [y * float(rng.normal(1.0, 0.05)) for y in clean]
    fit = fit_power_law(xs, ys)
    assert abs(fit.exponent - 0.5) < DEFAULT_K_SIGMA * fit.exponent_stderr


def test_the_lever_arm_is_what_shrinks_the_error_bar():
    """Same scatter, wider spread of x: the exponent gets better determined."""
    rng = np.random.default_rng(1)

    def noisy(xs):
        clean = [3.0 * x**0.5 for x in xs]
        return fit_power_law(xs, [y * float(rng.normal(1.0, 0.05)) for y in clean])

    narrow = noisy([20.0, 24.0, 28.0, 32.0, 36.0])
    wide = noisy([5.0, 12.0, 25.0, 50.0, 100.0])
    assert wide.decades > narrow.decades
    assert wide.exponent_stderr < narrow.exponent_stderr / 3


def test_the_standard_error_is_scatter_over_lever_arm():
    """The formula, computed by hand on the same points."""
    xs = [1.0, 10.0, 100.0, 1000.0]
    ys = [1.0, 3.0, 11.0, 28.0]
    fit = fit_power_law(xs, ys)

    log_x, log_y = np.log10(xs), np.log10(ys)
    slope, intercept = np.polyfit(log_x, log_y, 1)
    residuals = log_y - (slope * log_x + intercept)
    sigma = math.sqrt(float(np.sum(residuals**2)) / (len(xs) - 2))
    lever = math.sqrt(float(np.sum((log_x - log_x.mean()) ** 2)))
    assert fit.exponent_stderr == pytest.approx(sigma / lever)


def test_decades_is_the_span_of_x():
    fit = fit_power_law(*curve(0.5, xs=[1.0, 10.0, 100.0]))
    assert fit.decades == pytest.approx(2.0)


# --------------------------------------------------------------------------- #
# what it refuses to do
# --------------------------------------------------------------------------- #


def test_two_points_are_refused():
    """A line through two points is exact: no residual, no error bar, no
    information about whether the model fits."""
    with pytest.raises(FitError, match="at least 3"):
        fit_power_law([1.0, 2.0], [1.0, 2.0])


def test_x_all_the_same_is_refused():
    with pytest.raises(FitError, match="no slope"):
        fit_power_law([4.0, 4.0, 4.0], [1.0, 2.0, 3.0])


def test_mismatched_lengths_are_refused():
    with pytest.raises(FitError):
        fit_power_law([1.0, 2.0, 3.0], [1.0, 2.0])


def test_non_positive_points_are_dropped_and_counted():
    """A negative SNR has no logarithm, and dropping it silently would leave the
    fit describing fewer points than the plot shows."""
    xs = [5.0, 10.0, 20.0, 40.0]
    ys = [-1.0, 3.0, 4.2, 6.0]
    fit = fit_power_law(xs, ys)
    assert fit.n_points == 3
    assert fit.n_dropped == 1
    assert any("positive" in warning for warning in fit.warnings)


def test_dropping_too_many_points_becomes_a_refusal():
    with pytest.raises(FitError, match="dropped"):
        fit_power_law([5.0, 10.0, 20.0, 40.0], [-1.0, 0.0, 4.2, -3.0])


def test_three_points_warn_about_the_lone_degree_of_freedom():
    fit = fit_power_law(*curve(0.5, xs=[5.0, 25.0, 125.0]))
    assert fit.n_points == 3
    assert any("degree of freedom" in warning for warning in fit.warnings)


def test_a_short_lever_arm_warns():
    fit = fit_power_law(*curve(0.5, xs=[20.0, 24.0, 28.0, 32.0]))
    assert fit.decades < MIN_DECADES
    assert any("lever arm" in warning for warning in fit.warnings)
    assert not fit.has_lever_arm


# --------------------------------------------------------------------------- #
# the locked models
# --------------------------------------------------------------------------- #


def test_the_right_locked_model_wins_on_residual():
    xs, ys = curve(0.5)
    fit = fit_power_law(xs, ys, locked=(0.5, 1.0))
    assert fit.locked[0.5].residual_rms < fit.locked[1.0].residual_rms
    assert fit.locked[0.5].residual_rms == pytest.approx(0.0, abs=1e-12)


def test_a_locked_model_reproduces_the_data_it_matches():
    xs, ys = curve(0.5, amplitude=7.0)
    fit = fit_power_law(xs, ys, locked=(0.5,))
    assert fit.locked[0.5].amplitude == pytest.approx(7.0)
    assert fit.predict_locked(xs, 0.5) == pytest.approx(ys)


def test_the_free_fit_never_loses_to_a_locked_one():
    """One more free parameter cannot make least squares worse."""
    rng = np.random.default_rng(2)
    xs, clean = curve(0.72)
    ys = [y * float(rng.normal(1.0, 0.08)) for y in clean]
    fit = fit_power_law(xs, ys, locked=(0.5, 1.0))
    assert fit.residual_rms <= min(model.residual_rms for model in fit.locked.values())


def test_locked_fit_is_the_mean_offset():
    log_x = np.log10(POWERS)
    log_y = 0.5 * log_x + 0.6
    intercept, rms = locked_fit(log_x, log_y, 0.5)
    assert intercept == pytest.approx(0.6)
    assert rms == pytest.approx(0.0, abs=1e-12)


def test_predicting_an_unfitted_locked_model_is_an_error():
    fit = fit_power_law(*curve(0.5), locked=(0.5,))
    with pytest.raises(FitError, match="no locked model"):
        fit.predict_locked(POWERS, 1.0)


# --------------------------------------------------------------------------- #
# the verdict
# --------------------------------------------------------------------------- #


def test_a_clean_square_root_is_called_shot_noise_limited():
    rng = np.random.default_rng(3)
    xs, clean = curve(0.5)
    ys = [y * float(rng.normal(1.0, 0.02)) for y in clean]
    verdict = classify(fit_power_law(xs, ys, locked=tuple(TARGETS)), TARGETS)
    assert verdict.conclusion == "shot-noise limited"
    assert verdict.decided
    assert "excludes" in verdict.detail


def test_a_clean_line_is_called_detector_noise_limited():
    rng = np.random.default_rng(4)
    xs, clean = curve(1.0)
    ys = [y * float(rng.normal(1.0, 0.02)) for y in clean]
    verdict = classify(fit_power_law(xs, ys, locked=tuple(TARGETS)), TARGETS)
    assert verdict.conclusion == "detector-noise limited"
    assert verdict.decided


def test_an_interval_containing_both_refuses_to_choose():
    """0.64 is nearer 0.5, and that is not a reason to answer 0.5.

    Four points scattered by ±20%: a completely ordinary series, and it cannot
    tell the two regimes apart.
    """
    xs = [5.0, 10.0, 20.0, 40.0]
    ys = [3.0 * x**0.75 * p for x, p in zip(xs, [1.25, 0.8, 1.2, 0.85])]
    fit = fit_power_law(xs, ys, locked=tuple(TARGETS))
    verdict = classify(fit, TARGETS)
    assert fit.contains(0.5) and fit.contains(1.0)
    assert verdict.conclusion == "indeterminate"
    assert not verdict.decided
    assert "cannot separate" in verdict.detail


def test_an_exponent_near_neither_says_so():
    xs, ys = curve(2.0)
    verdict = classify(fit_power_law(xs, ys, locked=tuple(TARGETS)), TARGETS)
    assert verdict.conclusion == "neither"
    assert not verdict.decided


def test_a_short_lever_arm_withholds_the_verdict_however_clean_the_fit():
    """The refusal is about the geometry of the x axis, not about the scatter."""
    xs, ys = curve(0.5, xs=[20.0, 24.0, 28.0, 32.0])
    fit = fit_power_law(xs, ys, locked=tuple(TARGETS))
    assert fit.r_squared == pytest.approx(1.0)
    verdict = classify(fit, TARGETS)
    assert verdict.conclusion == "indeterminate"
    assert "decades" in verdict.detail


def test_the_interval_widens_with_k():
    fit = fit_power_law(*curve(0.5, xs=[5.0, 11.0, 23.0, 47.0]))
    narrow = fit.interval(1.0)
    wide = fit.interval(3.0)
    assert wide[0] <= narrow[0] and wide[1] >= narrow[1]
