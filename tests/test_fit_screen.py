"""The fit screen and the figure it draws."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from tif_analyzer import noise_regime, plotting
from tif_analyzer.analysis import MetricResult, RunReport
from tif_analyzer.discovery import FileRecord
from tif_analyzer.state import AppState
from tif_analyzer.ui.fit_screen import FitScreen

from .test_ui import ScriptedReader

POWERS = [5.0, 10.0, 20.0, 40.0, 80.0]


def snr_report(directory: Path, regime: str = "shot") -> RunReport:
    """A shot-limited series (the signal area's sigma is √S) or a detector-
    limited one (every sigma fixed). The signal is linear in power either way,
    and the noise area's sigma is flat in both — it holds no sample photons.
    """
    report = RunReport(config_snapshot={"directory": str(directory), "metric": "snr"})
    for power in POWERS:
        signal = 500.0 * power
        # Poisson over the area's *total* mean, background included.
        signal_sigma = math.sqrt(1000.0 + signal) if regime == "shot" else 12.0
        report.results.append(
            MetricResult(
                record=FileRecord(
                    directory / f"sample_{power:g}mW.tif", "power", power, f"{power:g}"
                ),
                values={
                    "snr": signal / 31.0,
                    "signal_mean": 1000.0 + signal,
                    "noise_mean": 1000.0,
                    "noise_std": 31.0,
                    "signal_std": signal_sigma,
                },
                duration_s=0.0,
                from_cache=False,
            )
        )
    return report


def make_screen(tmp_path, keys, report=None, metric_key="snr"):
    state = AppState(directory=tmp_path, metric_key=metric_key)
    state.report = report if report is not None else snr_report(tmp_path)
    state.selected_series = "snr" if metric_key == "snr" else None
    screen = FitScreen(state, ScriptedReader(keys).read, clear=False)
    screen.param_label = "power (mW)"
    return screen


# --------------------------------------------------------------------------- #
# what the screen says
# --------------------------------------------------------------------------- #


def test_the_screen_reports_the_exponent_and_the_verdict(tmp_path, capsys):
    screen = make_screen(tmp_path, "b")
    regime = screen.compute()
    assert regime.noise.fit.exponent == pytest.approx(1.0, abs=1e-6)
    screen.run()
    text = capsys.readouterr().out
    assert "snr vs power (mW)" in text
    assert "b = +1.000" in text  # the transfer curve's exponent
    assert "VERDICT: shot-noise limited" in text
    assert "lever arm" in text


def test_the_locked_models_are_listed_with_their_residuals(tmp_path, capsys):
    make_screen(tmp_path, "b").run()
    text = capsys.readouterr().out
    assert "b = 0.5 locked: residual rms" in text
    assert "b = 1 locked: residual rms" in text


def test_the_cross_checks_appear(tmp_path, capsys):
    make_screen(tmp_path, "b").run()
    text = capsys.readouterr().out
    assert "signal above background vs power (mW)" in text
    assert "photon transfer curve" in text


def test_the_report_says_the_thresholds_are_provisional(tmp_path, capsys):
    make_screen(tmp_path, "b").run()
    assert "Provisional" in capsys.readouterr().out


def test_a_report_with_no_results_is_not_a_crash(tmp_path, capsys):
    state = AppState(directory=tmp_path, metric_key="snr")
    FitScreen(state, ScriptedReader("b").read, clear=False).run()
    assert "run the analysis first" in capsys.readouterr().out


def test_an_unfittable_series_is_reported_not_raised(tmp_path, capsys):
    report = snr_report(tmp_path)
    del report.results[2:]  # two points left
    make_screen(tmp_path, "b", report=report).run()
    text = capsys.readouterr().out
    assert "at least 3" in text
    assert "VERDICT: no fit" in text


# --------------------------------------------------------------------------- #
# the keys
# --------------------------------------------------------------------------- #


def test_the_axes_toggle(tmp_path, capsys):
    screen = make_screen(tmp_path, ["l", "b"])
    assert screen.log_axes is True
    screen.run()
    assert screen.log_axes is False
    assert "axes: linear" in capsys.readouterr().out


def test_plotting_falls_back_to_a_png_when_headless(tmp_path, capsys):
    make_screen(tmp_path, ["v", "b"]).run()
    saved = list(tmp_path.glob("tif-analyzer_fit_*.png"))
    assert len(saved) == 1
    assert "figure saved to" in capsys.readouterr().out


def test_the_report_can_be_saved(tmp_path, monkeypatch, capsys):
    from tif_analyzer.ui import prompts

    target = tmp_path / "fit.txt"
    monkeypatch.setattr(prompts, "prompt_export_path", lambda *a, **k: target)
    make_screen(tmp_path, ["o", "b"]).run()
    assert target.exists()
    assert "VERDICT: shot-noise limited" in target.read_text()


def test_the_screen_recomputes_from_the_report_every_draw(tmp_path):
    """No fourth piece of derived state: change the report, the screen follows."""
    screen = make_screen(tmp_path, "b")
    first = screen.compute()
    screen.state.report = snr_report(tmp_path, regime="detector")
    second = screen.compute()
    assert first.conclusion == "shot-noise limited"
    assert second.conclusion == "detector-noise limited"


# --------------------------------------------------------------------------- #
# the figure
# --------------------------------------------------------------------------- #


def figure_panels(tmp_path):
    regime = noise_regime.analyse(snr_report(tmp_path), "snr", "power (mW)")
    return [panel for panel in regime.panels if panel.available]


def test_the_figure_has_one_panel_per_fitted_series(tmp_path):
    figure = plotting.build_fit_figure(figure_panels(tmp_path), "title", "subtitle")
    assert len(figure.axes) == 3


def test_every_panel_labels_the_fit_and_the_locked_models(tmp_path):
    figure = plotting.build_fit_figure(figure_panels(tmp_path), "title", "subtitle")
    labels = [text.get_text() for text in figure.axes[0].get_legend().get_texts()]
    assert labels[0] == "measured"
    assert labels[1].startswith("fit: b = ")
    assert any("locked" in label for label in labels[2:])


def test_log_axes_are_the_default_and_can_be_turned_off(tmp_path):
    panels = figure_panels(tmp_path)
    assert plotting.build_fit_figure(panels, "t", "s").axes[0].get_xscale() == "log"
    assert (
        plotting.build_fit_figure(panels, "t", "s", log_axes=False).axes[0].get_xscale()
        == "linear"
    )


def test_a_figure_with_nothing_fitted_is_refused(tmp_path):
    with pytest.raises(ValueError, match="nothing to plot"):
        plotting.build_fit_figure([], "t", "s")


# --------------------------------------------------------------------------- #
# the parameter cutoff
# --------------------------------------------------------------------------- #


def test_the_cutoff_starts_unset_and_is_shown(tmp_path, capsys):
    make_screen(tmp_path, "b").run()
    assert "[x] max power (mW): none" in capsys.readouterr().out


def test_setting_the_cutoff_refits_and_says_so(tmp_path, monkeypatch, capsys):
    from tif_analyzer.ui import prompts

    monkeypatch.setattr(prompts, "prompt_number", lambda *a, **k: 20.0)
    screen = make_screen(tmp_path, ["x", "b"])
    screen.run()
    assert screen.max_param == 20.0
    text = capsys.readouterr().out
    assert "[x] max power (mW): 20" in text
    assert "cutoff: power (mW) ≤ 20" in text
    assert "2 file(s) left out" in text


def test_the_cutoff_can_be_cleared(tmp_path, monkeypatch):
    from tif_analyzer.ui import prompts

    monkeypatch.setattr(prompts, "prompt_number", lambda *a, **k: prompts.CLEARED)
    screen = make_screen(tmp_path, ["x", "b"])
    screen.max_param = 20.0
    screen.run()
    assert screen.max_param is None


def test_cancelling_the_prompt_leaves_the_cutoff_alone(tmp_path, monkeypatch):
    from tif_analyzer.ui import prompts

    monkeypatch.setattr(prompts, "prompt_number", lambda *a, **k: None)
    screen = make_screen(tmp_path, ["x", "b"])
    screen.max_param = 20.0
    screen.run()
    assert screen.max_param == 20.0


def test_a_cutoff_above_the_series_warns_that_it_does_nothing(tmp_path, monkeypatch, capsys):
    from tif_analyzer.ui import prompts

    monkeypatch.setattr(prompts, "prompt_number", lambda *a, **k: 500.0)
    make_screen(tmp_path, ["x", "b"]).run()
    assert "so nothing is left out" in capsys.readouterr().out


def test_the_prompt_shows_the_range_the_series_covers(tmp_path):
    screen = make_screen(tmp_path, "b")
    assert screen.param_range() == (5.0, 80.0)


def test_excluded_points_are_drawn_hollow_not_hidden(tmp_path):
    regime = noise_regime.analyse(
        snr_report(tmp_path), "snr", "power (mW)", max_param=20.0
    )
    figure = plotting.build_fit_figure(regime.panels, "t", "s")
    labels = [text.get_text() for text in figure.axes[0].get_legend().get_texts()]
    assert "above the cutoff (not fitted)" in labels


def test_the_fitted_curve_spans_the_excluded_points(tmp_path):
    """So the extrapolation shows how far they fall off the line."""
    regime = noise_regime.analyse(
        snr_report(tmp_path), "snr", "power (mW)", max_param=20.0
    )
    figure = plotting.build_fit_figure(regime.panels, "t", "s")
    fit_line = figure.axes[0].get_lines()[2]  # measured, excluded, then the fit
    assert max(fit_line.get_xdata()) == pytest.approx(80.0)


def test_plotting_with_a_cutoff_still_produces_the_figure(tmp_path, capsys):
    screen = make_screen(tmp_path, ["v", "b"])
    screen.max_param = 20.0
    screen.run()
    assert (tmp_path / "tif-analyzer_fit_snr.png").exists()
    assert "figure saved to" in capsys.readouterr().out
