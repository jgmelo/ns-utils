from __future__ import annotations

import csv

import matplotlib
import pytest

matplotlib.use("Agg")

from tif_analyzer import analysis, discovery, export, metrics, plotting  # noqa: E402


@pytest.fixture
def report(series_dir):
    records = discovery.discover(series_dir).records
    snapshot = analysis.make_snapshot(
        directory=series_dir,
        glob="*.tif",
        pattern=discovery.DEFAULT_PATTERN,
        metric_key="std",
        steps=[],
        page_policy="error",
        allow_duplicates=False,
        n_files=len(records),
    )
    return analysis.run_blocking(records, metrics.get("std"), snapshot=snapshot)


def test_csv_is_long_format_with_commented_snapshot(report, tmp_path):
    path = export.write_csv(report, tmp_path / "out.csv")
    lines = path.read_text(encoding="utf-8").splitlines()
    comments = [line for line in lines if line.startswith("#")]
    assert any("metric: std" in line for line in comments)
    assert any("pattern:" in line for line in comments)

    rows = list(csv.reader(line for line in lines if not line.startswith("#")))
    assert rows[0] == ["filename", "param_name", "param_value", "output_key", "value"]
    assert len(rows) == 5
    assert rows[1][0] == "medida_001_1mW.tif"
    assert rows[1][1] == "power"
    assert rows[1][3] == "std"
    assert [float(row[2]) for row in rows[1:]] == [1.0, 2.0, 5.0, 10.0]


def test_csv_suffix_is_added(report, tmp_path):
    assert export.write_csv(report, tmp_path / "out").name == "out.csv"


def test_failures_are_noted_in_the_header(report, tmp_path):
    report.failures.append((tmp_path / "broken.tif", ValueError("nope")))
    text = export.write_csv(report, tmp_path / "out.csv").read_text(encoding="utf-8")
    assert "# failures: 1" in text
    assert "broken.tif" in text


# -- plotting ---------------------------------------------------------------- #


def test_agg_backend_is_detected_as_non_interactive():
    info = plotting.backend_info()
    assert info.name.lower() == "agg"
    assert info.interactive is False
    assert "saved as PNG" in info.explanation


def test_headless_falls_back_to_saving_a_png(report, tmp_path):
    outcome = plotting.show_or_save(
        report, metrics.get("std"), param_name="power", param_unit="mW"
    )
    assert outcome.shown is False
    assert outcome.saved_to is not None
    assert outcome.saved_to.exists()
    assert outcome.saved_to.stat().st_size > 0


def test_explicit_save_path_is_used(report, tmp_path):
    target = tmp_path / "figs" / "plot.png"
    outcome = plotting.show_or_save(report, metrics.get("std"), save_to=target)
    assert outcome.saved_to == target
    assert target.exists()


def test_figure_labels_and_title_are_self_describing(report):
    figure = plotting.build_figure(
        report, metrics.get("std"), "std", param_name="power", param_unit="mW"
    )
    axes = figure.axes[0]
    assert axes.get_xlabel() == "power (mW)"
    assert axes.get_ylabel() == "Standard deviation"
    assert axes.get_title(loc="left") == report.config_snapshot["directory"].split("/")[-1]
    subtitle = [t.get_text() for t in axes.texts]
    assert any("4 files" in text for text in subtitle)


def test_points_are_plotted_sorted_by_parameter(report):
    figure = plotting.build_figure(report, metrics.get("std"), param_name="power")
    xdata = list(figure.axes[0].lines[0].get_xdata())
    assert xdata == sorted(xdata) == [1.0, 2.0, 5.0, 10.0]


def test_line_and_markers_are_both_drawn(report):
    line = plotting.build_figure(report, metrics.get("std")).axes[0].lines[0]
    assert line.get_marker() not in ("", "None", None)
    assert line.get_linewidth() > 0


def test_empty_report_refuses_to_plot():
    with pytest.raises(ValueError, match="nothing to plot"):
        plotting.build_figure(analysis.RunReport(), metrics.get("std"))


def test_axis_label_without_unit():
    assert plotting.axis_label("power", None) == "power"
    assert plotting.axis_label("power", "mW") == "power (mW)"
