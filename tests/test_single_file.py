"""Single-file mode: naming one frame instead of matching a series.

The series contract — a regex with one named group whose capture parses as a
float — is right for a plot and wrong for opening one frame to look at it. These
tests pin the escape hatch and, just as importantly, pin what it refuses.
"""

from __future__ import annotations

import numpy as np
import pytest

from tif_analyzer import discovery
from tif_analyzer.discovery import DiscoveryError
from tif_analyzer.state import AppState

from .test_ui import make_app


@pytest.fixture
def unnamed_file(tmp_path, make_tif):
    """A frame whose name holds no number at all — unreachable before."""
    values = np.full((6, 6), 100, dtype=np.uint16)
    values[0, 0] = 900
    return make_tif(tmp_path / "data", "poliest.tif", values)


# --------------------------------------------------------------------------- #
# the report
# --------------------------------------------------------------------------- #


def test_one_named_file_becomes_one_record(unnamed_file):
    report = discovery.single_file_report(unnamed_file)
    assert [r.path for r in report.records] == [unnamed_file]
    assert report.n_globbed == 1
    assert report.n_parsed == 1


def test_the_record_carries_no_parameter(unnamed_file):
    record = discovery.single_file_report(unnamed_file).records[0]
    assert record.param_value is None
    assert record.param_name is None
    assert record.has_parameter is False


def test_a_matched_record_still_carries_one(series_dir):
    record = discovery.discover(series_dir, "*.tif", r"(?P<power>[\d.]+)mW").records[0]
    assert record.has_parameter is True
    assert record.param_name == "power"


def test_no_glob_no_regex_no_duplicate_check(unnamed_file):
    report = discovery.single_file_report(unnamed_file)
    assert report.pattern == ""
    assert report.duplicates == {}
    assert report.unmatched == [] and report.parse_failures == []


def test_a_missing_file_is_refused(tmp_path):
    with pytest.raises(DiscoveryError, match="not a file"):
        discovery.single_file_report(tmp_path / "nope.tif")


def test_a_directory_is_refused(tmp_path):
    with pytest.raises(DiscoveryError, match="not a file"):
        discovery.single_file_report(tmp_path)


def test_parameterless_records_never_collide(unnamed_file, series_dir):
    """Two files with no parameter are not duplicates of each other — there is
    no value for them to share."""
    records = [
        discovery.single_file_report(unnamed_file).records[0],
        discovery.single_file_report(unnamed_file).records[0],
    ]
    assert discovery.find_duplicates(records) == {}


# --------------------------------------------------------------------------- #
# the state
# --------------------------------------------------------------------------- #


def test_entering_the_mode_invalidates_results(series_dir, unnamed_file):
    state = AppState(directory=series_dir)
    state.report = object()
    assert state.update(single_file=unnamed_file) is True
    assert state.report is None
    assert state.single_file_mode is True


def test_the_series_settings_are_kept_while_the_mode_is_on(series_dir, unnamed_file):
    state = AppState(directory=series_dir, pattern=r"(?P<power>\d+)mW", glob="*.tiff")
    state.update(single_file=unnamed_file)
    assert state.pattern == r"(?P<power>\d+)mW"
    assert state.glob == "*.tiff"


# --------------------------------------------------------------------------- #
# the screen
# --------------------------------------------------------------------------- #


def test_the_header_swaps_pattern_and_glob_for_the_filename(unnamed_file, capsys):
    app = make_app(unnamed_file.parent, "q")
    app.state.update(single_file=unnamed_file)
    app.loop()
    text = capsys.readouterr().out
    assert "file:     poliest.tif" in text
    assert "pattern:" not in text
    assert "glob:" not in text
    assert "(1 file)" in text


def test_the_series_shaped_keys_say_why_they_are_off(unnamed_file, capsys):
    """Only the outputs that need an x axis go quiet. The metric still has a
    number to give, so [r] stays available and renames itself."""
    app = make_app(unnamed_file.parent, "q")
    app.state.update(single_file=unnamed_file)
    app.loop()
    text = capsys.readouterr().out
    reason = "single file mode — one point is not a series"
    for label in ("[p] pattern", "[g] glob filter", "[v] view plot",
                  "[e] export CSV", "[f] fit a power law"):
        assert f"{label}  ({reason})" in text
    assert "[r] measure this frame" in text
    assert f"[r] measure this frame  ({reason})" not in text


def test_the_key_label_names_the_action_not_the_state(series_dir, unnamed_file, capsys):
    app = make_app(series_dir, "q")
    app.loop()
    assert "[1] single file" in capsys.readouterr().out

    app = make_app(series_dir, "q")
    app.state.update(single_file=unnamed_file)
    app.loop()
    assert "[1] back to the series" in capsys.readouterr().out


def test_naming_a_file_enters_the_mode(series_dir, unnamed_file, monkeypatch, capsys):
    monkeypatch.setattr(prompts_module(), "prompt_single_file", lambda *a: unnamed_file)
    app = make_app(series_dir, ["1", "q"])
    app.loop()
    assert app.state.single_file == unnamed_file
    assert "✓ single file: poliest.tif" in capsys.readouterr().out


def test_pressing_it_again_leaves_the_mode(series_dir, unnamed_file, capsys):
    app = make_app(series_dir, ["1", "q"])
    app.state.update(single_file=unnamed_file)
    app.loop()
    assert app.state.single_file is None
    assert "back to the series" in capsys.readouterr().out


def test_leaving_restores_the_series_exactly(series_dir, unnamed_file):
    app = make_app(series_dir, ["1", "q"])
    app.state.update(single_file=unnamed_file)
    app.loop()
    assert app.state.records is not None
    assert [r.param_value for r in app.state.records] == [1.0, 2.0, 5.0, 10.0]


def test_cancelling_the_prompt_changes_nothing(series_dir, monkeypatch):
    monkeypatch.setattr(prompts_module(), "prompt_single_file", lambda *a: None)
    app = make_app(series_dir, ["1", "q"])
    app.loop()
    assert app.state.single_file is None
    assert app.state.has_records


def test_the_single_frame_tools_accept_a_parameterless_file(unnamed_file):
    """The point of the mode: [i] and friends read from records, and until now
    a name without a number could not produce one."""
    app = make_app(unnamed_file.parent, "q")
    app.state.update(single_file=unnamed_file)
    app.ensure_discovery()
    assert [r.path for r in app.inspectable_records()] == [unnamed_file]
    assert "i" not in app.blockers()


def test_a_vanished_file_is_reported_not_crashed(unnamed_file, capsys):
    app = make_app(unnamed_file.parent, "q")
    app.state.update(single_file=unnamed_file)
    unnamed_file.unlink()
    app.loop()
    text = capsys.readouterr().out
    assert "not a file" in text
    assert "(not found)" in text


def prompts_module():
    from tif_analyzer.ui import prompts

    return prompts


# --------------------------------------------------------------------------- #
# a blocked key must not act
# --------------------------------------------------------------------------- #


def test_the_run_key_measures_the_frame(unnamed_file):
    app = make_app(unnamed_file.parent, ["r", "", "q"])
    app.state.update(single_file=unnamed_file)
    app.loop()
    assert app.state.report is not None
    assert len(app.state.report.results) == 1
    assert "std" in app.state.report.results[0].values


def test_the_measurement_is_shown_in_full_and_in_the_header(unnamed_file, capsys):
    app = make_app(unnamed_file.parent, ["r", "", "q"])
    app.state.update(single_file=unnamed_file)
    app.loop()
    text = capsys.readouterr().out
    assert "Standard deviation:" in text          # the readout
    assert "measured: std " in text               # the header line afterwards


def test_the_header_says_to_press_r_before_measuring(unnamed_file, capsys):
    app = make_app(unnamed_file.parent, "q")
    app.state.update(single_file=unnamed_file)
    app.loop()
    assert "measured: — press [r]" in capsys.readouterr().out


def test_every_output_is_shown_not_just_the_selected_series(series_dir, make_tif):
    """With no plot to pick a series for, the numbers the ratio was built from
    are exactly what you want next to it."""
    import numpy as np

    from tif_analyzer.regions import Rect, RegionSet

    rng = np.random.default_rng(0)
    values = rng.integers(90, 111, size=(20, 20)).astype(np.uint16)
    values[2:8, 2:8] += 800
    path = make_tif(series_dir, "poliest.tif", values)

    app = make_app(series_dir, ["r", "", "q"])
    app.state.update(
        metric_key="snr",
        single_file=path,
        regions=RegionSet(
            signal=Rect(2, 2, 6, 6),
            noise=Rect(10, 10, 8, 8),
            image_shape=(20, 20),
        ),
    )
    app.loop()
    measured = app.state.report.results[0].values
    assert set(measured) == {"snr", "signal_mean", "noise_mean", "noise_std", "signal_std"}


def test_measuring_is_blocked_until_the_areas_exist(series_dir, make_tif, capsys):
    import numpy as np

    path = make_tif(series_dir, "poliest.tif", np.full((20, 20), 100, dtype=np.uint16))
    app = make_app(series_dir, ["r", "q"])
    app.state.update(metric_key="snr", single_file=path)
    app.loop()
    assert app.state.report is None
    assert "[r] measure this frame  (metric snr needs the areas — press [a])" in capsys.readouterr().out


def test_a_failing_frame_is_reported_not_crashed(series_dir, tmp_path, capsys):
    broken = series_dir / "poliest.tif"
    broken.write_bytes(b"not a tiff")
    app = make_app(series_dir, ["r", "", "q"])
    app.state.update(single_file=broken)
    app.loop()
    text = capsys.readouterr().out
    assert "✗ poliest.tif" in text


def test_the_pattern_and_glob_keys_do_nothing_in_the_mode(
    unnamed_file, monkeypatch
):
    monkeypatch.setattr(prompts_module(), "prompt_pattern", lambda *a: "CHANGED")
    monkeypatch.setattr(prompts_module(), "prompt_glob", lambda *a: "CHANGED")
    app = make_app(unnamed_file.parent, ["p", "g", "q"])
    app.state.update(single_file=unnamed_file)
    before = (app.state.pattern, app.state.glob)
    app.loop()
    assert (app.state.pattern, app.state.glob) == before


def test_the_duplicates_toggle_does_nothing_in_the_mode(unnamed_file, capsys):
    app = make_app(unnamed_file.parent, ["!", "q"])
    app.state.update(single_file=unnamed_file)
    app.loop()
    assert app.state.allow_duplicates is False
    assert "[!] allow duplicates: OFF  (single file mode" in capsys.readouterr().out
