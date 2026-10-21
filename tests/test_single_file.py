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


def test_the_series_keys_say_why_they_are_off(unnamed_file, capsys):
    app = make_app(unnamed_file.parent, "q")
    app.state.update(single_file=unnamed_file)
    app.loop()
    text = capsys.readouterr().out
    assert "[r] run analysis  (single file mode — no parameter to plot against)" in text
    for key in ("[p] pattern", "[g] glob filter"):
        assert key in text


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


def test_the_run_key_does_nothing_in_the_mode(unnamed_file):
    """Labelled as blocked is not enough: a series with no x axis would get all
    the way to plotting before anything noticed."""
    app = make_app(unnamed_file.parent, ["r", "q"])
    app.state.update(single_file=unnamed_file)
    app.loop()
    assert app.state.report is None


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
