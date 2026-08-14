from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

from tif_analyzer import app as app_mod, config as config_mod, discovery  # noqa: E402
from tif_analyzer.state import AppState  # noqa: E402
from tif_analyzer.ui import prompts, render  # noqa: E402
from tif_analyzer.ui.keys import KeyReader  # noqa: E402
from tif_analyzer.ui.menu import MenuApp  # noqa: E402


class ScriptedReader(KeyReader):
    """Replays a fixed key sequence; runs out into 'q' so no test can hang."""

    def __init__(self, keys: str | list[str]):
        super().__init__(force_line_mode=True)
        self.keys = list(keys)
        self.consumed: list[str] = []

    @property
    def hint(self) -> str:
        return "scripted"

    def read(self) -> str:
        key = self.keys.pop(0) if self.keys else "q"
        self.consumed.append(key)
        return key


def make_app(directory: Path, keys: str | list[str], **kwargs) -> MenuApp:
    state = AppState(directory=directory)
    return MenuApp(
        state,
        config_mod.Config(),
        reader=ScriptedReader(keys),
        persist=False,
        clear=False,
        **kwargs,
    )


def test_quit_immediately(series_dir, capsys):
    app = make_app(series_dir, "q")
    assert app.loop() == 0
    output = capsys.readouterr().out
    assert "TIF Series Analyzer" in output
    assert "[r] run analysis" in output


def test_header_shows_counts_after_discovery(series_dir, capsys):
    make_app(series_dir, "q").loop()
    output = capsys.readouterr().out
    assert "4 files, 4 matched" in output


def test_run_populates_the_report(series_dir, capsys):
    app = make_app(series_dir, ["r", "enter", "q"])
    app.loop()
    assert app.state.report is not None
    assert app.state.report.n_results == 4
    assert app.state.selected_series == "std"
    output = capsys.readouterr().out
    assert "4 result(s), 0 failure(s)" in output


def test_second_run_reports_cache_hits(series_dir, capsys):
    app = make_app(series_dir, ["r", "enter", "r", "enter", "q"])
    app.loop()
    assert app.cache.hits == 4
    assert "cached" in capsys.readouterr().out


def test_failures_are_reported_together_and_results_kept(series_dir, make_stack, capsys):
    make_stack(series_dir, "medida_005_20mW.tif", n_pages=2)
    app = make_app(series_dir, ["r", "enter", "q"])
    app.loop()
    output = capsys.readouterr().out
    assert "1 file(s) failed" in output
    assert "--page-policy first" in output  # actionable hint for stacks
    assert app.state.report is not None
    assert app.state.report.n_results == 4


def test_disabled_options_show_a_reason(tmp_path, capsys):
    (tmp_path / "unrelated.tif").write_bytes(b"")
    make_app(tmp_path, "q").loop()
    output = capsys.readouterr().out
    assert "[r] run analysis  (no matching files)" in output
    assert "[v] view plot  (no results" in output
    assert "[e] export CSV  (no results" in output


def test_unmatched_files_are_reported_never_dropped(tmp_path, capsys, make_tif):
    make_tif(tmp_path, "medida_001_1mW.tif")
    (tmp_path / "notes.tif").write_bytes(b"")
    app = make_app(tmp_path, ["r", "enter", "q"])
    app.loop()
    output = capsys.readouterr().out
    assert "1 of those matched the pattern" in output
    assert "notes.tif" in output


def test_duplicates_block_the_run_and_list_every_collision(tmp_path, capsys, make_tif):
    make_tif(tmp_path, "medida_001_5mW.tif")
    make_tif(tmp_path, "medida_002_5.0mW.tif")
    app = make_app(tmp_path, "q")
    app.loop()
    output = capsys.readouterr().out
    assert "duplicate parameter values" in output
    assert "medida_001_5mW.tif" in output
    assert "medida_002_5.0mW.tif" in output
    assert app.state.records is None


def test_duplicate_toggle_unblocks_the_run(tmp_path, capsys, make_tif):
    make_tif(tmp_path, "medida_001_5mW.tif")
    make_tif(tmp_path, "medida_002_5.0mW.tif")
    app = make_app(tmp_path, ["!", "r", "enter", "q"])
    app.loop()
    assert app.state.allow_duplicates is True
    assert app.state.report is not None
    assert app.state.report.n_results == 2


def test_plot_falls_back_to_png_and_reports_the_path(series_dir, capsys):
    app = make_app(series_dir, ["r", "enter", "v", "enter", "q"])
    app.loop()
    output = capsys.readouterr().out
    assert "plot saved to" in output
    saved = [p for p in series_dir.glob("*.png")]
    assert len(saved) == 1


def test_export_writes_the_csv(series_dir, monkeypatch, capsys):
    target = series_dir / "export.csv"
    monkeypatch.setattr(prompts, "prompt_export_path", lambda default: target)
    app = make_app(series_dir, ["r", "enter", "e", "q"])
    app.loop()
    assert target.exists()
    assert "exported to" in capsys.readouterr().out


def test_changing_the_pattern_invalidates_results(series_dir, monkeypatch):
    app = make_app(series_dir, ["r", "enter", "p", "q"])
    monkeypatch.setattr(
        prompts, "prompt_pattern", lambda default: r"(?P<exposure>[\d.]+)ms\.tiff?$"
    )
    app.loop()
    assert app.state.report is None
    assert app.state.pattern.endswith(r"ms\.tiff?$")


def test_cancelled_prompt_changes_nothing(series_dir, monkeypatch):
    monkeypatch.setattr(prompts, "prompt_pattern", lambda default: None)
    app = make_app(series_dir, ["p", "q"])
    app.loop()
    assert app.state.pattern == discovery.DEFAULT_PATTERN


def test_metric_menu_selection(series_dir, monkeypatch):
    app = make_app(series_dir, ["m", "q"])
    monkeypatch.setattr(prompts, "choose_from_menu", lambda *a, **k: "std")
    app.loop()
    assert app.state.metric_key == "std"


def test_unknown_key_is_ignored(series_dir):
    app = make_app(series_dir, ["z", "q"])
    assert app.loop() == 0


# -- rendering helpers ------------------------------------------------------- #


def test_progress_line_is_a_single_line():
    from tif_analyzer.analysis import ProgressEvent

    event = ProgressEvent(index=23, total=44, path=Path("medida_023_23mW.tif"), from_cache=True)
    line = render.progress_line(event, width=80)
    assert "\n" not in line
    assert "[ 23/44 ]" in line
    assert "(cached)" in line


def test_key_reader_normalisation():
    reader = KeyReader(force_line_mode=True)
    assert reader._normalise("D") == "d"
    assert reader._normalise("\x1b") == "esc"
    assert reader._normalise("\r") == "enter"
    assert reader._normalise("\x03") == "q"
    assert reader.single_key is False


# -- entry point ------------------------------------------------------------- #


def test_build_state_prefers_cli_directory(series_dir, monkeypatch):
    args = app_mod.build_parser().parse_args([str(series_dir), "--no-config"])
    state, _, notices = app_mod.build_state(args)
    assert state.directory == series_dir.resolve()
    assert notices == []


def test_build_state_notices_a_bad_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    args = app_mod.build_parser().parse_args([str(tmp_path / "nope"), "--no-config"])
    _, _, notices = app_mod.build_state(args)
    assert any("not a directory" in n for n in notices)


def test_build_state_rejects_a_stored_bad_pattern(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        config_mod, "load", lambda: config_mod.Config(last_pattern=r"no_group\.tif$")
    )
    args = app_mod.build_parser().parse_args([])
    state, _, notices = app_mod.build_state(args)
    assert state.pattern == discovery.DEFAULT_PATTERN
    assert any("stored pattern rejected" in n for n in notices)


def test_build_state_rejects_an_unknown_metric(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    args = app_mod.build_parser().parse_args(["--metric", "nope", "--no-config"])
    state, _, notices = app_mod.build_state(args)
    assert state.metric_key == "std"
    assert any("unknown metric" in n for n in notices)
