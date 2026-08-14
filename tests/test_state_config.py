from __future__ import annotations

import json
from pathlib import Path

import pytest

from tif_analyzer import config as config_mod
from tif_analyzer.analysis import RunReport
from tif_analyzer.discovery import DEFAULT_PATTERN, FileRecord
from tif_analyzer.state import INVALIDATING_FIELDS, AppState


def populated_state(tmp_path) -> AppState:
    state = AppState(directory=tmp_path)
    state.records = [FileRecord(tmp_path / "a.tif", "power", 1.0, "1")]
    state.report = RunReport()
    state.selected_series = "std"
    return state


@pytest.mark.parametrize(
    "field,value",
    [
        ("directory", Path("/elsewhere")),
        ("glob", "*.tiff"),
        ("pattern", r"(?P<t>[\d.]+)ms\.tif$"),
        ("metric_key", "other"),
        ("preprocess_steps", ["step"]),
        ("allow_duplicates", True),
        ("page_policy", "first"),
    ],
)
def test_input_changes_clear_derived_data(tmp_path, field, value):
    state = populated_state(tmp_path)
    assert state.update(**{field: value}) is True
    assert state.records is None
    assert state.report is None
    assert state.discovery is None


def test_selected_series_change_does_not_invalidate(tmp_path):
    state = populated_state(tmp_path)
    assert state.update(selected_series="mean") is False
    assert state.records is not None
    assert state.report is not None


def test_setting_the_same_value_is_a_no_op(tmp_path):
    state = populated_state(tmp_path)
    assert state.update(glob=state.glob, pattern=state.pattern) is False
    assert state.records is not None


def test_changing_metric_drops_the_selected_series(tmp_path):
    state = populated_state(tmp_path)
    state.update(metric_key="other")
    assert state.selected_series is None


def test_unknown_field_is_rejected(tmp_path):
    with pytest.raises(AttributeError, match="unknown AppState field"):
        AppState(directory=tmp_path).update(directoy=tmp_path)


def test_invalidating_fields_match_the_spec():
    assert {
        "directory",
        "glob",
        "pattern",
        "metric_key",
        "preprocess_steps",
        "allow_duplicates",
    } <= INVALIDATING_FIELDS
    assert "selected_series" not in INVALIDATING_FIELDS


# -- config ------------------------------------------------------------------ #


@pytest.fixture
def config_home(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "cfg"))
    monkeypatch.setattr("sys.platform", "linux")
    return tmp_path / "cfg" / "tif-analyzer" / "config.json"


def test_round_trip(config_home):
    saved = config_mod.Config(
        last_directory="/data", last_pattern=DEFAULT_PATTERN, allow_duplicates=True
    )
    assert config_mod.save(saved) == config_home
    loaded = config_mod.load()
    assert loaded.last_directory == "/data"
    assert loaded.allow_duplicates is True


def test_missing_config_is_a_non_event(config_home):
    assert config_mod.load() == config_mod.Config()


def test_corrupt_config_is_a_non_event(config_home):
    config_home.parent.mkdir(parents=True)
    config_home.write_text("{not json")
    assert config_mod.load() == config_mod.Config()


def test_wrongly_typed_values_are_ignored(config_home):
    config_home.parent.mkdir(parents=True)
    config_home.write_text(json.dumps({"allow_duplicates": "yes", "last_pattern": 12}))
    loaded = config_mod.load()
    assert loaded.allow_duplicates is False
    assert loaded.last_pattern == DEFAULT_PATTERN


def test_startup_falls_back_to_cwd_with_a_notice(config_home, tmp_path):
    config = config_mod.Config(last_directory=str(tmp_path / "gone"))
    startup = config_mod.resolve_startup(config, cwd=tmp_path)
    assert startup.directory == tmp_path.resolve()
    assert startup.notice is not None
    assert "no longer exists" in startup.notice


def test_startup_uses_the_remembered_directory(config_home, tmp_path):
    startup = config_mod.resolve_startup(
        config_mod.Config(last_directory=str(tmp_path)), cwd=Path("/")
    )
    assert startup.directory == tmp_path.resolve()
    assert startup.notice is None


def test_config_path_uses_appdata_on_windows(tmp_path, monkeypatch):
    monkeypatch.setattr("sys.platform", "win32")
    monkeypatch.setenv("APPDATA", str(tmp_path / "roaming"))
    assert config_mod.config_path() == tmp_path / "roaming" / "tif-analyzer" / "config.json"
