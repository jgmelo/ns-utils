"""Selection state machine, and the [a] key wired into the menu."""

from __future__ import annotations

import numpy as np
import pytest
import tifffile

import matplotlib

matplotlib.use("Agg")

from tif_analyzer import config as config_mod, roi_sidecar  # noqa: E402
from tif_analyzer.regions import Ellipse, Rect, RegionSet  # noqa: E402
from tif_analyzer.state import AppState  # noqa: E402
from tif_analyzer.ui import menu as menu_mod, prompts, selection  # noqa: E402
from tif_analyzer.ui.menu import MenuApp  # noqa: E402
from tif_analyzer.ui.selection import SelectionSession  # noqa: E402

from .test_ui import ScriptedReader  # noqa: E402

SHAPE = (20, 30)


@pytest.fixture
def snr_dir(tmp_path):
    directory = tmp_path / "data"
    directory.mkdir()
    for index, power in enumerate([1, 2, 4], start=1):
        image = np.zeros(SHAPE, dtype=np.uint16)
        image[2:6, 2:6] = 1000 * power
        image[2:8, 20:26] = (np.arange(36).reshape(6, 6) * power).astype(np.uint16)
        tifffile.imwrite(directory / f"medida_{index:03d}_{power}mW.tif", image)
    return directory


def region_set(**kwargs) -> RegionSet:
    defaults = dict(
        signal=Rect(x=2, y=2, width=4, height=4),
        noise=Rect(x=20, y=2, width=6, height=6),
        image_shape=SHAPE,
        source_file="medida_001_1mW.tif",
    )
    defaults.update(kwargs)
    return RegionSet(**defaults)  # type: ignore[arg-type]


def make_app(directory, keys, metric_key="snr", **kwargs) -> MenuApp:
    state = AppState(directory=directory, metric_key=metric_key)
    return MenuApp(
        state,
        config_mod.Config(),
        reader=ScriptedReader(keys),
        persist=False,
        clear=False,
        **kwargs,
    )


# -- session state machine --------------------------------------------------- #


def test_session_walks_signal_then_noise():
    session = SelectionSession(shape=SHAPE, source_file="a.tif")
    assert session.role == "signal"
    session.draw((2.0, 5.0, 2.0, 5.0))
    assert session.confirm() is True
    assert session.role == "noise"
    session.draw((20.0, 25.0, 2.0, 7.0))
    assert session.confirm() is True
    assert session.done

    result = session.result()
    assert result == RegionSet(
        signal=Rect(x=2, y=2, width=4, height=4),
        noise=Rect(x=20, y=2, width=6, height=6),
        image_shape=SHAPE,
        source_file="a.tif",
    )


def test_confirm_without_a_drag_explains_itself():
    session = SelectionSession(shape=SHAPE)
    assert session.confirm() is False
    assert "drag a box" in (session.error or "")
    assert session.role == "signal"


def test_a_one_pixel_area_is_refused():
    session = SelectionSession(shape=SHAPE)
    session.draw((5.0, 5.0, 5.0, 5.0))
    assert session.confirm() is False
    assert "too small" in (session.error or "")


def test_undo_drops_the_pending_box_then_the_confirmed_one():
    session = SelectionSession(shape=SHAPE)
    session.draw((2.0, 5.0, 2.0, 5.0))
    session.confirm()
    session.draw((20.0, 25.0, 2.0, 7.0))
    assert session.undo() is True  # drops the pending noise box
    assert session.role == "noise"
    assert session.undo() is True  # steps back to signal
    assert session.role == "signal"
    assert session.confirmed == {}
    assert session.undo() is False


def test_shape_toggle_changes_what_the_next_drag_produces():
    session = SelectionSession(shape=SHAPE)
    session.set_shape("ellipse")
    session.draw((4.0, 14.0, 5.0, 11.0))
    assert session.pending == Ellipse(cx=9.0, cy=8.0, rx=5.0, ry=3.0)
    with pytest.raises(ValueError, match="unknown shape"):
        session.set_shape("polygon")


def test_cancel_yields_nothing_even_when_complete():
    session = SelectionSession(shape=SHAPE)
    session.draw((2.0, 5.0, 2.0, 5.0))
    session.confirm()
    session.draw((20.0, 25.0, 2.0, 7.0))
    session.confirm()
    session.cancel()
    assert session.result() is None


def test_incomplete_session_yields_nothing():
    session = SelectionSession(shape=SHAPE)
    session.draw((2.0, 5.0, 2.0, 5.0))
    session.confirm()
    assert session.result() is None


def test_display_stretch_never_touches_the_data():
    image = np.full(SHAPE, 1000, dtype=np.uint16)
    image[0, 0] = 60000
    low, high = selection.display_limits(image)
    assert low < high
    assert image[0, 0] == 60000  # unchanged


def test_display_limits_survive_a_flat_image():
    low, high = selection.display_limits(np.zeros(SHAPE, dtype=np.uint16))
    assert high > low


def test_selection_is_blocked_without_a_gui():
    possible, reason = selection.can_select()
    assert possible is False
    assert "needs a GUI" in reason


# -- menu integration -------------------------------------------------------- #


def test_areas_option_is_disabled_headless(snr_dir, capsys):
    make_app(snr_dir, "q").loop()
    output = capsys.readouterr().out
    assert "[a] select areas  (" in output
    assert "area selection needs a GUI" in output


def test_run_is_blocked_until_areas_exist(snr_dir, capsys):
    make_app(snr_dir, "q").loop()
    output = capsys.readouterr().out
    assert "[r] run analysis  (metric snr needs the areas — press [a])" in output
    assert "areas:    none selected — press [a]" in output


def test_pressing_r_without_areas_does_nothing(snr_dir):
    app = make_app(snr_dir, ["r", "q"])
    app.loop()
    assert app.state.report is None


def test_saved_areas_are_loaded_on_startup_and_unblock_the_run(snr_dir, capsys):
    roi_sidecar.save(snr_dir, region_set())
    app = make_app(snr_dir, ["r", "enter", "q"])
    app.loop()
    assert app.state.regions == region_set()
    assert app.state.report is not None
    assert app.state.report.n_results == 3
    output = capsys.readouterr().out
    assert "areas loaded from .tif-analyzer-rois.json" in output


def test_areas_from_a_differently_sized_image_are_not_applied(snr_dir, capsys):
    roi_sidecar.save(snr_dir, region_set(image_shape=(40, 60)))
    app = make_app(snr_dir, "q")
    app.loop()
    assert app.state.regions is None
    assert "not applied, select them again" in capsys.readouterr().out


def test_selecting_areas_stores_and_persists_them(snr_dir, monkeypatch, capsys):
    chosen = region_set(source_file="medida_002_2mW.tif")
    monkeypatch.setattr(selection, "can_select", lambda: (True, ""))
    monkeypatch.setattr(menu_mod.selection, "select_regions", lambda image, name: chosen)
    monkeypatch.setattr(prompts, "prompt_reference_file", lambda records, **kw: records[1])

    app = make_app(snr_dir, ["a", "q"])
    app.loop()
    assert app.state.regions == chosen
    assert roi_sidecar.load(snr_dir).regions == chosen
    assert "areas set:" in capsys.readouterr().out


def test_cancelled_selection_leaves_state_alone(snr_dir, monkeypatch):
    roi_sidecar.save(snr_dir, region_set())
    monkeypatch.setattr(selection, "can_select", lambda: (True, ""))
    monkeypatch.setattr(menu_mod.selection, "select_regions", lambda image, name: None)
    monkeypatch.setattr(prompts, "prompt_reference_file", lambda records, **kw: records[0])

    app = make_app(snr_dir, ["a", "q"])
    app.loop()
    assert app.state.regions == region_set()
    assert roi_sidecar.load(snr_dir).regions == region_set()


def test_redrawing_areas_invalidates_the_report(snr_dir, monkeypatch):
    roi_sidecar.save(snr_dir, region_set())
    moved = region_set(noise=Rect(x=20, y=2, width=5, height=5))
    monkeypatch.setattr(selection, "can_select", lambda: (True, ""))
    monkeypatch.setattr(menu_mod.selection, "select_regions", lambda image, name: moved)
    monkeypatch.setattr(prompts, "prompt_reference_file", lambda records, **kw: records[0])

    app = make_app(snr_dir, ["r", "enter", "a", "q"])
    app.loop()
    assert app.state.regions == moved
    assert app.state.report is None  # the old numbers do not survive a redraw


def test_a_multipage_reference_is_reported_not_crashed(snr_dir, monkeypatch, capsys):
    stack = np.zeros((3, *SHAPE), dtype=np.uint16)
    tifffile.imwrite(snr_dir / "medida_007_7mW.tif", stack, photometric="minisblack")
    monkeypatch.setattr(selection, "can_select", lambda: (True, ""))
    monkeypatch.setattr(prompts, "prompt_reference_file", lambda records, **kw: records[-1])

    app = make_app(snr_dir, ["a", "q"])
    app.loop()
    assert app.state.regions is None
    assert "3 pages" in capsys.readouterr().out


def test_plot_and_csv_record_the_areas(snr_dir, monkeypatch, tmp_path, capsys):
    roi_sidecar.save(snr_dir, region_set())
    target = tmp_path / "out.csv"
    monkeypatch.setattr(prompts, "prompt_export_path", lambda default: target)
    app = make_app(snr_dir, ["r", "enter", "v", "enter", "e", "q"])
    app.loop()

    text = target.read_text(encoding="utf-8")
    assert "# regions: signal rect(x=2, y=2, w=4, h=4)" in text
    assert "# regions_source: medida_001_1mW.tif" in text
    assert ",snr," in text and ",noise_std," in text
    assert len(list(snr_dir.glob("*.png"))) == 1


def test_series_menu_appears_for_snr(snr_dir, capsys):
    roi_sidecar.save(snr_dir, region_set())
    make_app(snr_dir, "q").loop()
    assert "[s] select series" in capsys.readouterr().out


def test_changing_metric_to_std_stops_requiring_areas(snr_dir, monkeypatch, capsys):
    monkeypatch.setattr(prompts, "choose_from_menu", lambda *a, **k: "std")
    app = make_app(snr_dir, ["m", "r", "enter", "q"])
    app.loop()
    assert app.state.metric_key == "std"
    assert app.state.report is not None
    assert app.state.report.n_results == 3
