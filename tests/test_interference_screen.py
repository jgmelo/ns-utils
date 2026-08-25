"""The interference screen and its acquisition sidecar."""

from __future__ import annotations

import json

import pytest
import tifffile

from tif_analyzer import acquisition_sidecar
from tif_analyzer.discovery import FileRecord
from tif_analyzer.interference.params import Acquisition
from tif_analyzer.interference.synthetic import FrameSpec, make_pair
from tif_analyzer.ui import interference_screen as screen_mod, prompts
from tif_analyzer.ui.interference_screen import InterferenceScreen, InterferenceSettings

from .test_ui import ScriptedReader


@pytest.fixture
def frames_dir(tmp_path):
    signal, reference, _ = make_pair(
        FrameSpec(shape=(64, 512), period_px=16.0, amplitude=40.0, noise_sigma=5.0)
    )
    directory = tmp_path / "data"
    directory.mkdir()
    tifffile.imwrite(directory / "medida_001_1mW.tif", signal)
    tifffile.imwrite(directory / "medida_002_2mW.tif", reference)
    return directory


@pytest.fixture
def records(frames_dir):
    return [
        FileRecord(frames_dir / "medida_001_1mW.tif", "power", 1.0, "1"),
        FileRecord(frames_dir / "medida_002_2mW.tif", "power", 2.0, "2"),
    ]


def make_screen(records, frames_dir, keys, **kwargs):
    return InterferenceScreen(
        records, frames_dir, ScriptedReader(keys).read, clear=False, **kwargs
    )


# -- settings ---------------------------------------------------------------- #


def test_timing_needs_a_line_rate():
    settings = InterferenceSettings(frame_width=500)
    assert settings.timing() is None
    settings.line_rate_hz = 1000.0
    dwell, sampling, nyquist = settings.timing()
    assert dwell == pytest.approx(2e-6)
    assert sampling == pytest.approx(500_000.0)
    assert nyquist == pytest.approx(250_000.0)


def test_resolution_overrides_the_frame_width():
    settings = InterferenceSettings(frame_width=500, resolution=1000, line_rate_hz=1000.0)
    assert settings.pixels_per_line == 1000
    assert settings.timing()[0] == pytest.approx(1e-6)


def test_flyback_shortens_the_dwell_only_when_it_is_in_the_frame():
    base = InterferenceSettings(frame_width=500, line_rate_hz=1000.0, flyback_fraction=0.2)
    assert base.timing()[0] == pytest.approx(1.6e-6)
    base.frame_includes_flyback = True
    assert base.timing()[0] == pytest.approx(2e-6)


# -- screen ------------------------------------------------------------------ #


def test_screen_starts_on_the_first_record_and_reads_its_width(records, frames_dir):
    screen = make_screen(records, frames_dir, "b")
    assert screen.settings.signal == records[0]
    assert screen.settings.frame_width == 512
    assert screen.settings.reference is None


def test_header_shows_the_implied_nyquist(records, frames_dir):
    screen = make_screen(records, frames_dir, "b")
    assert "set the line rate" in screen.header()
    screen.settings.line_rate_hz = 1000.0
    header = screen.header()
    assert "dwell 1953.12 ns/px" in header
    assert "Nyquist 256 kHz" in header


def test_options_are_blocked_with_a_reason(records, frames_dir, capsys):
    screen = make_screen(records, frames_dir, "b")
    screen.run()
    output = capsys.readouterr().out
    assert "[o] save last report  (run the report first)" in output
    assert "[c] clear reference  (no reference set)" in output


def test_back_leaves_the_screen(records, frames_dir):
    screen = make_screen(records, frames_dir, "b")
    screen.run()  # returns rather than hanging
    assert screen.last_report is None


def test_line_rate_is_entered_and_persisted(records, frames_dir, monkeypatch):
    monkeypatch.setattr(prompts, "prompt_number", lambda *a, **k: 1000.0)
    screen = make_screen(records, frames_dir, ["l", "b"])
    screen.run()
    assert screen.settings.line_rate_hz == 1000.0
    assert acquisition_sidecar.load(frames_dir).acquisition.line_rate_hz == 1000.0


def test_clearing_a_field_unsets_it(records, frames_dir, monkeypatch):
    monkeypatch.setattr(prompts, "prompt_number", lambda *a, **k: prompts.CLEARED)
    screen = make_screen(records, frames_dir, ["l", "b"])
    screen.settings.line_rate_hz = 500.0
    screen.run()
    assert screen.settings.line_rate_hz is None


def test_cancelling_a_field_leaves_it_alone(records, frames_dir, monkeypatch):
    monkeypatch.setattr(prompts, "prompt_number", lambda *a, **k: None)
    screen = make_screen(records, frames_dir, ["l", "b"])
    screen.settings.line_rate_hz = 500.0
    screen.run()
    assert screen.settings.line_rate_hz == 500.0


def test_flyback_toggle_persists(records, frames_dir):
    screen = make_screen(records, frames_dir, ["w", "b"])
    screen.run()
    assert screen.settings.frame_includes_flyback is True
    assert acquisition_sidecar.load(frames_dir).acquisition.frame_includes_flyback is True


def test_reference_must_differ_from_the_signal(records, frames_dir, monkeypatch, capsys):
    monkeypatch.setattr(prompts, "prompt_reference_file", lambda *a, **k: records[0])
    screen = make_screen(records, frames_dir, ["r", "b"])
    screen.run()
    assert screen.settings.reference is None
    assert "must be a different file" in capsys.readouterr().out


def test_reference_is_set_and_cleared(records, frames_dir, monkeypatch):
    monkeypatch.setattr(prompts, "prompt_reference_file", lambda *a, **k: records[1])
    screen = make_screen(records, frames_dir, ["r", "b"])
    screen.run()
    assert screen.settings.reference == records[1]
    screen.read_key = ScriptedReader(["c", "b"]).read
    screen.run()
    assert screen.settings.reference is None


def test_running_produces_a_report(records, frames_dir, monkeypatch, capsys):
    monkeypatch.setattr(prompts, "prompt_reference_file", lambda *a, **k: records[1])
    screen = make_screen(records, frames_dir, ["r", "g", "enter", "b"])
    screen.settings.line_rate_hz = 1000.0
    screen.run()
    assert screen.last_report is not None
    output = capsys.readouterr().out
    assert "SUBTRACTION VIABLE" in output
    assert "NYQUIST LIMIT" in output


def test_running_without_a_line_rate_says_so(records, frames_dir, capsys):
    screen = make_screen(records, frames_dir, ["g", "enter", "b"])
    screen.run()
    assert "no line rate supplied" in capsys.readouterr().out


def test_report_can_be_saved(records, frames_dir, tmp_path, monkeypatch):
    target = tmp_path / "report.txt"
    monkeypatch.setattr(prompts, "prompt_export_path", lambda *a, **k: target)
    screen = make_screen(records, frames_dir, ["g", "enter", "o", "b"])
    screen.run()
    assert "FIXED-PATTERN INTERFERENCE" in target.read_text(encoding="utf-8")


def test_a_bad_frame_is_reported_not_crashed(records, frames_dir, capsys):
    (frames_dir / "medida_001_1mW.tif").write_bytes(b"not a tiff")
    screen = make_screen(records, frames_dir, ["g", "b"])
    screen.run()
    assert "✗" in capsys.readouterr().out
    assert screen.last_report is None


def test_persistence_can_be_turned_off(records, frames_dir):
    screen = make_screen(records, frames_dir, ["w", "b"], persist=False)
    screen.run()
    assert acquisition_sidecar.sidecar_path(frames_dir).exists() is False


# -- sidecar ----------------------------------------------------------------- #


def test_sidecar_round_trip(tmp_path):
    acquisition = Acquisition(
        line_rate_hz=1200.0, resolution=1024, flyback_fraction=0.15, frame_includes_flyback=True
    )
    path, notice = acquisition_sidecar.save(tmp_path, acquisition)
    assert notice is None
    assert path.name == ".tif-analyzer-acquisition.json"
    assert acquisition_sidecar.load(tmp_path).acquisition == acquisition


def test_missing_sidecar_is_a_non_event(tmp_path):
    loaded = acquisition_sidecar.load(tmp_path)
    assert loaded.acquisition is None and loaded.notice is None


def test_corrupt_sidecar_is_reported(tmp_path):
    acquisition_sidecar.sidecar_path(tmp_path).write_text("{oops")
    loaded = acquisition_sidecar.load(tmp_path)
    assert loaded.acquisition is None
    assert "unreadable" in loaded.notice


def test_impossible_sidecar_values_are_refused(tmp_path):
    acquisition_sidecar.sidecar_path(tmp_path).write_text(
        json.dumps({"line_rate_hz": -5.0})
    )
    loaded = acquisition_sidecar.load(tmp_path)
    assert loaded.acquisition is None
    assert "malformed" in loaded.notice


def test_saved_acquisition_is_picked_up_by_a_new_screen(records, frames_dir, capsys):
    acquisition_sidecar.save(frames_dir, Acquisition(line_rate_hz=750.0, flyback_fraction=0.1))
    screen = make_screen(records, frames_dir, "b")
    screen.run()
    assert screen.settings.line_rate_hz == 750.0
    assert screen.settings.flyback_fraction == pytest.approx(0.1)
    assert "acquisition loaded from" in capsys.readouterr().out


def test_screen_module_exposes_a_readable_hz_formatter():
    assert screen_mod._fmt_hz(500.0) == "500 Hz"
    assert screen_mod._fmt_hz(1500.0) == "1.5 kHz"
    assert screen_mod._fmt_hz(2.5e6) == "2.5 MHz"


# -- mask figure ------------------------------------------------------------- #


def test_mask_figure_has_three_aligned_panels(frames_dir):
    from tif_analyzer.interference.figures import build_mask_figure
    from tif_analyzer.interference.frames import load_frame
    from tif_analyzer.interference.mask import build_mask

    frame = load_frame(frames_dir / "medida_001_1mW.tif")
    result = build_mask(frame.data)
    figure = build_mask_figure(frame, result)
    image_axes, column_axes, histogram_axes = figure.axes[:3]

    assert len(figure.axes) == 3
    assert image_axes.get_title(loc="left").startswith("medida_001_1mW.tif")
    # The column panel shares the image's x axis, so a spike lines up with the
    # bead that caused it — that alignment is the point of the panel.
    assert column_axes.get_shared_x_axes().joined(column_axes, image_axes)
    assert histogram_axes.get_xlabel().startswith("pixel value")
    assert histogram_axes.get_yscale() == "log"


def test_mask_figure_marks_the_worst_column(frames_dir):
    import numpy as np
    import tifffile
    from tif_analyzer.interference.figures import build_mask_figure
    from tif_analyzer.interference.frames import load_frame
    from tif_analyzer.interference.mask import build_mask

    from tif_analyzer.interference.synthetic import make_frame

    path = frames_dir / "beady.tif"
    array, _ = make_frame(FrameSpec(shape=(64, 128), amplitude=20.0))
    array = np.asarray(array, dtype=np.float64)
    array[:, 40] = 60000.0  # one column entirely bead
    tifffile.imwrite(path, array.astype("uint16"))

    frame = load_frame(path)
    result = build_mask(frame.data)
    assert result.worst_column_index == 40
    figure = build_mask_figure(frame, result)
    labels = [text.get_text() for text in figure.axes[1].get_legend().get_texts()]
    assert any("worst: column 40" in label for label in labels)


def test_view_mask_saves_a_png_when_headless(records, frames_dir, capsys):
    screen = make_screen(records, frames_dir, ["v", "b"])
    screen.run()
    saved = list(frames_dir.glob("bead-mask_*.png"))
    assert len(saved) == 1
    assert saved[0].stat().st_size > 0
    assert "figure saved to" in capsys.readouterr().out


def test_view_mask_reports_a_bad_frame(records, frames_dir, capsys):
    (frames_dir / "medida_001_1mW.tif").write_bytes(b"not a tiff")
    screen = make_screen(records, frames_dir, ["v", "b"])
    screen.run()
    assert "✗" in capsys.readouterr().out
    assert not list(frames_dir.glob("bead-mask_*.png"))


# -- bead cut knob ----------------------------------------------------------- #


def test_default_bead_cut_matches_the_params_default():
    from tif_analyzer.interference.params import InterferenceParams

    assert InterferenceSettings().mad_k == InterferenceParams().mad_k
    assert InterferenceSettings().params().mad_k == InterferenceParams().mad_k


def test_bead_cut_is_shown_and_flagged_as_default(records, frames_dir):
    screen = make_screen(records, frames_dir, "b")
    assert "bead cut:    median + 3σ   (default)" in screen.header()
    screen.settings.mad_k = 5.0
    header = screen.header()
    assert "median + 5σ" in header
    assert "(default)" not in header


def test_bead_cut_can_be_changed(records, frames_dir, monkeypatch):
    monkeypatch.setattr(prompts, "prompt_number", lambda *a, **k: 5.0)
    screen = make_screen(records, frames_dir, ["k", "b"])
    screen.run()
    assert screen.settings.mad_k == 5.0
    assert screen.settings.params().mad_k == 5.0


def test_cancelling_the_bead_cut_leaves_it_alone(records, frames_dir, monkeypatch):
    monkeypatch.setattr(prompts, "prompt_number", lambda *a, **k: None)
    screen = make_screen(records, frames_dir, ["k", "b"])
    screen.run()
    assert screen.settings.mad_k == 3.0


def test_a_higher_cut_rejects_fewer_pixels(frames_dir):
    """The knob has to actually reach the mask, not just the header."""
    from tif_analyzer.interference.frames import load_frame
    from tif_analyzer.interference.mask import build_mask
    from tif_analyzer.interference.params import InterferenceParams

    data = load_frame(frames_dir / "medida_001_1mW.tif").data
    loose = build_mask(data, InterferenceParams(mad_k=8.0))
    tight = build_mask(data, InterferenceParams(mad_k=2.0))
    assert loose.threshold > tight.threshold
    assert loose.rejected_fraction <= tight.rejected_fraction


def test_the_report_names_the_cut_that_was_used(records, frames_dir, monkeypatch, capsys):
    screen = make_screen(records, frames_dir, ["g", "enter", "b"])
    screen.settings.mad_k = 5.0
    screen.run()
    assert "median + 5 sigma" in (screen.last_report or "")


# -- profile & spectrum figure ----------------------------------------------- #


def test_profile_figure_has_profile_spectrum_and_zoom(frames_dir):
    from tif_analyzer.interference import analyse_channel
    from tif_analyzer.interference.figures import build_profile_figure
    from tif_analyzer.interference.frames import load_frame

    analysis = analyse_channel(load_frame(frames_dir / "medida_001_1mW.tif"))
    figure = build_profile_figure(analysis)
    profile_axes, spectrum_axes, zoom_axes = figure.axes[:3]

    assert profile_axes.get_title(loc="left").startswith("medida_001_1mW.tif")
    assert "column" in profile_axes.get_xlabel()
    assert spectrum_axes.get_xlabel() == "cycles per pixel"
    assert spectrum_axes.get_yscale() == "log"
    assert "DETECTED" in spectrum_axes.get_title(loc="left")
    assert zoom_axes.get_ylabel() == "counts"


def test_the_drawn_component_matches_the_planted_amplitude(tmp_path):
    """The zoom fits a sinusoid by projection rather than inverting the cluster
    bins — leakage would otherwise draw a component smaller than the profile."""
    import tifffile
    from tif_analyzer.interference import analyse_channel
    from tif_analyzer.interference.figures import build_profile_figure
    from tif_analyzer.interference.frames import load_frame
    from tif_analyzer.interference.synthetic import FrameSpec, make_frame

    # A period that is deliberately not a whole number of bins.
    array, truth = make_frame(
        FrameSpec(shape=(64, 512), period_px=13.7, amplitude=45.0, noise_sigma=3.0)
    )
    path = tmp_path / "frame.tif"
    tifffile.imwrite(path, array)
    figure = build_profile_figure(analyse_channel(load_frame(path)))

    labels = [text.get_text() for text in figure.axes[2].get_legend().get_texts()]
    drawn = [label for label in labels if "single component" in label]
    assert drawn, labels
    amplitude = float(drawn[0].split("amplitude ")[1].split(" ")[0])
    assert amplitude == pytest.approx(truth.amplitude, rel=0.05)


def test_profile_figure_survives_a_frame_with_no_component(tmp_path):
    import tifffile
    from tif_analyzer.interference import analyse_channel
    from tif_analyzer.interference.figures import build_profile_figure
    from tif_analyzer.interference.frames import load_frame
    from tif_analyzer.interference.synthetic import FrameSpec, make_frame

    array, _ = make_frame(FrameSpec(shape=(64, 512), amplitude=0.0, noise_sigma=20.0))
    path = tmp_path / "noise.tif"
    tifffile.imwrite(path, array)
    analysis = analyse_channel(load_frame(path))
    assert analysis.spectrum.detected is False
    figure = build_profile_figure(analysis)
    assert "NOT DETECTED" in figure.axes[1].get_title(loc="left")


def test_view_profile_saves_a_png_when_headless(records, frames_dir, capsys):
    screen = make_screen(records, frames_dir, ["p", "b"])
    screen.run()
    saved = list(frames_dir.glob("profile_*.png"))
    assert len(saved) == 1
    assert saved[0].stat().st_size > 0
    assert "figure saved to" in capsys.readouterr().out


def test_view_profile_reports_a_bad_frame(records, frames_dir, capsys):
    (frames_dir / "medida_001_1mW.tif").write_bytes(b"not a tiff")
    screen = make_screen(records, frames_dir, ["p", "b"])
    screen.run()
    assert "✗" in capsys.readouterr().out
    assert not list(frames_dir.glob("profile_*.png"))


def test_both_figures_use_the_current_bead_cut(records, frames_dir):
    """The knob has to reach the figures, not just the report."""
    from tif_analyzer.interference.mask import build_mask

    screen = make_screen(records, frames_dir, "b")
    screen.settings.mad_k = 9.0
    analysis = screen._analyse_signal()
    frame = screen._load_signal()
    assert analysis.mask.threshold == build_mask(frame.data, screen.settings.params()).threshold
    assert analysis.mask.threshold > build_mask(frame.data).threshold
