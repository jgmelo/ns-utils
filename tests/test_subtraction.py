"""Fixed-pattern subtraction: the maths, the file it writes, and the screen."""

from __future__ import annotations

import json

import numpy as np
import pytest
import tifffile

from tif_analyzer.interference import analyse_channel, crosschannel
from tif_analyzer.interference.frames import frame_from_array
from tif_analyzer.interference.params import InterferenceParams
from tif_analyzer.interference.subtract import (
    SubtractionError,
    SubtractionPlan,
    alignment_shift,
    bandlimit_lines,
    fourier_shift,
    least_squares_scale,
    measure_stripe,
    robust_sigma,
    subtract,
)
from tif_analyzer.interference.synthetic import FrameSpec, make_frame, make_pair
from tif_analyzer.interference.writing import (
    default_output_path,
    read_provenance,
    write_corrected,
)

PARAMS = InterferenceParams()
#: Small, dim, and spaced by something that is not a whole number of periods —
#: beads placed every 5 periods add coherently at the stripe frequency and
#: quietly corrupt any amplitude measured from the frame.
BEADS = tuple((15 + i * 23, 47 + i * 91, 3.5, 4000.0) for i in range(5))


def pair(**kwargs):
    defaults = dict(shape=(128, 512), period_px=16.0, amplitude=40.0, noise_sigma=3.0)
    spec_kwargs = {k: v for k, v in kwargs.items() if k in FrameSpec.__dataclass_fields__}
    make_kwargs = {k: v for k, v in kwargs.items() if k not in spec_kwargs}
    defaults.update(spec_kwargs)
    return make_pair(FrameSpec(**defaults), **make_kwargs)


def analysed(array):
    return analyse_channel(frame_from_array(array))


def measured_shift(signal, reference):
    """The alignment shift the screen would compute, and the shared band."""
    a, b = analysed(signal), analysed(reference)
    cross = crosschannel.compare(
        a.profile.analysis_profile, b.profile.analysis_profile, a.spectrum, b.spectrum
    )
    band = (
        min(a.spectrum.cluster_low_bin, b.spectrum.cluster_low_bin),
        max(a.spectrum.cluster_high_bin, b.spectrum.cluster_high_bin),
    )
    return alignment_shift(cross.phase_offset_px), band, a


# -- building blocks --------------------------------------------------------- #


def test_integer_fourier_shift_matches_a_roll():
    array, _ = make_frame(FrameSpec(shape=(8, 64), period_px=16.0))
    data = np.asarray(array, dtype=np.float64)
    np.testing.assert_allclose(fourier_shift(data, 3), np.roll(data, 3, axis=-1), atol=1e-8)


def test_shifting_back_and_forth_returns_band_limited_data_exactly():
    """The round trip is exact for data with nothing at Nyquist — which is the
    claim the docstring makes. White noise reaches Nyquist and does not."""
    columns = np.arange(64)
    wave = (np.cos(2 * np.pi * columns / 16.0) + 0.4 * np.cos(2 * np.pi * columns / 7.0))[None, :]
    np.testing.assert_allclose(fourier_shift(fourier_shift(wave, 2.4), -2.4), wave, atol=1e-9)


def test_sub_pixel_shift_matches_the_analytic_sinusoid():
    columns = np.arange(256)
    wave = np.cos(2 * np.pi * columns / 16.0)[None, :]
    shifted = fourier_shift(wave, 1.7)
    expected = np.cos(2 * np.pi * (columns - 1.7) / 16.0)[None, :]
    np.testing.assert_allclose(shifted, expected, atol=1e-8)


def test_bandlimit_keeps_only_the_named_bins():
    columns = np.arange(256)
    slow = np.cos(2 * np.pi * 4 * columns / 256)[None, :]
    fast = np.cos(2 * np.pi * 40 * columns / 256)[None, :]
    np.testing.assert_allclose(bandlimit_lines(slow + fast, 3, 5), slow, atol=1e-8)


def test_least_squares_scale_is_exact_without_noise():
    reference = np.cos(2 * np.pi * np.arange(256) / 16.0)[None, :]
    assert least_squares_scale(2.5 * reference, reference) == pytest.approx(2.5)


def test_least_squares_scale_ignores_masked_pixels():
    reference = np.tile(np.cos(2 * np.pi * np.arange(256) / 16.0), (4, 1))
    signal = 2.0 * reference.copy()
    signal[0, :] += 5000.0  # a bead-like row that would drag the fit
    mask = np.ones_like(signal, dtype=bool)
    mask[0, :] = False
    assert least_squares_scale(signal, reference, mask) == pytest.approx(2.0)


def test_flat_reference_is_refused():
    with pytest.raises(SubtractionError, match="no variance"):
        least_squares_scale(np.ones((4, 8)), np.zeros((4, 8)))


def test_measure_stripe_recovers_the_planted_amplitude():
    array, _ = make_frame(FrameSpec(shape=(64, 512), period_px=16.0, amplitude=40.0, noise_sigma=2.0))
    assert measure_stripe(np.asarray(array, float), 1 / 16.0, PARAMS) == pytest.approx(40.0, rel=0.05)


def test_the_bead_mask_rescues_the_stripe_measurement():
    """Bright wide beads carry energy at the stripe frequency and drag the
    measurement well off the planted amplitude; masking them recovers most of it."""
    bright = tuple((20 + i * 30, 60 + i * 80, 6.0, 9000.0) for i in range(5))
    array, _ = make_frame(
        FrameSpec(shape=(128, 512), period_px=16.0, amplitude=40.0, noise_sigma=3.0, beads=bright)
    )
    data = np.asarray(array, dtype=np.float64)
    analysis = analysed(array)
    without = measure_stripe(data, 1 / 16.0, PARAMS)
    with_mask = measure_stripe(data, 1 / 16.0, PARAMS, analysis.mask.mask)
    assert without < 30.0  # badly off the planted 40
    assert with_mask > without + 5.0


def test_robust_sigma_ignores_the_stripe():
    """A frame that is all pattern and no noise must read as noiseless."""
    columns = np.arange(512)
    pure_stripe = np.tile(40 * np.cos(2 * np.pi * columns / 16.0), (64, 1))
    assert robust_sigma(pure_stripe) == pytest.approx(0.0, abs=1e-6)


# -- the alignment sign, which is where this goes wrong ---------------------- #


def test_alignment_shift_negates_the_measured_offset():
    assert alignment_shift(1.75) == pytest.approx(-1.75)
    assert alignment_shift(-1.75) == pytest.approx(1.75)


def test_the_right_sign_removes_the_stripe_and_the_wrong_one_does_not():
    signal, reference, _ = pair(phase_offset_deg=40.0, amplitude_ratio=0.6)
    shift, band, analysis = measured_shift(signal, reference)
    kwargs = dict(
        reference=reference, mask_result=analysis.mask, band=band,
        frequency=analysis.spectrum.centroid_frequency,
    )
    right = subtract(signal, SubtractionPlan(shift_px=shift), **kwargs)
    wrong = subtract(signal, SubtractionPlan(shift_px=-shift), **kwargs)
    assert right.removed_fraction > 0.8
    assert wrong.removed_fraction < 0.2


def test_a_correction_that_makes_it_worse_says_so():
    """Forcing the scale negative doubles the stripe instead of cancelling it."""
    signal, reference, _ = pair(phase_offset_deg=0.0, amplitude_ratio=0.6)
    shift, band, analysis = measured_shift(signal, reference)
    result = subtract(
        signal, SubtractionPlan(shift_px=shift, scale=-1.5), reference=reference,
        mask_result=analysis.mask, band=band, frequency=analysis.spectrum.centroid_frequency,
    )
    assert result.improved is False
    assert any("did not shrink" in warning for warning in result.warnings)


# -- the two modes ----------------------------------------------------------- #


def test_reference_mode_removes_most_of_the_stripe():
    signal, reference, _ = pair(phase_offset_deg=25.0, amplitude_ratio=0.6, beads=BEADS)
    shift, band, analysis = measured_shift(signal, reference)
    result = subtract(
        signal, SubtractionPlan(shift_px=shift), reference=reference,
        mask_result=analysis.mask, band=band, frequency=analysis.spectrum.centroid_frequency,
    )
    assert result.mode == "reference"
    assert result.removed_fraction > 0.8
    assert result.scale == pytest.approx(1 / 0.6, rel=0.15)
    assert result.corrected.shape == np.asarray(signal).shape


def test_profile_mode_needs_no_reference_and_adds_no_noise():
    array, _ = make_frame(
        FrameSpec(shape=(128, 512), period_px=16.0, amplitude=40.0, noise_sigma=3.0, beads=BEADS)
    )
    analysis = analysed(array)
    result = subtract(
        array, SubtractionPlan(mode="profile"), mask_result=analysis.mask,
        frequency=analysis.spectrum.centroid_frequency,
    )
    assert result.removed_fraction > 0.9
    assert abs(result.noise_change) < 0.05
    assert not result.warnings


def test_profile_mode_fails_when_the_pattern_drifts():
    """The honest caveat: profile subtraction only works line-locked."""
    array, _ = make_frame(
        FrameSpec(shape=(128, 512), period_px=16.0, amplitude=40.0, drift_deg_per_line=6.0)
    )
    analysis = analysed(array)
    result = subtract(
        array, SubtractionPlan(mode="profile"), mask_result=analysis.mask,
        frequency=analysis.spectrum.centroid_frequency,
    )
    assert result.removed_fraction < 0.3


def test_the_raw_reference_injects_its_noise_and_says_so():
    signal, reference, _ = pair(amplitude_ratio=0.6, reference_noise_sigma=25.0)
    shift, band, analysis = measured_shift(signal, reference)
    kwargs = dict(
        reference=reference, mask_result=analysis.mask, band=band,
        frequency=analysis.spectrum.centroid_frequency,
    )
    banded = subtract(signal, SubtractionPlan(shift_px=shift, band_limited=True), **kwargs)
    raw = subtract(signal, SubtractionPlan(shift_px=shift, band_limited=False), **kwargs)

    assert raw.noise_change > banded.noise_change
    assert raw.removed_fraction < banded.removed_fraction
    assert any("brought its own noise" in warning for warning in raw.warnings)


def test_per_line_fitting_produces_one_scale_per_line():
    signal, reference, _ = pair(phase_offset_deg=25.0, amplitude_ratio=0.6)
    shift, band, analysis = measured_shift(signal, reference)
    result = subtract(
        signal, SubtractionPlan(shift_px=shift, per_line=True), reference=reference,
        mask_result=analysis.mask, band=band, frequency=analysis.spectrum.centroid_frequency,
    )
    assert result.per_line
    assert result.scales_per_line is not None
    assert result.scales_per_line.size == np.asarray(signal).shape[0]
    assert result.removed_fraction > 0.8


def test_an_explicit_scale_overrides_the_fit():
    signal, reference, _ = pair()
    shift, band, analysis = measured_shift(signal, reference)
    result = subtract(
        signal, SubtractionPlan(shift_px=shift, scale=0.5), reference=reference,
        mask_result=analysis.mask, band=band, frequency=analysis.spectrum.centroid_frequency,
    )
    assert result.scale == pytest.approx(0.5)


def test_the_background_level_is_preserved():
    """Subtracting must not move the frame's own offset."""
    signal, reference, _ = pair(offset=1000.0)
    shift, band, analysis = measured_shift(signal, reference)
    result = subtract(
        signal, SubtractionPlan(shift_px=shift), reference=reference,
        mask_result=analysis.mask, band=band, frequency=analysis.spectrum.centroid_frequency,
    )
    assert float(np.mean(result.corrected)) == pytest.approx(float(np.mean(signal)), rel=1e-6)


# -- refusals ---------------------------------------------------------------- #


def test_reference_mode_without_a_reference_is_refused():
    array, _ = make_frame(FrameSpec(shape=(32, 128)))
    with pytest.raises(SubtractionError, match="needs a reference"):
        subtract(array, SubtractionPlan(mode="reference"))


def test_a_mismatched_reference_is_refused():
    signal, _ = make_frame(FrameSpec(shape=(64, 128)))
    reference, _ = make_frame(FrameSpec(shape=(64, 64)))
    with pytest.raises(SubtractionError, match="reference is"):
        subtract(signal, SubtractionPlan(), reference=reference)


def test_an_unknown_mode_is_refused():
    with pytest.raises(SubtractionError, match="unknown mode"):
        SubtractionPlan(mode="magic")


def test_a_non_2d_frame_is_refused():
    with pytest.raises(SubtractionError, match="expected a 2-D frame"):
        subtract(np.zeros((2, 8, 8)), SubtractionPlan(mode="profile"))


# -- writing ----------------------------------------------------------------- #


@pytest.fixture
def profile_result():
    array, _ = make_frame(
        FrameSpec(shape=(64, 256), period_px=16.0, amplitude=40.0, noise_sigma=3.0)
    )
    analysis = analysed(array)
    return subtract(
        array, SubtractionPlan(mode="profile"), mask_result=analysis.mask,
        frequency=analysis.spectrum.centroid_frequency,
    )


def test_written_frame_is_float32_and_loses_nothing(profile_result, tmp_path):
    """The corrected values are fractional and can go negative; float32 keeps
    both, where uint16 would round every pixel and clip anything below zero."""
    path, sidecar = write_corrected(tmp_path / "out.tif", profile_result)
    written = tifffile.imread(path)
    assert written.dtype == np.float32
    assert not np.allclose(written, np.round(written))  # genuinely fractional
    np.testing.assert_allclose(written, profile_result.corrected.astype(np.float32), rtol=1e-6)
    assert sidecar is not None and sidecar.exists()


def test_float32_survives_a_frame_that_does_go_negative(tmp_path):
    array, _ = make_frame(
        FrameSpec(shape=(32, 256), period_px=16.0, amplitude=40.0, offset=20.0, noise_sigma=2.0)
    )
    analysis = analysed(array)
    result = subtract(
        array, SubtractionPlan(mode="profile"), mask_result=analysis.mask,
        frequency=analysis.spectrum.centroid_frequency,
    )
    result.corrected = result.corrected - float(np.mean(result.corrected))
    path, _ = write_corrected(tmp_path / "neg.tif", result)
    assert tifffile.imread(path).min() < 0.0


def test_provenance_records_how_it_was_made(profile_result, tmp_path):
    path, _ = write_corrected(
        tmp_path / "out.tif", profile_result,
        signal_path=tmp_path / "signal.tif", reference_path=tmp_path / "reference.tif",
    )
    record = read_provenance(path)
    assert record["mode"] == "profile"
    assert record["dtype"] == "float32"
    assert record["stripe_before"] > record["stripe_after"]
    assert record["signal"].endswith("signal.tif")
    assert "period_px" in record


def test_provenance_also_travels_inside_the_tiff(profile_result, tmp_path):
    path, sidecar = write_corrected(tmp_path / "out.tif", profile_result)
    sidecar.unlink()
    with tifffile.TiffFile(path) as tif:
        record = json.loads(tif.pages[0].description)
    assert record["operation"] == "fixed-pattern subtraction"
    assert read_provenance(path)["mode"] == "profile"  # falls back to the TIFF


def test_a_missing_extension_becomes_tif(profile_result, tmp_path):
    path, _ = write_corrected(tmp_path / "out", profile_result)
    assert path.suffix == ".tif"


def test_default_output_names_say_which_mode(tmp_path):
    source = tmp_path / "medida_001_20mW.tif"
    assert default_output_path(source, "reference").name == "medida_001_20mW-corrected.tif"
    assert default_output_path(source, "profile").name == "medida_001_20mW-deprofiled.tif"


# -- the screen -------------------------------------------------------------- #


@pytest.fixture
def screen_dir(tmp_path):
    signal, reference, _ = pair(phase_offset_deg=25.0, amplitude_ratio=0.6, beads=BEADS)
    directory = tmp_path / "data"
    directory.mkdir()
    tifffile.imwrite(directory / "medida_001_20mW[PMT2].tif", signal)
    tifffile.imwrite(directory / "medida_002_20mW[PMT1].tif", reference)
    return directory


def make_subtraction_screen(screen_dir, keys, with_reference=True):
    from tif_analyzer.discovery import FileRecord
    from tif_analyzer.ui.interference_screen import InterferenceSettings
    from tif_analyzer.ui.subtraction_screen import SubtractionScreen

    from .test_ui import ScriptedReader

    settings = InterferenceSettings(
        signal=FileRecord(screen_dir / "medida_001_20mW[PMT2].tif", "power", 20.0, "20"),
        reference=(
            FileRecord(screen_dir / "medida_002_20mW[PMT1].tif", "power", 20.0, "20")
            if with_reference
            else None
        ),
    )
    return SubtractionScreen(settings, screen_dir, ScriptedReader(keys).read, clear=False)


def test_the_screen_defaults_to_profile_mode_without_a_reference(screen_dir):
    assert make_subtraction_screen(screen_dir, "b", with_reference=False).mode == "profile"
    assert make_subtraction_screen(screen_dir, "b").mode == "reference"


def test_reference_mode_is_blocked_without_a_reference(screen_dir, capsys):
    screen = make_subtraction_screen(screen_dir, ["m", "b"], with_reference=False)
    screen.mode = "reference"
    screen.run()
    output = capsys.readouterr().out
    assert "reference mode needs a reference frame" in output


def test_computing_reports_what_the_correction_did(screen_dir, capsys):
    screen = make_subtraction_screen(screen_dir, ["g", "b"])
    screen.run()
    assert screen.result is not None
    assert screen.result.removed_fraction > 0.8
    output = capsys.readouterr().out
    assert "removed)" in output
    assert "α" in output and "shift" in output


def test_changing_the_plan_drops_the_computed_result(screen_dir):
    screen = make_subtraction_screen(screen_dir, ["g", "n", "b"])
    screen.run()
    assert screen.result is None  # [n] invalidated it
    assert screen.band_limited is False


def test_profile_mode_computes_without_a_reference(screen_dir):
    # No [m] press: with no reference the screen already opens in profile mode.
    screen = make_subtraction_screen(screen_dir, ["g", "b"], with_reference=False)
    screen.run()
    assert screen.result is not None
    assert screen.result.mode == "profile"
    assert screen.result.removed_fraction > 0.9


def test_view_and_write_are_blocked_until_computed(screen_dir, capsys):
    screen = make_subtraction_screen(screen_dir, "b")
    screen.run()
    output = capsys.readouterr().out
    assert "[v] view before/after  (compute it first)" in output
    assert "[w] write the corrected frame  (compute it first)" in output


def test_writing_produces_a_float32_frame_and_a_sidecar(screen_dir, monkeypatch):
    target = screen_dir / "out.tif"
    from tif_analyzer.ui import prompts

    monkeypatch.setattr(prompts, "prompt_export_path", lambda *a, **k: target)
    screen = make_subtraction_screen(screen_dir, ["g", "w", "b"])
    screen.run()
    assert target.exists()
    assert tifffile.imread(target).dtype == np.float32
    assert (screen_dir / "out.subtraction.json").exists()
    record = json.loads((screen_dir / "out.subtraction.json").read_text())
    assert record["signal"].endswith("medida_001_20mW[PMT2].tif")
    assert record["reference"].endswith("medida_002_20mW[PMT1].tif")


def test_the_figure_falls_back_to_a_png_when_headless(screen_dir, capsys):
    screen = make_subtraction_screen(screen_dir, ["g", "v", "b"])
    screen.run()
    saved = list(screen_dir.glob("subtraction_*.png"))
    assert len(saved) == 1
    assert "figure saved to" in capsys.readouterr().out


def test_a_mismatched_reference_is_reported_not_crashed(screen_dir, capsys):
    small, _ = make_frame(FrameSpec(shape=(32, 64)))
    tifffile.imwrite(screen_dir / "medida_002_20mW[PMT1].tif", small)
    screen = make_subtraction_screen(screen_dir, ["g", "b"])
    screen.run()
    assert screen.result is None
    assert "✗" in capsys.readouterr().out


# -- the screen: applying it to the whole series ----------------------------- #


def screen_records(screen_dir):
    from tif_analyzer.discovery import FileRecord

    return [
        FileRecord(path, "power", 20.0, "20")
        for path in sorted(screen_dir.glob("medida_*.tif"))
    ]


def test_apply_to_all_is_blocked_without_matching_files(screen_dir, capsys):
    screen = make_subtraction_screen(screen_dir, "b")
    screen.run()
    assert "(no matching files)" in capsys.readouterr().out


def test_apply_to_all_corrects_every_file_into_its_own_directory(
    screen_dir, monkeypatch, capsys
):
    from tif_analyzer.ui import prompts

    output = screen_dir / "deprofiled"
    monkeypatch.setattr(prompts, "prompt_export_path", lambda *a, **k: output)
    screen = make_subtraction_screen(screen_dir, ["a", "", "b"], with_reference=False)
    screen.records = screen_records(screen_dir)
    screen.run()

    written = sorted(path.name for path in output.glob("*.tif"))
    assert written == ["medida_001_20mW[PMT2].tif", "medida_002_20mW[PMT1].tif"]
    text = capsys.readouterr().out
    assert "of the stripe removed" in text
    assert "2 frame(s) written" in text
    assert "point [d] at that directory" in text


def test_apply_to_all_takes_the_areas_along(screen_dir, monkeypatch, capsys):
    """Areas are pixel coordinates and the corrected frames keep their size."""
    from tif_analyzer import roi_sidecar
    from tif_analyzer.ui import prompts

    (screen_dir / roi_sidecar.SIDECAR_NAME).write_text("{}", encoding="utf-8")
    output = screen_dir / "deprofiled"
    monkeypatch.setattr(prompts, "prompt_export_path", lambda *a, **k: output)
    screen = make_subtraction_screen(screen_dir, ["a", "", "b"], with_reference=False)
    screen.records = screen_records(screen_dir)
    screen.run()
    assert (output / roi_sidecar.SIDECAR_NAME).exists()
    assert "copied across" in capsys.readouterr().out


def test_apply_to_all_warns_that_one_reference_serves_every_file(
    screen_dir, monkeypatch, capsys
):
    from tif_analyzer.ui import prompts

    monkeypatch.setattr(prompts, "prompt_export_path", lambda *a, **k: screen_dir / "corrected")
    screen = make_subtraction_screen(screen_dir, ["a", "", "b"])
    screen.records = screen_records(screen_dir)
    screen.run()
    assert "the same reference frame is used for every one of them" in capsys.readouterr().out


def test_apply_to_all_reports_a_batch_that_cannot_start(screen_dir, monkeypatch, capsys):
    """Writing beside the originals is refused, and the screen says so."""
    from tif_analyzer.ui import prompts

    monkeypatch.setattr(prompts, "prompt_export_path", lambda *a, **k: screen_dir)
    screen = make_subtraction_screen(screen_dir, ["a", "b"], with_reference=False)
    screen.records = screen_records(screen_dir)
    screen.run()
    assert "overwrite" in capsys.readouterr().out
