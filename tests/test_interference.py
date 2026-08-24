"""Fixed-pattern interference: every number checked against planted truth."""

from __future__ import annotations

import numpy as np
import pytest
import tifffile

from tif_analyzer.interference import (
    Acquisition,
    InterferenceParams,
    characterise_arrays,
    render_report,
)
from tif_analyzer.interference import dsp, mask as mask_mod, physical, profile as profile_mod
from tif_analyzer.interference.cli import main as cli_main
from tif_analyzer.interference.frames import (
    FrameError,
    ShapeMismatchError,
    frame_from_array,
    load_frame,
)
from tif_analyzer.interference.params import AcquisitionError
from tif_analyzer.interference.synthetic import FrameSpec, make_frame, make_pair

BEADS = tuple((40 + i * 30, 60 + i * 70, 6.0, 9000.0) for i in range(6))


# -- dsp helpers ------------------------------------------------------------- #


def test_analytic_signal_matches_the_textbook_definition():
    n = 64
    x = np.cos(2 * np.pi * 5 * np.arange(n) / n)
    analytic = dsp.analytic_signal(x)
    np.testing.assert_allclose(analytic.real, x, atol=1e-12)
    # The imaginary part is the Hilbert transform: cos -> sin.
    expected = np.sin(2 * np.pi * 5 * np.arange(n) / n)
    np.testing.assert_allclose(analytic.imag, expected, atol=1e-10)


def test_bandpass_keeps_only_the_named_bins():
    n = 128
    slow = np.sin(2 * np.pi * 4 * np.arange(n) / n)
    fast = np.sin(2 * np.pi * 40 * np.arange(n) / n)
    filtered = dsp.bandpass_by_bins(slow + fast, 3, 5)
    np.testing.assert_allclose(filtered, slow, atol=1e-10)


def test_pearson_signs_and_edges():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    assert dsp.pearson(a, a) == pytest.approx(1.0)
    assert dsp.pearson(a, -a) == pytest.approx(-1.0)
    assert np.isnan(dsp.pearson(a, np.ones(4)))


def test_wrap_deg():
    assert dsp.wrap_deg(370.0) == pytest.approx(10.0)
    assert dsp.wrap_deg(-190.0) == pytest.approx(170.0)
    assert dsp.wrap_deg(180.0) == pytest.approx(180.0)


# -- frames ------------------------------------------------------------------ #


def test_frame_records_shape_dtype_and_statistics():
    array, _ = make_frame(FrameSpec(shape=(64, 128), amplitude=0.0, noise_sigma=0.0, offset=500.0))
    frame = frame_from_array(array)
    assert frame.shape == (64, 128)
    assert frame.n_lines == 64 and frame.n_columns == 128
    assert frame.dtype == "uint16"
    assert frame.median == pytest.approx(500.0, abs=1.0)
    assert frame.saturated is False


def test_saturation_is_flagged_at_the_type_limit():
    array, _ = make_frame(FrameSpec(shape=(64, 128), saturate=True))
    frame = frame_from_array(array)
    assert frame.saturated is True
    assert frame.saturated_pixels > 0
    assert 0.0 < frame.saturated_fraction < 1.0


def test_float_frames_have_no_saturation_level():
    frame = frame_from_array(np.zeros((8, 16)), dtype="float32")
    assert frame.saturation_level is None
    assert frame.saturated is False


def test_non_2d_and_tiny_frames_are_rejected():
    with pytest.raises(FrameError, match="expected a 2-D frame"):
        frame_from_array(np.zeros((2, 4, 4)))
    with pytest.raises(FrameError, match="too small"):
        frame_from_array(np.zeros((1, 2)))


def test_mismatched_shapes_are_rejected():
    signal, _ = make_frame(FrameSpec(shape=(64, 128)))
    reference, _ = make_frame(FrameSpec(shape=(64, 64)))
    with pytest.raises(ShapeMismatchError, match="identical geometry"):
        characterise_arrays(signal, reference)


def test_load_frame_reads_the_on_disk_dtype(tmp_path):
    array, _ = make_frame(FrameSpec(shape=(32, 64)))
    path = tmp_path / "frame.tif"
    tifffile.imwrite(path, array)
    frame = load_frame(path)
    assert frame.dtype == "uint16"
    assert frame.shape == (32, 64)
    assert frame.path == path


# -- mask -------------------------------------------------------------------- #


def test_mask_keeps_background_and_drops_beads():
    array, _ = make_frame(FrameSpec(beads=BEADS))
    result = mask_mod.build_mask(np.asarray(array, dtype=np.float64))
    assert result.converged
    assert 0.0 < result.rejected_fraction < 0.2
    assert result.worst_column_rejected_fraction > result.rejected_fraction
    assert result.reliable


def test_mask_on_a_clean_frame_rejects_almost_nothing():
    array, _ = make_frame(FrameSpec(amplitude=0.0))
    result = mask_mod.build_mask(np.asarray(array, dtype=np.float64))
    assert result.rejected_fraction < 0.02


def test_mask_calls_itself_unreliable_when_nothing_survives():
    # A frame that is mostly bright signal: the background estimate is a fiction.
    array = np.full((64, 128), 50000.0)
    array[:2, :] = 100.0
    params = InterferenceParams(min_background_fraction=0.5)
    result = mask_mod.build_mask(array, params)
    assert result.reliable is False
    assert result.reason


def test_flat_frame_is_reported_not_crashed():
    result = mask_mod.build_mask(np.full((32, 64), 7.0))
    assert result.reliable is False
    assert "no measurable noise" in (result.reason or "")


# -- profile and axis -------------------------------------------------------- #


def test_profile_amplitude_matches_the_planted_stripes():
    array, truth = make_frame(FrameSpec(amplitude=40.0, noise_sigma=2.0))
    result = characterise_arrays(array)
    # A sine of amplitude A has std A/sqrt(2).
    assert result.signal.profile.std == pytest.approx(truth.amplitude / np.sqrt(2), rel=0.05)


def test_axis_check_is_quiet_for_vertical_stripes():
    array, _ = make_frame(FrameSpec(amplitude=40.0))
    check = profile_mod.axis_check(np.asarray(array, dtype=np.float64))
    assert check.warning is None
    assert check.ratio < 1.0


def test_axis_check_warns_when_the_pattern_runs_the_other_way():
    array, _ = make_frame(FrameSpec(amplitude=40.0))
    transposed = np.asarray(array, dtype=np.float64).T
    check = profile_mod.axis_check(transposed)
    assert check.warning is not None
    assert "wrong thing" in check.warning


def test_contamination_check_notices_beads_in_the_raw_profile():
    clean, _ = make_frame(FrameSpec(amplitude=40.0, beads=()))
    dirty, _ = make_frame(FrameSpec(amplitude=40.0, beads=BEADS))
    clean_ratio = characterise_arrays(clean).signal.profile.contamination_rms_ratio
    dirty_ratio = characterise_arrays(dirty).signal.profile.contamination_rms_ratio
    assert dirty_ratio > clean_ratio


def test_flyback_columns_are_dropped_when_they_are_in_the_frame():
    array, _ = make_frame(FrameSpec(shape=(64, 200)))
    acquisition = Acquisition(
        line_rate_hz=1000.0, flyback_fraction=0.1, frame_includes_flyback=True
    )
    result = characterise_arrays(array, acquisition=acquisition)
    assert result.signal.profile.flyback_columns_dropped == 20
    assert result.signal.profile.columns_used == 180


def test_flyback_is_not_dropped_when_the_frame_excludes_it():
    array, _ = make_frame(FrameSpec(shape=(64, 200)))
    result = characterise_arrays(
        array, acquisition=Acquisition(line_rate_hz=1000.0, flyback_fraction=0.1)
    )
    assert result.signal.profile.flyback_columns_dropped == 0
    assert result.signal.profile.columns_used == 200


# -- spectrum ---------------------------------------------------------------- #


@pytest.mark.parametrize("period", [7.3, 13.7, 16.0, 40.0])
def test_centroid_recovers_the_planted_period(period):
    array, _ = make_frame(FrameSpec(period_px=period, amplitude=40.0, noise_sigma=5.0))
    spectrum = characterise_arrays(array).signal.spectrum
    assert spectrum.detected
    assert spectrum.period_px == pytest.approx(period, rel=0.005)
    assert spectrum.centroid_frequency == pytest.approx(1.0 / period, rel=0.005)


def test_a_clean_tone_holds_nearly_all_the_power():
    array, _ = make_frame(FrameSpec(period_px=16.0, amplitude=40.0, noise_sigma=2.0))
    spectrum = characterise_arrays(array).signal.spectrum
    assert spectrum.power_fraction > 0.95
    assert spectrum.peak_to_floor > 100


def test_pure_noise_is_not_detected():
    array, _ = make_frame(FrameSpec(amplitude=0.0, noise_sigma=20.0))
    result = characterise_arrays(array)
    assert result.signal.spectrum.detected is False
    assert "broadband noise" in (result.signal.spectrum.reason or "")
    assert result.verdict.subtraction_viable is None
    assert "INCONCLUSIVE" in result.verdict.headline


def test_the_stronger_of_two_tones_wins():
    array, _ = make_frame(
        FrameSpec(period_px=16.0, amplitude=40.0, extra_tone=(5.0, 8.0), noise_sigma=2.0)
    )
    spectrum = characterise_arrays(array).signal.spectrum
    assert spectrum.period_px == pytest.approx(16.0, rel=0.01)


# -- physical frequency ------------------------------------------------------ #


def test_hz_conversion_follows_the_dwell_time():
    array, _ = make_frame(FrameSpec(shape=(64, 500), period_px=10.0, amplitude=40.0))
    acquisition = Acquisition(line_rate_hz=1000.0)  # 500 px in 1 ms -> 2 us dwell
    result = characterise_arrays(array, acquisition=acquisition)
    assert result.physical.available
    assert result.physical.dwell_s == pytest.approx(2e-6)
    assert result.physical.sampling_hz == pytest.approx(500_000.0)
    assert result.physical.nyquist_hz == pytest.approx(250_000.0)
    # 0.1 cycles/px at 500 kHz sampling -> 50 kHz.
    assert result.physical.observed_hz == pytest.approx(50_000.0, rel=0.01)


def test_flyback_shortens_the_dwell_time():
    array, _ = make_frame(FrameSpec(shape=(64, 500), period_px=10.0))
    plain = characterise_arrays(array, acquisition=Acquisition(line_rate_hz=1000.0))
    with_flyback = characterise_arrays(
        array, acquisition=Acquisition(line_rate_hz=1000.0, flyback_fraction=0.2)
    )
    assert with_flyback.physical.dwell_s == pytest.approx(0.8 * plain.physical.dwell_s)


def test_alias_family_is_emitted_around_the_sampling_rate():
    array, _ = make_frame(FrameSpec(shape=(64, 500), period_px=10.0, amplitude=40.0))
    result = characterise_arrays(array, acquisition=Acquisition(line_rate_hz=1000.0))
    aliases = result.physical.aliases
    assert aliases[0].is_baseband
    sampling = result.physical.sampling_hz
    observed = result.physical.observed_hz
    frequencies = [candidate.frequency_hz for candidate in aliases]
    assert any(f == pytest.approx(sampling - observed) for f in frequencies)
    assert any(f == pytest.approx(sampling + observed) for f in frequencies)
    assert all(f > 0 for f in frequencies)


def test_no_metadata_means_no_hz_rather_than_a_guess():
    array, _ = make_frame(FrameSpec(period_px=16.0, amplitude=40.0))
    result = characterise_arrays(array)
    assert result.physical.available is False
    assert "no line rate" in (result.physical.reason or "")


def test_impossible_acquisition_metadata_is_rejected():
    with pytest.raises(AcquisitionError, match="line rate must be positive"):
        Acquisition(line_rate_hz=-1.0)
    with pytest.raises(AcquisitionError, match="flyback"):
        Acquisition(flyback_fraction=1.5)


def test_alias_disambiguation_admits_when_it_cannot_decide():
    """A broad cluster makes the per-line prediction useless, and the result has
    to say so instead of picking a member anyway."""
    array, _ = make_frame(FrameSpec(shape=(128, 256), period_px=13.7, drift_deg_per_line=0.4))
    result = characterise_arrays(array, acquisition=Acquisition(line_rate_hz=1000.0))
    assert result.alias_match is not None
    assert result.alias_match.informative is False
    assert result.alias_match.candidate is None


def test_alias_match_accepts_a_consistent_prediction():
    predicted = physical._predicted_drift(1000.0, 0.001)  # 1 kHz on a 1 ms line
    assert predicted == pytest.approx(0.0, abs=1e-9)


# -- line stability ---------------------------------------------------------- #


@pytest.mark.parametrize("drift", [0.0, 0.25, -1.5])
def test_phase_drift_recovers_what_was_planted(drift):
    array, _ = make_frame(FrameSpec(period_px=16.0, amplitude=40.0, drift_deg_per_line=drift))
    stability = characterise_arrays(array).stability
    assert stability.available
    assert stability.phase_drift_deg_per_line == pytest.approx(drift, abs=0.01)


def test_amplitude_is_recovered_and_stable():
    array, _ = make_frame(FrameSpec(period_px=16.0, amplitude=40.0, noise_sigma=5.0))
    stability = characterise_arrays(array).stability
    assert stability.amplitude_mean == pytest.approx(40.0, rel=0.1)
    assert stability.amplitude_cv < 0.2


def test_a_locked_pattern_is_called_locked():
    array, _ = make_frame(FrameSpec(period_px=16.0, amplitude=40.0, drift_deg_per_line=0.0))
    result = characterise_arrays(array)
    assert result.stability.locked_to_scan is True
    assert any("locked to the line trigger" in note for note in result.verdict.notes)


def test_a_drifting_pattern_needs_a_simultaneous_reference():
    array, _ = make_frame(FrameSpec(period_px=16.0, amplitude=40.0, drift_deg_per_line=1.0))
    result = characterise_arrays(array)
    assert result.stability.locked_to_scan is False
    assert any("simultaneously acquired reference" in note for note in result.verdict.notes)


def test_lines_swamped_by_beads_are_skipped():
    spec = FrameSpec(shape=(64, 128), period_px=16.0, amplitude=40.0)
    array, _ = make_frame(spec)
    array = np.asarray(array, dtype=np.float64)
    array[10:14, :] = 60000.0  # four lines entirely bead
    result = characterise_arrays(array)
    assert result.stability.lines_used < result.stability.lines_total


# -- cross-channel and verdict ----------------------------------------------- #


def test_matching_channels_are_called_viable():
    signal, reference, _ = make_pair(
        FrameSpec(period_px=16.0, amplitude=40.0, noise_sigma=5.0), phase_offset_deg=0.0
    )
    result = characterise_arrays(signal, reference)
    assert result.cross is not None
    assert result.cross.correlation_band > 0.9
    assert result.verdict.subtraction_viable is True
    assert "VIABLE" in result.verdict.headline


def test_phase_offset_is_recovered_in_degrees_and_pixels():
    signal, reference, _ = make_pair(
        FrameSpec(period_px=16.0, amplitude=40.0, noise_sigma=2.0), phase_offset_deg=45.0
    )
    cross = characterise_arrays(signal, reference).cross
    assert cross is not None
    # The reference was advanced by 45 degrees, so the signal sits behind it.
    assert cross.phase_offset_deg == pytest.approx(-45.0, abs=2.0)
    assert cross.phase_offset_px == pytest.approx(-45.0 / 360.0 * 16.0, abs=0.1)


def test_amplitude_ratio_is_recovered():
    signal, reference, _ = make_pair(
        FrameSpec(period_px=16.0, amplitude=40.0, noise_sigma=2.0), amplitude_ratio=0.5
    )
    cross = characterise_arrays(signal, reference).cross
    assert cross is not None
    assert cross.amplitude_ratio == pytest.approx(2.0, rel=0.1)


def test_different_periods_block_the_verdict():
    signal, _ = make_frame(FrameSpec(period_px=16.0, amplitude=40.0, noise_sigma=2.0))
    reference, _ = make_frame(FrameSpec(period_px=11.0, amplitude=40.0, noise_sigma=2.0, seed=7))
    result = characterise_arrays(signal, reference)
    assert result.verdict.subtraction_viable is False
    assert "NOT SUPPORTED" in result.verdict.headline
    assert any(not check.passed for check in result.verdict.checks)


def test_a_noisy_reference_triggers_the_injection_warning():
    """Stripes agree, everything else does not: raw subtraction would swap the
    pattern for the reference's own noise."""
    signal, reference, _ = make_pair(
        FrameSpec(shape=(24, 512), period_px=16.0, amplitude=40.0, noise_sigma=3.0),
        reference_noise_sigma=400.0,
    )
    result = characterise_arrays(signal, reference)
    assert result.cross is not None
    assert result.cross.correlation_gap > 0.3
    assert any("inject the reference" in warning for warning in result.verdict.warnings)


def test_an_inverted_reference_is_called_out():
    signal, reference, _ = make_pair(
        FrameSpec(period_px=16.0, amplitude=40.0, noise_sigma=2.0), phase_offset_deg=180.0
    )
    result = characterise_arrays(signal, reference)
    assert result.cross is not None
    assert result.cross.correlation_band < -0.8
    assert any("ADDED, not subtracted" in warning for warning in result.verdict.warnings)


def test_no_reference_means_no_decision():
    array, _ = make_frame(FrameSpec(period_px=16.0, amplitude=40.0))
    result = characterise_arrays(array)
    assert result.cross is None
    assert result.verdict.subtraction_viable is None
    assert "no reference frame supplied" in result.verdict.headline


def test_thresholds_come_from_params():
    signal, reference, _ = make_pair(
        FrameSpec(period_px=16.0, amplitude=40.0, noise_sigma=5.0)
    )
    strict = characterise_arrays(
        signal, reference, params=InterferenceParams(band_corr_threshold=0.999999)
    )
    assert strict.verdict.subtraction_viable is False


# -- report and CLI ---------------------------------------------------------- #


def test_report_shows_every_value_next_to_its_threshold():
    signal, reference, _ = make_pair(FrameSpec(period_px=16.0, amplitude=40.0, noise_sigma=5.0))
    text = render_report(characterise_arrays(signal, reference))
    assert "VERDICT" in text
    assert "period agreement (relative)" in text
    assert "band-restricted correlation" in text
    assert ">=" in text and "<=" in text
    assert "provisional defaults" in text
    for heading in ("1. FRAMES", "AXIS CHECK", "2. BEAD MASK", "3. FIXED-PATTERN PROFILE",
                    "4. SPECTRAL ESTIMATE", "5. PHYSICAL FREQUENCY", "6. LINE STABILITY",
                    "7. CROSS-CHANNEL COMPARISON"):
        assert heading in text


def test_report_states_the_nyquist_limit_when_metadata_is_present():
    array, _ = make_frame(FrameSpec(shape=(64, 500), period_px=10.0, amplitude=40.0))
    text = render_report(characterise_arrays(array, acquisition=Acquisition(line_rate_hz=1000.0)))
    assert "NYQUIST LIMIT" in text
    assert "alias family" in text


def test_report_survives_a_frame_with_nothing_in_it():
    array, _ = make_frame(FrameSpec(amplitude=0.0, noise_sigma=20.0))
    text = render_report(characterise_arrays(array))
    assert "NOT DETECTED" in text
    assert "INCONCLUSIVE" in text


@pytest.fixture
def frame_pair(tmp_path):
    signal, reference, _ = make_pair(
        FrameSpec(period_px=16.0, amplitude=40.0, noise_sigma=5.0), phase_offset_deg=20.0
    )
    signal_path = tmp_path / "signal.tif"
    reference_path = tmp_path / "reference.tif"
    tifffile.imwrite(signal_path, signal)
    tifffile.imwrite(reference_path, reference)
    return signal_path, reference_path


def test_cli_returns_zero_when_subtraction_is_viable(frame_pair, capsys):
    signal_path, reference_path = frame_pair
    code = cli_main([str(signal_path), "-r", str(reference_path), "--line-rate", "1000"])
    assert code == 0
    assert "SUBTRACTION VIABLE" in capsys.readouterr().out


def test_cli_returns_four_without_a_reference(frame_pair, capsys):
    signal_path, _ = frame_pair
    assert cli_main([str(signal_path)]) == 4


def test_cli_writes_the_report_to_a_file(frame_pair, tmp_path, capsys):
    signal_path, reference_path = frame_pair
    target = tmp_path / "report.txt"
    cli_main([str(signal_path), "-r", str(reference_path), "-o", str(target)])
    assert "VERDICT" in target.read_text(encoding="utf-8")


def test_cli_reports_bad_metadata_without_a_traceback(frame_pair, capsys):
    signal_path, _ = frame_pair
    assert cli_main([str(signal_path), "--line-rate", "0"]) == 2
    assert "error:" in capsys.readouterr().err


def test_cli_reports_a_shape_mismatch(tmp_path, capsys):
    signal, _ = make_frame(FrameSpec(shape=(64, 128)))
    reference, _ = make_frame(FrameSpec(shape=(64, 64)))
    signal_path = tmp_path / "s.tif"
    reference_path = tmp_path / "r.tif"
    tifffile.imwrite(signal_path, signal)
    tifffile.imwrite(reference_path, reference)
    assert cli_main([str(signal_path), "-r", str(reference_path)]) == 1
    assert "identical geometry" in capsys.readouterr().err


def test_cli_thresholds_can_be_overridden(frame_pair):
    signal_path, reference_path = frame_pair
    code = cli_main(
        [str(signal_path), "-r", str(reference_path), "--band-corr", "0.999999"]
    )
    assert code == 3
