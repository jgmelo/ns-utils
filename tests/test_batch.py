"""Correcting a whole series: what gets written, where, and what still matches.

The point of the batch is that the corrected frames can be fed straight back
into the series pipeline, so most of these tests end by pointing ``discovery``
at the output directory.
"""

from __future__ import annotations

import csv

import numpy as np
import pytest
import tifffile

from tif_analyzer import discovery
from tif_analyzer.interference import batch as batch_mod
from tif_analyzer.interference.frames import frame_from_array
from tif_analyzer.interference.params import Acquisition, InterferenceParams
from tif_analyzer.interference.synthetic import FrameSpec, make_frame, make_pair
from tif_analyzer.interference.writing import SIDECAR_SUFFIX, read_provenance

PATTERN = r"(?P<power>\d+(?:\.\d+)?)mW"
GLOB = "*.tif"
POWERS = (5, 10, 20)


def write_frame(path, **kwargs):
    spec = FrameSpec(shape=(96, 256), period_px=16.0, amplitude=40.0, noise_sigma=3.0, **kwargs)
    array, _ = make_frame(spec)
    tifffile.imwrite(path, array)
    return array


@pytest.fixture
def series(tmp_path):
    """Three frames, one per power, all carrying the same locked stripe."""
    paths = []
    for index, power in enumerate(POWERS):
        path = tmp_path / f"sample_{power}mW.tif"
        write_frame(path, seed=index)
        paths.append(path)
    return paths


def run(paths, output_dir, **kwargs):
    kwargs.setdefault("mode", "profile")
    return batch_mod.run_batch_blocking(paths, output_dir, **kwargs)


# --------------------------------------------------------------------------- #
# one frame, the shared path
# --------------------------------------------------------------------------- #


def test_correct_frame_removes_the_locked_stripe():
    array, _ = make_frame(FrameSpec(shape=(96, 256), drift_deg_per_line=0.0, noise_sigma=3.0))
    correction = batch_mod.correct_frame(frame_from_array(array), mode="profile")
    assert correction.result.removed_fraction > 0.8
    assert correction.after is None  # not asked for


def test_correct_frame_analyses_the_result_when_asked():
    array, _ = make_frame(FrameSpec(shape=(96, 256)))
    correction = batch_mod.correct_frame(
        frame_from_array(array), mode="profile", analyse_after=True
    )
    assert correction.after is not None


def test_batch_and_screen_agree_on_the_same_frame(tmp_path, series):
    """The screen previews the batch, so both must go through one function."""
    report = run(series, tmp_path / "out")
    first = report.outcomes[0]
    alone = batch_mod.correct_frame(
        frame_from_array(tifffile.imread(series[0])), mode="profile"
    )
    assert first.result.removed_fraction == pytest.approx(alone.result.removed_fraction)


def test_reference_mode_needs_a_reference():
    array, _ = make_frame(FrameSpec(shape=(64, 128)))
    with pytest.raises(Exception):
        batch_mod.correct_frame(frame_from_array(array), mode="reference")


# --------------------------------------------------------------------------- #
# the run
# --------------------------------------------------------------------------- #


def test_every_file_is_written_under_its_own_name(tmp_path, series):
    report = run(series, tmp_path / "out")
    assert len(report.written) == len(series)
    assert not report.failures
    for source in series:
        written = report.output_dir / source.name
        assert written.exists()
        assert written.with_name(written.stem + SIDECAR_SUFFIX).exists()


def test_the_corrected_series_still_matches_the_pattern(tmp_path, series):
    """The reason names are preserved: the same glob and regex must find them."""
    report = run(series, tmp_path / "out")
    found = discovery.discover(report.output_dir, GLOB, PATTERN)
    assert [record.param_value for record in found.records] == [float(p) for p in POWERS]


def test_writing_beside_the_originals_is_refused(tmp_path, series):
    """Same names in the same directory would overwrite the raw data."""
    with pytest.raises(batch_mod.BatchError, match="overwrite"):
        run(series, tmp_path)


def test_the_originals_are_left_alone(tmp_path, series):
    before = [tifffile.imread(path) for path in series]
    run(series, tmp_path / "out")
    after = [tifffile.imread(path) for path in series]
    for original, current in zip(before, after):
        assert np.array_equal(original, current)


def test_the_corrected_frames_are_float32_with_the_background_kept(tmp_path, series):
    report = run(series, tmp_path / "out")
    for outcome in report.written:
        corrected = tifffile.imread(outcome.output)
        original = tifffile.imread(outcome.source)
        assert corrected.dtype == np.float32
        assert corrected.mean() == pytest.approx(original.mean(), rel=1e-3)


def test_each_frame_carries_its_own_provenance(tmp_path, series):
    report = run(series, tmp_path / "out")
    record = read_provenance(report.written[0].output)
    assert record["mode"] == "profile"
    assert record["batch"] is True
    assert record["signal"].endswith(series[0].name)


def test_a_bad_file_does_not_stop_the_others(tmp_path, series):
    broken = tmp_path / "sample_40mW.tif"
    broken.write_bytes(b"not a tiff at all")
    report = run(series + [broken], tmp_path / "out")
    assert len(report.written) == len(series)
    assert [outcome.source.name for outcome in report.failures] == [broken.name]


def test_progress_is_reported_once_per_file(tmp_path, series):
    generator = batch_mod.run_batch(series, tmp_path / "out", mode="profile")
    events = []
    while True:
        try:
            events.append(next(generator))
        except StopIteration:
            break
    assert [event.index for event in events] == [1, 2, 3]
    assert all(event.total == 3 for event in events)


def test_acquisition_and_params_reach_the_frames(tmp_path, series):
    """A different bead cut must change the profile, hence the result."""
    loose = run(series, tmp_path / "loose", params=InterferenceParams(mad_k=3.0))
    tight = run(series, tmp_path / "tight", params=InterferenceParams(mad_k=1.0))
    assert loose.written[0].result.stripe_after != tight.written[0].result.stripe_after
    # And the acquisition is accepted without complaint.
    run(series, tmp_path / "acq", acquisition=Acquisition(line_rate_hz=1000.0))


# --------------------------------------------------------------------------- #
# reference mode
# --------------------------------------------------------------------------- #


@pytest.fixture
def signal_and_reference(tmp_path):
    signal, reference, _ = make_pair(
        FrameSpec(shape=(96, 256), period_px=16.0, amplitude=40.0, noise_sigma=3.0),
        phase_offset_deg=30.0,
    )
    signal_path = tmp_path / "sample_5mW.tif"
    reference_path = tmp_path / "reference_0mW.tif"
    tifffile.imwrite(signal_path, signal)
    tifffile.imwrite(reference_path, reference)
    return signal_path, reference_path


def test_the_reference_is_not_corrected_against_itself(tmp_path, signal_and_reference):
    signal_path, reference_path = signal_and_reference
    report = run(
        [signal_path, reference_path],
        tmp_path / "out",
        mode="reference",
        reference_path=reference_path,
    )
    assert [outcome.source.name for outcome in report.skipped] == [reference_path.name]
    assert len(report.written) == 1


def test_reference_mode_removes_the_stripe(tmp_path, signal_and_reference):
    signal_path, reference_path = signal_and_reference
    report = run(
        [signal_path], tmp_path / "out", mode="reference", reference_path=reference_path
    )
    assert report.written[0].result.removed_fraction > 0.5
    assert read_provenance(report.written[0].output)["reference"].endswith(reference_path.name)


def test_reference_mode_without_a_reference_is_refused(tmp_path, series):
    with pytest.raises(batch_mod.BatchError, match="needs a reference"):
        run(series, tmp_path / "out", mode="reference")


def test_default_output_dir_names_the_mode(tmp_path):
    assert batch_mod.default_output_dir(tmp_path, "profile").name == "deprofiled"
    assert batch_mod.default_output_dir(tmp_path, "reference").name == "corrected"


# --------------------------------------------------------------------------- #
# the summary table
# --------------------------------------------------------------------------- #


def read_summary(report):
    with report.summary_path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_the_summary_has_one_row_per_file(tmp_path, series):
    report = run(series, tmp_path / "out")
    rows = read_summary(report)
    assert [row["filename"] for row in rows] == [path.name for path in series]
    assert all(float(row["removed_fraction"]) > 0.8 for row in rows)


def test_a_failure_is_named_in_the_summary(tmp_path, series):
    broken = tmp_path / "sample_40mW.tif"
    broken.write_bytes(b"not a tiff at all")
    report = run(series + [broken], tmp_path / "out")
    rows = {row["filename"]: row for row in read_summary(report)}
    assert rows[broken.name]["output"] == ""
    assert rows[broken.name]["warnings"]


def test_the_summary_lands_in_the_output_directory(tmp_path, series):
    report = run(series, tmp_path / "out")
    assert report.summary_path.parent == report.output_dir
    assert report.summary_path.name == batch_mod.SUMMARY_NAME


def test_removed_range_spans_the_series(tmp_path, series):
    report = run(series, tmp_path / "out")
    low, high = report.removed_range()
    fractions = [outcome.result.removed_fraction for outcome in report.written]
    assert low == pytest.approx(min(fractions))
    assert high == pytest.approx(max(fractions))
