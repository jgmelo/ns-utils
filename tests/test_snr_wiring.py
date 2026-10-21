"""Metric contract with regions, cache key, analysis wiring and the sidecar."""

from __future__ import annotations

import json

import numpy as np
import pytest
import tifffile

from tif_analyzer import analysis, discovery, metrics, preprocess, roi_sidecar
from tif_analyzer.cache import ResultCache, make_key
from tif_analyzer.metrics import MissingRegionsError, compute_validated
from tif_analyzer.regions import Ellipse, Rect, RegionOutOfBoundsError, RegionSet, regions_hash

SIGNAL_BOX = Rect(x=2, y=2, width=4, height=4)
NOISE_BOX = Rect(x=20, y=2, width=6, height=6)


def region_set(**kwargs) -> RegionSet:
    defaults = dict(
        signal=SIGNAL_BOX, noise=NOISE_BOX, image_shape=(20, 30), source_file="medida_001_1mW.tif"
    )
    defaults.update(kwargs)
    return RegionSet(**defaults)  # type: ignore[arg-type]


@pytest.fixture
def snr_dir(tmp_path):
    """Images where signal and noise are known exactly.

    Noise area holds the values [0, 1, ..., 35] scaled by the power, so its mean
    and sample std are analytic; the signal area is a flat plateau.
    """
    directory = tmp_path / "data"
    directory.mkdir()
    for index, power in enumerate([1, 2, 4], start=1):
        image = np.zeros((20, 30), dtype=np.uint16)
        image[2:6, 2:6] = 1000 * power
        image[2:8, 20:26] = (np.arange(36).reshape(6, 6) * power).astype(np.uint16)
        tifffile.imwrite(directory / f"medida_{index:03d}_{power}mW.tif", image)
    return directory


def records_of(directory):
    return discovery.discover(directory).records


# -- metric contract --------------------------------------------------------- #


def test_snr_declares_its_regions_and_outputs_statically():
    spec = metrics.get("snr")
    assert spec.requires_regions == ("signal", "noise")
    assert spec.needs_regions is True
    assert spec.output_keys == (
        "snr",
        "signal_mean",
        "noise_mean",
        "noise_std",
        "signal_std",
    )
    assert spec.default_series == "snr"


def test_std_stays_region_free():
    assert metrics.get("std").needs_regions is False
    assert metrics.get("std").requires_regions == ()


def test_snr_computes_the_declared_formula():
    signal = np.full(16, 1000.0)
    noise = np.arange(36, dtype=np.float64)
    values = compute_validated(metrics.get("snr"), np.zeros((2, 2)), {"signal": signal, "noise": noise})
    expected_std = float(np.std(noise, ddof=1))
    assert values["signal_mean"] == 1000.0
    assert values["noise_mean"] == pytest.approx(17.5)
    assert values["noise_std"] == pytest.approx(expected_std)
    assert values["snr"] == pytest.approx((1000.0 - 17.5) / expected_std)


def test_snr_also_reports_the_spread_inside_the_signal_area():
    """Not used by the ratio: it is the only output carrying the sample's own
    photon statistics, which the noise area cannot."""
    signal = np.arange(16, dtype=np.float64)
    noise = np.arange(36, dtype=np.float64)
    values = compute_validated(
        metrics.get("snr"), np.zeros((2, 2)), {"signal": signal, "noise": noise}
    )
    assert values["signal_std"] == pytest.approx(float(np.std(signal, ddof=1)))


def test_a_one_pixel_signal_area_is_refused():
    with pytest.raises(ValueError, match="signal area needs at least 2"):
        compute_validated(
            metrics.get("snr"),
            np.zeros((2, 2)),
            {"signal": np.array([5.0]), "noise": np.arange(9, dtype=np.float64)},
        )


def test_snr_without_regions_raises_the_precondition():
    with pytest.raises(MissingRegionsError, match="needs the signal, noise"):
        compute_validated(metrics.get("snr"), np.zeros((4, 4)))


def test_snr_with_only_one_region_names_what_is_missing():
    with pytest.raises(MissingRegionsError, match="missing: noise"):
        compute_validated(metrics.get("snr"), np.zeros((4, 4)), {"signal": np.arange(4.0)})


def test_flat_noise_area_is_a_per_file_error_not_infinity():
    with pytest.raises(ValueError, match="zero variance"):
        compute_validated(
            metrics.get("snr"),
            np.zeros((4, 4)),
            {"signal": np.full(4, 10.0), "noise": np.full(4, 3.0)},
        )


# -- analysis wiring --------------------------------------------------------- #


def test_run_computes_snr_for_every_file(snr_dir):
    report = analysis.run_blocking(
        records_of(snr_dir), metrics.get("snr"), regions=region_set()
    )
    assert report.n_results == 3
    assert report.n_failures == 0
    first = report.results[0]
    noise = np.arange(36, dtype=np.float64)
    assert first.values["signal_mean"] == pytest.approx(1000.0)
    assert first.values["noise_std"] == pytest.approx(float(np.std(noise, ddof=1)))


def test_snr_series_is_sorted_and_complete(snr_dir):
    report = analysis.run_blocking(
        records_of(snr_dir), metrics.get("snr"), regions=region_set()
    )
    xs, ys = report.series("snr")
    assert xs == [1.0, 2.0, 4.0]
    assert len(ys) == 3
    xs2, means = report.series("signal_mean")
    assert means == pytest.approx([1000.0, 2000.0, 4000.0])


def test_run_without_regions_fails_fast_before_any_file(snr_dir, monkeypatch):
    def explode(*args, **kwargs):  # pragma: no cover - must never run
        raise AssertionError("no file should be opened")

    monkeypatch.setattr("tifffile.TiffFile", explode)
    with pytest.raises(MissingRegionsError, match="select them before running"):
        analysis.run_blocking(records_of(snr_dir), metrics.get("snr"))


def test_smaller_image_becomes_a_collected_failure(snr_dir):
    tifffile.imwrite(snr_dir / "medida_009_9mW.tif", np.zeros((10, 10), dtype=np.uint16))
    report = analysis.run_blocking(
        records_of(snr_dir), metrics.get("snr"), regions=region_set()
    )
    assert report.n_results == 3
    assert report.n_failures == 1
    path, error = report.failures[0]
    assert path.name == "medida_009_9mW.tif"
    assert isinstance(error, RegionOutOfBoundsError)


def test_snapshot_records_the_areas(snr_dir):
    snapshot = analysis.make_snapshot(
        directory=snr_dir,
        glob="*.tif",
        pattern=discovery.DEFAULT_PATTERN,
        metric_key="snr",
        steps=[],
        page_policy="error",
        allow_duplicates=False,
        n_files=3,
        regions=region_set(),
    )
    assert "rect(x=2, y=2, w=4, h=4)" in snapshot["regions"]
    assert snapshot["regions_source"] == "medida_001_1mW.tif"
    assert snapshot["regions_hash"] == regions_hash(region_set())


def test_snapshot_omits_areas_for_region_free_metrics(snr_dir):
    snapshot = analysis.make_snapshot(
        directory=snr_dir,
        glob="*.tif",
        pattern=discovery.DEFAULT_PATTERN,
        metric_key="std",
        steps=[],
        page_policy="error",
        allow_duplicates=False,
        n_files=3,
    )
    assert "regions" not in snapshot


# -- cache ------------------------------------------------------------------- #


def test_moving_an_area_invalidates_the_cache(snr_dir):
    cache = ResultCache()
    first = analysis.run_blocking(
        records_of(snr_dir), metrics.get("snr"), cache=cache, regions=region_set()
    )
    moved = region_set(noise=Rect(x=20, y=2, width=5, height=5))
    second = analysis.run_blocking(
        records_of(snr_dir), metrics.get("snr"), cache=cache, regions=moved
    )
    assert all(not r.from_cache for r in second.results)
    assert first.results[0].values["noise_std"] != second.results[0].values["noise_std"]


def test_same_areas_hit_the_cache(snr_dir):
    cache = ResultCache()
    analysis.run_blocking(records_of(snr_dir), metrics.get("snr"), cache=cache, regions=region_set())
    again = analysis.run_blocking(
        records_of(snr_dir), metrics.get("snr"), cache=cache, regions=region_set()
    )
    assert all(r.from_cache for r in again.results)


def test_redrawing_the_same_boxes_on_another_file_still_hits_the_cache(snr_dir):
    cache = ResultCache()
    analysis.run_blocking(records_of(snr_dir), metrics.get("snr"), cache=cache, regions=region_set())
    again = analysis.run_blocking(
        records_of(snr_dir),
        metrics.get("snr"),
        cache=cache,
        regions=region_set(source_file="medida_002_2mW.tif"),
    )
    assert all(r.from_cache for r in again.results)


def test_key_has_six_elements_and_regions_last(tmp_path, make_tif):
    path = make_tif(tmp_path, "medida_001_1mW.tif")
    key = make_key(path, "snr", preprocess.preprocess_hash([]), regions_hash(region_set()))
    assert len(key) == 6
    assert key[-1] == regions_hash(region_set())
    assert key[-1] != make_key(path, "snr", preprocess.preprocess_hash([]))[-1]


def test_region_free_metrics_keep_an_empty_region_hash(snr_dir):
    cache = ResultCache()
    analysis.run_blocking(records_of(snr_dir), metrics.get("std"), cache=cache)
    key = next(iter(cache._entries))
    assert key[-1] == ""


# -- sidecar ----------------------------------------------------------------- #


def test_sidecar_round_trip(tmp_path):
    regions = region_set(noise=Ellipse(cx=22.0, cy=5.0, rx=3.0, ry=2.0))
    path, notice = roi_sidecar.save(tmp_path, regions)
    assert notice is None
    assert path == tmp_path / ".tif-analyzer-rois.json"
    loaded = roi_sidecar.load(tmp_path)
    assert loaded.regions == regions
    assert loaded.notice is None


def test_missing_sidecar_is_a_non_event(tmp_path):
    loaded = roi_sidecar.load(tmp_path)
    assert loaded.regions is None
    assert loaded.notice is None


def test_corrupt_sidecar_is_reported_not_fatal(tmp_path):
    (tmp_path / ".tif-analyzer-rois.json").write_text("{not json")
    loaded = roi_sidecar.load(tmp_path)
    assert loaded.regions is None
    assert "unreadable" in (loaded.notice or "")


def test_malformed_sidecar_is_reported(tmp_path):
    (tmp_path / ".tif-analyzer-rois.json").write_text(json.dumps({"regions": {}}))
    loaded = roi_sidecar.load(tmp_path)
    assert loaded.regions is None
    assert "malformed" in (loaded.notice or "")


def test_shape_mismatch_refuses_to_apply_the_areas(tmp_path):
    roi_sidecar.save(tmp_path, region_set())
    loaded = roi_sidecar.load(tmp_path, expected_shape=(40, 60))
    assert loaded.regions is None
    assert "drawn on a 30x20 px image but these are 60x40 px" in (loaded.notice or "")


def test_matching_shape_applies_the_areas(tmp_path):
    roi_sidecar.save(tmp_path, region_set())
    assert roi_sidecar.load(tmp_path, expected_shape=(20, 30)).regions == region_set()


def test_clear_removes_the_sidecar(tmp_path):
    roi_sidecar.save(tmp_path, region_set())
    assert roi_sidecar.clear(tmp_path) is True
    assert roi_sidecar.clear(tmp_path) is False
    assert roi_sidecar.load(tmp_path).regions is None


# -- snr_signal: the same areas, the other denominator ----------------------- #


def test_snr_signal_declares_its_regions_and_outputs_statically():
    spec = metrics.get("snr_signal")
    assert spec.requires_regions == ("signal", "noise")
    assert spec.output_keys == (
        "snr_signal",
        "signal_mean",
        "noise_mean",
        "signal_std",
        "noise_std",
    )
    assert spec.default_series == "snr_signal"


def test_snr_signal_divides_by_the_spread_inside_the_signal_area():
    signal = np.arange(16, dtype=np.float64) + 1000.0
    noise = np.arange(36, dtype=np.float64)
    values = compute_validated(
        metrics.get("snr_signal"), np.zeros((2, 2)), {"signal": signal, "noise": noise}
    )
    expected = (float(np.mean(signal)) - float(np.mean(noise))) / float(
        np.std(signal, ddof=1)
    )
    assert values["snr_signal"] == pytest.approx(expected)


def test_the_two_ratios_differ_on_the_same_areas():
    """Same numerator, different denominator — the point of having both."""
    rng = np.random.default_rng(0)
    signal = rng.normal(1000.0, 40.0, size=400)
    noise = rng.normal(100.0, 10.0, size=400)
    areas = {"signal": signal, "noise": noise}
    a = compute_validated(metrics.get("snr"), np.zeros((2, 2)), areas)
    b = compute_validated(metrics.get("snr_signal"), np.zeros((2, 2)), areas)
    assert a["signal_mean"] == b["signal_mean"]
    assert a["noise_mean"] == b["noise_mean"]
    assert a["snr"] != pytest.approx(b["snr_signal"])
    # The noise area is the quieter one here, so it gives the larger ratio.
    assert a["snr"] > b["snr_signal"]


def test_a_flat_signal_area_is_a_per_file_error_not_infinity():
    with pytest.raises(ValueError, match="signal area has zero variance"):
        compute_validated(
            metrics.get("snr_signal"),
            np.zeros((2, 2)),
            {"signal": np.full(16, 500.0), "noise": np.arange(36, dtype=np.float64)},
        )


def test_snr_signal_runs_over_a_series_like_any_other_metric(tmp_path):
    """Needs its own data: the shared fixture's signal area is a flat plateau,
    which ``snr`` tolerates and this metric cannot divide by."""
    directory = tmp_path / "data"
    directory.mkdir()
    rng = np.random.default_rng(0)
    for index, power in enumerate([1, 2, 4], start=1):
        image = rng.integers(90, 111, size=(20, 30)).astype(np.uint16)
        image[2:6, 2:6] += 1000 * power
        tifffile.imwrite(directory / f"medida_{index:03d}_{power}mW.tif", image)

    report = analysis.run_blocking(
        records_of(directory), metrics.get("snr_signal"), regions=region_set()
    )
    assert report.n_results == 3
    assert report.n_failures == 0
    assert all("snr_signal" in r.values for r in report.results)


def test_a_flat_plateau_is_measurable_by_snr_and_not_by_snr_signal(snr_dir):
    """The two metrics refuse different data, and each refusal is correct."""
    records = records_of(snr_dir)
    ok = analysis.run_blocking(records, metrics.get("snr"), regions=region_set())
    refused = analysis.run_blocking(
        records, metrics.get("snr_signal"), regions=region_set()
    )
    assert ok.n_failures == 0
    assert refused.n_results == 0
    assert "signal area has zero variance" in str(refused.failures[0][1])
