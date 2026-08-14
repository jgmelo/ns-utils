from __future__ import annotations

import numpy as np
import pytest

from tif_analyzer import analysis, discovery, loading, metrics, preprocess
from tif_analyzer.cache import ResultCache, make_key
from tif_analyzer.metrics import MetricContractError, MetricSpec, OutputSpec


def records_of(directory):
    return discovery.discover(directory).records


def test_run_blocking_produces_one_result_per_file(series_dir):
    report = analysis.run_blocking(records_of(series_dir), metrics.get("std"))
    assert report.n_results == 4
    assert report.n_failures == 0
    assert all(set(r.values) == {"std"} for r in report.results)


def test_progress_events_are_emitted_in_order(series_dir):
    generator = analysis.run(records_of(series_dir), metrics.get("std"))
    events = []
    while True:
        try:
            events.append(next(generator))
        except StopIteration as stop:
            report = stop.value
            break
    assert [e.index for e in events] == [1, 2, 3, 4]
    assert all(e.total == 4 for e in events)
    assert all(e.ok for e in events)
    assert report.n_results == 4


def test_series_is_sorted_by_parameter(series_dir):
    report = analysis.run_blocking(records_of(series_dir), metrics.get("std"))
    xs, ys = report.series("std")
    assert xs == sorted(xs)
    assert xs == [1.0, 2.0, 5.0, 10.0]
    assert len(ys) == 4


def test_per_file_failures_do_not_abort_the_run(series_dir, make_stack):
    make_stack(series_dir, "medida_005_20mW.tif", n_pages=2)
    records = records_of(series_dir)
    report = analysis.run_blocking(records, metrics.get("std"))
    assert report.n_results == 4
    assert report.n_failures == 1
    path, error = report.failures[0]
    assert path.name == "medida_005_20mW.tif"
    assert isinstance(error, loading.MultiPageError)


def test_failure_event_carries_the_exception(series_dir):
    (series_dir / "medida_009_9mW.tif").write_bytes(b"not a tiff")
    generator = analysis.run(records_of(series_dir), metrics.get("std"))
    failures = []
    while True:
        try:
            event = next(generator)
        except StopIteration:
            break
        if not event.ok:
            failures.append(event)
    assert len(failures) == 1
    assert isinstance(failures[0].error, loading.LoadError)


def test_metric_contract_errors_propagate(series_dir):
    broken = MetricSpec(
        key="broken",
        name="Broken",
        description="",
        outputs=(OutputSpec("a", "A", None),),
        compute=lambda image: {"b": 1.0},
    )
    with pytest.raises(MetricContractError):
        analysis.run_blocking(records_of(series_dir), broken)


def test_page_policy_is_honoured(tmp_path, make_stack):
    make_stack(tmp_path, "medida_001_1mW.tif", n_pages=2, shape=(2, 2))
    records = records_of(tmp_path)
    assert analysis.run_blocking(records, metrics.get("std")).n_failures == 1
    ok = analysis.run_blocking(records, metrics.get("std"), page_policy="first")
    assert ok.n_results == 1


# -- cache ------------------------------------------------------------------- #


def test_second_run_comes_from_cache(series_dir):
    cache = ResultCache()
    first = analysis.run_blocking(records_of(series_dir), metrics.get("std"), cache=cache)
    second = analysis.run_blocking(records_of(series_dir), metrics.get("std"), cache=cache)
    assert all(not r.from_cache for r in first.results)
    assert all(r.from_cache for r in second.results)
    assert [r.values for r in first.results] == [r.values for r in second.results]
    assert cache.hits == 4


def test_touching_a_file_invalidates_its_entry(series_dir, make_tif):
    cache = ResultCache()
    analysis.run_blocking(records_of(series_dir), metrics.get("std"), cache=cache)
    target = series_dir / "medida_001_1mW.tif"
    make_tif(series_dir, target.name, np.full((6, 6), 7, dtype=np.uint16))
    report = analysis.run_blocking(records_of(series_dir), metrics.get("std"), cache=cache)
    changed = [r for r in report.results if r.record.path.name == target.name][0]
    assert changed.from_cache is False
    assert changed.values["std"] == pytest.approx(0.0)


def test_key_includes_metric_and_preprocess_hash(tmp_path, make_tif):
    path = make_tif(tmp_path, "medida_001_1mW.tif")
    base = make_key(path, "std", preprocess.preprocess_hash([]))
    assert base != make_key(path, "mean", preprocess.preprocess_hash([]))
    assert base != make_key(path, "std", "otherhash")
    assert base == make_key(path, "std", preprocess.preprocess_hash([]))


def test_cache_returns_copies(tmp_path, make_tif):
    path = make_tif(tmp_path, "medida_001_1mW.tif")
    cache = ResultCache()
    key = make_key(path, "std", "h")
    cache.put(key, {"std": 1.0})
    got = cache.get(key)
    got["std"] = 99.0
    assert cache.get(key) == {"std": 1.0}


def test_snapshot_records_provenance(series_dir):
    snapshot = analysis.make_snapshot(
        directory=series_dir,
        glob="*.tif",
        pattern=discovery.DEFAULT_PATTERN,
        metric_key="std",
        steps=[],
        page_policy="error",
        allow_duplicates=False,
        n_files=4,
    )
    assert snapshot["metric"] == "std"
    assert snapshot["n_files"] == 4
    assert snapshot["preprocess"] == "none"
    assert snapshot["timestamp"]
