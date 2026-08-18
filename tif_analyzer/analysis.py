"""Orchestration: load → preprocess → compute.

:func:`run` is a generator so a UI can show progress, but it is equally
driveable from a script or a notebook via :func:`run_blocking`. Per-file
failures are collected rather than aborting the run; programming errors (a
metric breaking its own output contract) are allowed to propagate.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator, Iterable, Sequence

from . import loading, metrics, preprocess, regions as regions_mod
from .cache import ResultCache, make_key
from .discovery import FileRecord
from .loading import PagePolicy
from .metrics import MetricContractError, MetricSpec, MissingRegionsError
from .preprocess import PreprocessStep
from .regions import RegionSet


@dataclass
class MetricResult:
    record: FileRecord
    values: dict[str, float]
    duration_s: float
    from_cache: bool


@dataclass
class RunReport:
    results: list[MetricResult] = field(default_factory=list)
    failures: list[tuple[Path, Exception]] = field(default_factory=list)
    config_snapshot: dict = field(default_factory=dict)

    @property
    def n_results(self) -> int:
        return len(self.results)

    @property
    def n_failures(self) -> int:
        return len(self.failures)

    def series(self, output_key: str) -> tuple[list[float], list[float]]:
        """``(param_values, values)`` for *output_key*, sorted by parameter."""
        points = sorted(
            ((r.record.param_value, r.values[output_key]) for r in self.results),
            key=lambda pair: pair[0],
        )
        return [p[0] for p in points], [p[1] for p in points]


@dataclass(frozen=True)
class ProgressEvent:
    """Emitted once per file, before the next one is started."""

    index: int  # 1-based
    total: int
    path: Path
    from_cache: bool
    error: Exception | None = None

    @property
    def ok(self) -> bool:
        return self.error is None


def make_snapshot(
    directory: Path,
    glob: str,
    pattern: str,
    metric_key: str,
    steps: Sequence[PreprocessStep],
    page_policy: str,
    allow_duplicates: bool,
    n_files: int,
    regions: RegionSet | None = None,
) -> dict:
    """The provenance record carried by :class:`RunReport`, the plot title and
    the CSV header."""
    snapshot = {
        "directory": str(directory),
        "glob": glob,
        "pattern": pattern,
        "metric": metric_key,
        "preprocess": preprocess.describe(steps),
        "preprocess_hash": preprocess.preprocess_hash(steps),
        "page_policy": page_policy,
        "allow_duplicates": allow_duplicates,
        "n_files": n_files,
        "timestamp": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
    }
    if regions is not None:
        snapshot["regions"] = regions.describe()
        snapshot["regions_source"] = regions.source_file or "?"
        snapshot["regions_hash"] = regions_mod.regions_hash(regions)
    return snapshot


def run(
    records: Iterable[FileRecord],
    metric: MetricSpec,
    *,
    steps: Sequence[PreprocessStep] = (),
    cache: ResultCache | None = None,
    page_policy: PagePolicy = loading.DEFAULT_PAGE_POLICY,
    snapshot: dict | None = None,
    regions: RegionSet | None = None,
) -> Generator[ProgressEvent, None, RunReport]:
    """Process *records*, yielding one :class:`ProgressEvent` per file.

    The finished :class:`RunReport` is the generator's return value::

        gen = analysis.run(records, metric)
        while True:
            try:
                event = next(gen)
            except StopIteration as stop:
                report = stop.value
                break

    A region-aware metric needs *regions*; that precondition is checked once,
    before any file is opened, rather than failing identically on every file.
    """
    if metric.needs_regions and regions is None:
        raise MissingRegionsError(
            f"metric {metric.key!r} needs the {', '.join(metric.requires_regions)} area(s); "
            "select them before running"
        )

    records = list(records)
    total = len(records)
    report = RunReport(config_snapshot=dict(snapshot or {}))
    phash = preprocess.preprocess_hash(steps)
    rhash = regions_mod.regions_hash(regions) if metric.needs_regions else ""

    for index, record in enumerate(records, start=1):
        started = time.perf_counter()
        from_cache = False
        try:
            values: dict[str, float] | None = None
            key = None
            if cache is not None:
                key = make_key(record.path, metric.key, phash, rhash)
                values = cache.get(key)
                from_cache = values is not None

            if values is None:
                image = loading.load_image(record.path, page_policy)
                image = preprocess.apply_pipeline(image, steps)
                samples = regions.extract(image) if metric.needs_regions and regions else None
                values = metrics.compute_validated(metric, image, samples)
                if cache is not None and key is not None:
                    cache.put(key, values)

            report.results.append(
                MetricResult(
                    record=record,
                    values=values,
                    duration_s=time.perf_counter() - started,
                    from_cache=from_cache,
                )
            )
            yield ProgressEvent(index=index, total=total, path=record.path, from_cache=from_cache)
        except MetricContractError:
            # A metric breaking its own declaration is a bug, not a data
            # problem. Do not defend against your own bugs by swallowing them.
            raise
        except Exception as exc:  # noqa: BLE001 - per-file failures are collected
            report.failures.append((record.path, exc))
            yield ProgressEvent(
                index=index, total=total, path=record.path, from_cache=False, error=exc
            )

    return report


def run_blocking(
    records: Iterable[FileRecord],
    metric: MetricSpec,
    **kwargs,
) -> RunReport:
    """Drive :func:`run` to completion, discarding progress. For scripts and
    notebooks."""
    generator = run(records, metric, **kwargs)
    while True:
        try:
            next(generator)
        except StopIteration as stop:
            return stop.value
