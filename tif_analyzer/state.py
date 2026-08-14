"""Application state and its invalidation rule.

Every mutation goes through :meth:`AppState.update`. Stale-plot bugs come from
scattered assignments and they are hard to spot, because the output still looks
plausible.
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from .analysis import RunReport
from .discovery import DEFAULT_GLOB, DEFAULT_PATTERN, DiscoveryReport, FileRecord
from .loading import DEFAULT_PAGE_POLICY, PagePolicy
from .preprocess import PreprocessStep

#: Changing any of these clears ``records``, ``discovery`` and ``report``.
INVALIDATING_FIELDS = frozenset(
    {
        "directory",
        "glob",
        "pattern",
        "metric_key",
        "preprocess_steps",
        "allow_duplicates",
        "page_policy",
    }
)


@dataclass
class AppState:
    directory: Path
    glob: str = DEFAULT_GLOB
    pattern: str = DEFAULT_PATTERN
    metric_key: str = "std"
    preprocess_steps: list[PreprocessStep] = field(default_factory=list)
    allow_duplicates: bool = False
    page_policy: PagePolicy = DEFAULT_PAGE_POLICY
    records: list[FileRecord] | None = None
    discovery: DiscoveryReport | None = None
    report: RunReport | None = None
    selected_series: str | None = None

    def update(self, **changes: Any) -> bool:
        """Apply *changes*, clearing derived data when an input changed.

        Returns ``True`` if anything was actually invalidated. Changing
        ``selected_series`` never invalidates: it only affects rendering.
        """
        known = {f.name for f in fields(self)}
        unknown = set(changes) - known
        if unknown:
            raise AttributeError(f"unknown AppState field(s): {sorted(unknown)}")

        invalidate = False
        for name, value in changes.items():
            if getattr(self, name) == value:
                continue
            setattr(self, name, value)
            if name in INVALIDATING_FIELDS:
                invalidate = True
                if name == "metric_key":
                    # The selected series belongs to the previous metric.
                    self.selected_series = None

        if invalidate:
            self.records = None
            self.discovery = None
            self.report = None
        return invalidate

    # -- convenience -------------------------------------------------------- #

    @property
    def has_records(self) -> bool:
        return bool(self.records)

    @property
    def has_results(self) -> bool:
        return self.report is not None and bool(self.report.results)
