"""In-memory result cache (v1).

Keyed on ``(path, mtime_ns, size, metric_key, preprocess_hash)`` so that
re-running after a plot tweak does not re-read large images, while an edited or
replaced file is still recomputed.

A persistent on-disk cache (sidecar JSON) is the natural extension; it is
deliberately out of scope for v1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

CacheKey = tuple[str, int, int, str, str]


def make_key(path: Path, metric_key: str, preprocess_hash: str) -> CacheKey:
    """Build the cache key for *path*. Stats the file; raises ``OSError`` if it
    vanished since discovery (handled as an ordinary per-file failure)."""
    stat = Path(path).stat()
    return (str(Path(path).resolve()), stat.st_mtime_ns, stat.st_size, metric_key, preprocess_hash)


@dataclass
class ResultCache:
    """Metric values keyed as per locked decision 5."""

    _entries: dict[CacheKey, dict[str, float]] = field(default_factory=dict)
    hits: int = 0
    misses: int = 0

    def get(self, key: CacheKey) -> dict[str, float] | None:
        values = self._entries.get(key)
        if values is None:
            self.misses += 1
            return None
        self.hits += 1
        return dict(values)

    def put(self, key: CacheKey, values: dict[str, float]) -> None:
        self._entries[key] = dict(values)

    def clear(self) -> None:
        self._entries.clear()
        self.hits = 0
        self.misses = 0

    def __len__(self) -> int:
        return len(self._entries)
