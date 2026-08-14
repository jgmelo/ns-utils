"""CSV export, long format.

Long beats wide: it survives metrics with differing output sets and pivots
trivially in pandas. The ``config_snapshot`` is prepended as ``#``-commented
header lines, so an exported file records how it was produced.
"""

from __future__ import annotations

import csv
from pathlib import Path

from .analysis import RunReport

HEADER = ("filename", "param_name", "param_value", "output_key", "value")


def write_csv(report: RunReport, path: Path) -> Path:
    """Write *report* to *path* in long format and return the path."""
    path = Path(path).expanduser()
    if path.suffix == "":
        path = path.with_suffix(".csv")
    path.parent.mkdir(parents=True, exist_ok=True)

    rows = sorted(
        (
            (
                result.record.path.name,
                result.record.param_name,
                result.record.param_value,
                key,
                result.values[key],
            )
            for result in report.results
            for key in result.values
        ),
        key=lambda row: (row[2], row[0], row[3]),
    )

    with path.open("w", newline="", encoding="utf-8") as handle:
        for key, value in report.config_snapshot.items():
            handle.write(f"# {key}: {value}\n")
        if report.failures:
            handle.write(f"# failures: {len(report.failures)} file(s) not included\n")
            for failed, error in report.failures:
                handle.write(f"#   {failed.name}: {type(error).__name__}: {error}\n")
        writer = csv.writer(handle)
        writer.writerow(HEADER)
        for row in rows:
            writer.writerow([row[0], row[1], _fmt(row[2]), row[3], _fmt(row[4])])
    return path


def _fmt(value: float) -> str:
    """Compact but lossless-enough numeric text."""
    text = f"{value:.6g}"
    return text
