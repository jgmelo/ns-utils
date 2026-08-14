"""Screen rendering: header, discovery summary, progress, failure report."""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

from .. import metrics
from ..analysis import ProgressEvent, RunReport
from ..discovery import DiscoveryReport
from ..loading import MultiPageError
from ..state import AppState

TITLE = "TIF Series Analyzer"


def term_width(default: int = 100) -> int:
    try:
        return max(60, shutil.get_terminal_size((default, 24)).columns)
    except Exception:  # pragma: no cover
        return default


def clear_screen(enabled: bool = True) -> None:
    if enabled and sys.stdout.isatty():
        print("\x1b[2J\x1b[H", end="")
    else:
        print()


def rule() -> str:
    return "  " + "─" * (term_width() - 4)


def shorten(text: str, limit: int) -> str:
    return text if len(text) <= limit else "…" + text[-(limit - 1) :]


def header(state: AppState, backend_note: str, notices: list[str]) -> str:
    """Renders from ``AppState``, so the user always sees where they are."""
    lines = [f"\n  {TITLE}", ""]

    discovery = state.discovery
    if discovery is None:
        counts = "(not scanned yet)"
    else:
        counts = f"({discovery.n_globbed} files, {discovery.n_parsed} matched)"
        skipped = len(discovery.unmatched) + len(discovery.parse_failures)
        if skipped:
            counts += f"  {skipped} skipped"
    room = max(24, term_width() - len(counts) - 14)
    lines.append(f"  dir:      {shorten(str(state.directory), room)}  {counts}")
    lines.append(f"  pattern:  {state.pattern}")
    lines.append(f"  glob:     {state.glob}")

    spec = metrics.get(state.metric_key)
    lines.append(f"  metric:   {spec.key} — {spec.name}")
    if len(spec.outputs) > 1:
        lines.append(f"  series:   {state.selected_series or spec.default_series}")

    report = state.report
    if report is None:
        lines.append("  results:  —")
    else:
        failures = f"   ({report.n_failures} failures)" if report.n_failures else ""
        cached = sum(1 for r in report.results if r.from_cache)
        cache_note = f"   [{cached} cached]" if cached else ""
        lines.append(f"  results:  {report.n_results} points{failures}{cache_note}")

    for notice in notices:
        lines.append(f"  note:     {notice}")
    lines.append(f"  plot:     {backend_note}")
    return "\n".join(lines)


def menu(state: AppState, blockers: dict[str, str | None]) -> str:
    """Disabled options stay visible with a reason rather than vanishing."""

    def entry(key: str, label: str) -> tuple[str, str]:
        """``(label, reason)`` — the reason is rendered, never used to hide the
        option: a missing line puzzles, a disabled line teaches."""
        return f"[{key}] {label}", blockers.get(key) or ""

    spec = metrics.get(state.metric_key)
    left = [
        entry("d", "directory"),
        entry("m", "metric"),
        entry("r", "run analysis"),
        entry("e", "export CSV"),
    ]
    right = [
        entry("p", "pattern"),
        entry("g", "glob filter"),
        entry("v", "view plot"),
        (f"[!] allow duplicates: {'ON' if state.allow_duplicates else 'OFF'}", ""),
    ]
    if len(spec.outputs) > 1:
        left.append(entry("s", "select series"))
        right.append(("[q] quit", ""))
    else:
        right.append(("", ""))
        left.append(("[q] quit", ""))

    # The column width comes from the labels alone, so a long reason never
    # shifts the second column around between redraws.
    width = max(len(label) for label, _ in left + right) + 4
    lines = [""]
    for (left_label, left_reason), (right_label, right_reason) in zip(left, right):
        if left_reason:
            lines.append(f"  {left_label}  ({left_reason})")
            if right_label:
                lines.append(f"  {'':<{width}}{right_label}" + (f"  ({right_reason})" if right_reason else ""))
            continue
        row = f"  {left_label:<{width}}{right_label}"
        if right_reason:
            row += f"  ({right_reason})"
        lines.append(row.rstrip())
    return "\n".join(lines)


def discovery_summary(discovery: DiscoveryReport) -> str:
    """Three counts, always all three: a file that matched the glob but not the
    regex is reported, never silently dropped."""
    lines = [
        "",
        f"  {discovery.n_globbed} file(s) matched glob {discovery.glob!r}",
        f"  {discovery.n_regex_matched} of those matched the pattern",
        f"  {discovery.n_parsed} of those parsed as a number"
        + (f" (parameter: {discovery.param_name})" if discovery.param_name else ""),
    ]
    if discovery.unmatched:
        lines.append(f"\n  skipped — matched the glob but not the pattern ({len(discovery.unmatched)}):")
        lines.extend(f"    {p.name}" for p in discovery.unmatched[:10])
        if len(discovery.unmatched) > 10:
            lines.append(f"    … and {len(discovery.unmatched) - 10} more")
    if discovery.parse_failures:
        lines.append(f"\n  skipped — captured text is not a number ({len(discovery.parse_failures)}):")
        lines.extend(
            f"    {f.path.name}: captured {f.raw_match!r} ({f.reason})"
            for f in discovery.parse_failures[:10]
        )
        if len(discovery.parse_failures) > 10:
            lines.append(f"    … and {len(discovery.parse_failures) - 10} more")
    return "\n".join(lines)


def duplicate_report(duplicates: dict[float, list], allow_key: str = "!") -> str:
    """All collisions at once, grouped by value — one-at-a-time whack-a-mole
    across 40 files is miserable."""
    lines = [
        "",
        f"  ✗ duplicate parameter values ({len(duplicates)} value(s)) — the run is blocked.",
        "    Values are compared as parsed floats: 5mW, 5.0mW and 05mW are the same point.",
        "",
    ]
    for value, records in duplicates.items():
        lines.append(f"    {value:g}:")
        lines.extend(f"      {r.path.name}  (captured {r.raw_match!r})" for r in records)
    lines.append("")
    lines.append(
        f"    Fix the pattern, or press [{allow_key}] to allow duplicates if replicates are wanted."
    )
    return "\n".join(lines)


def progress_line(event: ProgressEvent, width: int | None = None) -> str:
    """One updating line: scrolling 44 lines buries the failure report."""
    width = width or term_width()
    digits = len(str(event.total))
    tag = ""
    if event.error is not None:
        tag = "  FAILED"
    elif event.from_cache:
        tag = "  (cached)"
    text = f"  [ {event.index:>{digits}}/{event.total} ] {event.path.name}{tag}"
    return text[: width - 1].ljust(width - 1)


def print_progress(event: ProgressEvent) -> None:
    print("\r" + progress_line(event), end="", flush=True)


def failure_report(report: RunReport) -> str:
    """Per-file failures presented together, with partial results still usable."""
    if not report.failures:
        return ""
    lines = ["", f"  {len(report.failures)} file(s) failed — partial results are still usable:"]
    for path, error in report.failures:
        lines.append(f"    {path.name}: {type(error).__name__}: {error}")
    if any(isinstance(error, MultiPageError) for _, error in report.failures):
        lines.append(
            "\n  hint: restart with --page-policy first  (or  mean)  to collapse those stacks."
        )
    return "\n".join(lines)


def run_summary(report: RunReport, elapsed: float) -> str:
    cached = sum(1 for r in report.results if r.from_cache)
    return (
        f"\n  done: {report.n_results} result(s), {report.n_failures} failure(s), "
        f"{cached} from cache, {elapsed:.2f}s"
    )


def pause(read_key, message: str = "  press any key to continue…") -> None:
    print(f"\n{message}", end="", flush=True)
    read_key()
    print()


def export_hint(path: Path) -> str:
    return f"  ✓ exported to {path}"
