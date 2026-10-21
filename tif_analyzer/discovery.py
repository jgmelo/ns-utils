"""Filename discovery and parsing. No image I/O happens in this module.

The pattern is a regex with *exactly one* named group: the group name becomes
the x-axis label and its captured text is parsed as a float.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

DEFAULT_PATTERN = r"medida_\d+_(?P<power>[\d.]+)mW\.tiff?$"
DEFAULT_GLOB = "*.tif"

#: Shown in the pattern prompt so users have something to edit rather than
#: compose from scratch. ``(description, pattern)``.
EXAMPLE_PATTERNS: tuple[tuple[str, str], ...] = (
    ("default: medida_001_1mW.tif", DEFAULT_PATTERN),
    ("power in mW, any prefix", r"(?P<power>[\d.]+)mW\.tiff?$"),
    ("exposure in ms", r"(?P<exposure>[\d.]+)ms\.tiff?$"),
    ("temperature in K, signed", r"(?P<temperature>-?[\d.]+)K\.tiff?$"),
    ("scientific notation gain", r"gain(?P<gain>[\d.eE+-]+)\.tiff?$"),
    ("trailing index, param first", r"(?P<angle>[\d.]+)deg_\d+\.tiff?$"),
)


class DiscoveryError(Exception):
    """Base class for problems that block a run at discovery time."""


class PatternError(DiscoveryError):
    """The regex does not satisfy the one-named-group contract."""


class NoMatchesError(DiscoveryError):
    """Nothing usable came out of discovery."""


@dataclass(frozen=True)
class FileRecord:
    path: Path
    #: ``None`` in single-file mode: the file was named outright rather than
    #: matched, so there is no parameter. The optional type is the point — it
    #: stops a parameterless record from reaching the series pipeline as a
    #: plausible-looking zero.
    param_name: str | None  # from the regex named group, e.g. "power"
    param_value: float | None
    raw_match: str  # the matched substring, for error messages

    @property
    def has_parameter(self) -> bool:
        return self.param_value is not None


@dataclass(frozen=True)
class ParseFailure:
    """Regex matched, but the captured text is not a float."""

    path: Path
    raw_match: str
    reason: str


@dataclass
class DiscoveryReport:
    """Counts and lists produced by :func:`discover`.

    Three counts are distinguished on purpose: a file that matched the glob but
    not the regex is reported, never silently dropped.
    """

    directory: Path
    glob: str
    pattern: str
    param_name: str | None
    records: list[FileRecord] = field(default_factory=list)
    globbed: list[Path] = field(default_factory=list)
    unmatched: list[Path] = field(default_factory=list)
    parse_failures: list[ParseFailure] = field(default_factory=list)
    duplicates: dict[float, list[FileRecord]] = field(default_factory=dict)

    @property
    def n_globbed(self) -> int:
        return len(self.globbed)

    @property
    def n_regex_matched(self) -> int:
        return len(self.records) + len(self.parse_failures)

    @property
    def n_parsed(self) -> int:
        return len(self.records)


class DuplicateParameterError(DiscoveryError):
    """Two or more files map to the same parameter value.

    Carries every collision at once (grouped by value) so the user can fix the
    pattern in one pass instead of playing whack-a-mole, plus the report so the
    caller can still show discovery counts.
    """

    def __init__(self, duplicates: dict[float, list[FileRecord]], report: DiscoveryReport):
        self.duplicates = duplicates
        self.report = report
        n_files = sum(len(v) for v in duplicates.values())
        super().__init__(
            f"{len(duplicates)} duplicate parameter value(s) across {n_files} files"
        )


def validate_pattern(pattern: str) -> re.Pattern[str]:
    """Compile *pattern* and enforce the exactly-one-named-group contract.

    Raises :class:`PatternError` with an actionable message; both checks are
    instant and catching them here beats a baffling zero-match result later.
    """
    try:
        compiled = re.compile(pattern)
    except re.error as exc:
        raise PatternError(f"not a valid regex: {exc}") from exc

    names = sorted(compiled.groupindex, key=lambda n: compiled.groupindex[n])
    if not names:
        raise PatternError(
            "no named group: there is no parameter to plot against. "
            "Name the parameter group, e.g. (?P<power>[\\d.]+)mW"
        )
    if len(names) > 1:
        joined = ", ".join(names)
        raise PatternError(
            f"ambiguous: {len(names)} named groups ({joined}) — which one is the x-axis? "
            "Only the parameter gets a name; match anything else with a bare group, e.g. \\d+"
        )
    return compiled


def param_name_of(pattern: str) -> str:
    """Return the single named group's name, validating the pattern."""
    return next(iter(validate_pattern(pattern).groupindex))


def infer_unit(pattern: str) -> str | None:
    """Best-effort unit for the x-axis, read from the literal text that follows
    the named group in *pattern* (``(?P<power>[\\d.]+)mW\\.tiff?$`` -> ``mW``).

    Returns ``None`` when nothing unit-like follows; the caller then labels the
    axis with the parameter name alone.
    """
    compiled = validate_pattern(pattern)
    name = next(iter(compiled.groupindex))
    marker = f"(?P<{name}>"
    start = pattern.find(marker)
    if start < 0:  # pragma: no cover - group syntax is fixed by re.compile
        return None

    # Walk to the closing parenthesis of the named group, honouring escapes,
    # character classes and nesting.
    i = start + len(marker)
    depth = 1
    in_class = False
    while i < len(pattern):
        ch = pattern[i]
        if ch == "\\":
            i += 2
            continue
        if in_class:
            if ch == "]":
                in_class = False
        elif ch == "[":
            in_class = True
        elif ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
            if depth == 0:
                i += 1
                break
        i += 1
    else:
        return None

    tail = pattern[i:]
    unit = re.match(r"[A-Za-zµμ%°]+", tail)
    if not unit:
        return None
    text = unit.group(0)
    # A trailing quantifier turns the last literal into an optional character
    # ("tiff?"), so it is not part of the unit.
    if len(text) > 1 and tail[len(text) : len(text) + 1] in {"?", "*", "+", "{"}:
        text = text[:-1]
    return text or None


def parse_records(
    paths: Iterable[Path], pattern: str, report: DiscoveryReport
) -> None:
    """Apply *pattern* to *paths*, filling ``records`` / ``unmatched`` /
    ``parse_failures`` on *report*."""
    compiled = validate_pattern(pattern)
    name = next(iter(compiled.groupindex))
    report.param_name = name

    for path in paths:
        match = compiled.search(path.name)
        if match is None:
            report.unmatched.append(path)
            continue
        raw = match.group(name)
        try:
            value = float(raw)
        except (TypeError, ValueError):
            report.parse_failures.append(
                ParseFailure(path, raw if raw is not None else "", "not a number")
            )
            continue
        if value != value or value in (float("inf"), float("-inf")):
            report.parse_failures.append(
                ParseFailure(path, raw, "not a finite number")
            )
            continue
        report.records.append(
            FileRecord(path=path, param_name=name, param_value=value, raw_match=raw)
        )


def find_duplicates(records: Iterable[FileRecord]) -> dict[float, list[FileRecord]]:
    """Group records by parsed float value and return only the collisions.

    Grouping is by the *parsed float*, not the raw string: ``5.0mW``, ``5.00mW``
    and ``05mW`` are the same point and a string comparison would let them
    through into a mangled plot.
    """
    groups: dict[float, list[FileRecord]] = {}
    for record in records:
        if record.param_value is None:
            continue  # parameterless records cannot collide on a value
        groups.setdefault(record.param_value, []).append(record)
    return {
        value: sorted(items, key=lambda r: r.path.name)
        for value, items in sorted(groups.items())
        if len(items) > 1
    }


def single_file_report(path: Path) -> DiscoveryReport:
    """One named file, with no parameter attached.

    The single-frame tools — the interference screen, the subtraction, the area
    selection — open one frame and never plot it against anything, but they read
    their file list from discovery, which until now could only produce records
    that carried a parsed float. A frame whose name holds no number was
    therefore unopenable, which is the series contract leaking into tools that
    are not series tools.

    No glob, no regex, no duplicate check: the user named the file, so there is
    nothing to match and nothing to collide with.
    """
    path = Path(path).expanduser()
    if not path.is_file():
        raise DiscoveryError(f"not a file: {path}")
    report = DiscoveryReport(
        directory=path.parent, glob=path.name, pattern="", param_name=None
    )
    report.globbed = [path]
    report.records = [
        FileRecord(path=path, param_name=None, param_value=None, raw_match=path.name)
    ]
    return report


def discover(
    directory: Path,
    glob: str = DEFAULT_GLOB,
    pattern: str = DEFAULT_PATTERN,
    allow_duplicates: bool = False,
) -> DiscoveryReport:
    """Glob *directory*, parse filenames and detect duplicates. No image I/O.

    Raises :class:`DuplicateParameterError` (before any file is opened) when
    duplicate parameter values exist and *allow_duplicates* is off.
    """
    directory = Path(directory)
    if not directory.is_dir():
        raise DiscoveryError(f"not a directory: {directory}")

    report = DiscoveryReport(
        directory=directory, glob=glob, pattern=pattern, param_name=None
    )
    report.globbed = sorted(p for p in directory.glob(glob) if p.is_file())
    parse_records(report.globbed, pattern, report)
    report.records.sort(key=lambda r: (r.param_value, r.path.name))

    duplicates = find_duplicates(report.records)
    report.duplicates = duplicates
    if duplicates and not allow_duplicates:
        raise DuplicateParameterError(duplicates, report)
    return report
