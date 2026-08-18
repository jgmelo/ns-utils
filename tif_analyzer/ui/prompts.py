"""Typed-text sub-prompts: line input with tab completion and history.

Entering a long absolute path without completion is unpleasant enough that users
avoid the feature, so path prompts get ``readline`` completion. Every prompt can
be abandoned with Esc or Ctrl-C, returning ``None`` and leaving state untouched.
"""

from __future__ import annotations

import glob as globmod
import os
from pathlib import Path
from typing import Callable, Sequence

from .. import metrics
from ..discovery import EXAMPLE_PATTERNS, PatternError, validate_pattern

try:  # not available on every platform
    import readline as _readline
except Exception:  # pragma: no cover - platform dependent
    _readline = None

CANCEL_HINT = "Esc + Enter (or Ctrl-C) to go back"


def _set_completer(completer: Callable[[str, int], str | None] | None) -> None:
    if _readline is None:
        return
    if completer is None:
        _readline.set_completer(None)
        return
    _readline.set_completer_delims(" \t\n")
    _readline.parse_and_bind("tab: complete")
    _readline.set_completer(completer)


def _path_completer(text: str, state: int) -> str | None:
    expanded = os.path.expanduser(text)
    matches = []
    for candidate in sorted(globmod.glob(expanded + "*")):
        if os.path.isdir(candidate):
            candidate += os.sep
        if text.startswith("~"):
            home = os.path.expanduser("~")
            if candidate.startswith(home):
                candidate = "~" + candidate[len(home) :]
        matches.append(candidate)
    return matches[state] if state < len(matches) else None


def prompt_line(
    message: str,
    default: str = "",
    completer: Callable[[str, int], str | None] | None = None,
) -> str | None:
    """Read a line, pre-filled with *default*. ``None`` means cancelled."""
    _set_completer(completer)
    if _readline is not None and default:
        _readline.set_startup_hook(lambda: _readline.insert_text(default))
    try:
        raw = input(message)
    except (EOFError, KeyboardInterrupt):
        print()
        return None
    finally:
        if _readline is not None:
            _readline.set_startup_hook()
        _set_completer(None)

    if "\x1b" in raw:  # Esc pressed before Enter
        return None
    raw = raw.strip()
    if not raw:
        # No readline (so no prefill): a bare Enter accepts the remembered value.
        return default if default else None
    return raw


def prompt_directory(default: Path) -> Path | None:
    """Ask for a directory until it exists, or the user backs out.

    ``Path(raw).expanduser().resolve()`` handles relative, absolute and ``~``
    uniformly; validation is immediate.
    """
    print(f"\n  Directory to scan   ({CANCEL_HINT}, Tab completes paths)")
    while True:
        raw = prompt_line("  dir> ", str(default), _path_completer)
        if raw is None:
            return None
        candidate = Path(raw).expanduser()
        try:
            candidate = candidate.resolve()
        except OSError as exc:
            print(f"  ! cannot resolve {raw!r}: {exc}")
            continue
        if candidate.is_dir():
            return candidate
        if candidate.exists():
            print(f"  ! not a directory: {candidate}")
        else:
            print(f"  ! does not exist: {candidate}")


def prompt_pattern(default: str) -> str | None:
    """Ask for the filename regex, validating it before accepting."""
    print(f"\n  Filename pattern — regex with exactly one named group   ({CANCEL_HINT})")
    print("  examples:")
    width = max(len(desc) for desc, _ in EXAMPLE_PATTERNS)
    for description, example in EXAMPLE_PATTERNS:
        print(f"    {description:<{width}}  {example}")
    print()
    while True:
        raw = prompt_line("  pattern> ", default)
        if raw is None:
            return None
        try:
            validate_pattern(raw)
        except PatternError as exc:
            print(f"  ! {exc}")
            continue
        return raw


def prompt_glob(default: str) -> str | None:
    print(f"\n  Glob filter applied to the directory   ({CANCEL_HINT})")
    print("  examples:  *.tif    *.tif*    medida_*.tif")
    raw = prompt_line("  glob> ", default)
    if raw is None:
        return None
    if any(sep in raw for sep in ("/", os.sep)):
        print("  ! recursive / nested globs are out of scope in v1 — keep it to a filename pattern")
        return None
    return raw


def prompt_export_path(default: Path) -> Path | None:
    print(f"\n  Export CSV to   ({CANCEL_HINT}, Tab completes paths)")
    raw = prompt_line("  file> ", str(default), _path_completer)
    if raw is None:
        return None
    return Path(raw).expanduser()


def choose_from_menu(
    title: str,
    entries: Sequence[tuple[str, str]],
    current: str | None,
    read_key: Callable[[], str],
) -> str | None:
    """Numbered single-key chooser. ``entries`` is ``(key, label)``."""
    print(f"\n  {title}   (Esc or b to go back)")
    for index, (key, label) in enumerate(entries, start=1):
        marker = "*" if key == current else " "
        print(f"   {marker} [{index}] {label}")
    print()
    while True:
        pressed = read_key()
        if pressed in ("esc", "b"):
            return None
        if pressed.isdigit():
            index = int(pressed) - 1
            if 0 <= index < len(entries):
                return entries[index][0]
        print("  ! pick a number from the list, or Esc/b to go back")


def metric_entries() -> list[tuple[str, str]]:
    entries = []
    for spec in metrics.available():
        needs = (
            f"  [needs the {', '.join(spec.requires_regions)} areas]"
            if spec.needs_regions
            else ""
        )
        entries.append((spec.key, f"{spec.key} — {spec.name}{needs}: {spec.description}"))
    return entries


def prompt_reference_file(records: Sequence, limit: int = 20):
    """Pick the image the areas are drawn on. ``None`` means cancelled.

    The default is the first matched file; a number picks from the list and any
    other text is matched against filenames, so a long series does not force
    scrolling through 44 numbered lines.
    """
    if not records:
        return None
    print(f"\n  Reference image for the areas   ({CANCEL_HINT})")
    for index, record in enumerate(records[:limit], start=1):
        print(f"    [{index:>2}] {record.path.name}   ({record.param_name}={record.param_value:g})")
    if len(records) > limit:
        print(f"    … and {len(records) - limit} more — type part of a filename to pick one")

    while True:
        # Default "1" so a bare Enter takes the first file; None is a real cancel.
        raw = prompt_line(f"  file [Enter = {records[0].path.name}]> ", "1")
        if raw is None:
            return None
        if raw.isdigit():
            index = int(raw) - 1
            if 0 <= index < len(records):
                return records[index]
            print(f"  ! pick a number between 1 and {len(records)}")
            continue
        matches = [r for r in records if raw.lower() in r.path.name.lower()]
        if len(matches) == 1:
            return matches[0]
        if not matches:
            print(f"  ! no filename contains {raw!r}")
        else:
            print(f"  ! {len(matches)} files match {raw!r} — be more specific")


def series_entries(spec: metrics.MetricSpec) -> list[tuple[str, str]]:
    return [
        (out.key, f"{out.key} — {out.label}" + (f" [{out.unit}]" if out.unit else ""))
        for out in spec.outputs
    ]
