"""Single-key input with a graceful degradation path.

``readchar`` is a soft dependency: if it is missing, or raw mode fails (some IDE
consoles), the reader degrades to key-plus-Enter via :func:`input` rather than
crashing.
"""

from __future__ import annotations

import sys

try:  # soft dependency
    import readchar as _readchar
except Exception:  # pragma: no cover - depends on the environment
    _readchar = None

ESC = "esc"
ENTER = "enter"


class KeyReader:
    """Reads one keypress, normalised to a lowercase single character, ``esc``
    or ``enter``."""

    def __init__(self, force_line_mode: bool = False):
        self._readchar = None if force_line_mode else _readchar
        self.degraded_reason: str | None = None
        if self._readchar is None and not force_line_mode:
            self.degraded_reason = "readchar not installed"
        elif force_line_mode:
            self.degraded_reason = "line mode requested"

    @property
    def single_key(self) -> bool:
        return self._readchar is not None

    @property
    def hint(self) -> str:
        if self.single_key:
            return "single keypress, no Enter needed"
        return f"key + Enter ({self.degraded_reason})"

    def _degrade(self, reason: str) -> None:
        self._readchar = None
        self.degraded_reason = reason

    def read(self) -> str:
        if self._readchar is not None:
            try:
                raw = self._readchar.readkey()
            except KeyboardInterrupt:
                raise
            except Exception as exc:  # raw mode unavailable in this console
                self._degrade(f"raw mode unavailable: {type(exc).__name__}")
            else:
                return self._normalise(raw)
        return self._read_line()

    def _read_line(self) -> str:
        try:
            raw = input()
        except EOFError:
            return "q"
        raw = raw.strip()
        if not raw:
            return ENTER
        return self._normalise(raw[0])

    @staticmethod
    def _normalise(raw: str) -> str:
        if not raw:
            return ENTER
        if raw in ("\r", "\n"):
            return ENTER
        if raw.startswith("\x1b"):
            return ESC
        if raw in ("\x03", "\x04"):  # Ctrl-C / Ctrl-D
            return "q"
        return raw[0].lower()


def supports_ansi() -> bool:
    return sys.stdout.isatty()
