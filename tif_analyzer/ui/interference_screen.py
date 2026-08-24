"""The interference screen: fill in the acquisition, then run the report.

A second screen rather than a straight-to-report key, because the physical
frequency needs metadata the series menu has nowhere to hold. Everything the
characterisation can take is editable here, and the screen shows the sampling
rate and Nyquist limit implied by the current numbers *before* anything runs —
so a wrong line rate is visible as a wrong Nyquist, not as a wrong answer.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .. import acquisition_sidecar, loading
from ..discovery import FileRecord
from ..interference import characterise, render_report
from ..interference.frames import FrameError
from ..interference.params import Acquisition, AcquisitionError
from . import prompts, render
from .prompts import CLEARED


@dataclass
class InterferenceSettings:
    """Everything the screen collects. Kept out of ``AppState`` on purpose:
    none of it touches the series pipeline, so none of it may invalidate it."""

    signal: FileRecord | None = None
    reference: FileRecord | None = None
    line_rate_hz: float | None = None
    resolution: int | None = None
    flyback_fraction: float = 0.0
    frame_includes_flyback: bool = False
    frame_width: int | None = None  # read from the signal frame's header

    def acquisition(self) -> Acquisition:
        return Acquisition(
            line_rate_hz=self.line_rate_hz,
            resolution=self.resolution,
            flyback_fraction=self.flyback_fraction,
            frame_includes_flyback=self.frame_includes_flyback,
        )

    def adopt(self, acquisition: Acquisition) -> None:
        self.line_rate_hz = acquisition.line_rate_hz
        self.resolution = acquisition.resolution
        self.flyback_fraction = acquisition.flyback_fraction
        self.frame_includes_flyback = acquisition.frame_includes_flyback

    @property
    def pixels_per_line(self) -> int | None:
        return self.resolution or self.frame_width

    def timing(self) -> tuple[float, float, float] | None:
        """``(dwell_s, sampling_hz, nyquist_hz)`` for the current numbers."""
        pixels = self.pixels_per_line
        if self.line_rate_hz is None or not pixels:
            return None
        dwell = self.acquisition().dwell_s(pixels)
        if dwell is None or dwell <= 0:
            return None
        sampling = 1.0 / dwell
        return dwell, sampling, sampling / 2.0


def _fmt_hz(value: float) -> str:
    if value >= 1e6:
        return f"{value / 1e6:.4g} MHz"
    if value >= 1e3:
        return f"{value / 1e3:.4g} kHz"
    return f"{value:.4g} Hz"


class InterferenceScreen:
    """Owns its own key dispatch; returns to MAIN on [b] or Esc."""

    def __init__(
        self,
        records: list[FileRecord],
        directory: Path,
        read_key,
        page_policy: str = "error",
        clear: bool = True,
        persist: bool = True,
    ):
        self.records = records
        self.directory = directory
        self.read_key = read_key
        self.page_policy = page_policy
        self.clear = clear
        self.persist = persist
        self.settings = InterferenceSettings()
        self.messages: list[str] = []
        self.last_report: str | None = None

        if records:
            self.settings.signal = records[0]
            self._read_frame_width()
        loaded = acquisition_sidecar.load(directory)
        if loaded.notice:
            self.messages.append(f"  ! {loaded.notice}")
        if loaded.acquisition is not None:
            self.settings.adopt(loaded.acquisition)
            self.messages.append(
                f"  ✓ acquisition loaded from {acquisition_sidecar.SIDECAR_NAME}"
            )

    # -- helpers ------------------------------------------------------------ #

    def _read_frame_width(self) -> None:
        """Header-only read, so the implied Nyquist can be shown before a run."""
        self.settings.frame_width = None
        if self.settings.signal is None:
            return
        try:
            self.settings.frame_width = loading.image_shape(self.settings.signal.path)[1]
        except loading.LoadError:
            pass

    def _persist(self) -> None:
        if not self.persist:
            return
        _, notice = acquisition_sidecar.save(self.directory, self.settings.acquisition())
        if notice:
            self.messages.append(f"  ! {notice}")

    def blockers(self) -> dict[str, str]:
        blocked: dict[str, str] = {}
        if self.settings.signal is None:
            blocked["g"] = "pick a signal frame first"
        if self.last_report is None:
            blocked["o"] = "run the report first"
        if self.settings.reference is None:
            blocked["c"] = "no reference set"
        return blocked

    # -- rendering ---------------------------------------------------------- #

    def header(self) -> str:
        settings = self.settings
        lines = ["\n  Interference — fixed-pattern characterisation", ""]
        lines.append(
            f"  signal:      {settings.signal.path.name if settings.signal else '(none — press [f])'}"
        )
        lines.append(
            f"  reference:   {settings.reference.path.name if settings.reference else '(none)'}"
        )
        lines.append(
            "  line rate:   "
            + (f"{settings.line_rate_hz:g} Hz" if settings.line_rate_hz else "(not set)")
        )
        resolution = (
            f"{settings.resolution} px"
            if settings.resolution
            else (
                f"(frame width: {settings.frame_width} px)"
                if settings.frame_width
                else "(not set)"
            )
        )
        lines.append(f"  resolution:  {resolution}")
        lines.append(
            f"  flyback:     {settings.flyback_fraction:.1%}"
            f"   (in the frame: {'YES' if settings.frame_includes_flyback else 'no'})"
        )

        timing = settings.timing()
        if timing is None:
            lines.append("  timing:      — set the line rate to get Hz instead of cycles/pixel")
        else:
            dwell, sampling, nyquist = timing
            lines.append(
                f"  timing:      dwell {dwell * 1e9:.2f} ns/px   "
                f"sampling {_fmt_hz(sampling)}   Nyquist {_fmt_hz(nyquist)}"
            )
        return "\n".join(lines)

    def menu(self) -> str:
        blocked = self.blockers()

        def entry(key: str, label: str) -> str:
            reason = blocked.get(key)
            text = f"[{key}] {label}"
            return f"{text}  ({reason})" if reason else text

        rows = [
            (entry("f", "signal frame"), entry("r", "reference frame")),
            (entry("l", "line rate"), entry("n", "resolution")),
            (
                entry("y", "flyback fraction"),
                f"[w] flyback in the frame: {'YES' if self.settings.frame_includes_flyback else 'NO'}",
            ),
            (entry("g", "run the report"), entry("o", "save last report")),
            (entry("c", "clear reference"), "[b] back to the main menu"),
        ]
        width = max(len(left) for left, _ in rows) + 4
        lines = [""]
        for left, right in rows:
            lines.append(f"  {left:<{width}}{right}".rstrip())
        return "\n".join(lines)

    def draw(self) -> None:
        render.clear_screen(self.clear)
        print(self.header())
        print(self.menu())
        for message in self.messages:
            print(f"\n{message}")
        self.messages.clear()
        print("\n  > ", end="", flush=True)

    # -- actions ------------------------------------------------------------ #

    def action_signal(self) -> None:
        chosen = prompts.prompt_reference_file(
            self.records, title="Signal frame — the one to characterise"
        )
        if chosen is None:
            return
        self.settings.signal = chosen
        self._read_frame_width()

    def action_reference(self) -> None:
        chosen = prompts.prompt_reference_file(
            self.records, title="Reference frame — acquired simultaneously with the signal"
        )
        if chosen is None:
            return
        if self.settings.signal is not None and chosen.path == self.settings.signal.path:
            self.messages.append("  ! the reference must be a different file from the signal")
            return
        self.settings.reference = chosen

    def action_clear_reference(self) -> None:
        self.settings.reference = None

    def action_line_rate(self) -> None:
        value = prompts.prompt_number(
            "Line rate in Hz (scan lines per second)",
            self.settings.line_rate_hz,
            kind=float,
            minimum=1e-9,
        )
        if value is None:
            return
        self.settings.line_rate_hz = None if value is CLEARED else float(value)
        self._persist()

    def action_resolution(self) -> None:
        value = prompts.prompt_number(
            "Pixels per scan line (blank/none = use the frame width)",
            self.settings.resolution,
            kind=int,
            minimum=2,
        )
        if value is None:
            return
        self.settings.resolution = None if value is CLEARED else int(value)
        self._persist()

    def action_flyback(self) -> None:
        value = prompts.prompt_number(
            "Flyback fraction of the line period (0–1, or a percentage like 10%)",
            self.settings.flyback_fraction,
            kind=float,
            minimum=0.0,
            maximum=1.0,
            percent_ok=True,
        )
        if value is None:
            return
        self.settings.flyback_fraction = 0.0 if value is CLEARED else float(value)
        self._persist()

    def action_toggle_includes_flyback(self) -> None:
        self.settings.frame_includes_flyback = not self.settings.frame_includes_flyback
        self._persist()

    def action_run(self) -> None:
        if self.settings.signal is None:
            return
        try:
            acquisition = self.settings.acquisition()
        except AcquisitionError as exc:
            self.messages.append(f"  ✗ {exc}")
            return

        render.clear_screen(self.clear)
        print(f"  reading {self.settings.signal.path.name}…")
        try:
            result = characterise(
                self.settings.signal.path,
                self.settings.reference.path if self.settings.reference else None,
                acquisition=acquisition,
                page_policy=self.page_policy,
            )
        except (loading.LoadError, FrameError, ValueError) as exc:
            self.messages.append(f"  ✗ {exc}")
            return

        self.last_report = render_report(result)
        print(self.last_report)
        render.pause(self.read_key)

    def action_save_report(self) -> None:
        if self.last_report is None:
            return
        default = self.directory / "interference-report.txt"
        target = prompts.prompt_export_path(default, title="Save the report to")
        if target is None:
            return
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(self.last_report + "\n", encoding="utf-8")
        except OSError as exc:
            self.messages.append(f"  ✗ could not write {target}: {exc}")
            return
        self.messages.append(f"  ✓ report written to {target}")

    # -- loop --------------------------------------------------------------- #

    def dispatch(self, key: str) -> bool:
        """Returns False to leave the screen."""
        if key in ("b", "esc", "q"):
            return False
        actions = {
            "f": self.action_signal,
            "r": self.action_reference,
            "c": self.action_clear_reference,
            "l": self.action_line_rate,
            "n": self.action_resolution,
            "y": self.action_flyback,
            "w": self.action_toggle_includes_flyback,
            "g": self.action_run,
            "o": self.action_save_report,
        }
        action = actions.get(key)
        if action is not None:
            action()
        return True

    def run(self) -> None:
        while True:
            self.draw()
            try:
                key = self.read_key()
            except KeyboardInterrupt:
                return
            print(key if len(key) == 1 else "")
            if not self.dispatch(key):
                return
