"""The subtraction screen: compute the correction, look at it, then write it.

Reached from the interference screen with ``[s]``. It assumes the caller has
already decided that subtraction is worth doing — it never re-litigates
viability — but it always reports what the correction actually did, because
"89.9% of the stripe removed, background noise up 80%" is a trade the user has
to see before keeping the file.

Nothing is written until ``[w]``.
"""

from __future__ import annotations

import shutil
from pathlib import Path

from .. import acquisition_sidecar, loading, plotting, roi_sidecar
from ..interference import batch as batch_mod
from ..interference.figures import build_subtraction_figure
from ..interference.frames import FrameError, load_frame
from ..interference.subtract import MODES, SubtractionError
from ..interference.writing import default_output_path, write_corrected
from . import prompts, render


class SubtractionScreen:
    """Its own key dispatch; returns to the interference screen on [b]."""

    def __init__(self, settings, directory: Path, read_key, page_policy: str = "error",
                 clear: bool = True, records: list | None = None):
        self.settings = settings
        self.directory = directory
        self.read_key = read_key
        self.page_policy = page_policy
        self.clear = clear
        #: The files ``[a]`` would correct: whatever the current glob and regex
        #: found, which is exactly the series that will later be plotted.
        self.records = list(records or [])

        self.mode = "reference" if settings.reference is not None else "profile"
        self.band_limited = True
        self.per_line = False

        self.messages: list[str] = []
        self.result = None
        self.before = None
        self.after = None
        self.written: Path | None = None

    # -- state --------------------------------------------------------------- #

    def _invalidate(self) -> None:
        """Any change to the plan drops the computed result.

        Same rule as the series pipeline: a stale number next to a changed
        setting is how someone ends up writing a file that does not match what
        the screen says.
        """
        self.result = None
        self.before = None
        self.after = None
        self.written = None

    def blockers(self) -> dict[str, str]:
        blocked: dict[str, str] = {}
        if self.settings.signal is None:
            blocked["g"] = "no signal frame"
        elif self.mode == "reference" and self.settings.reference is None:
            blocked["g"] = "reference mode needs a reference frame — set one, or switch to profile"
        if self.result is None:
            blocked["v"] = "compute it first"
            blocked["w"] = "compute it first"
        if self.mode == "profile":
            blocked["n"] = "profile mode uses no reference"
            blocked["e"] = "profile mode has nothing to scale"
        if not self.records:
            blocked["a"] = "no matching files"
        elif self.mode == "reference" and self.settings.reference is None:
            blocked["a"] = "reference mode needs a reference frame"
        return blocked

    # -- rendering ----------------------------------------------------------- #

    def header(self) -> str:
        settings = self.settings
        lines = ["\n  Subtraction — remove the fixed pattern", ""]
        lines.append(
            f"  signal:      {settings.signal.path.name if settings.signal else '(none)'}"
        )
        lines.append(
            f"  reference:   {settings.reference.path.name if settings.reference else '(none)'}"
        )
        lines.append(
            "  mode:        "
            + (
                "reference — signal minus a scaled, aligned reference"
                if self.mode == "reference"
                else "profile — signal minus its own column median (no reference)"
            )
        )
        if self.mode == "reference":
            lines.append(
                f"  reference:   {'band-limited to the stripe' if self.band_limited else 'RAW (brings its own noise)'}"
                f"   ·   fit {'per line' if self.per_line else 'global'}"
            )
        lines.append(f"  bead cut:    median + {settings.mad_k:g}σ")

        lines.append("")
        if self.result is None:
            lines.append("  computed:    — press [g]")
            return "\n".join(lines)

        result = self.result
        lines.append("  computed:")
        if result.frequency > 0:
            detail = f"period {1.0 / result.frequency:.3f} px"
        else:
            detail = "no dominant component (amplitudes measured as line std)"
        if result.mode == "reference":
            detail += f"   ·   α {result.scale:.4g}   ·   shift {result.shift_px:+.3f} px"
        lines.append(f"    {detail}")
        lines.append(
            f"    stripe   {result.stripe_before:8.4g} → {result.stripe_after:8.4g} counts"
            f"   ({result.removed_fraction:.1%} removed)"
        )
        lines.append(
            f"    noise    {result.sigma_before:8.4g} → {result.sigma_after:8.4g} counts"
            f"   ({result.noise_change:+.1%})"
        )
        if self.written:
            lines.append(f"    written to {self.written.name}")
        return "\n".join(lines)

    def menu(self) -> str:
        blocked = self.blockers()

        def entry(key: str, label: str) -> str:
            reason = blocked.get(key)
            text = f"[{key}] {label}"
            return f"{text}  ({reason})" if reason else text

        rows = [
            (entry("m", f"mode: {self.mode}"),
             entry("n", f"band-limited: {'ON' if self.band_limited else 'OFF'}")),
            (entry("e", f"fit per line: {'ON' if self.per_line else 'OFF'}"),
             entry("g", "compute")),
            (entry("v", "view before/after"), entry("w", "write the corrected frame")),
            (entry("a", f"apply to all {len(self.records)} matching file(s)"), ""),
            ("[b] back", ""),
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

    # -- actions ------------------------------------------------------------- #

    def action_mode(self) -> None:
        index = (MODES.index(self.mode) + 1) % len(MODES)
        self.mode = MODES[index]
        self._invalidate()

    def action_band(self) -> None:
        if self.mode != "reference":
            return
        self.band_limited = not self.band_limited
        self._invalidate()

    def action_per_line(self) -> None:
        if self.mode != "reference":
            return
        self.per_line = not self.per_line
        self._invalidate()

    def action_compute(self) -> None:
        """One frame, through the same function the batch uses.

        The screen is meant to be a preview of what ``[a]`` will do to the whole
        series; the only way to keep that true is for both to call the same
        code.
        """
        if self.settings.signal is None:
            return
        if self.mode == "reference" and self.settings.reference is None:
            return
        try:
            signal_frame = load_frame(self.settings.signal.path, self.page_policy)
            reference_frame = (
                load_frame(self.settings.reference.path, self.page_policy)
                if self.mode == "reference"
                else None
            )
        except (loading.LoadError, FrameError) as exc:
            self.messages.append(f"  ✗ {exc}")
            return

        try:
            correction = batch_mod.correct_frame(
                signal_frame,
                mode=self.mode,
                band_limited=self.band_limited,
                per_line=self.per_line,
                params=self.settings.params(),
                acquisition=self.settings.acquisition(),
                reference_frame=reference_frame,
                analyse_after=True,
            )
        except SubtractionError as exc:
            self.messages.append(f"  ✗ {exc}")
            return

        self.result = correction.result
        self.before = correction.before
        self.after = correction.after
        self.written = None
        for note in correction.notes:
            self.messages.append(f"  ! {note}")
        for warning in correction.result.warnings:
            self.messages.append(f"  ! {warning}")

    def action_view(self) -> None:
        if self.result is None:
            return
        figure = build_subtraction_figure(
            self.before, self.after, self.result, self.settings.params()
        )
        name = f"subtraction_{self.settings.signal.path.stem}.png"
        outcome = plotting.show_or_save_figure(
            figure, default_path=plotting.writable_path(name, self.directory)
        )
        if outcome.saved_to is not None:
            self.messages.append(
                f"  ✓ figure saved to {outcome.saved_to}  ({outcome.backend.name} backend)"
            )

    def action_write(self) -> None:
        if self.result is None or self.settings.signal is None:
            return
        default = default_output_path(self.settings.signal.path, self.result.mode)
        target = prompts.prompt_export_path(default, title="Write the corrected frame to")
        if target is None:
            return
        try:
            written, sidecar = write_corrected(
                target,
                self.result,
                signal_path=self.settings.signal.path,
                reference_path=(
                    self.settings.reference.path if self.settings.reference else None
                ),
            )
        except OSError as exc:
            self.messages.append(f"  ✗ could not write {target}: {exc}")
            return
        self.written = written
        self.messages.append(f"  ✓ float32 frame written to {written}")
        if sidecar is not None:
            self.messages.append(f"    provenance in {sidecar.name}")

    # -- the whole series ---------------------------------------------------- #

    def action_apply_all(self) -> None:
        """Correct every matching file into a directory of its own.

        The corrected frames keep their names, so the glob and the regex that
        found the originals find them too: point the analyzer at the new
        directory and the series is ready to run.
        """
        if not self.records:
            return
        if self.mode == "reference" and self.settings.reference is None:
            return

        paths = [record.path for record in self.records]
        default = batch_mod.default_output_dir(self.directory, self.mode)
        print(f"\n  {len(paths)} file(s) will be corrected in {self.mode} mode.")
        if self.mode == "reference":
            print(
                "  ! the same reference frame is used for every one of them — "
                "sound only if the pattern is locked, not drifting between acquisitions."
            )
        target = prompts.prompt_export_path(default, title="Write the corrected frames into")
        if target is None:
            return

        generator = batch_mod.run_batch(
            paths,
            target,
            mode=self.mode,
            band_limited=self.band_limited,
            per_line=self.per_line,
            params=self.settings.params(),
            acquisition=self.settings.acquisition(),
            reference_path=(
                self.settings.reference.path
                if self.mode == "reference" and self.settings.reference
                else None
            ),
            page_policy=self.page_policy,
        )
        print()
        report: batch_mod.BatchReport
        try:
            while True:
                try:
                    progress = next(generator)
                except StopIteration as stop:
                    report = stop.value
                    break
                print(f"  {self._progress_line(progress)}", flush=True)
        except batch_mod.BatchError as exc:
            self.messages.append(f"  ✗ {exc}")
            return

        carried = self._carry_sidecars(report.output_dir)

        print()
        for line in self._batch_summary(report):
            print(f"  {line}")
        for name in carried:
            print(f"  {name} copied across — the areas still fit, the frames kept their size")
        render.pause(self.read_key)

    def _carry_sidecars(self, output_dir: Path) -> list[str]:
        """Take the areas and the acquisition along to the corrected directory.

        Without this, pointing ``[d]`` at the output means redrawing the areas:
        they are pixel coordinates, they follow the directory, and the corrected
        frames have exactly the geometry they were drawn on.
        """
        carried = []
        for name in (roi_sidecar.SIDECAR_NAME, acquisition_sidecar.SIDECAR_NAME):
            source = self.directory / name
            if not source.is_file():
                continue
            try:
                shutil.copy2(source, output_dir / name)
            except OSError as exc:
                self.messages.append(f"  ! could not copy {name}: {exc}")
                continue
            carried.append(name)
        return carried

    @staticmethod
    def _progress_line(progress: batch_mod.BatchProgress) -> str:
        outcome = progress.outcome
        counter = f"[{progress.index}/{progress.total}]"
        name = outcome.source.name
        if outcome.skipped is not None:
            return f"{counter} {name}: skipped — {outcome.skipped}"
        if outcome.error is not None:
            return f"{counter} ✗ {name}: {type(outcome.error).__name__}: {outcome.error}"
        result = outcome.result
        assert result is not None
        return (
            f"{counter} ✓ {name}: {result.removed_fraction:6.1%} of the stripe removed, "
            f"noise {result.noise_change:+.1%}"
        )

    @staticmethod
    def _batch_summary(report: batch_mod.BatchReport) -> list[str]:
        lines = [
            f"✓ {len(report.written)} frame(s) written to {report.output_dir}",
        ]
        if report.summary_path is not None:
            lines.append(f"  summary table: {report.summary_path.name}")
        if report.skipped:
            lines.append(f"  {len(report.skipped)} skipped")
        if report.failures:
            lines.append(f"  ✗ {len(report.failures)} failed:")
            lines.extend(
                f"      {outcome.source.name}: {outcome.error}" for outcome in report.failures
            )
        spread = report.removed_range()
        if spread is not None:
            low, high = spread
            lines.append(f"  stripe removed: {low:.1%} … {high:.1%} across the series")
            if high - low > 0.15:
                lines.append(
                    "  ! the frames were not corrected alike — a spread that wide puts a "
                    "trend into the series that the sample does not have. Check the outliers "
                    "in the summary table before plotting."
                )
        lines.append("  point [d] at that directory and run the series as usual.")
        return lines

    # -- loop ---------------------------------------------------------------- #

    def dispatch(self, key: str) -> bool:
        if key in ("b", "esc", "q"):
            return False
        actions = {
            "m": self.action_mode,
            "n": self.action_band,
            "e": self.action_per_line,
            "g": self.action_compute,
            "v": self.action_view,
            "w": self.action_write,
            "a": self.action_apply_all,
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
