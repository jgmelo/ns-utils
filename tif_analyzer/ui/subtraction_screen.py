"""The subtraction screen: compute the correction, look at it, then write it.

Reached from the interference screen with ``[s]``. It assumes the caller has
already decided that subtraction is worth doing — it never re-litigates
viability — but it always reports what the correction actually did, because
"89.9% of the stripe removed, background noise up 80%" is a trade the user has
to see before keeping the file.

Nothing is written until ``[w]``.
"""

from __future__ import annotations

from pathlib import Path

from .. import loading, plotting
from ..interference import analyse_channel, crosschannel
from ..interference.figures import build_subtraction_figure
from ..interference.frames import FrameError, frame_from_array, load_frame
from ..interference.subtract import (
    MODES,
    SubtractionError,
    SubtractionPlan,
    alignment_shift,
    subtract,
)
from ..interference.writing import default_output_path, write_corrected
from . import prompts, render


class SubtractionScreen:
    """Its own key dispatch; returns to the interference screen on [b]."""

    def __init__(self, settings, directory: Path, read_key, page_policy: str = "error",
                 clear: bool = True):
        self.settings = settings
        self.directory = directory
        self.read_key = read_key
        self.page_policy = page_policy
        self.clear = clear

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
        if self.settings.signal is None:
            return
        if self.mode == "reference" and self.settings.reference is None:
            return
        params = self.settings.params()
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

        flyback = self.settings.acquisition().flyback_columns(signal_frame.n_columns)
        before = analyse_channel(signal_frame, flyback, params)
        frequency = before.spectrum.centroid_frequency if before.spectrum.detected else 0.0

        band = None
        shift = 0.0
        if reference_frame is not None:
            if reference_frame.shape != signal_frame.shape:
                self.messages.append(
                    f"  ✗ reference is {reference_frame.n_columns}x{reference_frame.n_lines} px "
                    f"but the signal is {signal_frame.n_columns}x{signal_frame.n_lines} px"
                )
                return
            reference_analysis = analyse_channel(reference_frame, flyback, params)
            if before.spectrum.detected and reference_analysis.spectrum.detected:
                cross = crosschannel.compare(
                    before.profile.analysis_profile,
                    reference_analysis.profile.analysis_profile,
                    before.spectrum,
                    reference_analysis.spectrum,
                )
                band = (
                    min(before.spectrum.cluster_low_bin, reference_analysis.spectrum.cluster_low_bin),
                    max(before.spectrum.cluster_high_bin, reference_analysis.spectrum.cluster_high_bin),
                )
                shift = alignment_shift(cross.phase_offset_px)
            else:
                self.messages.append(
                    "  ! no dominant component in one of the frames — subtracting unaligned, "
                    "with no band"
                )

        plan = SubtractionPlan(
            mode=self.mode,
            band_limited=self.band_limited and band is not None,
            per_line=self.per_line,
            shift_px=shift,
        )
        try:
            result = subtract(
                signal_frame.data,
                plan,
                reference=reference_frame.data if reference_frame else None,
                mask_result=before.mask,
                band=band,
                frequency=frequency,
                params=params,
            )
        except SubtractionError as exc:
            self.messages.append(f"  ✗ {exc}")
            return

        self.result = result
        self.before = before
        self.after = analyse_channel(
            frame_from_array(result.corrected, dtype="float32", path=self.settings.signal.path),
            flyback,
            params,
        )
        self.written = None
        for warning in result.warnings:
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
