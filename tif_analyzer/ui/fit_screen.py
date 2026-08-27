"""The fit screen: is this series shot-noise limited or detector-noise limited?

A screen of its own rather than an overlay on ``[v]``, for the same reason the
interference report has one: the answer is not a curve, it is an exponent with
an error bar and a verdict that may well be *indeterminate*, and none of that
fits in a plot legend.

The fit is recomputed every time the screen draws. It costs microseconds on a
handful of points, and it means there is no fourth piece of derived state to
keep in step with the run — the numbers on screen are always the numbers of the
report currently in ``AppState``.
"""

from __future__ import annotations

from pathlib import Path

from .. import metrics, noise_regime, plotting
from ..fitting import DEFAULT_K_SIGMA
from . import render


class FitScreen:
    """Owns its own key dispatch; returns to MAIN on [b] or Esc."""

    def __init__(self, state, read_key, clear: bool = True):
        self.state = state
        self.read_key = read_key
        self.clear = clear
        self.log_axes = True
        self.messages: list[str] = []
        self.param_label = "parameter"

    # -- the fit ------------------------------------------------------------ #

    @property
    def series_key(self) -> str:
        spec = metrics.get(self.state.metric_key)
        return self.state.selected_series or spec.default_series

    def compute(self) -> noise_regime.RegimeReport | None:
        if self.state.report is None or not self.state.report.results:
            return None
        return noise_regime.analyse(
            self.state.report, self.series_key, self.param_label
        )

    # -- screen ------------------------------------------------------------- #

    def draw(self, regime: noise_regime.RegimeReport | None) -> None:
        render.clear_screen(self.clear)
        print("\n  Fit — power law, y = a · x^b\n")
        if regime is None:
            print("  no results — run the analysis first\n")
        else:
            for line in self.report_lines(regime):
                print(line)
        print(self.menu())
        for message in self.messages:
            print(f"\n{message}")
        self.messages.clear()
        print("\n  > ", end="", flush=True)

    def report_lines(self, regime: noise_regime.RegimeReport) -> list[str]:
        lines: list[str] = []
        for panel in (regime.snr, regime.response, regime.noise):
            if panel is None:
                continue
            lines.append(f"  {panel.label}")
            if panel.error is not None:
                lines.append(f"    ✗ {panel.error}")
                lines.append("")
                continue
            fit = panel.fit
            assert fit is not None
            low, high = fit.interval(DEFAULT_K_SIGMA)
            lines.append(
                f"    b = {fit.exponent:+.3f} ± {fit.exponent_stderr:.3f}"
                f"   ({DEFAULT_K_SIGMA:g}·SE: [{low:+.2f}, {high:+.2f}])"
            )
            lines.append(
                f"    a = {fit.amplitude:.4g}   R² {fit.r_squared:.4f}"
                f"   residual rms {fit.residual_rms:.4f} (log10)"
            )
            lines.append(
                f"    {fit.n_points} point(s), lever arm {fit.decades:.2f} decades"
                + (f", {fit.n_dropped} dropped" if fit.n_dropped else "")
            )
            for exponent, model in sorted(fit.locked.items()):
                marker = "*" if model.residual_rms <= fit.residual_rms + 1e-12 else " "
                lines.append(
                    f"    {marker} b = {exponent:g} locked: residual rms "
                    f"{model.residual_rms:.4f}"
                )
            if panel.verdict is not None:
                lines.append(f"    → {panel.verdict.conclusion}: {panel.verdict.detail}")
            for warning in fit.warnings:
                lines.append(f"    ! {warning}")
            lines.append("")

        lines.append(f"  VERDICT: {regime.conclusion}")
        for reason in regime.reasons:
            lines.append(f"    · {reason}")
        lines.append("")
        lines.append(
            "  Provisional: the exponents are the physics, but the interval that "
            "decides between"
        )
        lines.append(
            "  them is a 3·SE rule chosen by principle, not calibrated against "
            "repeated series."
        )
        return lines

    def menu(self) -> str:
        axes = "log-log" if self.log_axes else "linear"
        return (
            f"\n  [l] axes: {axes}    [v] plot the fits    [o] save this report"
            "\n  [b] back to the main menu"
        )

    # -- actions ------------------------------------------------------------ #

    def action_toggle_axes(self) -> None:
        self.log_axes = not self.log_axes

    def action_plot(self, regime: noise_regime.RegimeReport | None) -> None:
        if regime is None:
            return
        panels = [panel for panel in regime.panels if panel.available]
        if not panels:
            self.messages.append("  ✗ nothing could be fitted")
            return
        snapshot = self.state.report.config_snapshot if self.state.report else {}
        directory = Path(snapshot.get("directory") or self.state.directory)
        title = directory.name or str(directory)
        subtitle = f"{self.series_key} · power-law fit · {regime.conclusion}"
        figure = plotting.build_fit_figure(
            panels, title, subtitle, log_axes=self.log_axes
        )
        outcome = plotting.show_or_save_figure(
            figure,
            None,
            plotting.writable_path(
                f"tif-analyzer_fit_{self.series_key}.png", directory
            ),
        )
        if outcome.saved_to is not None:
            self.messages.append(f"  ✓ figure saved to {outcome.saved_to}")

    def action_save(self, regime: noise_regime.RegimeReport | None) -> None:
        if regime is None:
            return
        from . import prompts

        snapshot = self.state.report.config_snapshot if self.state.report else {}
        directory = Path(snapshot.get("directory") or self.state.directory)
        default = directory / f"{self.series_key}_fit.txt"
        target = prompts.prompt_export_path(default, title="Save the fit report to")
        if target is None:
            return
        text = "\n".join(self.report_lines(regime))
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(text + "\n", encoding="utf-8")
        except OSError as exc:
            self.messages.append(f"  ✗ could not write {target}: {exc}")
            return
        self.messages.append(f"  ✓ report saved to {target}")

    # -- loop --------------------------------------------------------------- #

    def run(self) -> None:
        while True:
            regime = self.compute()
            self.draw(regime)
            try:
                key = self.read_key()
            except KeyboardInterrupt:
                return
            print(key if len(key) == 1 else "")
            if key in ("b", "esc", "q"):
                return
            if key == "l":
                self.action_toggle_axes()
            elif key == "v":
                self.action_plot(regime)
            elif key == "o":
                self.action_save(regime)
