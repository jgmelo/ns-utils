"""Menu loop and key dispatch.

The state machine of §11::

    MAIN ─┬─ [d] → PROMPT_DIR       line input, tab completion, validate, persist
          ├─ [p] → PROMPT_PATTERN   validate: compiles + exactly one named group
          ├─ [g] → PROMPT_GLOB
          ├─ [m] → MENU_METRIC
          ├─ [r] → RUNNING → REPORT (only if failures) → MAIN
          ├─ [s] → MENU_SERIES      shown only when metric has >1 output
          ├─ [v] → PLOT             blocking window; Agg fallback
          ├─ [e] → PROMPT_EXPORT
          ├─ [!] → toggle duplicates
          └─ [q] → exit
"""

from __future__ import annotations

import time

from .. import analysis, config as config_mod, discovery, export, metrics, plotting
from ..cache import ResultCache
from ..discovery import DiscoveryError, DuplicateParameterError
from ..state import AppState
from . import prompts, render
from .keys import KeyReader


class MenuApp:
    def __init__(
        self,
        state: AppState,
        config: config_mod.Config | None = None,
        reader: KeyReader | None = None,
        persist: bool = True,
        clear: bool = True,
    ):
        self.state = state
        self.config = config or config_mod.Config()
        self.reader = reader or KeyReader()
        self.persist = persist
        self.clear = clear
        self.cache = ResultCache()
        self.backend = plotting.backend_info()
        self.notices: list[str] = []
        self.messages: list[str] = []
        self.discovery_error: DiscoveryError | None = None
        self.running = False

    # -- persistence -------------------------------------------------------- #

    def _persist(self) -> None:
        """Written only after a directory has been validated, so a typo never
        becomes the next session's default."""
        if not self.persist:
            return
        self.config.last_directory = str(self.state.directory)
        self.config.last_pattern = self.state.pattern
        self.config.last_metric_key = self.state.metric_key
        self.config.last_glob = self.state.glob
        self.config.allow_duplicates = self.state.allow_duplicates
        if config_mod.save(self.config) is None:
            self.notices.append(f"could not write config to {config_mod.config_path()}")

    # -- discovery ---------------------------------------------------------- #

    def ensure_discovery(self) -> None:
        """Refresh filename discovery when state invalidated it. No image I/O,
        so this is cheap enough to do on every return to MAIN."""
        if self.state.records is not None and self.discovery_error is None:
            return
        self.discovery_error = None
        try:
            report = discovery.discover(
                self.state.directory,
                self.state.glob,
                self.state.pattern,
                self.state.allow_duplicates,
            )
        except DuplicateParameterError as exc:
            self.discovery_error = exc
            self.state.update(discovery=exc.report, records=None)
        except DiscoveryError as exc:
            self.discovery_error = exc
            self.state.update(discovery=None, records=None)
        else:
            self.state.update(discovery=report, records=report.records)

    def blockers(self) -> dict[str, str | None]:
        """Reasons a menu option is unavailable — shown, not hidden."""
        blocked: dict[str, str | None] = {}
        if isinstance(self.discovery_error, DuplicateParameterError):
            blocked["r"] = "duplicate parameter values — see below"
        elif self.discovery_error is not None:
            blocked["r"] = str(self.discovery_error)
        elif not self.state.has_records:
            blocked["r"] = "no matching files"
        if not self.state.has_results:
            blocked["v"] = "no results — run the analysis first"
            blocked["e"] = "no results — run the analysis first"
        elif not self.backend.interactive:
            blocked["v"] = f"{self.backend.name} backend — saves a PNG"
        return blocked

    # -- screen ------------------------------------------------------------- #

    def draw(self) -> None:
        render.clear_screen(self.clear)
        backend_note = (
            f"{self.backend.name} (interactive window)"
            if self.backend.interactive
            else f"{self.backend.name} — headless, plots are saved as PNG"
        )
        print(render.header(self.state, backend_note, self.notices))
        print(render.menu(self.state, self.blockers()))
        if isinstance(self.discovery_error, DuplicateParameterError):
            print(render.duplicate_report(self.discovery_error.duplicates))
        elif self.discovery_error is not None:
            print(f"\n  ✗ {self.discovery_error}")
        elif self.state.discovery is not None and (
            self.state.discovery.unmatched or self.state.discovery.parse_failures
        ):
            skipped = len(self.state.discovery.unmatched) + len(
                self.state.discovery.parse_failures
            )
            print(f"\n  {skipped} file(s) skipped — press [r] for the full breakdown")
        for message in self.messages:
            print(f"\n{message}")
        self.messages.clear()
        print(f"\n  input: {self.reader.hint}")
        print("  > ", end="", flush=True)

    # -- actions ------------------------------------------------------------ #

    def action_directory(self) -> None:
        chosen = prompts.prompt_directory(self.state.directory)
        if chosen is None:
            return
        self.state.update(directory=chosen)
        self.discovery_error = None
        self._persist()

    def action_pattern(self) -> None:
        chosen = prompts.prompt_pattern(self.state.pattern)
        if chosen is None:
            return
        self.state.update(pattern=chosen)
        self.discovery_error = None
        self._persist()

    def action_glob(self) -> None:
        chosen = prompts.prompt_glob(self.state.glob)
        if chosen is None:
            return
        self.state.update(glob=chosen)
        self.discovery_error = None
        self._persist()

    def action_metric(self) -> None:
        chosen = prompts.choose_from_menu(
            "Metric", prompts.metric_entries(), self.state.metric_key, self.reader.read
        )
        if chosen is None:
            return
        self.state.update(metric_key=chosen)
        self.discovery_error = None
        self._persist()

    def action_series(self) -> None:
        spec = metrics.get(self.state.metric_key)
        if len(spec.outputs) < 2:
            return
        chosen = prompts.choose_from_menu(
            "Series to plot",
            prompts.series_entries(spec),
            self.state.selected_series or spec.default_series,
            self.reader.read,
        )
        if chosen is None:
            return
        # Rendering-only: never invalidates the results.
        self.state.update(selected_series=chosen)

    def action_toggle_duplicates(self) -> None:
        self.state.update(allow_duplicates=not self.state.allow_duplicates)
        self.discovery_error = None
        self._persist()

    def action_run(self) -> None:
        self.ensure_discovery()
        if self.discovery_error is not None or not self.state.has_records:
            return
        assert self.state.discovery is not None

        render.clear_screen(self.clear)
        print(render.discovery_summary(self.state.discovery))
        print()

        spec = metrics.get(self.state.metric_key)
        snapshot = analysis.make_snapshot(
            directory=self.state.directory,
            glob=self.state.glob,
            pattern=self.state.pattern,
            metric_key=spec.key,
            steps=self.state.preprocess_steps,
            page_policy=self.state.page_policy,
            allow_duplicates=self.state.allow_duplicates,
            n_files=len(self.state.records or []),
        )
        generator = analysis.run(
            self.state.records or [],
            spec,
            steps=self.state.preprocess_steps,
            cache=self.cache,
            page_policy=self.state.page_policy,
            snapshot=snapshot,
        )
        started = time.perf_counter()
        report: analysis.RunReport
        while True:
            try:
                event = next(generator)
            except StopIteration as stop:
                report = stop.value
                break
            render.print_progress(event)
        print()
        self.state.update(
            report=report, selected_series=self.state.selected_series or spec.default_series
        )

        print(render.run_summary(report, time.perf_counter() - started))
        if report.failures:
            print(render.failure_report(report))
        render.pause(self.reader.read)

    def action_plot(self) -> None:
        if not self.state.has_results:
            return
        assert self.state.report is not None
        spec = metrics.get(self.state.metric_key)
        param_name = self.state.records[0].param_name if self.state.records else "parameter"
        unit = None
        try:
            unit = discovery.infer_unit(self.state.pattern)
        except DiscoveryError:  # pragma: no cover - pattern already validated
            unit = None

        if not self.backend.interactive:
            print(f"\n  {self.backend.explanation}")
        outcome = plotting.show_or_save(
            self.state.report,
            spec,
            self.state.selected_series,
            param_name=param_name,
            param_unit=unit,
        )
        if outcome.saved_to is not None:
            print(f"  ✓ plot saved to {outcome.saved_to}")
            render.pause(self.reader.read)

    def action_export(self) -> None:
        if not self.state.has_results:
            return
        assert self.state.report is not None
        default = self.state.directory / f"{self.state.directory.name}_{self.state.metric_key}.csv"
        target = prompts.prompt_export_path(default)
        if target is None:
            return
        try:
            written = export.write_csv(self.state.report, target)
        except OSError as exc:
            self.messages.append(f"  ✗ export failed: {exc}")
            return
        self.messages.append(render.export_hint(written))

    # -- loop --------------------------------------------------------------- #

    def dispatch(self, key: str) -> bool:
        """Handle one key. Returns ``False`` to quit."""
        if key == "q":
            return False
        actions = {
            "d": self.action_directory,
            "p": self.action_pattern,
            "g": self.action_glob,
            "m": self.action_metric,
            "s": self.action_series,
            "r": self.action_run,
            "v": self.action_plot,
            "e": self.action_export,
            "!": self.action_toggle_duplicates,
        }
        action = actions.get(key)
        if action is not None:
            action()
        return True

    def loop(self) -> int:
        self.running = True
        try:
            while self.running:
                self.ensure_discovery()
                self.draw()
                try:
                    key = self.reader.read()
                except KeyboardInterrupt:
                    print("\n  bye")
                    return 0
                print(key if len(key) == 1 else "")
                if not self.dispatch(key):
                    break
        finally:
            self.running = False
        print("\n  bye")
        return 0


def run(state: AppState, config: config_mod.Config | None = None, **kwargs) -> int:
    return MenuApp(state, config, **kwargs).loop()
