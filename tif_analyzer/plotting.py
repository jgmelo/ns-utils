"""Figure construction and display.

Interactive by design (locked decision 6): ``plt.show()`` blocks and the menu
loop resumes when the window is closed. When matplotlib resolves to a non-GUI
backend — ``Agg``, common over SSH without X forwarding and in bare containers —
the plot is written to a PNG and the path is reported instead. It must never
silently do nothing.
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import matplotlib

from .analysis import RunReport
from .metrics import MetricSpec

#: Backends that render to a file and cannot open a window.
NON_GUI_BACKENDS = frozenset(
    {"agg", "pdf", "ps", "svg", "template", "pgf", "cairo", "module://matplotlib_inline.backend_inline"}
)

# One series, one hue: recessive grid and axes, ink-coloured text, the data the
# only saturated thing on the surface.
SERIES_COLOR = "#2E6FD9"
GRID_COLOR = "#D8DCE3"
AXIS_COLOR = "#8A9099"
TEXT_PRIMARY = "#1B1F24"
TEXT_MUTED = "#5A6169"


@dataclass(frozen=True)
class BackendInfo:
    name: str
    interactive: bool

    @property
    def explanation(self) -> str:
        if self.interactive:
            return f"matplotlib backend: {self.name} (interactive window)"
        return (
            f"matplotlib backend: {self.name} — no GUI available "
            "(headless session or no display), so plots are saved as PNG instead"
        )


def backend_info() -> BackendInfo:
    """Resolve the backend once, at startup, so the UI can label the plot
    option honestly before the user asks for a window that cannot open."""
    name = matplotlib.get_backend()
    return BackendInfo(name=name, interactive=name.lower() not in NON_GUI_BACKENDS)


def axis_label(name: str, unit: str | None) -> str:
    return f"{name} ({unit})" if unit else name


def _title(snapshot: dict, metric: MetricSpec, n_points: int) -> tuple[str, str]:
    directory = snapshot.get("directory") or ""
    dir_name = Path(directory).name or directory or "?"
    stamp = snapshot.get("timestamp") or datetime.now().astimezone().isoformat(timespec="seconds")
    subtitle = f"{metric.key} · {n_points} files · {stamp}"
    return dir_name, subtitle


def build_figure(
    report: RunReport,
    metric: MetricSpec,
    series_key: str | None = None,
    param_name: str = "parameter",
    param_unit: str | None = None,
):
    """Build the metric-vs-parameter figure. Points are sorted by parameter
    value; markers *and* a connecting line, because small datasets read badly as
    bare points."""
    import matplotlib.pyplot as plt

    if not report.results:
        raise ValueError("nothing to plot: the run produced no results")

    key = series_key or metric.default_series
    output = metric.output(key)
    xs, ys = report.series(key)

    fig, ax = plt.subplots(figsize=(8.0, 5.0), layout="constrained")
    ax.plot(
        xs,
        ys,
        marker="o",
        markersize=7,
        linewidth=2.0,
        color=SERIES_COLOR,
        markerfacecolor=SERIES_COLOR,
        markeredgecolor="white",
        markeredgewidth=1.0,
        zorder=3,
    )

    ax.set_xlabel(axis_label(param_name, param_unit), color=TEXT_PRIMARY)
    ax.set_ylabel(axis_label(output.label, output.unit), color=TEXT_PRIMARY)

    title, subtitle = _title(report.config_snapshot, metric, len(xs))
    areas = report.config_snapshot.get("regions")
    pad = 30 if areas else 18
    ax.set_title(title, fontsize=13, color=TEXT_PRIMARY, loc="left", pad=pad)
    ax.annotate(
        subtitle,
        xy=(0.0, 1.02),
        xycoords="axes fraction",
        fontsize=9,
        color=TEXT_MUTED,
        annotation_clip=False,
    )
    if areas:
        # A saved figure has to say which areas produced these numbers.
        source = report.config_snapshot.get("regions_source")
        text = f"areas: {areas}" + (f"  ·  drawn on {source}" if source else "")
        ax.annotate(
            text,
            xy=(0.0, 1.06),
            xycoords="axes fraction",
            fontsize=8,
            color=TEXT_MUTED,
            annotation_clip=False,
        )

    ax.grid(True, color=GRID_COLOR, linewidth=0.8, alpha=0.9, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS_COLOR)
    ax.tick_params(colors=TEXT_MUTED, labelsize=9)
    return fig


def default_save_path(snapshot: dict, series_key: str) -> Path:
    """Where a headless plot lands: next to the data if that is writable, then
    the CWD, then a temporary directory."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    metric_key = str(snapshot.get("metric", "metric"))
    series = "" if series_key == metric_key else f"_{series_key}"
    name = f"tif-analyzer_{metric_key}{series}_{stamp}.png"
    candidates = []
    directory = snapshot.get("directory")
    if directory:
        candidates.append(Path(directory))
    candidates.extend([Path.cwd(), Path(tempfile.gettempdir())])
    for candidate in candidates:
        try:
            if candidate.is_dir():
                probe = candidate / f".tif-analyzer-write-test-{stamp}"
                probe.touch()
                probe.unlink()
                return candidate / name
        except OSError:
            continue
    return Path(tempfile.gettempdir()) / name


@dataclass(frozen=True)
class PlotOutcome:
    shown: bool
    saved_to: Path | None
    backend: BackendInfo


def show_or_save(
    report: RunReport,
    metric: MetricSpec,
    series_key: str | None = None,
    param_name: str = "parameter",
    param_unit: str | None = None,
    save_to: Path | None = None,
) -> PlotOutcome:
    """Show the figure in a blocking window, or save it when the backend cannot
    open one (or when *save_to* is given)."""
    import matplotlib.pyplot as plt

    info = backend_info()
    key = series_key or metric.default_series
    fig = build_figure(report, metric, key, param_name, param_unit)

    if save_to is None and info.interactive:
        try:
            plt.show()
            return PlotOutcome(shown=True, saved_to=None, backend=info)
        finally:
            plt.close(fig)

    path = Path(save_to) if save_to is not None else default_save_path(report.config_snapshot, key)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, facecolor="white")
    finally:
        plt.close(fig)
    return PlotOutcome(shown=False, saved_to=path, backend=info)
