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


#: Percentile clip used when an image is shown on screen.
DISPLAY_PERCENTILES = (1.0, 99.0)
#: A compact bright feature crushes the background to black under a linear
#: stretch, and the background is usually the part being judged.
DISPLAY_GAMMA = 0.4
STRETCHES = ("gamma", "linear")


def display_limits(image, percentiles: tuple[float, float] = DISPLAY_PERCENTILES):
    """Percentile clip for display. Never applied to the analysed data."""
    import numpy as np

    low, high = (float(v) for v in np.percentile(image, percentiles))
    if high <= low:
        low, high = float(np.min(image)), float(np.max(image))
    if high <= low:
        high = low + 1.0
    return low, high


def display_norm(image, stretch: str = "gamma"):
    """Colour normalisation for an image panel — display only.

    The array handed to any metric stays in raw counts; this only decides how
    those counts are painted.
    """
    from matplotlib.colors import Normalize, PowerNorm

    low, high = display_limits(image)
    if stretch == "gamma":
        return PowerNorm(gamma=DISPLAY_GAMMA, vmin=low, vmax=high)
    return Normalize(vmin=low, vmax=high)


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


# The fitted curve has to be distinguishable from the data without relying on
# colour alone: solid for the free fit, dashed for the models with the exponent
# held fixed.
FIT_COLOR = "#B4341C"
LOCKED_COLORS = ("#6B7280", "#9AA1AC")


def _curve_x(xs, log_axes: bool):
    """Where to sample a fitted curve: geometric spacing under log axes, so the
    line is smooth where the eye is looking."""
    import numpy as np

    low, high = float(min(xs)), float(max(xs))
    if log_axes and low > 0:
        return np.logspace(np.log10(low), np.log10(high), 200)
    return np.linspace(low, high, 200)


def _tidy_log_ticks(axis) -> None:
    """Plain numbers at 1, 2 and 5 per decade.

    A power-law series usually covers one or two decades, and matplotlib's
    default then labels every minor tick in scientific notation — ``3×10¹4×10¹``
    running into each other. Three labels a decade, written as ordinary numbers,
    is what a reader of this plot actually needs.
    """
    from matplotlib.ticker import FuncFormatter, LogLocator

    plain = FuncFormatter(lambda value, _: f"{value:g}")
    axis.set_major_locator(LogLocator(base=10.0))
    axis.set_minor_locator(LogLocator(base=10.0, subs=(2.0, 5.0)))
    axis.set_major_formatter(plain)
    axis.set_minor_formatter(plain)


def draw_fit_panel(ax, panel, log_axes: bool = True) -> None:
    """One fitted series: the points, the free fit, and the locked models.

    *panel* is a :class:`~tif_analyzer.noise_regime.Fitted` — anything with
    ``xs``, ``ys``, ``fit``, ``x_label`` and ``y_label``.
    """
    ax.plot(
        panel.xs,
        panel.ys,
        linestyle="none",
        marker="o",
        markersize=7,
        color=SERIES_COLOR,
        markerfacecolor=SERIES_COLOR,
        markeredgecolor="white",
        markeredgewidth=1.0,
        label="measured",
        zorder=4,
    )

    fit = panel.fit
    if fit is not None:
        curve_x = _curve_x(panel.xs, log_axes)
        ax.plot(
            curve_x,
            fit.predict(curve_x),
            color=FIT_COLOR,
            linewidth=2.0,
            zorder=3,
            label=f"fit: b = {fit.exponent:.2f} ± {fit.exponent_stderr:.2f}",
        )
        for index, (exponent, model) in enumerate(sorted(fit.locked.items())):
            ax.plot(
                curve_x,
                fit.predict_locked(curve_x, exponent),
                color=LOCKED_COLORS[index % len(LOCKED_COLORS)],
                linewidth=1.6,
                linestyle="--",
                zorder=2,
                label=f"b = {exponent:g} locked (rms {model.residual_rms:.3f})",
            )

    if log_axes:
        ax.set_xscale("log")
        ax.set_yscale("log")
        _tidy_log_ticks(ax.xaxis)
        _tidy_log_ticks(ax.yaxis)
    ax.set_xlabel(panel.x_label, color=TEXT_PRIMARY, fontsize=9)
    ax.set_ylabel(panel.y_label, color=TEXT_PRIMARY, fontsize=9)
    ax.set_title(panel.label, fontsize=10, color=TEXT_PRIMARY, loc="left")
    ax.grid(True, which="both", color=GRID_COLOR, linewidth=0.8, alpha=0.9, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(AXIS_COLOR)
    ax.tick_params(colors=TEXT_MUTED, labelsize=8)
    ax.legend(fontsize=7.5, frameon=False, labelcolor=TEXT_MUTED)


def build_fit_figure(panels, title: str, subtitle: str, log_axes: bool = True):
    """One panel per fitted series, side by side.

    Log axes by default: a power law is a straight line there, and the exponent
    is the slope the eye can read directly. On linear axes 0.5 and 1.0 both look
    like plausible curves through the same four points.
    """
    import matplotlib.pyplot as plt

    panels = [panel for panel in panels if panel.fit is not None]
    if not panels:
        raise ValueError("nothing to plot: no series could be fitted")

    width = 5.2 * len(panels)
    fig, axes = plt.subplots(1, len(panels), figsize=(width, 4.6), layout="constrained")
    if len(panels) == 1:
        axes = [axes]
    for ax, panel in zip(axes, panels):
        draw_fit_panel(ax, panel, log_axes)

    fig.suptitle(title, fontsize=13, color=TEXT_PRIMARY, x=0.01, ha="left")
    axes[0].annotate(
        subtitle,
        xy=(0.0, 1.10),
        xycoords="axes fraction",
        fontsize=9,
        color=TEXT_MUTED,
        annotation_clip=False,
    )
    return fig


def writable_path(name: str, directory: Path | str | None = None) -> Path:
    """Where a headless figure lands: next to the data if that is writable,
    then the CWD, then a temporary directory."""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    candidates = []
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


def default_save_path(snapshot: dict, series_key: str) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    metric_key = str(snapshot.get("metric", "metric"))
    series = "" if series_key == metric_key else f"_{series_key}"
    return writable_path(
        f"tif-analyzer_{metric_key}{series}_{stamp}.png", snapshot.get("directory")
    )


@dataclass(frozen=True)
class PlotOutcome:
    shown: bool
    saved_to: Path | None
    backend: BackendInfo


def show_or_save_figure(
    figure,
    save_to: Path | None = None,
    default_path: Path | None = None,
) -> PlotOutcome:
    """Show *figure* in a blocking window, or write it to a PNG when the backend
    cannot open one. Never silently does nothing — that is the failure mode §8
    exists to prevent."""
    import matplotlib.pyplot as plt

    info = backend_info()
    if save_to is None and info.interactive:
        try:
            plt.show()
            return PlotOutcome(shown=True, saved_to=None, backend=info)
        finally:
            plt.close(figure)

    path = Path(save_to) if save_to is not None else (default_path or Path("figure.png"))
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=150, facecolor="white")
    finally:
        plt.close(figure)
    return PlotOutcome(shown=False, saved_to=path, backend=info)


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
    key = series_key or metric.default_series
    figure = build_figure(report, metric, key, param_name, param_unit)
    return show_or_save_figure(
        figure, save_to, default_save_path(report.config_snapshot, key)
    )
