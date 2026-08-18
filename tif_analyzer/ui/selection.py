"""Interactive selection of the signal and noise areas.

One reference image, two drags, one blocking window — the same posture as the
plot (locked decision 6). The window is only ever opened on a GUI backend; the
caller is expected to check :func:`can_select` first and disable the option with
a visible reason otherwise.

The image is displayed with a percentile stretch so a 16-bit frame is actually
visible on screen. That stretch is **display only**: the array handed to the
metric stays in raw counts, as §6 requires.
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib
import numpy as np

from ..plotting import TEXT_MUTED, backend_info
from ..regions import (
    NOISE,
    SIGNAL,
    Region,
    RegionError,
    RegionSet,
    region_from_extents,
)

#: Blue / orange — distinguishable under the common colour-vision deficiencies,
#: and each box is labelled anyway, so identity never rests on colour alone.
ROLE_COLORS = {SIGNAL: "#2E6FD9", NOISE: "#D97706"}
ROLE_ORDER = (SIGNAL, NOISE)
ROLE_HELP = {
    SIGNAL: "area with the signal of interest",
    NOISE: "background area, used for the noise statistics",
}

SHAPES = ("rect", "ellipse")
DISPLAY_PERCENTILES = (1.0, 99.0)
#: A compact bright feature crushes the background to black under a linear
#: stretch, which is exactly the part you need to see to place a noise area.
#: The gamma stretch lifts it; it is a display curve and nothing else.
DISPLAY_GAMMA = 0.4
STRETCHES = ("gamma", "linear")


def can_select() -> tuple[bool, str]:
    """``(possible, reason)`` — selection needs a window."""
    info = backend_info()
    if info.interactive:
        return True, ""
    return False, f"{info.name} backend — area selection needs a GUI"


def display_limits(image: np.ndarray) -> tuple[float, float]:
    """Percentile clip for display. Never applied to the analysed data."""
    low, high = (float(v) for v in np.percentile(image, DISPLAY_PERCENTILES))
    if high <= low:
        low, high = float(np.min(image)), float(np.max(image))
    if high <= low:
        high = low + 1.0
    return low, high


def display_norm(image: np.ndarray, stretch: str = "gamma"):
    """Colour normalisation for the reference image — display only.

    The array handed to the metric stays in raw counts (§6); this only decides
    how those counts are painted on screen.
    """
    from matplotlib.colors import Normalize, PowerNorm

    low, high = display_limits(image)
    if stretch == "gamma":
        return PowerNorm(gamma=DISPLAY_GAMMA, vmin=low, vmax=high)
    return Normalize(vmin=low, vmax=high)


@dataclass
class SelectionSession:
    """The role-by-role state machine, kept free of matplotlib.

    Split out so the flow is testable without opening a window: the widget
    callbacks are thin wrappers around :meth:`confirm`, :meth:`undo` and
    :meth:`set_shape`.
    """

    shape: tuple[int, int]
    source_file: str | None = None
    kind: str = "rect"
    stretch: str = "gamma"
    index: int = 0
    pending: Region | None = None
    error: str | None = None
    cancelled: bool = False

    def __post_init__(self) -> None:
        self.confirmed: dict[str, Region] = {}

    @property
    def done(self) -> bool:
        return self.index >= len(ROLE_ORDER)

    @property
    def role(self) -> str | None:
        return None if self.done else ROLE_ORDER[self.index]

    def set_shape(self, kind: str) -> None:
        if kind not in SHAPES:
            raise ValueError(f"unknown shape {kind!r}")
        self.kind = kind
        self.pending = None

    def draw(self, extents: tuple[float, float, float, float]) -> None:
        """Record a drag, without accepting it yet."""
        if self.done:
            return
        try:
            self.pending = region_from_extents(self.kind, extents, self.shape)
            self.error = None
        except RegionError as exc:
            self.pending = None
            self.error = str(exc)

    def confirm(self) -> bool:
        """Accept the pending region for the current role. ``True`` if it moved on."""
        role = self.role
        if role is None:
            return False
        if self.pending is None:
            self.error = f"drag a box for the {role} area first"
            return False
        try:
            self.pending.validate_for(self.shape)
            if self.pending.npixels(self.shape) < 2:
                raise RegionError(f"{role} area is too small to measure — make it bigger")
        except RegionError as exc:
            self.error = str(exc)
            return False
        self.confirmed[role] = self.pending
        self.pending = None
        self.error = None
        self.index += 1
        return True

    def undo(self) -> bool:
        """Step back to the previous role. ``True`` if anything was undone."""
        if self.pending is not None:
            self.pending = None
            self.error = None
            return True
        if self.index == 0:
            return False
        self.index -= 1
        self.confirmed.pop(ROLE_ORDER[self.index], None)
        self.error = None
        return True

    def cancel(self) -> None:
        self.cancelled = True

    def result(self) -> RegionSet | None:
        if self.cancelled or not self.done:
            return None
        return RegionSet(
            signal=self.confirmed[SIGNAL],
            noise=self.confirmed[NOISE],
            image_shape=self.shape,
            source_file=self.source_file,
        )

    def instructions(self) -> str:
        role = self.role
        if role is None:
            return "done"
        state = f"pending {self.pending.describe()}" if self.pending else "drag to draw"
        return (
            f"[{self.index + 1}/{len(ROLE_ORDER)}] {role.upper()} — {ROLE_HELP[role]}   ({state})\n"
            f"shape: {self.kind}   ·   [r] rectangle   [e] ellipse   "
            f"[enter] confirm   [u] undo   [esc] cancel\n"
            f"display: {self.stretch} stretch   ·   [l] toggle stretch"
        )


def _patch_for(region: Region, color: str):
    from matplotlib import patches

    style = dict(edgecolor=color, linewidth=2.0, facecolor=color, alpha=0.15)
    if region.kind == "rect":
        return patches.Rectangle(
            (region.x - 0.5, region.y - 0.5), region.width, region.height, **style
        )
    return patches.Ellipse(
        (region.cx, region.cy), 2 * region.rx, 2 * region.ry, **style
    )


class SelectionWindow:
    """The matplotlib side of the selection: figure, selectors, key handling.

    Separated from :func:`select_regions` so the drawing can be built and
    inspected — or saved to a file — without an interactive backend.
    """

    def __init__(self, image: np.ndarray, source_file: str | None = None):
        import matplotlib.pyplot as plt
        from matplotlib.widgets import EllipseSelector, RectangleSelector

        if image.ndim != 2:
            raise RegionError(f"expected a 2-D image, got shape {image.shape}")
        self.session = SelectionSession(
            shape=(int(image.shape[0]), int(image.shape[1])), source_file=source_file
        )
        self._plt = plt
        self._patches: list = []
        self.closed = False

        self.figure, self.axes = plt.subplots(figsize=(9.0, 7.0), layout="constrained")
        self._image = image
        low, high = display_limits(image)
        self._artist = self.axes.imshow(
            image,
            cmap="gray",
            norm=display_norm(image, self.session.stretch),
            interpolation="nearest",
        )
        self.axes.set_xlabel(
            f"display clipped to {DISPLAY_PERCENTILES[0]:g}-{DISPLAY_PERCENTILES[1]:g}% "
            f"({low:g}-{high:g} counts) — the metric always uses the raw counts",
            fontsize=8,
            color=TEXT_MUTED,
        )
        self.figure.suptitle(source_file or "reference image", fontsize=11)
        self.banner = self.axes.set_title(
            self.session.instructions(), fontsize=9, loc="left", family="monospace"
        )

        common = dict(useblit=False, button=[1], interactive=True, spancoords="data")
        self.selectors = {
            "rect": RectangleSelector(self.axes, self._on_select, **common),
            "ellipse": EllipseSelector(self.axes, self._on_select, **common),
        }
        self._activate(self.session.kind)
        self.figure.canvas.mpl_connect("key_press_event", self.on_key)
        self.redraw()

    # -- drawing ------------------------------------------------------------ #

    def _activate(self, kind: str) -> None:
        for name, selector in self.selectors.items():
            selector.set_active(name == kind)
            selector.set_visible(name == kind)

    def redraw(self) -> None:
        for artist in self._patches:
            artist.remove()
        self._patches.clear()
        for role, region in self.session.confirmed.items():
            patch = _patch_for(region, ROLE_COLORS[role])
            self.axes.add_patch(patch)
            self._patches.append(patch)
            self._patches.append(
                self.axes.annotate(
                    role,
                    xy=(region.bounds[0], region.bounds[1] - 2),
                    color=ROLE_COLORS[role],
                    fontsize=9,
                    fontweight="bold",
                )
            )
        text = self.session.instructions()
        if self.session.error:
            text += f"\n! {self.session.error}"
        self.banner.set_text(text)
        self.figure.canvas.draw_idle()

    # -- events ------------------------------------------------------------- #

    def _on_select(self, eclick, erelease) -> None:
        selector = self.selectors[self.session.kind]
        self.session.draw(tuple(float(v) for v in selector.extents))
        self.redraw()

    def close(self) -> None:
        self.closed = True
        self._plt.close(self.figure)

    def on_key(self, event) -> None:
        """Handle one keypress. Accepts anything with a ``key`` attribute."""
        key = (getattr(event, "key", "") or "").lower()
        if key in ("r", "e"):
            self.session.set_shape("rect" if key == "r" else "ellipse")
            self._activate(self.session.kind)
        elif key == "enter":
            if self.session.confirm() and self.session.done:
                self.redraw()
                self.close()
                return
        elif key == "l":
            index = (STRETCHES.index(self.session.stretch) + 1) % len(STRETCHES)
            self.session.stretch = STRETCHES[index]
            self._artist.set_norm(display_norm(self._image, self.session.stretch))
        elif key == "u":
            self.session.undo()
        elif key == "escape":
            self.session.cancel()
            self.close()
            return
        self.redraw()


def select_regions(image: np.ndarray, source_file: str | None = None) -> RegionSet | None:
    """Open the blocking selection window. ``None`` if the user cancelled.

    Blocking, like the plot window (locked decision 6): the menu loop resumes
    when the window closes.
    """
    import matplotlib.pyplot as plt

    # Our own single-letter keys would otherwise collide with matplotlib's
    # navigation shortcuts — 'r' resets the view, 'q' quits the window.
    keymap_off = {key: [] for key in matplotlib.rcParams if key.startswith("keymap.")}
    with matplotlib.rc_context(keymap_off):
        window = SelectionWindow(image, source_file)
        try:
            plt.show()
        finally:
            if not window.closed:
                window.close()
    return window.session.result()
