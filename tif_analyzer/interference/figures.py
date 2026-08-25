"""Figures for the interference characterisation.

The spec put plotting out of scope for v0; this was added afterwards, on
request. It stays on the analysis side of the line — these functions build
figures and never show or save them, so the caller decides between a blocking
window and a PNG, exactly as the series plot does.

First figure: the bead mask. The report gives two numbers about it — the
fraction rejected overall and in the worst column — and neither tells you
*where* the rejection happened or whether the threshold sits in a sensible
place. This shows all three at once.
"""

from __future__ import annotations

import numpy as np

from ..plotting import (
    AXIS_COLOR,
    GRID_COLOR,
    TEXT_MUTED,
    TEXT_PRIMARY,
    display_limits,
    display_norm,
)
from .dsp import goertzel, window
from .frames import Frame
from .mask import MaskResult
from .params import InterferenceParams

#: Warm hue for what was removed, against the greyscale frame underneath.
#: Blue/orange stays separable under the common colour-vision deficiencies, and
#: every panel is labelled anyway.
REJECT_COLOR = "#D97706"
KEEP_COLOR = "#2E6FD9"

DISPLAY_NOTE = (
    "{stretch} stretch, {low:.4g}-{high:.4g} counts (raw counts are what gets measured)"
)


def _style_axes(axes) -> None:
    axes.grid(True, color=GRID_COLOR, linewidth=0.8, alpha=0.9, zorder=0)
    axes.set_axisbelow(True)
    for side in ("top", "right"):
        axes.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        axes.spines[side].set_color(AXIS_COLOR)
    axes.tick_params(colors=TEXT_MUTED, labelsize=8)


def build_mask_figure(
    frame: Frame,
    mask_result: MaskResult,
    params: InterferenceParams | None = None,
    stretch: str = "gamma",
):
    """Three aligned views of the bead mask.

    1. the frame, with the rejected pixels painted over it;
    2. the rejected fraction per column, sharing the x axis with the image, so a
       spike lines up with the bead that caused it;
    3. the pixel histogram with the median and the cut, which is what says
       whether ``mad_k`` is doing something sensible.
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap

    params = params or InterferenceParams()
    data = frame.data
    mask = mask_result.mask
    rejected = ~mask
    n_lines, n_columns = data.shape

    figure = plt.figure(figsize=(10.0, 8.0), layout="constrained")
    grid = figure.add_gridspec(3, 1, height_ratios=[3.0, 1.1, 1.1], hspace=0.06)
    image_axes = figure.add_subplot(grid[0])
    column_axes = figure.add_subplot(grid[1], sharex=image_axes)
    histogram_axes = figure.add_subplot(grid[2])

    # -- 1. the frame, with rejections painted on top --------------------------
    low, high = display_limits(data)
    image_axes.imshow(
        data,
        cmap="gray",
        norm=display_norm(data, stretch),
        interpolation="nearest",
        aspect="auto",
    )
    overlay = np.ma.masked_where(~rejected, rejected.astype(float))
    image_axes.imshow(
        overlay,
        cmap=ListedColormap([REJECT_COLOR]),
        alpha=0.55,
        interpolation="nearest",
        aspect="auto",
        vmin=0.0,
        vmax=1.0,
    )
    image_axes.set_ylabel("scan line", color=TEXT_PRIMARY, fontsize=9)
    image_axes.tick_params(labelbottom=False, colors=TEXT_MUTED, labelsize=8)
    image_axes.set_title(
        f"{frame.name} — bead mask", loc="left", fontsize=12, color=TEXT_PRIMARY, pad=16
    )
    image_axes.annotate(
        f"{mask_result.rejected_fraction:.2%} of pixels rejected in "
        f"{mask_result.iterations} iteration(s)"
        + ("" if mask_result.converged else ", DID NOT CONVERGE")
        + f"   ·   display {DISPLAY_NOTE.format(low=low, high=high, stretch=stretch)}",
        xy=(0.0, 1.015),
        xycoords="axes fraction",
        fontsize=8,
        color=TEXT_MUTED,
        annotation_clip=False,
    )

    # -- 2. rejected fraction per column ---------------------------------------
    per_column = 1.0 - mask.mean(axis=0)
    columns = np.arange(n_columns)
    column_axes.fill_between(columns, per_column, color=REJECT_COLOR, alpha=0.35, zorder=2)
    column_axes.plot(columns, per_column, color=REJECT_COLOR, linewidth=1.0, zorder=3)
    column_axes.axhline(
        mask_result.rejected_fraction,
        color=KEEP_COLOR,
        linewidth=1.2,
        linestyle="--",
        zorder=4,
        label=f"frame mean {mask_result.rejected_fraction:.2%}",
    )

    # Above this fraction a column has too few rows left and falls back to the
    # unmasked median — the point where its profile value stops being trustworthy.
    fallback = 1.0 - params.min_background_rows_per_column / n_lines
    if 0.0 < fallback < 1.0:
        column_axes.axhline(
            fallback,
            color=TEXT_MUTED,
            linewidth=1.0,
            linestyle=":",
            zorder=4,
            label=f"falls back to the raw median above {fallback:.1%}",
        )

    worst = mask_result.worst_column_index
    column_axes.plot(
        [worst],
        [mask_result.worst_column_rejected_fraction],
        marker="o",
        markersize=7,
        color=REJECT_COLOR,
        markeredgecolor="white",
        markeredgewidth=1.0,
        zorder=5,
        label=f"worst: column {worst} at {mask_result.worst_column_rejected_fraction:.1%}",
    )
    column_axes.set_ylabel("rejected\nper column", color=TEXT_PRIMARY, fontsize=9)
    column_axes.set_xlabel("column (time within the scan line)", color=TEXT_PRIMARY, fontsize=9)
    column_axes.set_xlim(0, n_columns - 1)
    column_axes.set_ylim(0, max(0.05, float(per_column.max()) * 1.15))
    column_axes.legend(fontsize=7, loc="upper right", framealpha=0.9)
    _style_axes(column_axes)

    # -- 3. where the cut sits in the value distribution -----------------------
    values = data.ravel()
    # Zoom on the background and the cut. Beads reach tens of thousands of
    # counts, and letting them set the range squashes the only part of the
    # distribution this panel exists to let you judge.
    span = max(mask_result.sigma, 1.0)
    lower = min(float(values.min()), mask_result.median - 5.0 * span)
    upper = max(mask_result.threshold + 5.0 * span, mask_result.median + 8.0 * span)
    edges = np.linspace(lower, upper, 200)
    inside = values[(values >= lower) & (values <= upper)]
    histogram_axes.hist(inside, bins=edges, color=KEEP_COLOR, alpha=0.75, zorder=2)
    above = int(np.count_nonzero(values > upper))
    if above:
        histogram_axes.annotate(
            f"{above} px above {upper:.4g} (beads), off scale",
            xy=(0.98, 0.55),
            xycoords="axes fraction",
            ha="right",
            fontsize=7,
            color=TEXT_MUTED,
        )
    histogram_axes.set_xlim(lower, upper)
    histogram_axes.axvline(
        mask_result.median,
        color=TEXT_PRIMARY,
        linewidth=1.2,
        zorder=4,
        label=f"median {mask_result.median:.4g}",
    )
    histogram_axes.axvline(
        mask_result.threshold,
        color=REJECT_COLOR,
        linewidth=1.6,
        zorder=4,
        label=f"cut at median + {params.mad_k:g}σ = {mask_result.threshold:.4g}",
    )
    histogram_axes.set_yscale("log")
    histogram_axes.set_xlabel("pixel value (raw counts)", color=TEXT_PRIMARY, fontsize=9)
    histogram_axes.set_ylabel("pixels", color=TEXT_PRIMARY, fontsize=9)
    histogram_axes.legend(fontsize=7, loc="upper right", framealpha=0.9)
    _style_axes(histogram_axes)
    if not mask_result.reliable and mask_result.reason:
        histogram_axes.annotate(
            f"! {mask_result.reason}",
            xy=(0.0, -0.45),
            xycoords="axes fraction",
            fontsize=8,
            color=REJECT_COLOR,
            annotation_clip=False,
            wrap=True,
        )
    return figure



def build_profile_figure(
    analysis,
    params: InterferenceParams | None = None,
):
    """The fixed-pattern profile beside its spectrum, plus a zoom.

    Left: the profile the FFT actually read, with the raw one behind it — the
    gap between them is the bead contamination the report quantifies.
    Right: the power spectrum, with the cluster shaded and the detection
    threshold drawn, so "detected" is a thing you can see rather than trust.
    Bottom: a few periods, with the cluster's own reconstruction on top, which
    is what says whether one component really describes this profile.
    """
    import matplotlib.pyplot as plt

    params = params or InterferenceParams()
    profile = analysis.profile
    spectrum = analysis.spectrum
    values = profile.analysis_profile
    columns = np.arange(values.size)

    figure = plt.figure(figsize=(12.0, 7.0), layout="constrained")
    grid = figure.add_gridspec(2, 2, width_ratios=[2.0, 1.0], height_ratios=[1.0, 0.85])
    profile_axes = figure.add_subplot(grid[0, 0])
    spectrum_axes = figure.add_subplot(grid[0, 1])
    zoom_axes = figure.add_subplot(grid[1, :])

    # -- the profile -----------------------------------------------------------
    if profile.masked_values is not None:
        profile_axes.plot(
            columns,
            profile.values,
            color=TEXT_MUTED,
            linewidth=0.8,
            alpha=0.7,
            zorder=2,
            label=f"raw column median (std {profile.std:.4g})",
        )
        label = f"bead-masked (std {profile.masked_std:.4g})"
    else:
        label = f"column median (std {profile.std:.4g})"
    profile_axes.plot(columns, values, color=KEEP_COLOR, linewidth=1.1, zorder=3, label=label)
    profile_axes.set_ylabel("counts, mean-centred", color=TEXT_PRIMARY, fontsize=9)
    profile_axes.set_xlabel("column (time within the scan line)", color=TEXT_PRIMARY, fontsize=9)
    profile_axes.set_xlim(0, max(1, values.size - 1))
    profile_axes.set_title(
        f"{analysis.frame.name} — fixed-pattern profile",
        loc="left",
        fontsize=12,
        color=TEXT_PRIMARY,
        pad=14,
    )
    if profile.flyback_columns_dropped:
        profile_axes.annotate(
            f"{profile.flyback_columns_dropped} flyback column(s) dropped",
            xy=(0.0, 1.02),
            xycoords="axes fraction",
            fontsize=8,
            color=TEXT_MUTED,
            annotation_clip=False,
        )
    profile_axes.legend(fontsize=7, loc="upper right", framealpha=0.9)
    _style_axes(profile_axes)

    # -- the spectrum ----------------------------------------------------------
    frequencies = spectrum.frequencies
    power = spectrum.power
    low = max(1, params.exclude_low_bins)
    spectrum_axes.plot(
        frequencies[low:], power[low:], color=KEEP_COLOR, linewidth=1.0, zorder=3
    )
    if low > 1:
        spectrum_axes.axvspan(
            frequencies[1], frequencies[low], color=TEXT_MUTED, alpha=0.12, zorder=1,
            label=f"not searched ({low} low bins)",
        )
    band_low, band_high = spectrum.cluster_frequency_range
    spectrum_axes.axvspan(
        band_low, band_high, color=REJECT_COLOR, alpha=0.22, zorder=2,
        label=f"cluster, {spectrum.power_fraction:.0%} of the power",
    )
    spectrum_axes.axvline(
        spectrum.centroid_frequency,
        color=REJECT_COLOR,
        linewidth=1.4,
        zorder=4,
        label=f"centroid {spectrum.centroid_frequency:.5f} c/px  ({spectrum.period_px:.2f} px)",
    )
    if spectrum.noise_floor > 0:
        spectrum_axes.axhline(
            spectrum.noise_floor * spectrum.required_ratio,
            color=TEXT_PRIMARY,
            linestyle="--",
            linewidth=1.0,
            zorder=4,
            label=f"detection needs {spectrum.required_ratio:.0f}x the floor",
        )
        spectrum_axes.axhline(
            spectrum.noise_floor, color=TEXT_MUTED, linestyle=":", linewidth=1.0, zorder=4,
            label="noise floor (median)",
        )
    spectrum_axes.set_yscale("log")
    # From the first non-DC bin: the period axis on top is 1/f, and including
    # zero would ask it to label an infinite period.
    spectrum_axes.set_xlim(float(frequencies[1]), 0.5)
    spectrum_axes.set_xlabel("cycles per pixel", color=TEXT_PRIMARY, fontsize=9)
    spectrum_axes.set_ylabel("power", color=TEXT_PRIMARY, fontsize=9)
    spectrum_axes.set_title(
        "spectrum — " + ("DETECTED" if spectrum.detected else "NOT DETECTED"),
        loc="left",
        fontsize=10,
        color=TEXT_PRIMARY if spectrum.detected else REJECT_COLOR,
        pad=14,
    )
    period_axis = spectrum_axes.secondary_xaxis(
        "top", functions=(_safe_reciprocal, _safe_reciprocal)
    )
    # Explicit ticks, chosen to be spread along the *frequency* axis: 1/f
    # bunches long periods against the left edge, and the default locator turns
    # the period axis into an unreadable smear there.
    periods = [
        p
        for p in (2, 3, 4, 6, 10, 20, 40)
        if 0.02 <= 1.0 / p <= 0.5 and p <= values.size / 4
    ]
    period_axis.set_xticks(periods)
    period_axis.set_xticklabels([str(p) for p in periods])
    period_axis.set_xlabel("period (px)", color=TEXT_MUTED, fontsize=8)
    period_axis.tick_params(colors=TEXT_MUTED, labelsize=7)
    spectrum_axes.legend(fontsize=6.5, loc="upper right", framealpha=0.9)
    _style_axes(spectrum_axes)

    # -- a few periods, with the cluster's reconstruction -----------------------
    if spectrum.detected and np.isfinite(spectrum.period_px):
        span = int(min(values.size, max(40, round(6 * spectrum.period_px))))
    else:
        span = int(min(values.size, 120))
    # A single sinusoid at the centroid, fitted by projection — not an inverse
    # transform of the cluster bins. The cluster was found in the *windowed*
    # spectrum, and a real period is never exactly on a bin, so inverting those
    # bins on the unwindowed profile loses amplitude to leakage and draws a
    # component visibly smaller than the profile it is supposed to explain.
    reconstruction = np.zeros_like(values)
    amplitude = float("nan")
    if spectrum.detected and spectrum.centroid_frequency > 0:
        weights = window(params.window, values.size)
        projection = goertzel(values - values.mean(), spectrum.centroid_frequency, weights)
        amplitude = 2.0 * abs(projection) / float(np.sum(weights))
        phase = float(np.angle(projection))
        reconstruction = amplitude * np.cos(
            2.0 * np.pi * spectrum.centroid_frequency * columns + phase
        )
    zoom_axes.plot(
        columns[:span], values[:span], color=KEEP_COLOR, linewidth=1.2, marker="o",
        markersize=3, zorder=3, label="profile",
    )
    zoom_axes.plot(
        columns[:span], reconstruction[:span], color=REJECT_COLOR, linewidth=1.8, zorder=4,
        label=f"single component at the centroid (amplitude {amplitude:.4g} counts)",
    )
    residual = float(np.std(values - reconstruction))
    headroom = float(np.max(np.abs(values[:span]))) * 1.55
    zoom_axes.set_ylim(-headroom / 1.35, headroom)
    zoom_axes.set_xlabel(
        f"column — first {span} of {values.size}", color=TEXT_PRIMARY, fontsize=9
    )
    zoom_axes.set_ylabel("counts", color=TEXT_PRIMARY, fontsize=9)
    zoom_axes.set_title(
        f"the dominant component against the profile   ·   residual std {residual:.4g} counts",
        loc="left",
        fontsize=9,
        color=TEXT_MUTED,
        pad=8,
    )
    zoom_axes.legend(fontsize=7, loc="upper right", framealpha=0.9)
    _style_axes(zoom_axes)
    return figure


def _safe_reciprocal(values):
    """1/x for the period axis, kept finite so the axis can always be scaled."""
    values = np.asarray(values, dtype=np.float64)
    safe = np.where(np.abs(values) < 1e-12, 1e-12, values)
    return 1.0 / safe
