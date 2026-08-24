"""Tunables and acquisition metadata, all in one place.

Every threshold the verdict depends on lives here, so it can be audited and
changed without reading the maths. The defaults are **provisional**: they were
chosen from first principles, not calibrated against real ImSpector frames, and
the report says so.
"""

from __future__ import annotations

from dataclasses import dataclass


class AcquisitionError(ValueError):
    """Acquisition metadata that cannot be true."""


@dataclass(frozen=True)
class Acquisition:
    """Metadata needed to turn cycles/pixel into Hz. All of it optional.

    ``resolution`` is pixels per scan line; when omitted the frame's column
    count is used.

    ``frame_includes_flyback`` decides two coupled things at once, which is why
    it is a single flag rather than two:

    * ``False`` (default) — the saved columns cover only the active part of the
      line, so ``t_dwell = T_line·(1 − flyback) / N`` and no column is dropped.
    * ``True`` — the saved columns span the whole line period, so
      ``t_dwell = T_line / N`` and the flyback columns are excluded from the
      profile, since they carry no signal.
    """

    line_rate_hz: float | None = None
    resolution: int | None = None
    flyback_fraction: float = 0.0
    frame_includes_flyback: bool = False

    def __post_init__(self) -> None:
        if self.line_rate_hz is not None and self.line_rate_hz <= 0:
            raise AcquisitionError(f"line rate must be positive, got {self.line_rate_hz}")
        if self.resolution is not None and self.resolution < 2:
            raise AcquisitionError(f"resolution must be at least 2 px, got {self.resolution}")
        if not 0.0 <= self.flyback_fraction < 1.0:
            raise AcquisitionError(
                f"flyback fraction must be in [0, 1), got {self.flyback_fraction}"
            )

    @property
    def complete(self) -> bool:
        """Whether Hz can be derived at all."""
        return self.line_rate_hz is not None

    @property
    def line_period_s(self) -> float | None:
        return None if self.line_rate_hz is None else 1.0 / self.line_rate_hz

    def dwell_s(self, n_columns: int) -> float | None:
        """Time per pixel, from the line rate and the pixels per line."""
        if self.line_rate_hz is None:
            return None
        pixels = self.resolution or n_columns
        active = 1.0 if self.frame_includes_flyback else (1.0 - self.flyback_fraction)
        return self.line_period_s * active / pixels  # type: ignore[operator]

    def flyback_columns(self, n_columns: int) -> int:
        """Columns to drop from the profile (only when they are in the frame)."""
        if not self.frame_includes_flyback or self.flyback_fraction <= 0:
            return 0
        return min(n_columns - 2, int(round(n_columns * self.flyback_fraction)))


@dataclass(frozen=True)
class InterferenceParams:
    """Knobs for the whole pipeline. Provisional until calibrated on real data."""

    # -- bead mask ---------------------------------------------------------- #
    #: Rejection threshold in robust sigmas above the background median.
    mad_k: float = 3.0
    mask_max_iterations: int = 10
    #: Below this surviving fraction the background estimate is called unreliable.
    min_background_fraction: float = 0.10
    #: A column with fewer surviving rows than this falls back to the raw median.
    min_background_rows_per_column: int = 8

    # -- spectrum ----------------------------------------------------------- #
    window: str = "hann"
    #: Bins skipped at the low end: DC plus window leakage from any residual trend.
    exclude_low_bins: int = 2
    #: Cluster grows around the peak while power stays above this fraction of
    #: it and keeps descending. Low on purpose: the window's own side lobes
    #: belong to the component, and excluding them would understate its power.
    cluster_power_drop: float = 0.05
    #: Absolute floor for the peak-to-noise ratio, whatever the bin count says.
    detection_ratio: float = 4.0
    #: Tolerated probability of calling pure noise a detection. The real
    #: threshold is derived from this and the number of bins searched: the
    #: periodogram of white noise is exponentially distributed, so the tallest
    #: of N bins sits around ln(N)/ln(2) times the median by chance alone —
    #: about 9x for a 512-column frame. A fixed low threshold would report a
    #: period for every noise frame handed to it.
    detection_false_alarm: float = 0.01

    # -- line stability ----------------------------------------------------- #
    #: Lines with more than this fraction of pixels rejected are skipped.
    max_line_reject_fraction: float = 0.5

    # -- axis sanity -------------------------------------------------------- #
    #: Warn when the other axis carries at least this much of the primary's
    #: stripe strength — the usual sign the fast axis was set the wrong way.
    axis_warn_ratio: float = 1.0

    # -- verdict (the provisional numbers) ---------------------------------- #
    #: Periods agree when they differ by less than this, relative.
    period_tolerance: float = 0.02
    #: Band-restricted correlation needed to call subtraction viable.
    band_corr_threshold: float = 0.80
    #: Gap between band and raw correlation that means raw subtraction would
    #: inject the reference's broadband noise.
    raw_band_gap_threshold: float = 0.30

    #: Alias candidates emitted per direction.
    alias_orders: int = 3
