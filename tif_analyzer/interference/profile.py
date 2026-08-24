"""Operation 3 — the fixed-pattern profile, plus the axis sanity check.

Each row is a scan line and the column index is time within the line, so the
fixed pattern is the **column-wise median**: for every position in the line, the
typical value across all lines. Median, not mean, so a bead crossing a few lines
does not drag the point with it.

The profile is mean-centred; its standard deviation is the amplitude of the
fixed pattern in raw counts.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .dsp import pearson
from .mask import MaskResult
from .params import InterferenceParams


@dataclass(frozen=True)
class AxisCheck:
    """Stripe strength on both axes.

    Analysing the wrong axis produces perfectly plausible numbers, so the frame
    is measured both ways and the result says which way the pattern actually
    runs.
    """

    column_profile_std: float
    row_profile_std: float
    ratio: float  # row / column
    warning: str | None

    @property
    def consistent(self) -> bool:
        return self.warning is None


@dataclass(frozen=True)
class ProfileResult:
    values: np.ndarray  # mean-centred, flyback-trimmed
    std: float
    masked_values: np.ndarray | None
    masked_std: float | None
    contamination_rms_ratio: float | None
    contamination_correlation: float | None
    columns_used: int
    flyback_columns_dropped: int
    columns_without_background: int

    @property
    def analysis_profile(self) -> np.ndarray:
        """The profile the spectrum should read.

        The masked one when it exists: the raw profile can carry bead signal,
        and that is exactly what the contamination check is there to reveal.
        """
        return self.values if self.masked_values is None else self.masked_values


def axis_check(image: np.ndarray, params: InterferenceParams | None = None) -> AxisCheck:
    """Compare stripe strength along both axes."""
    params = params or InterferenceParams()
    data = np.asarray(image, dtype=np.float64)
    column_profile = np.median(data, axis=0)
    row_profile = np.median(data, axis=1)
    column_std = float(np.std(column_profile))
    row_std = float(np.std(row_profile))

    warning = None
    if column_std <= 0.0:
        ratio = float("inf") if row_std > 0 else float("nan")
    else:
        ratio = row_std / column_std
    if np.isfinite(ratio) and ratio >= params.axis_warn_ratio:
        warning = (
            f"the pattern is at least as strong across rows as across columns "
            f"(row/column stripe strength = {ratio:.2f}) — this analysis assumes each ROW is a "
            "scan line with time running along the columns; if the fast axis is the other way, "
            "every number below is measuring the wrong thing"
        )
    return AxisCheck(
        column_profile_std=column_std, row_profile_std=row_std, ratio=ratio, warning=warning
    )


def _trim(data: np.ndarray, flyback_columns: int) -> np.ndarray:
    """Drop the flyback columns from the end of every line."""
    return data if flyback_columns <= 0 else data[:, : data.shape[1] - flyback_columns]


def build_profile(
    image: np.ndarray,
    mask_result: MaskResult | None = None,
    flyback_columns: int = 0,
    params: InterferenceParams | None = None,
) -> ProfileResult:
    """Column-wise median profile, raw and bead-masked, plus their discrepancy."""
    params = params or InterferenceParams()
    data = _trim(np.asarray(image, dtype=np.float64), flyback_columns)

    raw = np.median(data, axis=0)
    raw = raw - raw.mean()

    masked = None
    masked_std = None
    rms_ratio = None
    correlation = None
    columns_without_background = 0

    if mask_result is not None:
        mask = _trim(mask_result.mask, flyback_columns)
        masked = np.empty(data.shape[1], dtype=np.float64)
        for column in range(data.shape[1]):
            kept = data[mask[:, column], column]
            if kept.size < params.min_background_rows_per_column:
                # Too little background left here to trust; fall back to the raw
                # median rather than inventing a value, and count the column.
                columns_without_background += 1
                masked[column] = np.median(data[:, column])
            else:
                masked[column] = np.median(kept)
        masked = masked - masked.mean()
        masked_std = float(np.std(masked))

        difference = raw - masked
        raw_std = float(np.std(raw))
        rms = float(np.sqrt(np.mean(difference**2)))
        rms_ratio = rms / raw_std if raw_std > 0 else float("nan")
        correlation = pearson(raw, masked) if raw.size >= 2 else float("nan")

    return ProfileResult(
        values=raw,
        std=float(np.std(raw)),
        masked_values=masked,
        masked_std=masked_std,
        contamination_rms_ratio=rms_ratio,
        contamination_correlation=correlation,
        columns_used=int(data.shape[1]),
        flyback_columns_dropped=int(flyback_columns),
        columns_without_background=columns_without_background,
    )
