"""Operation 1 — load and validate.

Read as float64 with no rescaling (the whole point is raw counts), record the
shape, the on-disk dtype and the basic statistics, and flag saturation at the
type limit. Frames that do not have the same shape are rejected: a
cross-channel comparison between differently sized frames is meaningless.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .. import loading
from ..loading import DEFAULT_PAGE_POLICY, PagePolicy


class FrameError(Exception):
    """A frame that cannot be characterised."""


class ShapeMismatchError(FrameError):
    """Signal and reference frames disagree on shape."""


@dataclass(frozen=True)
class Frame:
    """One validated 2-D frame in raw counts."""

    data: np.ndarray
    path: Path | None
    dtype: str
    saturation_level: float | None
    saturated_pixels: int

    @property
    def shape(self) -> tuple[int, int]:
        return int(self.data.shape[0]), int(self.data.shape[1])

    @property
    def n_lines(self) -> int:
        return self.shape[0]

    @property
    def n_columns(self) -> int:
        return self.shape[1]

    @property
    def name(self) -> str:
        return self.path.name if self.path is not None else "<array>"

    @property
    def minimum(self) -> float:
        return float(np.min(self.data))

    @property
    def maximum(self) -> float:
        return float(np.max(self.data))

    @property
    def median(self) -> float:
        return float(np.median(self.data))

    @property
    def saturated_fraction(self) -> float:
        return self.saturated_pixels / self.data.size

    @property
    def saturated(self) -> bool:
        return self.saturated_pixels > 0


def _saturation_level(dtype: np.dtype) -> float | None:
    """The value that means "the detector ran out of range", or None when the
    dtype has no such limit."""
    if dtype.kind in "ui":
        return float(np.iinfo(dtype).max)
    return None


def frame_from_array(
    data: np.ndarray, dtype: np.dtype | str = np.uint16, path: Path | None = None
) -> Frame:
    """Build a :class:`Frame` from an in-memory array, for synthetic data."""
    array = np.asarray(data, dtype=np.float64)
    if array.ndim != 2:
        raise FrameError(f"expected a 2-D frame, got shape {array.shape}")
    if array.shape[0] < 2 or array.shape[1] < 4:
        raise FrameError(f"frame is too small to characterise: {array.shape}")
    resolved = np.dtype(dtype)
    level = _saturation_level(resolved)
    saturated = int(np.count_nonzero(array >= level)) if level is not None else 0
    return Frame(
        data=array, path=path, dtype=resolved.name, saturation_level=level, saturated_pixels=saturated
    )


def load_frame(path: Path, page_policy: PagePolicy = DEFAULT_PAGE_POLICY) -> Frame:
    """Read *path* as a single 2-D float64 frame in raw counts."""
    path = Path(path)
    array = loading.load_image(path, page_policy)
    dtype = loading.image_dtype(path)
    return frame_from_array(array, dtype=dtype, path=path)


def validate_pair(signal: Frame, reference: Frame) -> None:
    """Reject mismatched shapes before anything is computed from them."""
    if signal.shape != reference.shape:
        raise ShapeMismatchError(
            f"signal {signal.name} is {signal.n_columns}x{signal.n_lines} px but reference "
            f"{reference.name} is {reference.n_columns}x{reference.n_lines} px — "
            "a cross-channel comparison needs identical geometry"
        )
