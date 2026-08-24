"""TIFF loading.

``tifffile``, not PIL: PIL is quietly unreliable on 16/32-bit scientific TIFFs
and on unusual photometric interpretations.

Images are read **as-is** — never normalised or rescaled — and converted to
float64 before any accumulation, so a standard deviation comes out in raw counts
and integer types cannot overflow.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, get_args

import numpy as np
import tifffile

PagePolicy = Literal["error", "first", "mean"]
PAGE_POLICIES: tuple[str, ...] = get_args(PagePolicy)
DEFAULT_PAGE_POLICY: PagePolicy = "error"

PAGE_POLICY_HELP: dict[str, str] = {
    "error": "refuse multi-page files (default — a stack never silently becomes one number)",
    "first": "use the first page",
    "mean": "average all pages",
}


class LoadError(Exception):
    """A file could not be turned into a single 2-D float64 image."""


class MultiPageError(LoadError):
    """A multi-page TIFF was found under the ``error`` policy."""


class ColorImageError(LoadError):
    """Colour / multi-channel images are out of scope in v1."""


class UnsupportedImageError(LoadError):
    """Shape or dtype the pipeline cannot reduce to a 2-D plane."""


def _as_plane(array: np.ndarray, path: Path, page_index: int | None = None) -> np.ndarray:
    """Validate that *array* is a single 2-D plane and return it as float64."""
    where = f"{path.name}" if page_index is None else f"{path.name} (page {page_index})"
    if array.ndim == 3 and array.shape[-1] == 1:
        array = array[..., 0]
    if array.ndim == 3:
        raise ColorImageError(
            f"{where}: {array.shape[-1]}-channel image; colour/multi-channel is out of scope in v1"
        )
    if array.ndim != 2:
        raise UnsupportedImageError(
            f"{where}: expected a 2-D image, got shape {array.shape}"
        )
    if array.dtype.kind not in "uifb":
        raise UnsupportedImageError(f"{where}: unsupported pixel dtype {array.dtype}")
    return np.asarray(array, dtype=np.float64)


def page_count(path: Path) -> int:
    """Number of pages in *path*, without decoding pixel data."""
    with tifffile.TiffFile(path) as tif:
        return len(tif.pages)


def image_dtype(path: Path) -> np.dtype:
    """On-disk pixel dtype, read from the header only.

    ``load_image`` converts to float64 on the way out, so the original type —
    and with it the saturation level — has to be read separately.
    """
    try:
        with tifffile.TiffFile(path) as tif:
            if not tif.pages:
                raise UnsupportedImageError(f"{Path(path).name}: file contains no image pages")
            return np.dtype(tif.pages[0].dtype)
    except LoadError:
        raise
    except (OSError, ValueError, tifffile.TiffFileError) as exc:
        raise LoadError(f"{Path(path).name}: could not read TIFF header ({exc})") from exc


def image_shape(path: Path) -> tuple[int, int]:
    """``(height, width)`` of the first page, read from the header only.

    Cheap enough to call when entering a directory, which is what checking the
    saved areas against the data needs.
    """
    try:
        with tifffile.TiffFile(path) as tif:
            if not tif.pages:
                raise UnsupportedImageError(f"{Path(path).name}: file contains no image pages")
            shape = tuple(int(v) for v in tif.pages[0].shape[:2])
    except LoadError:
        raise
    except (OSError, ValueError, tifffile.TiffFileError) as exc:
        raise LoadError(f"{Path(path).name}: could not read TIFF header ({exc})") from exc
    if len(shape) != 2:
        raise UnsupportedImageError(f"{Path(path).name}: unexpected page shape {shape}")
    return shape[0], shape[1]


def load_image(path: Path, policy: PagePolicy = DEFAULT_PAGE_POLICY) -> np.ndarray:
    """Open *path* and return a single 2-D float64 image.

    One data point per file: a multi-page TIFF is resolved by *policy*
    (``error`` by default) rather than being silently collapsed.
    """
    path = Path(path)
    if policy not in PAGE_POLICIES:
        raise ValueError(f"unknown page policy {policy!r}; expected one of {PAGE_POLICIES}")

    try:
        with tifffile.TiffFile(path) as tif:
            pages = tif.pages
            n_pages = len(pages)
            if n_pages == 0:
                raise UnsupportedImageError(f"{path.name}: file contains no image pages")

            first = pages[0]
            samples = int(getattr(first, "samplesperpixel", 1) or 1)
            if samples > 1:
                raise ColorImageError(
                    f"{path.name}: {samples} samples per pixel; "
                    "colour/multi-channel is out of scope in v1"
                )

            if n_pages == 1:
                return _as_plane(first.asarray(), path)

            if policy == "error":
                raise MultiPageError(
                    f"{path.name}: {n_pages} pages. One data point per file — choose the "
                    "'first' or 'mean' page policy if collapsing the stack is what you want."
                )

            if policy == "first":
                return _as_plane(first.asarray(), path, page_index=0)

            # policy == "mean": accumulate in float64 so integer pixels cannot overflow.
            accumulator: np.ndarray | None = None
            for index, page in enumerate(pages):
                plane = _as_plane(page.asarray(), path, page_index=index)
                if accumulator is None:
                    accumulator = plane
                    continue
                if plane.shape != accumulator.shape:
                    raise UnsupportedImageError(
                        f"{path.name}: page {index} has shape {plane.shape}, "
                        f"page 0 has {accumulator.shape}; cannot average"
                    )
                accumulator += plane
            assert accumulator is not None
            return accumulator / n_pages
    except LoadError:
        raise
    except (OSError, ValueError, tifffile.TiffFileError) as exc:
        raise LoadError(f"{path.name}: could not read TIFF ({exc})") from exc
