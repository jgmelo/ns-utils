"""Named image regions (signal / noise) and their pixel extraction.

Regions are *metric input*, not preprocessing: a preprocess step transforms one
image into another image, while these are two named samples with different
roles, consumed side by side by a single metric. So they travel alongside the
preprocess pipeline rather than inside it, and the v1 pipeline stays empty as
the spec fixed it.

Coordinates are pixel indices in the reference image: pixel ``i`` sits at
coordinate ``i``, which is what matplotlib's ``imshow`` uses, so a selection
made on screen maps back without an off-by-half.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Iterator, Mapping

import numpy as np

#: Role names the v1 metrics ask for.
SIGNAL = "signal"
NOISE = "noise"


class RegionError(Exception):
    """Base class for region problems."""


class RegionOutOfBoundsError(RegionError):
    """The region does not fit the image it is being applied to."""


class EmptyRegionError(RegionError):
    """The region covers too few pixels to compute anything from."""


@dataclass(frozen=True)
class Rect:
    """Axis-aligned rectangle, in pixels. ``x``/``y`` are the top-left pixel."""

    x: int
    y: int
    width: int
    height: int

    kind = "rect"

    def __post_init__(self) -> None:
        if self.width < 1 or self.height < 1:
            raise RegionError(f"rectangle must be at least 1x1, got {self.width}x{self.height}")

    @property
    def bounds(self) -> tuple[int, int, int, int]:
        """``(x0, y0, x1, y1)``, right/bottom exclusive."""
        return self.x, self.y, self.x + self.width, self.y + self.height

    def mask(self, shape: tuple[int, int]) -> np.ndarray:
        self.validate_for(shape)
        result = np.zeros(shape, dtype=bool)
        x0, y0, x1, y1 = self.bounds
        result[y0:y1, x0:x1] = True
        return result

    def validate_for(self, shape: tuple[int, int]) -> None:
        height, width = shape
        x0, y0, x1, y1 = self.bounds
        if x0 < 0 or y0 < 0 or x1 > width or y1 > height:
            raise RegionOutOfBoundsError(
                f"rectangle {self.describe()} does not fit an image of {width}x{height} px"
            )

    def npixels(self, shape: tuple[int, int] | None = None) -> int:
        return self.width * self.height

    def describe(self) -> str:
        return f"rect(x={self.x}, y={self.y}, w={self.width}, h={self.height})"

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "x": int(self.x),
            "y": int(self.y),
            "width": int(self.width),
            "height": int(self.height),
        }


@dataclass(frozen=True)
class Ellipse:
    """Axis-aligned ellipse, in pixels, centre plus semi-axes."""

    cx: float
    cy: float
    rx: float
    ry: float

    kind = "ellipse"

    def __post_init__(self) -> None:
        if self.rx <= 0 or self.ry <= 0:
            raise RegionError(f"ellipse radii must be positive, got rx={self.rx}, ry={self.ry}")

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return self.cx - self.rx, self.cy - self.ry, self.cx + self.rx, self.cy + self.ry

    def mask(self, shape: tuple[int, int]) -> np.ndarray:
        self.validate_for(shape)
        height, width = shape
        ys = np.arange(height, dtype=np.float64)[:, None]
        xs = np.arange(width, dtype=np.float64)[None, :]
        return ((xs - self.cx) / self.rx) ** 2 + ((ys - self.cy) / self.ry) ** 2 <= 1.0

    def validate_for(self, shape: tuple[int, int]) -> None:
        height, width = shape
        x0, y0, x1, y1 = self.bounds
        if x0 < -0.5 or y0 < -0.5 or x1 > width - 0.5 or y1 > height - 0.5:
            raise RegionOutOfBoundsError(
                f"ellipse {self.describe()} does not fit an image of {width}x{height} px"
            )

    def npixels(self, shape: tuple[int, int] | None = None) -> int:
        if shape is None:
            return int(round(np.pi * self.rx * self.ry))
        return int(self.mask(shape).sum())

    def describe(self) -> str:
        return f"ellipse(cx={self.cx:g}, cy={self.cy:g}, rx={self.rx:g}, ry={self.ry:g})"

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "cx": float(self.cx),
            "cy": float(self.cy),
            "rx": float(self.rx),
            "ry": float(self.ry),
        }


Region = Rect | Ellipse

_KINDS: dict[str, type[Rect] | type[Ellipse]] = {"rect": Rect, "ellipse": Ellipse}
#: Minimum pixels in a region — the noise metric uses ddof=1.
MIN_PIXELS = 2


def region_from_dict(payload: Mapping[str, Any]) -> Region:
    try:
        kind = payload["kind"]
        cls = _KINDS[kind]
    except (KeyError, TypeError) as exc:
        raise RegionError(f"unknown or missing region kind in {payload!r}") from exc
    fields = {k: v for k, v in payload.items() if k != "kind"}
    try:
        return cls(**fields)  # type: ignore[arg-type]
    except (TypeError, ValueError) as exc:
        raise RegionError(f"malformed {kind} region: {exc}") from exc


def region_from_extents(
    kind: str, extents: tuple[float, float, float, float], shape: tuple[int, int]
) -> Region:
    """Build a region from a selector's ``(xmin, xmax, ymin, ymax)``.

    Coordinates are snapped to whole pixels and clipped to the image, so a drag
    that overshoots the edge still produces a usable region instead of an error.
    """
    height, width = shape
    xmin, xmax, ymin, ymax = extents
    if xmin > xmax:
        xmin, xmax = xmax, xmin
    if ymin > ymax:
        ymin, ymax = ymax, ymin

    if kind == "rect":
        x0 = int(np.clip(round(xmin), 0, width - 1))
        y0 = int(np.clip(round(ymin), 0, height - 1))
        x1 = int(np.clip(round(xmax), 0, width - 1))
        y1 = int(np.clip(round(ymax), 0, height - 1))
        return Rect(x=x0, y=y0, width=max(1, x1 - x0 + 1), height=max(1, y1 - y0 + 1))

    if kind == "ellipse":
        x0 = float(np.clip(xmin, -0.5, width - 0.5))
        x1 = float(np.clip(xmax, -0.5, width - 0.5))
        y0 = float(np.clip(ymin, -0.5, height - 0.5))
        y1 = float(np.clip(ymax, -0.5, height - 0.5))
        rx = max((x1 - x0) / 2.0, 0.5)
        ry = max((y1 - y0) / 2.0, 0.5)
        return Ellipse(cx=(x0 + x1) / 2.0, cy=(y0 + y1) / 2.0, rx=rx, ry=ry)

    raise RegionError(f"unknown region kind {kind!r}")


@dataclass(frozen=True)
class RegionSet:
    """The signal and noise regions, plus where they were drawn.

    ``image_shape`` and ``source_file`` are provenance: they say which image the
    boxes were drawn on, so a mismatch later can be explained rather than just
    failing.
    """

    signal: Region
    noise: Region
    image_shape: tuple[int, int]
    source_file: str | None = None

    def named(self) -> dict[str, Region]:
        return {SIGNAL: self.signal, NOISE: self.noise}

    def __iter__(self) -> Iterator[tuple[str, Region]]:
        return iter(self.named().items())

    def validate_for(self, shape: tuple[int, int]) -> None:
        """Check both regions fit *shape* and are big enough to measure."""
        for name, region in self.named().items():
            try:
                region.validate_for(shape)
            except RegionOutOfBoundsError as exc:
                raise RegionOutOfBoundsError(f"{name} region: {exc}") from None
            if region.npixels(shape) < MIN_PIXELS:
                raise EmptyRegionError(
                    f"{name} region covers fewer than {MIN_PIXELS} pixels: {region.describe()}"
                )

    def extract(self, image: np.ndarray) -> dict[str, np.ndarray]:
        """Pixel samples per role, flattened to 1-D float64.

        Flattened on purpose: mean and standard deviation do not care about
        shape, and an ellipse has no rectangular sub-image to hand back. A
        metric that needs 2-D geometry would need more than this.
        """
        if image.ndim != 2:
            raise RegionError(f"expected a 2-D image, got shape {image.shape}")
        shape = (int(image.shape[0]), int(image.shape[1]))
        self.validate_for(shape)
        samples = {}
        for name, region in self.named().items():
            samples[name] = np.asarray(image, dtype=np.float64)[region.mask(shape)].ravel()
        return samples

    def describe(self) -> str:
        return f"signal {self.signal.describe()} · noise {self.noise.describe()}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": 1,
            "image_shape": [int(self.image_shape[0]), int(self.image_shape[1])],
            "source_file": self.source_file,
            "regions": {name: region.to_dict() for name, region in self.named().items()},
        }


def region_set_from_dict(payload: Mapping[str, Any]) -> RegionSet:
    if not isinstance(payload, Mapping):
        raise RegionError("region set must be a JSON object")
    regions = payload.get("regions")
    if not isinstance(regions, Mapping) or SIGNAL not in regions or NOISE not in regions:
        raise RegionError(f"region set must define both {SIGNAL!r} and {NOISE!r}")
    shape = payload.get("image_shape")
    if (
        not isinstance(shape, (list, tuple))
        or len(shape) != 2
        or not all(isinstance(v, int) and v > 0 for v in shape)
    ):
        raise RegionError("region set must record a valid image_shape [height, width]")
    source = payload.get("source_file")
    return RegionSet(
        signal=region_from_dict(regions[SIGNAL]),
        noise=region_from_dict(regions[NOISE]),
        image_shape=(int(shape[0]), int(shape[1])),
        source_file=source if isinstance(source, str) else None,
    )


def regions_hash(region_set: RegionSet | None) -> str:
    """Stable hash of the geometry alone — the sixth element of the cache key.

    Provenance (source file, reference shape) is deliberately excluded: the key
    must change exactly when the extracted pixels would change, and not when
    the same boxes were merely drawn on a different file.
    """
    if region_set is None:
        return ""
    payload = json.dumps(
        {name: region.to_dict() for name, region in region_set.named().items()},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
