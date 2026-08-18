"""Per-directory persistence of the selected areas.

The areas belong to an experiment, not to the user: they are pixel coordinates
that only mean something for one set of images. So they live in a sidecar file
next to the data instead of in the global config — the selection travels with
the directory, survives moving between machines, and two experiments never
inherit each other's boxes.

A missing or corrupt sidecar is a non-event, exactly like the config file: the
app carries on with no areas selected.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .regions import RegionError, RegionSet, region_set_from_dict

SIDECAR_NAME = ".tif-analyzer-rois.json"


def sidecar_path(directory: Path) -> Path:
    return Path(directory) / SIDECAR_NAME


@dataclass(frozen=True)
class LoadedRegions:
    """What was found on disk, plus a note when it could not be used."""

    regions: RegionSet | None = None
    notice: str | None = None


def load(directory: Path, expected_shape: tuple[int, int] | None = None) -> LoadedRegions:
    """Read the sidecar for *directory*.

    When *expected_shape* is given and does not match the shape the areas were
    drawn on, the areas are **not** applied and a notice explains why — a silent
    mismatch would produce numbers from the wrong pixels.
    """
    path = sidecar_path(directory)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return LoadedRegions()
    except (OSError, ValueError) as exc:
        return LoadedRegions(notice=f"ignoring unreadable {SIDECAR_NAME} ({exc})")

    try:
        regions = region_set_from_dict(payload)
    except RegionError as exc:
        return LoadedRegions(notice=f"ignoring malformed {SIDECAR_NAME} ({exc})")

    if expected_shape is not None and tuple(regions.image_shape) != tuple(expected_shape):
        stored = "x".join(str(v) for v in reversed(regions.image_shape))
        found = "x".join(str(v) for v in reversed(expected_shape))
        return LoadedRegions(
            notice=(
                f"saved areas were drawn on a {stored} px image but these are {found} px — "
                "not applied, select them again"
            )
        )
    return LoadedRegions(regions=regions)


def save(directory: Path, regions: RegionSet) -> tuple[Path | None, str | None]:
    """Write the sidecar. Returns ``(path, notice)``; a write failure is
    reported, never fatal — losing persistence must not lose the selection."""
    path = sidecar_path(directory)
    try:
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(regions.to_dict(), indent=2) + "\n", encoding="utf-8")
        tmp.replace(path)
        return path, None
    except OSError as exc:
        return None, f"could not save areas to {path} ({exc})"


def clear(directory: Path) -> bool:
    """Delete the sidecar. Returns whether a file was removed."""
    try:
        sidecar_path(directory).unlink()
        return True
    except (FileNotFoundError, OSError):
        return False
