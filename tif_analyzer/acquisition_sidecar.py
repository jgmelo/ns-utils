"""Per-directory persistence of acquisition metadata.

Line rate, resolution and flyback describe how one set of images was acquired,
not how the user likes to work — so they follow the directory, in a sidecar,
for the same reason the selected areas do. Retyping the line rate every session
would be the fastest way to end up with a typo in a reported frequency.

A missing or corrupt sidecar is a non-event: the fields simply start empty.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .interference.params import Acquisition, AcquisitionError

SIDECAR_NAME = ".tif-analyzer-acquisition.json"


def sidecar_path(directory: Path) -> Path:
    return Path(directory) / SIDECAR_NAME


@dataclass(frozen=True)
class LoadedAcquisition:
    acquisition: Acquisition | None = None
    notice: str | None = None


def to_dict(acquisition: Acquisition) -> dict:
    return {
        "version": 1,
        "line_rate_hz": acquisition.line_rate_hz,
        "resolution": acquisition.resolution,
        "flyback_fraction": acquisition.flyback_fraction,
        "frame_includes_flyback": acquisition.frame_includes_flyback,
    }


def from_dict(payload: dict) -> Acquisition:
    """Build an :class:`Acquisition`, letting its own validation reject nonsense."""
    if not isinstance(payload, dict):
        raise AcquisitionError("acquisition sidecar must be a JSON object")
    line_rate = payload.get("line_rate_hz")
    resolution = payload.get("resolution")
    flyback = payload.get("flyback_fraction", 0.0)
    includes = payload.get("frame_includes_flyback", False)
    return Acquisition(
        line_rate_hz=float(line_rate) if isinstance(line_rate, (int, float)) else None,
        resolution=int(resolution) if isinstance(resolution, int) else None,
        flyback_fraction=float(flyback) if isinstance(flyback, (int, float)) else 0.0,
        frame_includes_flyback=bool(includes),
    )


def load(directory: Path) -> LoadedAcquisition:
    path = sidecar_path(directory)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return LoadedAcquisition()
    except (OSError, ValueError) as exc:
        return LoadedAcquisition(notice=f"ignoring unreadable {SIDECAR_NAME} ({exc})")
    try:
        return LoadedAcquisition(acquisition=from_dict(payload))
    except (AcquisitionError, TypeError, ValueError) as exc:
        return LoadedAcquisition(notice=f"ignoring malformed {SIDECAR_NAME} ({exc})")


def save(directory: Path, acquisition: Acquisition) -> tuple[Path | None, str | None]:
    """Write the sidecar. A failure is reported, never fatal."""
    path = sidecar_path(directory)
    try:
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(to_dict(acquisition), indent=2) + "\n", encoding="utf-8")
        tmp.replace(path)
        return path, None
    except OSError as exc:
        return None, f"could not save acquisition metadata to {path} ({exc})"
