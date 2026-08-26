"""Writing a corrected frame, with a record of how it was made.

float32 on purpose. The subtraction preserves the frame's own background level,
so negatives are not the common case — but the corrected values are fractional
(``α`` is), and uint16 would round every pixel. With a background sigma of a few
counts, quantising to integers throws away part of the noise distribution that a
signal-to-noise measurement is about to measure. And where a frame does dip
below zero, uint16 would clip it away silently.

A corrected image without a record of how it was corrected is a trap set for
whoever opens it six months later, so the parameters go both into the TIFF
description and into a JSON sidecar.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tifffile

from .. import __version__
from .subtract import SubtractionResult

SIDECAR_SUFFIX = ".subtraction.json"


def provenance(
    result: SubtractionResult,
    signal_path: Path | None,
    reference_path: Path | None,
    extra: dict | None = None,
) -> dict:
    """Everything needed to explain, or repeat, this correction."""
    record = {
        "tool": f"tif-analyzer {__version__}",
        "operation": "fixed-pattern subtraction",
        "timestamp": datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
        "mode": result.mode,
        "signal": str(signal_path) if signal_path else None,
        "reference": str(reference_path) if reference_path else None,
        "scale": result.scale,
        "shift_px": result.shift_px,
        "per_line": result.per_line,
        "band_limited": result.band_limited,
        "band_bins": list(result.band) if result.band else None,
        "frequency_cycles_px": result.frequency,
        "period_px": (1.0 / result.frequency) if result.frequency > 0 else None,
        "stripe_before": result.stripe_before,
        "stripe_after": result.stripe_after,
        "stripe_removed_fraction": result.removed_fraction,
        "noise_before": result.sigma_before,
        "noise_after": result.sigma_after,
        "noise_change_fraction": result.noise_change,
        "warnings": list(result.warnings),
        "dtype": "float32",
        "note": (
            "float32: the corrected values are fractional and may be negative; "
            "the frame's original background level is preserved"
        ),
    }
    if extra:
        record.update(extra)
    return record


def default_output_path(signal_path: Path, mode: str) -> Path:
    suffix = "-corrected" if mode == "reference" else "-deprofiled"
    return signal_path.with_name(f"{signal_path.stem}{suffix}.tif")


def write_corrected(
    path: Path,
    result: SubtractionResult,
    signal_path: Path | None = None,
    reference_path: Path | None = None,
    extra: dict | None = None,
) -> tuple[Path, Path | None]:
    """Write the corrected frame and its sidecar. Returns both paths."""
    path = Path(path).expanduser()
    if path.suffix.lower() not in (".tif", ".tiff"):
        path = path.with_suffix(".tif")
    path.parent.mkdir(parents=True, exist_ok=True)

    record = provenance(result, signal_path, reference_path, extra)
    tifffile.imwrite(
        path,
        np.asarray(result.corrected, dtype=np.float32),
        description=json.dumps(record),
    )

    sidecar = path.with_name(path.stem + SIDECAR_SUFFIX)
    try:
        sidecar.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    except OSError:
        return path, None
    return path, sidecar


def read_provenance(path: Path) -> dict | None:
    """Read back what a corrected frame records about itself."""
    path = Path(path)
    sidecar = path.with_name(path.stem + SIDECAR_SUFFIX)
    for candidate in (sidecar,):
        try:
            return json.loads(candidate.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
    try:
        with tifffile.TiffFile(path) as tif:
            description = tif.pages[0].description
        return json.loads(description) if description else None
    except (OSError, ValueError, KeyError, tifffile.TiffFileError):
        return None
