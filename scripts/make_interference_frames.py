#!/usr/bin/env python3
"""Write synthetic frames for trying the interference characterisation out.

    python scripts/make_interference_frames.py /tmp/if --drift 0.3 --beads
    tif-interference /tmp/if/signal.tif -r /tmp/if/reference.tif --line-rate 1000

The stripes have a known period, phase and drift, so the report can be checked
against what was planted.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import tifffile

# Runnable straight from a checkout, without installing the package first.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tif_analyzer.interference.synthetic import FrameSpec, make_pair  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--period", type=float, default=13.7, help="stripe period in pixels")
    parser.add_argument("--amplitude", type=float, default=45.0, help="stripe amplitude in counts")
    parser.add_argument("--noise", type=float, default=6.0, help="background sigma in counts")
    parser.add_argument("--drift", type=float, default=0.0, help="phase drift in degrees per line")
    parser.add_argument("--phase-offset", type=float, default=25.0, help="reference phase lead")
    parser.add_argument("--amplitude-ratio", type=float, default=0.6, help="reference/signal")
    parser.add_argument(
        "--reference-noise", type=float, help="reference background sigma (default: same as signal)"
    )
    parser.add_argument("--beads", action="store_true", help="add bright beads to the signal frame")
    parser.add_argument("--lines", type=int, default=256)
    parser.add_argument("--columns", type=int, default=512)
    args = parser.parse_args()

    beads = (
        tuple((40 + i * 30, 60 + i * 70, 6.0, 9000.0) for i in range(6)) if args.beads else ()
    )
    spec = FrameSpec(
        shape=(args.lines, args.columns),
        period_px=args.period,
        amplitude=args.amplitude,
        noise_sigma=args.noise,
        drift_deg_per_line=args.drift,
        beads=beads,
    )
    signal, reference, truth = make_pair(
        spec,
        phase_offset_deg=args.phase_offset,
        amplitude_ratio=args.amplitude_ratio,
        reference_noise_sigma=args.reference_noise,
    )

    directory = args.directory.expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    tifffile.imwrite(directory / "signal.tif", signal)
    tifffile.imwrite(directory / "reference.tif", reference)

    print(f"wrote signal.tif and reference.tif to {directory}")
    print(
        f"planted: period {truth.period_px} px, amplitude {truth.amplitude} counts, "
        f"drift {truth.drift_deg_per_line}°/line, reference phase lead {args.phase_offset}°"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
