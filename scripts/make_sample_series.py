#!/usr/bin/env python3
"""Generate a synthetic TIF series for trying the analyzer out.

    python scripts/make_sample_series.py /tmp/sample --powers 1 2 5 10 12.5

Noise grows with the parameter, so the std-vs-power plot is a visible trend.
Options exist for the awkward cases the app has to handle: an extra file that
does not match the pattern, a duplicate parameter value and a multi-page stack.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tifffile


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--powers", nargs="+", type=float, default=[1, 2, 5, 10, 12.5])
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--beam",
        action="store_true",
        help="add a gaussian spot whose brightness scales with the parameter, "
        "so the signal/noise areas have something to measure",
    )
    parser.add_argument("--with-unmatched", action="store_true", help="add notes.tif")
    parser.add_argument("--with-duplicate", action="store_true", help="repeat the first power")
    parser.add_argument("--with-stack", action="store_true", help="add a 3-page TIFF")
    args = parser.parse_args()

    directory = args.directory.expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    shape = (args.size, args.size)

    beam = None
    if args.beam:
        ys, xs = np.mgrid[0 : shape[0], 0 : shape[1]]
        cy, cx = shape[0] * 0.35, shape[1] * 0.65
        radius = min(shape) * 0.12
        beam = np.exp(-(((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * radius**2)))

    for index, power in enumerate(args.powers, start=1):
        noise = rng.normal(1000.0, 5.0 + 3.0 * power, size=shape)
        if beam is not None:
            noise = noise + 800.0 * power * beam
        image = np.clip(noise, 0, 65535).astype(np.uint16)
        text = f"{power:g}"
        tifffile.imwrite(directory / f"medida_{index:03d}_{text}mW.tif", image)

    if args.with_duplicate and args.powers:
        power = args.powers[0]
        image = rng.normal(1000.0, 5.0, size=shape).clip(0, 65535).astype(np.uint16)
        tifffile.imwrite(directory / f"medida_900_{power:.2f}mW.tif", image)

    if args.with_unmatched:
        tifffile.imwrite(directory / "notes.tif", np.zeros(shape, dtype=np.uint16))

    if args.with_stack:
        stack = rng.normal(1000.0, 20.0, size=(3, *shape)).clip(0, 65535).astype(np.uint16)
        tifffile.imwrite(directory / "medida_901_99mW.tif", stack, photometric="minisblack")

    print(f"wrote {len(list(directory.glob('*.tif')))} file(s) to {directory}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
