"""Command-line entry point: ``tif-interference``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .. import __version__
from ..loading import DEFAULT_PAGE_POLICY, PAGE_POLICIES, LoadError
from .analyzer import characterise
from .frames import FrameError
from .params import Acquisition, AcquisitionError, InterferenceParams
from .report import render_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tif-interference",
        description=(
            "Characterise fixed-pattern interference in an ImSpector TIFF frame and report "
            "whether cross-channel subtraction is viable. Characterisation only — nothing is "
            "corrected."
        ),
        epilog=(
            "Assumes each row is a scan line, with time running along the columns "
            "(unidirectional). The report warns if the frame disagrees."
        ),
    )
    parser.add_argument("signal", type=Path, help="signal frame")
    parser.add_argument(
        "-r", "--reference", type=Path, help="simultaneously-acquired reference frame"
    )

    acquisition = parser.add_argument_group("acquisition metadata (optional; needed for Hz)")
    acquisition.add_argument("--line-rate", type=float, metavar="HZ", help="scan lines per second")
    acquisition.add_argument(
        "--resolution", type=int, metavar="PX", help="pixels per line (default: frame width)"
    )
    acquisition.add_argument(
        "--flyback", type=float, default=0.0, metavar="F", help="flyback fraction of the line period"
    )
    acquisition.add_argument(
        "--frame-includes-flyback",
        action="store_true",
        help="the saved columns span the whole line period, flyback included; "
        "they are then dropped from the profile and the dwell time changes accordingly",
    )

    tuning = parser.add_argument_group("tuning (defaults are provisional)")
    tuning.add_argument("--mad-k", type=float, help="bead rejection threshold, in robust sigmas")
    tuning.add_argument("--period-tolerance", type=float, help="relative period agreement")
    tuning.add_argument("--band-corr", type=float, help="band-restricted correlation threshold")
    tuning.add_argument("--raw-band-gap", type=float, help="gap that triggers the noise warning")
    tuning.add_argument(
        "--page-policy", choices=PAGE_POLICIES, default=DEFAULT_PAGE_POLICY,
        help=f"multi-page TIFF policy (default: {DEFAULT_PAGE_POLICY})",
    )

    parser.add_argument("-o", "--output", type=Path, help="write the report to a file as well")
    parser.add_argument("--version", action="version", version=f"tif-interference {__version__}")
    return parser


def params_from(args: argparse.Namespace) -> InterferenceParams:
    defaults = InterferenceParams()
    return InterferenceParams(
        mad_k=args.mad_k if args.mad_k is not None else defaults.mad_k,
        period_tolerance=(
            args.period_tolerance if args.period_tolerance is not None else defaults.period_tolerance
        ),
        band_corr_threshold=(
            args.band_corr if args.band_corr is not None else defaults.band_corr_threshold
        ),
        raw_band_gap_threshold=(
            args.raw_band_gap if args.raw_band_gap is not None else defaults.raw_band_gap_threshold
        ),
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        acquisition = Acquisition(
            line_rate_hz=args.line_rate,
            resolution=args.resolution,
            flyback_fraction=args.flyback,
            frame_includes_flyback=args.frame_includes_flyback,
        )
    except AcquisitionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2

    try:
        result = characterise(
            args.signal,
            args.reference,
            acquisition=acquisition,
            params=params_from(args),
            page_policy=args.page_policy,
        )
    except (LoadError, FrameError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    text = render_report(result)
    print(text)
    if args.output:
        args.output.write_text(text + "\n", encoding="utf-8")
        print(f"\nreport written to {args.output}")

    # Exit code carries the verdict, for scripting.
    if result.verdict.subtraction_viable is True:
        return 0
    if result.verdict.subtraction_viable is False:
        return 3
    return 4


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
