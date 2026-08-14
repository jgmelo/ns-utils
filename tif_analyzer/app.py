"""Entry point: wires configuration, state and the UI together."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__, config as config_mod, discovery, metrics
from .loading import PAGE_POLICIES, DEFAULT_PAGE_POLICY
from .state import AppState
from .ui.keys import KeyReader
from .ui.menu import MenuApp


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="tif-analyzer",
        description="Scan a directory of TIFs, compute a metric per image and plot it "
        "against a parameter read from the filename.",
    )
    parser.add_argument("directory", nargs="?", type=Path, help="directory to scan")
    parser.add_argument("-g", "--glob", help=f"glob filter (default: {discovery.DEFAULT_GLOB})")
    parser.add_argument("-p", "--pattern", help="filename regex with exactly one named group")
    parser.add_argument("-m", "--metric", help="metric key (default: std)")
    parser.add_argument(
        "--page-policy",
        choices=PAGE_POLICIES,
        help=f"multi-page TIFF policy (default: {DEFAULT_PAGE_POLICY})",
    )
    parser.add_argument(
        "--allow-duplicates",
        action="store_true",
        help="do not treat duplicate parameter values as an error",
    )
    parser.add_argument(
        "--no-config",
        action="store_true",
        help="ignore and do not write the persisted configuration",
    )
    parser.add_argument(
        "--line-input",
        action="store_true",
        help="use key+Enter instead of single-key input",
    )
    parser.add_argument("--no-clear", action="store_true", help="do not clear the screen")
    parser.add_argument("--version", action="version", version=f"tif-analyzer {__version__}")
    return parser


def build_state(args: argparse.Namespace) -> tuple[AppState, config_mod.Config, list[str]]:
    notices: list[str] = []
    config = config_mod.Config() if args.no_config else config_mod.load()
    startup = config_mod.resolve_startup(config)
    if startup.notice and not args.no_config:
        notices.append(startup.notice)

    directory = startup.directory
    if args.directory is not None:
        candidate = args.directory.expanduser()
        if candidate.is_dir():
            directory = candidate.resolve()
        else:
            notices.append(f"{candidate} is not a directory — using {directory}")

    metric_key = args.metric or config.last_metric_key
    if metric_key not in metrics.REGISTRY:
        notices.append(f"unknown metric {metric_key!r} — falling back to 'std'")
        metric_key = "std"

    pattern = args.pattern or config.last_pattern
    try:
        discovery.validate_pattern(pattern)
    except discovery.PatternError as exc:
        notices.append(f"stored pattern rejected ({exc}) — using the default")
        pattern = discovery.DEFAULT_PATTERN

    state = AppState(
        directory=directory,
        glob=args.glob or config.last_glob,
        pattern=pattern,
        metric_key=metric_key,
        allow_duplicates=bool(args.allow_duplicates or config.allow_duplicates),
        page_policy=args.page_policy or DEFAULT_PAGE_POLICY,
    )
    return state, config, notices


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    state, config, notices = build_state(args)
    app = MenuApp(
        state,
        config,
        reader=KeyReader(force_line_mode=args.line_input),
        persist=not args.no_config,
        clear=not args.no_clear,
    )
    app.notices.extend(notices)
    return app.loop()


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
