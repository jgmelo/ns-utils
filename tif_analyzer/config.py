"""Persisted user configuration.

``$XDG_CONFIG_HOME/tif-analyzer/config.json``, falling back to
``~/.config/tif-analyzer/config.json``; ``%APPDATA%`` on Windows. No extra
dependency.

A missing or corrupt config file is a non-event: defaults are used and the app
carries on. The file is written **only after a directory has been validated**,
so a typo never becomes the next session's default.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

from .discovery import DEFAULT_GLOB, DEFAULT_PATTERN

APP_DIR_NAME = "tif-analyzer"
CONFIG_FILE_NAME = "config.json"


@dataclass
class Config:
    last_directory: str | None = None
    last_pattern: str = DEFAULT_PATTERN
    last_metric_key: str = "std"
    last_glob: str = DEFAULT_GLOB
    allow_duplicates: bool = False


def config_dir() -> Path:
    if sys.platform.startswith("win"):
        base = os.environ.get("APPDATA") or (Path.home() / "AppData" / "Roaming")
    else:
        base = os.environ.get("XDG_CONFIG_HOME") or (Path.home() / ".config")
    return Path(base) / APP_DIR_NAME


def config_path() -> Path:
    return config_dir() / CONFIG_FILE_NAME


def load() -> Config:
    """Load the config, tolerating anything the file might have become."""
    path = config_path()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return Config()
    if not isinstance(raw, dict):
        return Config()

    config = Config()
    for key, value in raw.items():
        if not hasattr(config, key):
            continue
        default = getattr(config, key)
        if key == "last_directory":
            if isinstance(value, str) or value is None:
                config.last_directory = value
        elif isinstance(default, bool):
            if isinstance(value, bool):
                setattr(config, key, value)
        elif isinstance(value, str) and value:
            setattr(config, key, value)
    return config


def save(config: Config) -> Path | None:
    """Persist *config*. Returns the path written, or ``None`` on failure —
    losing the convenience of a remembered directory must never break a run."""
    path = config_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(asdict(config), indent=2) + "\n", encoding="utf-8")
        tmp.replace(path)
        return path
    except OSError:
        return None


@dataclass
class StartupConfig:
    """What the app resolved at startup, plus a note if a fallback happened."""

    config: Config
    directory: Path
    notice: str | None = None


def resolve_startup(config: Config | None = None, cwd: Path | None = None) -> StartupConfig:
    """Validate the remembered directory, falling back to the CWD with an
    explicit notice rather than crashing or silently pretending it worked."""
    config = config if config is not None else load()
    cwd = Path(cwd) if cwd is not None else Path.cwd()

    remembered = config.last_directory
    if remembered:
        candidate = Path(remembered).expanduser()
        if candidate.is_dir():
            return StartupConfig(config, candidate.resolve())
        return StartupConfig(
            config,
            cwd.resolve(),
            notice=f"remembered directory no longer exists: {candidate} — using {cwd.resolve()}",
        )
    return StartupConfig(config, cwd.resolve())
