"""Terminal UI. The only package that touches stdin/stdout.

Everything below it (discovery, loading, preprocess, metrics, analysis, cache,
plotting, export) is importable and driveable from a script or a notebook with
no UI present.
"""

from __future__ import annotations

from .menu import MenuApp, run

__all__ = ["MenuApp", "run"]
