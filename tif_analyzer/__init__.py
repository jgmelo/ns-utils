"""TIF Series Analyzer.

Scans a directory of ``.tif`` images, extracts a numeric parameter from each
filename, computes a metric per image and plots metric-vs-parameter.

Nothing outside :mod:`tif_analyzer.ui` reads from or writes to the terminal:
the analysis layer is importable from a script or a notebook with no UI present.
"""

from __future__ import annotations

__version__ = "1.0.0"

__all__ = ["__version__"]
