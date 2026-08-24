"""Fixed-pattern interference characterisation for ImSpector TIFF frames.

Characterisation only — v0 never corrects anything. It answers one question:
*is cross-channel subtraction viable for this pair of frames?*

This is a single-frame tool, so it lives beside the series pipeline rather than
inside it: it shares :mod:`tif_analyzer.loading` and nothing else. Its output is
a structured result object plus a plain-text report; no plotting.

Axis convention (confirmed at design time): each **row** is a scan line and the
column index is time within the line, so the interference shows up as vertical
stripes and the fixed-pattern profile is the column-wise median. The frame is
checked against the other axis and warns loudly if the pattern is stronger
there — analysing the wrong axis yields perfectly plausible, wrong numbers.
"""

from __future__ import annotations

from .analyzer import InterferenceResult, characterise, characterise_arrays
from .params import Acquisition, InterferenceParams
from .report import render_report

__all__ = [
    "Acquisition",
    "InterferenceParams",
    "InterferenceResult",
    "characterise",
    "characterise_arrays",
    "render_report",
]
