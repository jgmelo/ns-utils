"""Operation 6 — is the pattern the same on every line?

The phase of the dominant component is measured line by line, by projecting each
line onto the frequency the spectrum already found. A full periodogram per line
would be noisy — one scan line is short — but a projection onto a known
frequency is well conditioned.

What the drift means, and why it decides the whole question:

* **drift ≈ 0** — the pattern is locked to the line trigger. It is genuinely
  *fixed*: the same stripes sit in the same place in every frame, so a stored
  pattern could be subtracted.
* **drift ≠ 0** — the interference is not locked to the scan. It moves between
  frames, so only a **simultaneously acquired** reference can cancel it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .dsp import goertzel, window
from .mask import MaskResult
from .params import InterferenceParams


@dataclass(frozen=True)
class StabilityResult:
    available: bool
    reason: str | None = None
    lines_used: int = 0
    lines_total: int = 0
    amplitude_mean: float = float("nan")
    amplitude_cv: float = float("nan")
    phase_drift_deg_per_line: float = float("nan")
    phase_drift_residual_deg: float = float("nan")
    phase_drift_stderr_deg: float = float("nan")
    phase_span_deg: float = float("nan")
    amplitudes: np.ndarray | None = None
    phases_deg: np.ndarray | None = None

    @property
    def locked_to_scan(self) -> bool:
        """Whether the drift is indistinguishable from zero.

        Compared against the standard error of the fitted slope, not against the
        line-to-line scatter: with hundreds of lines, a drift far below the
        per-line noise still accumulates into a large phase span across the
        frame, and calling that "locked" would be wrong.
        """
        if not self.available or not np.isfinite(self.phase_drift_stderr_deg):
            return False
        return abs(self.phase_drift_deg_per_line) <= 3.0 * self.phase_drift_stderr_deg


def analyse(
    image: np.ndarray,
    frequency_cycles_px: float,
    mask_result: MaskResult | None = None,
    flyback_columns: int = 0,
    params: InterferenceParams | None = None,
) -> StabilityResult:
    """Per-line amplitude and phase of the component at *frequency_cycles_px*."""
    params = params or InterferenceParams()
    data = np.asarray(image, dtype=np.float64)
    if flyback_columns > 0:
        data = data[:, : data.shape[1] - flyback_columns]
    n_lines, n_columns = data.shape

    if frequency_cycles_px <= 0:
        return StabilityResult(
            available=False, reason="no dominant frequency to track", lines_total=n_lines
        )

    mask = None
    if mask_result is not None:
        mask = mask_result.mask
        if flyback_columns > 0:
            mask = mask[:, : mask.shape[1] - flyback_columns]

    weights = window(params.window, n_columns)
    weight_sum = float(np.sum(weights))

    amplitudes: list[float] = []
    phases: list[float] = []
    used_lines: list[int] = []

    for index in range(n_lines):
        line = data[index]
        if mask is not None:
            keep = mask[index]
            rejected = 1.0 - float(np.count_nonzero(keep)) / keep.size
            if rejected > params.max_line_reject_fraction:
                continue
            if rejected > 0:
                # Fill rejected pixels with this line's background median: a
                # single-bin DFT with holes in it is biased, and filling with a
                # flat value adds no energy at the frequency being measured.
                line = line.copy()
                background = float(np.median(line[keep])) if np.any(keep) else float(np.median(line))
                line[~keep] = background
        centred = line - line.mean()
        z = goertzel(centred, frequency_cycles_px, weights)
        amplitudes.append(2.0 * abs(z) / weight_sum)
        phases.append(float(np.angle(z)))
        used_lines.append(index)

    if len(used_lines) < 3:
        return StabilityResult(
            available=False,
            reason=(
                f"only {len(used_lines)} of {n_lines} lines survived the bead mask — "
                "not enough to measure stability"
            ),
            lines_used=len(used_lines),
            lines_total=n_lines,
        )

    amplitude_array = np.asarray(amplitudes, dtype=np.float64)
    unwrapped = np.unwrap(np.asarray(phases, dtype=np.float64))
    positions = np.asarray(used_lines, dtype=np.float64)

    slope, intercept = np.polyfit(positions, unwrapped, 1)
    residuals = unwrapped - (slope * positions + intercept)

    # Standard error of the slope: residual scatter spread over the lever arm.
    spread = float(np.std(positions))
    degrees_of_freedom = max(1, positions.size - 2)
    residual_sigma = float(np.sqrt(np.sum(residuals**2) / degrees_of_freedom))
    slope_stderr = (
        residual_sigma / (spread * np.sqrt(positions.size)) if spread > 0 else float("nan")
    )

    mean_amplitude = float(np.mean(amplitude_array))
    return StabilityResult(
        available=True,
        lines_used=len(used_lines),
        lines_total=n_lines,
        amplitude_mean=mean_amplitude,
        amplitude_cv=float(np.std(amplitude_array) / mean_amplitude) if mean_amplitude > 0 else float("nan"),
        phase_drift_deg_per_line=float(np.degrees(slope)),
        phase_drift_residual_deg=float(np.degrees(np.std(residuals))),
        phase_drift_stderr_deg=float(np.degrees(slope_stderr)),
        phase_span_deg=float(np.degrees(unwrapped[-1] - unwrapped[0])),
        amplitudes=amplitude_array,
        phases_deg=np.degrees(unwrapped),
    )
