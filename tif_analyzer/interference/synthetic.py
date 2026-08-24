"""Synthetic frames with known ground truth.

The only honest way to check that a centroid, a phase drift or a cross-channel
offset is right is to plant one and try to recover it. Used by the tests, and by
``scripts/make_interference_frames.py`` to produce files you can run the CLI on.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SyntheticTruth:
    """What was planted, so a test can assert against it."""

    period_px: float
    amplitude: float
    phase_deg: float
    drift_deg_per_line: float
    offset: float
    noise_sigma: float

    @property
    def frequency_cycles_px(self) -> float:
        return 1.0 / self.period_px


@dataclass
class FrameSpec:
    """Recipe for one synthetic frame."""

    shape: tuple[int, int] = (256, 512)  # lines, columns
    period_px: float = 16.0
    amplitude: float = 40.0
    phase_deg: float = 0.0
    drift_deg_per_line: float = 0.0
    offset: float = 1000.0
    noise_sigma: float = 5.0
    beads: tuple[tuple[int, int, float, float], ...] = ()  # (line, column, radius, amplitude)
    saturate: bool = False
    dtype: str = "uint16"
    seed: int = 0
    extra_tone: tuple[float, float] | None = None  # (period_px, amplitude)


def make_frame(spec: FrameSpec) -> tuple[np.ndarray, SyntheticTruth]:
    """Build a frame with vertical stripes of a known period, phase and drift."""
    lines, columns = spec.shape
    rng = np.random.default_rng(spec.seed)

    x = np.arange(columns, dtype=np.float64)[None, :]
    line_index = np.arange(lines, dtype=np.float64)[:, None]

    phase = np.radians(spec.phase_deg + spec.drift_deg_per_line * line_index)
    stripes = spec.amplitude * np.sin(2.0 * np.pi * x / spec.period_px + phase)

    frame = spec.offset + stripes + rng.normal(0.0, spec.noise_sigma, size=(lines, columns))

    if spec.extra_tone is not None:
        period, amplitude = spec.extra_tone
        frame = frame + amplitude * np.sin(2.0 * np.pi * x / period)

    for line, column, radius, amplitude in spec.beads:
        ys = np.arange(lines, dtype=np.float64)[:, None]
        xs = np.arange(columns, dtype=np.float64)[None, :]
        frame = frame + amplitude * np.exp(
            -(((xs - column) ** 2 + (ys - line) ** 2) / (2.0 * radius**2))
        )

    dtype = np.dtype(spec.dtype)
    if dtype.kind in "ui":
        limit = np.iinfo(dtype).max
        if spec.saturate:
            frame[: max(1, lines // 20), : max(1, columns // 20)] = limit + 1000.0
        frame = np.clip(np.round(frame), 0, limit)
    array = frame.astype(dtype)

    truth = SyntheticTruth(
        period_px=spec.period_px,
        amplitude=spec.amplitude,
        phase_deg=spec.phase_deg,
        drift_deg_per_line=spec.drift_deg_per_line,
        offset=spec.offset,
        noise_sigma=spec.noise_sigma,
    )
    return array, truth


def make_pair(
    signal: FrameSpec,
    phase_offset_deg: float = 0.0,
    amplitude_ratio: float = 1.0,
    reference_noise_sigma: float | None = None,
    reference_seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray, SyntheticTruth]:
    """A signal frame plus a reference carrying the same stripes, shifted.

    Independent noise seeds on purpose: that is what makes raw correlation drop
    below band-restricted correlation, which is the situation the verdict has to
    detect and warn about.
    """
    signal_array, truth = make_frame(signal)
    reference_spec = FrameSpec(
        shape=signal.shape,
        period_px=signal.period_px,
        amplitude=signal.amplitude * amplitude_ratio,
        phase_deg=signal.phase_deg + phase_offset_deg,
        drift_deg_per_line=signal.drift_deg_per_line,
        offset=signal.offset,
        noise_sigma=signal.noise_sigma if reference_noise_sigma is None else reference_noise_sigma,
        beads=(),
        saturate=False,
        dtype=signal.dtype,
        seed=signal.seed + 1000 if reference_seed is None else reference_seed,
    )
    reference_array, _ = make_frame(reference_spec)
    return signal_array, reference_array, truth
