"""Removing the fixed pattern from a frame.

Two modes, both leaving the frame's own background level untouched:

* **reference** — ``S − α·R_aligned``. The reference is aligned to the signal's
  stripes by a sub-pixel Fourier shift, scaled by least squares over the
  background pixels, and optionally restricted to the stripe band first.
* **profile** — ``S − profile(x)``. The column-wise median is itself the fixed
  pattern; subtracting it needs no second channel and injects almost no noise,
  because the profile is a median over hundreds of lines. It only works when
  the pattern is locked to the line, and it removes any genuine vertical
  structure along with it.

Nothing here decides whether subtraction is a good idea — the caller has
already decided that. What it does refuse to do is stay quiet when the result
is worse than the input.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .dsp import MAD_TO_SIGMA, window
from .mask import MaskResult, build_mask
from .params import InterferenceParams

MODES = ("reference", "profile")


class SubtractionError(Exception):
    """The subtraction cannot be computed as asked."""


# --------------------------------------------------------------------------- #
# building blocks
# --------------------------------------------------------------------------- #


def fourier_shift(image: np.ndarray, shift_px: float) -> np.ndarray:
    """Shift every line by *shift_px* columns, sub-pixel, via a phase ramp.

    Exact for band-limited data, and it invents no intermediate values the way
    linear interpolation does. Positive *shift_px* moves the content towards
    higher column indices.

    The transform is circular: content leaving one edge reappears at the other.
    For a stripe pattern that spans the whole line this is what you want; the
    edge columns are still the least trustworthy part of the result.
    """
    data = np.asarray(image, dtype=np.float64)
    if shift_px == 0.0:
        return data.copy()
    n = data.shape[-1]
    spectrum = np.fft.rfft(data, axis=-1)
    frequencies = np.fft.rfftfreq(n, d=1.0)
    ramp = np.exp(-2j * np.pi * frequencies * shift_px)
    if n % 2 == 0:
        # The Nyquist bin cannot carry a phase for real output; letting it try
        # leaks a half-sample error into every line.
        ramp[-1] = np.real(ramp[-1])
    return np.fft.irfft(spectrum * ramp, n=n, axis=-1)


def alignment_shift(phase_offset_px: float) -> float:
    """Shift to apply to the reference, from the measured phase offset.

    ``phase_offset_px`` is *signal minus reference*: positive means the signal's
    stripes sit ahead. To bring the reference's stripes onto the signal's, the
    reference has to move by the **opposite** amount, so the shift is negated.

    This is a one-character mistake with a silent failure mode — the wrong sign
    leaves the stripe untouched and produces a plausible-looking scale — so it
    lives here, verified by a test, instead of at each call site.
    """
    return -float(phase_offset_px)


def bandlimit_lines(image: np.ndarray, low_bin: int, high_bin: int) -> np.ndarray:
    """Keep only rfft bins ``[low_bin, high_bin]`` of every line.

    This is what keeps the reference's broadband noise out of the corrected
    frame: only the part of the reference that carries the stripe is subtracted.
    """
    data = np.asarray(image, dtype=np.float64)
    spectrum = np.fft.rfft(data, axis=-1)
    keep = np.zeros_like(spectrum)
    low = max(0, low_bin)
    high = min(spectrum.shape[-1] - 1, high_bin)
    if high >= low:
        keep[..., low : high + 1] = spectrum[..., low : high + 1]
    return np.fft.irfft(keep, n=data.shape[-1], axis=-1)


def least_squares_scale(
    signal: np.ndarray, reference: np.ndarray, mask: np.ndarray | None = None
) -> float:
    """``α = ⟨S,R⟩ / ⟨R,R⟩`` over the pixels *mask* keeps.

    Both sides are used as given, so the caller is responsible for having
    removed their means — otherwise the DC levels dominate the fit.

    Excluding the beads matters: a bright grain present in the signal and
    absent from the reference pulls α up, and the subtraction then starts
    eating real signal.
    """
    signal = np.asarray(signal, dtype=np.float64)
    reference = np.asarray(reference, dtype=np.float64)
    if mask is not None:
        signal = signal[mask]
        reference = reference[mask]
    denominator = float(np.dot(reference.ravel(), reference.ravel()))
    if denominator <= 0:
        raise SubtractionError("the reference has no variance to scale — nothing to subtract")
    return float(np.dot(signal.ravel(), reference.ravel()) / denominator)


def measure_stripe(
    image: np.ndarray,
    frequency: float,
    params: InterferenceParams,
    mask: np.ndarray | None = None,
) -> float:
    """Median over lines of the stripe amplitude at *frequency*.

    Measured **per line**, not on the column-median profile. That choice is not
    cosmetic: profile mode subtracts the column median, so the column median of
    its output is identically zero and a profile-based metric would always
    report a perfect 100% removal — measuring exactly what was subtracted. Per
    line, both modes have to earn the number.

    Beads are filled with their line's background median before projecting. A
    bright object is not narrow in frequency: it carries energy at the stripe
    frequency too, with a phase set by where it sits along the line, and it can
    add to or cancel the stripe. Leaving beads in makes the before/after
    numbers depend on where the sample happened to be.
    """
    data = np.asarray(image, dtype=np.float64)
    n_columns = data.shape[1]
    if mask is not None:
        data = data.copy()
        for index in range(data.shape[0]):
            keep = mask[index]
            if keep.all() or not keep.any():
                continue
            data[index, ~keep] = float(np.median(data[index, keep]))
    centred = data - data.mean(axis=1, keepdims=True)
    if frequency <= 0:
        return float(np.median(np.std(centred, axis=1)))
    weights = window(params.window, n_columns)
    phasor = np.exp(-2j * np.pi * frequency * np.arange(n_columns))
    projections = centred @ (weights * phasor)
    return float(np.median(2.0 * np.abs(projections) / float(np.sum(weights))))


def robust_sigma(image: np.ndarray, mask: np.ndarray | None = None) -> float:
    """Pixel noise, with the fixed pattern taken out first.

    Each column's own median is removed before measuring, so the number is the
    scatter *around* the pattern rather than the scatter of the pattern itself.
    Without that, the stripe dominates the sigma and removing it looks like a
    huge noise improvement — flattering, and wrong.
    """
    data = np.asarray(image, dtype=np.float64)
    detrended = data - np.median(data, axis=0)[None, :]
    sample = detrended[mask] if mask is not None else detrended.ravel()
    if sample.size == 0:
        return float("nan")
    median = float(np.median(sample))
    return float(np.median(np.abs(sample - median)) * MAD_TO_SIGMA)


# --------------------------------------------------------------------------- #
# plan and result
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class SubtractionPlan:
    """What to do. Every field is a decision the caller can inspect."""

    mode: str = "reference"
    #: Restrict the reference to the stripe band before subtracting, so its own
    #: broadband noise is not injected into the corrected frame.
    band_limited: bool = True
    #: Fit the scale (and use it) line by line instead of once for the frame.
    per_line: bool = False
    #: Override the fitted scale, or ``None`` to fit it.
    scale: float | None = None
    #: Override the alignment shift, or ``None`` to take it from the phase offset.
    shift_px: float | None = None

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise SubtractionError(f"unknown mode {self.mode!r}; expected one of {MODES}")


@dataclass
class SubtractionResult:
    corrected: np.ndarray
    mode: str
    scale: float
    shift_px: float
    per_line: bool
    band_limited: bool
    band: tuple[int, int] | None
    frequency: float
    stripe_before: float
    stripe_after: float
    sigma_before: float
    sigma_after: float
    scales_per_line: np.ndarray | None = None
    warnings: tuple[str, ...] = field(default_factory=tuple)

    @property
    def removed_fraction(self) -> float:
        if self.stripe_before <= 0:
            return float("nan")
        return 1.0 - self.stripe_after / self.stripe_before

    @property
    def noise_change(self) -> float:
        """Relative change in background sigma. Positive means noise went up."""
        if not np.isfinite(self.sigma_before) or self.sigma_before <= 0:
            return float("nan")
        return self.sigma_after / self.sigma_before - 1.0

    @property
    def improved(self) -> bool:
        return self.stripe_after < self.stripe_before


# --------------------------------------------------------------------------- #
# the two modes
# --------------------------------------------------------------------------- #


def _prepare_reference(
    reference: np.ndarray,
    shift_px: float,
    band: tuple[int, int] | None,
    band_limited: bool,
) -> np.ndarray:
    """Align, optionally band-limit, and remove the mean.

    The mean has to go: subtracting a reference that carries its own DC level
    would move the corrected frame's background, and every count in this
    pipeline is meant to stay comparable to the original.
    """
    prepared = fourier_shift(reference, shift_px)
    if band_limited and band is not None:
        prepared = bandlimit_lines(prepared, band[0], band[1])
    else:
        prepared = prepared - prepared.mean()
    return prepared


def subtract(
    signal: np.ndarray,
    plan: SubtractionPlan,
    reference: np.ndarray | None = None,
    mask_result: MaskResult | None = None,
    band: tuple[int, int] | None = None,
    frequency: float = 0.0,
    params: InterferenceParams | None = None,
) -> SubtractionResult:
    """Remove the fixed pattern from *signal* and measure what changed."""
    params = params or InterferenceParams()
    data = np.asarray(signal, dtype=np.float64)
    if data.ndim != 2:
        raise SubtractionError(f"expected a 2-D frame, got shape {data.shape}")

    mask_result = mask_result or build_mask(data, params)
    mask = mask_result.mask
    warnings: list[str] = []

    stripe_before = measure_stripe(data, frequency, params, mask)
    sigma_before = robust_sigma(data, mask)

    if plan.mode == "profile":
        corrected, scale, shift = _subtract_profile(data, mask, params)
        scales_per_line = None
        band_used = None
        band_limited = False
    else:
        if reference is None:
            raise SubtractionError("reference mode needs a reference frame")
        reference = np.asarray(reference, dtype=np.float64)
        if reference.shape != data.shape:
            raise SubtractionError(
                f"reference is {reference.shape} but the signal is {data.shape}"
            )
        corrected, scale, shift, scales_per_line = _subtract_reference(
            data, reference, mask, plan, band
        )
        band_used = band if plan.band_limited else None
        band_limited = plan.band_limited

    stripe_after = measure_stripe(corrected, frequency, params, mask)
    sigma_after = robust_sigma(corrected, mask)

    if stripe_after >= stripe_before:
        warnings.append(
            f"the stripe did not shrink ({stripe_before:.4g} → {stripe_after:.4g} counts). "
            "Check the alignment sign and the scale before using this frame."
        )
    if np.isfinite(sigma_before) and sigma_before > 0 and sigma_after > 1.2 * sigma_before:
        warnings.append(
            f"background noise rose {sigma_after / sigma_before - 1:.0%} "
            f"({sigma_before:.4g} → {sigma_after:.4g} counts) — the reference brought its own "
            "noise in; try the band-limited mode"
        )

    return SubtractionResult(
        corrected=corrected,
        mode=plan.mode,
        scale=scale,
        shift_px=shift,
        per_line=plan.per_line,
        band_limited=band_limited,
        band=band_used,
        frequency=frequency,
        stripe_before=stripe_before,
        stripe_after=stripe_after,
        sigma_before=sigma_before,
        sigma_after=sigma_after,
        scales_per_line=scales_per_line,
        warnings=tuple(warnings),
    )


def _subtract_profile(
    data: np.ndarray, mask: np.ndarray, params: InterferenceParams
) -> tuple[np.ndarray, float, float]:
    """``S − profile(x)``, with the profile taken over background pixels only."""
    n_lines, n_columns = data.shape
    profile = np.empty(n_columns, dtype=np.float64)
    for column in range(n_columns):
        kept = data[mask[:, column], column]
        if kept.size < params.min_background_rows_per_column:
            profile[column] = np.median(data[:, column])
        else:
            profile[column] = np.median(kept)
    profile = profile - profile.mean()
    return data - profile[None, :], 1.0, 0.0


def _subtract_reference(
    data: np.ndarray,
    reference: np.ndarray,
    mask: np.ndarray,
    plan: SubtractionPlan,
    band: tuple[int, int] | None,
) -> tuple[np.ndarray, float, float, np.ndarray | None]:
    shift = 0.0 if plan.shift_px is None else float(plan.shift_px)
    prepared = _prepare_reference(reference, shift, band, plan.band_limited)
    centred = data - data.mean()

    if not plan.per_line:
        scale = (
            float(plan.scale)
            if plan.scale is not None
            else least_squares_scale(centred, prepared, mask)
        )
        return data - scale * prepared, scale, shift, None

    # Per line: each fit sees one line of data, so it follows amplitude that
    # varies along the acquisition at the cost of a noisier estimate.
    corrected = data.copy()
    scales = np.zeros(data.shape[0], dtype=np.float64)
    for index in range(data.shape[0]):
        line_mask = mask[index]
        line_signal = centred[index]
        line_reference = prepared[index]
        if plan.scale is not None:
            scale = float(plan.scale)
        elif np.count_nonzero(line_mask) < 8:
            scale = 0.0  # nothing trustworthy to fit against on this line
        else:
            try:
                scale = least_squares_scale(line_signal, line_reference, line_mask)
            except SubtractionError:
                scale = 0.0
        scales[index] = scale
        corrected[index] = data[index] - scale * line_reference
    return corrected, float(np.median(scales)), shift, scales
