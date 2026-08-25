"""Operation 2 — bead mask by iterative median/MAD.

Beads are bright, so rejection is one-sided: pixels far *above* the background
median are dropped, low outliers (dead pixels) are kept, since they do not bias
a median-based profile the way a bright bead does.

The worst column matters as much as the overall fraction: one column swamped by
a bead has an unreliable median, and that single point sits in the middle of the
profile that the FFT then reads.

Known limitation, visible in the mask figure: the MAD is computed on the frame
as it is, so a strong fixed pattern inflates it and pushes the cut higher than
the pixel noise alone would. The mask is then less aggressive on a striped frame
than on a flat one. Masking on a stripe-subtracted frame would fix it, and would
also make the mask depend on the profile it is meant to clean — that circularity
is why v0 does not do it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .dsp import MAD_TO_SIGMA
from .params import InterferenceParams


@dataclass(frozen=True)
class MaskResult:
    """Which pixels count as background, and how much was thrown away."""

    mask: np.ndarray  # True = background, kept
    median: float
    mad: float
    sigma: float
    threshold: float
    iterations: int
    converged: bool
    rejected_fraction: float
    worst_column_rejected_fraction: float
    worst_column_index: int
    reliable: bool
    reason: str | None = None

    @property
    def kept_fraction(self) -> float:
        return 1.0 - self.rejected_fraction


def build_mask(image: np.ndarray, params: InterferenceParams | None = None) -> MaskResult:
    """Iterate median/MAD until the background selection stops changing."""
    params = params or InterferenceParams()
    data = np.asarray(image, dtype=np.float64)
    mask = np.ones(data.shape, dtype=bool)

    median = float(np.median(data))
    scatter = 0.0
    sigma = 0.0
    threshold = float("inf")
    converged = False
    iterations = 0

    for iterations in range(1, params.mask_max_iterations + 1):
        sample = data[mask]
        if sample.size < 2:
            break
        median = float(np.median(sample))
        scatter = float(np.median(np.abs(sample - median)))
        sigma = scatter * MAD_TO_SIGMA
        if sigma <= 0.0:
            # A perfectly flat background: nothing to clip against.
            converged = True
            break
        threshold = median + params.mad_k * sigma
        updated = data <= threshold
        if np.array_equal(updated, mask):
            converged = True
            break
        mask = updated

    rejected = 1.0 - float(np.count_nonzero(mask)) / mask.size
    per_column = 1.0 - mask.mean(axis=0)
    worst_index = int(np.argmax(per_column))
    worst = float(per_column[worst_index])

    reason = None
    reliable = True
    if mask.mean() < params.min_background_fraction:
        reliable = False
        reason = (
            f"only {mask.mean():.1%} of pixels survived the bead mask "
            f"(floor is {params.min_background_fraction:.0%}) — the background estimate, "
            "and everything derived from it, is unreliable"
        )
    elif sigma <= 0.0:
        reliable = False
        reason = "background scatter is zero — the frame has no measurable noise"

    return MaskResult(
        mask=mask,
        median=median,
        mad=scatter,
        sigma=sigma,
        threshold=threshold,
        iterations=iterations,
        converged=converged,
        rejected_fraction=rejected,
        worst_column_rejected_fraction=worst,
        worst_column_index=worst_index,
        reliable=reliable,
        reason=reason,
    )
