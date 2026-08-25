"""Orchestration: the seven operations, in order, into one result object.

Importable from a script or a notebook with no UI present, like the rest of the
analysis layer — nothing here prints.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ..loading import DEFAULT_PAGE_POLICY, PagePolicy
from . import crosschannel, mask as mask_mod, physical, profile as profile_mod, spectrum as spectrum_mod, stability as stability_mod, verdict as verdict_mod
from .crosschannel import CrossChannelResult
from .frames import Frame, frame_from_array, load_frame, validate_pair
from .mask import MaskResult
from .params import Acquisition, InterferenceParams
from .physical import AliasMatch, PhysicalResult
from .profile import AxisCheck, ProfileResult
from .spectrum import SpectrumResult
from .stability import StabilityResult
from .verdict import Verdict


@dataclass(frozen=True)
class ChannelAnalysis:
    """Everything derived from one frame on its own."""

    frame: Frame
    mask: MaskResult
    profile: ProfileResult
    spectrum: SpectrumResult


@dataclass(frozen=True)
class InterferenceResult:
    """The structured result the spec asks for. :func:`render_report` turns it
    into text; nothing in here is formatted."""

    signal: ChannelAnalysis
    reference: ChannelAnalysis | None
    axis: AxisCheck
    physical: PhysicalResult
    stability: StabilityResult
    alias_match: AliasMatch | None
    cross: CrossChannelResult | None
    verdict: Verdict
    params: InterferenceParams
    acquisition: Acquisition
    timestamp: str


def analyse_channel(
    frame: Frame, flyback_columns: int = 0, params: InterferenceParams | None = None
) -> ChannelAnalysis:
    """Mask, profile and spectrum for one frame on its own.

    Public because the figures need exactly this much and nothing more: a
    cross-channel comparison is not required to look at one channel's profile.
    """
    params = params or InterferenceParams()
    mask_result = mask_mod.build_mask(frame.data, params)
    profile_result = profile_mod.build_profile(frame.data, mask_result, flyback_columns, params)
    spectrum_result = spectrum_mod.analyse(profile_result.analysis_profile, params)
    return ChannelAnalysis(
        frame=frame, mask=mask_result, profile=profile_result, spectrum=spectrum_result
    )


def characterise_arrays(
    signal: np.ndarray,
    reference: np.ndarray | None = None,
    acquisition: Acquisition | None = None,
    params: InterferenceParams | None = None,
    dtype: str = "uint16",
) -> InterferenceResult:
    """Characterise in-memory frames, for synthetic data and notebooks."""
    signal_frame = frame_from_array(signal, dtype=dtype)
    reference_frame = None if reference is None else frame_from_array(reference, dtype=dtype)
    return _characterise(signal_frame, reference_frame, acquisition, params)


def characterise(
    signal_path: Path,
    reference_path: Path | None = None,
    acquisition: Acquisition | None = None,
    params: InterferenceParams | None = None,
    page_policy: PagePolicy = DEFAULT_PAGE_POLICY,
) -> InterferenceResult:
    """Characterise a signal frame, optionally against a reference frame."""
    signal_frame = load_frame(signal_path, page_policy)
    reference_frame = (
        None if reference_path is None else load_frame(reference_path, page_policy)
    )
    return _characterise(signal_frame, reference_frame, acquisition, params)


def _characterise(
    signal_frame: Frame,
    reference_frame: Frame | None,
    acquisition: Acquisition | None,
    params: InterferenceParams | None,
) -> InterferenceResult:
    params = params or InterferenceParams()
    acquisition = acquisition or Acquisition()

    if reference_frame is not None:
        validate_pair(signal_frame, reference_frame)

    flyback_columns = acquisition.flyback_columns(signal_frame.n_columns)

    axis = profile_mod.axis_check(signal_frame.data, params)
    signal_analysis = analyse_channel(signal_frame, flyback_columns, params)
    reference_analysis = (
        None
        if reference_frame is None
        else analyse_channel(reference_frame, flyback_columns, params)
    )

    physical_result = (
        physical.resolve(
            signal_analysis.spectrum.centroid_frequency,
            signal_frame.n_columns,
            acquisition,
            params,
        )
        if signal_analysis.spectrum.detected
        else PhysicalResult(
            available=False, reason="not computed: no dominant component to convert"
        )
    )

    stability_result = stability_mod.analyse(
        signal_frame.data,
        signal_analysis.spectrum.centroid_frequency if signal_analysis.spectrum.detected else 0.0,
        signal_analysis.mask,
        flyback_columns,
        params,
    )

    alias_match = None
    if physical_result.available and stability_result.available:
        # Half the cluster width is the honest spread of the centroid, and it is
        # what decides whether the drift can separate the alias family at all.
        uncertainty_hz = (
            0.5 * signal_analysis.spectrum.cluster_width_cycles * physical_result.sampling_hz
        )
        alias_match = physical.match_alias(
            physical_result, stability_result.phase_drift_deg_per_line, uncertainty_hz
        )

    cross = None
    if reference_analysis is not None and signal_analysis.spectrum.detected:
        cross = crosschannel.compare(
            signal_analysis.profile.analysis_profile,
            reference_analysis.profile.analysis_profile,
            signal_analysis.spectrum,
            reference_analysis.spectrum,
        )

    decision = verdict_mod.decide(
        signal_analysis.spectrum, stability_result, cross, params
    )

    return InterferenceResult(
        signal=signal_analysis,
        reference=reference_analysis,
        axis=axis,
        physical=physical_result,
        stability=stability_result,
        alias_match=alias_match,
        cross=cross,
        verdict=decision,
        params=params,
        acquisition=acquisition,
        timestamp=datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds"),
    )
