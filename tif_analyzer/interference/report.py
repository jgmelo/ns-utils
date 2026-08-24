"""Plain-text report. The only module that formats anything."""

from __future__ import annotations

import numpy as np

from .analyzer import ChannelAnalysis, InterferenceResult

WIDTH = 78


def _rule(char: str = "─") -> str:
    return char * WIDTH


def _heading(text: str) -> str:
    return f"\n{text}\n{_rule()}"


def scored_label(match) -> str:
    """Name of the closest candidate, even when it was not accepted."""
    candidate = match.candidate
    return candidate.label if candidate is not None else "the closest candidate"


def _fmt(value: float | None, spec: str = ".6g", missing: str = "—") -> str:
    if value is None:
        return missing
    if isinstance(value, float) and not np.isfinite(value):
        return "inf" if value > 0 else ("-inf" if value < 0 else "nan")
    return format(value, spec)


def _frame_block(analysis: ChannelAnalysis, label: str) -> list[str]:
    frame = analysis.frame
    lines = [
        f"  {label}: {frame.name}",
        f"    {frame.n_columns} x {frame.n_lines} px (columns x lines), on-disk dtype {frame.dtype}",
        f"    min {_fmt(frame.minimum)}   median {_fmt(frame.median)}   max {_fmt(frame.maximum)}",
    ]
    if frame.saturation_level is None:
        lines.append("    saturation: not applicable to a floating-point frame")
    elif frame.saturated:
        lines.append(
            f"    SATURATED: {frame.saturated_pixels} px at the {frame.dtype} limit "
            f"({frame.saturated_fraction:.3%}) — clipped pixels distort every statistic below"
        )
    else:
        lines.append(f"    saturation: none at the {frame.dtype} limit ({_fmt(frame.saturation_level)})")
    return lines


def _mask_block(analysis: ChannelAnalysis) -> list[str]:
    result = analysis.mask
    lines = [
        f"    background median {_fmt(result.median)} counts, MAD {_fmt(result.mad)} "
        f"(robust sigma {_fmt(result.sigma)}), rejecting above {_fmt(result.threshold)}",
        f"    rejected {result.rejected_fraction:.2%} of pixels in {result.iterations} iteration(s)"
        + ("" if result.converged else ", DID NOT CONVERGE"),
        f"    worst column: #{result.worst_column_index} with "
        f"{result.worst_column_rejected_fraction:.2%} rejected",
    ]
    if not result.reliable and result.reason:
        lines.append(f"    ! {result.reason}")
    return lines


def _profile_block(analysis: ChannelAnalysis) -> list[str]:
    result = analysis.profile
    lines = [
        f"    columns used {result.columns_used}"
        + (
            f" ({result.flyback_columns_dropped} flyback columns dropped)"
            if result.flyback_columns_dropped
            else ""
        ),
        f"    raw profile std {_fmt(result.std)} counts",
    ]
    if result.masked_std is not None:
        lines.append(f"    bead-masked profile std {_fmt(result.masked_std)} counts")
        lines.append(
            f"    contamination check: rms(raw − masked) is "
            f"{result.contamination_rms_ratio:.2%} of the raw amplitude, "
            f"correlation {_fmt(result.contamination_correlation, '.4f')}"
        )
        if result.contamination_rms_ratio is not None and result.contamination_rms_ratio > 0.1:
            lines.append(
                "    ! the two profiles differ by more than 10% — beads are leaking into the raw "
                "profile; the masked one is used for everything below"
            )
    if result.columns_without_background:
        lines.append(
            f"    ! {result.columns_without_background} column(s) had too little background left "
            "and fell back to the unmasked median"
        )
    return lines


def render_report(result: InterferenceResult) -> str:
    """Render the whole characterisation as plain text."""
    out: list[str] = []
    out.append(_rule("═"))
    out.append("  FIXED-PATTERN INTERFERENCE — CHARACTERISATION (v0, no correction applied)")
    out.append(f"  {result.timestamp}")
    out.append(_rule("═"))

    out.append(_heading("VERDICT"))
    out.append(f"  {result.verdict.headline}")
    for check in result.verdict.checks:
        out.append(f"    {check.summary}")
        if check.detail:
            out.append(f"        {check.detail}")
    for warning in result.verdict.warnings:
        out.append(f"  ! {warning}")
    for note in result.verdict.notes:
        if note:
            out.append(f"  · {note}")
    out.append(
        "  (thresholds are provisional defaults, not calibrated against real frames — "
        "see InterferenceParams)"
    )

    out.append(_heading("1. FRAMES"))
    out.extend(_frame_block(result.signal, "signal"))
    if result.reference is not None:
        out.append("")
        out.extend(_frame_block(result.reference, "reference"))

    out.append(_heading("AXIS CHECK"))
    out.append(
        f"  stripe strength: {result.axis.column_profile_std:.4g} across columns vs "
        f"{result.axis.row_profile_std:.4g} across rows (ratio {result.axis.ratio:.3f})"
    )
    if result.axis.warning:
        out.append(f"  ! {result.axis.warning}")
    else:
        out.append("  consistent with rows being scan lines and time running along columns")

    out.append(_heading("2. BEAD MASK (signal)"))
    out.extend(_mask_block(result.signal))

    out.append(_heading("3. FIXED-PATTERN PROFILE (signal)"))
    out.extend(_profile_block(result.signal))

    out.append(_heading("4. SPECTRAL ESTIMATE (signal)"))
    spectrum = result.signal.spectrum
    if not spectrum.detected:
        out.append(f"    NOT DETECTED — {spectrum.reason}")
    else:
        low, high = spectrum.cluster_frequency_range
        out.append(
            f"    dominant cluster: bins {spectrum.cluster_low_bin}–{spectrum.cluster_high_bin} "
            f"({low:.5f}–{high:.5f} cycles/px)"
        )
        out.append(f"    power-weighted centroid {spectrum.centroid_frequency:.6f} cycles/px")
        out.append(
            f"    period {spectrum.period_px:.3f} px    "
            f"cluster width {spectrum.cluster_width_cycles:.5f} cycles/px"
        )
        out.append(f"    holds {spectrum.power_fraction:.1%} of the profile's non-DC power")
        out.append(
            f"    peak is {_fmt(spectrum.peak_to_floor, '.1f')}x the noise floor "
            f"(detection needs {spectrum.required_ratio:.1f}x at this bin count)"
        )

    out.append(_heading("5. PHYSICAL FREQUENCY"))
    physical = result.physical
    if not physical.available:
        out.append(f"    {physical.reason}")
    else:
        out.append(
            f"    dwell {physical.dwell_s * 1e9:.2f} ns/px   "
            f"sampling {physical.sampling_hz / 1e3:.3f} kHz   "
            f"line period {physical.line_period_s * 1e6:.2f} µs"
        )
        out.append(
            f"    NYQUIST LIMIT {physical.nyquist_hz / 1e3:.3f} kHz — nothing above this can be "
            "distinguished from its alias"
        )
        out.append(f"    observed component: {physical.observed_hz / 1e3:.4f} kHz")
        out.append("    alias family (all consistent with the same samples):")
        for candidate in physical.aliases:
            drift = candidate.predicted_drift_deg_per_line
            predicted = f"predicts {drift:+7.2f}°/line" if drift is not None else ""
            marker = " "
            if (
                result.alias_match is not None
                and result.alias_match.candidate is candidate
                and result.alias_match.unambiguous
            ):
                marker = "→"
            out.append(
                f"     {marker} {candidate.label:<16} {candidate.frequency_hz / 1e3:12.4f} kHz   {predicted}"
            )
        if result.alias_match is not None:
            match = result.alias_match
            if not match.informative:
                out.append(
                    f"    ! the drift cannot separate these: the centroid is known to about "
                    f"±{match.drift_uncertainty_deg:.0f}° of per-line phase, which is more than a "
                    "full turn's worth of ambiguity — narrow the component (longer lines, or a "
                    "cleaner background) before trusting any single member"
                )
            elif not match.consistent:
                out.append(
                    f"    ! no member explains the measured drift: the closest is "
                    f"{scored_label(match)} at {match.residual_deg:.1f}°/line, outside the "
                    f"±{max(match.drift_uncertainty_deg, 5.0):.1f}° this measurement can resolve. "
                    "The interference may be drifting in frequency, or the line rate may be wrong."
                )
            elif match.unambiguous and match.candidate is not None:
                out.append(
                    f"    → the measured drift singles out {match.candidate.label} "
                    f"(residual {match.residual_deg:.1f}°/line; next candidate "
                    f"{match.runner_up_residual_deg:.1f}° away)"
                )
            elif match.candidate is not None:
                out.append(
                    f"    · {match.candidate.label} fits the drift "
                    f"(residual {match.residual_deg:.1f}°/line), but another candidate is "
                    f"about as close — the drift does not settle it"
                )

    out.append(_heading("6. LINE STABILITY"))
    stability = result.stability
    if not stability.available:
        out.append(f"    {stability.reason}")
    else:
        skipped = stability.lines_total - stability.lines_used
        out.append(
            f"    {stability.lines_used} of {stability.lines_total} lines used"
            + (f" ({skipped} skipped: too many bead pixels)" if skipped else "")
        )
        out.append(
            f"    amplitude {stability.amplitude_mean:.4g} counts, "
            f"coefficient of variation {stability.amplitude_cv:.3f}"
        )
        out.append(
            f"    phase drift {stability.phase_drift_deg_per_line:+.4f}°/line "
            f"(residual scatter {stability.phase_drift_residual_deg:.3f}°, "
            f"total span {stability.phase_span_deg:+.1f}°)"
        )
        out.append(
            "    → locked to the line trigger: the pattern repeats frame to frame"
            if stability.locked_to_scan
            else "    → not locked to the scan: the pattern moves between frames"
        )

    out.append(_heading("7. CROSS-CHANNEL COMPARISON"))
    cross = result.cross
    if cross is None:
        out.append(
            "    no reference frame supplied"
            if result.reference is None
            else "    not computed: no dominant component in the signal frame"
        )
    else:
        out.append(
            f"    period: signal {cross.period_signal_px:.3f} px, reference "
            f"{cross.period_reference_px:.3f} px, difference "
            f"{cross.period_relative_difference:.3%}"
        )
        out.append(
            f"    stripe amplitude: signal {cross.signal_band_rms:.4g} vs reference "
            f"{cross.reference_band_rms:.4g} counts rms "
            f"(signal/reference = {cross.amplitude_ratio:.3f})"
        )
        out.append(f"    correlation, raw profiles:      {cross.correlation_raw:+.4f}")
        out.append(
            f"    correlation, stripe band only:  {cross.correlation_band:+.4f}   "
            f"(bins {cross.band_low_bin}–{cross.band_high_bin})"
        )
        out.append(f"    gap (band − |raw|): {cross.correlation_gap:+.4f}")
        out.append(
            f"    phase offset: {cross.phase_offset_deg:+.2f}° = {cross.phase_offset_px:+.3f} px"
        )
        if result.reference is not None:
            out.append("")
            out.append("    reference channel, on its own:")
            out.extend(f"    {line}" for line in _mask_block(result.reference))

    out.append("")
    out.append(_rule("═"))
    return "\n".join(out)
