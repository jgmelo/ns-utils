"""Correcting one frame, and correcting a whole series the same way.

Two things live here, and the second is built out of the first on purpose: the
batch must not be able to drift from what the subtraction screen showed. If a
single frame corrected interactively and the same frame corrected in a batch
could disagree, the screen would stop being a preview of the batch — which is
the only reason to have a preview at all.

Nothing here prints.

**Where the corrected frames go.** Into a directory of their own, never beside
the originals, and under their original names. Both halves of that matter:

* Writing beside the originals would put two files carrying the same parameter
  value in one directory, and the series pipeline refuses that — correctly. The
  corrected series would be unplottable the moment it was written.
* Keeping the names means the glob and the regex that found the originals find
  the corrected frames unchanged. Point the analyzer at the new directory and
  everything else is already set up.

The provenance is not lost to the renaming: every frame still carries its own
``.subtraction.json``, and the batch writes one summary table for the run.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator, Iterable, Sequence

from ..loading import DEFAULT_PAGE_POLICY, PagePolicy
from . import crosschannel
from .analyzer import ChannelAnalysis, analyse_channel
from .frames import Frame, FrameError, frame_from_array, load_frame
from .params import Acquisition, InterferenceParams
from .subtract import SubtractionError, SubtractionPlan, SubtractionResult, alignment_shift, subtract
from .writing import write_corrected

SUMMARY_NAME = "subtraction-summary.csv"


class BatchError(Exception):
    """The batch cannot start. Distinct from a single file failing inside it."""


# --------------------------------------------------------------------------- #
# one frame
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Correction:
    """One frame's correction, plus what it took to compute it."""

    result: SubtractionResult
    before: ChannelAnalysis
    after: ChannelAnalysis | None = None
    notes: tuple[str, ...] = field(default_factory=tuple)


def correct_frame(
    signal_frame: Frame,
    *,
    mode: str,
    band_limited: bool = True,
    per_line: bool = False,
    params: InterferenceParams | None = None,
    acquisition: Acquisition | None = None,
    reference_frame: Frame | None = None,
    reference_analysis: ChannelAnalysis | None = None,
    analyse_after: bool = False,
) -> Correction:
    """Analyse *signal_frame*, plan the subtraction, and carry it out.

    The analysis is not optional overhead: the stripe frequency sets the band,
    the bead mask keeps grains out of the fit, and the phase offset against the
    reference sets the alignment. A subtraction planned without them is a
    subtraction planned blind.

    *reference_analysis* is an optimisation for batches — the reference does not
    change from file to file, so it need not be analysed again for each one.
    """
    params = params or InterferenceParams()
    acquisition = acquisition or Acquisition()
    notes: list[str] = []

    flyback = acquisition.flyback_columns(signal_frame.n_columns)
    before = analyse_channel(signal_frame, flyback, params)
    frequency = before.spectrum.centroid_frequency if before.spectrum.detected else 0.0

    band = None
    shift = 0.0
    if mode == "reference":
        if reference_frame is None:
            raise SubtractionError("reference mode needs a reference frame")
        if reference_frame.shape != signal_frame.shape:
            raise SubtractionError(
                f"reference is {reference_frame.n_columns}x{reference_frame.n_lines} px "
                f"but the signal is {signal_frame.n_columns}x{signal_frame.n_lines} px"
            )
        if reference_analysis is None:
            reference_analysis = analyse_channel(reference_frame, flyback, params)
        if before.spectrum.detected and reference_analysis.spectrum.detected:
            cross = crosschannel.compare(
                before.profile.analysis_profile,
                reference_analysis.profile.analysis_profile,
                before.spectrum,
                reference_analysis.spectrum,
            )
            band = (
                min(before.spectrum.cluster_low_bin, reference_analysis.spectrum.cluster_low_bin),
                max(before.spectrum.cluster_high_bin, reference_analysis.spectrum.cluster_high_bin),
            )
            shift = alignment_shift(cross.phase_offset_px)
        else:
            notes.append(
                "no dominant component in one of the frames — subtracting unaligned, with no band"
            )

    plan = SubtractionPlan(
        mode=mode,
        band_limited=band_limited and band is not None,
        per_line=per_line,
        shift_px=shift,
    )
    result = subtract(
        signal_frame.data,
        plan,
        reference=reference_frame.data if reference_frame is not None else None,
        mask_result=before.mask,
        band=band,
        frequency=frequency,
        params=params,
    )

    after = None
    if analyse_after:
        after = analyse_channel(
            frame_from_array(result.corrected, dtype="float32", path=signal_frame.path),
            flyback,
            params,
        )
    return Correction(result=result, before=before, after=after, notes=tuple(notes))


# --------------------------------------------------------------------------- #
# the series
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class BatchOutcome:
    """What happened to one file. Exactly one of *result* / *error* is set."""

    source: Path
    output: Path | None = None
    result: SubtractionResult | None = None
    error: Exception | None = None
    skipped: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.skipped is None


@dataclass(frozen=True)
class BatchProgress:
    """Emitted once per file, after it is written."""

    index: int  # 1-based
    total: int
    outcome: BatchOutcome


@dataclass
class BatchReport:
    output_dir: Path
    mode: str
    outcomes: list[BatchOutcome] = field(default_factory=list)
    summary_path: Path | None = None

    @property
    def written(self) -> list[BatchOutcome]:
        return [outcome for outcome in self.outcomes if outcome.ok]

    @property
    def failures(self) -> list[BatchOutcome]:
        return [outcome for outcome in self.outcomes if outcome.error is not None]

    @property
    def skipped(self) -> list[BatchOutcome]:
        return [outcome for outcome in self.outcomes if outcome.skipped is not None]

    def removed_range(self) -> tuple[float, float] | None:
        """Best and worst stripe removal across the run.

        A series corrected frame by frame can only be compared point to point if
        every frame was corrected about as well; a run spanning 20% to 90% has
        put a trend into the data that was not in the sample.
        """
        fractions = [
            outcome.result.removed_fraction
            for outcome in self.written
            if outcome.result is not None
        ]
        if not fractions:
            return None
        return min(fractions), max(fractions)


def default_output_dir(directory: Path, mode: str) -> Path:
    name = "corrected" if mode == "reference" else "deprofiled"
    return Path(directory) / name


def run_batch(
    paths: Sequence[Path],
    output_dir: Path,
    *,
    mode: str,
    band_limited: bool = True,
    per_line: bool = False,
    params: InterferenceParams | None = None,
    acquisition: Acquisition | None = None,
    reference_path: Path | None = None,
    page_policy: PagePolicy = DEFAULT_PAGE_POLICY,
) -> Generator[BatchProgress, None, BatchReport]:
    """Correct every frame in *paths*, yielding one event per file.

    The finished :class:`BatchReport` is the generator's return value, the same
    contract the series run uses. A file that fails is recorded and the run goes
    on: one unreadable frame in a series of twenty should not cost the other
    nineteen.
    """
    paths = [Path(path) for path in paths]
    output_dir = Path(output_dir).expanduser()
    if mode == "reference" and reference_path is None:
        raise BatchError("reference mode needs a reference frame")

    collisions = [path for path in paths if _resolve(path.parent) == _resolve(output_dir)]
    if collisions:
        raise BatchError(
            f"the output directory is where the originals live ({output_dir}) — "
            "the corrected frames keep their names and would overwrite them"
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    report = BatchReport(output_dir=output_dir, mode=mode)

    reference_frame = None
    reference_analysis = None
    if reference_path is not None:
        try:
            reference_frame = load_frame(reference_path, page_policy)
        except (OSError, FrameError, ValueError) as exc:
            raise BatchError(f"could not read the reference {reference_path.name}: {exc}") from exc
        # The reference is the same frame for every file: analyse it once here
        # rather than once per file. Every signal frame must match its shape, so
        # the flyback trim is the same for all of them.
        reference_flyback = (acquisition or Acquisition()).flyback_columns(
            reference_frame.n_columns
        )
        reference_analysis = analyse_channel(
            reference_frame, reference_flyback, params or InterferenceParams()
        )

    total = len(paths)
    for index, path in enumerate(paths, start=1):
        outcome = _correct_one(
            path,
            output_dir,
            mode=mode,
            band_limited=band_limited,
            per_line=per_line,
            params=params,
            acquisition=acquisition,
            reference_frame=reference_frame,
            reference_path=reference_path,
            reference_analysis=reference_analysis,
            page_policy=page_policy,
        )
        report.outcomes.append(outcome)
        yield BatchProgress(index=index, total=total, outcome=outcome)

    report.summary_path = write_summary(report, output_dir / SUMMARY_NAME)
    return report


def _resolve(path: Path) -> Path:
    try:
        return path.resolve()
    except OSError:  # pragma: no cover - unresolvable paths are rare
        return path


def _correct_one(
    path: Path,
    output_dir: Path,
    *,
    mode: str,
    band_limited: bool,
    per_line: bool,
    params: InterferenceParams | None,
    acquisition: Acquisition | None,
    reference_frame: Frame | None,
    reference_path: Path | None,
    reference_analysis: ChannelAnalysis | None,
    page_policy: PagePolicy,
) -> BatchOutcome:
    if reference_path is not None and _resolve(path) == _resolve(reference_path):
        return BatchOutcome(source=path, skipped="this is the reference frame")
    try:
        signal_frame = load_frame(path, page_policy)
        correction = correct_frame(
            signal_frame,
            mode=mode,
            band_limited=band_limited,
            per_line=per_line,
            params=params,
            acquisition=acquisition,
            reference_frame=reference_frame,
            reference_analysis=reference_analysis,
        )
        written, _ = write_corrected(
            output_dir / path.name,
            correction.result,
            signal_path=path,
            reference_path=reference_path,
            extra={"batch": True},
        )
    except Exception as exc:  # noqa: BLE001 - per-file failures are collected
        return BatchOutcome(source=path, error=exc)
    return BatchOutcome(source=path, output=written, result=correction.result)


SUMMARY_HEADER = (
    "filename",
    "output",
    "mode",
    "period_px",
    "scale",
    "shift_px",
    "stripe_before",
    "stripe_after",
    "removed_fraction",
    "noise_before",
    "noise_after",
    "noise_change",
    "warnings",
)


def write_summary(report: BatchReport, path: Path) -> Path | None:
    """One row per corrected frame, so the run can be checked as a whole.

    The per-file sidecars say what happened to each frame; this says whether the
    frames were treated alike, which is the question a series plot depends on.
    """
    path = Path(path)
    try:
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(SUMMARY_HEADER)
            for outcome in report.outcomes:
                if outcome.result is None:
                    reason = outcome.skipped or f"{type(outcome.error).__name__}: {outcome.error}"
                    writer.writerow([outcome.source.name, "", report.mode] + [""] * 9 + [reason])
                    continue
                result = outcome.result
                writer.writerow(
                    [
                        outcome.source.name,
                        outcome.output.name if outcome.output else "",
                        result.mode,
                        _fmt(1.0 / result.frequency) if result.frequency > 0 else "",
                        _fmt(result.scale) if result.mode == "reference" else "",
                        _fmt(result.shift_px) if result.mode == "reference" else "",
                        _fmt(result.stripe_before),
                        _fmt(result.stripe_after),
                        _fmt(result.removed_fraction),
                        _fmt(result.sigma_before),
                        _fmt(result.sigma_after),
                        _fmt(result.noise_change),
                        " | ".join(result.warnings),
                    ]
                )
    except OSError:
        return None
    return path


def _fmt(value: float) -> str:
    return f"{value:.6g}"


def run_batch_blocking(paths: Iterable[Path], output_dir: Path, **kwargs) -> BatchReport:
    """The generator, driven to the end, for scripts that want no progress."""
    generator = run_batch(list(paths), output_dir, **kwargs)
    while True:
        try:
            next(generator)
        except StopIteration as stop:
            return stop.value
