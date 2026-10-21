"""Metric registry. Adding a metric must require touching only this module.

Metrics always return ``dict[str, float]``, never a bare float, so a metric with
several outputs (mean + std) does not force a plumbing rewrite.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import numpy as np

from .regions import NOISE, SIGNAL


@dataclass(frozen=True)
class OutputSpec:
    key: str  # "std"
    label: str  # "Standard deviation"
    unit: str | None  # "counts", or None


@dataclass(frozen=True)
class MetricSpec:
    key: str
    name: str
    description: str
    outputs: tuple[OutputSpec, ...]
    compute: Callable[..., dict[str, float]]
    #: Region roles this metric needs, declared statically like ``outputs`` so
    #: the UI can say "select the areas first" without running anything.
    #: Empty tuple -> ``compute(image)``; otherwise ``compute(image, regions)``
    #: where ``regions`` maps each declared role to its 1-D pixel samples.
    requires_regions: tuple[str, ...] = ()

    @property
    def needs_regions(self) -> bool:
        return bool(self.requires_regions)

    @property
    def output_keys(self) -> tuple[str, ...]:
        return tuple(o.key for o in self.outputs)

    def output(self, key: str) -> OutputSpec:
        for spec in self.outputs:
            if spec.key == key:
                return spec
        raise KeyError(f"metric {self.key!r} declares no output {key!r}")

    @property
    def default_series(self) -> str:
        return self.outputs[0].key


class MetricContractError(RuntimeError):
    """A metric returned something other than its declared output keys.

    This is a bug in the metric, not a data problem: it is raised loudly and
    never collected as a per-file failure.
    """


class MissingRegionsError(ValueError):
    """A region-aware metric was run without the regions it declares.

    A precondition, not a per-file problem: the caller is expected to check
    ``spec.requires_regions`` first, so this is raised once, up front, rather
    than turning into one identical failure per file.
    """


REGISTRY: dict[str, MetricSpec] = {}


def register(
    key: str,
    name: str,
    description: str,
    outputs: tuple[OutputSpec, ...],
    requires_regions: tuple[str, ...] = (),
) -> Callable[[Callable[..., dict[str, float]]], Callable[..., dict[str, float]]]:
    """Decorator: register the decorated function as a metric.

    Nothing outside this module hardcodes a metric list.
    """

    def decorator(func: Callable[..., dict[str, float]]) -> Callable[..., dict[str, float]]:
        if key in REGISTRY:
            raise ValueError(f"metric already registered: {key!r}")
        if not outputs:
            raise ValueError(f"metric {key!r} declares no outputs")
        seen = [o.key for o in outputs]
        if len(set(seen)) != len(seen):
            raise ValueError(f"metric {key!r} declares duplicate output keys: {seen}")
        if len(set(requires_regions)) != len(requires_regions):
            raise ValueError(f"metric {key!r} declares duplicate region roles: {requires_regions}")
        REGISTRY[key] = MetricSpec(
            key=key,
            name=name,
            description=description,
            outputs=outputs,
            compute=func,
            requires_regions=tuple(requires_regions),
        )
        return func

    return decorator


def get(key: str) -> MetricSpec:
    try:
        return REGISTRY[key]
    except KeyError:
        known = ", ".join(sorted(REGISTRY)) or "<none>"
        raise KeyError(f"unknown metric {key!r}; registered: {known}") from None


def available() -> list[MetricSpec]:
    return [REGISTRY[k] for k in sorted(REGISTRY)]


def compute_validated(
    spec: MetricSpec,
    image: np.ndarray,
    regions: Mapping[str, np.ndarray] | None = None,
) -> dict[str, float]:
    """Run *spec* on *image* and enforce the runtime output contract.

    Region-aware metrics are called as ``compute(image, regions)``; region-free
    ones keep the original ``compute(image)`` signature. The branch is explicit
    rather than inferred from the function signature.
    """
    if spec.needs_regions:
        missing = [role for role in spec.requires_regions if role not in (regions or {})]
        if missing:
            raise MissingRegionsError(
                f"metric {spec.key!r} needs the {', '.join(spec.requires_regions)} region(s); "
                f"missing: {', '.join(missing)}"
            )
        assert regions is not None
        values = spec.compute(image, {role: regions[role] for role in spec.requires_regions})
    else:
        values = spec.compute(image)
    if not isinstance(values, dict):
        raise MetricContractError(
            f"metric {spec.key!r} returned {type(values).__name__}, expected dict[str, float]"
        )
    expected = set(spec.output_keys)
    got = set(values)
    if got != expected:
        missing = sorted(expected - got)
        extra = sorted(got - expected)
        detail = []
        if missing:
            detail.append(f"missing {missing}")
        if extra:
            detail.append(f"unexpected {extra}")
        raise MetricContractError(
            f"metric {spec.key!r} output keys do not match its declaration: " + ", ".join(detail)
        )
    return {key: float(values[key]) for key in spec.output_keys}


# --------------------------------------------------------------------------- #
# v1 metrics
# --------------------------------------------------------------------------- #


@register(
    key="std",
    name="Standard deviation",
    description=(
        "Sample standard deviation (ddof=1) of all pixel values, in raw counts. "
        "Sample, not population: the distinction matters for noise analysis."
    ),
    outputs=(OutputSpec("std", "Standard deviation", None),),
)
def _std(image: np.ndarray) -> dict[str, float]:
    if image.size < 2:
        raise ValueError(
            f"sample standard deviation needs at least 2 pixels, image has {image.size}"
        )
    return {"std": float(np.std(image, ddof=1))}


@register(
    key="snr",
    name="Signal-to-noise ratio",
    description=(
        "From two selected areas of the same image: "
        "snr = (signal_mean - noise_mean) / noise_std, with the sample standard "
        "deviation (ddof=1) over the noise area. All values in raw counts; the "
        "ratio is dimensionless."
    ),
    outputs=(
        OutputSpec("snr", "Signal-to-noise ratio", None),
        OutputSpec("signal_mean", "Signal mean", "counts"),
        OutputSpec("noise_mean", "Noise mean", "counts"),
        OutputSpec("noise_std", "Noise standard deviation", "counts"),
        # Spread *inside* the signal area, which the ratio itself does not use.
        # It is the only output that carries the sample's own photon statistics:
        # the noise area holds none of the sample's photons, so its sigma cannot
        # grow with the sample's brightness however shot-limited the image is.
        OutputSpec("signal_std", "Signal-area standard deviation", "counts"),
    ),
    requires_regions=(SIGNAL, NOISE),
)
def _snr(image: np.ndarray, regions: Mapping[str, np.ndarray]) -> dict[str, float]:
    signal = regions[SIGNAL]
    noise = regions[NOISE]
    if noise.size < 2:
        raise ValueError(
            f"noise area needs at least 2 pixels for ddof=1, it has {noise.size}"
        )
    if signal.size < 2:
        raise ValueError(
            f"signal area needs at least 2 pixels for ddof=1, it has {signal.size}"
        )
    noise_std = float(np.std(noise, ddof=1))
    if noise_std == 0.0:
        raise ValueError(
            "noise area has zero variance — every pixel is identical, so the ratio "
            "is undefined; pick an area with actual noise in it"
        )
    signal_mean = float(np.mean(signal))
    noise_mean = float(np.mean(noise))
    return {
        "snr": (signal_mean - noise_mean) / noise_std,
        "signal_mean": signal_mean,
        "noise_mean": noise_mean,
        "noise_std": noise_std,
        "signal_std": float(np.std(signal, ddof=1)),
    }


@register(
    key="snr_signal",
    name="Signal-to-noise ratio, signal-referenced",
    description=(
        "From the same two areas, but divided by the spread *inside* the signal "
        "area: snr_signal = (signal_mean - noise_mean) / signal_std. Where snr "
        "asks how far the sample stands out from the background, this asks how "
        "precisely the sample's own brightness is known. Two caveats: it is "
        "valid only where the signal area is uniform, since spread across real "
        "structure is not noise; and its 0.5 / 1.0 exponents against the "
        "parameter are asymptotes, reached only when the sample far outshines "
        "the background. With a pedestal, sigma is sqrt(background + signal) and "
        "the exponent lands between the two — 0.73 for a Poisson series sitting "
        "on 1000 counts. The photon transfer curve has no such problem."
    ),
    outputs=(
        OutputSpec("snr_signal", "Signal-to-noise ratio (signal-referenced)", None),
        OutputSpec("signal_mean", "Signal mean", "counts"),
        OutputSpec("noise_mean", "Noise mean", "counts"),
        OutputSpec("signal_std", "Signal-area standard deviation", "counts"),
        OutputSpec("noise_std", "Noise standard deviation", "counts"),
    ),
    requires_regions=(SIGNAL, NOISE),
)
def _snr_signal(image: np.ndarray, regions: Mapping[str, np.ndarray]) -> dict[str, float]:
    signal = regions[SIGNAL]
    noise = regions[NOISE]
    if signal.size < 2:
        raise ValueError(
            f"signal area needs at least 2 pixels for ddof=1, it has {signal.size}"
        )
    if noise.size < 2:
        raise ValueError(
            f"noise area needs at least 2 pixels for ddof=1, it has {noise.size}"
        )
    signal_std = float(np.std(signal, ddof=1))
    if signal_std == 0.0:
        raise ValueError(
            "signal area has zero variance — every pixel is identical, so the ratio "
            "is undefined; pick an area with actual noise in it"
        )
    signal_mean = float(np.mean(signal))
    noise_mean = float(np.mean(noise))
    return {
        "snr_signal": (signal_mean - noise_mean) / signal_std,
        "signal_mean": signal_mean,
        "noise_mean": noise_mean,
        "signal_std": signal_std,
        "noise_std": float(np.std(noise, ddof=1)),
    }
