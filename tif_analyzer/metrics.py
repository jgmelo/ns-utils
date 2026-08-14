"""Metric registry. Adding a metric must require touching only this module.

Metrics always return ``dict[str, float]``, never a bare float, so a metric with
several outputs (mean + std) does not force a plumbing rewrite.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


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
    compute: Callable[[np.ndarray], dict[str, float]]

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


REGISTRY: dict[str, MetricSpec] = {}


def register(
    key: str,
    name: str,
    description: str,
    outputs: tuple[OutputSpec, ...],
) -> Callable[[Callable[[np.ndarray], dict[str, float]]], Callable[[np.ndarray], dict[str, float]]]:
    """Decorator: register the decorated function as a metric.

    Nothing outside this module hardcodes a metric list.
    """

    def decorator(
        func: Callable[[np.ndarray], dict[str, float]]
    ) -> Callable[[np.ndarray], dict[str, float]]:
        if key in REGISTRY:
            raise ValueError(f"metric already registered: {key!r}")
        if not outputs:
            raise ValueError(f"metric {key!r} declares no outputs")
        seen = [o.key for o in outputs]
        if len(set(seen)) != len(seen):
            raise ValueError(f"metric {key!r} declares duplicate output keys: {seen}")
        REGISTRY[key] = MetricSpec(
            key=key, name=name, description=description, outputs=outputs, compute=func
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


def compute_validated(spec: MetricSpec, image: np.ndarray) -> dict[str, float]:
    """Run *spec* on *image* and enforce the runtime output contract."""
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
