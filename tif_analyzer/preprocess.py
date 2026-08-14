"""Ordered preprocessing pipeline, canonical serialization and hashing.

Empty in v1 — the pipeline is a pass-through — but the plumbing exists so that
ROI crop / dark-frame subtraction can be added without touching the layers that
carry the preprocess hash around (the cache key, the config snapshot, the CSV
header).
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Callable, Iterable, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np


@runtime_checkable
class PreprocessStep(Protocol):
    """One step of the pipeline.

    ``params`` must be JSON-serializable: it is what makes the step's identity
    reproducible across runs, and therefore what the cache key is built from.
    """

    name: str

    @property
    def params(self) -> Mapping[str, Any]: ...

    def apply(self, image: np.ndarray) -> np.ndarray: ...


#: Steps self-register here, mirroring the metric registry. Empty in v1.
REGISTRY: dict[str, Callable[..., PreprocessStep]] = {}


def register(name: str) -> Callable[[Callable[..., PreprocessStep]], Callable[..., PreprocessStep]]:
    """Register a step factory under *name*."""

    def decorator(factory: Callable[..., PreprocessStep]) -> Callable[..., PreprocessStep]:
        if name in REGISTRY:
            raise ValueError(f"preprocess step already registered: {name!r}")
        REGISTRY[name] = factory
        return factory

    return decorator


def serialize(steps: Sequence[PreprocessStep]) -> list[dict[str, Any]]:
    """Canonical, order-preserving description of the pipeline."""
    return [
        {"name": step.name, "params": dict(step.params)} for step in steps
    ]


def preprocess_hash(steps: Sequence[PreprocessStep]) -> str:
    """Stable hash of the pipeline, used in the cache key and the snapshot.

    The empty pipeline has a stable hash of its own, so v1 results stay cached
    across runs within a session.
    """
    payload = json.dumps(serialize(steps), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def describe(steps: Sequence[PreprocessStep]) -> str:
    """Human-readable one-liner for the UI header."""
    if not steps:
        return "none"
    return " → ".join(step.name for step in steps)


def apply_pipeline(image: np.ndarray, steps: Iterable[PreprocessStep]) -> np.ndarray:
    """Run *image* through *steps* in order. Pass-through when empty."""
    result = image
    for step in steps:
        result = step.apply(result)
        if not isinstance(result, np.ndarray):
            raise TypeError(
                f"preprocess step {step.name!r} returned {type(result).__name__}, expected ndarray"
            )
    return result
