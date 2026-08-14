from __future__ import annotations

import numpy as np
import pytest

from tif_analyzer import metrics, preprocess
from tif_analyzer.metrics import MetricContractError, MetricSpec, OutputSpec, compute_validated


def test_std_metric_is_registered_with_static_outputs():
    spec = metrics.get("std")
    assert spec.name == "Standard deviation"
    assert spec.output_keys == ("std",)
    assert "ddof=1" in spec.description


def test_std_uses_sample_ddof_one():
    image = np.array([[1.0, 2.0], [3.0, 4.0]])
    values = compute_validated(metrics.get("std"), image)
    assert values == pytest.approx({"std": float(np.std(image, ddof=1))})
    assert values["std"] != pytest.approx(float(np.std(image)))


def test_metrics_return_dicts_not_floats():
    values = metrics.get("std").compute(np.array([[1.0, 5.0]]))
    assert isinstance(values, dict)


def test_std_needs_two_pixels():
    with pytest.raises(ValueError, match="at least 2 pixels"):
        compute_validated(metrics.get("std"), np.array([[7.0]]))


def test_output_contract_violation_raises():
    broken = MetricSpec(
        key="broken",
        name="Broken",
        description="",
        outputs=(OutputSpec("mean", "Mean", None), OutputSpec("std", "Std", None)),
        compute=lambda image: {"mean": 1.0},
    )
    with pytest.raises(MetricContractError, match="missing"):
        compute_validated(broken, np.zeros((2, 2)))


def test_unexpected_output_key_raises():
    broken = MetricSpec(
        key="broken",
        name="Broken",
        description="",
        outputs=(OutputSpec("mean", "Mean", None),),
        compute=lambda image: {"mean": 1.0, "extra": 2.0},
    )
    with pytest.raises(MetricContractError, match="unexpected"):
        compute_validated(broken, np.zeros((2, 2)))


def test_bare_float_return_raises():
    broken = MetricSpec(
        key="broken",
        name="Broken",
        description="",
        outputs=(OutputSpec("mean", "Mean", None),),
        compute=lambda image: 1.0,  # type: ignore[return-value]
    )
    with pytest.raises(MetricContractError, match="expected dict"):
        compute_validated(broken, np.zeros((2, 2)))


def test_unknown_metric_lists_registered_ones():
    with pytest.raises(KeyError, match="registered: std"):
        metrics.get("nope")


# -- preprocess -------------------------------------------------------------- #


def test_v1_pipeline_is_empty_and_pass_through():
    assert preprocess.REGISTRY == {}
    image = np.arange(9.0).reshape(3, 3)
    assert preprocess.apply_pipeline(image, []) is image
    assert preprocess.describe([]) == "none"


def test_empty_pipeline_hash_is_stable():
    assert preprocess.preprocess_hash([]) == preprocess.preprocess_hash([])
    assert preprocess.serialize([]) == []


class _Scale:
    name = "scale"

    def __init__(self, factor: float):
        self.factor = factor

    @property
    def params(self):
        return {"factor": self.factor}

    def apply(self, image):
        return image * self.factor


def test_hash_depends_on_steps_and_order():
    a, b = _Scale(2.0), _Scale(3.0)
    assert preprocess.preprocess_hash([a]) != preprocess.preprocess_hash([])
    assert preprocess.preprocess_hash([a, b]) != preprocess.preprocess_hash([b, a])
    assert preprocess.preprocess_hash([a]) == preprocess.preprocess_hash([_Scale(2.0)])


def test_pipeline_runs_steps_in_order():
    image = np.ones((2, 2))
    result = preprocess.apply_pipeline(image, [_Scale(2.0), _Scale(3.0)])
    np.testing.assert_allclose(result, np.full((2, 2), 6.0))
    assert preprocess.describe([_Scale(2.0), _Scale(3.0)]) == "scale → scale"
