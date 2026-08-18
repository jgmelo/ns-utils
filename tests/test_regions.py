from __future__ import annotations

import numpy as np
import pytest

from tif_analyzer.regions import (
    Ellipse,
    EmptyRegionError,
    Rect,
    RegionError,
    RegionOutOfBoundsError,
    RegionSet,
    region_from_dict,
    region_from_extents,
    region_set_from_dict,
    regions_hash,
)

SHAPE = (20, 30)  # height, width


def test_rect_mask_covers_exactly_its_pixels():
    mask = Rect(x=2, y=3, width=4, height=5).mask(SHAPE)
    assert mask.sum() == 20
    assert mask[3, 2] and mask[7, 5]
    assert not mask[2, 2] and not mask[8, 6]


def test_rect_bounds_are_right_exclusive():
    assert Rect(x=2, y=3, width=4, height=5).bounds == (2, 3, 6, 8)


def test_rect_must_have_positive_size():
    with pytest.raises(RegionError, match="at least 1x1"):
        Rect(x=0, y=0, width=0, height=3)


def test_ellipse_mask_is_centred_and_roughly_elliptical():
    ellipse = Ellipse(cx=10.0, cy=10.0, rx=5.0, ry=3.0)
    mask = ellipse.mask(SHAPE)
    assert mask[10, 10]
    assert mask[10, 15] and mask[13, 10]  # on the axes, at the radius
    assert not mask[10, 16] and not mask[14, 10]
    assert not mask[13, 14]  # corner of the bounding box is outside
    # Area is within a pixel-rounding margin of pi*rx*ry.
    assert mask.sum() == pytest.approx(np.pi * 5 * 3, rel=0.15)


def test_ellipse_radii_must_be_positive():
    with pytest.raises(RegionError, match="radii must be positive"):
        Ellipse(cx=1.0, cy=1.0, rx=0.0, ry=2.0)


@pytest.mark.parametrize(
    "region",
    [
        Rect(x=28, y=0, width=5, height=5),
        Rect(x=-1, y=0, width=5, height=5),
        Rect(x=0, y=18, width=5, height=5),
        Ellipse(cx=29.0, cy=10.0, rx=5.0, ry=3.0),
    ],
)
def test_out_of_bounds_regions_are_rejected(region):
    with pytest.raises(RegionOutOfBoundsError):
        region.mask(SHAPE)


def test_serialization_round_trip():
    for region in (Rect(x=1, y=2, width=3, height=4), Ellipse(cx=5.5, cy=6.5, rx=2.0, ry=3.0)):
        assert region_from_dict(region.to_dict()) == region


def test_unknown_kind_is_rejected():
    with pytest.raises(RegionError, match="unknown or missing region kind"):
        region_from_dict({"kind": "polygon", "points": []})


def test_malformed_region_payload_is_rejected():
    with pytest.raises(RegionError, match="malformed rect"):
        region_from_dict({"kind": "rect", "x": 1})


# -- selector geometry ------------------------------------------------------- #


def test_rect_from_extents_is_inclusive_of_both_ends():
    rect = region_from_extents("rect", (2.0, 5.0, 3.0, 4.0), SHAPE)
    assert rect == Rect(x=2, y=3, width=4, height=2)


def test_extents_are_clipped_to_the_image():
    rect = region_from_extents("rect", (-5.0, 100.0, -2.0, 100.0), SHAPE)
    assert rect == Rect(x=0, y=0, width=30, height=20)


def test_reversed_drag_is_normalised():
    assert region_from_extents("rect", (5.0, 2.0, 4.0, 3.0), SHAPE) == region_from_extents(
        "rect", (2.0, 5.0, 3.0, 4.0), SHAPE
    )


def test_ellipse_from_extents_takes_centre_and_semi_axes():
    ellipse = region_from_extents("ellipse", (4.0, 14.0, 5.0, 11.0), SHAPE)
    assert ellipse == Ellipse(cx=9.0, cy=8.0, rx=5.0, ry=3.0)


def test_a_click_without_a_drag_still_yields_a_usable_region():
    rect = region_from_extents("rect", (7.0, 7.0, 7.0, 7.0), SHAPE)
    assert rect == Rect(x=7, y=7, width=1, height=1)
    ellipse = region_from_extents("ellipse", (7.0, 7.0, 7.0, 7.0), SHAPE)
    assert ellipse.rx == 0.5 and ellipse.ry == 0.5


# -- region sets ------------------------------------------------------------- #


def make_set(**kwargs) -> RegionSet:
    defaults = dict(
        signal=Rect(x=2, y=2, width=4, height=4),
        noise=Rect(x=20, y=10, width=5, height=5),
        image_shape=SHAPE,
        source_file="medida_003_5mW.tif",
    )
    defaults.update(kwargs)
    return RegionSet(**defaults)  # type: ignore[arg-type]


def test_extract_returns_flat_float64_samples():
    image = np.arange(600, dtype=np.uint16).reshape(SHAPE)
    samples = make_set().extract(image)
    assert set(samples) == {"signal", "noise"}
    assert samples["signal"].ndim == 1
    assert samples["signal"].dtype == np.float64
    assert samples["signal"].size == 16
    assert samples["noise"].size == 25


def test_extract_reads_the_right_pixels():
    image = np.zeros(SHAPE, dtype=np.uint16)
    image[2:6, 2:6] = 500
    samples = make_set().extract(image)
    assert samples["signal"].mean() == 500.0
    assert samples["noise"].mean() == 0.0


def test_extract_on_a_smaller_image_reports_the_mismatch():
    small = np.zeros((10, 10), dtype=np.uint16)
    with pytest.raises(RegionOutOfBoundsError, match="noise region"):
        make_set().extract(small)


def test_extract_rejects_non_2d_images():
    with pytest.raises(RegionError, match="expected a 2-D image"):
        make_set().extract(np.zeros((2, 20, 30)))


def test_tiny_regions_are_rejected():
    with pytest.raises(EmptyRegionError, match="fewer than 2 pixels"):
        make_set(noise=Rect(x=0, y=0, width=1, height=1)).validate_for(SHAPE)


def test_region_set_round_trip():
    original = make_set(noise=Ellipse(cx=20.0, cy=10.0, rx=4.0, ry=4.0))
    assert region_set_from_dict(original.to_dict()) == original


@pytest.mark.parametrize(
    "payload",
    [
        {"regions": {"signal": {"kind": "rect", "x": 0, "y": 0, "width": 2, "height": 2}}},
        {"regions": {}, "image_shape": [10, 10]},
        {
            "regions": {
                "signal": {"kind": "rect", "x": 0, "y": 0, "width": 2, "height": 2},
                "noise": {"kind": "rect", "x": 4, "y": 4, "width": 2, "height": 2},
            }
        },
    ],
)
def test_malformed_region_sets_are_rejected(payload):
    with pytest.raises(RegionError):
        region_set_from_dict(payload)


# -- hashing ----------------------------------------------------------------- #


def test_hash_is_empty_when_there_are_no_regions():
    assert regions_hash(None) == ""


def test_hash_changes_when_geometry_changes():
    base = regions_hash(make_set())
    assert base == regions_hash(make_set())
    assert base != regions_hash(make_set(noise=Rect(x=21, y=10, width=5, height=5)))
    assert base != regions_hash(make_set(signal=Ellipse(cx=4.0, cy=4.0, rx=2.0, ry=2.0)))


def test_hash_ignores_provenance():
    """Same boxes drawn on another file must not invalidate cached results."""
    assert regions_hash(make_set(source_file="other.tif")) == regions_hash(make_set())
    assert regions_hash(make_set(image_shape=(40, 60))) == regions_hash(make_set())


def test_swapping_roles_changes_the_hash():
    swapped = make_set(signal=make_set().noise, noise=make_set().signal)
    assert regions_hash(swapped) != regions_hash(make_set())


def test_describe_is_readable():
    text = make_set().describe()
    assert "signal rect(x=2, y=2, w=4, h=4)" in text
    assert "noise rect(x=20, y=10, w=5, h=5)" in text
