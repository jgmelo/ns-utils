from __future__ import annotations

import numpy as np
import pytest
import tifffile

from tif_analyzer import loading
from tif_analyzer.loading import (
    ColorImageError,
    LoadError,
    MultiPageError,
    load_image,
)


def test_single_page_is_read_as_float64_without_rescaling(tmp_path, make_tif):
    values = np.array([[0, 1000], [65535, 42]], dtype=np.uint16)
    path = make_tif(tmp_path, "medida_001_1mW.tif", values)
    image = load_image(path)
    assert image.dtype == np.float64
    np.testing.assert_array_equal(image, values.astype(np.float64))


def test_32bit_float_tif_round_trips(tmp_path):
    values = np.array([[0.5, -3.25], [1e6, 2.0]], dtype=np.float32)
    path = tmp_path / "medida_001_1mW.tif"
    tifffile.imwrite(path, values)
    np.testing.assert_allclose(load_image(path), values.astype(np.float64))


def test_multipage_errors_by_default(tmp_path, make_stack):
    path = make_stack(tmp_path, "medida_001_1mW.tif", n_pages=3)
    with pytest.raises(MultiPageError, match="3 pages"):
        load_image(path)


def test_multipage_first_policy(tmp_path, make_stack):
    path = make_stack(tmp_path, "medida_001_1mW.tif", n_pages=3, shape=(2, 2))
    image = load_image(path, "first")
    np.testing.assert_array_equal(image, np.arange(4, dtype=np.float64).reshape(2, 2))


def test_multipage_mean_policy(tmp_path, make_stack):
    path = make_stack(tmp_path, "medida_001_1mW.tif", n_pages=3, shape=(2, 2))
    expected = np.arange(12, dtype=np.float64).reshape(3, 2, 2).mean(axis=0)
    np.testing.assert_allclose(load_image(path, "mean"), expected)


def test_mean_policy_does_not_overflow_integer_pixels(tmp_path):
    pages = np.full((4, 2, 2), 65535, dtype=np.uint16)
    path = tmp_path / "medida_001_1mW.tif"
    tifffile.imwrite(path, pages, photometric="minisblack")
    np.testing.assert_allclose(load_image(path, "mean"), np.full((2, 2), 65535.0))


def test_colour_image_is_rejected(tmp_path):
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    path = tmp_path / "medida_001_1mW.tif"
    tifffile.imwrite(path, rgb, photometric="rgb")
    with pytest.raises(ColorImageError):
        load_image(path)


def test_unknown_policy_is_a_programming_error(tmp_path, make_tif):
    path = make_tif(tmp_path, "medida_001_1mW.tif")
    with pytest.raises(ValueError, match="unknown page policy"):
        load_image(path, "average")  # type: ignore[arg-type]


def test_corrupt_file_raises_load_error(tmp_path):
    path = tmp_path / "medida_001_1mW.tif"
    path.write_bytes(b"definitely not a tiff")
    with pytest.raises(LoadError):
        load_image(path)


def test_missing_file_raises_load_error(tmp_path):
    with pytest.raises(LoadError):
        load_image(tmp_path / "absent.tif")


def test_page_count(tmp_path, make_stack):
    path = make_stack(tmp_path, "s.tif", n_pages=5)
    assert loading.page_count(path) == 5
