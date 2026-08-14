from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import tifffile


@pytest.fixture
def make_tif():
    """Write a synthetic single-page TIFF with a known standard deviation."""

    def _make(directory: Path, name: str, values: np.ndarray | None = None, dtype=np.uint16):
        directory.mkdir(parents=True, exist_ok=True)
        if values is None:
            rng = np.random.default_rng(abs(hash(name)) % (2**32))
            values = rng.integers(0, 1000, size=(8, 8))
        array = np.asarray(values, dtype=dtype)
        path = directory / name
        tifffile.imwrite(path, array)
        return path

    return _make


@pytest.fixture
def make_stack():
    def _make(directory: Path, name: str, n_pages: int = 3, shape=(4, 4), dtype=np.uint16):
        directory.mkdir(parents=True, exist_ok=True)
        stack = np.arange(n_pages * shape[0] * shape[1], dtype=dtype).reshape(
            (n_pages, *shape)
        )
        path = directory / name
        # minisblack, so the stack is written as separate pages rather than as
        # one RGB page with separate component planes.
        tifffile.imwrite(path, stack, photometric="minisblack")
        return path

    return _make


@pytest.fixture
def series_dir(tmp_path, make_tif):
    """Four files, four distinct powers, deterministic pixel content."""
    directory = tmp_path / "data"
    for index, power in enumerate([1, 2, 5, 10], start=1):
        values = np.full((6, 6), 100, dtype=np.uint16)
        values[0, 0] = 100 + power * 10
        make_tif(directory, f"medida_{index:03d}_{power}mW.tif", values)
    return directory
