from __future__ import annotations

from pathlib import Path

import pytest

from tif_analyzer import discovery
from tif_analyzer.discovery import (
    DEFAULT_PATTERN,
    DiscoveryError,
    DuplicateParameterError,
    PatternError,
    discover,
    find_duplicates,
    infer_unit,
    validate_pattern,
)


def touch(directory: Path, *names: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name in names:
        (directory / name).write_bytes(b"")


# -- pattern validation ------------------------------------------------------ #


def test_default_pattern_matches_spec_examples():
    compiled = validate_pattern(DEFAULT_PATTERN)
    for name, expected in [
        ("medida_001_1mW.tif", 1.0),
        ("medida_002_2mW.tif", 2.0),
        ("medida_017_12.5mW.tiff", 12.5),
    ]:
        match = compiled.search(name)
        assert match is not None, name
        assert float(match.group("power")) == expected


def test_pattern_without_named_group_is_rejected():
    with pytest.raises(PatternError, match="no named group"):
        validate_pattern(r"medida_\d+_[\d.]+mW\.tif$")


def test_pattern_with_two_named_groups_is_rejected():
    with pytest.raises(PatternError, match="ambiguous"):
        validate_pattern(r"medida_(?P<index>\d+)_(?P<power>[\d.]+)mW\.tif$")


def test_uncompilable_pattern_is_rejected():
    with pytest.raises(PatternError, match="not a valid regex"):
        validate_pattern(r"medida_(?P<power>[\d.+mW")


@pytest.mark.parametrize(
    "pattern,expected",
    [
        (DEFAULT_PATTERN, "mW"),
        (r"(?P<exposure>[\d.]+)ms\.tiff?$", "ms"),
        (r"(?P<t>[\d.]+)\.tif$", None),
        (r"(?P<a>(?:[\d.]+))uW\.tif$", "uW"),
        (r"(?P<frac>[\d.]+)%\.tif$", "%"),
    ],
)
def test_infer_unit(pattern, expected):
    assert infer_unit(pattern) == expected


# -- discovery counts -------------------------------------------------------- #


def test_three_counts_are_distinguished(tmp_path):
    touch(
        tmp_path,
        "medida_001_1mW.tif",
        "medida_002_2mW.tif",
        "notes.tif",  # glob yes, regex no
        "medida_003_..mW.tif",  # regex yes, float no
        "ignored.txt",  # glob no
    )
    report = discover(tmp_path, "*.tif", DEFAULT_PATTERN)
    assert report.n_globbed == 4
    assert report.n_regex_matched == 3
    assert report.n_parsed == 2
    assert [p.name for p in report.unmatched] == ["notes.tif"]
    assert [f.path.name for f in report.parse_failures] == ["medida_003_..mW.tif"]
    assert report.param_name == "power"


def test_records_are_sorted_by_parameter(tmp_path):
    touch(tmp_path, "medida_001_10mW.tif", "medida_002_2mW.tif", "medida_003_1.5mW.tif")
    report = discover(tmp_path, "*.tif", DEFAULT_PATTERN)
    assert [r.param_value for r in report.records] == [1.5, 2.0, 10.0]
    assert report.records[0].raw_match == "1.5"


def test_glob_filter_is_honoured(tmp_path):
    touch(tmp_path, "medida_001_1mW.tif", "medida_002_2mW.tiff")
    assert discover(tmp_path, "*.tif", DEFAULT_PATTERN).n_globbed == 1
    assert discover(tmp_path, "*.tif*", DEFAULT_PATTERN).n_globbed == 2


def test_missing_directory_raises(tmp_path):
    with pytest.raises(DiscoveryError, match="not a directory"):
        discover(tmp_path / "nope")


# -- duplicates -------------------------------------------------------------- #


def test_duplicates_are_grouped_by_parsed_float(tmp_path):
    touch(tmp_path, "medida_001_5mW.tif", "medida_002_5.00mW.tif", "medida_003_05mW.tif")
    with pytest.raises(DuplicateParameterError) as excinfo:
        discover(tmp_path, "*.tif", DEFAULT_PATTERN)
    duplicates = excinfo.value.duplicates
    assert list(duplicates) == [5.0]
    assert len(duplicates[5.0]) == 3
    # The report travels with the error so the UI can still show counts.
    assert excinfo.value.report.n_parsed == 3


def test_all_collisions_reported_at_once(tmp_path):
    touch(
        tmp_path,
        "medida_001_1mW.tif",
        "medida_002_1.0mW.tif",
        "medida_003_3mW.tif",
        "medida_004_3mW.tif",
        "medida_005_7mW.tif",
    )
    with pytest.raises(DuplicateParameterError) as excinfo:
        discover(tmp_path, "*.tif", DEFAULT_PATTERN)
    assert sorted(excinfo.value.duplicates) == [1.0, 3.0]


def test_duplicates_allowed_by_toggle(tmp_path):
    touch(tmp_path, "medida_001_5mW.tif", "medida_002_5.0mW.tif")
    report = discover(tmp_path, "*.tif", DEFAULT_PATTERN, allow_duplicates=True)
    assert report.n_parsed == 2
    assert list(report.duplicates) == [5.0]


def test_duplicate_check_does_no_file_io(tmp_path, monkeypatch):
    touch(tmp_path, "medida_001_5mW.tif", "medida_002_5.0mW.tif")

    def explode(*args, **kwargs):  # pragma: no cover - must never run
        raise AssertionError("discovery must not open files")

    monkeypatch.setattr("tifffile.TiffFile", explode)
    with pytest.raises(DuplicateParameterError):
        discover(tmp_path, "*.tif", DEFAULT_PATTERN)


def test_find_duplicates_returns_empty_when_unique():
    from tif_analyzer.discovery import FileRecord

    records = [
        FileRecord(Path("a.tif"), "power", 1.0, "1"),
        FileRecord(Path("b.tif"), "power", 2.0, "2"),
    ]
    assert find_duplicates(records) == {}


def test_example_patterns_all_validate():
    for _, pattern in discovery.EXAMPLE_PATTERNS:
        validate_pattern(pattern)
