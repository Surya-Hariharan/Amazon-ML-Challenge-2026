"""Tests for src.io_utils.write_submission and ID-list helpers."""

import pandas as pd
import pytest

from src.io_utils import format_id_list, parse_id_list, read_tsv, write_submission

S1 = ["S1-1", "S1-2", "S1-3"]
VALID = {"S2-10", "S2-11", "S3-20", "S3-21"}


def _write(tmp_path, matches, candidates, s1_ids=S1, valid=VALID):
    """Call write_submission into tmp_path."""
    return write_submission(matches, candidates, s1_ids, valid, out_dir=tmp_path)


def test_happy_path_format(tmp_path):
    """Headers, one row per S1 in order, comma lists, empty string for no match."""
    m_path, c_path = _write(
        tmp_path,
        matches={"S1-1": ["S2-10", "S3-20"]},
        candidates={"S1-1": ["S2-10", "S3-20", "S2-11"], "S1-2": ["S3-21"]},
    )
    assert m_path.read_bytes() == (
        b"source1_entity_id\tmatched_entity_ids\n"
        b"S1-1\tS2-10,S3-20\n"
        b"S1-2\t\n"
        b"S1-3\t\n"
    )
    assert c_path.read_bytes() == (
        b"source1_entity_id\tcandidate_entity_ids\n"
        b"S1-1\tS2-10,S3-20,S2-11\n"
        b"S1-2\tS3-21\n"
        b"S1-3\t\n"
    )
    df = read_tsv(m_path)
    assert list(df.columns) == ["source1_entity_id", "matched_entity_ids"]
    assert df["source1_entity_id"].tolist() == S1
    assert df["matched_entity_ids"].tolist() == ["S2-10,S3-20", "", ""]


def test_duplicate_ids_in_list_are_dropped(tmp_path):
    """Duplicates within a list are removed, order kept."""
    m_path, c_path = _write(
        tmp_path,
        matches={"S1-1": ["S2-10", "S2-10"]},
        candidates={"S1-1": ["S2-10", "S2-10", "S3-20"]},
    )
    assert read_tsv(m_path)["matched_entity_ids"][0] == "S2-10"
    assert read_tsv(c_path)["candidate_entity_ids"][0] == "S2-10,S3-20"


@pytest.mark.parametrize(
    "matches, candidates, s1_ids, match",
    [
        ({"S1-1": ["S2-10"]}, {"S1-1": ["S3-20"]}, S1, "not in its candidates"),
        ({}, {"S1-1": ["S1-2"]}, S1, "not an S2-/S3- ID"),
        ({}, {"S1-1": ["S2-99"]}, S1, "not in the source files"),
        ({}, {"S1-1": ["S2-10 "]}, S1, None),  # stripped -> valid, no error
        ({}, {"S1-9": ["S2-10"]}, S1, "not in s1_ids"),
        ({"S1-9": []}, {}, S1, "not in s1_ids"),
        ({}, {}, ["S1-1", "S1-1"], "duplicates"),
        ({}, {}, ["S2-10"], "non-S1"),
    ],
)
def test_rule_violations_raise(tmp_path, matches, candidates, s1_ids, match):
    """Every format rule violation raises ValueError and writes nothing."""
    if match is None:
        _write(tmp_path, matches, candidates, s1_ids)
        return
    with pytest.raises(ValueError, match=match):
        _write(tmp_path, matches, candidates, s1_ids)
    assert not any(tmp_path.iterdir())


def test_id_with_separator_rejected(tmp_path):
    """IDs containing commas/quotes/inner spaces are rejected even if 'valid'."""
    bad = "S2-1,S2-2"
    with pytest.raises(ValueError, match="separator or quote"):
        _write(tmp_path, {}, {"S1-1": [bad]}, valid=VALID | {bad})


def test_round_trip_matches_s1_ids(tmp_path):
    """Output has exactly one row per S1 and no duplicate S1 rows."""
    s1 = [f"S1-{i}" for i in range(50)]
    m_path, _ = _write(tmp_path, {}, {}, s1_ids=s1)
    df = read_tsv(m_path)
    assert len(df) == 50 and df["source1_entity_id"].is_unique
    assert (df["matched_entity_ids"] == "").all()


def test_id_list_helpers():
    """parse/format are inverses and handle the empty cell."""
    assert parse_id_list("") == []
    assert parse_id_list("S2-1,S3-2") == ["S2-1", "S3-2"]
    assert format_id_list(["S2-1", "S3-2", "S2-1"]) == "S2-1,S3-2"
    assert format_id_list([]) == ""


def test_read_tsv_keeps_strings(tmp_path):
    """IDs stay strings and empty cells stay empty strings, not NaN."""
    p = tmp_path / "x.tsv"
    p.write_text("entity_id\tbusiness_name\n007\t\n", encoding="utf-8")
    df = read_tsv(p)
    assert df["entity_id"][0] == "007"
    assert df["business_name"][0] == ""
    assert isinstance(df, pd.DataFrame)
