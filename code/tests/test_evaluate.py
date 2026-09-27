"""Tests for src.evaluate (macro F0.5 and blocking metrics)."""

import pytest

from src.evaluate import (
    blocking_recall,
    fbeta_single,
    macro_fbeta,
    reduction_ratio,
    score_report,
)


def test_worked_example():
    """pred [47,193,812] vs truth [47,812] -> 0.714 (P=2/3, R=1)."""
    assert fbeta_single(["47", "193", "812"], ["47", "812"]) == pytest.approx(0.714, abs=5e-4)
    assert fbeta_single(["47", "193", "812"], ["47", "812"]) == pytest.approx(5 / 7)


@pytest.mark.parametrize(
    "pred, truth, expected",
    [
        ([], [], 1.0),                # singleton, correctly empty
        (["S2-1"], [], 0.0),          # singleton, any prediction
        (["S2-1", "S3-2"], [], 0.0),
        ([], ["S2-1"], 0.0),          # missed all matches
        (["S2-9"], ["S2-1"], 0.0),    # no overlap
        (["S2-1"], ["S2-1"], 1.0),    # exact
    ],
)
def test_singleton_and_edge_cases(pred, truth, expected):
    """Singleton rules from the challenge metric plus exact/no-overlap cases."""
    assert fbeta_single(pred, truth) == expected


def test_precision_weighted_over_recall():
    """F0.5 punishes a false positive more than a false negative."""
    truth = ["a", "b"]
    assert fbeta_single(["a"], truth) > fbeta_single(["a", "b", "x"], truth)


def test_duplicates_in_pred_ignored():
    """Duplicate predicted IDs do not change the score."""
    assert fbeta_single(["a", "a", "b"], ["a", "b"]) == 1.0


def test_macro_average_includes_singletons_and_missing_preds():
    """Macro average over the S1 universe; missing preds are empty lists."""
    truth = {"S1-1": ["47", "812"], "S1-2": [], "S1-3": ["5"]}
    pred = {"S1-1": ["47", "193", "812"], "S1-2": []}  # S1-3 missing -> empty -> 0
    assert macro_fbeta(pred, truth) == pytest.approx((5 / 7 + 1.0 + 0.0) / 3)


def test_macro_with_explicit_universe():
    """S1s outside ``truth`` but in ``s1_ids`` are singletons."""
    truth = {"S1-1": ["a"]}
    pred = {"S1-1": ["a"], "S1-2": ["b"]}
    assert macro_fbeta(pred, truth, s1_ids=["S1-1", "S1-2", "S1-3"]) == pytest.approx(2 / 3)


def test_score_report_breakdown():
    """Report splits singletons vs non-singletons and per group."""
    truth = {"S1-1": ["a", "b"], "S1-2": []}
    pred = {"S1-1": ["a"], "S1-2": ["z"]}
    rep = score_report(pred, truth, groups={"S1-1": "US", "S1-2": "India"})
    f_a = fbeta_single(["a"], ["a", "b"])
    assert rep["macro_f05"] == pytest.approx(f_a / 2)
    assert rep["singleton_f05"] == 0.0
    assert rep["non_singleton_f05"] == pytest.approx(f_a)
    assert rep["pair_precision"] == pytest.approx(0.5)
    assert rep["pair_recall"] == pytest.approx(0.5)
    assert rep["f05_US"] == pytest.approx(f_a)
    assert rep["f05_India"] == 0.0


def test_blocking_recall_and_reduction_ratio():
    """Pair recall, full-S1 recall and mean candidate count."""
    truth = {"S1-1": ["a", "b"], "S1-2": ["c"], "S1-3": []}
    cands = {"S1-1": ["a", "x"], "S1-2": ["c", "y", "z"], "S1-3": ["q"]}
    rep = blocking_recall(cands, truth)
    assert rep["pair_recall"] == pytest.approx(2 / 3)
    assert rep["s1_full_recall"] == pytest.approx(0.5)
    assert rep["mean_candidates"] == pytest.approx(2.0)
    assert reduction_ratio(6, 3, 10) == pytest.approx(0.8)
