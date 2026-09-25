"""Tests for src.decide: one-to-one, thresholding, and the vectorised sweep."""

import numpy as np
import pandas as pd
import pytest

from src.decide import apply_threshold, assign_one_to_one, sweep_thresholds, tune_threshold
from src.evaluate import macro_fbeta


def _scored(rows):
    """Scored-pair frame from (s1, cand, prob) tuples."""
    return pd.DataFrame(rows, columns=["s1_id", "cand_id", "prob"])


def test_one_to_one_keeps_best_s1_per_candidate():
    """A candidate listed under two S1s goes to the higher probability only."""
    df = _scored([("S1-a", "S2-x", 0.9), ("S1-b", "S2-x", 0.7), ("S1-b", "S3-y", 0.8),
                  ("S1-c", "S3-z", 0.5), ("S1-d", "S3-z", 0.5)])
    out = assign_one_to_one(df)
    pairs = set(zip(out["s1_id"], out["cand_id"]))
    assert pairs == {("S1-a", "S2-x"), ("S1-b", "S3-y"), ("S1-c", "S3-z")}  # tie -> smaller id
    assert out["cand_id"].is_unique


def test_apply_threshold_emits_every_s1():
    """Every S1 in scope gets a row; lists sorted by probability; one-to-one applied."""
    df = _scored([("S1-a", "S2-x", 0.9), ("S1-b", "S2-x", 0.95), ("S1-a", "S3-y", 0.7),
                  ("S1-a", "S3-w", 0.4)])
    out = apply_threshold(df, tau=0.6, s1_ids=["S1-a", "S1-b", "S1-c"])
    assert out == {"S1-a": ["S3-y"], "S1-b": ["S2-x"], "S1-c": []}
    out = apply_threshold(df, tau=0.6, s1_ids=["S1-a", "S1-b"], one_to_one=False)
    assert out["S1-a"] == ["S2-x", "S3-y"]


def test_singleton_threshold():
    """An S1 whose best score is below singleton_tau is emptied."""
    df = _scored([("S1-a", "S2-x", 0.65), ("S1-b", "S2-y", 0.95), ("S1-b", "S3-z", 0.62)])
    out = apply_threshold(df, tau=0.6, singleton_tau=0.8)
    assert out == {"S1-a": [], "S1-b": ["S2-y", "S3-z"]}


def test_sweep_matches_reference_metric():
    """The vectorised sweep equals evaluate.macro_fbeta at every threshold."""
    rng = np.random.default_rng(0)
    s1 = [f"S1-{i}" for i in range(60)]
    truth = {s: ([f"S2-{i}"] if i % 7 else []) + ([f"S3-{i}"] if i % 3 == 0 else [])
             for i, s in enumerate(s1)}
    rows = []
    for i, s in enumerate(s1[:50]):  # S1s 50-59 have no candidates at all
        for c in [f"S2-{i}", f"S3-{i}", f"S2-{(i + 1) % 60}", f"S3-orphan{i}"]:
            rows.append((s, c, float(rng.random())))
    df = _scored(rows)
    taus = [0.2, 0.5, 0.8]
    for one_to_one in (True, False):
        grid = sweep_thresholds(df, truth, s1, taus, one_to_one=one_to_one)
        for tau, got in zip(taus, grid["macro_f05"]):
            pred = apply_threshold(df, tau, s1_ids=s1, singleton_tau=None, one_to_one=one_to_one)
            assert got == pytest.approx(macro_fbeta(pred, truth, s1_ids=s1)), (tau, one_to_one)
    grid = sweep_thresholds(df, truth, s1, [0.5], singleton_taus=[0.9])
    pred = apply_threshold(df, 0.5, s1_ids=s1, singleton_tau=0.9)
    assert grid["macro_f05"][0] == pytest.approx(macro_fbeta(pred, truth, s1_ids=s1))


def test_tune_threshold_prefers_precision_and_counts_orphans():
    """Orphan false positives at mid scores push the best tau above them."""
    truth = {"S1-1": ["S2-1"], "S1-2": ["S2-2"], "S1-3": [], "S1-4": []}
    df = _scored([("S1-1", "S2-1", 0.9), ("S1-2", "S2-2", 0.85),
                  ("S1-3", "S3-orphan1", 0.7), ("S1-4", "S3-orphan2", 0.72)])
    tau, stau, score, grid = tune_threshold(df, truth, list(truth), taus=[0.5, 0.6, 0.75, 0.95])
    assert tau == 0.75 and score == pytest.approx(1.0)
    assert {"tau", "singleton_tau", "macro_f05", "singleton_f05"} <= set(grid.columns)


def test_all_empty_baseline():
    """Nothing above tau -> score equals the singleton share."""
    truth = {"a": [], "b": ["S2-1"], "c": ["S2-2"], "d": ["S3-3"]}
    grid = sweep_thresholds(_scored([("b", "S2-1", 0.1)]), truth, list(truth), [0.5])
    assert grid["macro_f05"][0] == pytest.approx(0.25)
