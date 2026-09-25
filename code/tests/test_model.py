"""Tests for src.model on synthetic features."""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score

from src.blocking import generate_candidates
from src.features import build_features, feature_columns, label_pairs
from src.model import (
    feature_importance,
    group_folds,
    load_model,
    predict,
    save_model,
    train_full,
    train_oof,
)
from src.normalize import normalize_frame
from tests.synthetic import make_dataset

SMALL = {"objective": "binary", "learning_rate": 0.1, "num_leaves": 15,
         "min_child_samples": 5, "verbose": -1, "seed": 42, "deterministic": True,
         "num_threads": 1}


@pytest.fixture(scope="module")
def data():
    """Blocked + featurised synthetic dataset with labels."""
    s1, s2, s3, truth = make_dataset(n_s1=200, seed=5)
    s1n = normalize_frame(s1)
    on = normalize_frame(pd.concat([s2, s3], ignore_index=True))
    cands = generate_candidates(s1n, on, verbose=False)
    feats = build_features(cands, s1n, on, verbose=False)
    feats["label"] = label_pairs(feats, truth)
    return feats


def test_group_folds_never_split_a_group():
    """Every group lands in exactly one fold."""
    groups = np.repeat([f"S1-{i}" for i in range(23)], np.arange(1, 24) % 5 + 1)
    folds = group_folds(groups, 5)
    per_group = pd.Series(folds).groupby(groups).nunique()
    assert (per_group == 1).all()
    assert set(folds) == set(range(5))
    assert group_folds(["a", "a"], 5).tolist() == [0, 0]  # one group -> one fold


def test_train_oof_learns_and_is_deterministic(data):
    """OOF AUC is high on easy synthetic data; reruns are identical (seeded)."""
    cols = feature_columns(data)
    X, y, g = data[cols], data["label"].to_numpy(), data["s1_id"]
    oof, models = train_oof(X, y, g, n_folds=3, params=SMALL, num_boost_round=200,
                            early_stopping=20, verbose=False)
    assert len(models) == 3 and oof.shape == (len(X),)
    assert 0 <= oof.min() and oof.max() <= 1
    assert roc_auc_score(y, oof) > 0.95
    oof2, _ = train_oof(X, y, g, n_folds=3, params=SMALL, num_boost_round=200,
                        early_stopping=20, verbose=False)
    np.testing.assert_array_equal(oof, oof2)


def test_train_full_save_load_roundtrip(data, tmp_path):
    """Full model predicts the same after save/load; importance lists every feature."""
    cols = feature_columns(data)
    X, y = data[cols], data["label"].to_numpy()
    _, models = train_oof(X, y, data["s1_id"], n_folds=3, params=SMALL,
                          num_boost_round=100, early_stopping=10, verbose=False)
    model = train_full(X, y, params=SMALL, fold_models=models)
    p = predict(model, X)
    save_model(model, tmp_path / "m.txt")
    np.testing.assert_allclose(predict(load_model(tmp_path / "m.txt"), X), p, rtol=1e-6)
    imp = feature_importance(model)
    assert set(imp["feature"]) == set(cols)
    # Averaging a list of fold models works and column order does not matter.
    assert predict(models, X[cols[::-1]]).shape == (len(X),)
