"""LightGBM pair classifier: training, OOF predictions, save/load (CLAUDE.md §6.4).

LightGBM is MIT-licensed and trained from scratch on the provided data (see MODELS.md).
Cross-validation uses GroupKFold grouped by S1 ID, so an S1 never appears in both a
training and a validation fold; its out-of-fold predictions feed the threshold sweep
in ``decide.py``.

The model trains on whatever numeric columns ``features.feature_columns`` returns, so
the same code runs on synthetic test fixtures now and on real SageMaker data later.
"""

from __future__ import annotations

import time
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from . import config


def _log(msg: str) -> None:
    """Print a timestamped progress line."""
    print(f"[model {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def group_folds(groups: pd.Series | np.ndarray, n_folds: int = config.N_FOLDS) -> np.ndarray:
    """Return a fold id per row such that every group sits in exactly one fold.

    Uses sklearn's deterministic GroupKFold. ``n_folds`` is capped at the number of
    distinct groups.
    """
    groups = np.asarray(groups)
    n = min(n_folds, len(np.unique(groups)))
    fold = np.full(len(groups), -1, dtype=np.int16)
    if n < 2:
        return np.zeros(len(groups), dtype=np.int16)
    for i, (_, va) in enumerate(GroupKFold(n_splits=n).split(groups, groups=groups)):
        fold[va] = i
    return fold


def train_oof(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: pd.Series | np.ndarray,
    n_folds: int = config.N_FOLDS,
    params: dict | None = None,
    num_boost_round: int = config.LGB_NUM_BOOST_ROUND,
    early_stopping: int = config.LGB_EARLY_STOPPING,
    verbose: bool = True,
) -> tuple[np.ndarray, list[lgb.Booster]]:
    """Train one model per GroupKFold fold and return out-of-fold probabilities.

    Args:
        X: feature matrix (float32 columns).
        y: 0/1 labels.
        groups: S1 ID per row. Folds never split a group.
        n_folds: number of folds.
        params: LightGBM params (defaults to ``config.LGB_PARAMS``).
        num_boost_round / early_stopping: boosting rounds, stopped on the fold's
            validation log-loss.
        verbose: print per-fold progress.

    Returns:
        ``(oof_probabilities, fold_models)``. Each fold model's ``best_iteration``
        is set.
    """
    params = dict(config.LGB_PARAMS if params is None else params)
    y = np.asarray(y)
    folds = group_folds(groups, n_folds)
    oof = np.zeros(len(X), dtype=np.float32)
    models: list[lgb.Booster] = []
    for f in range(int(folds.max()) + 1):
        tr, va = folds != f, folds == f
        if not tr.any():  # a single group: nothing to train on out-of-fold
            continue
        dtr = lgb.Dataset(X[tr], y[tr], free_raw_data=True)
        dva = lgb.Dataset(X[va], y[va], reference=dtr)
        model = lgb.train(params, dtr, num_boost_round=num_boost_round, valid_sets=[dva],
                          callbacks=[lgb.early_stopping(early_stopping, verbose=False)])
        oof[va] = model.predict(X[va], num_iteration=model.best_iteration)
        models.append(model)
        if verbose:
            _log(f"fold {f}: {tr.sum():,} train / {va.sum():,} valid rows, "
                 f"best_iter={model.best_iteration}")
    return oof, models


def train_full(
    X: pd.DataFrame,
    y: np.ndarray,
    num_boost_round: int | None = None,
    params: dict | None = None,
    fold_models: list[lgb.Booster] | None = None,
) -> lgb.Booster:
    """Train the final model on all labelled pairs.

    The number of rounds defaults to the mean fold ``best_iteration`` scaled by
    ``n_folds / (n_folds - 1)`` (more data -> more rounds), or
    ``config.LGB_NUM_BOOST_ROUND // 4`` without fold models.
    """
    params = dict(config.LGB_PARAMS if params is None else params)
    if num_boost_round is None:
        if fold_models:
            k = len(fold_models)
            mean_best = np.mean([max(m.best_iteration, 1) for m in fold_models])
            num_boost_round = max(1, int(round(mean_best * k / max(k - 1, 1))))
        else:
            num_boost_round = config.LGB_NUM_BOOST_ROUND // 4
    return lgb.train(params, lgb.Dataset(X, np.asarray(y)), num_boost_round=num_boost_round)


def predict(model: lgb.Booster | list[lgb.Booster], X: pd.DataFrame) -> np.ndarray:
    """Match probabilities; a list of models is averaged."""
    models = model if isinstance(model, list) else [model]
    preds = [m.predict(X[m.feature_name()], num_iteration=m.best_iteration or None)
             for m in models]
    return np.mean(preds, axis=0).astype(np.float32)


def feature_importance(model: lgb.Booster) -> pd.DataFrame:
    """Gain and split importance per feature, sorted by gain."""
    return pd.DataFrame({
        "feature": model.feature_name(),
        "gain": model.feature_importance("gain"),
        "split": model.feature_importance("split"),
    }).sort_values("gain", ascending=False).reset_index(drop=True)


def save_model(model: lgb.Booster, path: Path) -> None:
    """Persist a trained model (text format) under ``artifacts/``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(path))


def load_model(path: Path) -> lgb.Booster:
    """Load a model saved by :func:`save_model`."""
    return lgb.Booster(model_file=str(path))
