"""Thresholding and one-to-one assignment -> final match lists (CLAUDE.md §6.5).

* **One-to-one** (confirmed in the CP1 audit: no S2/S3 record matches more than one S1
  in train). Each candidate is kept only under the S1 where its probability is
  highest, before thresholding. This alone removes many false merges when chain names
  put one record on several S1 candidate lists.
* **Threshold tau** is tuned to maximise *macro F0.5 including singletons*, not
  pair-level F1. The sweep scores every S1 in scope: S1s with no candidates, true
  singletons (5.58% of train), and the false positives that orphan S2/S3 records
  (~25% of them have no S1 parent) cause. A precision-heavy metric plus orphans means
  the best tau is expected above 0.5.
* **Optional singleton threshold**: an S1 whose top probability is below
  ``singleton_tau`` gets an empty list even if some pairs pass tau.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from . import config


def assign_one_to_one(scored: pd.DataFrame) -> pd.DataFrame:
    """Keep each ``cand_id`` only under the S1 where its ``prob`` is highest.

    Ties go to the smallest ``s1_id`` (deterministic). Returns the reduced frame with
    the original index.
    """
    if scored.empty:
        return scored
    order = scored.sort_values(["cand_id", "prob", "s1_id"], ascending=[True, False, True],
                               kind="mergesort")
    return scored.loc[order.drop_duplicates("cand_id", keep="first").index].sort_index()


def apply_threshold(
    scored: pd.DataFrame,
    tau: float = config.MATCH_THRESHOLD,
    s1_ids: Iterable[str] | None = None,
    singleton_tau: float | None = config.SINGLETON_THRESHOLD,
    one_to_one: bool = config.ONE_TO_ONE,
) -> dict[str, list[str]]:
    """Final ``{s1_id: [matched ids]}`` from scored pairs.

    Args:
        scored: frame with ``s1_id``, ``cand_id``, ``prob``.
        tau: keep pairs with ``prob >= tau``.
        s1_ids: every S1 to emit (S1s without kept pairs get ``[]``). Defaults to
            the S1s present in ``scored``.
        singleton_tau: if set, S1s whose top probability is below it get ``[]``.
        one_to_one: enforce :func:`assign_one_to_one` first.

    Returns:
        Matches ordered by descending probability.
    """
    df = assign_one_to_one(scored) if one_to_one else scored
    keep = df["prob"] >= tau
    if singleton_tau is not None:
        top = scored.groupby("s1_id")["prob"].transform("max")
        keep &= top.loc[df.index] >= singleton_tau
    kept = df[keep].sort_values(["s1_id", "prob", "cand_id"], ascending=[True, False, True])
    out = {s: list(g) for s, g in kept.groupby("s1_id", sort=False)["cand_id"]}
    universe = scored["s1_id"].unique() if s1_ids is None else s1_ids
    return {s: out.get(s, []) for s in universe}


def sweep_thresholds(
    scored: pd.DataFrame,
    truth: Mapping[str, Sequence[str]],
    s1_ids: Iterable[str],
    taus: Sequence[float] = config.TAU_GRID,
    singleton_taus: Sequence[float | None] = (None,),
    one_to_one: bool = config.ONE_TO_ONE,
    beta: float = config.BETA,
) -> pd.DataFrame:
    """Macro F-beta (singletons included) for every ``(tau, singleton_tau)`` pair.

    Vectorised with ``np.bincount``, so a full grid over tens of millions of OOF pairs
    is cheap. ``truth`` gives the *full* true lists, so true matches that blocking
    missed still count as false negatives.

    Returns one row per grid point: ``tau``, ``singleton_tau``, ``macro_f05``,
    ``precision`` / ``recall`` (pair-level), ``singleton_f05``, ``non_singleton_f05``
    and ``mean_pred``.
    """
    universe = pd.Index(pd.unique(pd.Series(list(s1_ids), dtype=object)))
    n = len(universe)
    n_true = np.array([len(set(truth.get(s, ()))) for s in universe], dtype=np.float64)
    df = assign_one_to_one(scored) if one_to_one else scored
    code = universe.get_indexer(df["s1_id"])
    ok = code >= 0
    code, prob = code[ok], df["prob"].to_numpy(np.float64)[ok]
    true_sets = {s: set(truth.get(s, ())) for s in universe}
    label = np.fromiter((c in true_sets[s] for s, c in
                         zip(df["s1_id"].to_numpy()[ok], df["cand_id"].to_numpy()[ok])),
                        dtype=np.float64, count=int(ok.sum()))
    top = np.full(n, -np.inf)
    # Singleton gate uses the S1's top probability before one-to-one assignment.
    all_code = universe.get_indexer(scored["s1_id"])
    m = all_code >= 0
    np.maximum.at(top, all_code[m], scored["prob"].to_numpy(np.float64)[m])
    b2 = beta * beta
    singleton = n_true == 0
    rows = []
    for stau in singleton_taus:
        gate = np.ones(n, bool) if stau is None else top >= stau
        for tau in taus:
            keep = (prob >= tau) & gate[code]
            npred = np.bincount(code, weights=keep, minlength=n)
            tp = np.bincount(code, weights=keep * label, minlength=n)
            with np.errstate(divide="ignore", invalid="ignore"):
                p = np.where(npred > 0, tp / npred, 0.0)
                r = np.where(n_true > 0, tp / n_true, 0.0)
                f = np.where(tp > 0, (1 + b2) * p * r / (b2 * p + r), 0.0)
            f = np.where(singleton, (npred == 0).astype(float), f)
            rows.append({
                "tau": float(tau),
                "singleton_tau": stau,
                "macro_f05": float(f.mean()) if n else 0.0,
                "precision": float(tp.sum() / npred.sum()) if npred.sum() else 0.0,
                "recall": float(tp.sum() / n_true.sum()) if n_true.sum() else 0.0,
                "singleton_f05": float(f[singleton].mean()) if singleton.any() else np.nan,
                "non_singleton_f05": float(f[~singleton].mean()) if (~singleton).any() else np.nan,
                "mean_pred": float(npred.mean()) if n else 0.0,
            })
    return pd.DataFrame(rows)


def tune_threshold(
    scored: pd.DataFrame,
    truth: Mapping[str, Sequence[str]],
    s1_ids: Iterable[str],
    taus: Sequence[float] = config.TAU_GRID,
    singleton_taus: Sequence[float | None] | None = None,
    one_to_one: bool = config.ONE_TO_ONE,
) -> tuple[float, float | None, float, pd.DataFrame]:
    """Pick the grid point with the best macro F0.5 on OOF predictions.

    ``singleton_taus`` defaults to ``None`` plus every tau value. Ties go to the
    smaller threshold pair (grid order), so results are deterministic.

    Returns ``(best_tau, best_singleton_tau, best_macro_f05, full_grid)``.
    """
    if singleton_taus is None:
        singleton_taus = [None, *taus]
    grid = sweep_thresholds(scored, truth, s1_ids, taus, singleton_taus, one_to_one)
    best = grid.loc[grid["macro_f05"].idxmax()]
    stau = best["singleton_tau"]
    stau = None if stau is None or (isinstance(stau, float) and np.isnan(stau)) else float(stau)
    return float(best["tau"]), stau, float(best["macro_f05"]), grid
