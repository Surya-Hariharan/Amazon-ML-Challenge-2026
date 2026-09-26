"""Label-free train-vs-test distribution comparison (roadmap v3 Phase 0, E-drift).

Every covariate here is computable from S1/S2/S3 fields and
``normalize.normalize_frame`` alone -- no ground truth is used, so this is
safe to run on the real, unlabelled test split. Distinguishes STATISTICAL
difference (KS test) from PRACTICAL / potentially relevant distribution shift
(Cohen's d effect size) per roadmap v3's instruction not to rely on p-values
alone: a large sample can make a trivial shift "significant" (tiny p-value,
tiny d), and a small slice can hide a real shift behind a large p-value.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

import numpy as np
import pandas as pd
from scipy import stats

from . import config
from .normalize import normalize_frame

NON_LATIN_RE = re.compile(r"[^\x00-\x7F]")

_QUANTILES = (0.05, 0.25, 0.50, 0.75, 0.95)


def record_covariates(raw: pd.DataFrame) -> pd.DataFrame:
    """Return one row of label-free covariates per record of a raw source frame.

    Columns: ``entity_id``, ``country``, ``name_len``/``addr_len`` (raw
    character lengths), ``name_missing``/``addr_missing``, ``name_tokens``/
    ``addr_tokens`` (token counts on the *normalised* fields), ``digit_count``,
    ``has_long_num`` (a 5+ digit token present -- ZIP/PIN/CP proxy),
    ``non_latin_name``/``non_latin_addr`` (raw text outside ASCII -- a
    script/transliteration proxy), ``name_core_len``/``addr_norm_len``
    (post-normalisation lengths).
    """
    n = normalize_frame(raw)
    name_raw = raw[config.NAME_COL].fillna("")
    addr_raw = raw[config.ADDRESS_COL].fillna("")
    return pd.DataFrame({
        "entity_id": raw[config.ID_COL].to_numpy(),
        "country": raw[config.COUNTRY_COL].to_numpy(),
        "name_len": name_raw.str.len(),
        "addr_len": addr_raw.str.len(),
        "name_missing": (name_raw.str.len() == 0).astype(int),
        "addr_missing": (addr_raw.str.len() == 0).astype(int),
        "name_tokens": n["name_norm"].str.split().map(len),
        "addr_tokens": n["addr_norm"].str.split().map(len),
        "digit_count": n["digits"].str.split().map(len),
        "has_long_num": (n["long_nums"].str.len() > 0).astype(int),
        "non_latin_name": name_raw.map(lambda s: bool(NON_LATIN_RE.search(s))).astype(int),
        "non_latin_addr": addr_raw.map(lambda s: bool(NON_LATIN_RE.search(s))).astype(int),
        "name_core_len": n["name_core"].str.len(),
        "addr_norm_len": n["addr_norm"].str.len(),
    })


def candidate_covariates(cands: pd.DataFrame, s1_ids: Sequence[str]) -> pd.DataFrame:
    """Per-S1 candidate-count and top-candidate-score covariates.

    ``cands`` is a blocking output (``blocking.generate_candidates``) or a
    scored frame; ``top_score`` takes the row-wise max of ``prob`` and any
    ``score_*`` column present, as a rough top-candidate-confidence proxy.
    S1s absent from ``cands`` get ``n_candidates=0`` and ``top_score=NaN``.
    """
    score_cols = [c for c in cands.columns if c == "prob" or c.startswith("score_")]
    row_score = (cands[score_cols].max(axis=1, skipna=True) if score_cols
                else pd.Series(np.nan, index=cands.index))
    grouped = cands.assign(_score=row_score).groupby("s1_id")["_score"]
    n_cand = cands.groupby("s1_id").size()
    top = grouped.max()
    idx = pd.Index(s1_ids)
    return pd.DataFrame({
        "entity_id": idx.to_numpy(),
        "n_candidates": n_cand.reindex(idx, fill_value=0).to_numpy(),
        "top_score": top.reindex(idx).to_numpy(),
    })


def _quantile_summary(x: pd.Series) -> dict[str, float]:
    """count/mean/median/P5/P25/P50/P75/P95 of a numeric series (NaNs dropped)."""
    x = x.dropna()
    out: dict[str, float] = {"count": float(len(x)),
                             "mean": float(x.mean()) if len(x) else float("nan")}
    qs = (x.quantile(list(_QUANTILES)) if len(x)
         else pd.Series([float("nan")] * len(_QUANTILES), index=list(_QUANTILES)))
    for q in _QUANTILES:
        out[f"p{int(q * 100)}"] = float(qs.loc[q])
    out["median"] = out["p50"]
    return out


def cohens_d(a: pd.Series, b: pd.Series) -> float:
    """Standardised mean difference (pooled std) between two samples.

    The practical-significance measure used alongside (never instead of) the
    KS test. |d| is conventionally read as ~0.2 small, ~0.5 medium, ~0.8 large.
    Returns NaN if either sample has fewer than 2 non-null values or the
    pooled standard deviation is zero.
    """
    a = a.dropna().to_numpy(np.float64)
    b = b.dropna().to_numpy(np.float64)
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    na, nb = len(a), len(b)
    pooled_var = ((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2)
    pooled_std = np.sqrt(pooled_var)
    if not pooled_std > 0:
        return float("nan")
    return float((a.mean() - b.mean()) / pooled_std)


def _effect_bucket(d: float) -> str:
    """Cohen's-d bucket label, or ``"n/a"`` for NaN."""
    if np.isnan(d):
        return "n/a"
    ad = abs(d)
    if ad < 0.2:
        return "negligible"
    if ad < 0.5:
        return "small"
    if ad < 0.8:
        return "medium"
    return "large"


def compare_distributions(
    train: pd.DataFrame, test: pd.DataFrame, columns: Sequence[str] | None = None
) -> pd.DataFrame:
    """Train-vs-test distribution report for every numeric column in ``columns``.

    Returns one row per column with train/test count/mean/median/P5/P25/P75/
    P95, a KS-test statistic and p-value (STATISTICAL DIFFERENCE), and
    Cohen's d with a bucket label (PRACTICAL / POTENTIALLY RELEVANT
    DISTRIBUTION SHIFT). The two are reported side by side and never
    collapsed into a single verdict.
    """
    if columns is None:
        columns = [c for c in train.columns if c not in ("entity_id", "country")
                  and pd.api.types.is_numeric_dtype(train[c])]
    rows = []
    for col in columns:
        tr, te = train[col], test[col]
        row: dict[str, float | str] = {"covariate": col}
        for split, s in (("train", tr), ("test", te)):
            for k, v in _quantile_summary(s).items():
                row[f"{split}_{k}"] = v
        tr_d, te_d = tr.dropna(), te.dropna()
        if len(tr_d) >= 2 and len(te_d) >= 2:
            ks_stat, ks_p = stats.ks_2samp(tr_d, te_d)
        else:
            ks_stat, ks_p = float("nan"), float("nan")
        d = cohens_d(tr, te)
        row["ks_stat"] = float(ks_stat)
        row["ks_pvalue"] = float(ks_p)
        row["cohens_d"] = d
        row["effect_size_bucket"] = _effect_bucket(d)
        rows.append(row)
    return pd.DataFrame(rows)


def compare_country_share(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Country distribution (share of records) in train vs. test, side by side.

    Expects a ``country`` column (e.g. the output of :func:`record_covariates`,
    or a raw source frame's own ``config.COUNTRY_COL`` renamed to ``country``).
    """
    tr = train["country"].value_counts(normalize=True).rename("train_share")
    te = test["country"].value_counts(normalize=True).rename("test_share")
    out = pd.concat([tr, te], axis=1).fillna(0.0).reset_index(names="country")
    return out.sort_values("test_share", ascending=False).reset_index(drop=True)
