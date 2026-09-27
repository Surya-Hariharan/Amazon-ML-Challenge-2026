"""Pairwise features for (S1, candidate) pairs (CLAUDE.md §6.3).

Design inputs (CP1 audit):

* 38% of S1 records share their name with another S1 (one name is used 253 times),
  so name similarity alone cannot separate them. The address block below is as rich
  as the name block, with explicit *conflict* features (house number, postal code,
  digits, region) that let the model reject same-name/different-place pairs.
* Some true matches differ entirely on name (native script, website form) and are
  decided by address alone.
* S2/S3 addresses are often empty. Missing information is encoded as NaN, never as
  a 0 similarity, so the model can tell "unknown" from "different".
* ~25% of S2/S3 records are orphans. Context features (rank within the S1's list,
  reverse rank among S1s competing for the same candidate, gap to best) are the main
  weapon against them.

No country one-hots and no raw ID features (CLAUDE.md §2.4); ``same_country`` is the
only country-derived feature. Computation is chunked (``config.FEATURE_CHUNK`` pairs)
and each chunk is written directly into a preallocated output buffer sized for the
full pair count, so memory stays bounded by the final matrix size (~1x) rather than
growing with the number of chunks retained in memory.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler
from sklearn.feature_extraction.text import TfidfVectorizer

from . import config
from .blocking import BASE_PASSES, PASS_BITS
from .normalize import SUFFIX_FAMILY

KEY_COLS = ["s1_id", "cand_id"]


def _log(msg: str) -> None:
    """Print a timestamped progress line."""
    print(f"[features {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _rss_mb() -> float:
    """Current process resident memory in MiB, or NaN if psutil is unavailable."""
    try:
        import psutil
        return psutil.Process().memory_info().rss / (1024 ** 2)
    except Exception:
        return float("nan")


# --- fitted context ------------------------------------------------------------------

class FeatureContext:
    """TF-IDF vectorizers for name/address cosine, fitted on one split's records.

    Fitted per split (train for training, test for inference) on a seeded sample of
    that split's own S1+S2+S3 records, so IDF reflects the records being compared,
    including French n-grams that never occur in train.
    """

    def __init__(self, name_vec: TfidfVectorizer, addr_vec: TfidfVectorizer):
        """Store fitted vectorizers."""
        self.name_vec = name_vec
        self.addr_vec = addr_vec

    @classmethod
    def fit(cls, s1: pd.DataFrame, others: pd.DataFrame,
            sample: int = config.FEATURE_TFIDF_FIT_SAMPLE, seed: int = config.SEED
            ) -> "FeatureContext":
        """Fit char 3-4-gram TF-IDF on name_core and addr_norm of a record sample."""
        both = pd.concat([s1[["name_core", "addr_norm"]], others[["name_core", "addr_norm"]]],
                         ignore_index=True)
        if len(both) > sample:
            both = both.sample(sample, random_state=seed)

        def fit_one(texts: pd.Series) -> TfidfVectorizer:
            """Fit one vectorizer (falls back to no min_df on tiny inputs)."""
            texts = texts[texts.str.len() > 0]
            vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 4), dtype=np.float32,
                                  sublinear_tf=True, min_df=2 if len(texts) > 1000 else 1)
            vec.fit(texts if len(texts) else pd.Series(["empty"]))
            return vec

        return cls(fit_one(both["name_core"]), fit_one(both["addr_norm"]))


# --- vectorised pair helpers ---------------------------------------------------------

def _cpdist(a: list[str], b: list[str], scorer) -> np.ndarray:
    """Element-wise rapidfuzz similarity of two aligned string lists, in [0, 1]."""
    if not a:
        return np.zeros(0, dtype=np.float32)
    out = process.cpdist(a, b, scorer=scorer, workers=-1, dtype=np.float32)
    return (out / 100.0 if scorer is not JaroWinkler.normalized_similarity else out
            ).astype(np.float32)


def _row_cosine(vec: TfidfVectorizer, a: pd.Series, b: pd.Series) -> np.ndarray:
    """Row-wise TF-IDF cosine of aligned texts; NaN where either side is empty."""
    ma, mb = vec.transform(a), vec.transform(b)
    cos = np.asarray(ma.multiply(mb).sum(axis=1)).ravel().astype(np.float32)
    empty = (a.str.len().to_numpy() == 0) | (b.str.len().to_numpy() == 0)
    cos[empty] = np.nan
    return cos


def _token_sets(values: pd.Series) -> np.ndarray:
    """Per-record frozenset of space-separated tokens (object array)."""
    return np.array([frozenset(v.split()) for v in values], dtype=object)


def _jaccard(a: np.ndarray, b: np.ndarray, nan_if_empty: bool = True) -> np.ndarray:
    """Element-wise Jaccard of two aligned arrays of sets; NaN if either set is empty."""
    out = np.empty(len(a), dtype=np.float32)
    for i, (x, y) in enumerate(zip(a, b)):
        if not x or not y:
            out[i] = np.nan if nan_if_empty else 0.0
        else:
            out[i] = len(x & y) / len(x | y)
    return out


def _overlap_count(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Element-wise size of the intersection of two aligned set arrays."""
    return np.fromiter((len(x & y) for x, y in zip(a, b)), dtype=np.float32, count=len(a))


def _eq_or_nan(a: pd.Series, b: pd.Series) -> np.ndarray:
    """1.0 if equal, 0.0 if different, NaN if either side is empty."""
    a, b = a.to_numpy(), b.to_numpy()
    out = (a == b).astype(np.float32)
    out[(a == "") | (b == "")] = np.nan
    return out


def _suffix_features(a: pd.Series, b: pd.Series) -> dict[str, np.ndarray]:
    """Legal-suffix agreement: both present, equal, family conflict, one missing."""
    fa = [frozenset(SUFFIX_FAMILY[t] for t in s.split() if t in SUFFIX_FAMILY) for s in a]
    fb = [frozenset(SUFFIX_FAMILY[t] for t in s.split() if t in SUFFIX_FAMILY) for s in b]
    both = np.array([bool(x) and bool(y) for x, y in zip(fa, fb)])
    conflict = np.array([bool(x) and bool(y) and not (x & y) for x, y in zip(fa, fb)])
    one = np.array([bool(x) != bool(y) for x, y in zip(fa, fb)])
    return {
        "suffix_equal": (a.to_numpy() == b.to_numpy()).astype(np.float32),
        "suffix_both": both.astype(np.float32),
        "suffix_conflict": conflict.astype(np.float32),
        "suffix_one_missing": one.astype(np.float32),
    }


# --- pairwise features ---------------------------------------------------------------

def embedding_cosine(e1: np.ndarray, e2: np.ndarray, i1: np.ndarray, i2: np.ndarray,
                     block: int = 65_536) -> np.ndarray:
    """``emb_cos`` for pairs ``(i1[j], i2[j])`` of the full embedding matrices.

    Gathers and multiplies ``block`` rows at a time, so the transient is
    ``3 * block * dim`` floats instead of three full chunk-sized matrices. Each row's
    value is computed exactly as ``(a * b).sum(axis=1)`` on the gathered rows, so the
    result is bit-identical to gathering the whole chunk at once.
    """
    out = np.empty(len(i1), dtype=np.float32)
    for s in range(0, len(i1), block):
        a = np.asarray(e1[i1[s:s + block]], dtype=np.float32)
        b = np.asarray(e2[i2[s:s + block]], dtype=np.float32)
        out[s:s + block] = (a * b).sum(axis=1)
    return out


def pair_features(r1: pd.DataFrame, r2: pd.DataFrame, ctx: FeatureContext,
                  emb: tuple[np.ndarray, np.ndarray] | None = None,
                  emb_cos: np.ndarray | None = None) -> pd.DataFrame:
    """Name + address features for aligned record frames ``r1`` (S1) and ``r2``.

    ``emb`` is ``(e1, e2)`` row-aligned with ``r1``/``r2`` (already gathered), or None.
    ``emb_cos``, when given, is the precomputed embedding cosine (see
    :func:`embedding_cosine`) and takes precedence over ``emb``.
    """
    f: dict[str, np.ndarray] = {}
    c1, c2 = r1["name_core"].tolist(), r2["name_core"].tolist()
    # --- name
    f["name_jw"] = _cpdist(c1, c2, JaroWinkler.normalized_similarity)
    f["name_ratio"] = _cpdist(c1, c2, fuzz.ratio)
    f["name_token_set"] = _cpdist(c1, c2, fuzz.token_set_ratio)
    f["name_token_sort"] = _cpdist(c1, c2, fuzz.token_sort_ratio)
    f["name_partial"] = _cpdist(c1, c2, fuzz.partial_ratio)
    f["name_full_token_set"] = _cpdist(r1["name_norm"].tolist(), r2["name_norm"].tolist(),
                                       fuzz.token_set_ratio)
    k1, k2 = r1["name_compact"].tolist(), r2["name_compact"].tolist()
    f["name_compact_ratio"] = _cpdist(k1, k2, fuzz.ratio)
    f["name_compact_partial"] = _cpdist(k1, k2, fuzz.partial_ratio)
    f["name_tfidf_cos"] = _row_cosine(ctx.name_vec, r1["name_core"], r2["name_core"])
    f["name_core_exact"] = (r1["name_core"].to_numpy() == r2["name_core"].to_numpy()
                            ).astype(np.float32)
    f["name_compact_exact"] = (np.array(k1, dtype=object) == np.array(k2, dtype=object)
                               ).astype(np.float32)
    f.update(_suffix_features(r1["name_suffix"], r2["name_suffix"]))
    a1, a2 = r1["acronym"].to_numpy(), r2["acronym"].to_numpy()
    k1a, k2a = np.array(k1, dtype=object), np.array(k2, dtype=object)
    f["acronym_match"] = (((a1 != "") & (a1 == k2a)) | ((a2 != "") & (a2 == k1a))
                          ).astype(np.float32)
    t1, t2 = _token_sets(r1["name_core"]), _token_sets(r2["name_core"])
    f["name_token_jaccard"] = _jaccard(t1, t2, nan_if_empty=False)
    l1 = r1["name_core"].str.len().to_numpy(np.float32)
    l2 = r2["name_core"].str.len().to_numpy(np.float32)
    f["name_len_ratio"] = np.minimum(l1, l2) / np.maximum(np.maximum(l1, l2), 1)
    f["first_token_equal"] = _eq_or_nan(r1["first_token"], r2["first_token"])
    f["cand_is_website"] = r2["is_website"].to_numpy(np.float32)
    f["cand_has_dba"] = (r2["name_alias"].to_numpy() != "").astype(np.float32)
    # --- address
    e1, e2 = r1["addr_empty"].to_numpy(bool), r2["addr_empty"].to_numpy(bool)
    f["addr_empty_s1"] = e1.astype(np.float32)
    f["addr_empty_cand"] = e2.astype(np.float32)
    any_empty = e1 | e2
    ad1, ad2 = r1["addr_norm"].tolist(), r2["addr_norm"].tolist()
    for name, scorer in (("addr_ratio", fuzz.ratio), ("addr_token_set", fuzz.token_set_ratio),
                         ("addr_token_sort", fuzz.token_sort_ratio),
                         ("addr_partial", fuzz.partial_ratio)):
        v = _cpdist(ad1, ad2, scorer)
        v[any_empty] = np.nan
        f[name] = v
    f["addr_tfidf_cos"] = _row_cosine(ctx.addr_vec, r1["addr_norm"], r2["addr_norm"])
    f["addr_token_jaccard"] = _jaccard(_token_sets(r1["addr_norm"]), _token_sets(r2["addr_norm"]))
    d1, d2 = _token_sets(r1["digits"]), _token_sets(r2["digits"])
    f["digit_jaccard"] = _jaccard(d1, d2)
    f["digit_shared"] = _overlap_count(d1, d2)
    has_d = np.array([bool(x) and bool(y) for x, y in zip(d1, d2)])
    f["digit_conflict"] = np.where(has_d, (f["digit_shared"] == 0).astype(np.float32), np.nan)
    f["digit_cand_subset"] = np.where(
        has_d, np.fromiter((y <= x for x, y in zip(d1, d2)), dtype=np.float32, count=len(d1)),
        np.nan)
    f["house_equal"] = _eq_or_nan(r1["house_no"], r2["house_no"])
    ln1, ln2 = _token_sets(r1["long_nums"]), _token_sets(r2["long_nums"])
    f["long_num_equal"] = np.where(
        np.array([bool(x) and bool(y) for x, y in zip(ln1, ln2)]),
        (_overlap_count(ln1, ln2) > 0).astype(np.float32), np.nan)
    f["landmark_jaccard"] = _jaccard(_token_sets(r1["landmark"]), _token_sets(r2["landmark"]))
    f["region_equal"] = _eq_or_nan(r1["region"], r2["region"])
    f["same_country"] = (r1[config.COUNTRY_COL].to_numpy() == r2[config.COUNTRY_COL].to_numpy()
                         ).astype(np.float32)
    # --- embedding
    if emb_cos is not None:
        f["emb_cos"] = np.asarray(emb_cos, dtype=np.float32)
    elif emb is not None:
        f["emb_cos"] = (np.asarray(emb[0], dtype=np.float32)
                        * np.asarray(emb[1], dtype=np.float32)).sum(axis=1)
    else:
        f["emb_cos"] = np.full(len(r1), np.nan, dtype=np.float32)
    return pd.DataFrame({k: np.asarray(v, dtype=np.float32) for k, v in f.items()})


# --- context -------------------------------------------------------------------------

def context_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add rank / gap / reverse-rank / count features to a full pair frame (in place).

    Uses ``pair_sim`` = mean of name token-set and address token-set similarity
    (name only when an address is missing) as the ranking score:

    * ``rank_in_s1`` / ``gap_to_best_s1`` / ``n_cands_s1``: this candidate vs. the S1's
      other candidates.
    * ``rank_in_cand`` / ``gap_to_best_cand`` / ``n_s1_for_cand``: reverse view. Because
      each S2/S3 record belongs to at most one S1 (CP1 audit), a candidate that is
      clearly better matched to another S1 is probably not this S1's match.
    * ``is_s3``: candidate source.
    """
    addr = df["addr_token_set"].to_numpy()
    name = df["name_token_set"].to_numpy()
    sim = np.where(np.isnan(addr), name, 0.5 * name + 0.5 * addr).astype(np.float32)
    df["pair_sim"] = sim
    g1 = df.groupby("s1_id", sort=False)["pair_sim"]
    df["rank_in_s1"] = g1.rank(ascending=False, method="min").astype(np.float32)
    df["gap_to_best_s1"] = (g1.transform("max") - df["pair_sim"]).astype(np.float32)
    df["n_cands_s1"] = g1.transform("size").astype(np.float32)
    g2 = df.groupby("cand_id", sort=False)["pair_sim"]
    df["rank_in_cand"] = g2.rank(ascending=False, method="min").astype(np.float32)
    df["gap_to_best_cand"] = (g2.transform("max") - df["pair_sim"]).astype(np.float32)
    df["n_s1_for_cand"] = g2.transform("size").astype(np.float32)
    df["is_s3"] = df["cand_id"].str.startswith("S3-").astype(np.float32)
    return df


def blocking_features(cands: pd.DataFrame) -> pd.DataFrame:
    """Pass-membership flags, pass count and per-pass blocking scores."""
    out = {}
    passes = cands["passes"].to_numpy()
    # Base passes always; optional passes only when their score column is present
    # (i.e. the pass was enabled when these candidates were generated).
    names = [p for p in PASS_BITS if p in BASE_PASSES or f"score_{p}" in cands]
    for name in names:
        bit = PASS_BITS[name]
        out[f"in_{name}"] = ((passes & bit) > 0).astype(np.float32)
        col = f"score_{name}"
        out[f"block_{name}"] = (cands[col].to_numpy(np.float32) if col in cands
                                else np.full(len(cands), np.nan, np.float32))
    out["n_passes"] = sum(out[f"in_{n}"] for n in names)
    return pd.DataFrame(out, index=cands.index)


# --- driver --------------------------------------------------------------------------

def build_features(
    cands: pd.DataFrame,
    s1: pd.DataFrame,
    others: pd.DataFrame,
    embeddings: tuple[np.ndarray, np.ndarray] | None = None,
    ctx: FeatureContext | None = None,
    chunk: int | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Return one float32 feature row per candidate pair, keyed by ``s1_id``/``cand_id``.

    Args:
        cands: blocking output (``s1_id``, ``cand_id``, ``passes``, ``score_*``).
        s1 / others: normalised frames (``normalize.normalize_frame``).
        embeddings: ``(s1_emb, others_emb)`` row-aligned with the frames, or None
            (``emb_cos`` becomes NaN).
        ctx: fitted :class:`FeatureContext`; fitted on ``s1``+``others`` if None.
        chunk: pairs per chunk; None reads ``config.FEATURE_CHUNK`` at call time.
        verbose: print progress.
    """
    chunk = config.FEATURE_CHUNK if chunk is None else chunk
    s1 = s1.reset_index(drop=True)
    others = others.reset_index(drop=True)
    cands = cands.reset_index(drop=True)
    if ctx is None:
        ctx = FeatureContext.fit(s1, others)
    i1 = pd.Index(s1[config.ID_COL]).get_indexer(cands["s1_id"])
    i2 = pd.Index(others[config.ID_COL]).get_indexer(cands["cand_id"])
    if (i1 < 0).any() or (i2 < 0).any():
        raise ValueError("candidate IDs missing from the normalised frames")

    n = len(cands)
    if verbose:
        _log(f"before build_features: rss={_rss_mb():.0f}MiB")
    # Every chunk is written into ONE preallocated float32 block of shape
    # (n_features, n_pairs). Wrapping its transpose in a DataFrame reuses the memory
    # as pandas' single float block, so assembling the output frame no longer copies
    # every feature column (measured +214 B/pair transient with per-column buffers).
    block: np.ndarray | None = None
    names: list[str] = []
    t0 = time.time()
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        a, b = i1[start:end], i2[start:end]
        r1 = s1.iloc[a].reset_index(drop=True)
        r2 = others.iloc[b].reset_index(drop=True)
        cos = (embedding_cosine(embeddings[0], embeddings[1], a, b)
               if embeddings is not None else None)
        part = pair_features(r1, r2, ctx, emb_cos=cos)
        del r1, r2, cos
        if block is None:
            names = list(part.columns)
            block = np.empty((len(names), n), dtype=np.float32)
        for j, col in enumerate(names):
            block[j, start:end] = part[col].to_numpy()
        del part
        if verbose:
            _log(f"{end:,}/{n:,} pairs ({time.time() - t0:.0f}s) rss={_rss_mb():.0f}MiB")
    if block is None:
        names = list(pair_features(s1.iloc[:0], others.iloc[:0], ctx, None).columns)
        block = np.empty((len(names), 0), dtype=np.float32)

    if verbose:
        _log(f"before assembling output frame: rss={_rss_mb():.0f}MiB")
    out = pd.DataFrame(block.T, columns=names, copy=False)
    del block  # the frame now owns the memory
    out.insert(0, KEY_COLS[1], cands["cand_id"].to_numpy())
    out.insert(0, KEY_COLS[0], cands["s1_id"].to_numpy())
    bf = blocking_features(cands)
    for col in bf.columns:
        out[col] = bf[col].to_numpy()
    del bf
    if verbose:
        _log(f"after assembling output frame: rss={_rss_mb():.0f}MiB")
    out = context_features(out)
    if verbose:
        _log(f"after context_features: rss={_rss_mb():.0f}MiB")
    return out


def label_pairs(pairs: pd.DataFrame, truth: dict[str, list[str]]) -> np.ndarray:
    """1 where ``cand_id`` is a true match of ``s1_id``, else 0."""
    true = pd.DataFrame([(s, m) for s, ms in truth.items() for m in ms], columns=KEY_COLS)
    true["label"] = 1
    lab = pairs[KEY_COLS].merge(true.drop_duplicates(KEY_COLS), how="left", on=KEY_COLS)["label"]
    return lab.fillna(0).astype(np.int8).to_numpy()


def feature_columns(df: pd.DataFrame) -> list[str]:
    """Model input columns: every numeric column except keys and the label."""
    return [c for c in df.columns if c not in (*KEY_COLS, "label")]
