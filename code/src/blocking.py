

from __future__ import annotations

import hashlib
import math
import time
from collections.abc import Callable, Iterator, Mapping

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer

from . import config
from .evaluate import reduction_ratio

#: Bit per pass in the ``passes`` column of the candidate frame.
PASS_BITS = {"tfidf": 1, "embed": 2, "rare": 4, "digit": 8, "address": 16}

#: Street words too common to be address block keys.
_ADDRESS_STOPWORDS = frozenset({
    "road", "street", "avenue", "lane", "drive", "court", "place", "boulevard", "rue",
    "floor", "flat", "house", "building", "near", "opposite", "nagar", "colony", "main",
    "cross", "sector", "phase", "block", "plot", "shop", "unit", "suite", "apartment",
    "north", "south", "east", "west", "the", "and", "des", "del", "les", "chemin",
    "allee", "impasse", "route", "city", "district", "village", "post", "office",
    "ground", "first", "second", "third", "way", "circle", "highway", "tehsil", "taluk",
})

Encoder = Callable[[list[str]], np.ndarray]


def _log(msg: str) -> None:
    """Print a timestamped progress line."""
    print(f"[blocking {time.strftime('%H:%M:%S')}] {msg}", flush=True)


def country_groups(
    s1: pd.DataFrame, others: pd.DataFrame, within_country: bool = config.BLOCK_WITHIN_COUNTRY
) -> Iterator[tuple[str, np.ndarray, np.ndarray]]:
    """Yield ``(country, s1_positions, other_positions)`` blocking groups.

    With ``within_country`` off, a single group ``"*"`` holds everything. Countries
    come from the S1 frame (open set). S1 rows whose country has no S2/S3 records
    yield an empty ``other_positions`` array and simply get no candidates.
    """
    if not within_country:
        yield "*", np.arange(len(s1)), np.arange(len(others))
        return
    other_groups = others.groupby(config.COUNTRY_COL, sort=True).indices
    for country, q_idx in s1.groupby(config.COUNTRY_COL, sort=True).indices.items():
        yield country, q_idx, other_groups.get(country, np.array([], dtype=np.int64))


# --- top-k primitives ----------------------------------------------------------------

def _csr_row_topk(mat: sparse.csr_matrix, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return ``(row, col, value)`` of the k largest stored values in each CSR row."""
    mat.eliminate_zeros()
    counts = np.diff(mat.indptr)
    rows = np.repeat(np.arange(mat.shape[0]), counts)
    if rows.size == 0:
        e = np.array([], dtype=np.int64)
        return e, e, np.array([], dtype=np.float32)
    order = np.lexsort((-mat.data, rows))
    rank = np.arange(rows.size) - mat.indptr[rows[order]]
    keep = order[rank < k]
    return rows[keep], mat.indices[keep], mat.data[keep].astype(np.float32)


def prune_rows(mat: sparse.csr_matrix, n_terms: int) -> sparse.csr_matrix:
    """Keep only the ``n_terms`` largest entries of each CSR row (query pruning).

    For TF-IDF rows the largest weights are the rarest n-grams, which carry the
    identity and have the shortest posting lists.
    """
    mat = mat.tocsr()
    r, c, v = _csr_row_topk(mat.copy(), n_terms)
    return sparse.csr_matrix((v, (r, c)), shape=mat.shape, dtype=mat.dtype)


def sparse_topk(
    queries: sparse.csr_matrix,
    index: sparse.csr_matrix,
    k: int,
    max_product_nnz: int = config.TFIDF_MAX_PRODUCT_NNZ,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Top-k by sparse dot product, in cost-adaptive query chunks.

    A query's cost is the total posting-list length of its terms, which bounds how
    many entries it adds to the product. Chunks are cut so that one chunk's product
    stays below ``max_product_nnz`` (a single very expensive query still forms its own
    chunk). Returns ``(query_row, index_row, score)``.
    """
    index_t = index.T.tocsr()
    postings = np.diff(index_t.indptr).astype(np.int64)  # records per term
    q_bin = queries.copy()
    q_bin.data = np.ones_like(q_bin.data)
    cost = np.asarray(q_bin @ postings).ravel()
    qs, xs, vs = [], [], []
    start, n = 0, queries.shape[0]
    while start < n:
        cum = np.cumsum(cost[start:])
        end = start + max(1, int(np.searchsorted(cum, max_product_nnz, side="right")))
        prod = (queries[start:end] @ index_t).tocsr()
        r, c, v = _csr_row_topk(prod, k)
        qs.append(r + start)
        xs.append(c)
        vs.append(v)
        start = end
    if not qs:
        e = np.array([], dtype=np.int64)
        return e, e, np.array([], dtype=np.float32)
    return np.concatenate(qs), np.concatenate(xs), np.concatenate(vs)


def _torch_device():
    """Return a CUDA torch device if available, else None (numpy path)."""
    try:
        import torch
    except ImportError:
        return None
    return torch.device("cuda") if torch.cuda.is_available() else None


def dense_topk(
    queries: np.ndarray,
    index: np.ndarray,
    k: int,
    query_chunk: int = config.DENSE_QUERY_CHUNK,
    index_chunk: int = config.DENSE_INDEX_CHUNK,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Exact inner-product top-k, streaming index blocks and keeping a running top-k.

    Each index block is loaded once (to the GPU when torch+CUDA are available) and
    scored against every query chunk. Peak memory is about
    ``query_chunk * index_chunk`` scores plus ``n_queries * k`` running results, so it
    scales to millions of rows per side. Rows should be L2-normalised (cosine).
    Returns flat ``(query_row, index_row, score)`` arrays.
    """
    nq, nx = len(queries), len(index)
    k = min(k, nx)
    if nq == 0 or k == 0:
        e = np.array([], dtype=np.int64)
        return e, e, np.array([], dtype=np.float32)
    best_s = np.full((nq, k), -np.inf, dtype=np.float32)
    best_i = np.full((nq, k), -1, dtype=np.int64)
    device = _torch_device()
    if device is not None:
        import torch
    for xs in range(0, nx, index_chunk):
        xblock = np.asarray(index[xs:xs + index_chunk], dtype=np.float32)
        kb = min(k, len(xblock))
        if device is not None:
            xt = torch.from_numpy(xblock).to(device, dtype=torch.float16)
        for qs in range(0, nq, query_chunk):
            qblock = np.asarray(queries[qs:qs + query_chunk], dtype=np.float32)
            if device is not None:
                qt = torch.from_numpy(qblock).to(device, dtype=torch.float16)
                sv, si = torch.topk((qt @ xt.T).float(), kb, dim=1)
                sv, si = sv.cpu().numpy(), si.cpu().numpy().astype(np.int64)
            else:
                sims = qblock @ xblock.T
                si = np.argpartition(-sims, kb - 1, axis=1)[:, :kb]
                sv = np.take_along_axis(sims, si, axis=1)
            si = si + xs
            cat_s = np.concatenate([best_s[qs:qs + query_chunk], sv], axis=1)
            cat_i = np.concatenate([best_i[qs:qs + query_chunk], si], axis=1)
            top = np.argpartition(-cat_s, k - 1, axis=1)[:, :k]
            best_s[qs:qs + query_chunk] = np.take_along_axis(cat_s, top, axis=1)
            best_i[qs:qs + query_chunk] = np.take_along_axis(cat_i, top, axis=1)
    rows = np.repeat(np.arange(nq), k)
    flat_i, flat_s = best_i.ravel(), best_s.ravel()
    ok = flat_i >= 0
    return rows[ok], flat_i[ok], flat_s[ok]


# --- embeddings ----------------------------------------------------------------------

def embedding_text(frame: pd.DataFrame) -> list[str]:
    """Text fed to the embedding model: lightly cleaned raw ``name | address``.

    Raw (not transliterated) so the multilingual model sees native scripts itself.
    """
    name = frame[config.NAME_COL].fillna("").str.replace(r"\s+", " ", regex=True)
    addr = frame[config.ADDRESS_COL].fillna("").str.replace(r"\s+", " ", regex=True)
    name = name.str.strip().str.lstrip("#->< ").str.lower()
    addr = addr.str.strip().str.lstrip("#->< ").str.lower()
    return (name + " | " + addr).tolist()


def load_encoder(model_name: str = config.EMBEDDING_MODEL,
                 revision: str = config.EMBEDDING_REVISION) -> Encoder:
    """Load the pinned sentence-transformers model (licence logged in MODELS.md).

    Returns a callable mapping a list of strings to L2-normalised float32 vectors.
    """
    from sentence_transformers import SentenceTransformer

    device = "cuda" if _torch_device() is not None else "cpu"
    if device == "cpu":
        _log("WARNING: no CUDA device; embedding millions of records on CPU is very slow")
    model = SentenceTransformer(model_name, revision=revision, device=device)

    def encode(texts: list[str]) -> np.ndarray:
        """Encode texts with the loaded model."""
        return model.encode(texts, batch_size=config.EMBEDDING_BATCH,
                            normalize_embeddings=True, convert_to_numpy=True,
                            show_progress_bar=len(texts) > 100_000).astype(np.float32)

    return encode


def _texts_key(texts: list[str], model_name: str) -> str:
    """Stable cache key for a list of texts and a model."""
    h = hashlib.sha1(model_name.encode())
    h.update(config.EMBEDDING_REVISION.encode())
    h.update(pd.util.hash_pandas_object(pd.Series(texts), index=False).values.tobytes())
    return h.hexdigest()[:20]


def compute_embeddings(frame: pd.DataFrame, encoder: Encoder | None = None,
                       cache: bool = True) -> np.ndarray:
    """Return L2-normalised float16 embeddings for every row of ``frame``.

    Cached in ``artifacts/emb_<hash>.npy`` keyed by input texts + model (CLAUDE.md §7),
    so blocking and features share one encoding pass per split.
    """
    texts = embedding_text(frame)
    path = config.ARTIFACTS_DIR / f"emb_{_texts_key(texts, config.EMBEDDING_MODEL)}.npy"
    if cache and path.exists():
        return np.load(path, mmap_mode="r")
    if encoder is None:
        encoder = load_encoder()
    uniq, inverse = np.unique(np.asarray(texts, dtype=object), return_inverse=True)
    vecs = encoder(list(uniq)).astype(np.float16)[inverse]
    if cache:
        config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        np.save(path, vecs)
    return vecs


# --- passes --------------------------------------------------------------------------

def _empty_pass() -> pd.DataFrame:
    """An empty pass result frame."""
    return pd.DataFrame({"q": pd.Series([], dtype=np.int64),
                         "x": pd.Series([], dtype=np.int64),
                         "score": pd.Series([], dtype=np.float32)})


def tfidf_name_pass(q_names: pd.Series, x_names: pd.Series, k: int = config.K_TFIDF_NAME
                    ) -> pd.DataFrame:
    """Pass 1: char 3-4-gram TF-IDF on ``name_core``, cosine top-k per query.

    The vectorizer is fitted on this group's queries + index. Returns positions
    ``q`` (into ``q_names``), ``x`` (into ``x_names``) and ``score``.
    """
    if len(q_names) == 0 or len(x_names) == 0:
        return _empty_pass()
    n_docs = len(q_names) + len(x_names)
    big = n_docs >= config.TFIDF_MAX_DF_MIN_DOCS
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=config.TFIDF_NGRAM,
                          min_df=config.TFIDF_MIN_DF if big else 1,
                          max_df=config.TFIDF_MAX_DF if big else 1.0,
                          dtype=np.float32, sublinear_tf=True)
    try:
        vec.fit(pd.concat([q_names, x_names], ignore_index=True))
    except ValueError:  # empty vocabulary
        return _empty_pass()
    queries = prune_rows(vec.transform(q_names), config.TFIDF_QUERY_TERMS)
    q, x, s = sparse_topk(queries, vec.transform(x_names), k)
    return pd.DataFrame({"q": q, "x": x, "score": s})


def embedding_pass(q_emb: np.ndarray, x_emb: np.ndarray, k: int = config.K_EMBEDDING
                   ) -> pd.DataFrame:
    """Pass 2: dense embedding cosine top-k per query."""
    q, x, s = dense_topk(q_emb, x_emb, k)
    return pd.DataFrame({"q": q, "x": x, "score": s})


def key_pass(q_keys: pd.Series, x_keys: pd.Series, k: int, max_block: int,
             idf_weight: bool = True, chunk: int = config.KEY_PASS_CHUNK) -> pd.DataFrame:
    """Generic inverted-index pass: pair records sharing a block key.

    Args:
        q_keys / x_keys: per-record lists of hashable keys (position-aligned).
        k: keep the k best index records per query.
        max_block: keys held by more than this many index records are skipped. This
            bounds the join size and drops uninformative keys.
        idf_weight: score = sum of log-IDF of shared keys (else count of shared keys).
        chunk: queries per join chunk (bounds memory).
    """
    xe = x_keys.explode().dropna()
    qe = q_keys.explode().dropna()
    if xe.empty or qe.empty:
        return _empty_pass()
    x_df = pd.DataFrame({"x": xe.index.to_numpy(np.int64), "key": xe.to_numpy()})
    q_df = pd.DataFrame({"q": qe.index.to_numpy(np.int64), "key": qe.to_numpy()})
    x_df = x_df.drop_duplicates()
    q_df = q_df.drop_duplicates()
    block = x_df["key"].value_counts()
    block = block[block <= max_block]
    if block.empty:
        return _empty_pass()
    x_df = x_df[x_df["key"].isin(block.index)]
    q_df = q_df[q_df["key"].isin(block.index)]
    n_total = len(x_keys) + len(q_keys)
    weight = (np.log(n_total / block) if idf_weight else block * 0 + 1.0).astype(np.float32)
    x_df = x_df.assign(w=x_df["key"].map(weight).to_numpy(np.float32))
    out = []
    q_ids = q_df["q"].unique()
    for start in range(0, len(q_ids), chunk):
        part = q_df[q_df["q"].isin(q_ids[start:start + chunk])]
        joined = part.merge(x_df, on="key", how="inner")
        if joined.empty:
            continue
        agg = joined.groupby(["q", "x"], sort=False)["w"].sum().reset_index()
        agg = agg.sort_values(["q", "w", "x"], ascending=[True, False, True])
        out.append(agg.groupby("q", sort=False).head(k))
    if not out:
        return _empty_pass()
    res = pd.concat(out, ignore_index=True).rename(columns={"w": "score"})
    return res.astype({"q": np.int64, "x": np.int64, "score": np.float32})


def _name_tokens(frame: pd.DataFrame) -> pd.Series:
    """Distinct name_core tokens (length >= 2) per record."""
    return frame["name_core"].str.split().map(
        lambda ts: list(dict.fromkeys(t for t in ts if len(t) >= 2)))


def rare_token_pass(q: pd.DataFrame, x: pd.DataFrame, k: int = config.K_RARE_TOKEN
                    ) -> pd.DataFrame:
    """Pass 3: records sharing one of the query's rarest name tokens.

    Each query uses its ``RARE_TOKENS_PER_RECORD`` lowest-frequency tokens (frequency
    counted over this group's queries + index). The index side uses all its tokens.
    """
    q_tok, x_tok = _name_tokens(q).reset_index(drop=True), _name_tokens(x).reset_index(drop=True)
    df = pd.concat([q_tok, x_tok]).explode().value_counts()
    n = config.RARE_TOKENS_PER_RECORD
    q_rare = q_tok.map(lambda ts: sorted(ts, key=lambda t: (df.get(t, 0), t))[:n])
    return key_pass(q_rare, x_tok, k, config.RARE_MAX_BLOCK, idf_weight=True)


def _digit_name_keys(frame: pd.DataFrame) -> pd.Series:
    """Keys ``"<digit>|<first name token>"`` for pass 4."""
    return pd.Series([
        [f"{d}|{f}" for d in digits.split()] if f else []
        for digits, f in zip(frame["digits"], frame["first_token"])
    ])


def digit_token_pass(q: pd.DataFrame, x: pd.DataFrame, k: int = config.K_POSTAL_TOKEN
                     ) -> pd.DataFrame:
    """Pass 4: shared address digit token (house no. / postal code) + first name token."""
    return key_pass(_digit_name_keys(q), _digit_name_keys(x), k, config.DIGIT_MAX_BLOCK,
                    idf_weight=True)


def _address_keys(frame: pd.DataFrame, word_df: pd.Series | None = None) -> pd.Series:
    """Address-only keys: first digit pair, and house number + two rarest street words."""
    keys = []
    for addr, digits in zip(frame["addr_norm"], frame["digits"]):
        ds = digits.split()
        ks: list[str] = []
        if len(ds) >= 2:
            ks.append(f"{ds[0]}#{ds[1]}")
        if ds:
            words = [w for w in addr.split()
                     if len(w) >= 3 and w.isalpha() and w not in _ADDRESS_STOPWORDS]
            words = list(dict.fromkeys(words))
            if word_df is not None:
                words.sort(key=lambda w: (word_df.get(w, 0), w))
            ks.extend(f"{ds[0]}@{w}" for w in words[:2])
        keys.append(ks)
    return pd.Series(keys)


def address_pass(q: pd.DataFrame, x: pd.DataFrame, k: int = config.K_ADDRESS
                 ) -> pd.DataFrame:
    """Pass 5: address-only keys, for matches whose names differ entirely."""
    words = pd.concat([q["addr_norm"], x["addr_norm"]]).str.split().explode()
    word_df = words.value_counts()
    return key_pass(_address_keys(q, word_df), _address_keys(x, word_df), k,
                    config.ADDRESS_MAX_BLOCK, idf_weight=True)


# --- union ---------------------------------------------------------------------------

def generate_candidates(
    s1: pd.DataFrame,
    others: pd.DataFrame,
    embeddings: tuple[np.ndarray, np.ndarray] | None = None,
    within_country: bool = config.BLOCK_WITHIN_COUNTRY,
    use_address_pass: bool = config.USE_ADDRESS_PASS,
    verbose: bool = True,
    k_overrides: Mapping[str, int] | None = None,
) -> pd.DataFrame:
    """Run every pass per blocking group and return the de-duplicated union.

    Args:
        s1: normalised S1 frame (``normalize.normalize_frame``).
        others: normalised S2+S3 frame.
        embeddings: ``(s1_emb, others_emb)`` row-aligned with the frames, or None to
            skip the embedding pass.
        within_country: block within each country string.
        use_address_pass: include pass 5.
        verbose: print per-group progress.
        k_overrides: optional ``{pass_name: k}`` (keys from ``PASS_BITS``, i.e.
            ``"tfidf"``/``"embed"``/``"rare"``/``"digit"``/``"address"``) to run a pass
            at a k other than its ``config.K_*`` default. Missing keys keep the
            default k for that pass. ``None`` (the default) reproduces production
            behaviour exactly -- this parameter exists only for the blocking-K
            ablation diagnostic (``src.diagnose blocking-ablation``), never for
            production runs.

    Returns:
        One row per (S1, candidate) pair with ``s1_id``, ``cand_id``, ``passes``
        (bitmask of ``PASS_BITS``) and ``score_<pass>`` (NaN when that pass did not
        return the pair).
    """
    ko = k_overrides or {}
    s1 = s1.reset_index(drop=True)
    others = others.reset_index(drop=True)
    frames = []
    for country, q_idx, x_idx in country_groups(s1, others, within_country):
        if len(x_idx) == 0:
            continue
        t0 = time.time()
        q, x = s1.iloc[q_idx].reset_index(drop=True), others.iloc[x_idx].reset_index(drop=True)
        runs = {
            "tfidf": lambda: tfidf_name_pass(q["name_core"], x["name_core"],
                                             k=ko.get("tfidf", config.K_TFIDF_NAME)),
            "rare": lambda: rare_token_pass(q, x, k=ko.get("rare", config.K_RARE_TOKEN)),
            "digit": lambda: digit_token_pass(q, x, k=ko.get("digit", config.K_POSTAL_TOKEN)),
        }
        if embeddings is not None:
            runs["embed"] = lambda: embedding_pass(
                np.asarray(embeddings[0][q_idx]), np.asarray(embeddings[1][x_idx]),
                k=ko.get("embed", config.K_EMBEDDING))
        if use_address_pass:
            runs["address"] = lambda: address_pass(q, x, k=ko.get("address", config.K_ADDRESS))
        for name, run in runs.items():
            res = run()
            res = pd.DataFrame({
                "q": q_idx[res["q"].to_numpy()],
                "x": x_idx[res["x"].to_numpy()],
                "bit": np.full(len(res), PASS_BITS[name], dtype=np.uint8),
                f"score_{name}": res["score"].to_numpy(np.float32),
            })
            frames.append(res)
            if verbose:
                _log(f"{country}: pass {name} -> {len(res):,} pairs")
        if verbose:
            _log(f"{country}: {len(q_idx):,} S1 x {len(x_idx):,} others in "
                 f"{time.time() - t0:.1f}s")
    score_cols = [f"score_{p}" for p in PASS_BITS]
    cols = ["s1_id", "cand_id", "passes"] + score_cols
    if not frames:
        empty = {c: pd.Series(dtype=object) for c in cols[:2]}
        empty["passes"] = pd.Series(dtype=np.uint8)
        empty.update({c: pd.Series(dtype=np.float32) for c in score_cols})
        return pd.DataFrame(empty)
    allp = pd.concat(frames, ignore_index=True)
    for c in score_cols:
        if c not in allp:
            allp[c] = np.float32(np.nan)
    # Each pass yields a (q, x) pair at most once and groups have disjoint queries,
    # so summing the distinct pass bits equals their bitwise OR.
    agg = {"bit": "sum", **{c: "max" for c in score_cols}}
    out = allp.groupby(["q", "x"], sort=True).agg(agg).reset_index()
    out = out.rename(columns={"bit": "passes"}).astype({"passes": np.uint8})
    out.insert(0, "s1_id", s1[config.ID_COL].to_numpy()[out["q"].to_numpy()])
    out.insert(1, "cand_id", others[config.ID_COL].to_numpy()[out["x"].to_numpy()])
    out[score_cols] = out[score_cols].astype(np.float32)
    return out[cols].reset_index(drop=True)


def candidates_to_lists(cands: pd.DataFrame) -> dict[str, list[str]]:
    """Convert a candidate frame to ``{s1_id: [cand_id, ...]}``."""
    return {s: list(g) for s, g in cands.groupby("s1_id", sort=False)["cand_id"]}


def report_blocking_stats(
    cands: pd.DataFrame,
    truth: Mapping[str, list[str]],
    s1_ids: list[str],
    n_other: int,
    s1_country: Mapping[str, str] | None = None,
    verbose: bool = True,
) -> dict[str, float]:
    """Blocking quality for docs/planning/plan.md CP3: recall, candidates/S1, reduction ratio.

    Args:
        cands: output of :func:`generate_candidates`.
        truth: S1 ID -> true S2/S3 IDs (S1s missing from it are singletons).
        s1_ids: every S1 ID in scope (S1s with no candidates count in the means).
        n_other: number of S2+S3 records blocked against.
        s1_country: optional S1 ID -> country, for per-country pair recall.
        verbose: print the report.

    Returns:
        ``pair_recall``, ``s1_full_recall``, ``mean_candidates``, ``max_candidates``,
        ``reduction_ratio``, ``n_pairs``, per-pass ``recall_<p>`` / ``unique_<p>`` (true
        pairs found by that pass only) / ``pairs_<p>``, and ``recall_<country>``.
    """
    s1_set = pd.Index(pd.unique(pd.Series(list(s1_ids), dtype=object)))
    n_s1 = max(len(s1_set), 1)
    truth_df = pd.DataFrame(
        [(s, m) for s in s1_set for m in dict.fromkeys(truth.get(s, ()))],
        columns=["s1_id", "cand_id"], dtype=object)
    in_scope = cands[cands["s1_id"].isin(s1_set)]
    found = truth_df.merge(in_scope[["s1_id", "cand_id", "passes"]], how="left",
                           on=["s1_id", "cand_id"])
    hit = found["passes"].notna().to_numpy()
    passes = found["passes"].fillna(0).astype(np.uint8).to_numpy()
    n_true = max(len(found), 1)
    sizes = in_scope.groupby("s1_id").size()
    stats: dict[str, float] = {
        "pair_recall": float(hit.sum()) / n_true if len(found) else 1.0,
        "s1_full_recall": float(pd.Series(hit).groupby(found["s1_id"].to_numpy()).all().mean())
        if len(found) else 1.0,
        "mean_candidates": len(in_scope) / n_s1,
        "max_candidates": float(sizes.max()) if len(sizes) else 0.0,
        "reduction_ratio": reduction_ratio(len(in_scope), len(s1_set), n_other),
        "n_pairs": float(len(in_scope)),
    }
    all_passes = in_scope["passes"].to_numpy()
    for name, bit in PASS_BITS.items():
        stats[f"pairs_{name}"] = float(((all_passes & bit) > 0).sum())
        stats[f"recall_{name}"] = float(((passes & bit) > 0).sum()) / n_true
        stats[f"unique_{name}"] = float((passes == bit).sum()) / n_true
    if s1_country is not None and len(found):
        country = found["s1_id"].map(s1_country).fillna("").to_numpy()
        for c, rate in pd.Series(hit).groupby(country).mean().items():
            stats[f"recall_{c}"] = float(rate)
    if verbose:
        for kname, v in stats.items():
            _log(f"{kname:>22}: {v:,.4f}" if not math.isnan(v) else f"{kname}: nan")
    return stats
