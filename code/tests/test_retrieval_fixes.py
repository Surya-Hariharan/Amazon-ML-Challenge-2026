"""Tests for the retrieval + memory fixes: digit-run blocking digits, all-digit address
keys, the street-word and name+address TF-IDF passes, hashed key joins, copy-free
feature assembly, chunked embedding cosine / encoding, and code-versioned caches.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import blocking, config, features
from src import run_pipeline as rp
from src.normalize import normalize_frame
from tests.synthetic import make_dataset
from tests.test_blocking import fake_encoder


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    """Artifacts to tmp so cache tests never touch the real artifacts dir."""
    monkeypatch.setattr(config, "ARTIFACTS_DIR", tmp_path / "artifacts")
    return tmp_path


def _addr_frame(addrs: list[str], digits: list[str] | None = None) -> pd.DataFrame:
    """Minimal frame with the fields the address/digit key builders read."""
    return pd.DataFrame({"addr_norm": addrs,
                         "digits": digits if digits is not None else [""] * len(addrs),
                         "first_token": ["acme"] * len(addrs)})


@pytest.fixture(scope="module")
def synth():
    """Normalised synthetic S1/others with truth (includes an unseen country)."""
    s1, s2, s3, truth = make_dataset(n_s1=150, seed=3, countries=("US", "India", "France"))
    return normalize_frame(s1), normalize_frame(pd.concat([s2, s3], ignore_index=True)), truth


# --- C. blocking-only digit runs -------------------------------------------------------

def test_block_digits_tokens_mode_is_the_normalised_field(monkeypatch):
    """Default mode returns exactly the normalised digits (no behaviour change)."""
    monkeypatch.setattr(config, "BLOCK_DIGIT_SOURCE", "tokens")
    f = _addr_frame(["24637b gregory road", "12 napean"], ["", "12"])
    assert blocking.block_digits(f) == [[], ["12"]]


def test_block_digits_runs_keep_alphanumeric_house_numbers(monkeypatch):
    """Digit runs inside alphanumeric tokens survive; zeros stripped; order kept."""
    monkeypatch.setattr(config, "BLOCK_DIGIT_SOURCE", "runs")
    f = _addr_frame(["24637b gregory road", "f 45d gtb enclave", "h a26 1 rajeev 026",
                     "office bw9021 9 floor", "no digits here"])
    assert blocking.block_digits(f) == [["24637"], ["45"], ["26", "1"], ["9021", "9"], []]


def test_block_digits_rejects_unknown_mode(monkeypatch):
    """A typo in the flag must fail loudly."""
    monkeypatch.setattr(config, "BLOCK_DIGIT_SOURCE", "nope")
    with pytest.raises(ValueError):
        blocking.block_digits(_addr_frame(["1 a"]))


# --- A. address keys -------------------------------------------------------------------

def test_address_keys_all_mode_shares_later_digits(monkeypatch):
    """"12 napean" vs "145 12 napean": no shared key in first mode, shared in all mode."""
    f = _addr_frame(["12 napean sea", "145 12 napean sea"], ["12", "145 12"])
    wdf = {"napean": 1, "sea": 5}
    monkeypatch.setattr(config, "ADDRESS_KEY_MODE", "first")
    a, b = blocking._address_keys(f, wdf)
    assert not set(a) & set(b)
    monkeypatch.setattr(config, "ADDRESS_KEY_MODE", "all")
    a, b = blocking._address_keys(f, wdf)
    assert "12@napean" in set(a) & set(b)


def test_address_keys_first_mode_matches_original_definition():
    """First mode keeps the original keys: digit pair + first digit @ 2 rarest words."""
    f = _addr_frame(["7 3 main bazaar kotwali"], ["7 3"])
    keys = blocking._address_keys(f, {"bazaar": 2, "kotwali": 1})[0]
    assert keys == ["7#3", "7@kotwali", "7@bazaar"]


def test_street_keys_are_rare_word_pairs_without_digits(monkeypatch):
    """Digit-free addresses get street-pair keys; stopwords never form keys."""
    monkeypatch.setattr(config, "STREET_WORDS", 3)
    f = _addr_frame(["geeta nilyam umesh cinema road hajipur"])
    wdf = {"geeta": 1, "nilyam": 1, "umesh": 2, "cinema": 9, "hajipur": 3}
    keys = blocking._street_keys(f, wdf)[0]
    assert keys == ["geeta+nilyam", "geeta+umesh", "nilyam+umesh"]
    assert not any("road" in k for k in keys)


# --- hashed key joins ------------------------------------------------------------------

def test_key_pass_hashed_join_matches_string_join():
    """Hashing keys does not change which pairs are joined or their scores."""
    rng = np.random.default_rng(0)
    vocab = [f"k{i}" for i in range(40)]
    q = pd.Series([list(rng.choice(vocab, 3)) for _ in range(60)], dtype=object)
    x = pd.Series([list(rng.choice(vocab, 3)) for _ in range(200)], dtype=object)
    got = blocking.key_pass(q, x, k=5, max_block=30)
    # reference: plain string join with the same scoring/tie-breaking
    xe, qe = x.explode(), q.explode()
    xd = pd.DataFrame({"x": xe.index, "key": xe.values}).drop_duplicates()
    qd = pd.DataFrame({"q": qe.index, "key": qe.values}).drop_duplicates()
    blk = xd["key"].value_counts()
    blk = blk[blk <= 30]
    w = np.log((len(q) + len(x)) / blk).astype(np.float32)
    j = qd[qd["key"].isin(blk.index)].merge(xd.assign(w=xd["key"].map(w)), on="key")
    ref = (j.groupby(["q", "x"])["w"].sum().reset_index()
           .sort_values(["q", "w", "x"], ascending=[True, False, True]).groupby("q").head(5))
    assert set(zip(got["q"], got["x"])) == set(zip(ref["q"], ref["x"]))


# --- pass wiring -----------------------------------------------------------------------

def test_flags_off_keep_the_original_five_pass_schema(synth):
    """With every new flag off, candidates and features carry only the base passes."""
    s1n, others, _ = synth
    cands = blocking.generate_candidates(s1n, others, verbose=False)
    assert [c for c in cands.columns if c.startswith("score_")] == \
        [f"score_{p}" for p in blocking.BASE_PASSES]
    bf = features.blocking_features(cands)
    assert "in_street" not in bf and "in_tfidf_addr" not in bf


def test_new_passes_add_bits_scores_and_features(synth, monkeypatch):
    """Enabled passes get their own bit, score column and in_/block_ features, and the
    union only grows."""
    s1n, others, truth = synth
    base = blocking.generate_candidates(s1n, others, verbose=False)
    for flag, val in (("USE_STREET_PASS", True), ("USE_TFIDF_ADDR_PASS", True),
                      ("ADDRESS_KEY_MODE", "all"), ("BLOCK_DIGIT_SOURCE", "runs")):
        monkeypatch.setattr(config, flag, val)
    cands = blocking.generate_candidates(s1n, others, verbose=False)
    for p in ("street", "tfidf_addr"):
        assert f"score_{p}" in cands
        has = (cands["passes"].to_numpy() & blocking.PASS_BITS[p]) > 0
        assert has.any()
        assert cands.loc[has, f"score_{p}"].notna().all()
        assert cands.loc[~has, f"score_{p}"].isna().all()
    assert len(cands) >= len(base)
    bf = features.blocking_features(cands)
    assert {"in_street", "block_street", "in_tfidf_addr", "block_tfidf_addr"} <= set(bf)
    ids = list(s1n["entity_id"])
    r_base = blocking.report_blocking_stats(base, truth, ids, len(others), verbose=False)
    r_new = blocking.report_blocking_stats(cands, truth, ids, len(others), verbose=False)
    assert r_new["pair_recall"] >= r_base["pair_recall"]
    assert "recall_street" in r_new and "recall_tfidf_addr" in r_new


def test_tfidf_addr_index_pruning_bounds_index_terms(synth, monkeypatch):
    """Index pruning still returns candidates and never more than k per query."""
    s1n, others, _ = synth
    monkeypatch.setattr(config, "TFIDF_ADDR_INDEX_TERMS", 8)
    q, x = s1n.reset_index(drop=True), others.reset_index(drop=True)
    res = blocking.tfidf_addr_pass(q, x, k=5)
    assert len(res) > 0 and res.groupby("q").size().max() <= 5


# --- D. memory: features ---------------------------------------------------------------

def test_embedding_cosine_is_bit_identical_to_full_gather():
    """Sub-block gathering gives exactly the full-gather row sums."""
    rng = np.random.default_rng(1)
    e1 = rng.standard_normal((50, 16)).astype(np.float32)
    e2 = rng.standard_normal((70, 16)).astype(np.float32)
    i1, i2 = rng.integers(0, 50, 1000), rng.integers(0, 70, 1000)
    ref = (e1[i1] * e2[i2]).sum(axis=1)
    assert np.array_equal(features.embedding_cosine(e1, e2, i1, i2, block=37), ref)


def test_build_features_chunking_and_no_copy_assembly(synth):
    """Chunk size never changes values; the float block is shared, not copied."""
    s1n, others, _ = synth
    cands = blocking.generate_candidates(s1n, others, verbose=False)
    emb = (fake_encoder(blocking.embedding_text(s1n)), fake_encoder(blocking.embedding_text(others)))
    ctx = features.FeatureContext.fit(s1n, others)
    a = features.build_features(cands, s1n, others, emb, ctx=ctx, chunk=97, verbose=False)
    b = features.build_features(cands, s1n, others, emb, ctx=ctx, chunk=10 ** 7, verbose=False)
    pd.testing.assert_frame_equal(a, b)
    assert list(a.columns[:2]) == ["s1_id", "cand_id"]
    # emb_cos equals the straightforward gathered product
    i1 = pd.Index(s1n["entity_id"]).get_indexer(cands["s1_id"])
    i2 = pd.Index(others["entity_id"]).get_indexer(cands["cand_id"])
    assert np.array_equal(a["emb_cos"].to_numpy(), (emb[0][i1] * emb[1][i2]).sum(axis=1))


# --- D. memory: embeddings -------------------------------------------------------------

def test_compute_embeddings_chunked_cache_equals_uncached(synth, monkeypatch):
    """Chunked encoding + memmap cache returns exactly the uncached vectors, leaves no
    temp file, and a second call is a cache hit."""
    s1n, _, _ = synth
    monkeypatch.setattr(config, "EMBEDDING_ENCODE_CHUNK", 7)
    plain = blocking.compute_embeddings(s1n, fake_encoder, cache=False)
    cached = blocking.compute_embeddings(s1n, fake_encoder, cache=True)
    assert isinstance(cached, np.memmap)
    assert np.array_equal(np.asarray(cached), plain)
    assert not list(config.ARTIFACTS_DIR.glob("*.tmp.npy"))
    calls = []
    again = blocking.compute_embeddings(s1n, lambda t: calls.append(t) or fake_encoder(t), cache=True)
    assert not calls and np.array_equal(np.asarray(again), plain)


# --- E. cache correctness --------------------------------------------------------------

def test_code_version_changes_with_source(tmp_path, monkeypatch):
    """The cache-key code hash reflects file content and ignores CRLF vs LF."""
    src = tmp_path / "src"
    src.mkdir()
    fake = src / "run_pipeline.py"
    fake.write_text("")
    monkeypatch.setattr(rp, "__file__", str(fake))
    (src / "normalize.py").write_bytes(b"x = 1\n")
    v1 = rp.code_version("normalize")
    (src / "normalize.py").write_bytes(b"x = 1\r\n")
    assert rp.code_version("normalize") == v1
    (src / "normalize.py").write_bytes(b"x = 2\n")
    assert rp.code_version("normalize") != v1


def test_normalisation_cache_key_includes_code_version(monkeypatch):
    """A normalize.py change (simulated) produces a new cache file, not a stale hit."""
    raw = pd.DataFrame({"entity_id": ["S1-1"], "business_name": ["Acme Inc"],
                        "business_address": ["1 Main St"], "country": ["US"]})
    monkeypatch.setattr(rp, "code_version", lambda *m: "v1")
    rp._normalize_and_cache("s1", raw, use_cache=True)
    monkeypatch.setattr(rp, "code_version", lambda *m: "v2")
    rp._normalize_and_cache("s1", raw, use_cache=True)
    assert len(list(config.ARTIFACTS_DIR.glob("norm_s1_*.parquet"))) == 2


def test_new_flags_are_tunables_so_they_key_the_candidate_cache():
    """Every new blocking flag is in TUNABLES, which run_pipeline.block hashes."""
    for name in ("ADDRESS_KEY_MODE", "ADDRESS_MAX_DIGITS", "BLOCK_DIGIT_SOURCE",
                 "USE_STREET_PASS", "K_STREET", "STREET_MAX_BLOCK", "STREET_WORDS",
                 "USE_TFIDF_ADDR_PASS", "K_TFIDF_ADDR", "TFIDF_ADDR_INDEX_TERMS"):
        assert name in config.TUNABLES
