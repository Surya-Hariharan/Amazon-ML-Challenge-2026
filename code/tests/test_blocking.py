"""Tests for src.blocking on synthetic data (chunking forced with tiny chunk sizes)."""

import numpy as np
import pandas as pd
import pytest
from scipy import sparse
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize as l2

from src import blocking, config
from src.blocking import (
    PASS_BITS,
    candidates_to_lists,
    compute_embeddings,
    country_groups,
    dense_topk,
    generate_candidates,
    key_pass,
    report_blocking_stats,
    sparse_topk,
)
from src.normalize import normalize_frame
from tests.synthetic import make_dataset


def fake_encoder(texts):
    """Deterministic stand-in for the sentence-transformer (char 3-gram hashing)."""
    hv = HashingVectorizer(analyzer="char_wb", ngram_range=(3, 3), n_features=256,
                           alternate_sign=False)
    return l2(hv.transform(texts)).toarray().astype(np.float32)


def _brute_topk_sets(q, x, k):
    """Reference top-k index sets by brute force."""
    sims = q @ x.T
    sims = sims.toarray() if sparse.issparse(sims) else sims
    return [set(np.argsort(-row, kind="stable")[:k]) for row in sims]


def test_sparse_topk_matches_brute_force():
    """Cost-chunked sparse top-k equals brute force (tiny budget forces many chunks)."""
    q = l2(sparse.random(23, 40, density=0.3, random_state=1, format="csr"))
    x = l2(sparse.random(57, 40, density=0.3, random_state=2, format="csr"))
    dense = (q @ x.T).toarray()
    for budget in (1, 30, 10**9):
        qi, xi, s = sparse_topk(q, x, k=5, max_product_nnz=budget)
        got = pd.DataFrame({"q": qi, "x": xi}).groupby("q")["x"].apply(set)
        for r in range(q.shape[0]):
            row = dense[r]
            n_pos = min(5, int((row > 0).sum()))
            kth = np.sort(row)[::-1][n_pos - 1] if n_pos else np.inf
            chosen = got.get(r, set())
            assert len(chosen) == n_pos
            assert all(row[i] >= kth - 1e-6 for i in chosen)  # ties may pick either
        np.testing.assert_allclose(s, dense[qi, xi], rtol=1e-5)


def test_prune_rows_keeps_largest_terms():
    """Query pruning keeps the n largest weights of each row."""
    m = sparse.csr_matrix(np.array([[0.1, 0.5, 0.3, 0.0], [0.0, 0.0, 0.2, 0.9]], np.float32))
    out = blocking.prune_rows(m, 2).toarray()
    np.testing.assert_allclose(out, [[0, 0.5, 0.3, 0], [0, 0, 0.2, 0.9]], rtol=1e-6)


def test_dense_topk_numpy_matches_brute_force(monkeypatch):
    """Streaming dense top-k (numpy path) equals brute force across chunk borders."""
    monkeypatch.setattr(blocking, "_torch_device", lambda: None)
    rng = np.random.default_rng(1)
    q = l2(rng.normal(size=(31, 16))).astype(np.float32)
    x = l2(rng.normal(size=(97, 16))).astype(np.float32)
    qi, xi, s = dense_topk(q, x, k=6, query_chunk=7, index_chunk=11)
    got = pd.DataFrame({"q": qi, "x": xi}).groupby("q")["x"].apply(set)
    ref = _brute_topk_sets(q, x, 6)
    for r, expected in enumerate(ref):
        assert got[r] == expected
    assert len(qi) == 31 * 6
    np.testing.assert_allclose(s, (q[qi] * x[xi]).sum(1), rtol=1e-5)


def test_dense_topk_gpu_path_if_available():
    """The torch (FP32) path agrees exactly with brute force -- no FP16 tolerance needed.

    dense_topk's CUDA branch computes the similarity matmul in float32 (never
    float16, per CLAUDE.md's FP32 retrieval requirement -- see
    test_dense_topk_cuda_branch_never_requests_float16 for a host-independent
    regression guard on that). With a real CUDA device this exercises that branch
    directly; it is skipped where no CUDA device is available.
    """
    if blocking._torch_device() is None:
        pytest.skip("no CUDA device")
    rng = np.random.default_rng(2)
    q = l2(rng.normal(size=(50, 32))).astype(np.float32)
    x = l2(rng.normal(size=(300, 32))).astype(np.float32)
    qi, xi, _ = dense_topk(q, x, k=5, query_chunk=16, index_chunk=64)
    got = pd.DataFrame({"q": qi, "x": xi}).groupby("q")["x"].apply(set)
    ref = _brute_topk_sets(q, x, 5)
    for r in range(50):
        assert got[r] == ref[r]


def test_dense_topk_cuda_branch_never_requests_float16(monkeypatch):
    """dense_topk's torch branch must request float32 tensors, never float16.

    Forces the "device is not None" branch (as CUDA would) by monkeypatching
    ``_torch_device`` to a CPU torch device -- so this runs identically on a
    CPU-only CI host, unlike test_dense_topk_gpu_path_if_available, which needs
    real CUDA hardware to exercise that branch at all. This is a direct
    regression guard for the production-audit finding that the CUDA branch used
    to hardcode ``torch.float16`` for the similarity matmul, contradicting the
    FP32 retrieval requirement.
    """
    torch = pytest.importorskip("torch")
    monkeypatch.setattr(blocking, "_torch_device", lambda: torch.device("cpu"))
    seen_dtypes = []
    orig_to = torch.Tensor.to

    def spy_to(self, *args, **kwargs):
        """Record any dtype requested by a ``.to(...)`` call, then delegate."""
        dtype = kwargs.get("dtype")
        for a in args:
            if isinstance(a, torch.dtype):
                dtype = a
        if dtype is not None:
            seen_dtypes.append(dtype)
        return orig_to(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", spy_to)
    rng = np.random.default_rng(3)
    q = l2(rng.normal(size=(9, 6))).astype(np.float32)
    x = l2(rng.normal(size=(13, 6))).astype(np.float32)
    dense_topk(q, x, k=4, query_chunk=3, index_chunk=4)
    assert seen_dtypes, "expected the torch branch to run and request a dtype"
    assert torch.float16 not in seen_dtypes
    assert all(d == torch.float32 for d in seen_dtypes)


def test_dense_topk_k_larger_than_index(monkeypatch):
    """k > index size returns every index row once."""
    monkeypatch.setattr(blocking, "_torch_device", lambda: None)
    q = np.eye(3, dtype=np.float32)
    x = np.eye(3, dtype=np.float32)[:2]
    qi, xi, _ = dense_topk(q, x, k=10, query_chunk=2, index_chunk=1)
    assert len(qi) == 6 and set(xi) == {0, 1}


def test_key_pass_caps_blocks_and_ranks_by_idf():
    """Keys over max_block are ignored; rarer shared keys rank first."""
    q_keys = pd.Series([["rare", "common"], ["common"], []])
    x_keys = pd.Series([["common"], ["common"], ["common"], ["rare", "common"], ["rare"]])
    res = key_pass(q_keys, x_keys, k=5, max_block=2, chunk=1)
    assert set(res["q"]) == {0}  # "common" (4 holders) is over the cap
    assert set(res["x"]) == {3, 4}
    res = key_pass(q_keys, x_keys, k=1, max_block=10)
    assert res[res["q"] == 0]["x"].tolist() == [3]  # shares rare + common


def test_country_groups_open_set():
    """Groups come from the data; an S1-only country gets an empty index."""
    s1 = pd.DataFrame({"country": ["US", "France", "Atlantis", "US"]})
    ot = pd.DataFrame({"country": ["France", "US", "US"]})
    groups = {c: (list(q), list(x)) for c, q, x in country_groups(s1, ot, True)}
    assert groups["US"] == ([0, 3], [1, 2])
    assert groups["France"] == ([1], [0])
    assert groups["Atlantis"] == ([2], [])
    assert [c for c, _, _ in country_groups(s1, ot, False)] == ["*"]


@pytest.fixture(scope="module")
def synthetic_blocked():
    """Normalised synthetic data with a third country, blocked with all passes."""
    s1, s2, s3, truth = make_dataset(n_s1=240, seed=3, countries=("US", "India", "France"))
    s1n = normalize_frame(s1)
    others = normalize_frame(pd.concat([s2, s3], ignore_index=True))
    emb = (fake_encoder(blocking.embedding_text(s1n)), fake_encoder(blocking.embedding_text(others)))
    cands = generate_candidates(s1n, others, embeddings=emb, verbose=False)
    return s1n, others, truth, cands


def test_generate_candidates_schema_and_recall(synthetic_blocked):
    """Union has one row per pair, valid pass bits, and high recall on easy data."""
    s1n, others, truth, cands = synthetic_blocked
    active = blocking.active_passes()
    assert list(cands.columns) == ["s1_id", "cand_id", "passes"] + [f"score_{p}" for p in active]
    assert not cands.duplicated(["s1_id", "cand_id"]).any()
    assert cands["passes"].between(1, sum(PASS_BITS[p] for p in active)).all()
    assert cands["cand_id"].str.match(r"^S[23]-").all()
    stats = report_blocking_stats(cands, truth, list(s1n["entity_id"]), len(others),
                                  dict(zip(s1n["entity_id"], s1n["country"])), verbose=False)
    assert stats["pair_recall"] >= 0.95
    assert 0 < stats["reduction_ratio"] < 1
    assert "recall_France" in stats  # unseen-in-training country flows through
    for p in active:
        assert stats[f"pairs_{p}"] > 0, p


def test_generate_candidates_k_overrides_change_only_the_named_pass(synthetic_blocked):
    """k_overrides changes exactly the requested pass's k, at the actual call site,
    without mutating config -- the P1 blocking-ablation diagnostic's core contract."""
    s1n, others, _, baseline_cands = synthetic_blocked
    emb = (fake_encoder(blocking.embedding_text(s1n)), fake_encoder(blocking.embedding_text(others)))
    before = {name: getattr(config, attr) for name, attr in (
        ("tfidf", "K_TFIDF_NAME"), ("embed", "K_EMBEDDING"), ("rare", "K_RARE_TOKEN"),
        ("digit", "K_POSTAL_TOKEN"), ("address", "K_ADDRESS"))}

    cands = generate_candidates(s1n, others, embeddings=emb, verbose=False,
                                k_overrides={"tfidf": 1})

    # Production config is never mutated by passing k_overrides.
    assert config.K_TFIDF_NAME == before["tfidf"]
    assert config.K_EMBEDDING == before["embed"]
    assert config.K_RARE_TOKEN == before["rare"]
    assert config.K_POSTAL_TOKEN == before["digit"]
    assert config.K_ADDRESS == before["address"]

    # k=1 for tfidf can only ever shrink (never grow) that pass's pair count relative
    # to the production-default run, while every other pass's pair count is untouched.
    base_by_pass = {p: (baseline_cands["passes"].to_numpy() & bit > 0).sum()
                    for p, bit in PASS_BITS.items()}
    new_by_pass = {p: (cands["passes"].to_numpy() & bit > 0).sum()
                  for p, bit in PASS_BITS.items()}
    assert new_by_pass["tfidf"] <= base_by_pass["tfidf"]
    for p in ("rare", "digit", "address", "embed"):
        assert new_by_pass[p] == base_by_pass[p]


def test_generate_candidates_k_overrides_none_matches_production_default(synthetic_blocked):
    """k_overrides=None (the default) reproduces the exact pre-existing candidate set --
    adding the parameter changes nothing about ordinary production calls."""
    s1n, others, _, baseline_cands = synthetic_blocked
    emb = (fake_encoder(blocking.embedding_text(s1n)), fake_encoder(blocking.embedding_text(others)))
    cands = generate_candidates(s1n, others, embeddings=emb, verbose=False)
    pd.testing.assert_frame_equal(
        cands.sort_values(["s1_id", "cand_id"]).reset_index(drop=True),
        baseline_cands.sort_values(["s1_id", "cand_id"]).reset_index(drop=True))


def test_generate_candidates_k_overrides_reaches_the_pass_function(monkeypatch):
    """An explicit k_override is actually threaded into the underlying pass call,
    not merely accepted and ignored."""
    s1 = pd.DataFrame({"entity_id": ["S1-1"], "country": ["US"], "name_core": ["acme"],
                       "digits": [""], "first_token": ["acme"], "addr_norm": ["1 main st"]})
    x = pd.DataFrame({"entity_id": ["S2-1"], "country": ["US"], "name_core": ["acme"],
                      "digits": [""], "first_token": ["acme"], "addr_norm": ["1 main st"]})
    seen_k = {}

    def spy_tfidf(q_names, x_names, k=config.K_TFIDF_NAME):
        seen_k["tfidf"] = k
        return blocking._empty_pass()

    monkeypatch.setattr(blocking, "tfidf_name_pass", spy_tfidf)
    # the name+address pass also routes through tfidf_name_pass; keep this about pass 1
    monkeypatch.setattr(config, "USE_TFIDF_ADDR_PASS", False)
    generate_candidates(s1, x, embeddings=None, use_address_pass=False, verbose=False,
                        k_overrides={"tfidf": 7})
    assert seen_k["tfidf"] == 7


def test_no_cross_country_candidates(synthetic_blocked):
    """Within-country blocking never proposes a candidate from another country."""
    s1n, others, _, cands = synthetic_blocked
    c1 = cands["s1_id"].map(dict(zip(s1n["entity_id"], s1n["country"])))
    c2 = cands["cand_id"].map(dict(zip(others["entity_id"], others["country"])))
    assert (c1 == c2).all()


def test_candidates_to_lists(synthetic_blocked):
    """Lists round-trip the pair frame."""
    _, _, _, cands = synthetic_blocked
    lists = candidates_to_lists(cands)
    assert sum(len(v) for v in lists.values()) == len(cands)


def test_report_blocking_stats_exact_numbers():
    """Hand-checked stats: 2 of 3 true pairs found, one S1 fully covered."""
    cands = pd.DataFrame({"s1_id": ["S1-1", "S1-1", "S1-2"], "cand_id": ["S2-1", "S3-9", "S2-5"],
                          "passes": np.array([1, 2, 3], dtype=np.uint8)})
    truth = {"S1-1": ["S2-1", "S3-2"], "S1-2": ["S2-5"], "S1-3": []}
    st = report_blocking_stats(cands, truth, ["S1-1", "S1-2", "S1-3"], n_other=10, verbose=False)
    assert st["pair_recall"] == pytest.approx(2 / 3)
    assert st["s1_full_recall"] == pytest.approx(0.5)
    assert st["mean_candidates"] == pytest.approx(1.0)
    assert st["reduction_ratio"] == pytest.approx(1 - 3 / 30)
    assert st["recall_tfidf"] == pytest.approx(2 / 3)
    assert st["unique_tfidf"] == pytest.approx(1 / 3)
    assert st["recall_embed"] == pytest.approx(1 / 3)


def test_compute_embeddings_cached(tmp_path, monkeypatch):
    """Embeddings are cached by content; the encoder runs once per distinct text."""
    monkeypatch.setattr(config, "ARTIFACTS_DIR", tmp_path)
    calls = []

    def enc(texts):
        """Counting fake encoder."""
        calls.append(len(texts))
        return fake_encoder(texts)

    df = pd.DataFrame({"business_name": ["A Co", "A Co", "B Inc"],
                       "business_address": ["1 X Rd", "1 X Rd", ""]})
    e1 = compute_embeddings(df, enc)
    e2 = compute_embeddings(df, enc)
    assert calls == [2]
    assert e1.shape == (3, 256) and np.array_equal(np.asarray(e1), np.asarray(e2))


def test_compute_embeddings_round_trips_at_float32(tmp_path, monkeypatch):
    """Embeddings are stored and reloaded at float32, matching dense_topk's FP32
    retrieval precision -- never float16 (production-audit fix: a prior version cast
    to float16 before caching, silently handing FP32-retrieval code lower-precision
    vectors on every cache hit)."""
    monkeypatch.setattr(config, "ARTIFACTS_DIR", tmp_path)
    df = pd.DataFrame({"business_name": ["A Co", "B Inc"],
                       "business_address": ["1 X Rd", "2 Y St"]})
    computed = compute_embeddings(df, fake_encoder, cache=True)
    assert computed.dtype == np.float32
    cache_files = list(tmp_path.glob("embf32_*.npy"))
    assert len(cache_files) == 1
    reloaded = compute_embeddings(df, encoder=None, cache=True)
    assert reloaded.dtype == np.float32
    np.testing.assert_array_equal(np.asarray(computed), np.asarray(reloaded))


def test_compute_embeddings_ignores_stale_float16_cache_file(tmp_path, monkeypatch):
    """A leftover float16 cache file from the old ``emb_<hash>.npy`` naming is never
    matched by the current float32 cache lookup -- the prefixes cannot collide, so a
    stale FP16 cache can never be silently reused as FP32 embeddings."""
    monkeypatch.setattr(config, "ARTIFACTS_DIR", tmp_path)
    tmp_path.mkdir(parents=True, exist_ok=True)
    # Simulate a leftover artifact from the old (pre-fix) float16 cache scheme: same
    # directory, old "emb_" prefix, wrong dtype and shape, so a false cache hit would
    # be easy to detect.
    stale = tmp_path / "emb_deadbeefdeadbeefdead.npy"
    np.save(stale, np.zeros((1, 1), dtype=np.float16))
    df = pd.DataFrame({"business_name": ["A Co"], "business_address": ["1 X Rd"]})
    out = compute_embeddings(df, fake_encoder, cache=True)
    assert out.dtype == np.float32
    assert out.shape == (1, 256)
    # The stale file is untouched and still float16 -- proving it was never read as
    # this call's cache.
    assert np.load(stale).dtype == np.float16
