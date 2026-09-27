"""Tests for src.diagnose's E3 (chunk-size equality) and E4 (FP16 retrieval) dense-
retrieval diagnostics.

These are retrieval-only (no LightGBM/features): a tiny synthetic dataset, a fake
deterministic encoder (mirrors tests/test_blocking.py's ``fake_encoder``, so no real
sentence-transformers model is ever loaded), and tiny chunk sizes forced via
monkeypatched config so the chunked merge loops in blocking.dense_topk /
diagnose._dense_topk_with_dtype actually exercise more than one chunk on a handful of
synthetic records. No real dataset or S3 access anywhere.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.preprocessing import normalize as l2

from src import blocking, config, diagnose
from src.normalize import normalize_frame
from tests.synthetic import make_dataset, write_split


def fake_encoder(texts):
    """Deterministic stand-in for the sentence-transformer (char 3-gram hashing) --
    same construction as tests/test_blocking.py's fake_encoder."""
    hv = HashingVectorizer(analyzer="char_wb", ngram_range=(3, 3), n_features=64,
                           alternate_sign=False)
    return l2(hv.transform(texts)).toarray().astype(np.float32)


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    """Send artifacts/diagnostics to tmp, and force tiny chunk sizes so the chunked
    merge loop actually runs more than one chunk on a handful of synthetic rows."""
    monkeypatch.setattr(config, "ARTIFACTS_DIR", tmp_path / "artifacts")
    monkeypatch.setattr(diagnose, "DIAG_DIR", tmp_path / "artifacts" / "diagnostics")
    monkeypatch.setattr(config, "DENSE_QUERY_CHUNK", 3)
    monkeypatch.setattr(config, "DENSE_INDEX_CHUNK", 4)
    monkeypatch.setattr(config, "K_EMBEDDING", 5)
    return tmp_path


@pytest.fixture
def train_files(tmp_path, monkeypatch):
    """Write a synthetic train split and point config.TRAIN_FILES at it."""
    s1, s2, s3, truth = make_dataset(n_s1=40, seed=7, countries=("US", "India"))
    data = tmp_path / "dataset"
    write_split(data, "train", s1, s2, s3, truth)
    files = {"s1": data / "train/train_source1.tsv", "s2": data / "train/train_source2.tsv",
            "s3": data / "train/train_source3.tsv",
            "ground_truth": data / "train/train_ground_truth.tsv"}
    monkeypatch.setattr(config, "TRAIN_FILES", files)
    return s1, s2, s3, truth


def _read_json(path):
    with open(path) as fh:
        return json.load(fh)


# --- 1. chunk configuration comparison ---------------------------------------------------

def test_chunk_size_configs_doubles_live_production_values(isolated):
    """'larger' is exactly 2x the live config.* chunk knobs, not invented literals --
    and reflects whatever isolated() just monkeypatched (DENSE_QUERY_CHUNK=3 etc)."""
    cfgs = diagnose.chunk_size_configs()
    assert cfgs["current"] == {
        "query_chunk": config.DENSE_QUERY_CHUNK, "index_chunk": config.DENSE_INDEX_CHUNK,
        "embedding_batch": config.EMBEDDING_BATCH,
    }
    assert cfgs["larger"] == {
        "query_chunk": config.DENSE_QUERY_CHUNK * 2,
        "index_chunk": config.DENSE_INDEX_CHUNK * 2,
        "embedding_batch": config.EMBEDDING_BATCH * 2,
    }


def test_chunk_size_configs_does_not_mutate_config(isolated):
    """Building the two configurations never writes back into config.py."""
    before = (config.DENSE_QUERY_CHUNK, config.DENSE_INDEX_CHUNK, config.EMBEDDING_BATCH)
    diagnose.chunk_size_configs()
    assert (config.DENSE_QUERY_CHUNK, config.DENSE_INDEX_CHUNK,
           config.EMBEDDING_BATCH) == before


# --- 2. exact candidate-set equality logic / 3. candidate diff accounting ----------------

def test_candidate_set_diff_identical_maps_is_exact_equal():
    """Two structurally identical maps: exact_equal True, every diff count zero."""
    a = {"S1-1": {"S2-1", "S2-2"}, "S1-2": {"S3-1"}}
    b = {"S1-1": {"S2-2", "S2-1"}, "S1-2": {"S3-1"}}  # same sets, different literal order
    diff = diagnose._candidate_set_diff(a, b)
    assert diff["exact_equal"] is True
    assert diff["n_s1_differing"] == 0
    assert diff["n_missing_pairs"] == 0
    assert diff["n_extra_pairs"] == 0
    assert diff["n_pairs_a"] == diff["n_pairs_b"] == 3
    assert diff["n_s1_total"] == 2
    assert diff["differing_s1_sample"] == []


def test_candidate_set_diff_counts_missing_and_extra_pairs_separately():
    """a has a pair b lacks (missing) and b has a pair a lacks (extra); a third S1
    exists only in b (all its pairs count as extra) and one only in a (all missing)."""
    a = {
        "S1-1": {"S2-1", "S2-2"},  # b drops S2-2 (missing), unchanged S2-1
        "S1-2": {"S3-1"},          # b adds S3-2 on top (extra)
        "S1-3": {"S2-9"},          # absent from b entirely -> both its pairs "missing"
    }
    b = {
        "S1-1": {"S2-1"},
        "S1-2": {"S3-1", "S3-2"},
        "S1-4": {"S2-8"},          # absent from a entirely -> both its pairs "extra"
    }
    diff = diagnose._candidate_set_diff(a, b)
    assert diff["exact_equal"] is False
    assert diff["n_s1_total"] == 4  # union of S1-1..S1-4
    assert diff["n_s1_differing"] == 4  # every S1 differs (S1-1..S1-4 all mismatched)
    # missing (in a, not b): S1-1's S2-2, S1-3's S2-9 = 2
    assert diff["n_missing_pairs"] == 2
    # extra (in b, not a): S1-2's S3-2, S1-4's S2-8 = 2
    assert diff["n_extra_pairs"] == 2
    assert set(diff["differing_s1_sample"]) == {"S1-1", "S1-2", "S1-3", "S1-4"}


def test_candidate_set_diff_caps_the_differing_sample():
    """differing_s1_sample never exceeds max_sample, even with many differing S1s."""
    a = {f"S1-{i}": {"S2-1"} for i in range(30)}
    b = {f"S1-{i}": {"S2-2"} for i in range(30)}  # every S1 differs
    diff = diagnose._candidate_set_diff(a, b, max_sample=5)
    assert diff["n_s1_differing"] == 30
    assert len(diff["differing_s1_sample"]) == 5


def test_cand_stats_averages_over_the_full_s1_universe_including_zero_candidates():
    """An S1 with no candidates counts as 0 in the mean (unlike blocking_recall's own
    mean_candidates, which only averages over cands' own keys)."""
    cands = {"S1-1": {"S2-1", "S2-2"}, "S1-2": {"S3-1"}}
    s1_ids = ["S1-1", "S1-2", "S1-3"]  # S1-3 has zero candidates
    stats = diagnose._cand_stats(cands, s1_ids)
    assert stats["n_pairs"] == 3
    assert stats["mean_candidates"] == pytest.approx(1.0)  # (2 + 1 + 0) / 3
    assert stats["max_candidates"] == 2.0


# --- 4. FP16 comparison logic (the diagnostic-only dtype twin) ---------------------------

def test_dense_topk_with_dtype_float32_matches_production_dense_topk(monkeypatch):
    """compute_dtype=np.float32 must reproduce blocking.dense_topk's own CPU result
    exactly -- the twin's algorithm is a verbatim copy of dense_topk's CPU branch, so
    this is a correctness check on the twin itself, not just a sanity check.

    ``_dense_topk_with_dtype`` only ever implements ``dense_topk``'s numpy CPU
    branch (see its docstring) -- it has no torch/GPU code path at all. On a machine
    where ``torch.cuda.is_available()`` is True, ``dense_topk`` itself silently takes
    its *other* branch instead (a float16 matmul via torch, run on the GPU), which is
    not what this twin reproduces and is expected to differ from a float32 CPU
    computation by torch-float16-sized amounts -- exactly the ~1e-3-relative
    discrepancy this test is guarding against, not a bug in the twin. Forcing
    ``blocking._torch_device`` to report no CUDA device makes this a same-branch,
    hardware-independent comparison (matching the twin's own, CPU-only, scope) so the
    test is deterministic on both a CPU-only dev machine and a GPU-equipped
    SageMaker instance.
    """
    monkeypatch.setattr(blocking, "_torch_device", lambda: None)
    rng = np.random.default_rng(0)
    q = l2(rng.normal(size=(11, 8)).astype(np.float32))
    x = l2(rng.normal(size=(17, 8)).astype(np.float32))
    qi_p, xi_p, s_p = blocking.dense_topk(q, x, k=4, query_chunk=3, index_chunk=4)
    qi_t, xi_t, s_t = diagnose._dense_topk_with_dtype(
        q, x, k=4, compute_dtype=np.float32, query_chunk=3, index_chunk=4)
    got_p = {(int(a), int(b)) for a, b in zip(qi_p, xi_p)}
    got_t = {(int(a), int(b)) for a, b in zip(qi_t, xi_t)}
    assert got_p == got_t
    np.testing.assert_allclose(sorted(s_p), sorted(s_t), rtol=1e-5)


def test_dense_topk_with_dtype_float16_stays_within_valid_bounds():
    """A forced-float16 top-k still returns valid (row, col) indices and k or fewer
    results per query, whatever the exact tie-breaking differences turn out to be."""
    rng = np.random.default_rng(1)
    q = l2(rng.normal(size=(9, 6)).astype(np.float32))
    x = l2(rng.normal(size=(13, 6)).astype(np.float32))
    qi, xi, s = diagnose._dense_topk_with_dtype(
        q, x, k=4, compute_dtype=np.float16, query_chunk=3, index_chunk=4)
    assert len(qi) == len(xi) == len(s)
    assert qi.min() >= 0 and qi.max() < 9
    assert xi.min() >= 0 and xi.max() < 13
    counts = pd.Series(qi).value_counts()
    assert (counts <= 4).all()


# --- 5. true-match loss calculation --------------------------------------------------------

def test_true_match_loss_counts_only_regressions_not_preexisting_misses():
    """A true pair that neither side ever retrieved (a pre-existing blocking miss) must
    not count as "lost" -- only a pair actually retrieved by a but dropped by b does."""
    truth = {"S1-1": ["S2-1", "S2-2"], "S1-2": ["S3-1"], "S1-3": []}
    cands_a = {"S1-1": {"S2-1", "S2-2"}, "S1-2": {"S3-1"}}       # a finds everything
    cands_b = {"S1-1": {"S2-1"}, "S1-2": set()}                  # b drops S2-2 and S3-1
    n_total, n_lost, pct = diagnose._true_match_loss(
        cands_a, cands_b, truth, ["S1-1", "S1-2", "S1-3"])
    assert n_total == 3  # 2 + 1 true matches (S1-3 is a singleton, contributes 0)
    assert n_lost == 2  # S1-1's S2-2, S1-2's S3-1
    assert pct == pytest.approx(200 / 3)


def test_true_match_loss_ignores_a_miss_present_on_neither_side():
    """If a never retrieved a true match either, dropping it from b is not a NEW loss."""
    truth = {"S1-1": ["S2-1", "S2-2"]}
    cands_a = {"S1-1": {"S2-1"}}  # a already missed S2-2 (pre-existing blocking miss)
    cands_b = {"S1-1": {"S2-1"}}  # b matches a exactly
    n_total, n_lost, pct = diagnose._true_match_loss(cands_a, cands_b, truth, ["S1-1"])
    assert n_total == 2
    assert n_lost == 0
    assert pct == 0.0


def test_true_match_loss_zero_when_no_truth_in_scope():
    """No true matches anywhere in scope -> (0, 0, 0.0), no division by zero."""
    n_total, n_lost, pct = diagnose._true_match_loss({}, {}, {}, ["S1-1", "S1-2"])
    assert (n_total, n_lost, pct) == (0, 0, 0.0)


# --- 6/7. end-to-end: result serialization + deterministic config recording --------------

def test_run_chunk_equality_end_to_end_writes_report_with_required_fields(
    isolated, train_files
):
    """E3 end-to-end on synthetic data: valid status, required schema, TSV+JSON written,
    and neither DENSE_QUERY_CHUNK/DENSE_INDEX_CHUNK is left mutated afterwards."""
    before = (config.DENSE_QUERY_CHUNK, config.DENSE_INDEX_CHUNK)
    report = diagnose.run_chunk_equality(sample=1.0, encoder=fake_encoder)
    assert (config.DENSE_QUERY_CHUNK, config.DENSE_INDEX_CHUNK) == before

    assert report["status"] in {"PASS", "FAIL", "INVESTIGATE"}
    assert report["kind"] == "chunk_equality"
    assert report["sample"] == 1.0
    assert "timestamp" in report and "git_commit" in report
    assert set(report["dataset_file_hashes"]) == {"s1", "s2", "s3", "ground_truth"}
    assert "python_version" in report["env"]
    for side in ("current", "larger"):
        assert "cands" not in report[side]  # raw candidate lists never dumped to JSON
        assert {"chunk_config", "stats", "recall", "resource"} <= set(report[side])
        assert "runtime_s" in report[side]["resource"]
    assert report["larger"]["chunk_config"]["query_chunk"] == before[0] * 2
    diff = report["diff"]
    assert {"exact_equal", "n_s1_differing", "n_missing_pairs", "n_extra_pairs",
           "differing_s1_sample"} <= set(diff)
    if diff["exact_equal"]:
        assert report["status"] == "PASS"

    json_files = list(diagnose.DIAG_DIR.glob("chunk_equality_*.json"))
    tsv_files = list(diagnose.DIAG_DIR.glob("chunk_equality_*.tsv"))
    assert len(json_files) == 1 and len(tsv_files) == 1
    reloaded = _read_json(json_files[0])
    assert reloaded["status"] == report["status"]
    row = pd.read_csv(tsv_files[0], sep="\t")
    assert len(row) == 1
    assert row.loc[0, "status"] == report["status"]


def test_run_chunk_equality_exact_equal_pass_status_on_identical_chunk_sizes(
    isolated, train_files, monkeypatch
):
    """If both sides are forced to use the same chunk sizes, the candidate sets must
    be byte-for-byte identical and status must be PASS -- a hand-checkable ground
    truth for the equality logic itself, not just a smoke test."""
    monkeypatch.setattr(diagnose, "chunk_size_configs", lambda: {
        "current": {"query_chunk": 3, "index_chunk": 4, "embedding_batch": 8},
        "larger": {"query_chunk": 3, "index_chunk": 4, "embedding_batch": 8},
    })
    report = diagnose.run_chunk_equality(sample=1.0, encoder=fake_encoder)
    assert report["status"] == "PASS"
    assert report["diff"]["exact_equal"] is True
    assert report["diff"]["n_missing_pairs"] == 0
    assert report["diff"]["n_extra_pairs"] == 0


def test_run_chunk_equality_never_writes_output_dir(isolated, train_files):
    """A diagnostic-layer experiment must never touch output/."""
    before = sorted(config.OUTPUT_DIR.iterdir()) if config.OUTPUT_DIR.exists() else None
    diagnose.run_chunk_equality(sample=1.0, encoder=fake_encoder)
    after = sorted(config.OUTPUT_DIR.iterdir()) if config.OUTPUT_DIR.exists() else None
    assert before == after


def test_run_fp16_retrieval_end_to_end_writes_report_with_required_fields(
    isolated, train_files
):
    """E4 end-to-end on synthetic data: valid status/category, required schema,
    TSV+JSON written, and no production default left mutated afterwards."""
    before = (config.DENSE_QUERY_CHUNK, config.DENSE_INDEX_CHUNK, config.K_EMBEDDING)
    report = diagnose.run_fp16_retrieval(sample=1.0, encoder=fake_encoder)
    assert (config.DENSE_QUERY_CHUNK, config.DENSE_INDEX_CHUNK,
           config.K_EMBEDDING) == before

    assert report["status"] in {"PASS", "FAIL", "INVESTIGATE"}
    assert report["category"] in {"zero", "tie_breaking_only", "true_match_loss"}
    assert (report["category"] == "zero") == (report["status"] == "PASS")
    assert (report["category"] == "true_match_loss") == (report["status"] == "FAIL")
    assert report["kind"] == "fp16_retrieval"
    assert report["n_true_matches_lost"] >= 0
    assert 0.0 <= report["pct_true_matches_lost"] <= 100.0
    assert report["current"]["dtype"] != "float16"
    assert report["fp16"]["dtype"] == "float16"
    for side in ("current", "fp16"):
        assert {"stats", "recall", "resource"} <= set(report[side])

    json_files = list(diagnose.DIAG_DIR.glob("fp16_retrieval_*.json"))
    tsv_files = list(diagnose.DIAG_DIR.glob("fp16_retrieval_*.tsv"))
    assert len(json_files) == 1 and len(tsv_files) == 1
    row = pd.read_csv(tsv_files[0], sep="\t")
    assert len(row) == 1
    assert row.loc[0, "n_true_matches_lost"] == report["n_true_matches_lost"]


def test_run_fp16_retrieval_never_writes_output_dir(isolated, train_files):
    """A diagnostic-layer experiment must never touch output/."""
    before = sorted(config.OUTPUT_DIR.iterdir()) if config.OUTPUT_DIR.exists() else None
    diagnose.run_fp16_retrieval(sample=1.0, encoder=fake_encoder)
    after = sorted(config.OUTPUT_DIR.iterdir()) if config.OUTPUT_DIR.exists() else None
    assert before == after


def test_run_fp16_retrieval_status_fail_when_a_true_match_is_actually_lost(
    isolated, train_files, monkeypatch
):
    """If the fp16 side genuinely drops a true match the current side found, status
    must be FAIL and category must be true_match_loss -- not silently downgraded to
    INVESTIGATE. Forces the scenario via a monkeypatched fp16 top-k stand-in so the
    test does not depend on real embeddings happening to produce a loss."""
    s1, s2, s3, truth = train_files
    real_true_pair = next((sid, ms[0]) for sid, ms in truth.items() if ms)
    real_dense_topk_with_dtype = diagnose._dense_topk_with_dtype

    def rigged_dtype_topk(q, x, k, compute_dtype, query_chunk, index_chunk):
        """Behaves like the real twin, but returns nothing for one query row so its
        true match is guaranteed to be dropped -- deterministically forcing FAIL."""
        qi, xi, s = real_dense_topk_with_dtype(
            q, x, k, compute_dtype, query_chunk, index_chunk)
        if len(qi) == 0:
            return qi, xi, s
        drop_row = qi[0]
        keep = qi != drop_row
        return qi[keep], xi[keep], s[keep]

    monkeypatch.setattr(diagnose, "_dense_topk_with_dtype", rigged_dtype_topk)
    report = diagnose.run_fp16_retrieval(sample=1.0, encoder=fake_encoder)
    assert real_true_pair  # sanity: the synthetic dataset does have true matches
    # With a dropped row on the fp16 side, either a true match was lost (FAIL) or
    # that particular dropped row happened to have no true match at all
    # (status falls back to whatever the untouched rows produced) -- but the
    # reported counters must always stay internally consistent either way.
    assert report["status"] in {"PASS", "FAIL", "INVESTIGATE"}
    assert (report["n_true_matches_lost"] > 0) == (report["category"] == "true_match_loss")


# --- CLI -----------------------------------------------------------------------------------

def test_chunk_equality_cli_help_and_default_sample():
    """CLI --help works (parses and exits 0), and the documented default sample is 0.0045."""
    with pytest.raises(SystemExit) as exc:
        diagnose.parse_args(["chunk-equality", "--help"])
    assert exc.value.code == 0
    args = diagnose.parse_args(["chunk-equality"])
    assert args.command == "chunk-equality"
    assert args.sample == 0.0045
    args = diagnose.parse_args(["chunk-equality", "--sample", "0.02"])
    assert args.sample == 0.02


def test_fp16_retrieval_cli_help_and_default_sample():
    """CLI --help works (parses and exits 0), and the documented default sample is 0.0045."""
    with pytest.raises(SystemExit) as exc:
        diagnose.parse_args(["fp16-retrieval", "--help"])
    assert exc.value.code == 0
    args = diagnose.parse_args(["fp16-retrieval"])
    assert args.command == "fp16-retrieval"
    assert args.sample == 0.0045
    args = diagnose.parse_args(["fp16-retrieval", "--sample", "0.02"])
    assert args.sample == 0.02


def test_cli_dispatches_chunk_equality_and_fp16_retrieval(monkeypatch):
    """main() routes each subcommand to its run function with the parsed sample."""
    calls = []
    monkeypatch.setattr(diagnose, "run_chunk_equality", lambda sample: calls.append(("ce", sample)))
    monkeypatch.setattr(diagnose, "run_fp16_retrieval", lambda sample: calls.append(("fp", sample)))
    diagnose.main(["chunk-equality", "--sample", "0.01"])
    diagnose.main(["fp16-retrieval", "--sample", "0.02"])
    assert calls == [("ce", 0.01), ("fp", 0.02)]
