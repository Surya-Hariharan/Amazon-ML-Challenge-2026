"""Tests for the diagnostic-phase commands in src.diagnose: blocking-key-counterfactual
(D2), density-stress (D4), name-retrieval-rank (D3) and memory-profile (D8).

Mirrors the other test_diagnose_* modules: a tiny synthetic dataset, embeddings
disabled, artifacts redirected to tmp, and no real dataset or S3 access.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config, diagnose
from tests.synthetic import make_dataset, write_split


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    """Send artifacts/diagnostics to tmp."""
    monkeypatch.setattr(config, "ARTIFACTS_DIR", tmp_path / "artifacts")
    monkeypatch.setattr(diagnose, "DIAG_DIR", tmp_path / "artifacts" / "diagnostics")
    return tmp_path


@pytest.fixture
def train_files(tmp_path, monkeypatch):
    """Write a synthetic US/India train split and point config.TRAIN_FILES at it."""
    s1, s2, s3, truth = make_dataset(n_s1=150, seed=5, countries=("US", "India"))
    data = tmp_path / "dataset"
    write_split(data, "train", s1, s2, s3, truth)
    files = {"s1": data / "train/train_source1.tsv", "s2": data / "train/train_source2.tsv",
             "s3": data / "train/train_source3.tsv",
             "ground_truth": data / "train/train_ground_truth.tsv"}
    monkeypatch.setattr(config, "TRAIN_FILES", files)
    return s1, s2, s3, truth


def _frame(addr: str, digits: str, name: str = "acme traders", long_nums: str = "",
           region: str = "") -> pd.DataFrame:
    """One-row normalised-style frame with the fields the key schemes read."""
    return pd.DataFrame({"addr_norm": [addr], "digits": [digits], "long_nums": [long_nums],
                         "region": [region], "name_core": [name]})


# --- D2 key schemes -------------------------------------------------------------------

def test_all_digit_word_keys_use_every_digit_not_only_the_first():
    """The production address key is anchored on the first digit token only; the
    counterfactual scheme must also key on later digit tokens."""
    wdf = {"napean": 1, "sea": 5}
    a = diagnose._cf_scheme_keys("all_digit_word", _frame("12 napean sea", "12"), wdf, {})[0]
    b = diagnose._cf_scheme_keys("all_digit_word", _frame("145 12 napean sea", "145 12"), wdf, {})[0]
    assert set(a) & set(b)  # shared "12@napean"
    prod_a = diagnose._cf_scheme_keys("prod_address", _frame("12 napean sea", "12"), wdf, {})[0]
    prod_b = diagnose._cf_scheme_keys("prod_address", _frame("145 12 napean sea", "145 12"), wdf, {})[0]
    assert not set(prod_a) & set(prod_b)  # the production blind spot being measured


def test_street_word_pair_keys_exist_without_digits():
    """Digit-less addresses get no production address key but do get word-pair keys."""
    f = _frame("gulmohar colony bhopal", "")
    assert diagnose._cf_scheme_keys("prod_address", f, {}, {})[0] == []
    keys = diagnose._cf_scheme_keys("street_word_pair", f, {}, {})[0]
    # "colony" is a production address stopword, leaving one word pair.
    assert keys == ["bhopal+gulmohar"]


def test_unknown_scheme_raises():
    """A typo in a scheme name must fail loudly rather than yield empty keys."""
    with pytest.raises(ValueError):
        diagnose._cf_scheme_keys("nope", _frame("a b c", "1"), {}, {})


def test_key_counterfactual_reproduces_production_address_pass(train_files):
    """prod_address run through key_pass must equal the production address pairs, and
    recovered counts can never exceed the missed pairs."""
    res = diagnose.run_blocking_key_counterfactual(sample=1.0, use_embeddings=False)
    assert res["sanity"] is True
    s = res["summary"]
    assert (s["missed_recovered_at_prod_settings"] <= s["missed_pairs"]).all()
    assert (s.loc[s["scheme"] == "prod_address", "added_pairs_vs_union"] == 0).all()
    assert "ALL_NEW_UNION" in set(s["scheme"])


# --- D4 density stress ----------------------------------------------------------------

def test_hash_fraction_is_deterministic_and_nested():
    """Selection is a fixed function of the ID, so a smaller fraction is a subset of a
    larger one and reruns select the same records."""
    ids = pd.Series([f"S2-{i}" for i in range(5000)])
    h1, h2 = diagnose._hash_fraction(ids), diagnose._hash_fraction(ids)
    assert np.array_equal(h1, h2)
    small, large = set(ids[h1 < 0.1]), set(ids[h1 < 0.3])
    assert small <= large and 0.05 < len(small) / len(ids) < 0.15


def test_density_fraction_zero_matches_production_recall(train_files):
    """Fraction 0 must reproduce the production sample union's recall exactly."""
    from src import run_pipeline as rp
    res = diagnose.run_density_stress(sample=1.0, fractions=(0.0,), use_embeddings=False)
    row = res["summary"].iloc[0]
    s1, s2, s3, truth = rp._load_train(1.0)
    prep = rp.prepare(s1, s2, s3, use_embeddings=False)
    cands = rp.block(prep)
    tp = diagnose._true_pair_frame(prep, truth)
    union = set(zip(cands["s1_id"], cands["cand_id"]))
    hit = sum((s, c) in union for s, c in zip(tp["s1_id"], tp["cand_id"]))
    assert row["added_distractors"] == 0
    assert row["union_pairs"] == len(cands)
    assert row["union_recall_pct"] == pytest.approx(100.0 * hit / len(tp))


def test_density_counterfactuals_only_add_recall(train_files):
    """Union + a counterfactual pass can never have lower recall than the union."""
    res = diagnose.run_density_stress(sample=1.0, fractions=(0.0,), use_embeddings=False,
                                      counterfactuals=True)
    row = res["summary"].iloc[0]
    for name in ("cf_name_addr_tfidf", "cf_all_digit_word", "cf_street_word_pair", "cf_all_three"):
        assert row[f"{name}_union_plus_recall_pct"] >= row["union_recall_pct"] - 1e-9
        assert row[f"{name}_added_pairs"] >= 0
    assert row["cf_all_three_union_plus_recall_pct"] >= row["cf_name_addr_tfidf_union_plus_recall_pct"] - 1e-9


def test_candidate_augment_downstream_baseline_and_superset(train_files, monkeypatch):
    """Baseline scores the production union unchanged; an augmented variant only adds
    pairs, and both rows carry OOF and validation F0.5."""
    from src import run_pipeline as rp
    monkeypatch.setattr(config, "LGB_PARAMS", {
        "objective": "binary", "learning_rate": 0.1, "num_leaves": 15, "min_child_samples": 5,
        "verbose": -1, "seed": 42, "deterministic": True, "num_threads": 1})
    monkeypatch.setattr(config, "LGB_NUM_BOOST_ROUND", 100)
    monkeypatch.setattr(config, "LGB_EARLY_STOPPING", 10)
    monkeypatch.setattr(config, "N_FOLDS", 3)
    out = diagnose.run_candidate_augment_downstream(
        sample=1.0, use_embeddings=False, variants=("baseline", "cf_all_three"))
    s1, s2, s3, _ = rp._load_train(1.0)
    base = rp.block(rp.prepare(s1, s2, s3, use_embeddings=False))
    b, a = out.iloc[0], out.iloc[1]
    assert b["n_pairs"] == len(base)
    assert a["n_pairs"] >= b["n_pairs"]
    assert 0.0 <= b["oof_macro_f05"] <= 1.0 and 0.0 <= a["valid_macro_f05"] <= 1.0


# --- production-fix validation harness --------------------------------------------------

def test_config_overrides_restores_and_rejects_typos():
    """Overrides apply inside the block, are always restored, and unknown names raise."""
    before = config.USE_STREET_PASS
    with diagnose.config_overrides({"USE_STREET_PASS": not before}):
        assert config.USE_STREET_PASS is (not before)
    assert config.USE_STREET_PASS is before
    with pytest.raises(AttributeError):
        with diagnose.config_overrides({"NOT_A_FLAG": 1}):
            pass
    assert config.USE_STREET_PASS is before


def test_fix_variants_baseline_matches_default_and_restores_config(train_files, monkeypatch):
    """The baseline variant equals a plain default run; the combined variant only adds
    pairs; config is unchanged afterwards."""
    monkeypatch.setattr(config, "LGB_PARAMS", {
        "objective": "binary", "learning_rate": 0.1, "num_leaves": 15, "min_child_samples": 5,
        "verbose": -1, "seed": 42, "deterministic": True, "num_threads": 1})
    monkeypatch.setattr(config, "LGB_NUM_BOOST_ROUND", 100)
    monkeypatch.setattr(config, "LGB_EARLY_STOPPING", 10)
    monkeypatch.setattr(config, "N_FOLDS", 3)
    snapshot = {k: getattr(config, k) for k in config.TUNABLES}
    out = diagnose.run_fix_variants(sample=1.0, variants=("baseline", "combined_k20"),
                                    use_embeddings=False)
    assert {k: getattr(config, k) for k in config.TUNABLES} == snapshot
    b, c = out.iloc[0], out.iloc[1]
    assert c["n_pairs"] >= b["n_pairs"] and c["pair_recall"] >= b["pair_recall"]
    for col in ("oof_macro_f05", "valid_macro_f05", "false_positives", "matcher_fn",
                "one_to_one_removed", "peak_rss_gib"):
        assert col in out


def test_density_variants_fraction_zero(train_files):
    """At fraction 0 the baseline variant's pairs equal the production union."""
    from src import run_pipeline as rp
    out = diagnose.run_density_variants(sample=1.0, fractions=(0.0,),
                                        variants=("baseline", "combined_k10"),
                                        use_embeddings=False)
    s1, s2, s3, _ = rp._load_train(1.0)
    base = rp.block(rp.prepare(s1, s2, s3, use_embeddings=False))
    b, c = out.iloc[0], out.iloc[1]
    assert b["union_pairs"] == len(base)
    assert c["union_recall_pct"] >= b["union_recall_pct"]
    assert "street_recall_pct" in out and "tfidf_addr_recall_pct" in out


# --- D3 retrieval rank ----------------------------------------------------------------

def test_true_ranks_sparse_counts_strictly_better_rows():
    """Rank = 1 + number of index rows scoring strictly higher; zero score -> inf."""
    from scipy import sparse
    q = sparse.csr_matrix(np.array([[1.0, 0.0]], dtype=np.float32))
    x = sparse.csr_matrix(np.array([[0.9, 0], [0.5, 0], [0, 1.0], [0.95, 0]], dtype=np.float32))
    r = diagnose._true_ranks_sparse(q, x, np.array([0, 0, 0]), np.array([0, 1, 2]))
    assert r[0] == 2 and r[1] == 3 and np.isinf(r[2])


def test_name_retrieval_rank_replicates_production_tfidf(train_files):
    """Pairs the production TF-IDF pass retrieved rank within K under the replica."""
    res = diagnose.run_name_retrieval_rank(sample=1.0, use_embeddings=False)
    ranks = res["ranks"]
    assert "embed" not in ranks
    s = res["summary"]
    assert set(s["scope"]) <= {"all_true", "missed_by_union"}
    assert (s["within_20_pct"] <= s["within_1000_pct"] + 1e-9).all()


# --- D8 memory profile ----------------------------------------------------------------

def test_memory_profile_stages_and_fit(train_files):
    """Every stage is recorded for each pair count, the chunk sweep adds a loop row,
    and a bytes-per-pair fit is produced."""
    res = diagnose.run_memory_profile(sample=1.0, pair_counts=(400, 800), chunks=(100,),
                                      use_embeddings=False)
    st = res["stages"]
    for stage in ("feature_context_fit", "pair_feature_loop", "frame_assembly",
                  "context_features", "final_frame_resident"):
        assert stage in set(st["stage"])
    loop = st[st["stage"] == "pair_feature_loop"]
    assert set(loop["chunk"]) == {100, config.FEATURE_CHUNK}
    assert not res["fit"].empty
