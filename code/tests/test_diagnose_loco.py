"""Tests for src.diagnose's LOCO country-generalization comparison (baseline 20/20
vs both40 40/40 candidate K, evaluated per country-held-out direction).

Mirrors test_diagnose_decision_validation.py's conventions: a tiny synthetic
dataset, a fast LightGBM config, embeddings disabled by default, and no real
dataset or S3 access anywhere. Focus areas: (1) the paired baseline-vs-both40
comparison changes only K, (2) country-held-out split correctness (no S1/record
crossing the split), (3) no validation-label leakage -- including the
FeatureContext-fit-scope fix relative to run_pipeline.valid_run's own loco=True
branch, (4) configuration recording, (5) output schema, (6) France/no-ground-truth
handling, and (7) deterministic behaviour.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src import config, diagnose, features
from tests.synthetic import make_dataset, write_split

FAST_LGB = {"objective": "binary", "learning_rate": 0.1, "num_leaves": 15,
            "min_child_samples": 5, "verbose": -1, "seed": 42, "deterministic": True,
            "num_threads": 1}


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    """Send artifacts/experiments/diagnostics to tmp; use a small, fast LightGBM config."""
    monkeypatch.setattr(config, "ARTIFACTS_DIR", tmp_path / "artifacts")
    monkeypatch.setattr(config, "EXPERIMENTS_CSV", tmp_path / "experiments.csv")
    monkeypatch.setattr(config, "LGB_PARAMS", FAST_LGB)
    monkeypatch.setattr(config, "LGB_NUM_BOOST_ROUND", 200)
    monkeypatch.setattr(config, "LGB_EARLY_STOPPING", 20)
    monkeypatch.setattr(config, "N_FOLDS", 3)
    monkeypatch.setattr(diagnose, "DIAG_DIR", tmp_path / "artifacts" / "diagnostics")
    return tmp_path


@pytest.fixture
def train_files(tmp_path, monkeypatch):
    """Write a synthetic US/India train split and point config.TRAIN_FILES at it."""
    s1, s2, s3, truth = make_dataset(n_s1=260, seed=31, countries=("US", "India"))
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


# --- 1. paired baseline vs both40 configuration ----------------------------------------

def test_loco_configs_differ_only_in_tfidf_and_embed_k():
    """The only intended architecture difference between the two configurations is
    tfidf/embed K -- rare/digit/address must be identical."""
    by_name = {c["name"]: c for c in diagnose.LOCO_CONFIGS}
    assert set(by_name) == {"baseline", "both40"}
    assert by_name["baseline"]["tfidf"] == 20 and by_name["baseline"]["embed"] == 20
    assert by_name["both40"]["tfidf"] == 40 and by_name["both40"]["embed"] == 40
    for key in ("rare", "digit", "address"):
        assert by_name["baseline"][key] == by_name["both40"][key]


def test_loco_run_produces_one_row_per_config_per_direction(isolated, train_files):
    """2 configs x 2 directions (US, India) = 4 rows, each recording its own K."""
    out = diagnose.run_loco_comparison(sample=1.0, use_embeddings=False)
    assert len(out) == 4
    assert set(zip(out["configuration"], out["held_out_country"])) == {
        ("baseline", "US"), ("baseline", "India"),
        ("both40", "US"), ("both40", "India"),
    }
    baseline = out[out["configuration"] == "baseline"]
    both40 = out[out["configuration"] == "both40"]
    assert (baseline["k_tfidf"] == 20).all() and (baseline["k_embed"] == 20).all()
    assert (both40["k_tfidf"] == 40).all() and (both40["k_embed"] == 40).all()
    assert (baseline["k_rare"] == both40["k_rare"].to_numpy()).all()


# --- 2. country-held-out split correctness -----------------------------------------------

def test_country_split_partitions_without_overlap_or_loss(isolated, train_files):
    """Every S1/other record ends up in exactly one side, and each side is pure in
    its own country."""
    s1, s2, s3, truth = train_files
    prep = diagnose.rp.prepare(s1, s2, s3, use_embeddings=False, use_cache=False)
    for country in ("US", "India"):
        tr, ho = diagnose._country_split(prep, country)
        assert (ho["s1"][config.COUNTRY_COL] == country).all()
        assert (tr["s1"][config.COUNTRY_COL] != country).all()
        assert len(tr["s1"]) + len(ho["s1"]) == len(prep["s1"])
        assert (ho["others"][config.COUNTRY_COL] == country).all()
        assert (tr["others"][config.COUNTRY_COL] != country).all()
        assert len(tr["others"]) + len(ho["others"]) == len(prep["others"])
        tr_ids = set(tr["s1"][config.ID_COL])
        ho_ids = set(ho["s1"][config.ID_COL])
        assert tr_ids.isdisjoint(ho_ids)


def test_loco_fold_never_mixes_train_and_held_out_s1_ids(isolated, train_files):
    """_loco_fold's own n_train_s1 + n_valid_s1 accounts for every S1 in that
    country's partition, with no double-count."""
    s1, s2, s3, truth = train_files
    prep = diagnose.rp.prepare(s1, s2, s3, use_embeddings=False, use_cache=False)
    k_overrides = {"tfidf": 20, "embed": 20, "rare": 10, "digit": 10, "address": 10}
    row = diagnose._loco_fold(prep, truth, k_overrides, "US")
    assert row["n_train_s1"] + row["n_valid_s1"] == len(prep["s1"])
    assert row["n_valid_s1"] == (prep["s1"][config.COUNTRY_COL] == "US").sum()


# --- 3. no validation-label leakage ------------------------------------------------------

def test_feature_context_is_fit_only_on_training_side(isolated, train_files, monkeypatch):
    """The leakage fix: FeatureContext.fit must be called with the training-country
    frames only, and its result must be the SAME object passed as ctx= to both
    build_features calls (train and held-out) -- never re-fit on pooled data."""
    seen_fit_args = []
    seen_ctx_ids = []
    real_fit = features.FeatureContext.fit
    real_build = diagnose.build_features

    def spy_fit(s1, others, *a, **kw):
        ctx = real_fit(s1, others, *a, **kw)
        seen_fit_args.append((set(s1[config.COUNTRY_COL]), set(others[config.COUNTRY_COL])))
        return ctx

    def spy_build(cands, s1, others, embeddings=None, ctx=None, **kw):
        seen_ctx_ids.append(id(ctx))
        return real_build(cands, s1, others, embeddings, ctx=ctx, **kw)

    monkeypatch.setattr(features.FeatureContext, "fit", spy_fit)
    monkeypatch.setattr(diagnose, "build_features", spy_build)

    s1, s2, s3, truth = train_files
    prep = diagnose.rp.prepare(s1, s2, s3, use_embeddings=False, use_cache=False)
    k_overrides = {"tfidf": 20, "embed": 20, "rare": 10, "digit": 10, "address": 10}
    diagnose._loco_fold(prep, truth, k_overrides, "US")

    assert len(seen_fit_args) == 1  # fit exactly once per fold
    countries_seen = seen_fit_args[0][0] | seen_fit_args[0][1]
    assert "US" not in countries_seen  # never saw the held-out country's own records
    assert len(seen_ctx_ids) == 2  # once for train pairs, once for held-out pairs
    assert seen_ctx_ids[0] == seen_ctx_ids[1]  # the SAME fitted context, not refit


def test_held_out_labels_never_reach_training(isolated, train_files, monkeypatch):
    """label_pairs on the training feature frame must never be able to see a
    held-out-country S1's true matches -- verified by checking that every
    (s1_id, cand_id) labelled 1 in feats_tr belongs to a training-country S1."""
    s1, s2, s3, truth = train_files
    prep = diagnose.rp.prepare(s1, s2, s3, use_embeddings=False, use_cache=False)
    tr, ho = diagnose._country_split(prep, "US")
    tr_ids = set(tr["s1"][config.ID_COL])

    from src.blocking import generate_candidates
    from src.features import FeatureContext, build_features, label_pairs
    k_overrides = {"tfidf": 20, "embed": 20, "rare": 10, "digit": 10, "address": 10}
    cands_tr = generate_candidates(tr["s1"], tr["others"], embeddings=None, verbose=False,
                                   k_overrides=k_overrides)
    ctx = FeatureContext.fit(tr["s1"], tr["others"])
    feats_tr = build_features(cands_tr, tr["s1"], tr["others"], None, ctx=ctx, verbose=False)
    feats_tr["label"] = label_pairs(feats_tr, truth)
    assert set(feats_tr["s1_id"]) <= tr_ids  # candidate generation never crossed the split


def test_tune_threshold_universe_excludes_held_out_country(isolated, train_files, monkeypatch):
    """decide.tune_threshold must be called with s1_ids restricted to the training
    side only -- never the full S1 universe, which would let held-out-country truth
    enter the threshold sweep."""
    seen_s1_ids = []
    real_tune = diagnose.tune_threshold

    def spy_tune(scored, truth, s1_ids, *a, **kw):
        seen_s1_ids.append(list(s1_ids))
        return real_tune(scored, truth, s1_ids, *a, **kw)

    monkeypatch.setattr(diagnose, "tune_threshold", spy_tune)
    s1, s2, s3, truth = train_files
    prep = diagnose.rp.prepare(s1, s2, s3, use_embeddings=False, use_cache=False)
    k_overrides = {"tfidf": 20, "embed": 20, "rare": 10, "digit": 10, "address": 10}
    diagnose._loco_fold(prep, truth, k_overrides, "US")

    assert len(seen_s1_ids) == 1
    tuned_ids = set(seen_s1_ids[0])
    ho_ids = set(prep["s1"].loc[prep["s1"][config.COUNTRY_COL] == "US", config.ID_COL])
    assert tuned_ids.isdisjoint(ho_ids)


# --- 4. configuration recording -----------------------------------------------------------

def test_json_sidecar_records_configs_directions_and_leakage_checks(isolated, train_files):
    """The JSON sidecar records the exact configurations, directions, skipped
    directions, limitations, leakage-check documentation, git commit, dataset
    hashes and env info."""
    diagnose.run_loco_comparison(sample=1.0, use_embeddings=False)
    json_files = list(diagnose.DIAG_DIR.glob("loco_comparison_*.json"))
    assert len(json_files) == 1
    meta = _read_json(json_files[0])

    assert meta["kind"] == "loco_comparison"
    assert meta["configs"] == diagnose.LOCO_CONFIGS
    assert set(meta["directions"]) == {"US", "India"}
    assert meta["countries_in_sample"] == sorted({"US", "India"})
    assert meta["leakage_checks"] == diagnose.LOCO_LEAKAGE_CHECKS
    assert any("France" in s for s in meta["limitations"])
    assert "git_commit" in meta
    assert set(meta["dataset_file_hashes"]) == {"s1", "s2", "s3", "ground_truth"}
    assert "python_version" in meta["env"]
    assert len(meta["rows"]) == 4


# --- 5. output schema -----------------------------------------------------------------------

def test_output_schema_has_the_required_columns(isolated, train_files):
    """Every row carries the comparison-table columns the task requires, plus the
    fuller per-direction metric set, and TSV/JSON sidecars are written."""
    out = diagnose.run_loco_comparison(sample=1.0, use_embeddings=False)
    required = {
        "configuration", "held_out_country", "n_train_s1", "n_valid_s1", "n_pairs",
        "block_pair_recall", "block_s1_full_recall", "block_mean_candidates",
        "block_max_candidates", "oof_macro_f05", "macro_f05", "precision", "recall",
        "singleton_f05", "non_singleton_f05", "tau", "singleton_tau",
        "fold_f05", "fold_f05_mean", "fold_f05_std",
        "runtime_seconds", "peak_rss_bytes", "current_rss_bytes", "sample",
    }
    assert required <= set(out.columns)

    tsv_files = list(diagnose.DIAG_DIR.glob("loco_comparison_*.tsv"))
    json_files = list(diagnose.DIAG_DIR.glob("loco_comparison_*.json"))
    assert len(tsv_files) == 1
    assert len(json_files) == 1
    roundtrip = pd.read_csv(tsv_files[0], sep="\t")
    assert len(roundtrip) == len(out)


def test_never_touches_output_dir(isolated, train_files):
    """A diagnostic-layer experiment must never touch output/ (CLAUDE.md rule)."""
    before = sorted(config.OUTPUT_DIR.iterdir()) if config.OUTPUT_DIR.exists() else None
    diagnose.run_loco_comparison(sample=1.0, use_embeddings=False)
    after = sorted(config.OUTPUT_DIR.iterdir()) if config.OUTPUT_DIR.exists() else None
    assert before == after


# --- 6. France / no-ground-truth handling -------------------------------------------------

def test_france_is_never_a_direction_when_absent_from_training_sample(isolated, train_files):
    """With a US/India-only training sample (the real CLAUDE.md scenario), France
    must never appear as a LOCO direction, and no France row is fabricated."""
    out = diagnose.run_loco_comparison(sample=1.0, use_embeddings=False)
    assert "France" not in set(out["held_out_country"])
    assert "France" not in diagnose.LOCO_CONFIGS[0]  # sanity: not baked into configs either


def test_country_with_no_remaining_training_country_is_skipped_not_fabricated(isolated, tmp_path,
                                                                               monkeypatch):
    """If a sample happens to contain only ONE country, that country cannot be a
    LOCO holdout (no remaining-country data to train on) -- it must be skipped and
    recorded, never silently given a fabricated result."""
    s1, s2, s3, truth = make_dataset(n_s1=80, seed=5, countries=("US",))
    data = tmp_path / "dataset"
    write_split(data, "train", s1, s2, s3, truth)
    monkeypatch.setattr(config, "TRAIN_FILES", {
        "s1": data / "train/train_source1.tsv", "s2": data / "train/train_source2.tsv",
        "s3": data / "train/train_source3.tsv",
        "ground_truth": data / "train/train_ground_truth.tsv"})
    out = diagnose.run_loco_comparison(sample=1.0, use_embeddings=False)
    assert out.empty
    json_files = list(diagnose.DIAG_DIR.glob("loco_comparison_*.json"))
    meta = _read_json(json_files[0])
    assert meta["skipped_directions"] == ["US"]
    assert meta["directions"] == []
    assert any("Skipped as LOCO holdouts" in s for s in meta["limitations"])


# --- 7. deterministic behaviour -------------------------------------------------------------

def test_run_is_deterministic_given_the_same_seeded_inputs(isolated, train_files):
    """Two runs on the same synthetic data/config produce identical metrics -- no
    hidden randomness beyond the seeded pipeline."""
    out1 = diagnose.run_loco_comparison(sample=1.0, use_embeddings=False)
    out2 = diagnose.run_loco_comparison(sample=1.0, use_embeddings=False)
    cols = ["configuration", "held_out_country", "macro_f05", "oof_macro_f05", "tau"]
    pd.testing.assert_frame_equal(
        out1[cols].reset_index(drop=True), out2[cols].reset_index(drop=True))


# --- CLI ----------------------------------------------------------------------------------

def test_cli_help_and_default_sample():
    """CLI --help works (parses and exits 0), and the documented default sample is
    0.0045, matching every other diagnostic."""
    with pytest.raises(SystemExit) as exc:
        diagnose.parse_args(["loco", "--help"])
    assert exc.value.code == 0

    args = diagnose.parse_args(["loco"])
    assert args.command == "loco"
    assert args.sample == 0.0045
    assert args.no_embeddings is False


def test_cli_dispatches_to_run_function(monkeypatch):
    """main() routes the subcommand to run_loco_comparison with parsed args."""
    calls = []
    monkeypatch.setattr(diagnose, "run_loco_comparison",
                        lambda sample, use_embeddings: calls.append((sample, use_embeddings)))
    diagnose.main(["loco", "--sample", "0.1", "--no-embeddings"])
    assert calls == [(0.1, False)]
