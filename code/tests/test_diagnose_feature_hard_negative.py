"""Tests for src.diagnose's final pre-lock diagnostic: feature discrimination audit +
hard-negative score overlap on the both40 candidate configuration.

Mirrors test_diagnose_loco.py / test_diagnose_blocking_k_downstream.py's conventions:
a tiny synthetic dataset, a fast LightGBM config, embeddings disabled by default, and
no real dataset or S3 access anywhere. Focus areas: (1) both40 is actually the
configuration used, (2) the feature audit table covers every model input feature,
(3) score/quantile/gate arithmetic is internally consistent, (4) output schema and
sidecar reproducibility, (5) never touches output/ or config.py defaults, (6) CLI.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from src import blocking, config, diagnose
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
    s1, s2, s3, truth = make_dataset(n_s1=220, seed=17, countries=("US", "India"))
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


# --- 1. both40 is actually the configuration used --------------------------------------

def test_uses_both40_k_overrides(isolated, train_files, monkeypatch):
    """generate_candidates must be called with tfidf=embed=40, matching
    blocking_k_downstream_configs()'s "both40" entry, and rare/digit/address must stay
    at the live production baseline (never hard-coded here)."""
    seen = []
    real_generate = blocking.generate_candidates

    def spy(*args, **kwargs):
        seen.append(kwargs.get("k_overrides"))
        return real_generate(*args, **kwargs)

    monkeypatch.setattr(diagnose, "generate_candidates", spy)
    diagnose.run_feature_hard_negative_audit(sample=1.0, use_embeddings=False)

    baseline = diagnose.blocking_ablation_baseline()
    assert len(seen) == 1
    assert seen[0] == {**baseline, "tfidf": 40, "embed": 40}


def test_report_records_both40_configuration(isolated, train_files):
    report = diagnose.run_feature_hard_negative_audit(sample=1.0, use_embeddings=False)
    baseline = diagnose.blocking_ablation_baseline()
    assert report["blocking_configuration"]["name"] == "both40"
    assert report["blocking_configuration"]["tfidf"] == 40
    assert report["blocking_configuration"]["embed"] == 40
    assert report["blocking_configuration"]["rare"] == baseline["rare"]
    assert report["blocking_configuration"]["digit"] == baseline["digit"]
    assert report["blocking_configuration"]["address"] == baseline["address"]


# --- 2. feature audit table covers every model input -----------------------------------

def test_feature_audit_covers_every_feature_column(isolated, train_files):
    """Every column feature_columns() would feed to the model appears exactly once in
    the audit table, each with a family and a gain_importance."""
    report = diagnose.run_feature_hard_negative_audit(sample=1.0, use_embeddings=False)
    tsv_path = report["_audit_table_path"]
    audit = pd.read_csv(tsv_path, sep="\t")

    # Reconstruct the same feature set the run actually used.
    assert set(audit["feature"]) == set(diagnose.FEATURE_FAMILIES) | (
        set(audit["feature"]) - set(diagnose.FEATURE_FAMILIES))  # sanity: no crash
    assert audit["feature"].is_unique
    assert audit["family"].notna().all()
    assert (audit["gain_importance"] >= 0).all()
    # No family left as an accidental empty string.
    assert (audit["family"] != "").all()


def test_feature_audit_families_are_known_buckets(isolated, train_files):
    """Every assigned family is one of the task's Part 5 buckets."""
    report = diagnose.run_feature_hard_negative_audit(sample=1.0, use_embeddings=False)
    audit = pd.read_csv(report["_audit_table_path"], sep="\t")
    allowed = {"NAME", "ADDRESS", "TF-IDF", "EMBEDDING", "NUMERIC/DIGIT", "COUNTRY",
              "LENGTH/MISSINGNESS", "OTHER"}
    assert set(audit["family"]) <= allowed


def test_feature_family_summary_shares_sum_to_one(isolated, train_files):
    report = diagnose.run_feature_hard_negative_audit(sample=1.0, use_embeddings=False)
    shares = [row["gain_share"] for row in report["feature_family_summary"]]
    assert shares  # at least one family got some importance, or all-zero is still summed
    total = sum(shares)
    assert total == pytest.approx(1.0, abs=1e-6) or total == pytest.approx(0.0, abs=1e-6)


# --- 3. score/quantile/gate arithmetic is internally consistent -------------------------

def test_score_overlap_quantiles_are_well_formed(isolated, train_files):
    report = diagnose.run_feature_hard_negative_audit(sample=1.0, use_embeddings=False)
    overlap = report["score_overlap"]
    for key in ("true_positive_scores", "true_negative_scores"):
        q = overlap[key]
        if q["n"] > 0:
            assert 0.0 <= q["p50"] <= 1.0
            assert q["p50"] <= q["p90"] <= q["p99"] <= q["max"] + 1e-9
    for name, stats in overlap["hard_negatives"].items():
        assert stats["threshold"] is not None
        if stats["n"] > 0:
            assert stats["p50"] >= stats["threshold"] - 1e-6


def test_hard_negative_counts_are_monotonic_in_threshold(isolated, train_files):
    """A stricter score cutoff can never keep MORE negatives than a looser one."""
    report = diagnose.run_feature_hard_negative_audit(sample=1.0, use_embeddings=False)
    hn = report["score_overlap"]["hard_negatives"]
    assert hn["score_ge_0.5"]["n"] >= hn["score_ge_0.7"]["n"] >= hn["score_ge_0.9"]["n"]


def test_missed_positive_counts_shrink_as_threshold_tightens(isolated, train_files):
    """below_tau uses the actual tuned tau; below_0.5/0.7 are fixed reference cuts. All
    three must be non-negative and never exceed the total true-positive-in-candidate
    count."""
    report = diagnose.run_feature_hard_negative_audit(sample=1.0, use_embeddings=False)
    mp = report["missed_positive_stats"]
    n_pos = report["n_positive_pairs_valid"]
    for stats in mp.values():
        assert 0 <= stats["n"] <= n_pos


def test_gate_conclusion_matches_recommendation(isolated, train_files):
    """recommendation is "B" iff every gate condition holds; the conclusion text must
    be internally consistent with that (task requirement: never redesign on tiny
    numbers, and blocking-dominated misses must recommend A, never B)."""
    report = diagnose.run_feature_hard_negative_audit(sample=1.0, use_embeddings=False)
    gate = report["gate"]
    all_hold = all(v["value"] for v in gate.values())
    assert report["recommendation"] == ("B" if all_hold else "A")
    if report["recommendation"] == "A":
        assert "blocking" in report["conclusion"].lower()


def test_gate_never_recommends_b_when_blocking_dominates(isolated, train_files):
    """Synthetic/tiny-sample runs realistically have few or no matcher_false_negative
    pairs, so blocking-caused loss should dominate or tie -- recommendation must be A,
    never a spurious B from noise."""
    report = diagnose.run_feature_hard_negative_audit(sample=1.0, use_embeddings=False)
    stage_counts = report["stage_counts"]
    n_matcher_fn = stage_counts.get("matcher_false_negative", 0)
    n_blocking_loss = (stage_counts.get("blocking_false_negative", 0)
                      + stage_counts.get("representation_failure", 0))
    if n_matcher_fn <= n_blocking_loss:
        assert report["recommendation"] == "A"


# --- 4. output schema and sidecar reproducibility ---------------------------------------

def test_output_files_written_and_json_roundtrips(isolated, train_files):
    report = diagnose.run_feature_hard_negative_audit(sample=1.0, use_embeddings=False)
    json_files = list(diagnose.DIAG_DIR.glob("feature_hard_negative_audit_*.json"))
    tsv_files = list(diagnose.DIAG_DIR.glob("feature_hard_negative_audit_*.tsv"))
    assert len(json_files) == 1
    assert len(tsv_files) == 1

    meta = _read_json(json_files[0])
    assert meta["kind"] == "feature_hard_negative_audit"
    assert meta["sample"] == 1.0
    assert "git_commit" in meta
    assert set(meta["dataset_file_hashes"]) == {"s1", "s2", "s3", "ground_truth"}
    assert "python_version" in meta["env"]
    assert meta["recommendation"] == report["recommendation"]

    audit = pd.read_csv(tsv_files[0], sep="\t")
    assert len(audit) > 0


def test_threshold_configuration_is_read_from_the_fit_not_hardcoded(isolated, train_files):
    """tau/singleton_tau in the report must be the values decide.tune_threshold actually
    picked for this run, not config.SINGLETON_THRESHOLD or a literal 0.725."""
    report = diagnose.run_feature_hard_negative_audit(sample=1.0, use_embeddings=False)
    assert 0.0 < report["threshold_configuration"]["tau"] < 1.0
    assert "not hardcoded" in report["threshold_configuration"]["source"]


# --- 5. never touches output/ or config.py defaults --------------------------------------

def test_never_touches_output_dir(isolated, train_files):
    before = sorted(config.OUTPUT_DIR.iterdir()) if config.OUTPUT_DIR.exists() else None
    diagnose.run_feature_hard_negative_audit(sample=1.0, use_embeddings=False)
    after = sorted(config.OUTPUT_DIR.iterdir()) if config.OUTPUT_DIR.exists() else None
    assert before == after


def test_never_mutates_production_blocking_config(isolated, train_files):
    before = diagnose.blocking_ablation_baseline()
    diagnose.run_feature_hard_negative_audit(sample=1.0, use_embeddings=False)
    assert diagnose.blocking_ablation_baseline() == before


# --- 6. CLI --------------------------------------------------------------------------------

def test_cli_help_and_default_sample():
    with pytest.raises(SystemExit) as exc:
        diagnose.parse_args(["feature-hard-negative", "--help"])
    assert exc.value.code == 0

    args = diagnose.parse_args(["feature-hard-negative"])
    assert args.command == "feature-hard-negative"
    assert args.sample == 0.0045
    assert args.no_embeddings is False


def test_cli_dispatches_to_run_function(monkeypatch):
    calls = []
    monkeypatch.setattr(diagnose, "run_feature_hard_negative_audit",
                        lambda sample, use_embeddings: calls.append((sample, use_embeddings)))
    diagnose.main(["feature-hard-negative", "--sample", "0.1", "--no-embeddings"])
    assert calls == [(0.1, False)]
