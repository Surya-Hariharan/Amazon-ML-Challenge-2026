"""Tests for src.diagnose's P1 blocking-K ablation (blocking-only, no model/features/OOF).

These are cheap: a tiny synthetic dataset, embeddings disabled by default in most
tests (a fake deterministic encoder is used where embeddings are exercised), and no
real dataset or S3 access anywhere. Mirrors the conventions in test_diagnose_cli.py.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src import blocking, config, diagnose
from tests.synthetic import make_dataset, write_split


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    """Send artifacts/diagnostics to tmp, exactly like test_diagnose_cli.py's fixture."""
    monkeypatch.setattr(config, "ARTIFACTS_DIR", tmp_path / "artifacts")
    monkeypatch.setattr(diagnose, "DIAG_DIR", tmp_path / "artifacts" / "diagnostics")
    return tmp_path


@pytest.fixture
def train_files(tmp_path, monkeypatch):
    """Write a synthetic train split and point config.TRAIN_FILES at it."""
    s1, s2, s3, truth = make_dataset(n_s1=80, seed=11, countries=("US", "India"))
    data = tmp_path / "dataset"
    write_split(data, "train", s1, s2, s3, truth)
    files = {"s1": data / "train/train_source1.tsv", "s2": data / "train/train_source2.tsv",
            "s3": data / "train/train_source3.tsv",
            "ground_truth": data / "train/train_ground_truth.tsv"}
    monkeypatch.setattr(config, "TRAIN_FILES", files)
    return s1, s2, s3, truth


# --- 1. configuration generation / 2. baseline values / 3. one-factor-at-a-time -----------

def test_blocking_ablation_baseline_reads_live_config_values():
    """Baseline dict mirrors config.K_* exactly, so a future default change is picked
    up automatically instead of drifting out of sync with a hard-coded literal."""
    baseline = diagnose.blocking_ablation_baseline()
    assert baseline == {
        "tfidf": config.K_TFIDF_NAME, "embed": config.K_EMBEDDING,
        "rare": config.K_RARE_TOKEN, "digit": config.K_POSTAL_TOKEN,
        "address": config.K_ADDRESS,
    }


def test_blocking_ablation_configs_total_16_and_one_factor_at_a_time():
    """1 baseline + 5 passes x 3 sweep values = 16 configurations; every non-baseline
    config differs from the baseline in exactly one pass's k."""
    baseline = diagnose.blocking_ablation_baseline()
    configs = diagnose.blocking_ablation_configs()
    assert len(configs) == 16
    assert configs[0]["name"] == "baseline"
    assert {k: v for k, v in configs[0].items() if k != "name"} == baseline

    names = [c["name"] for c in configs]
    assert len(names) == len(set(names))  # every configuration name is unique

    for cfg in configs[1:]:
        k_values = {k: v for k, v in cfg.items() if k != "name"}
        changed = [p for p in baseline if k_values[p] != baseline[p]]
        assert len(changed) <= 1, cfg  # OFAT: at most one factor differs from baseline


def test_blocking_ablation_configs_cover_exact_requested_sweep_values():
    """Every requested sweep value for every pass appears in some configuration."""
    configs = diagnose.blocking_ablation_configs()
    expected = {
        "tfidf": {10, 20, 40}, "embed": {10, 20, 40}, "rare": {5, 10, 20},
        "digit": {5, 10, 20}, "address": {5, 10, 20},
    }
    for factor, values in expected.items():
        seen = {cfg[factor] for cfg in configs}
        assert values <= seen, factor


def test_blocking_ablation_configs_production_config_not_mutated():
    """Building the configuration matrix never writes back into config.py."""
    before = diagnose.blocking_ablation_baseline()
    diagnose.blocking_ablation_configs()
    assert diagnose.blocking_ablation_baseline() == before


# --- 4. explicit K values are passed into the blocking passes -------------------------------

def test_blocking_ablation_run_passes_explicit_k_into_generate_candidates(
    isolated, train_files, monkeypatch
):
    """run_blocking_ablation calls generate_candidates once per configuration, with
    that configuration's k_overrides -- not the production config.K_* defaults."""
    seen_overrides = []
    real = blocking.generate_candidates

    def spy(*args, **kwargs):
        seen_overrides.append(kwargs.get("k_overrides"))
        return real(*args, **kwargs)

    monkeypatch.setattr(diagnose, "generate_candidates", spy)
    out = diagnose.run_blocking_ablation(sample=1.0, use_embeddings=False)

    configs = diagnose.blocking_ablation_configs()
    assert len(seen_overrides) == len(configs) == len(out)
    for cfg, override in zip(configs, seen_overrides):
        assert override == {k: v for k, v in cfg.items() if k != "name"}


def test_blocking_ablation_run_does_not_mutate_config(isolated, train_files):
    """Running the sweep end-to-end never leaves config.py's K_* defaults changed."""
    before = diagnose.blocking_ablation_baseline()
    diagnose.run_blocking_ablation(sample=1.0, use_embeddings=False)
    assert diagnose.blocking_ablation_baseline() == before


# --- 5. (explicit K passed in -- covered above) / 6. output schema -------------------------

def test_blocking_ablation_run_output_schema_and_one_row_per_config(isolated, train_files):
    """One row per configuration, with every required metric, and a TSV
    + JSON sidecar written under artifacts/diagnostics/."""
    out = diagnose.run_blocking_ablation(sample=1.0, use_embeddings=False)
    configs = diagnose.blocking_ablation_configs()

    assert len(out) == len(configs) == 16
    assert list(out["experiment"]) == [c["name"] for c in configs]
    assert not out["experiment"].duplicated().any()

    required = {
        "experiment", "k_tfidf", "k_embed", "k_rare", "k_digit", "k_address",
        "sample", "n_s1", "n_other",
        "pair_recall", "s1_full_recall", "mean_candidates", "max_candidates",
        "n_pairs", "reduction_ratio",
        "recall_tfidf", "recall_embed", "recall_rare", "recall_digit", "recall_address",
        "runtime_seconds", "peak_rss_bytes", "current_rss_bytes",
    }
    assert required <= set(out.columns)
    assert {"recall_US", "recall_India"} <= set(out.columns)  # per-country recall

    tsv_files = list(diagnose.DIAG_DIR.glob("blocking_ablation_*.tsv"))
    json_files = list(diagnose.DIAG_DIR.glob("blocking_ablation_*.json"))
    assert len(tsv_files) == 1
    assert len(json_files) == 1
    roundtrip = pd.read_csv(tsv_files[0], sep="\t")
    assert len(roundtrip) == len(out)


def test_blocking_ablation_run_never_writes_output_dir(isolated, train_files):
    """A blocking-only diagnostic must never touch output/ --
    whatever output/ already held (e.g. README.md, .gitkeep) is unchanged."""
    before = sorted(config.OUTPUT_DIR.iterdir()) if config.OUTPUT_DIR.exists() else None
    diagnose.run_blocking_ablation(sample=1.0, use_embeddings=False)
    after = sorted(config.OUTPUT_DIR.iterdir()) if config.OUTPUT_DIR.exists() else None
    assert before == after


def test_blocking_ablation_cli_help_and_default_sample():
    """CLI --help works (parses and exits 0), and the documented default sample is 0.0045."""
    with pytest.raises(SystemExit) as exc:
        diagnose.parse_args(["blocking-ablation", "--help"])
    assert exc.value.code == 0

    args = diagnose.parse_args(["blocking-ablation"])
    assert args.command == "blocking-ablation"
    assert args.sample == 0.0045
    assert args.no_embeddings is False

    args = diagnose.parse_args(["blocking-ablation", "--sample", "0.5", "--no-embeddings"])
    assert args.sample == 0.5
    assert args.no_embeddings is True


# --- 7. evaluation metrics on a tiny synthetic dataset --------------------------------------

def test_blocking_ablation_metrics_on_tiny_hand_checked_dataset(isolated, monkeypatch):
    """report_blocking_stats' numbers (reused, not reimplemented) are hand-checkable
    on a 2-S1 toy union, wired through run_blocking_ablation end to end."""
    s1, s2, s3, truth = make_dataset(n_s1=6, seed=5, countries=("US",))
    monkeypatch.setattr(diagnose, "blocking_ablation_configs",
                       lambda: [{"name": "baseline", "tfidf": 20, "embed": 20, "rare": 10,
                                "digit": 10, "address": 10}])

    def fake_load_train(sample):
        return s1, s2, s3, truth

    monkeypatch.setattr(diagnose.rp, "_load_train", fake_load_train)
    out = diagnose.run_blocking_ablation(sample=1.0, use_embeddings=False)

    assert len(out) == 1
    row = out.iloc[0]
    assert row["n_s1"] == len(s1)
    assert 0.0 <= row["pair_recall"] <= 1.0
    assert 0.0 <= row["s1_full_recall"] <= 1.0
    assert row["mean_candidates"] >= 0.0
    assert row["n_pairs"] >= 0
    assert 0.0 <= row["reduction_ratio"] <= 1.0
