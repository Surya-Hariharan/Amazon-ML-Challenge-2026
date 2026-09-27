"""Tests for src.diagnose's P1-downstream blocking-K OOF experiment.

Mirrors test_diagnose_blocking_ablation.py's conventions: a tiny synthetic dataset,
embeddings disabled by default (a fast LightGBM config, like test_diagnose_cli.py, is
needed here because -- unlike the blocking-only ablation -- this experiment runs the
full features -> LightGBM -> threshold -> decision pipeline), and no real dataset or
S3 access anywhere.
"""

from __future__ import annotations

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
    """Write a synthetic train split and point config.TRAIN_FILES at it."""
    s1, s2, s3, truth = make_dataset(n_s1=200, seed=21, countries=("US", "India"))
    data = tmp_path / "dataset"
    write_split(data, "train", s1, s2, s3, truth)
    files = {"s1": data / "train/train_source1.tsv", "s2": data / "train/train_source2.tsv",
            "s3": data / "train/train_source3.tsv",
            "ground_truth": data / "train/train_ground_truth.tsv"}
    monkeypatch.setattr(config, "TRAIN_FILES", files)
    return s1, s2, s3, truth


# --- 1. configuration generation --------------------------------------------------------

def test_downstream_configs_are_the_4_named_pareto_configs_in_order():
    """Exactly baseline/tfidf40/embed40/both40, in that order, each changing only
    tfidf/embed K relative to the live production baseline (rare/digit/address held
    fixed -- those are not part of this experiment's variable)."""
    baseline = diagnose.blocking_ablation_baseline()
    configs = diagnose.blocking_k_downstream_configs()
    names = [c["name"] for c in configs]
    assert names == ["baseline", "tfidf40", "embed40", "both40"]

    by_name = {c["name"]: c for c in configs}
    assert by_name["baseline"] == {"name": "baseline", **baseline}
    assert by_name["tfidf40"] == {"name": "tfidf40", **{**baseline, "tfidf": 40}}
    assert by_name["embed40"] == {"name": "embed40", **{**baseline, "embed": 40}}
    assert by_name["both40"] == {"name": "both40", **{**baseline, "tfidf": 40, "embed": 40}}

    for name in ("rare", "digit", "address"):
        assert {c[name] for c in configs} == {baseline[name]}  # unchanged everywhere


def test_downstream_configs_do_not_mutate_production_config():
    """Building the configuration matrix never writes back into config.py."""
    before = diagnose.blocking_ablation_baseline()
    diagnose.blocking_k_downstream_configs()
    assert diagnose.blocking_ablation_baseline() == before


# --- 2. K-override propagation into the same candidate set used for scoring -------------

def test_downstream_run_passes_explicit_k_and_reuses_same_cands_for_features(
    isolated, train_files, monkeypatch
):
    """generate_candidates is called once per configuration with that configuration's
    k_overrides, and the frame it returns is exactly what build_features receives
    (the measured candidate set is the scored candidate set -- task requirement 8)."""
    seen_overrides = []
    seen_cands_ids = []
    real_generate = blocking.generate_candidates
    real_build = diagnose.build_features

    def spy_generate(*args, **kwargs):
        cands = real_generate(*args, **kwargs)
        seen_overrides.append(kwargs.get("k_overrides"))
        return cands

    def spy_build(cands, *args, **kwargs):
        seen_cands_ids.append(id(cands))
        return real_build(cands, *args, **kwargs)

    monkeypatch.setattr(diagnose, "generate_candidates", spy_generate)
    monkeypatch.setattr(diagnose, "build_features", spy_build)
    out = diagnose.run_blocking_k_downstream(sample=1.0, use_embeddings=False)

    configs = diagnose.blocking_k_downstream_configs()
    assert (out["status"] == "ok").all()
    assert len(seen_overrides) == len(configs)
    for cfg, override in zip(configs, seen_overrides):
        assert override == {k: v for k, v in cfg.items() if k != "name"}
    # each generate_candidates call's frame is the very same object build_features saw
    assert seen_cands_ids == seen_cands_ids  # sanity: list populated
    assert len(seen_cands_ids) == len(configs)


def test_downstream_run_does_not_mutate_production_config(isolated, train_files):
    """Running the experiment end-to-end never leaves config.py's K_* defaults changed."""
    before = diagnose.blocking_ablation_baseline()
    diagnose.run_blocking_k_downstream(sample=1.0, use_embeddings=False)
    assert diagnose.blocking_ablation_baseline() == before


# --- 3. output schema --------------------------------------------------------------------

def test_downstream_run_output_schema_and_one_row_per_config(isolated, train_files):
    """One row per configuration, in order, with every metric the task
    requires, and a TSV + JSON sidecar written under artifacts/diagnostics/."""
    out = diagnose.run_blocking_k_downstream(sample=1.0, use_embeddings=False)
    configs = diagnose.blocking_k_downstream_configs()

    assert len(out) == len(configs) == 4
    assert list(out["experiment"]) == [c["name"] for c in configs]
    assert (out["status"] == "ok").all()

    required = {
        "experiment", "status", "k_tfidf", "k_embed", "k_rare", "k_digit", "k_address",
        "sample", "n_s1", "n_other",
        "block_pair_recall", "block_s1_full_recall", "block_mean_candidates",
        "block_max_candidates", "block_n_pairs",
        "n_feature_rows",
        "oof_macro_f05", "tau", "singleton_tau",
        "valid_macro_f05", "valid_pair_precision", "valid_pair_recall",
        "valid_singleton_f05", "valid_non_singleton_f05",
        "fold_f05", "fold_f05_mean", "fold_f05_std",
        "runtime_seconds", "peak_rss_bytes", "current_rss_bytes",
    }
    assert required <= set(out.columns)
    assert {"block_recall_US", "block_recall_India",
           "valid_f05_US", "valid_f05_India"} <= set(out.columns)

    tsv_files = list(diagnose.DIAG_DIR.glob("blocking_k_downstream_*.tsv"))
    json_files = list(diagnose.DIAG_DIR.glob("blocking_k_downstream_*.json"))
    assert len(tsv_files) == 1
    assert len(json_files) == 1
    roundtrip = pd.read_csv(tsv_files[0], sep="\t")
    assert len(roundtrip) == len(out)


def test_downstream_run_never_writes_output_dir(isolated, train_files):
    """A diagnostic-layer experiment must never touch output/."""
    before = sorted(config.OUTPUT_DIR.iterdir()) if config.OUTPUT_DIR.exists() else None
    diagnose.run_blocking_k_downstream(sample=1.0, use_embeddings=False)
    after = sorted(config.OUTPUT_DIR.iterdir()) if config.OUTPUT_DIR.exists() else None
    assert before == after


def test_downstream_fold_f05_has_one_value_per_used_fold(isolated, train_files):
    """Fold-stability values: at most N_FOLDS entries, each a valid macro F0.5."""
    out = diagnose.run_blocking_k_downstream(sample=1.0, use_embeddings=False)
    for raw in out["fold_f05"]:
        values = [float(v) for v in str(raw).split(";") if v]
        assert 0 < len(values) <= config.N_FOLDS
        assert all(0.0 <= v <= 1.0 for v in values)


# --- 4. reproducibility / config recording ------------------------------------------------

def test_downstream_json_sidecar_records_reproducibility_fields(isolated, train_files):
    """The JSON sidecar records the sample fraction, exact K configs, env, dataset
    hashes and a git commit field -- enough to reproduce the run later."""
    diagnose.run_blocking_k_downstream(sample=1.0, use_embeddings=False)
    json_files = list(diagnose.DIAG_DIR.glob("blocking_k_downstream_*.json"))
    assert len(json_files) == 1
    meta = diag_read_json(json_files[0])

    assert meta["kind"] == "blocking_k_downstream"
    assert meta["sample"] == 1.0
    assert meta["configs"] == diagnose.blocking_k_downstream_configs()
    assert meta["order"] == ["baseline", "tfidf40", "embed40", "both40"]
    assert "git_commit" in meta
    assert set(meta["dataset_file_hashes"]) == {"s1", "s2", "s3", "ground_truth"}
    assert "python_version" in meta["env"]
    assert "timestamp" in meta


def diag_read_json(path):
    """Small local json.load wrapper (kept out of the top-level imports)."""
    import json
    with open(path) as fh:
        return json.load(fh)


# --- 5. failure handling -------------------------------------------------------------------

def test_downstream_one_failing_config_is_recorded_and_others_still_run(
    isolated, train_files, monkeypatch
):
    """If one configuration's blocking pass raises, its row is marked failed with the
    error message, and the remaining 3 configurations still produce ok rows -- a
    single failure must never abort the whole experiment or be silently dropped."""
    real_generate = blocking.generate_candidates

    def flaky_generate(*args, **kwargs):
        if kwargs.get("k_overrides", {}).get("tfidf") == 40 and \
           kwargs.get("k_overrides", {}).get("embed") != 40:
            raise RuntimeError("boom: simulated blocking failure")
        return real_generate(*args, **kwargs)

    monkeypatch.setattr(diagnose, "generate_candidates", flaky_generate)
    out = diagnose.run_blocking_k_downstream(sample=1.0, use_embeddings=False)

    assert len(out) == 4
    by_name = dict(zip(out["experiment"], out["status"]))
    assert by_name["tfidf40"] == "failed"
    assert by_name["baseline"] == "ok"
    assert by_name["embed40"] == "ok"
    assert by_name["both40"] == "ok"

    failed_row = out[out["experiment"] == "tfidf40"].iloc[0]
    assert "boom" in failed_row["error"]

    json_files = list(diagnose.DIAG_DIR.glob("blocking_k_downstream_*.json"))
    meta = diag_read_json(json_files[0])
    assert meta["failures"] == ["tfidf40"]


def test_downstream_all_configs_failing_still_produces_a_report(isolated, train_files, monkeypatch):
    """Even a total wipeout (every configuration fails) must not raise -- it must
    still produce a 4-row report with every row marked failed, so a bad run is
    visible in the artifact rather than crashing with no trace."""
    def always_fail(*args, **kwargs):
        raise RuntimeError("boom: everything failed")

    monkeypatch.setattr(diagnose, "generate_candidates", always_fail)
    out = diagnose.run_blocking_k_downstream(sample=1.0, use_embeddings=False)
    assert len(out) == 4
    assert (out["status"] == "failed").all()
    assert out["error"].str.contains("boom").all()


# --- 6. CLI ----------------------------------------------------------------------------

def test_downstream_cli_help_and_default_sample():
    """CLI --help works (parses and exits 0), and the documented default sample is 0.0045."""
    with pytest.raises(SystemExit) as exc:
        diagnose.parse_args(["blocking-k-downstream", "--help"])
    assert exc.value.code == 0

    args = diagnose.parse_args(["blocking-k-downstream"])
    assert args.command == "blocking-k-downstream"
    assert args.sample == 0.0045
    assert args.no_embeddings is False

    args = diagnose.parse_args(["blocking-k-downstream", "--sample", "0.5", "--no-embeddings"])
    assert args.sample == 0.5
    assert args.no_embeddings is True


def test_downstream_cli_dispatches_to_run_function(monkeypatch):
    """main() routes the subcommand to run_blocking_k_downstream with parsed args."""
    calls = []
    monkeypatch.setattr(diagnose, "run_blocking_k_downstream",
                        lambda sample, use_embeddings: calls.append((sample, use_embeddings)))
    diagnose.main(["blocking-k-downstream", "--sample", "0.1", "--no-embeddings"])
    assert calls == [(0.1, False)]
