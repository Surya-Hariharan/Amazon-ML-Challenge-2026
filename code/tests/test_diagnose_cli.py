"""Smoke tests for src.diagnose (roadmap v3 Phase 0 CLI), on synthetic data only.

These exercise the orchestration (baseline/LOCO logging, convergence looping,
error-decomposition wiring, drift comparison, and the resource-qualification
stop-on-failure control flow) at toy scale, with embeddings disabled and a
fast LightGBM config -- exactly like the existing src.run_pipeline tests. The
real, expensive resource-qualification stage functions (which need a GPU and
the full dataset) are stubbed out here; only the driver that runs and stops
them is under test.
"""

import pytest

from src import config
from src import diagnose
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
    s1, s2, s3, truth = make_dataset(n_s1=200, seed=21)
    data = tmp_path / "dataset"
    write_split(data, "train", s1, s2, s3, truth)
    files = {"s1": data / "train/train_source1.tsv", "s2": data / "train/train_source2.tsv",
            "s3": data / "train/train_source3.tsv",
            "ground_truth": data / "train/train_ground_truth.tsv"}
    monkeypatch.setattr(config, "TRAIN_FILES", files)
    return s1, s2, s3, truth


def test_run_baseline_produces_a_report_with_loco(isolated, train_files):
    """E0a/E0b/E0c: baseline + LOCO produces a report with env/hash/metrics/resource fields."""
    report = diagnose.run_baseline(sample=1.0, loco=True, use_embeddings=False)
    assert report["metrics"]["valid_macro_f05"] >= 0
    assert {"loco_f05_US", "loco_f05_India"} <= set(report["metrics"])
    assert set(report["dataset_file_hashes"]) == {"s1", "s2", "s3", "ground_truth"}
    assert "python_version" in report["env"]
    assert "runtime_s" in report
    files = list(diagnose.DIAG_DIR.glob("baseline_*.json"))
    assert len(files) == 1


def test_run_convergence_holds_settings_fixed_across_sizes(isolated, train_files):
    """E-convergence runs at each requested size and reports per-size metrics,
    including the total candidate-pair count (not just the mean)."""
    out = diagnose.run_convergence([60, 120], use_embeddings=False)
    assert list(out["requested_n"]) == [60, 120]
    assert (out["actual_n_s1"] <= [60, 120]).all()
    assert out["valid_macro_f05"].notna().all()
    assert out["block_n_pairs"].notna().all() and (out["block_n_pairs"] > 0).all()
    files = list(diagnose.DIAG_DIR.glob("convergence_*.tsv"))
    assert len(files) == 1


def test_run_error_decomposition_classifies_every_true_pair(isolated, train_files):
    """E1: every true pair in the held-out validation split gets exactly one stage."""
    out = diagnose.run_error_decomposition(sample=1.0, use_embeddings=False)
    classified = out["classified"]
    valid_stages = {"representation_failure", "blocking_false_negative",
                    "matcher_false_negative", "decision_false_negative", "correct_match"}
    assert not classified.empty
    assert set(classified["stage"]) <= valid_stages
    dirs = list(diagnose.DIAG_DIR.glob("errors_*"))
    assert len(dirs) == 1
    for name in ("classified", "false_positives", "multi_match", "one_to_one_removals", "slices"):
        assert (dirs[0] / f"{name}.tsv").exists()


def test_run_drift_compares_train_and_test_without_labels(isolated, train_files, tmp_path,
                                                           monkeypatch):
    """E-drift: a distribution report compares train vs. test S1 covariates using
    only unlabelled fields, and shows a country test/train share exactly as written."""
    t1, t2, t3, _ = make_dataset(n_s1=120, seed=22, countries=("US", "India", "France"))
    data = tmp_path / "dataset"
    write_split(data, "test", t1, t2, t3)
    monkeypatch.setattr(config, "TEST_FILES", {
        "s1": data / "test/test_source1.tsv", "s2": data / "test/test_source2.tsv",
        "s3": data / "test/test_source3.tsv"})
    out = diagnose.run_drift(sample=1.0, with_candidates=False, use_embeddings=False)
    dist = out["distributions"]
    assert "name_len" in set(dist["covariate"])
    assert {"ks_stat", "ks_pvalue", "cohens_d", "effect_size_bucket"} <= set(dist.columns)
    shares = out["country_share"].set_index("country")
    assert "France" in shares.index
    assert shares.loc["France", "train_share"] == 0.0  # France never appears in train


def test_run_drift_without_candidates_never_loads_s2_or_s3(isolated, train_files, tmp_path,
                                                            monkeypatch):
    """The plain drift comparison (no --with-candidates) must load only S1 from each
    split -- load_split (which would materialise every S2/S3 record too) must not be
    called at all in that path."""
    t1, t2, t3, _ = make_dataset(n_s1=50, seed=23)
    data = tmp_path / "dataset"
    write_split(data, "test", t1, t2, t3)
    monkeypatch.setattr(config, "TEST_FILES", {
        "s1": data / "test/test_source1.tsv", "s2": data / "test/test_source2.tsv",
        "s3": data / "test/test_source3.tsv"})

    def fail_if_called(split):
        raise AssertionError(f"load_split({split!r}) must not be called without --with-candidates")

    monkeypatch.setattr(diagnose, "load_split", fail_if_called)
    out = diagnose.run_drift(sample=1.0, with_candidates=False, use_embeddings=False)
    assert not out["distributions"].empty


def test_warn_if_missing_logs_only_when_cache_absent(isolated, capsys):
    """_warn_if_missing prints a warning when the glob matches nothing, and stays
    silent when a matching artifact is present."""
    diagnose._warn_if_missing("norm_*.parquet", "embed", "normalize")
    assert "WARNING" in capsys.readouterr().out

    (config.ARTIFACTS_DIR).mkdir(parents=True, exist_ok=True)
    (config.ARTIFACTS_DIR / "norm_s1_deadbeef.parquet").write_bytes(b"")
    diagnose._warn_if_missing("norm_*.parquet", "embed", "normalize")
    assert "WARNING" not in capsys.readouterr().out


def test_run_resource_stage_stops_after_first_failure(isolated, monkeypatch):
    """E2: the qualification driver stops at the first failing stage and reports it,
    never invoking the stages after it."""
    calls: list[str] = []

    def ok_stage():
        calls.append("normalize")
        return {"name": "normalize", "success": True, "runtime_s": 0.0,
               "peak_rss_bytes": None, "peak_gpu_bytes": None}

    def failing_stage():
        calls.append("embed")
        return {"name": "embed", "success": False, "error": "simulated failure",
               "runtime_s": None, "peak_rss_bytes": None, "peak_gpu_bytes": None}

    def never_called():
        calls.append("should-not-run")
        return {"name": "unused", "success": True}

    monkeypatch.setitem(diagnose._STAGES, "normalize", ok_stage)
    monkeypatch.setitem(diagnose._STAGES, "embed", failing_stage)
    monkeypatch.setitem(diagnose._STAGES, "block", never_called)
    monkeypatch.setitem(diagnose._STAGES, "features", never_called)

    reports = diagnose.run_resource_stage("all")
    assert calls == ["normalize", "embed"]
    assert reports[-1]["success"] is False


def test_cli_parses_every_subcommand():
    """Every documented subcommand parses without error and dispatches correctly."""
    for argv in (["baseline", "--sample", "0.1", "--loco"],
                ["drift", "--with-candidates"],
                ["convergence", "--sizes", "100", "200"],
                ["resource-stage", "--stage", "embed"],
                ["errors", "--no-embeddings"]):
        args = diagnose.parse_args(argv)
        assert args.command == argv[0]
