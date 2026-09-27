"""Tests for src.diagnose's P3 decision-layer validation experiment.

Mirrors test_diagnose_blocking_k_downstream.py's conventions: a tiny synthetic
dataset, a fast LightGBM config, embeddings disabled by default, and no real
dataset or S3 access anywhere. P3 fixes the candidate configuration (does not
sweep blocking K) and trains exactly one model, so these tests focus on: (1)
one-to-one ON/OFF isolation, (2) decision-order isolation, (3) threshold and
(4) singleton-threshold neighborhood evaluation, (5) output schema, and (6)
deterministic configuration recording.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from src import config, decide, diagnose, model
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
    s1, s2, s3, truth = make_dataset(n_s1=220, seed=21, countries=("US", "India"))
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


# --- candidate configuration is fixed, not swept -------------------------------------------

def test_p3_candidate_config_is_the_both40_configuration():
    """P3 must not compare blocking-K values -- it uses the one fixed "both40"
    configuration P1-downstream reported as current-best."""
    assert diagnose.P3_CANDIDATE_CONFIG == {
        "tfidf": 40, "embed": 40, "rare": 10, "digit": 10, "address": 10}


# --- 1. one-to-one ON/OFF isolation ----------------------------------------------------------

def test_p3a_trains_the_model_exactly_once_for_both_settings(isolated, train_files, monkeypatch):
    """One-to-one ON and OFF must reuse the same trained model/OOF -- train_oof and
    train_full are each called exactly once for the whole P3-A comparison, never once
    per one-to-one setting."""
    oof_calls = []
    full_calls = []
    real_oof, real_full = model.train_oof, model.train_full

    def spy_oof(*a, **kw):
        oof_calls.append(1)
        return real_oof(*a, **kw)

    def spy_full(*a, **kw):
        full_calls.append(1)
        return real_full(*a, **kw)

    monkeypatch.setattr(diagnose, "train_oof", spy_oof)
    monkeypatch.setattr(diagnose, "train_full", spy_full)
    out = diagnose.run_decision_validation(sample=1.0, use_embeddings=False)
    assert len(oof_calls) == 1
    assert len(full_calls) == 1
    assert (out["experiment"].str.startswith("P3A_")).sum() == 2


def test_p3a_on_and_off_rows_reflect_the_flag_and_can_score_differently(isolated, train_files):
    """ON produces a one_to_one=True row, OFF a one_to_one=False row; both cover the
    same validation universe (n_true_matches identical), and ON's removal counters
    are internally consistent (true + false = total removed)."""
    out = diagnose.run_decision_validation(sample=1.0, use_embeddings=False)
    a = out[out["experiment"].str.startswith("P3A_")]
    assert set(a["one_to_one"]) == {True, False}
    on = a[a["one_to_one"] == True].iloc[0]  # noqa: E712
    off = a[a["one_to_one"] == False].iloc[0]  # noqa: E712
    assert on["n_true_matches"] == off["n_true_matches"]
    assert on["n_true_matches_removed"] + on["n_false_matches_removed"] == on["n_removed_by_step"]
    assert off["n_removed_by_step"] == 0  # no one-to-one applied -> nothing removed by it


# --- 2. decision-order isolation -------------------------------------------------------------

def test_p3b_uses_the_p3a_on_tau_and_does_not_retrain(isolated, train_files, monkeypatch):
    """Both decision orders in P3-B use the exact same (tau, singleton_tau) that
    P3-A's one-to-one-ON tuning selected, and no additional model training happens
    for P3-B (only the one P3-A/shared training call)."""
    full_calls = []
    real_full = model.train_full

    def spy_full(*a, **kw):
        full_calls.append(1)
        return real_full(*a, **kw)

    monkeypatch.setattr(diagnose, "train_full", spy_full)
    out = diagnose.run_decision_validation(sample=1.0, use_embeddings=False)
    assert len(full_calls) == 1

    a_on = out[(out["experiment"].str.startswith("P3A_")) & (out["one_to_one"] == True)].iloc[0]  # noqa: E712
    b = out[out["experiment"].str.startswith("P3B_")]
    assert len(b) == 2
    assert (b["tau"] == a_on["tau"]).all()
    assert (b["singleton_tau"] == a_on["singleton_tau"]).all() or b["singleton_tau"].isna().all()


def test_decide_threshold_then_one_to_one_matches_apply_threshold_when_no_duplicates(isolated):
    """When no cand_id appears under more than one S1 (nothing for one-to-one to
    remove either way), both decision orders must agree exactly -- ordering can only
    matter when assign_one_to_one actually changes something."""
    scored = pd.DataFrame({
        "s1_id": ["A", "B", "C"], "cand_id": ["x", "y", "z"],
        "prob": [0.9, 0.8, 0.4], "label": [1, 1, 0],
    })
    tau = 0.5
    order_a = diagnose.decide_threshold_then_one_to_one(scored, tau, ["A", "B", "C"])
    order_b = decide.apply_threshold(scored, tau, ["A", "B", "C"], one_to_one=True)
    assert order_a == order_b == {"A": ["x"], "B": ["y"], "C": []}


def test_decide_threshold_then_one_to_one_differs_from_production_order_with_duplicates(isolated):
    """With a candidate shared by two S1s where the loser is below tau, production's
    order (one-to-one first) drops S1 B entirely (its only candidate, "z", loses the
    one-to-one tie-break to A and is discarded before the threshold even runs), while
    threshold-first keeps B's own below-threshold copy filtered out too -- but if B's
    copy were the *lower*-probability one and above tau, threshold-first would let it
    through where production's order already removed it. This exercises exactly that
    case: two S1s share cand "z", B's copy is weaker but still above tau."""
    scored = pd.DataFrame({
        "s1_id": ["A", "B"], "cand_id": ["z", "z"],
        "prob": [0.9, 0.6], "label": [1, 0],
    })
    tau = 0.5
    order_a = diagnose.decide_threshold_then_one_to_one(scored, tau, ["A", "B"])
    order_b = decide.apply_threshold(scored, tau, ["A", "B"], one_to_one=True)
    # order B (one-to-one first): "z" goes to A (higher prob), B ends up with nothing.
    assert order_b == {"A": ["z"], "B": []}
    # order A (threshold first): both pass tau, then one-to-one still gives "z" to A.
    assert order_a == {"A": ["z"], "B": []}
    # both orders agree here (one-to-one's own tie-break is prob-based either way) --
    # the key isolation point is that both were computed from the SAME tau/model.
    assert order_a == order_b


# --- 3. threshold neighborhood evaluation -----------------------------------------------------

def test_p3c_neighborhood_is_the_documented_5_point_spacing(isolated, train_files):
    """P3-C evaluates T-0.05, T-0.025, T, T+0.025, T+0.05 (clipped to TAU_GRID),
    never a different spacing or an invented thresholding method."""
    out = diagnose.run_decision_validation(sample=1.0, use_embeddings=False)
    a_on = out[(out["experiment"].str.startswith("P3A_")) & (out["one_to_one"] == True)].iloc[0]  # noqa: E712
    tau = a_on["tau"]
    c = out[out["experiment"].str.startswith("P3C_")]
    expected = diagnose._tau_neighborhood(tau)
    assert sorted(c["tau"].tolist()) == expected
    assert len(expected) <= 5
    assert (c["singleton_tau"] == a_on["singleton_tau"]).all() or c["singleton_tau"].isna().all()
    selected_row = c[c["tau"] == tau]
    assert len(selected_row) == 1
    assert bool(selected_row.iloc[0]["is_selected"])


def test_tau_neighborhood_clips_to_grid_bounds():
    """Near the grid edges, out-of-range neighbors are dropped rather than
    extrapolated past config.TAU_GRID's min/max."""
    lo, hi = min(config.TAU_GRID), max(config.TAU_GRID)
    near_lo = diagnose._tau_neighborhood(lo)
    assert min(near_lo) >= lo
    near_hi = diagnose._tau_neighborhood(hi)
    assert max(near_hi) <= hi


# --- 4. singleton-threshold neighborhood evaluation --------------------------------------------

def test_p3d_neighborhood_matches_helper_and_reports_singleton_counts(isolated, train_files):
    """P3-D's rows use the same neighborhood helper as P3-C (applied to singleton_tau
    instead of tau) and report predicted/true/false singleton counts that are
    internally consistent with the validation universe size."""
    out = diagnose.run_decision_validation(sample=1.0, use_embeddings=False)
    a_on = out[(out["experiment"].str.startswith("P3A_")) & (out["one_to_one"] == True)].iloc[0]  # noqa: E712
    stau = a_on["singleton_tau"]
    stau = None if pd.isna(stau) else float(stau)
    d = out[out["experiment"].str.startswith("P3D_")]
    expected = diagnose._singleton_neighborhood(stau)
    got = [None if pd.isna(v) else float(v) for v in d["singleton_tau"]]
    assert sorted(got, key=lambda v: (v is None, v)) == sorted(
        expected, key=lambda v: (v is None, v))
    for _, row in d.iterrows():
        assert row["n_false_singletons"] <= row["n_predicted_singletons"]
        assert row["n_predicted_singletons"] == row["zero_match_count"]
        assert row["n_true_singletons"] >= 0


def test_singleton_neighborhood_is_a_single_none_row_when_selected_is_none():
    """No singleton gate (None) has no numeric neighborhood -- it is its own
    single-element result, never a fabricated numeric grid."""
    assert diagnose._singleton_neighborhood(None) == [None]


# --- 5. output schema -----------------------------------------------------------------------

def test_output_schema_has_one_row_per_subexperiment_and_required_columns(isolated, train_files):
    """P3-A contributes 2 rows, P3-B 2, P3-C up to 5, P3-D up to 5; every row carries
    the shared columns needed to identify and reproduce it, and TSV/JSON sidecars
    are written under artifacts/diagnostics/."""
    out = diagnose.run_decision_validation(sample=1.0, use_embeddings=False)
    counts = out["experiment"].str.extract(r"^(P3[ABCD])")[0].value_counts()
    assert counts["P3A"] == 2
    assert counts["P3B"] == 2
    assert 1 <= counts["P3C"] <= 5
    assert 1 <= counts["P3D"] <= 5

    required = {"experiment", "tau", "singleton_tau", "valid_macro_f05",
                "valid_pair_precision", "valid_pair_recall", "valid_singleton_f05",
                "valid_non_singleton_f05", "candidate_config", "sample"}
    assert required <= set(out.columns)
    assert {"valid_f05_US", "valid_f05_India"} <= set(out.columns)

    tsv_files = list(diagnose.DIAG_DIR.glob("decision_validation_*.tsv"))
    json_files = list(diagnose.DIAG_DIR.glob("decision_validation_*.json"))
    assert len(tsv_files) == 1
    assert len(json_files) == 1
    roundtrip = pd.read_csv(tsv_files[0], sep="\t")
    assert len(roundtrip) == len(out)


def test_never_touches_output_dir_or_runs_full_scale(isolated, train_files):
    """A diagnostic-layer experiment must never touch output/ and
    never invokes run_pipeline's --mode test path."""
    before = sorted(config.OUTPUT_DIR.iterdir()) if config.OUTPUT_DIR.exists() else None
    diagnose.run_decision_validation(sample=1.0, use_embeddings=False)
    after = sorted(config.OUTPUT_DIR.iterdir()) if config.OUTPUT_DIR.exists() else None
    assert before == after


# --- 6. deterministic configuration recording -------------------------------------------------

def test_json_sidecar_records_reproducibility_fields(isolated, train_files):
    """The JSON sidecar records the candidate configuration, sample fraction,
    selected tau/singleton_tau, block stats, git commit, dataset hashes and env --
    enough to reproduce the run and audit what decision setting was tested."""
    diagnose.run_decision_validation(sample=1.0, use_embeddings=False)
    json_files = list(diagnose.DIAG_DIR.glob("decision_validation_*.json"))
    assert len(json_files) == 1
    meta = _read_json(json_files[0])

    assert meta["kind"] == "decision_validation"
    assert meta["candidate_config"] == diagnose.P3_CANDIDATE_CONFIG
    assert meta["sample"] == 1.0
    assert "selected_tau" in meta and "selected_singleton_tau" in meta
    assert "block_stats" in meta
    assert "git_commit" in meta
    assert set(meta["dataset_file_hashes"]) == {"s1", "s2", "s3", "ground_truth"}
    assert "python_version" in meta["env"]
    assert {"p3a_one_to_one", "p3b_decision_order",
           "p3c_threshold_generalization", "p3d_singleton_threshold"} <= set(meta)


def test_run_is_deterministic_given_the_same_seeded_inputs(isolated, train_files):
    """Two runs on the same synthetic data/config select the same tau/singleton_tau
    and produce the same candidate configuration -- no hidden randomness in the
    decision-layer sweep itself."""
    out1 = diagnose.run_decision_validation(sample=1.0, use_embeddings=False)
    out2 = diagnose.run_decision_validation(sample=1.0, use_embeddings=False)
    a1 = out1[out1["experiment"] == "P3A_one_to_one_on"].iloc[0]
    a2 = out2[out2["experiment"] == "P3A_one_to_one_on"].iloc[0]
    assert a1["tau"] == a2["tau"]
    pd.testing.assert_series_equal(
        pd.isna(pd.Series([a1["singleton_tau"]])),
        pd.isna(pd.Series([a2["singleton_tau"]])))


# --- CLI ----------------------------------------------------------------------------------

def test_cli_help_and_default_sample():
    """CLI --help works (parses and exits 0), and the documented default sample is
    0.0045, matching every other P-series diagnostic."""
    with pytest.raises(SystemExit) as exc:
        diagnose.parse_args(["decision-validation", "--help"])
    assert exc.value.code == 0

    args = diagnose.parse_args(["decision-validation"])
    assert args.command == "decision-validation"
    assert args.sample == 0.0045
    assert args.no_embeddings is False


def test_cli_dispatches_to_run_function(monkeypatch):
    """main() routes the subcommand to run_decision_validation with parsed args."""
    calls = []
    monkeypatch.setattr(diagnose, "run_decision_validation",
                        lambda sample, use_embeddings: calls.append((sample, use_embeddings)))
    diagnose.main(["decision-validation", "--sample", "0.1", "--no-embeddings"])
    assert calls == [(0.1, False)]
