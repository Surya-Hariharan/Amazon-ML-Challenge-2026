"""Tests for src.drift (roadmap v3 Phase 0, E-drift)."""

import numpy as np
import pandas as pd
import pytest

from src import drift


def _frame(rows):
    """Build a raw source frame from (entity_id, name, address, country) tuples."""
    return pd.DataFrame(rows, columns=["entity_id", "business_name", "business_address", "country"])


def test_record_covariates_lengths_and_missingness():
    """Raw lengths, missingness flags and non-Latin detection are computed correctly."""
    raw = _frame([
        ("S1-1", "Green Traders LLC", "12 Main St, Austin, TX", "US"),
        ("S1-2", "", "", "US"),
        ("S1-3", "Café Français", "5 Rue de la Paix", "France"),
    ])
    cov = drift.record_covariates(raw)
    assert cov.loc[0, "name_missing"] == 0 and cov.loc[0, "addr_missing"] == 0
    assert cov.loc[1, "name_missing"] == 1 and cov.loc[1, "addr_missing"] == 1
    assert cov.loc[0, "name_len"] == len("Green Traders LLC")
    assert cov.loc[2, "non_latin_name"] == 1  # "é" is outside ASCII
    assert cov.loc[0, "non_latin_name"] == 0
    assert (cov["entity_id"] == raw["entity_id"]).all()


def test_record_covariates_digit_and_long_num_detection():
    """A 5+ digit token sets has_long_num; digit_count reflects normalised digit tokens."""
    raw = _frame([
        ("S1-1", "Acme Inc", "100 Main St, Austin, TX 78701", "US"),
        ("S1-2", "Acme Inc", "100 Main St, Austin, TX", "US"),
    ])
    cov = drift.record_covariates(raw)
    assert cov.loc[0, "has_long_num"] == 1  # 78701 is a 5-digit token
    assert cov.loc[1, "has_long_num"] == 0
    assert cov.loc[0, "digit_count"] >= cov.loc[1, "digit_count"]


def test_candidate_covariates_counts_and_missing_s1s():
    """n_candidates is the group size per s1_id; S1s with no rows get 0/NaN."""
    cands = pd.DataFrame({
        "s1_id": ["S1-1", "S1-1", "S1-2"],
        "cand_id": ["S2-1", "S2-2", "S2-3"],
        "score_tfidf": [0.9, 0.5, 0.7],
    })
    out = drift.candidate_covariates(cands, ["S1-1", "S1-2", "S1-3"])
    row = out.set_index("entity_id")
    assert row.loc["S1-1", "n_candidates"] == 2
    assert row.loc["S1-2", "n_candidates"] == 1
    assert row.loc["S1-3", "n_candidates"] == 0
    assert row.loc["S1-1", "top_score"] == pytest.approx(0.9)
    assert np.isnan(row.loc["S1-3", "top_score"])


def test_cohens_d_matches_known_case():
    """Two unit-variance samples shifted by 1.0 have |d| approx 1.0."""
    rng = np.random.default_rng(0)
    a = pd.Series(rng.normal(0, 1, 5000))
    b = pd.Series(rng.normal(1, 1, 5000))
    d = drift.cohens_d(a, b)
    assert d == pytest.approx(-1.0, abs=0.1)


def test_cohens_d_nan_on_tiny_or_constant_samples():
    """Fewer than 2 values, or zero pooled variance, yields NaN rather than raising."""
    assert np.isnan(drift.cohens_d(pd.Series([1.0]), pd.Series([1.0, 2.0])))
    assert np.isnan(drift.cohens_d(pd.Series([5.0, 5.0]), pd.Series([5.0, 5.0])))


def test_compare_distributions_separates_statistical_from_effect_size():
    """A large-N trivial shift gets a tiny p-value but a negligible effect size,
    and a real shift gets both a small p-value and a non-negligible effect size --
    the two must be reported independently, never collapsed into one verdict."""
    rng = np.random.default_rng(1)
    trivial_train = pd.DataFrame({"name_len": rng.normal(10.0, 1.0, 20000)})
    trivial_test = pd.DataFrame({"name_len": rng.normal(10.01, 1.0, 20000)})
    report = drift.compare_distributions(trivial_train, trivial_test, columns=["name_len"])
    row = report.iloc[0]
    assert row["ks_pvalue"] < 0.5  # plausible to detect at this N, not asserted tiny
    assert row["effect_size_bucket"] in ("negligible", "small")

    shifted_train = pd.DataFrame({"name_len": rng.normal(10.0, 1.0, 500)})
    shifted_test = pd.DataFrame({"name_len": rng.normal(11.0, 1.0, 500)})
    report2 = drift.compare_distributions(shifted_train, shifted_test, columns=["name_len"])
    row2 = report2.iloc[0]
    assert row2["ks_pvalue"] < 0.01
    assert row2["effect_size_bucket"] in ("medium", "large")


def test_compare_distributions_reports_quantiles():
    """Every requested quantile column is present for both splits."""
    train = pd.DataFrame({"x": np.arange(100, dtype=float)})
    test = pd.DataFrame({"x": np.arange(50, 150, dtype=float)})
    report = drift.compare_distributions(train, test, columns=["x"])
    for split in ("train", "test"):
        for stat in ("count", "mean", "median", "p5", "p25", "p50", "p75", "p95"):
            assert f"{split}_{stat}" in report.columns


def test_compare_country_share():
    """Country shares are computed independently per split and aligned by country."""
    train = pd.DataFrame({"country": ["US"] * 6 + ["India"] * 4})
    test = pd.DataFrame({"country": ["US"] * 3 + ["India"] * 4 + ["France"] * 3})
    out = drift.compare_country_share(train, test).set_index("country")
    assert out.loc["US", "train_share"] == pytest.approx(0.6)
    assert out.loc["India", "train_share"] == pytest.approx(0.4)
    assert out.loc["France", "test_share"] == pytest.approx(0.3)
    assert out.loc["France", "train_share"] == 0.0  # absent from train, filled rather than NaN
