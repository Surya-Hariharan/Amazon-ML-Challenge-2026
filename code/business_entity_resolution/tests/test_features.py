"""Tests for src.features on small synthetic pairs."""

import numpy as np
import pandas as pd
import pytest

from src.blocking import PASS_BITS
from src.features import (
    FeatureContext,
    build_features,
    context_features,
    feature_columns,
    label_pairs,
)
from src.normalize import normalize_frame

COLS = ["entity_id", "business_name", "business_address", "country"]


def _frames():
    """S1 with a chain name at two addresses; noisy S2/S3 copies and an orphan."""
    s1 = pd.DataFrame([
        ("S1-1", "Primary Care Group LLC", "1115 Friendly Avenue, Greensboro, NC", "US"),
        ("S1-2", "Primary Care Group LLC", "809 Edgecliff Terrace, Austin, TX", "US"),
        ("S1-3", "Swastik Exports Private Limited", "P No 145 Anand Nagar, Medak, Telangana", "India"),
        ("S1-4", "International Business Machines Corp", "1 New Orchard Road, Armonk, NY", "US"),
        ("S1-5", "Bordeaux Parents SAS", "107 Rue du Tondu, Bordeaux, Nouvelle-Aquitaine", "France"),
    ], columns=COLS)
    others = pd.DataFrame([
        ("S2-10", "PRIMARY CARE GROUP", "1115 FRIENDLY AVE, GREENSBORO, NC", "US"),
        ("S3-11", "Primary Care Group LLC", "809 Edgecliff Ter, TX, Austin", "US"),
        ("S2-12", "స్వస్తిక్ ఎక్స్‌పోర్ట్స్ ప్రైవేట్ లిమిటెడ్", "P NO 145 ANAND NAGAR, MEDAK, తెలంగాణ", "India"),
        ("S3-13", "IBM Inc", "", "US"),
        ("S2-14", "BORDEAUX PARENTS SARL", "NO 107 R. DU TONDU, BORDEAUX, Gironde", "France"),
        ("S3-15", "Primary Care Group Corp", "55 Other Street, Dallas, TX", "US"),
    ], columns=COLS)
    return normalize_frame(s1), normalize_frame(others)


def _cands(pairs):
    """Candidate frame with the blocking schema."""
    df = pd.DataFrame(pairs, columns=["s1_id", "cand_id"])
    df["passes"] = np.uint8(PASS_BITS["tfidf"])
    for p in PASS_BITS:
        df[f"score_{p}"] = np.float32(np.nan)
    df["score_tfidf"] = np.float32(0.5)
    return df


@pytest.fixture(scope="module")
def feats():
    """Features for a hand-picked pair set."""
    s1, others = _frames()
    cands = _cands([("S1-1", "S2-10"), ("S1-1", "S3-11"), ("S1-1", "S3-15"),
                    ("S1-2", "S2-10"), ("S1-2", "S3-11"), ("S1-2", "S3-15"),
                    ("S1-3", "S2-12"), ("S1-4", "S3-13"), ("S1-5", "S2-14")])
    out = build_features(cands, s1, others, chunk=4, verbose=False)
    return out.set_index(["s1_id", "cand_id"])


def test_shape_dtypes_no_forbidden_columns(feats):
    """float32 features, no country one-hots or raw ID features."""
    cols = feature_columns(feats.reset_index())
    assert len(cols) > 40
    assert all(feats[c].dtype == np.float32 for c in cols)
    assert not any("country_" in c or c.endswith("_id") for c in cols)
    assert len(feats) == 9


def test_address_separates_chain_names(feats):
    """Same chain name: the address decides (house number equal vs conflict)."""
    right, wrong = feats.loc[("S1-1", "S2-10")], feats.loc[("S1-1", "S3-11")]
    assert right["name_token_set"] == pytest.approx(wrong["name_token_set"])
    assert right["house_equal"] == 1.0 and wrong["house_equal"] == 0.0
    assert right["digit_conflict"] == 0.0 and wrong["digit_conflict"] == 1.0
    assert right["addr_token_set"] > wrong["addr_token_set"]
    assert right["pair_sim"] > wrong["pair_sim"]


def test_native_script_name_matched_by_address(feats):
    """A Telugu-script copy scores high on address and region; name via transliteration."""
    f = feats.loc[("S1-3", "S2-12")]
    assert f["house_equal"] == 1.0
    assert f["region_equal"] == 1.0
    assert f["addr_token_set"] > 0.9
    assert f["name_token_set"] > 0.5


def test_empty_address_is_nan_not_zero(feats):
    """Missing address -> NaN similarity + missing flag; acronym match detected."""
    f = feats.loc[("S1-4", "S3-13")]
    assert f["addr_empty_cand"] == 1.0
    assert np.isnan(f["addr_token_set"]) and np.isnan(f["house_equal"])
    assert f["acronym_match"] == 1.0
    assert f["suffix_both"] == 1.0 and f["suffix_conflict"] == 0.0  # corp ~ inc family


def test_french_pair(feats):
    """French abbreviations, departement vs region and suffix conflict."""
    f = feats.loc[("S1-5", "S2-14")]
    assert f["name_core_exact"] == 1.0
    assert f["addr_token_set"] == pytest.approx(1.0)
    assert f["region_equal"] == 1.0
    assert f["suffix_conflict"] == 1.0  # SAS vs SARL are different legal forms
    assert f["same_country"] == 1.0


def test_context_features(feats):
    """Rank / reverse rank / counts computed over the pair set."""
    assert feats.loc[("S1-1", "S2-10"), "rank_in_s1"] == 1.0
    assert feats.loc[("S1-1", "S2-10"), "n_cands_s1"] == 3.0
    assert feats.loc[("S1-1", "S2-10"), "gap_to_best_s1"] == 0.0
    # S2-10 is listed by S1-1 and S1-2 and belongs to S1-1 (same address).
    assert feats.loc[("S1-1", "S2-10"), "rank_in_cand"] == 1.0
    assert feats.loc[("S1-2", "S2-10"), "rank_in_cand"] == 2.0
    assert feats.loc[("S1-2", "S2-10"), "n_s1_for_cand"] == 2.0
    assert feats.loc[("S1-1", "S3-11"), "is_s3"] == 1.0
    assert feats.loc[("S1-1", "S2-10"), "is_s3"] == 0.0


def test_blocking_features(feats):
    """Pass flags and blocking scores are carried through."""
    f = feats.loc[("S1-1", "S2-10")]
    assert f["in_tfidf"] == 1.0 and f["in_embed"] == 0.0 and f["n_passes"] == 1.0
    assert f["block_tfidf"] == pytest.approx(0.5)
    assert np.isnan(f["emb_cos"])  # no embeddings given


def test_embedding_cosine_used_when_given():
    """emb_cos is the row-wise dot product of the supplied vectors."""
    s1, others = _frames()
    cands = _cands([("S1-1", "S2-10"), ("S1-2", "S2-10")])
    e1 = np.zeros((len(s1), 2), np.float16)
    e2 = np.zeros((len(others), 2), np.float16)
    e1[0] = [1, 0]
    e1[1] = [0, 1]
    e2[0] = [1, 0]
    out = build_features(cands, s1, others, embeddings=(e1, e2), verbose=False)
    assert out["emb_cos"].tolist() == [1.0, 0.0]


def test_chunking_invariant():
    """Chunk size does not change the result."""
    s1, others = _frames()
    cands = _cands([(a, b) for a in s1["entity_id"] for b in others["entity_id"]])
    ctx = FeatureContext.fit(s1, others)
    x = build_features(cands, s1, others, ctx=ctx, chunk=3, verbose=False)
    y = build_features(cands, s1, others, ctx=ctx, chunk=1000, verbose=False)
    pd.testing.assert_frame_equal(x, y)


def test_label_pairs():
    """Labels from the truth dict."""
    pairs = pd.DataFrame({"s1_id": ["S1-1", "S1-1", "S1-2"], "cand_id": ["S2-1", "S3-2", "S2-1"]})
    assert label_pairs(pairs, {"S1-1": ["S2-1"], "S1-2": []}).tolist() == [1, 0, 0]
    assert label_pairs(pairs, {}).tolist() == [0, 0, 0]


def test_context_features_standalone():
    """Reverse rank uses pair_sim across S1s sharing a candidate."""
    df = pd.DataFrame({"s1_id": ["A", "B"], "cand_id": ["X", "X"],
                       "name_token_set": [0.9, 0.5], "addr_token_set": [np.nan, 0.5]})
    out = context_features(df)
    assert out["rank_in_cand"].tolist() == [1.0, 2.0]
    assert out["pair_sim"].tolist() == pytest.approx([0.9, 0.5])
