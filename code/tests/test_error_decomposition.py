"""Tests for src.error_decomposition (error-decomposition diagnostic)."""

import pandas as pd
import pytest

from src import config, error_decomposition as edecomp


def _row(name="", addr="", name_core="", addr_norm="", digits="", house_no="",
        long_nums="", landmark="", region=""):
    """A plain dict standing in for one normalised record's relevant fields."""
    return {
        config.NAME_COL: name, config.ADDRESS_COL: addr,
        "name_core": name_core, "addr_norm": addr_norm, "digits": digits,
        "house_no": house_no, "long_nums": long_nums, "landmark": landmark, "region": region,
    }


# --- representation_loss_flags: each flag mechanically isolated ----------------------

def test_no_loss_when_shared_tokens_survive_normalisation():
    """Identical shared name/address/digit content that fully survives normalisation
    triggers no loss flag."""
    s1 = _row(name="Green Traders", addr="12 Main St", name_core="green traders",
             addr_norm="12 main street", digits="12", house_no="12")
    cand = _row(name="Green Traders LLC", addr="12 Main Street", name_core="green traders",
               addr_norm="12 main street", digits="12", house_no="12")
    flags = edecomp.representation_loss_flags(s1, cand)
    assert not any(flags.values()), flags


def test_postal_or_long_num_lost_when_shared_zip_missing_from_normalised_digits():
    """A shared 5-digit token in raw text that the normalised digits field lost."""
    s1 = _row(addr="78701 Main St, Austin", digits="", long_nums="")
    cand = _row(addr="78701 Main Street, Austin", digits="78701", long_nums="78701")
    flags = edecomp.representation_loss_flags(s1, cand)
    assert flags["postal_or_long_num_lost"] is True


def test_house_number_lost_when_shared_short_digit_missing_from_normalised_digits():
    """A shared short (house-number-like) digit token missing from normalised digits."""
    s1 = _row(addr="12 Main St, Austin", digits="", house_no="")
    cand = _row(addr="12 Main Street, Austin", digits="12", house_no="12")
    flags = edecomp.representation_loss_flags(s1, cand)
    assert flags["house_number_lost"] is True
    assert flags["postal_or_long_num_lost"] is False


def test_name_token_lost_when_shared_brand_word_missing_from_name_core():
    """A shared brand word present in both raw names but missing from one side's name_core."""
    s1 = _row(name="Zephay Traders", name_core="")  # normalisation dropped everything
    cand = _row(name="Zephay Traders LLC", name_core="zephay traders")
    flags = edecomp.representation_loss_flags(s1, cand)
    assert flags["name_token_lost"] is True


def test_name_token_lost_false_when_only_shared_token_is_a_legal_suffix_variant():
    """A shared legal-suffix word (its removal is intentional canonicalisation, not loss)."""
    s1 = _row(name="Acme Private Limited", name_core="acme")
    cand = _row(name="Acme Pvt Ltd", name_core="acme")
    flags = edecomp.representation_loss_flags(s1, cand)
    assert flags["name_token_lost"] is False


def test_address_token_lost_when_shared_street_word_missing_from_addr_norm():
    """A shared street/locality word present in both raw addresses but missing from
    normalised addr_norm (and not moved to landmark/region either)."""
    s1 = _row(addr="Velachery Main Road", addr_norm="")
    cand = _row(addr="Velachery Main Road", addr_norm="velachery main road")
    flags = edecomp.representation_loss_flags(s1, cand)
    assert flags["address_token_lost"] is True


def test_address_token_lost_false_when_shared_word_moved_to_landmark():
    """A shared word that legitimately moved to the landmark field is not "lost"."""
    s1 = _row(addr="Near SBI Bank", addr_norm="", landmark="sbi bank")
    cand = _row(addr="Opposite SBI Bank", addr_norm="", landmark="sbi bank")
    flags = edecomp.representation_loss_flags(s1, cand)
    assert flags["address_token_lost"] is False


def test_script_info_lost_when_transliteration_produces_nothing_usable():
    """Raw non-Latin script content whose normalised fields are both empty despite
    non-empty raw text."""
    s1 = _row(name="भारत ट्रेडर्स", addr="", name_core="", addr_norm="")
    cand = _row(name="Bharat Traders", addr="", name_core="bharat traders", addr_norm="")
    flags = edecomp.representation_loss_flags(s1, cand)
    assert flags["script_info_lost"] is True


def test_script_info_lost_false_when_transliteration_produces_usable_text():
    """Non-Latin raw text that *does* produce a non-empty normalised name_core."""
    s1 = _row(name="भारत ट्रेडर्स", addr="", name_core="bharat traders", addr_norm="")
    cand = _row(name="Bharat Traders", addr="", name_core="bharat traders", addr_norm="")
    flags = edecomp.representation_loss_flags(s1, cand)
    assert flags["script_info_lost"] is False


def test_other_normalization_loss_catches_cross_field_token_movement():
    """A token shared across raw name (S1) and raw address (candidate) that ends up
    in neither side's normalised fields anywhere -- caught only by the catch-all."""
    s1 = _row(name="Acme", addr="", name_core="", addr_norm="")
    cand = _row(name="", addr="Acme Road", name_core="", addr_norm="")
    flags = edecomp.representation_loss_flags(s1, cand)
    assert flags["other_normalization_loss"] is True
    assert flags["name_token_lost"] is False
    assert flags["address_token_lost"] is False


# --- classify_true_pairs: one representative pair per stage ---------------------------

def _frame(rows):
    """Build a normalised-frame stand-in from row dicts (see _row) plus an entity_id
    and country, matching run_pipeline.prepare's s1/others frame shape."""
    df = pd.DataFrame(rows)
    df[config.COUNTRY_COL] = "US"
    df["addr_empty"] = df[config.ADDRESS_COL].str.len() == 0
    return df


@pytest.fixture
def scenario():
    """Five (S1, candidate) true pairs, one per failure stage."""
    s1_rows = [
        {config.ID_COL: "S1-1", **_row(name="Rare Traders", addr="99999 Oak St",
                                       name_core="rare traders", addr_norm="99999 oak street",
                                       digits="")},  # representation_failure: digits dropped
        {config.ID_COL: "S1-2", **_row(name="Foo Bar", addr="1 Road",
                                       name_core="foo bar", addr_norm="1 road", digits="1",
                                       house_no="1")},  # blocking_false_negative: clean, just missed
        {config.ID_COL: "S1-3", **_row(name="Matcher Miss", addr="2 Road",
                                       name_core="matcher miss", addr_norm="2 road")},
        {config.ID_COL: "S1-4", **_row(name="Decision Drop", addr="3 Road",
                                       name_core="decision drop", addr_norm="3 road")},
        {config.ID_COL: "S1-5", **_row(name="Correct Match", addr="4 Road",
                                       name_core="correct match", addr_norm="4 road")},
    ]
    o_rows = [
        {config.ID_COL: "S2-1", **_row(name="Rare Traders Co", addr="99999 Oak Street",
                                       name_core="rare traders", addr_norm="99999 oak street",
                                       digits="99999", long_nums="99999")},
        {config.ID_COL: "S2-2", **_row(name="Foo Bar", addr="1 Road",
                                       name_core="foo bar", addr_norm="1 road", digits="1",
                                       house_no="1")},
        {config.ID_COL: "S2-3", **_row(name="Matcher Miss", addr="2 Road",
                                       name_core="matcher miss", addr_norm="2 road")},
        {config.ID_COL: "S2-4", **_row(name="Decision Drop", addr="3 Road",
                                       name_core="decision drop", addr_norm="3 road")},
        {config.ID_COL: "S2-5", **_row(name="Correct Match", addr="4 Road",
                                       name_core="correct match", addr_norm="4 road")},
    ]
    s1 = _frame(s1_rows)
    others = _frame(o_rows)
    truth = {f"S1-{i}": [f"S2-{i}"] for i in range(1, 6)}
    # Only S1-3/4/5 pairs survived blocking (S1-1/S1-2 were missed entirely).
    cands = pd.DataFrame({
        "s1_id": ["S1-3", "S1-4", "S1-5"], "cand_id": ["S2-3", "S2-4", "S2-5"],
    })
    scored = pd.DataFrame({
        "s1_id": ["S1-3", "S1-4", "S1-5"], "cand_id": ["S2-3", "S2-4", "S2-5"],
        "prob": [0.2, 0.9, 0.9],
    })
    tau = 0.5
    # S1-4 scored above tau but was dropped by the decision stage (e.g. one-to-one).
    pred = {"S1-3": [], "S1-4": [], "S1-5": ["S2-5"]}
    return dict(s1=s1, others=others, cands=cands, scored=scored, pred=pred, truth=truth, tau=tau)


def test_classify_true_pairs_assigns_each_stage_correctly(scenario):
    """Every stage in the decomposition tree is reachable and correctly assigned."""
    out = edecomp.classify_true_pairs(
        scenario["s1"], scenario["others"], scenario["cands"], scenario["scored"],
        scenario["pred"], scenario["truth"], scenario["tau"])
    stage = out.set_index("s1_id")["stage"]
    assert stage["S1-1"] == "representation_failure"
    assert stage["S1-2"] == "blocking_false_negative"
    assert stage["S1-3"] == "matcher_false_negative"
    assert stage["S1-4"] == "decision_false_negative"
    assert stage["S1-5"] == "correct_match"
    # Representation-loss flags are only meaningful (and only computed) for the
    # pairs classified as representation_failure/blocking_false_negative.
    by_s1 = out.set_index("s1_id")
    assert bool(by_s1.loc["S1-1", "postal_or_long_num_lost"])
    assert not bool(by_s1.loc["S1-2", "postal_or_long_num_lost"])


def test_classify_true_pairs_with_no_candidates_at_all(scenario):
    """An empty candidate frame -- blocking found nothing for anyone -- must resolve
    every true pair to representation_failure or blocking_false_negative, never to
    a matcher/decision/correct stage (those require the pair to be a candidate)."""
    empty_cands = scenario["cands"].iloc[0:0]
    empty_scored = scenario["scored"].iloc[0:0]
    out = edecomp.classify_true_pairs(
        scenario["s1"], scenario["others"], empty_cands, empty_scored,
        {}, scenario["truth"], scenario["tau"])
    assert set(out["stage"]) <= {"representation_failure", "blocking_false_negative"}
    assert len(out) == sum(len(v) for v in scenario["truth"].values())


def test_classify_true_pairs_empty_truth_returns_empty_frame(scenario):
    """No true pairs -> an empty, correctly-columned frame, not an error."""
    out = edecomp.classify_true_pairs(
        scenario["s1"], scenario["others"], scenario["cands"], scenario["scored"],
        scenario["pred"], {}, scenario["tau"])
    assert out.empty
    assert "stage" in out.columns


# --- false_positive_report / multi_match_report / one_to_one_removals -----------------

def test_false_positive_report_ranks_top_wrong_candidate():
    """The highest-probability wrong prediction for an S1 is top_wrong; the rest aren't."""
    scored = pd.DataFrame({
        "s1_id": ["S1-1", "S1-1", "S1-1"], "cand_id": ["S2-1", "S2-2", "S2-3"],
        "prob": [0.9, 0.8, 0.95],
    })
    pred = {"S1-1": ["S2-1", "S2-2", "S2-3"]}
    truth = {"S1-1": ["S2-3"]}  # only S2-3 is a true match; S2-1/S2-2 are false positives
    fp = edecomp.false_positive_report(scored, pred, truth).set_index("cand_id")
    assert set(fp.index) == {"S2-1", "S2-2"}
    assert fp.loc["S2-1", "top_wrong"]  # 0.9 is the highest among the two wrong ones
    assert not fp.loc["S2-2", "top_wrong"]


def test_multi_match_report_flags_singleton_fp_and_under_over_prediction():
    """Singleton false positives and multi-match under/over-prediction are detected."""
    pred = {"S1-1": ["S2-1"], "S1-2": [], "S1-3": ["S2-3"], "S1-4": ["S2-4", "S2-5", "S2-6"]}
    truth = {"S1-1": [], "S1-2": [], "S1-3": ["S2-3", "S2-3b"], "S1-4": ["S2-4", "S2-5"]}
    mm = edecomp.multi_match_report(pred, truth).set_index("s1_id")
    assert mm.loc["S1-1", "singleton_false_positive"]
    assert not mm.loc["S1-2", "singleton_false_positive"]
    assert mm.loc["S1-3", "multi_underpredicted"]
    assert mm.loc["S1-4", "multi_overpredicted"]


def test_one_to_one_removals_detects_true_pair_losing_to_a_rival():
    """A true pair whose candidate a rival S1 scored higher is flagged as removed."""
    scored = pd.DataFrame({
        "s1_id": ["S1-1", "S1-2"], "cand_id": ["S2-1", "S2-1"], "prob": [0.6, 0.9],
    })
    truth = {"S1-1": ["S2-1"]}  # S1-1's true match, but S1-2 outscores it for S2-1
    out = edecomp.one_to_one_removals(scored, truth)
    row = out.iloc[0]
    assert bool(row["was_scored"])
    assert bool(row["removed_by_one_to_one"])


def test_one_to_one_removals_no_removal_when_pair_wins():
    """A true pair that wins the one-to-one contest is not flagged as removed."""
    scored = pd.DataFrame({
        "s1_id": ["S1-1", "S1-2"], "cand_id": ["S2-1", "S2-1"], "prob": [0.9, 0.6],
    })
    truth = {"S1-1": ["S2-1"]}
    out = edecomp.one_to_one_removals(scored, truth)
    assert not bool(out.iloc[0]["removed_by_one_to_one"])


# --- slice_report -----------------------------------------------------------------------

def test_slice_report_shares_sum_to_one_per_slice_value(scenario):
    """Every slice value's stage shares sum to 1.0."""
    classified = edecomp.classify_true_pairs(
        scenario["s1"], scenario["others"], scenario["cands"], scenario["scored"],
        scenario["pred"], scenario["truth"], scenario["tau"])
    s1_country = {s: "US" for s in scenario["truth"]}
    sl = edecomp.slice_report(classified, scenario["s1"], scenario["others"],
                              scenario["cands"], s1_country)
    assert not sl.empty
    totals = sl.groupby(["slice_dim", "slice_value"])["share"].sum()
    assert (totals.round(6) == 1.0).all()


def test_slice_report_without_country_still_produces_other_dims(scenario):
    """s1_country=None skips the country dimension but every other slice still works."""
    classified = edecomp.classify_true_pairs(
        scenario["s1"], scenario["others"], scenario["cands"], scenario["scored"],
        scenario["pred"], scenario["truth"], scenario["tau"])
    sl = edecomp.slice_report(classified, scenario["s1"], scenario["others"],
                              scenario["cands"], s1_country=None)
    assert not sl.empty
    assert "country" not in set(sl["slice_dim"])
    assert "candidate_bucket" in set(sl["slice_dim"])


def test_slice_report_empty_classified_returns_empty_frame(scenario):
    """An empty classified frame (e.g. no true pairs) yields an empty, correctly-
    columned slice report rather than raising."""
    empty = edecomp.classify_true_pairs(
        scenario["s1"], scenario["others"], scenario["cands"], scenario["scored"],
        scenario["pred"], {}, scenario["tau"])
    sl = edecomp.slice_report(empty, scenario["s1"], scenario["others"], scenario["cands"])
    assert sl.empty
    assert list(sl.columns) == ["slice_dim", "slice_value", "stage", "count", "share"]
