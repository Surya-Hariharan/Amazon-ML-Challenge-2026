"""True-match failure-stage classification (roadmap v3 Phase 0, E1).

For every train ground-truth true pair, classifies which stage lost it:

    TRUE MATCH
     |
     +-- representation_failure   (mechanical raw-vs-normalised signal loss,
     |                              detected BEFORE blocking is blamed)
     +-- blocking_false_negative  (not in the candidate set, no signal loss found)
     +-- matcher_false_negative   (candidate, but OOF/valid prob < tau)
     +-- decision_false_negative  (scored >= tau, but absent from the final
     |                              prediction -- one-to-one/singleton removal)
     +-- correct_match

Every representation-loss flag is a literal token/digit-set membership check
between raw text and the pipeline's own normalised fields -- never a
subjective "would a human see this" judgement. This is intentionally
conservative: it will also flag some *legitimate* canonicalisations (e.g. a
digit-typo repair, an address-abbreviation expansion producing a raw/normalised
word mismatch) as "loss". Flagged pairs are a cheap first-pass signal to
inspect, not proof of an actual defect -- read them alongside the raw/
normalised text before concluding normalisation is at fault.

Nothing here changes normalisation, blocking, feature, model, decision or
submission behaviour -- every function only reads already-computed objects
(blocking output, scored predictions, the final decision dict, and the
normalised S1/others frames that `run_pipeline.prepare` already produces).
"""

from __future__ import annotations

import re
from collections.abc import Mapping

import numpy as np
import pandas as pd

from . import config
from .drift import NON_LATIN_RE
from .normalize import HONORIFICS, SUFFIX_FAMILY

_DIGIT_RE = re.compile(r"\d+")
_ALPHA_RE = re.compile(r"[^\W\d_]+", re.UNICODE)

#: Tokens whose removal by normalisation is intentional, not information loss.
#: SUFFIX_FAMILY's keys are canonical legal-suffix tokens; the literal spellings
#: below are common raw variants that get canonicalised/removed before that.
_IGNORED_NAME_TOKENS = (
    HONORIFICS | set(SUFFIX_FAMILY)
    | {"and", "the", "of", "de", "et", "private", "limited", "incorporated",
       "corporation", "company", "messrs", "the"}
)

LOSS_KEYS = ("postal_or_long_num_lost", "house_number_lost", "name_token_lost",
             "address_token_lost", "script_info_lost", "other_normalization_loss")
_NO_LOSS = dict.fromkeys(LOSS_KEYS, False)


def _raw_digit_tokens(text: str, min_len: int) -> set[str]:
    """Digit substrings of at least ``min_len`` characters, leading zeros stripped."""
    return {t.lstrip("0") or "0" for t in _DIGIT_RE.findall(text or "") if len(t) >= min_len}


def _raw_alpha_tokens(text: str) -> set[str]:
    """Casefolded alpha tokens (letters of any script, length >= 3)."""
    return {t.casefold() for t in _ALPHA_RE.findall(text or "") if len(t) >= 3}


def _norm_digit_set(row: Mapping) -> set[str]:
    """Union of a normalised record's digits/house_no/long_nums tokens."""
    parts = [row.get("digits", ""), row.get("house_no", ""), row.get("long_nums", "")]
    return {t for p in parts for t in str(p).split() if t}


def _norm_name_set(row: Mapping) -> set[str]:
    """Tokens of a normalised record's ``name_core``."""
    return set(str(row.get("name_core", "")).split())


def _norm_addr_set(row: Mapping) -> set[str]:
    """Tokens of a normalised record's ``addr_norm`` + ``landmark`` + ``region``
    (a token legitimately moved into one of those fields is not a loss)."""
    return (set(str(row.get("addr_norm", "")).split())
           | set(str(row.get("landmark", "")).split())
           | set(str(row.get("region", "")).split()))


def representation_loss_flags(s1_row: Mapping, cand_row: Mapping) -> dict[str, bool]:
    """Mechanical raw-vs-normalised comparison for one (S1, candidate) pair.

    Args:
        s1_row / cand_row: mapping-like objects (e.g. a ``pandas.Series`` row)
            carrying both the raw ``business_name``/``business_address``
            columns and ``normalize.normalize_frame``'s output columns --
            i.e. a row from ``run_pipeline.prepare``'s ``s1``/``others`` frames.

    Returns:
        ``postal_or_long_num_lost``: a raw digit token of >= 4 digits, shared
            between the two raw texts, is missing from either side's
            normalised digit fields.
        ``house_number_lost``: same, for raw digit tokens of 1-3 digits.
        ``name_token_lost``: a raw alpha token (len >= 3, not an honorific/
            legal-suffix variant) shared between the two raw *names* is
            missing from either side's ``name_core`` tokens.
        ``address_token_lost``: same, for the two raw *addresses* vs.
            ``addr_norm``/``landmark``/``region`` tokens.
        ``script_info_lost``: one side's raw name/address contains non-ASCII
            (script) characters, but that side's normalised ``name_core`` and
            ``addr_norm`` are both empty despite the raw fields being non-empty.
        ``other_normalization_loss``: broader catch-all -- a shared raw alpha
            token (name or address, either side, cross-field) absent from the
            union of both records' normalised name/address/digit fields.
    """
    s1_name, s1_addr = str(s1_row.get(config.NAME_COL, "")), str(s1_row.get(config.ADDRESS_COL, ""))
    c_name, c_addr = str(cand_row.get(config.NAME_COL, "")), str(cand_row.get(config.ADDRESS_COL, ""))

    s1_digits_raw = _raw_digit_tokens(s1_name + " " + s1_addr, 1)
    c_digits_raw = _raw_digit_tokens(c_name + " " + c_addr, 1)
    shared_digits = s1_digits_raw & c_digits_raw
    s1_norm_digits, c_norm_digits = _norm_digit_set(s1_row), _norm_digit_set(cand_row)
    shared_long = {d for d in shared_digits if len(d) >= 4}
    shared_short = shared_digits - shared_long
    postal_lost = any(d not in s1_norm_digits or d not in c_norm_digits for d in shared_long)
    house_lost = any(d not in s1_norm_digits or d not in c_norm_digits for d in shared_short)

    s1_name_raw = _raw_alpha_tokens(s1_name) - _IGNORED_NAME_TOKENS
    c_name_raw = _raw_alpha_tokens(c_name) - _IGNORED_NAME_TOKENS
    shared_name = s1_name_raw & c_name_raw
    s1_name_norm, c_name_norm = _norm_name_set(s1_row), _norm_name_set(cand_row)
    name_lost = any(t not in s1_name_norm or t not in c_name_norm for t in shared_name)

    s1_addr_raw, c_addr_raw = _raw_alpha_tokens(s1_addr), _raw_alpha_tokens(c_addr)
    shared_addr = s1_addr_raw & c_addr_raw
    s1_addr_norm, c_addr_norm = _norm_addr_set(s1_row), _norm_addr_set(cand_row)
    addr_lost = any(t not in s1_addr_norm or t not in c_addr_norm for t in shared_addr)

    def _script_lost(name: str, addr: str, core: str, norm_addr: str) -> bool:
        """One side's raw script content produced no usable normalised text."""
        has_script = bool(NON_LATIN_RE.search(name)) or bool(NON_LATIN_RE.search(addr))
        raw_nonempty = bool(name.strip()) or bool(addr.strip())
        norm_empty = not core.strip() and not norm_addr.strip()
        return has_script and raw_nonempty and norm_empty

    script_lost = (
        _script_lost(s1_name, s1_addr, str(s1_row.get("name_core", "")), str(s1_row.get("addr_norm", "")))
        or _script_lost(c_name, c_addr, str(cand_row.get("name_core", "")), str(cand_row.get("addr_norm", "")))
    )

    s1_combined_raw = (_raw_alpha_tokens(s1_name) | _raw_alpha_tokens(s1_addr)) - _IGNORED_NAME_TOKENS
    c_combined_raw = (_raw_alpha_tokens(c_name) | _raw_alpha_tokens(c_addr)) - _IGNORED_NAME_TOKENS
    shared_combined = s1_combined_raw & c_combined_raw
    combined_norm = (s1_name_norm | s1_addr_norm | s1_norm_digits
                     | c_name_norm | c_addr_norm | c_norm_digits)
    other_lost = any(t not in combined_norm for t in shared_combined)

    return {
        "postal_or_long_num_lost": bool(postal_lost),
        "house_number_lost": bool(house_lost),
        "name_token_lost": bool(name_lost),
        "address_token_lost": bool(addr_lost),
        "script_info_lost": bool(script_lost),
        "other_normalization_loss": bool(other_lost),
    }


def classify_true_pairs(
    s1: pd.DataFrame,
    others: pd.DataFrame,
    cands: pd.DataFrame,
    scored: pd.DataFrame,
    pred: Mapping[str, list[str]],
    truth: Mapping[str, list[str]],
    tau: float,
) -> pd.DataFrame:
    """Classify every ``truth`` true pair into one failure stage.

    Args:
        s1 / others: normalised frames (``run_pipeline.prepare``'s output;
            must carry both raw and normalised columns).
        cands: blocking output (``s1_id``, ``cand_id``, ...).
        scored: frame with ``s1_id``, ``cand_id``, ``prob`` (OOF or held-out
            predictions).
        pred: final ``{s1_id: [matched ids]}`` after the full decision logic
            (thresholding + one-to-one/singleton gates).
        truth: ``{s1_id: [true matched ids]}`` restricted to the S1 universe
            being diagnosed (e.g. the held-out validation S1s).
        tau: the threshold used to produce ``pred`` from ``scored``.

    Returns:
        One row per ``(s1_id, cand_id)`` true pair, with a ``stage`` column
        (see module docstring) plus every :func:`representation_loss_flags` key.
    """
    true_pairs = [(s, m) for s, ms in truth.items() for m in ms]
    if not true_pairs:
        return pd.DataFrame(columns=["s1_id", "cand_id", "stage", *LOSS_KEYS])

    df = pd.DataFrame(true_pairs, columns=["s1_id", "cand_id"])
    cand_key = set(zip(cands["s1_id"], cands["cand_id"]))
    scored_prob = (scored.set_index(["s1_id", "cand_id"])["prob"]
                  if len(scored) else pd.Series(dtype=float))
    pred_key = {(s, c) for s, cs in pred.items() for c in cs}

    keys = list(zip(df["s1_id"], df["cand_id"]))
    in_cands = np.fromiter((k in cand_key for k in keys), dtype=bool, count=len(keys))
    s1_pos = pd.Index(s1[config.ID_COL]).get_indexer(df["s1_id"])
    o_pos = pd.Index(others[config.ID_COL]).get_indexer(df["cand_id"])

    # Representation-loss flags only ever mean something for a pair blocking
    # missed -- default every row to "no loss" and only run the (comparatively
    # expensive) per-pair mechanical check for that (usually small) subset,
    # instead of allocating a fresh all-False dict per row for every pair
    # blocking already found (the majority population when recall is good).
    loss_df = pd.DataFrame(False, index=df.index, columns=list(LOSS_KEYS))
    stages = np.empty(len(df), dtype=object)

    for i in np.flatnonzero(~in_cands):
        flags = (representation_loss_flags(s1.iloc[s1_pos[i]], others.iloc[o_pos[i]])
                if s1_pos[i] >= 0 and o_pos[i] >= 0 else dict(_NO_LOSS))
        loss_df.iloc[i] = [flags[k] for k in LOSS_KEYS]
        stages[i] = ("representation_failure" if any(flags.values())
                    else "blocking_false_negative")

    cand_idx = np.flatnonzero(in_cands)
    if len(cand_idx):
        cand_keys = [keys[i] for i in cand_idx]
        probs = np.fromiter((scored_prob.get(k, np.nan) for k in cand_keys),
                            dtype=float, count=len(cand_keys))
        below_tau = ~(probs >= tau)
        not_predicted = np.fromiter((k not in pred_key for k in cand_keys),
                                    dtype=bool, count=len(cand_keys))
        stages[cand_idx] = np.where(
            below_tau, "matcher_false_negative",
            np.where(not_predicted, "decision_false_negative", "correct_match"))

    return pd.concat([df.assign(stage=stages), loss_df], axis=1)


def false_positive_report(
    scored: pd.DataFrame, pred: Mapping[str, list[str]], truth: Mapping[str, list[str]]
) -> pd.DataFrame:
    """Predicted-but-wrong pairs, ranked as top-competitor vs. low-rank noise.

    ``top_wrong`` marks a wrong prediction that was the *highest*-probability
    prediction for its S1 among that S1's wrong predictions (a hard
    confusable); the rest are lower-ranked noise picks a stricter threshold
    would likely also have caught.
    """
    pred_key = {(s, c) for s, cs in pred.items() for c in cs}
    true_key = {(s, m) for s, ms in truth.items() for m in ms}
    rows = [(s, c, p) for s, c, p in zip(scored["s1_id"], scored["cand_id"], scored["prob"])
            if (s, c) in pred_key and (s, c) not in true_key]
    fp = pd.DataFrame(rows, columns=["s1_id", "cand_id", "prob"])
    if fp.empty:
        return fp.assign(rank=pd.Series(dtype=float), top_wrong=pd.Series(dtype=bool))
    fp["rank"] = fp.groupby("s1_id")["prob"].rank(ascending=False, method="min")
    fp["top_wrong"] = fp["rank"] == 1
    return fp


def multi_match_report(
    pred: Mapping[str, list[str]], truth: Mapping[str, list[str]]
) -> pd.DataFrame:
    """Per-S1 predicted vs. true match counts, for singleton/multi-match analysis."""
    ids = sorted(set(pred) | set(truth))
    n_true = [len(set(truth.get(s, ()))) for s in ids]
    n_pred = [len(set(pred.get(s, ()))) for s in ids]
    df = pd.DataFrame({"s1_id": ids, "n_true": n_true, "n_pred": n_pred})
    df["is_singleton"] = df["n_true"] == 0
    df["singleton_false_positive"] = df["is_singleton"] & (df["n_pred"] > 0)
    df["is_multi_match"] = df["n_true"] >= 2
    df["multi_underpredicted"] = df["is_multi_match"] & (df["n_pred"] < df["n_true"])
    df["multi_overpredicted"] = df["is_multi_match"] & (df["n_pred"] > df["n_true"])
    return df


def one_to_one_removals(
    scored: pd.DataFrame, truth: Mapping[str, list[str]]
) -> pd.DataFrame:
    """True pairs whose candidate was assigned away by one-to-one assignment.

    A true pair "loses" its candidate under one-to-one assignment iff a rival
    S1 scored that same candidate higher (see ``decide.assign_one_to_one``).
    """
    from .decide import assign_one_to_one

    rows = [(s, m) for s, ms in truth.items() for m in ms]
    df = pd.DataFrame(rows, columns=["s1_id", "cand_id"])
    if df.empty:
        return df.assign(was_scored=pd.Series(dtype=bool),
                         removed_by_one_to_one=pd.Series(dtype=bool))
    kept = set(zip(*assign_one_to_one(scored)[["s1_id", "cand_id"]].to_numpy().T)) \
        if len(scored) else set()
    scored_key = set(zip(scored["s1_id"], scored["cand_id"]))
    was_scored = [(s, c) in scored_key for s, c in zip(df["s1_id"], df["cand_id"])]
    removed = [w and (s, c) not in kept
              for w, s, c in zip(was_scored, df["s1_id"], df["cand_id"])]
    return df.assign(was_scored=was_scored, removed_by_one_to_one=removed)


def slice_report(
    classified: pd.DataFrame,
    s1: pd.DataFrame,
    others: pd.DataFrame,
    cands: pd.DataFrame,
    s1_country: Mapping[str, str] | None = None,
) -> pd.DataFrame:
    """Stage-share breakdown, sliced by country / candidate-count / quality buckets.

    Adds, per true pair already in ``classified``: ``country``,
    ``candidate_bucket`` (0 / 1-5 / 6-20 / 21-50 / 50+ candidates for that
    S1), ``name_quality_bucket`` (``exact_core`` if the two records'
    ``name_core`` are identical, else ``different_core``),
    ``addr_quality_bucket`` (``both_present``/``one_missing``/``both_missing``),
    ``transliteration_bucket`` (``any_non_latin``/``all_latin``, from the raw
    text), and ``any_representation_loss`` (any :data:`LOSS_KEYS` flag set).

    Returns one row per ``(slice_dim, slice_value, stage)`` with a count and
    that slice value's share across stages.
    """
    df = classified.copy()
    if df.empty:
        return pd.DataFrame(columns=["slice_dim", "slice_value", "stage", "count", "share"])

    n_cand = cands.groupby("s1_id").size()
    df["n_candidates"] = df["s1_id"].map(n_cand).fillna(0)
    df["candidate_bucket"] = pd.cut(
        df["n_candidates"], bins=[-1, 0, 5, 20, 50, np.inf],
        labels=["0", "1-5", "6-20", "21-50", "50+"])
    if s1_country:
        df["country"] = df["s1_id"].map(s1_country).fillna("unknown")

    s1_pos = pd.Index(s1[config.ID_COL]).get_indexer(df["s1_id"])
    o_pos = pd.Index(others[config.ID_COL]).get_indexer(df["cand_id"])
    ok = (s1_pos >= 0) & (o_pos >= 0)
    s1_core = np.where(ok, s1["name_core"].to_numpy()[np.clip(s1_pos, 0, None)], "")
    c_core = np.where(ok, others["name_core"].to_numpy()[np.clip(o_pos, 0, None)], "")
    df["name_quality_bucket"] = np.where(ok & (s1_core == c_core), "exact_core", "different_core")

    s1_ae = np.where(ok, s1["addr_empty"].to_numpy()[np.clip(s1_pos, 0, None)], False)
    c_ae = np.where(ok, others["addr_empty"].to_numpy()[np.clip(o_pos, 0, None)], False)
    n_missing = s1_ae.astype(int) + c_ae.astype(int)
    df["addr_quality_bucket"] = np.select(
        [n_missing == 0, n_missing == 1, n_missing == 2],
        ["both_present", "one_missing", "both_missing"], default="unknown")

    s1_name_raw = np.where(ok, s1[config.NAME_COL].to_numpy()[np.clip(s1_pos, 0, None)], "")
    s1_addr_raw = np.where(ok, s1[config.ADDRESS_COL].to_numpy()[np.clip(s1_pos, 0, None)], "")
    c_name_raw = np.where(ok, others[config.NAME_COL].to_numpy()[np.clip(o_pos, 0, None)], "")
    c_addr_raw = np.where(ok, others[config.ADDRESS_COL].to_numpy()[np.clip(o_pos, 0, None)], "")
    any_non_latin = [
        any(bool(NON_LATIN_RE.search(str(t))) for t in (sn, sa, cn, ca))
        for sn, sa, cn, ca in zip(s1_name_raw, s1_addr_raw, c_name_raw, c_addr_raw)
    ]
    df["transliteration_bucket"] = np.where(any_non_latin, "any_non_latin", "all_latin")
    loss_cols = [c for c in LOSS_KEYS if c in df.columns]
    df["any_representation_loss"] = df[loss_cols].any(axis=1) if loss_cols else False

    dims = [d for d in ("country", "candidate_bucket", "name_quality_bucket",
                        "addr_quality_bucket", "transliteration_bucket",
                        "any_representation_loss") if d in df.columns]
    out = []
    for dim in dims:
        g = df.groupby([dim, "stage"], observed=True).size().rename("count").reset_index()
        g["share"] = g["count"] / g.groupby(dim, observed=True)["count"].transform("sum")
        out.append(g.rename(columns={dim: "slice_value"}).assign(slice_dim=dim))
    return (pd.concat(out, ignore_index=True) if out
           else pd.DataFrame(columns=["slice_dim", "slice_value", "stage", "count", "share"]))
