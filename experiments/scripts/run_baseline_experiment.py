"""Baseline validation experiment driver (experiments/, not code/src/).

Runs the REAL, unmodified pipeline (src.run_pipeline.valid_run) on a deterministic,
S1-level sample of the training data (ground truth preserved) and dumps a JSON of
every metric run_pipeline already computes, plus a few extra diagnostics (feature
separability, score percentiles, threshold sweep, error-budget, resource usage)
computed by calling the same real modules directly. No source files are modified.

Usage: python run_baseline_experiment.py <target_n_s1> <tag>
"""
import json
import sys
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_DIR = REPO_ROOT / "code"
sys.path.insert(0, str(CODE_DIR))

from src import config  # noqa: E402
from src.io_utils import load_split, parse_id_list  # noqa: E402
from src.run_pipeline import (  # noqa: E402
    subsample_train, truth_from_frame, prepare, block, split_s1,
)
from src.blocking import report_blocking_stats, candidates_to_lists  # noqa: E402
from src.features import build_features, feature_columns, label_pairs  # noqa: E402
from src.model import train_oof, train_full, predict  # noqa: E402
from src.decide import tune_threshold, apply_threshold  # noqa: E402
from src.evaluate import score_report  # noqa: E402

TOTAL_TRAIN_S1 = 2_206_821  # from CLAUDE.md / prior audit, verified below


def main():
    target_n = int(sys.argv[1])
    tag = sys.argv[2]
    out_path = REPO_ROOT / "experiments" / "results" / f"{tag}.json"

    t_all0 = time.time()
    timings = {}
    result = {"target_n_s1": target_n, "tag": tag}

    t0 = time.time()
    data = load_split("train")
    s1, s2, s3, gt = data["s1"], data["s2"], data["s3"], data["ground_truth"]
    truth_full = truth_from_frame(gt)
    timings["load_data"] = time.time() - t0
    result["measured_total_train_s1"] = len(s1)
    result["measured_total_train_s2"] = len(s2)
    result["measured_total_train_s3"] = len(s3)

    frac = min(1.0, target_n / len(s1))
    t0 = time.time()
    s1_s, s2_s, s3_s, truth = subsample_train(s1, s2, s3, truth_full, frac, seed=config.SEED)
    timings["subsample"] = time.time() - t0
    result["actual_n_s1"] = len(s1_s)
    result["actual_n_s2"] = len(s2_s)
    result["actual_n_s3"] = len(s3_s)
    result["frac_used"] = frac

    # ---- Stage A: data / ground-truth characterization ----
    match_counts = {s: len(truth.get(s, [])) for s in s1_s[config.ID_COL]}
    mc = pd.Series(match_counts)
    s1_country = dict(zip(s1_s[config.ID_COL], s1_s[config.COUNTRY_COL]))
    countries = pd.Series(s1_country)
    zero = int((mc == 0).sum())
    one = int((mc == 1).sum())
    multi = int((mc > 1).sum())
    all_matched_ids = {m for ms in truth.values() for m in ms}
    s2_ids = set(s2_s[config.ID_COL])
    s3_ids = set(s3_s[config.ID_COL])
    n_to_s2 = sum(1 for m in all_matched_ids if m in s2_ids)
    n_to_s3 = sum(1 for m in all_matched_ids if m in s3_ids)
    # one-to-one check: does any S2/S3 id appear in >1 S1's true match list?
    owner = {}
    conflicts = 0
    for s, ms in truth.items():
        for m in ms:
            if m in owner and owner[m] != s:
                conflicts += 1
            owner[m] = s
    result["stage_A_characterization"] = {
        "n_s1": len(s1_s), "n_s2": len(s2_s), "n_s3": len(s3_s),
        "zero_match_s1": zero, "one_match_s1": one, "multi_match_s1": multi,
        "zero_match_pct": 100.0 * zero / max(len(s1_s), 1),
        "one_match_pct": 100.0 * one / max(len(s1_s), 1),
        "multi_match_pct": 100.0 * multi / max(len(s1_s), 1),
        "mean_matches_per_s1": float(mc.mean()), "median_matches_per_s1": float(mc.median()),
        "max_matches_per_s1": int(mc.max()) if len(mc) else 0,
        "country_counts_s1": countries.value_counts().to_dict(),
        "matched_ids_in_s2": n_to_s2, "matched_ids_in_s3": n_to_s3,
        "matched_id_reuse_conflicts_ONE_TO_ONE_check": conflicts,
        "france_present_in_train_sample": bool((countries == "France").any()),
    }

    # ---- prepare (normalize + embed) with resource profiling ----
    tracemalloc.start()
    t0 = time.time()
    prep = prepare(s1_s, s2_s, s3_s, config.USE_EMBEDDINGS, None, use_cache=False)
    timings["prepare_normalize_embed"] = time.time() - t0
    cur, peak_normemb = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    try:
        import torch
        torch.cuda.reset_peak_memory_stats()
        gpu_before = torch.cuda.max_memory_allocated()
    except Exception:
        torch = None

    # ---- Stage B: blocking ----
    tracemalloc.start()
    t0 = time.time()
    cands = block(prep, use_cache=False)
    timings["blocking"] = time.time() - t0
    cur, peak_block = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    gpu_peak_mb = None
    if torch is not None:
        try:
            gpu_peak_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
        except Exception:
            gpu_peak_mb = None

    s1_ids_all = list(prep["s1"][config.ID_COL])
    bstats = report_blocking_stats(cands, truth, s1_ids_all, len(prep["others"]), s1_country)
    result["stage_B_blocking"] = bstats
    result["stage_14_resource_blocking"] = {
        "runtime_s": timings["blocking"], "peak_ram_mb_traced": peak_block / (1024 ** 2),
    }
    result["stage_14_resource_prepare"] = {
        "runtime_s": timings["prepare_normalize_embed"],
        "peak_ram_mb_traced": peak_normemb / (1024 ** 2),
        "gpu_peak_mb": gpu_peak_mb,
    }

    # candidate volume distribution
    sizes = cands.groupby("s1_id").size()
    sizes_full = sizes.reindex(s1_ids_all, fill_value=0)
    result["stage_B_candidate_volume"] = {
        "total_candidates": int(len(cands)),
        "mean": float(sizes_full.mean()), "median": float(sizes_full.median()),
        "p95": float(sizes_full.quantile(0.95)), "p99": float(sizes_full.quantile(0.99)),
        "max": int(sizes_full.max()) if len(sizes_full) else 0,
        "search_space_full": len(s1_ids_all) * len(prep["others"]),
        "search_space_retained": int(len(cands)),
    }
    # source-wise recall (S1->S2 vs S1->S3), country-wise already in bstats via recall_<country>

    if target_n <= 500:
        # too small a sample to safely run full feature/model stage meaningfully beyond this
        pass

    # ---- Stage C: features ----
    t0 = time.time()
    feats = build_features(cands, prep["s1"], prep["others"], prep["emb"])
    feats["label"] = label_pairs(feats, truth)
    timings["features"] = time.time() - t0
    fcols = feature_columns(feats)
    pos = feats[feats["label"] == 1]
    neg = feats[feats["label"] == 0]
    feat_diag = {}
    for c in fcols:
        pv = pos[c].astype(float)
        nv = neg[c].astype(float)
        missing_rate = float(feats[c].isna().mean())
        pv_f = pv.dropna(); nv_f = nv.dropna()
        pooled_std = np.sqrt((pv_f.var(ddof=1) + nv_f.var(ddof=1)) / 2) if len(pv_f) > 1 and len(nv_f) > 1 else np.nan
        smd = float((pv_f.mean() - nv_f.mean()) / pooled_std) if pooled_std and not np.isnan(pooled_std) and pooled_std > 0 else None
        is_const = bool(feats[c].nunique(dropna=True) <= 1)
        if is_const:
            bucket = "constant/near-constant"
        elif smd is None:
            bucket = "undefined (insufficient variance)"
        elif abs(smd) >= 0.8:
            bucket = "clearly informative"
        elif abs(smd) >= 0.3:
            bucket = "weakly informative"
        else:
            bucket = "highly overlapping"
        feat_diag[c] = {
            "pos_mean": float(pv_f.mean()) if len(pv_f) else None,
            "neg_mean": float(nv_f.mean()) if len(nv_f) else None,
            "pos_median": float(pv_f.median()) if len(pv_f) else None,
            "neg_median": float(nv_f.median()) if len(nv_f) else None,
            "pos_p10": float(pv_f.quantile(0.10)) if len(pv_f) else None,
            "pos_p90": float(pv_f.quantile(0.90)) if len(pv_f) else None,
            "neg_p10": float(nv_f.quantile(0.10)) if len(nv_f) else None,
            "neg_p90": float(nv_f.quantile(0.90)) if len(nv_f) else None,
            "smd": smd, "missing_rate": missing_rate, "is_constant": is_const,
            "bucket": bucket,
        }
    result["stage_C_feature_diagnostics"] = feat_diag
    result["stage_C_meta"] = {"n_pairs": len(feats), "n_features": len(fcols),
                              "positive_rate": float(feats["label"].mean()),
                              "n_positive": int(pos.shape[0]), "n_negative": int(neg.shape[0])}

    # ---- Stage D/E/F: model, scores, threshold sweep ----
    train_ids, valid_ids = split_s1(s1_ids_all, truth)
    in_train = feats["s1_id"].isin(set(train_ids)).to_numpy()
    train_feats = feats[in_train]
    X, y = train_feats[fcols], train_feats["label"].to_numpy()
    t0 = time.time()
    oof, fold_models = train_oof(X, y, train_feats["s1_id"], verbose=False)
    timings["train_oof"] = time.time() - t0
    n_groups = train_feats["s1_id"].nunique()
    result["stage_D_matcher"] = {
        "n_train_pairs": int(len(train_feats)), "n_positive": int(y.sum()),
        "n_negative": int((y == 0).sum()), "positive_rate": float(y.mean()) if len(y) else None,
        "n_groups_s1": int(n_groups), "n_folds": config.N_FOLDS,
        "train_oof_runtime_s": timings["train_oof"],
    }

    scored_oof = train_feats[["s1_id", "cand_id"]].assign(prob=oof, label=y)
    t0 = time.time()
    tau, stau, f05_oof, grid = tune_threshold(scored_oof, truth, train_ids)
    timings["tune_threshold"] = time.time() - t0

    def pct(a):
        if len(a) == 0:
            return {}
        qs = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        return {f"p{q:02d}": float(np.percentile(a, q)) for q in qs}

    result["stage_E_score_analysis"] = {
        "positive_prob_percentiles": pct(oof[y == 1]),
        "negative_prob_percentiles": pct(oof[y == 0]),
    }

    # threshold sweep table (own diagnostic sweep, distinct from decide.py's tune_threshold)
    sweep = []
    for t in config.TAU_GRID:
        pred_pairs = scored_oof[scored_oof["prob"] >= t]
        tp = int((pred_pairs["label"] == 1).sum())
        fp = int((pred_pairs["label"] == 0).sum())
        fn = int(((scored_oof["label"] == 1) & (scored_oof["prob"] < t)).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        b2 = config.BETA ** 2
        f05 = (1 + b2) * precision * recall / (b2 * precision + recall) if (precision + recall) > 0 else 0.0
        pred_map = {}
        for s1id, cid in zip(pred_pairs["s1_id"], pred_pairs["cand_id"]):
            pred_map.setdefault(s1id, []).append(cid)
        n_zero_match_s1 = sum(1 for s in train_ids if not pred_map.get(s))
        n_multi_match_s1 = sum(1 for v in pred_map.values() if len(v) > 1)
        macro = score_report({s: pred_map.get(s, []) for s in train_ids}, truth, train_ids)["macro_f05"]
        sweep.append({"threshold": t, "pair_precision": precision, "pair_recall": recall,
                      "pair_f05": f05, "macro_f05": macro, "n_predicted_pairs": int(len(pred_pairs)),
                      "n_zero_match_s1": n_zero_match_s1, "n_multi_match_s1": n_multi_match_s1})
    result["stage_F_threshold_sweep"] = sweep
    result["stage_F_selected_by_existing_tuning"] = {
        "tau": tau, "singleton_tau": stau, "oof_macro_f05": f05_oof,
    }

    # ---- full model + validation (Stage G, final challenge metrics) ----
    t0 = time.time()
    model = train_full(X, y, fold_models=fold_models)
    timings["train_full"] = time.time() - t0
    valid_feats = feats[~in_train]
    t0 = time.time()
    valid_prob = predict(model, valid_feats[fcols])
    timings["predict_valid"] = time.time() - t0
    scored_valid = valid_feats[["s1_id", "cand_id", "label"]].assign(prob=valid_prob)
    pred = apply_threshold(scored_valid, tau, valid_ids, stau)
    report = score_report(pred, truth, valid_ids, groups=s1_country)
    per_scores = {s: report for s in ()}  # placeholder unused
    result["stage_G_final_challenge_metrics"] = report
    result["stage_G_meta"] = {
        "valid_n_s1": len(valid_ids), "train_n_s1": len(train_ids),
        "candidates_per_s1_valid": float(scored_valid.groupby("s1_id").size().reindex(valid_ids, fill_value=0).mean()),
    }

    # macro F0.5 distribution
    from src.evaluate import per_entity_scores
    scores_map = per_entity_scores(pred, truth, valid_ids)
    arr = np.array(list(scores_map.values()))
    result["stage_G_f05_distribution"] = {
        "mean": float(arr.mean()), "median": float(np.median(arr)),
        "p10": float(np.percentile(arr, 10)), "p25": float(np.percentile(arr, 25)),
        "p75": float(np.percentile(arr, 75)), "p90": float(np.percentile(arr, 90)),
        "pct_zero": float((arr == 0).mean() * 100), "pct_one": float((arr == 1).mean() * 100),
    }

    # ---- Stage 12: error-budget analysis (missed true matches, earliest stage lost) ----
    # Universe: all true pairs for valid_ids.
    true_pairs = [(s, m) for s in valid_ids for m in truth.get(s, [])]
    cand_set = set(zip(scored_valid["s1_id"], scored_valid["cand_id"]))
    pred_set = {(s, c) for s, cs in pred.items() for c in cs}
    # need per-pair prob to know "scored below threshold" vs conflict-removed
    prob_map = {(r.s1_id, r.cand_id): r.prob for r in scored_valid.itertuples(index=False)}
    n_blocking_fn = 0
    n_below_threshold = 0
    n_removed_by_1to1 = 0
    n_other_missing = 0
    for s, m in true_pairs:
        if (s, m) not in cand_set:
            n_blocking_fn += 1
        elif (s, m) not in pred_set:
            p = prob_map.get((s, m))
            if p is not None and p >= tau:
                n_removed_by_1to1 += 1
            else:
                n_below_threshold += 1
        # else: matched correctly
    n_missed_total = n_blocking_fn + n_below_threshold + n_removed_by_1to1 + n_other_missing
    result["stage_12_error_budget"] = {
        "n_true_pairs_valid": len(true_pairs),
        "n_missed_total": n_missed_total,
        "blocking_false_negative": n_blocking_fn,
        "scored_below_threshold": n_below_threshold,
        "removed_by_one_to_one": n_removed_by_1to1,
        "pct_blocking_fn": 100.0 * n_blocking_fn / max(n_missed_total, 1),
        "pct_below_threshold": 100.0 * n_below_threshold / max(n_missed_total, 1),
        "pct_removed_by_1to1": 100.0 * n_removed_by_1to1 / max(n_missed_total, 1),
    }

    # ---- Stage 13: one-to-one / assignment analysis ----
    cand_owner_count = scored_valid.groupby("cand_id")["s1_id"].nunique()
    n_candidate_conflicts = int((cand_owner_count > 1).sum())
    pred_owner_count = {}
    for s, cs in pred.items():
        for c in cs:
            pred_owner_count[c] = pred_owner_count.get(c, 0) + 1
    n_assignment_conflicts = sum(1 for v in pred_owner_count.values() if v > 1)
    result["stage_13_one_to_one"] = {
        "n_candidates_with_multiple_s1_owners": n_candidate_conflicts,
        "n_final_assignment_conflicts": n_assignment_conflicts,
        "n_true_matches_removed_by_1to1_constraint": n_removed_by_1to1,
    }

    timings["total"] = time.time() - t_all0
    result["stage_14_timings_s"] = timings
    result["stage_14_gpu_peak_mb_overall"] = gpu_peak_mb

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"WROTE {out_path}")
    print(f"n_s1={len(s1_s)} blocking pair_recall={bstats.get('pair_recall'):.4f} "
          f"s1_full_recall={bstats.get('s1_full_recall'):.4f} "
          f"mean_cands={bstats.get('mean_candidates'):.2f} "
          f"final macro_f05={report['macro_f05']:.4f} total_time={timings['total']:.1f}s")


if __name__ == "__main__":
    main()
