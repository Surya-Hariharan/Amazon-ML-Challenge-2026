"""CLI for roadmap-v3 Phase-0 diagnostics: baseline, LOCO, drift, sample-size
convergence, staged resource qualification, and error decomposition.

This module only *calls* the existing pipeline (``run_pipeline``, ``blocking``,
``features``, ``model``, ``decide``) through its existing public functions, in
the same order and with the same arguments those functions already use
elsewhere. It changes no normalisation, blocking, feature, model, threshold,
one-to-one or submission behaviour -- it only wraps those calls with
measurement (env/hash/resource logging, distribution comparison, and failure-
stage classification) and writes its own diagnostic artifacts under
``artifacts/diagnostics/``.

Usage (from code/business_entity_resolution/)::

    python -m src.diagnose baseline [--sample 0.1] [--loco] [--no-embeddings]
    python -m src.diagnose drift [--sample 0.1] [--with-candidates] [--no-embeddings]
    python -m src.diagnose convergence --sizes 2000 5000 10000 25000 [--no-embeddings]
    python -m src.diagnose resource-stage --stage normalize|embed|block|features|all
    python -m src.diagnose errors [--sample 0.1] [--no-embeddings]
    python -m src.diagnose blocking-ablation [--sample 0.0045] [--no-embeddings]
    python -m src.diagnose blocking-k-downstream [--sample 0.0045] [--no-embeddings]
    python -m src.diagnose chunk-equality [--sample 0.0045]
    python -m src.diagnose fp16-retrieval [--sample 0.0045]
    python -m src.diagnose decision-validation [--sample 0.0045] [--no-embeddings]
    python -m src.diagnose loco [--sample 0.0045] [--no-embeddings]
    python -m src.diagnose feature-hard-negative [--sample 0.0045] [--no-embeddings]
    python -m src.diagnose blocking-fn-attribution [--sample 0.0045] [--no-embeddings]
    python -m src.diagnose blocking-fn-forensics [--sample 0.0045] [--no-embeddings]

Every subcommand is measurement-only: none of them changes ``output/``, and
none of them is wired to run automatically -- each is a separate, explicit CLI
invocation, deliberately left for the user to trigger.
"""

from __future__ import annotations

import argparse
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

from . import config
from . import diagnostics as diag
from . import drift as drift_mod
from . import error_decomposition as edecomp
from . import run_pipeline as rp
from .blocking import (address_pass, country_groups, dense_topk, digit_token_pass,
                       embedding_pass, generate_candidates, rare_token_pass,
                       report_blocking_stats, tfidf_name_pass)
from .decide import apply_threshold, assign_one_to_one, tune_threshold
from .evaluate import blocking_recall, score_report
from .features import (FeatureContext, build_features, feature_columns, label_pairs,
                       pair_features)
from .io_utils import load_source, load_split
from .model import feature_importance, group_folds, predict, train_full, train_oof

Encoder = Callable[[list[str]], np.ndarray]

DIAG_DIR = config.ARTIFACTS_DIR / "diagnostics"


def _log(msg: str) -> None:
    """Print a timestamped progress line."""
    print(f"[diagnose {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --- E0a/E0b/E0c: baseline + LOCO -----------------------------------------------------

def run_baseline(sample: float = 1.0, loco: bool = False,
                 use_embeddings: bool = config.USE_EMBEDDINGS) -> dict:
    """E0a/E0b/E0c: baseline reproduction with full env/hash/resource logging.

    Calls the existing, unmodified ``run_pipeline.run_valid`` (which already
    implements LOCO when ``loco=True``, producing ``loco_f05_<country>`` for
    every train country -- e.g. LOCO-US and LOCO-India). Wraps that call with
    :class:`diagnostics.ResourceTracker`, environment info, and dataset file
    hashes, and writes a sidecar JSON report (this does not change
    ``experiments.csv``'s existing schema, written unmodified by
    ``run_pipeline.log_experiment`` as it already does).
    """
    hashes = diag.dataset_file_hashes(config.TRAIN_FILES)
    env = diag.env_info()
    with diag.ResourceTracker() as rt:
        metrics = rp.run_valid("all", sample=sample, loco=loco, use_embeddings=use_embeddings)
    report = {
        "kind": "baseline", "sample": sample, "loco": loco,
        "dataset_file_hashes": hashes, "env": env, "metrics": metrics,
        **rt.report,
    }
    ts = time.strftime("%Y%m%d_%H%M%S")
    diag.save_json(report, DIAG_DIR / f"baseline_{ts}.json")
    _log(f"baseline report written to {DIAG_DIR / f'baseline_{ts}.json'}")
    return report


# --- E-drift ---------------------------------------------------------------------------

def run_drift(sample: float = 1.0, with_candidates: bool = False,
             use_embeddings: bool = config.USE_EMBEDDINGS) -> dict[str, pd.DataFrame]:
    """E-drift: label-free train-vs-test covariate distribution comparison.

    Loads only S1 from both splits when ``with_candidates=False`` (the plain
    covariate comparison never reads S2/S3) -- ``load_split`` would otherwise
    materialise every S2/S3 record of both splits (~20M unused raw rows at
    full scale) just to leave them unread for the entire call. S1+S2+S3 are
    only loaded (via :func:`io_utils.load_split`) when ``with_candidates``
    actually needs them for blocking.

    Computes :func:`drift.record_covariates`, and reports
    :func:`drift.compare_distributions` plus :func:`drift.compare_country_share`.
    If ``with_candidates``, also runs blocking on both splits (this is the
    expensive path -- it needs embeddings/candidate generation) and adds
    :func:`drift.candidate_covariates` (candidate-count and top-candidate-
    score distributions) to the comparison. No test label is read at any point.
    """
    if with_candidates:
        train = load_split("train")
        test = load_split("test")
    else:
        train = {"s1": load_source("train", "s1")}
        test = {"s1": load_source("test", "s1")}
    if sample < 1.0:
        train["s1"] = train["s1"].sample(frac=sample, random_state=config.SEED)

    tr_cov = drift_mod.record_covariates(train["s1"])
    te_cov = drift_mod.record_covariates(test["s1"])

    if with_candidates:
        prep_tr = rp.prepare(train["s1"], train["s2"], train["s3"], use_embeddings)
        prep_te = rp.prepare(test["s1"], test["s2"], test["s3"], use_embeddings)
        cands_tr = rp.block(prep_tr)
        cands_te = rp.block(prep_te)
        cc_tr = drift_mod.candidate_covariates(cands_tr, list(prep_tr["s1"][config.ID_COL]))
        cc_te = drift_mod.candidate_covariates(cands_te, list(prep_te["s1"][config.ID_COL]))
        tr_cov = tr_cov.merge(cc_tr, on="entity_id", how="left")
        te_cov = te_cov.merge(cc_te, on="entity_id", how="left")

    report = drift_mod.compare_distributions(tr_cov, te_cov)
    country_share = drift_mod.compare_country_share(tr_cov, te_cov)
    ts = time.strftime("%Y%m%d_%H%M%S")
    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    report.to_csv(DIAG_DIR / f"drift_distributions_{ts}.tsv", sep="\t", index=False)
    country_share.to_csv(DIAG_DIR / f"drift_country_share_{ts}.tsv", sep="\t", index=False)
    _log(f"drift report written under {DIAG_DIR} (drift_distributions_{ts}.tsv, "
        f"drift_country_share_{ts}.tsv)")
    return {"distributions": report, "country_share": country_share}


# --- E-convergence -----------------------------------------------------------------------

def run_convergence(sizes: list[int], use_embeddings: bool = config.USE_EMBEDDINGS,
                    loco: bool = False) -> pd.DataFrame:
    """E-convergence: re-run validation at increasing S1 sample sizes.

    Loads the full training split once, then for each ``N`` in ``sizes``
    (each converted to the equivalent ``TRAIN_SAMPLE_FRAC`` via
    ``N / len(s1)``, clipped to 1.0) calls ``run_pipeline.valid_run`` on
    ``run_pipeline.subsample_train``'s output for that fraction -- i.e. every
    other setting (config, architecture, seed) is held fixed, exactly as
    roadmap v3 requires. Returns one row per size with pair recall, S1 full
    recall, mean candidates, total candidate pairs, precision, recall, macro
    F0.5 and runtime; also written to ``artifacts/diagnostics/convergence_<ts>.tsv``.

    Two measurement caveats, worth reading before comparing rows:

    * Each ``N``'s subsample is an *independent* random draw from the full
      S1 set (``subsample_train`` calls ``.sample(frac=..., random_state=
      config.SEED)`` fresh per ``N``). Pandas' ``.sample()`` does not
      guarantee that a larger ``frac``'s sample is a superset of a smaller
      ``frac``'s sample even with the same seed, so a non-monotonic wobble
      between two ``N`` values can be sampling-composition variance, not
      "instability from more data" -- it is not a nested/telescoping series.
    * ``peak_rss_bytes`` in each row is a *process-lifetime* high-water mark
      (see :class:`diagnostics.ResourceTracker`) and this function measures
      every ``N`` inside one process, in a loop -- so a later row's
      ``peak_rss_bytes`` can be inflated by an earlier row's memory use, not
      isolated to that row alone. ``current_rss_bytes`` (a genuine snapshot,
      not a running maximum) is the less-contaminated column to compare
      across rows.
    """
    s1, s2, s3, truth = rp._load_train(1.0)
    rows = []
    for n in sizes:
        frac = min(1.0, n / max(len(s1), 1))
        a1, a2, a3, at = rp.subsample_train(s1, s2, s3, truth, frac)
        _log(f"convergence: N={n} (frac={frac:.4f}, actual S1={len(a1)})")
        with diag.ResourceTracker() as rt:
            m = rp.valid_run(a1, a2, a3, at, stage="all", loco=loco,
                             use_embeddings=use_embeddings, use_cache=False)
        rows.append({
            "requested_n": n, "actual_n_s1": len(a1),
            "block_pair_recall": m.get("block_pair_recall"),
            "block_s1_full_recall": m.get("block_s1_full_recall"),
            "block_mean_candidates": m.get("block_mean_candidates"),
            "block_n_pairs": m.get("block_n_pairs"),
            "valid_pair_precision": m.get("valid_pair_precision"),
            "valid_pair_recall": m.get("valid_pair_recall"),
            "valid_macro_f05": m.get("valid_macro_f05"),
            "tau": m.get("tau"), **rt.report,
        })
    out = pd.DataFrame(rows)
    ts = time.strftime("%Y%m%d_%H%M%S")
    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(DIAG_DIR / f"convergence_{ts}.tsv", sep="\t", index=False)
    _log(f"convergence report written to {DIAG_DIR / f'convergence_{ts}.tsv'}")
    return out


# --- E2: staged resource qualification ---------------------------------------------------

def _warn_if_missing(glob: str, this_stage: str, prior_stage: str) -> None:
    """Log a warning if a prior stage's expected cache artifacts are absent.

    Each staged function below assumes the prior stage already ran and left
    its cache on disk, so its own measured window covers only its own work.
    If that assumption doesn't hold (e.g. a clean ``artifacts/`` directory,
    or stages run out of order), the missing prior stage's normalisation/
    embedding/blocking work happens silently *inside* this stage's measured
    time/memory instead -- this warning makes that mislabelling visible in
    the log rather than letting the report imply a number it doesn't mean.
    """
    if not list(config.ARTIFACTS_DIR.glob(glob)):
        _log(f"WARNING: no '{glob}' cache found -- stage '{this_stage}' will also "
            f"perform (and be charged for) stage '{prior_stage}''s work, since its "
            f"prerequisite cache is missing")


def _stage_normalize() -> dict:
    """Stage 1: full-scale sequential normalisation of the configured train files."""
    def fn():
        rp.prepare_from_files(config.TRAIN_FILES, use_embeddings=False, use_cache=True)
    artifacts = list((config.ARTIFACTS_DIR).glob("norm_*.parquet"))
    return diag.run_stage("normalize", fn, artifacts, DIAG_DIR)


def _stage_embed() -> dict:
    """Stage 2: full-scale embedding generation (requires stage 1's normalised cache)."""
    _warn_if_missing("norm_*.parquet", "embed", "normalize")

    def fn():
        rp.prepare_from_files(config.TRAIN_FILES, use_embeddings=True, use_cache=True)
    artifacts = list((config.ARTIFACTS_DIR).glob("embf32_*.npy"))
    return diag.run_stage("embed", fn, artifacts, DIAG_DIR)


def _stage_block() -> dict:
    """Stage 3: full-scale blocking/dense retrieval (requires stages 1-2's caches)."""
    _warn_if_missing("norm_*.parquet", "block", "normalize")
    _warn_if_missing("embf32_*.npy", "block", "embed")

    def fn():
        prep = rp.prepare_from_files(config.TRAIN_FILES, use_embeddings=True, use_cache=True)
        rp.block(prep, use_cache=True)
    artifacts = list((config.ARTIFACTS_DIR).glob("cands_*.parquet"))
    return diag.run_stage("block", fn, artifacts, DIAG_DIR)


def _stage_features() -> dict:
    """Stage 4: full-scale feature generation (requires stages 1-3's caches)."""
    _warn_if_missing("norm_*.parquet", "features", "normalize")
    _warn_if_missing("embf32_*.npy", "features", "embed")
    _warn_if_missing("cands_*.parquet", "features", "block")

    def fn():
        prep = rp.prepare_from_files(config.TRAIN_FILES, use_embeddings=True, use_cache=True)
        cands = rp.block(prep, use_cache=True)
        build_features(cands, prep["s1"], prep["others"], prep["emb"])
    return diag.run_stage("features", fn, (), DIAG_DIR)


_STAGES: dict[str, Callable[[], dict]] = {
    "normalize": _stage_normalize, "embed": _stage_embed,
    "block": _stage_block, "features": _stage_features,
}


def run_resource_stage(stage: str) -> list[dict]:
    """E2: run one (or, for ``"all"``, every) resource-qualification stage.

    Stops after the first failing stage (per roadmap v3 O0: "if a stage
    fails, STOP -- do not redesign the architecture automatically, report the
    exact failure"). Each stage's report is written to
    ``artifacts/diagnostics/stage_<name>_<timestamp>.json`` by
    :func:`diagnostics.run_stage`; this function also prints a one-line
    summary per stage.
    """
    order = list(_STAGES) if stage == "all" else [stage]
    reports = []
    for name in order:
        _log(f"resource qualification: stage '{name}' starting")
        report = _STAGES[name]()
        reports.append(report)
        status = "OK" if report["success"] else "FAILED"
        _log(f"resource qualification: stage '{name}' {status} "
            f"(runtime={report.get('runtime_s')}, peak_rss={report.get('peak_rss_bytes')}, "
            f"peak_gpu={report.get('peak_gpu_bytes')})")
        if not report["success"]:
            _log(f"STOPPING: stage '{name}' failed -- {report.get('error')}")
            break
    return reports


# --- E1: error decomposition -------------------------------------------------------------

def run_valid_for_diagnostics(
    s1: pd.DataFrame, s2: pd.DataFrame, s3: pd.DataFrame, truth: dict[str, list[str]],
    use_embeddings: bool = config.USE_EMBEDDINGS, use_cache: bool = True,
) -> dict:
    """Reproduce ``run_pipeline.valid_run``'s computation, exposing intermediates.

    Calls the exact same public functions valid_run uses, in the same order
    and with the same arguments, so results are identical to a real
    ``--mode valid`` run -- this exists only because ``valid_run`` returns a
    flat metrics dict, and error decomposition needs the intermediate
    objects (blocking output, scored predictions, the final decision dict,
    tau) that dict summarises. This function does not modify
    ``run_pipeline.py`` and calls nothing that isn't already public there.
    """
    prep = rp.prepare(s1, s2, s3, use_embeddings, use_cache=use_cache)
    s1_ids = list(prep["s1"][config.ID_COL])
    cands = rp.block(prep, use_cache)
    feats = build_features(cands, prep["s1"], prep["others"], prep["emb"])
    feats["label"] = label_pairs(feats, truth)
    train_ids, valid_ids = rp.split_s1(s1_ids, truth)
    in_train = feats["s1_id"].isin(set(train_ids)).to_numpy()
    fitted = rp.fit_and_tune(feats[in_train], truth, train_ids)
    valid_feats = feats[~in_train]
    scored = valid_feats[["s1_id", "cand_id", "label"]].assign(
        prob=predict(fitted["model"], valid_feats[feature_columns(feats)]))
    pred = apply_threshold(scored, fitted["tau"], valid_ids, fitted["singleton_tau"])
    return {
        "prep": prep, "cands": cands, "feats": feats, "scored": scored, "pred": pred,
        "tau": fitted["tau"], "singleton_tau": fitted["singleton_tau"],
        "valid_ids": valid_ids, "train_ids": train_ids, "fitted": fitted,
    }


def run_error_decomposition(
    sample: float = 1.0, use_embeddings: bool = config.USE_EMBEDDINGS
) -> dict[str, pd.DataFrame]:
    """E1: build and persist the full error-decomposition report on a validation split.

    Writes ``classified.tsv`` (per-true-pair stage classification, plus
    representation-loss flags), ``false_positives.tsv``, ``multi_match.tsv``,
    ``one_to_one_removals.tsv`` and ``slices.tsv`` under
    ``artifacts/diagnostics/errors_<timestamp>/``.
    """
    s1, s2, s3, truth = rp._load_train(sample)
    ctx = run_valid_for_diagnostics(s1, s2, s3, truth, use_embeddings)
    valid_truth = {s: truth.get(s, []) for s in ctx["valid_ids"]}
    valid_scored = ctx["scored"]
    s1_country = dict(zip(ctx["prep"]["s1"][config.ID_COL],
                          ctx["prep"]["s1"][config.COUNTRY_COL]))

    classified = edecomp.classify_true_pairs(
        ctx["prep"]["s1"], ctx["prep"]["others"], ctx["cands"], valid_scored,
        ctx["pred"], valid_truth, ctx["tau"])
    fp = edecomp.false_positive_report(valid_scored, ctx["pred"], valid_truth)
    mm = edecomp.multi_match_report(ctx["pred"], valid_truth)
    o2o = edecomp.one_to_one_removals(valid_scored, valid_truth)
    slices = edecomp.slice_report(classified, ctx["prep"]["s1"], ctx["prep"]["others"],
                                  ctx["cands"], s1_country)

    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = DIAG_DIR / f"errors_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in (("classified", classified), ("false_positives", fp),
                     ("multi_match", mm), ("one_to_one_removals", o2o), ("slices", slices)):
        df.to_csv(out_dir / f"{name}.tsv", sep="\t", index=False)
    stage_counts = classified["stage"].value_counts(normalize=True) if len(classified) else pd.Series()
    _log(f"error decomposition written under {out_dir}")
    _log(f"stage shares: {stage_counts.to_dict()}")
    return {"classified": classified, "false_positives": fp, "multi_match": mm,
           "one_to_one_removals": o2o, "slices": slices}


# --- P1: blocking-K ablation ---------------------------------------------------------------

#: One-factor-at-a-time sweep values per blocking pass (production baseline is
#: config.K_TFIDF_NAME / K_EMBEDDING / K_RARE_TOKEN / K_POSTAL_TOKEN / K_ADDRESS).
#: Keys match blocking.PASS_BITS / generate_candidates' k_overrides.
_ABLATION_SWEEPS: dict[str, list[int]] = {
    "tfidf": [10, 20, 40],
    "embed": [10, 20, 40],
    "rare": [5, 10, 20],
    "digit": [5, 10, 20],
    "address": [5, 10, 20],
}


def blocking_ablation_baseline() -> dict[str, int]:
    """Production blocking-K baseline, read live from ``config`` (never hard-coded).

    Reading these values here rather than duplicating literals means a future
    change to a ``config.K_*`` default is picked up automatically and never
    silently drifts out of sync with what ``generate_candidates`` actually
    runs in production.
    """
    return {
        "tfidf": config.K_TFIDF_NAME, "embed": config.K_EMBEDDING,
        "rare": config.K_RARE_TOKEN, "digit": config.K_POSTAL_TOKEN,
        "address": config.K_ADDRESS,
    }


def blocking_ablation_configs() -> list[dict]:
    """16 one-factor-at-a-time blocking-K configurations: 1 baseline + 5x3 sweeps.

    Each non-baseline entry copies the baseline dict and changes exactly one
    pass's k to one of ``_ABLATION_SWEEPS[pass]``'s three values (including,
    for completeness, the value that already equals the baseline -- so every
    listed value is run as its own named configuration, matching the
    requested 1 + 5*3 = 16 total exactly). Each dict has ``"name"`` plus one
    key per pass (``tfidf``/``embed``/``rare``/``digit``/``address``); this
    is the exact shape ``generate_candidates(..., k_overrides=...)`` expects
    once ``"name"`` is stripped.
    """
    baseline = blocking_ablation_baseline()
    configs = [{"name": "baseline", **baseline}]
    for factor, values in _ABLATION_SWEEPS.items():
        for v in values:
            cfg = dict(baseline)
            cfg[factor] = v
            configs.append({"name": f"{factor}_{v}", **cfg})
    return configs


def run_blocking_ablation(
    sample: float = 0.0045, use_embeddings: bool = True
) -> pd.DataFrame:
    """P1: cheap blocking-only screening sweep, one K at a time.

    Loads the training split once (via ``run_pipeline._load_train``, the same
    helper ``errors``/``convergence`` use) and normalises + embeds it once
    (via the existing, unmodified ``run_pipeline.prepare``, with its normal
    on-disk normalisation/embedding caches, per CLAUDE.md 7 -- embeddings are
    never recomputed per configuration). For every configuration in
    :func:`blocking_ablation_configs`, calls the existing, unmodified
    ``blocking.generate_candidates`` -- the same union/de-duplication code
    path production blocking uses -- passing that configuration's k values
    through its ``k_overrides`` parameter; ``config.py``'s production K
    defaults are never read for anything but the baseline row and are never
    mutated. Scores every configuration with the existing, unmodified
    ``blocking.report_blocking_stats`` (pair recall, S1 full recall, mean/max
    candidates, n_pairs, reduction ratio, per-pass recall, per-country
    recall). No feature generation, model training, threshold tuning or OOF
    happens anywhere in this path, and ``output/`` and ``run_pipeline.py``'s
    behaviour are untouched.

    Writes one row per configuration to
    ``artifacts/diagnostics/blocking_ablation_<timestamp>.tsv`` and a sidecar
    ``blocking_ablation_<timestamp>.json`` with the configuration matrix and
    environment info (mirroring ``run_baseline``'s report shape). Returns the
    same frame as a :class:`pandas.DataFrame`.
    """
    s1, s2, s3, truth = rp._load_train(sample)
    prep = rp.prepare(s1, s2, s3, use_embeddings=use_embeddings, use_cache=True)
    s1_ids = list(prep["s1"][config.ID_COL])
    s1_country = dict(zip(prep["s1"][config.ID_COL], prep["s1"][config.COUNTRY_COL]))
    n_other = len(prep["others"])
    configs = blocking_ablation_configs()
    _log(f"blocking ablation: {len(configs)} configurations, "
        f"{len(s1_ids):,} S1 x {n_other:,} others (sample={sample})")

    rows = []
    for cfg in configs:
        name = cfg["name"]
        k_overrides = {k: v for k, v in cfg.items() if k != "name"}
        _log(f"blocking ablation: running '{name}' (k={k_overrides})")
        with diag.ResourceTracker() as rt:
            cands = generate_candidates(
                prep["s1"], prep["others"], embeddings=prep["emb"],
                verbose=False, k_overrides=k_overrides)
        stats = report_blocking_stats(cands, truth, s1_ids, n_other, s1_country,
                                      verbose=False)
        row = {
            "experiment": name, **{f"k_{k}": v for k, v in k_overrides.items()},
            "sample": sample, "n_s1": len(s1_ids), "n_other": n_other,
            **stats,
            "runtime_seconds": rt.report["runtime_s"],
            "peak_rss_bytes": rt.report["peak_rss_bytes"],
            "current_rss_bytes": rt.report["current_rss_bytes"],
        }
        rows.append(row)
        _log(f"blocking ablation: '{name}' pair_recall={stats['pair_recall']:.4f} "
            f"mean_candidates={stats['mean_candidates']:.2f} "
            f"runtime={row['runtime_seconds']:.1f}s")

    out = pd.DataFrame(rows)
    ts = time.strftime("%Y%m%d_%H%M%S")
    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DIAG_DIR / f"blocking_ablation_{ts}.tsv"
    out.to_csv(out_path, sep="\t", index=False)
    meta = {
        "kind": "blocking_ablation", "sample": sample, "use_embeddings": use_embeddings,
        "n_s1": len(s1_ids), "n_other": n_other, "configs": configs,
        "dataset_file_hashes": diag.dataset_file_hashes(config.TRAIN_FILES),
        "env": diag.env_info(),
    }
    diag.save_json(meta, DIAG_DIR / f"blocking_ablation_{ts}.json")
    _log(f"blocking ablation report written to {out_path}")
    return out


# --- P1-downstream: blocking-K OOF validation ---------------------------------------------

#: The 4 Pareto-competitive configurations P1's blocking-only screening surfaced
#: (roadmap v3 P1 follow-up). Unlike ``_ABLATION_SWEEPS`` (16 one-factor-at-a-time
#: blocking-only rows), this is a small, fixed, *named* set run through the full
#: downstream pipeline (features -> LightGBM -> threshold -> decision), in this
#: exact order, because that ordering is part of the deliverable.
_DOWNSTREAM_ORDER = ("baseline", "tfidf40", "embed40", "both40")


def blocking_k_downstream_configs() -> list[dict]:
    """The 4 named blocking-K configurations for the P1-downstream OOF experiment.

    Reads the production baseline live from ``config`` (via
    :func:`blocking_ablation_baseline`, never duplicated as literals), then applies
    exactly the K changes P1's blocking-only screening reported as Pareto-competitive:
    ``tfidf40`` raises only TF-IDF K to 40, ``embed40`` raises only embedding K to 40,
    ``both40`` raises both. ``rare``/``digit``/``address`` K stay at the production
    baseline in every configuration -- the only experimental variable across these 4
    rows is TF-IDF/embedding K, per the task's methodological requirement. Returned in
    the deterministic order (baseline, tfidf40, embed40, both40) the experiment must
    run in.
    """
    baseline = blocking_ablation_baseline()
    by_name = {
        "baseline": dict(baseline),
        "tfidf40": {**baseline, "tfidf": 40},
        "embed40": {**baseline, "embed": 40},
        "both40": {**baseline, "tfidf": 40, "embed": 40},
    }
    return [{"name": n, **by_name[n]} for n in _DOWNSTREAM_ORDER]


def _fit_and_tune_with_oof(feats: pd.DataFrame, truth: dict[str, list[str]],
                           s1_ids: list[str]) -> dict:
    """Reproduce ``run_pipeline.fit_and_tune``'s exact computation, additionally
    exposing the OOF probabilities and GroupKFold fold id per row.

    ``fit_and_tune`` itself only returns a flat summary (model, tau, singleton_tau,
    oof_f05, grid, fold_models) -- fold-stability reporting needs the per-row OOF
    predictions and which fold each row was held out in, so this calls the exact same
    public functions (``model.group_folds``, ``model.train_oof``,
    ``decide.tune_threshold``, ``model.train_full``), in the same order, with the same
    arguments ``fit_and_tune`` uses, purely to additionally capture those
    intermediates. This changes no training, tuning or fold-assignment behaviour and
    does not modify ``model.py``/``run_pipeline.py`` -- ``model.group_folds`` is
    deterministic (sklearn ``GroupKFold``), so calling it here on the same ``groups``
    ``train_oof`` uses internally reproduces the identical fold assignment.
    """
    cols = feature_columns(feats)
    X, y, groups = feats[cols], feats["label"].to_numpy(), feats["s1_id"]
    folds = group_folds(groups, config.N_FOLDS)
    oof, fold_models = train_oof(X, y, groups, verbose=False)
    scored = feats[["s1_id", "cand_id"]].assign(prob=oof)
    tau, stau, f05, grid = tune_threshold(scored, truth, s1_ids)
    model = train_full(X, y, fold_models=fold_models)
    return {"model": model, "tau": tau, "singleton_tau": stau, "oof_f05": f05,
            "grid": grid, "fold_models": fold_models, "oof": oof, "folds": folds,
            "scored": scored}


def _fold_stability(scored: pd.DataFrame, folds: np.ndarray, tau: float,
                    singleton_tau: float | None, truth: dict[str, list[str]],
                    one_to_one: bool = config.ONE_TO_ONE,
                    decide_fn: Callable[..., dict] | None = None) -> dict:
    """Per-fold macro F0.5 from OOF predictions, using the one tuned threshold pair.

    Each GroupKFold fold holds a disjoint set of S1 groups (``model.group_folds``'
    contract), so ``scored`` rows in fold ``f`` are exactly the out-of-fold
    predictions for the S1s validated in that fold -- unbiased, the same way the
    headline OOF macro F0.5 is. The single ``(tau, singleton_tau)`` from the full OOF
    sweep is reused for every fold (task requirement 10: no per-fold threshold
    re-tuning), so this measures how stable the *chosen* operating point is across
    folds, not how well each fold could have done with its own threshold.

    ``one_to_one`` (used only when ``decide_fn`` is not given) and ``decide_fn`` let
    P3 reuse this exact per-fold-stability computation for one-to-one ON/OFF
    (:func:`decide.apply_threshold` with the requested flag) and for the P3-B
    decision-order comparison (an arbitrary ``(scored, tau, s1_ids, singleton_tau)
    -> pred`` callable, e.g. :func:`decide_threshold_then_one_to_one`) without
    duplicating the fold-slicing loop. Existing callers that pass neither argument
    get exactly the prior behaviour (production ``apply_threshold`` with
    ``one_to_one=True``).
    """
    s1_col = scored["s1_id"].to_numpy()
    per_fold: list[float] = []
    for f in sorted(set(int(v) for v in folds) - {-1}):
        mask = folds == f
        fold_s1 = sorted(set(s1_col[mask]))
        if not fold_s1:
            continue
        if decide_fn is not None:
            pred = decide_fn(scored[mask], tau, fold_s1, singleton_tau)
        else:
            pred = apply_threshold(scored[mask], tau, fold_s1, singleton_tau, one_to_one=one_to_one)
        per_fold.append(score_report(pred, truth, fold_s1)["macro_f05"])
    arr = np.array(per_fold, dtype=float)
    return {
        "fold_f05": per_fold,
        "fold_f05_mean": float(arr.mean()) if len(arr) else float("nan"),
        "fold_f05_std": float(arr.std(ddof=0)) if len(arr) else float("nan"),
    }


def run_blocking_k_downstream(sample: float = 0.0045, use_embeddings: bool = True) -> pd.DataFrame:
    """P1-downstream: full blocking->features->LightGBM->threshold->decision OOF
    validation for the 4 Pareto-competitive blocking-K configurations P1's cheap
    blocking-only screening surfaced (``blocking_k_downstream_configs``).

    P1's ``blocking-ablation`` only measures blocking recall; it cannot say whether
    the extra candidates it buys actually improve the complete pipeline's macro F0.5,
    since more candidates also means more opportunities for the matcher to produce a
    false positive. This closes that gap by running each configuration all the way
    through -- exactly the existing, unmodified pipeline functions
    ``run_pipeline.prepare``, ``blocking.generate_candidates``,
    ``blocking.report_blocking_stats``, ``features.build_features``,
    ``features.label_pairs``, the GroupKFold OOF machinery in ``model.py``,
    ``decide.tune_threshold``/``apply_threshold`` and ``evaluate.score_report`` -- in
    the same order and with the same arguments production ``run_pipeline.valid_run``
    uses, so the only experimental variable across the 4 rows is blocking K.

    Normalisation and embeddings are computed exactly once (via ``run_pipeline.prepare``
    with its normal on-disk caches), then reused for every configuration by calling
    ``blocking.generate_candidates`` directly with that configuration's ``k_overrides``
    -- mirroring ``run_blocking_ablation``'s cache-reuse strategy. The
    train/valid S1 split (``run_pipeline.split_s1``) is computed once, before the
    configuration loop, and reused identically for every configuration, so a
    difference between rows can never be attributed to a different split.

    Each configuration's ``cands`` frame is used, unmodified, for both its blocking
    stats *and* its feature generation -- the same candidate set that is measured is
    the one the matcher scores (task requirement 8). If a configuration raises, its
    row is recorded with ``status="failed"`` and the exception message, and the
    remaining configurations still run (task requirement 12) -- nothing here silently
    swallows a failure or skips a configuration without a row.

    Writes one row per configuration to
    ``artifacts/diagnostics/blocking_k_downstream_<timestamp>.tsv`` and a sidecar
    ``blocking_k_downstream_<timestamp>.json`` (git commit, dataset hashes, env info,
    the exact configuration matrix, and any failures) mirroring
    ``run_blocking_ablation``'s report shape. Returns the same frame. Does not touch
    ``output/``, ``run_pipeline.py`` or any ``config.py`` default -- it only ever
    calls ``blocking.generate_candidates`` with an explicit ``k_overrides`` dict built
    from configuration values read live off ``config`` at call time.
    """
    s1, s2, s3, truth = rp._load_train(sample)
    prep = rp.prepare(s1, s2, s3, use_embeddings=use_embeddings, use_cache=True)
    s1_ids = list(prep["s1"][config.ID_COL])
    s1_country = dict(zip(prep["s1"][config.ID_COL], prep["s1"][config.COUNTRY_COL]))
    n_other = len(prep["others"])
    train_ids, valid_ids = rp.split_s1(s1_ids, truth)
    configs = blocking_k_downstream_configs()
    _log(f"blocking-K downstream: {len(configs)} configurations, "
        f"{len(s1_ids):,} S1 x {n_other:,} others (sample={sample})")

    rows: list[dict] = []
    failures: list[str] = []
    for cfg in configs:
        name = cfg["name"]
        k_overrides = {k: v for k, v in cfg.items() if k != "name"}
        _log(f"blocking-K downstream: running '{name}' (k={k_overrides})")
        try:
            with diag.ResourceTracker() as rt:
                cands = generate_candidates(
                    prep["s1"], prep["others"], embeddings=prep["emb"],
                    verbose=False, k_overrides=k_overrides)
                block_stats = report_blocking_stats(
                    cands, truth, s1_ids, n_other, s1_country, verbose=False)
                feats = build_features(cands, prep["s1"], prep["others"], prep["emb"])
                feats["label"] = label_pairs(feats, truth)
                in_train = feats["s1_id"].isin(set(train_ids)).to_numpy()
                feats_train = feats[in_train]
                fit = _fit_and_tune_with_oof(feats_train, truth, train_ids)
                stability = _fold_stability(fit["scored"], fit["folds"], fit["tau"],
                                            fit["singleton_tau"], truth)
                valid_feats = feats[~in_train]
                valid_scored = valid_feats[["s1_id", "cand_id"]].assign(
                    prob=predict(fit["model"], valid_feats[feature_columns(feats)]))
                pred = apply_threshold(valid_scored, fit["tau"], valid_ids, fit["singleton_tau"])
                report = score_report(pred, truth, valid_ids, groups=s1_country)
            row = {
                "experiment": name, "status": "ok",
                **{f"k_{k}": v for k, v in k_overrides.items()},
                "sample": sample, "n_s1": len(s1_ids), "n_other": n_other,
                **{f"block_{k}": v for k, v in block_stats.items()},
                "n_feature_rows": len(feats),
                "oof_macro_f05": fit["oof_f05"],
                "tau": fit["tau"], "singleton_tau": fit["singleton_tau"],
                **{f"valid_{k}": v for k, v in report.items()},
                "fold_f05": ";".join(f"{v:.4f}" for v in stability["fold_f05"]),
                "fold_f05_mean": stability["fold_f05_mean"],
                "fold_f05_std": stability["fold_f05_std"],
                "runtime_seconds": rt.report["runtime_s"],
                "peak_rss_bytes": rt.report["peak_rss_bytes"],
                "current_rss_bytes": rt.report["current_rss_bytes"],
            }
            _log(f"blocking-K downstream: '{name}' OOF macro F0.5={fit['oof_f05']:.4f} "
                f"valid macro F0.5={report['macro_f05']:.4f} "
                f"(fold mean={stability['fold_f05_mean']:.4f} "
                f"std={stability['fold_f05_std']:.4f})")
        except Exception as exc:  # noqa: BLE001 -- one failing config must not stop the rest
            row = {
                "experiment": name, "status": "failed", "error": str(exc),
                **{f"k_{k}": v for k, v in k_overrides.items()},
                "sample": sample, "n_s1": len(s1_ids), "n_other": n_other,
            }
            failures.append(name)
            _log(f"blocking-K downstream: '{name}' FAILED: {exc}")
        rows.append(row)

    out = pd.DataFrame(rows)
    ts = time.strftime("%Y%m%d_%H%M%S")
    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DIAG_DIR / f"blocking_k_downstream_{ts}.tsv"
    out.to_csv(out_path, sep="\t", index=False)
    meta = {
        "kind": "blocking_k_downstream", "timestamp": ts, "sample": sample,
        "use_embeddings": use_embeddings, "n_s1": len(s1_ids), "n_other": n_other,
        "configs": configs, "order": list(_DOWNSTREAM_ORDER), "failures": failures,
        "git_commit": rp._git_hash(),
        "dataset_file_hashes": diag.dataset_file_hashes(config.TRAIN_FILES),
        "env": diag.env_info(),
    }
    diag.save_json(meta, DIAG_DIR / f"blocking_k_downstream_{ts}.json")
    _log(f"blocking-K downstream report written to {out_path}"
        + (f" ({len(failures)} failure(s): {failures})" if failures else ""))
    return out


# --- E3/E4 shared helpers: dense-retrieval candidate comparison ---------------------------

def _dense_candidates_grouped(
    s1: pd.DataFrame, others: pd.DataFrame, embeddings: tuple[np.ndarray, np.ndarray],
    k: int, topk_fn: Callable[[np.ndarray, np.ndarray, int], tuple],
    within_country: bool = config.BLOCK_WITHIN_COUNTRY,
) -> dict[str, set[str]]:
    """``{s1_id: candidate_id set}`` from one dense-embedding top-k pass.

    Groups by country with the exact production primitive (``blocking.country_groups``
    -- the same call ``blocking.generate_candidates`` makes for every pass), so which
    S1/candidate pairs are even eligible to be compared is identical to production.
    The actual top-k search is supplied as ``topk_fn`` so E3 (chunk size) and E4
    (compute dtype) can each plug in a different variant of the retrieval primitive
    while everything else about the comparison -- grouping, ID mapping, candidate-set
    construction -- stays shared and identical between the two experiments.
    """
    s1 = s1.reset_index(drop=True)
    others = others.reset_index(drop=True)
    out: dict[str, set[str]] = {}
    s1_ids_all = s1[config.ID_COL].to_numpy()
    cand_ids_all = others[config.ID_COL].to_numpy()
    for _country, q_idx, x_idx in country_groups(s1, others, within_country):
        if len(x_idx) == 0:
            continue
        q_emb = np.asarray(embeddings[0][q_idx])
        x_emb = np.asarray(embeddings[1][x_idx])
        qi, xi, _s = topk_fn(q_emb, x_emb, k)
        s1g, candg = s1_ids_all[q_idx], cand_ids_all[x_idx]
        for qq, xx in zip(qi, xi):
            out.setdefault(s1g[qq], set()).add(candg[xx])
    return out


def _candidate_set_diff(a: dict[str, set[str]], b: dict[str, set[str]],
                        max_sample: int = 20) -> dict:
    """Exact-equality diff between two ``{s1_id: candidate_id set}`` maps.

    ``a`` is the reference (current-production) side. Returns the total pairs on
    each side, how many S1s have a differing candidate set, the total number of
    pairs present only in ``a`` ("missing" -- lost by switching to ``b``) or only
    in ``b`` ("extra" -- gained by switching to ``b``), whether the two maps are
    byte-for-byte identical, and a small sample (capped at ``max_sample``) of the
    differing S1 IDs for spot-checking -- never the full candidate lists, so the
    report stays a bounded size even at larger sample fractions.
    """
    all_s1 = sorted(set(a) | set(b))
    n_diff_s1 = 0
    n_missing = 0  # in a, not in b
    n_extra = 0  # in b, not in a
    diff_sample: list[str] = []
    for s in all_s1:
        sa, sb = a.get(s, set()), b.get(s, set())
        if sa != sb:
            n_diff_s1 += 1
            n_missing += len(sa - sb)
            n_extra += len(sb - sa)
            if len(diff_sample) < max_sample:
                diff_sample.append(s)
    return {
        "n_s1_total": len(all_s1),
        "n_pairs_a": sum(len(v) for v in a.values()),
        "n_pairs_b": sum(len(v) for v in b.values()),
        "n_s1_differing": n_diff_s1,
        "n_missing_pairs": n_missing,
        "n_extra_pairs": n_extra,
        "exact_equal": n_diff_s1 == 0,
        "differing_s1_sample": diff_sample,
    }


def _cand_stats(cands: dict[str, set[str]], s1_ids: list[str]) -> dict:
    """Mean/max candidates and total pairs over the *full* S1 universe.

    Unlike ``evaluate.blocking_recall``'s own ``mean_candidates`` (which averages
    only over the S1s present as keys in ``cands``), this averages over every S1 in
    ``s1_ids`` -- an S1 with zero candidates counts as 0, matching
    ``blocking.report_blocking_stats``' production semantics.
    """
    sizes = [len(cands.get(s, ())) for s in s1_ids]
    return {
        "mean_candidates": sum(sizes) / len(sizes) if sizes else 0.0,
        "max_candidates": float(max(sizes)) if sizes else 0.0,
        "n_pairs": sum(sizes),
    }


def _true_match_loss(
    cands_a: dict[str, set[str]], cands_b: dict[str, set[str]],
    truth: dict[str, list[str]], s1_ids: list[str],
) -> tuple[int, int, float]:
    """Count true (S1, match) pairs retrieved by ``a`` but lost by ``b``.

    A "true match lost" is a ground-truth pair that side ``a`` (the reference/
    current-precision side) actually retrieved as a candidate, but side ``b`` (the
    lower-precision side) did not -- i.e. an actual retrieval regression, not merely
    "a true pair blocking never found in the first place" (that is a pre-existing
    blocking miss on *both* sides, already covered by each side's own ``pair_recall``,
    and is deliberately not double-counted here). Returns ``(n_true_total,
    n_true_lost, pct_true_lost)`` over every S1 in ``s1_ids`` that has at least one
    true match; singletons (no true matches) contribute nothing to either count.
    """
    n_true_total = 0
    n_true_lost = 0
    for s in s1_ids:
        true_matches = set(truth.get(s, ()))
        if not true_matches:
            continue
        n_true_total += len(true_matches)
        retrieved_a = true_matches & cands_a.get(s, set())
        n_true_lost += len(retrieved_a - cands_b.get(s, set()))
    pct_true_lost = (n_true_lost / n_true_total * 100.0) if n_true_total else 0.0
    return n_true_total, n_true_lost, pct_true_lost


def _prepare_for_dense_diagnostic(
    sample: float, encoder: Encoder | None
) -> tuple[dict, dict[str, list[str]], list[str]]:
    """Shared E3/E4 setup: load + normalise + embed the sample exactly once.

    Calls the existing, unmodified ``run_pipeline._load_train`` and
    ``run_pipeline.prepare`` (``use_embeddings=True``) -- the same normalisation and
    embedding-cache path production uses -- so both experiments compare candidate IDs
    computed from the *same* normalised records and the *same* embeddings, per the
    task's methodology ("same normalization... same embeddings"). Raises if
    embeddings could not be produced, since both E3 and E4 are dense-retrieval-only
    diagnostics with nothing to compare without them.
    """
    s1, s2, s3, truth = rp._load_train(sample)
    prep = rp.prepare(s1, s2, s3, use_embeddings=True, encoder=encoder, use_cache=True)
    if prep["emb"] is None:
        raise RuntimeError("prepare() produced no embeddings; E3/E4 need use_embeddings=True")
    s1_ids = list(prep["s1"][config.ID_COL])
    return prep, truth, s1_ids


# --- E3: dense-retrieval chunk-size equality ------------------------------------------

def chunk_size_configs() -> dict[str, dict]:
    """'current' (live production) and 'larger' chunk configurations for E3.

    Every chunk-size knob affecting embedding generation or dense embedding retrieval
    lives in ``config.py``: ``DENSE_QUERY_CHUNK``/``DENSE_INDEX_CHUNK`` (the query/
    index block sizes ``blocking.dense_topk`` chunks the exact top-k search into) and
    ``EMBEDDING_BATCH`` (the encoder's ``model.encode(batch_size=...)``). ``'larger'``
    doubles each, derived from the live production values rather than invented
    outright: doubling both ``DENSE_QUERY_CHUNK`` and ``DENSE_INDEX_CHUNK`` roughly
    4x's a per-chunk similarity block's element count (query_chunk x index_chunk, see
    ``blocking.dense_topk``'s docstring for that memory model) -- at the production
    defaults (4096 x 262_144) that block is already sized to fit comfortably in a T4's
    16GB VRAM in float16, so doubling both still leaves headroom for the embedding
    model itself; doubling ``EMBEDDING_BATCH`` is the same fewer-larger-calls
    reasoning for the encode() batch.

    NOTE: ``run_chunk_equality`` only exercises ``query_chunk``/``index_chunk`` as the
    experimental variable in its candidate-ID comparison -- per the task's own
    methodology, the *same* precomputed embeddings are reused for both configurations
    (E3 is not re-encoding text twice), so ``embedding_batch`` cannot itself change a
    dense-retrieval candidate ID here even though it is a real chunk-size knob (it
    would need its own, separate re-encoding comparison to test). It is still reported
    on every row for completeness, per the task's "chunk sizes" output field.
    """
    return {
        "current": {
            "query_chunk": config.DENSE_QUERY_CHUNK,
            "index_chunk": config.DENSE_INDEX_CHUNK,
            "embedding_batch": config.EMBEDDING_BATCH,
        },
        "larger": {
            "query_chunk": config.DENSE_QUERY_CHUNK * 2,
            "index_chunk": config.DENSE_INDEX_CHUNK * 2,
            "embedding_batch": config.EMBEDDING_BATCH * 2,
        },
    }


def run_chunk_equality(sample: float = 0.0045, encoder: Encoder | None = None) -> dict:
    """E3: exact dense-retrieval candidate-ID equality across chunk-size configurations.

    For the same deterministic S1/S2/S3 sample, the same normalisation, the same
    embeddings (computed once, reused by both sides), the same blocking K
    (``config.K_EMBEDDING``) and the same country restriction (production
    ``blocking.country_groups``), this runs ``blocking.dense_topk`` -- the exact
    production dense-retrieval primitive, unmodified -- twice: once with the current
    production ``(DENSE_QUERY_CHUNK, DENSE_INDEX_CHUNK)`` and once with
    :func:`chunk_size_configs`'s ``'larger'`` values (explicit keyword arguments,
    since ``dense_topk``'s own default arguments are bound to ``config.*`` once at
    import time, so mutating ``config`` at runtime would not otherwise reach a call
    that omits them). The candidate-ID sets produced by each side are then compared
    for exact equality (:func:`_candidate_set_diff`), and each side's true-pair
    recall, mean/max candidates and resource usage are reported alongside.

    Never modifies ``config.DENSE_QUERY_CHUNK``/``DENSE_INDEX_CHUNK`` -- both values
    are only ever passed as explicit ``query_chunk``/``index_chunk`` keyword
    arguments into ``dense_topk`` calls local to this function.

    Sets ``status`` to ``"PASS"`` if the two candidate-ID sets are byte-for-byte
    identical; ``"FAIL"`` if they differ *and* the larger-chunk side's true-pair
    recall is lower (an actual true match was lost, not just tie-breaking noise);
    otherwise ``"INVESTIGATE"`` (candidate sets differ, but no true match was lost --
    e.g. a tie at the k-th position broken differently by chunk boundary). This
    function never declares a production change safe by itself; it only measures.

    Writes ``artifacts/diagnostics/chunk_equality_<timestamp>.json`` (full report,
    reproducibility fields, git commit, dataset hashes, env info) and a one-row
    ``chunk_equality_<timestamp>.tsv`` summary. Returns the JSON-serialisable report
    dict. Never touches ``output/``, ``run_pipeline.py`` or any ``config.py`` default.
    """
    prep, truth, s1_ids = _prepare_for_dense_diagnostic(sample, encoder)
    n_other = len(prep["others"])
    chunks = chunk_size_configs()
    k = config.K_EMBEDDING
    _log(f"E3 chunk equality: {len(s1_ids):,} S1 x {n_other:,} others "
        f"(sample={sample}, k={k}), configs={chunks}")

    sides: dict[str, dict] = {}
    for name, cfg in chunks.items():
        def topk(q, x, kk, qc=cfg["query_chunk"], ic=cfg["index_chunk"]):
            """Production dense_topk, called with this side's explicit chunk sizes."""
            return dense_topk(q, x, kk, query_chunk=qc, index_chunk=ic)
        with diag.ResourceTracker() as rt:
            cands = _dense_candidates_grouped(prep["s1"], prep["others"], prep["emb"], k, topk)
        sides[name] = {
            "cands": cands, "chunk_config": cfg,
            "stats": _cand_stats(cands, s1_ids),
            "recall": blocking_recall(cands, truth),
            "resource": dict(rt.report),
        }
        _log(f"E3 chunk equality: '{name}' n_pairs={sides[name]['stats']['n_pairs']} "
            f"pair_recall={sides[name]['recall']['pair_recall']:.4f} "
            f"runtime={rt.report['runtime_s']:.2f}s")

    diff = _candidate_set_diff(sides["current"]["cands"], sides["larger"]["cands"])
    if diff["exact_equal"]:
        status = "PASS"
    elif sides["larger"]["recall"]["pair_recall"] < sides["current"]["recall"]["pair_recall"]:
        status = "FAIL"
    else:
        status = "INVESTIGATE"

    report = {
        "kind": "chunk_equality", "status": status, "sample": sample,
        "k_embedding": k, "block_within_country": config.BLOCK_WITHIN_COUNTRY,
        "n_s1": len(s1_ids), "n_other": n_other,
        "current": {k2: v for k2, v in sides["current"].items() if k2 != "cands"},
        "larger": {k2: v for k2, v in sides["larger"].items() if k2 != "cands"},
        "diff": diff,
        "git_commit": rp._git_hash(),
        "dataset_file_hashes": diag.dataset_file_hashes(config.TRAIN_FILES),
        "env": diag.env_info(),
    }
    ts = time.strftime("%Y%m%d_%H%M%S")
    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    report["timestamp"] = ts
    diag.save_json(report, DIAG_DIR / f"chunk_equality_{ts}.json")
    summary = pd.DataFrame([{
        "timestamp": ts, "status": status, "sample": sample, "n_s1": len(s1_ids),
        "current_query_chunk": chunks["current"]["query_chunk"],
        "current_index_chunk": chunks["current"]["index_chunk"],
        "larger_query_chunk": chunks["larger"]["query_chunk"],
        "larger_index_chunk": chunks["larger"]["index_chunk"],
        "exact_equal": diff["exact_equal"], "n_s1_differing": diff["n_s1_differing"],
        "n_missing_pairs": diff["n_missing_pairs"], "n_extra_pairs": diff["n_extra_pairs"],
        "current_pair_recall": sides["current"]["recall"]["pair_recall"],
        "larger_pair_recall": sides["larger"]["recall"]["pair_recall"],
        "current_runtime_s": sides["current"]["resource"]["runtime_s"],
        "larger_runtime_s": sides["larger"]["resource"]["runtime_s"],
    }])
    summary.to_csv(DIAG_DIR / f"chunk_equality_{ts}.tsv", sep="\t", index=False)
    _log(f"E3 chunk equality: status={status} "
        f"(n_s1_differing={diff['n_s1_differing']}, missing={diff['n_missing_pairs']}, "
        f"extra={diff['n_extra_pairs']}); report written to "
        f"{DIAG_DIR / f'chunk_equality_{ts}.json'}")
    return report


# --- E4: FP16 embedding-precision retrieval correctness -------------------------------

def _dense_topk_with_dtype(
    queries: np.ndarray, index: np.ndarray, k: int, compute_dtype: type,
    query_chunk: int = config.DENSE_QUERY_CHUNK, index_chunk: int = config.DENSE_INDEX_CHUNK,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Diagnostic-only twin of ``blocking.dense_topk``'s CPU chunked top-k merge,
    parametrised by the similarity-matmul dtype.

    ``blocking.dense_topk``'s own CPU branch always upcasts both operands to
    ``float32`` (``np.asarray(..., dtype=np.float32)``) before the matmul,
    *regardless* of the input embeddings' own dtype -- ``blocking.compute_embeddings``
    already stores them as float16 on disk, so on a CPU-only machine (no CUDA) simply
    feeding float16 arrays into the real ``dense_topk`` would silently be re-upcast
    and never actually exercise a float16 matmul. This function exists only so E4 can
    measure what changes if that matmul itself runs at a lower precision: it
    reproduces ``dense_topk``'s exact chunked running-top-k merge algorithm (the same
    ``best_s``/``best_i`` running update, the same ``np.argpartition``-based per-chunk
    and per-merge top-k selection, the same chunk loop structure) verbatim, with
    ``compute_dtype`` substituted for the hard-coded ``np.float32`` upcast in the
    similarity matmul. The merge/comparison bookkeeping (``best_s``/``best_i``) still
    accumulates in float32 -- only the similarity *values themselves* are computed at
    ``compute_dtype`` -- matching what an actual float16 matmul followed by a
    float32-accumulated top-k merge would produce (this mirrors ``dense_topk``'s own
    GPU branch, which does exactly this: a float16 matmul via
    ``torch.float16``, then ``.float()`` before ``torch.topk``).

    On hardware where ``dense_topk`` already runs on CUDA (float16 matmul via torch),
    this twin's ``compute_dtype=np.float16`` result should closely match what
    ``dense_topk`` itself already does there -- this function's real purpose is
    making the float16-vs-float32 comparison possible on a CPU-only development
    machine, where ``dense_topk`` otherwise always computes in float32. It is never
    called anywhere outside ``run_fp16_retrieval`` and does not modify
    ``blocking.py``.
    """
    nq, nx = len(queries), len(index)
    k = min(k, nx)
    if nq == 0 or k == 0:
        e = np.array([], dtype=np.int64)
        return e, e, np.array([], dtype=np.float32)
    best_s = np.full((nq, k), -np.inf, dtype=np.float32)
    best_i = np.full((nq, k), -1, dtype=np.int64)
    for xs in range(0, nx, index_chunk):
        xblock = np.asarray(index[xs:xs + index_chunk], dtype=compute_dtype)
        kb = min(k, len(xblock))
        for qs in range(0, nq, query_chunk):
            qblock = np.asarray(queries[qs:qs + query_chunk], dtype=compute_dtype)
            sims = (qblock @ xblock.T).astype(np.float32)
            si = np.argpartition(-sims, kb - 1, axis=1)[:, :kb]
            sv = np.take_along_axis(sims, si, axis=1)
            si = si + xs
            cat_s = np.concatenate([best_s[qs:qs + query_chunk], sv], axis=1)
            cat_i = np.concatenate([best_i[qs:qs + query_chunk], si], axis=1)
            top = np.argpartition(-cat_s, k - 1, axis=1)[:, :k]
            best_s[qs:qs + query_chunk] = np.take_along_axis(cat_s, top, axis=1)
            best_i[qs:qs + query_chunk] = np.take_along_axis(cat_i, top, axis=1)
    rows = np.repeat(np.arange(nq), k)
    flat_i, flat_s = best_i.ravel(), best_s.ravel()
    ok = flat_i >= 0
    return rows[ok], flat_i[ok], flat_s[ok]


def run_fp16_retrieval(sample: float = 0.0045, encoder: Encoder | None = None) -> dict:
    """E4: true-match retrieval loss from computing dense-retrieval similarity at
    float16 instead of ``blocking.dense_topk``'s current per-hardware precision.

    For the same deterministic sample, the same normalised records, the same model
    weights/embeddings (computed once, reused by both sides), the same blocking K and
    the same query/index chunk sizes, this compares two dense-retrieval candidate
    sets: the current production path (``blocking.dense_topk`` called unmodified --
    float32 matmul on CPU, float16 matmul on CUDA, whichever this machine already
    does) against a forced-float16-matmul path (:func:`_dense_topk_with_dtype` with
    ``compute_dtype=np.float16`` -- see that function's docstring for why a genuine
    twin, not the real ``dense_topk``, is needed to exercise float16 on a CPU-only
    machine).

    Beyond the exact-equality diff (:func:`_candidate_set_diff`) and both sides'
    blocking recall/candidate stats, this additionally counts **true matches lost**:
    true (S1, match) pairs from the ground truth that the current-precision side
    retrieved but the float16 side did not, and the percentage of all true matches in
    scope that represents. Classifies the outcome as one of:

    * ``"zero"`` -- the two candidate-ID sets are exactly identical.
    * ``"tie_breaking_only"`` -- candidate sets differ, but zero true matches were
      lost (the differences are all among non-matching candidates, or a tie at the
      k-th position broken differently).
    * ``"true_match_loss"`` -- at least one true match present under the current
      precision path is absent under float16.

    ``status`` is ``"PASS"`` for ``"zero"``, ``"INVESTIGATE"`` for
    ``"tie_breaking_only"``, and ``"FAIL"`` for ``"true_match_loss"`` -- this function
    never itself declares float16 safe for production; it only measures and reports.

    Writes ``artifacts/diagnostics/fp16_retrieval_<timestamp>.json`` (full report,
    reproducibility fields, git commit, dataset hashes, env info) and a one-row
    ``fp16_retrieval_<timestamp>.tsv`` summary. Returns the JSON-serialisable report
    dict. Never modifies embedding precision, chunk sizes or any other production
    default -- ``_dense_topk_with_dtype`` is a diagnostic-only function called from
    nowhere else, and ``blocking.py``/``config.py`` are untouched.
    """
    prep, truth, s1_ids = _prepare_for_dense_diagnostic(sample, encoder)
    n_other = len(prep["others"])
    k = config.K_EMBEDDING
    qc, ic = config.DENSE_QUERY_CHUNK, config.DENSE_INDEX_CHUNK
    _log(f"E4 fp16 retrieval: {len(s1_ids):,} S1 x {n_other:,} others "
        f"(sample={sample}, k={k}, query_chunk={qc}, index_chunk={ic})")

    with diag.ResourceTracker() as rt_current:
        cands_current = _dense_candidates_grouped(
            prep["s1"], prep["others"], prep["emb"], k,
            lambda q, x, kk: dense_topk(q, x, kk, query_chunk=qc, index_chunk=ic))
    with diag.ResourceTracker() as rt_fp16:
        cands_fp16 = _dense_candidates_grouped(
            prep["s1"], prep["others"], prep["emb"], k,
            lambda q, x, kk: _dense_topk_with_dtype(q, x, kk, np.float16, qc, ic))

    diff = _candidate_set_diff(cands_current, cands_fp16)
    recall_current = blocking_recall(cands_current, truth)
    recall_fp16 = blocking_recall(cands_fp16, truth)
    stats_current = _cand_stats(cands_current, s1_ids)
    stats_fp16 = _cand_stats(cands_fp16, s1_ids)

    n_true_total, n_true_lost, pct_true_lost = _true_match_loss(
        cands_current, cands_fp16, truth, s1_ids)

    if n_true_lost > 0:
        category, status = "true_match_loss", "FAIL"
    elif not diff["exact_equal"]:
        category, status = "tie_breaking_only", "INVESTIGATE"
    else:
        category, status = "zero", "PASS"

    report = {
        "kind": "fp16_retrieval", "status": status, "category": category,
        "sample": sample, "k_embedding": k, "query_chunk": qc, "index_chunk": ic,
        "block_within_country": config.BLOCK_WITHIN_COUNTRY,
        "n_s1": len(s1_ids), "n_other": n_other,
        "current": {"dtype": "float32/hardware-default", "stats": stats_current,
                    "recall": recall_current, "resource": dict(rt_current.report)},
        "fp16": {"dtype": "float16", "stats": stats_fp16,
                "recall": recall_fp16, "resource": dict(rt_fp16.report)},
        "diff": diff,
        "n_true_matches_total": n_true_total,
        "n_true_matches_lost": n_true_lost,
        "pct_true_matches_lost": pct_true_lost,
        "git_commit": rp._git_hash(),
        "dataset_file_hashes": diag.dataset_file_hashes(config.TRAIN_FILES),
        "env": diag.env_info(),
    }
    ts = time.strftime("%Y%m%d_%H%M%S")
    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    report["timestamp"] = ts
    diag.save_json(report, DIAG_DIR / f"fp16_retrieval_{ts}.json")
    summary = pd.DataFrame([{
        "timestamp": ts, "status": status, "category": category, "sample": sample,
        "n_s1": len(s1_ids), "exact_equal": diff["exact_equal"],
        "n_s1_differing": diff["n_s1_differing"], "n_missing_pairs": diff["n_missing_pairs"],
        "n_extra_pairs": diff["n_extra_pairs"],
        "n_true_matches_total": n_true_total, "n_true_matches_lost": n_true_lost,
        "pct_true_matches_lost": pct_true_lost,
        "current_pair_recall": recall_current["pair_recall"],
        "fp16_pair_recall": recall_fp16["pair_recall"],
        "current_s1_full_recall": recall_current["s1_full_recall"],
        "fp16_s1_full_recall": recall_fp16["s1_full_recall"],
        "current_runtime_s": rt_current.report["runtime_s"],
        "fp16_runtime_s": rt_fp16.report["runtime_s"],
    }])
    summary.to_csv(DIAG_DIR / f"fp16_retrieval_{ts}.tsv", sep="\t", index=False)
    _log(f"E4 fp16 retrieval: status={status} category={category} "
        f"(true matches lost={n_true_lost}/{n_true_total} = {pct_true_lost:.2f}%); "
        f"report written to {DIAG_DIR / f'fp16_retrieval_{ts}.json'}")
    return report


# --- P3: decision-layer validation ---------------------------------------------------------

#: Candidate configuration held fixed for the whole P3 experiment: the "both40"
#: configuration P1-downstream reported as the current best candidate (task
#: requirement -- P3 does not compare blocking-K values, it isolates the decision
#: layer). ``rare``/``digit``/``address`` stay at their P1-downstream values too.
P3_CANDIDATE_CONFIG: dict[str, int] = {"tfidf": 40, "embed": 40, "rare": 10, "digit": 10,
                                       "address": 10}


def decide_threshold_then_one_to_one(
    scored: pd.DataFrame, tau: float, s1_ids: Iterable[str] | None,
    singleton_tau: float | None = None,
) -> dict[str, list[str]]:
    """Diagnostic-only decision order: threshold (+ singleton gate) first, one-to-one
    assignment second -- the reverse of production's order.

    ``decide.apply_threshold`` (unmodified, called elsewhere in this module as-is)
    always applies :func:`decide.assign_one_to_one` *before* the ``prob >= tau``
    filter, i.e. one-to-one-then-threshold; that is P3-B's "Order B". This function
    exists solely so P3-B can measure the other ordering ("Order A": threshold first,
    one-to-one only among the pairs that already passed the threshold/singleton gate)
    without changing ``decide.py`` itself. It calls the exact same primitives
    (``assign_one_to_one``) production uses, in a different sequence.
    """
    keep = scored["prob"].to_numpy() >= tau
    if singleton_tau is not None:
        top = scored.groupby("s1_id")["prob"].transform("max").to_numpy()
        keep &= top >= singleton_tau
    final = assign_one_to_one(scored[keep])
    kept = final.sort_values(["s1_id", "prob", "cand_id"], ascending=[True, False, True])
    out = {s: list(g) for s, g in kept.groupby("s1_id", sort=False)["cand_id"]}
    universe = scored["s1_id"].unique() if s1_ids is None else s1_ids
    return {s: out.get(s, []) for s in universe}


#: The two decision orders P3-B compares, as ``(scored, tau, s1_ids, singleton_tau)
#: -> {s1_id: [matched ids]}`` callables sharing one signature so P3-B's loop can
#: treat them identically. "order_b" is production's own ``apply_threshold``,
#: unmodified, called with ``one_to_one=True`` -- not a re-implementation.
_DECISION_ORDERS: dict[str, Callable[..., dict]] = {
    "order_A_threshold_then_one_to_one": decide_threshold_then_one_to_one,
    "order_B_one_to_one_then_threshold": lambda scored, tau, s1_ids, singleton_tau: apply_threshold(
        scored, tau, s1_ids, singleton_tau, one_to_one=True),
}


def _label_lookup(scored: pd.DataFrame) -> dict[tuple[str, str], int]:
    """``{(s1_id, cand_id): label}`` for a scored frame that carries a ``label`` column."""
    return {(s, c): int(l) for s, c, l in
           zip(scored["s1_id"], scored["cand_id"], scored["label"])}


def _removal_counts(pre: dict[str, list[str]], post: dict[str, list[str]],
                    label_lookup: dict[tuple[str, str], int]) -> dict:
    """Pairs kept by ``pre`` but dropped by ``post`` -- i.e. what going from ``pre``
    to ``post`` removed, split into true (label 1) and false (label 0) matches.

    Used by P3-A (``pre`` = threshold-only/one-to-one-off, ``post`` = one-to-one-on,
    at the same tau) and P3-B (``pre`` = threshold-only, ``post`` = one order's
    final decision) to answer "how many matches, and how many of them true, did
    one-to-one remove".
    """
    pre_pairs = {(s, c) for s, cs in pre.items() for c in cs}
    post_pairs = {(s, c) for s, cs in post.items() for c in cs}
    removed = pre_pairs - post_pairs
    n_true_removed = sum(1 for p in removed if label_lookup.get(p, 0) == 1)
    return {
        "n_predicted_pre": len(pre_pairs), "n_predicted_post": len(post_pairs),
        "n_removed_by_step": len(removed), "n_true_matches_removed": n_true_removed,
        "n_false_matches_removed": len(removed) - n_true_removed,
    }


def _tau_neighborhood(tau: float, grid: tuple[float, ...] = config.TAU_GRID,
                      step: float = 0.025, n: int = 2) -> list[float]:
    """5-point neighborhood ``T-2*step .. T+2*step`` around ``tau``, clipped to the
    existing ``config.TAU_GRID``'s own min/max so every returned value is one
    ``tune_threshold`` could actually have chosen (task requirement: "appropriate to
    the existing threshold grid", "only evaluate thresholds that are valid").
    """
    lo, hi = min(grid), max(grid)
    cand = (round(tau + i * step, 3) for i in range(-n, n + 1))
    return sorted({c for c in cand if lo <= c <= hi})


def _singleton_neighborhood(singleton_tau: float | None,
                            grid: tuple[float, ...] = config.TAU_GRID,
                            step: float = 0.025, n: int = 2) -> list[float | None]:
    """Neighborhood around the selected singleton threshold, reusing
    :func:`_tau_neighborhood`'s clipping. ``None`` (no singleton gate) has no numeric
    neighborhood, so it is returned as its own single-element list -- this never
    invents a new thresholding scheme for the "no gate" case (task requirement).
    """
    if singleton_tau is None:
        return [None]
    return _tau_neighborhood(singleton_tau, grid, step, n)


def _p3_prepare(sample: float, use_embeddings: bool) -> dict:
    """P3 shared setup: load, normalise/embed, block (fixed ``P3_CANDIDATE_CONFIG``)
    and featurise the sample exactly once, plus one train/valid S1 split.

    Every P3 sub-experiment (A/B/C/D) reuses this same ``cands``/``feats``/split --
    only the decision-layer variable may differ between them (task's "experiment
    isolation" requirement). Calls the existing, unmodified ``run_pipeline.prepare``,
    ``blocking.generate_candidates``, ``blocking.report_blocking_stats``,
    ``features.build_features``, ``features.label_pairs`` and ``run_pipeline.split_s1``
    -- the same functions ``run_pipeline.valid_run`` uses, in the same order.
    """
    s1, s2, s3, truth = rp._load_train(sample)
    prep = rp.prepare(s1, s2, s3, use_embeddings=use_embeddings, use_cache=True)
    s1_ids = list(prep["s1"][config.ID_COL])
    s1_country = dict(zip(prep["s1"][config.ID_COL], prep["s1"][config.COUNTRY_COL]))
    n_other = len(prep["others"])
    cands = generate_candidates(prep["s1"], prep["others"], embeddings=prep["emb"],
                                verbose=False, k_overrides=dict(P3_CANDIDATE_CONFIG))
    block_stats = report_blocking_stats(cands, truth, s1_ids, n_other, s1_country, verbose=False)
    feats = build_features(cands, prep["s1"], prep["others"], prep["emb"])
    feats["label"] = label_pairs(feats, truth)
    train_ids, valid_ids = rp.split_s1(s1_ids, truth)
    return {
        "prep": prep, "cands": cands, "feats": feats, "truth": truth,
        "train_ids": train_ids, "valid_ids": valid_ids, "s1_country": s1_country,
        "block_stats": block_stats, "s1_ids": s1_ids, "n_other": n_other,
    }


def _p3_train(ctx: dict) -> dict:
    """Train the ONE LightGBM model/OOF the whole P3 experiment reuses.

    Exactly one ``model.group_folds`` + ``model.train_oof`` + ``model.train_full``
    call happens here, on ``ctx``'s training S1s -- no P3 sub-experiment (A/B/C/D)
    retrains anything (task's explicit "do NOT retrain" requirement for P3-B/C, and
    the shared-model requirement for P3-A). Returns the OOF-scored training pairs
    (with ``label``, for removal-count bookkeeping), the GroupKFold fold id per
    training row (for fold-stability), the trained full model, and that model's
    scored predictions on the held-out validation pairs.
    """
    feats = ctx["feats"]
    in_train = feats["s1_id"].isin(set(ctx["train_ids"])).to_numpy()
    feats_train = feats[in_train]
    cols = feature_columns(feats)
    X, y, groups = feats_train[cols], feats_train["label"].to_numpy(), feats_train["s1_id"]
    folds = group_folds(groups, config.N_FOLDS)
    oof, fold_models = train_oof(X, y, groups, verbose=False)
    oof_scored = feats_train[["s1_id", "cand_id", "label"]].assign(prob=oof)
    model = train_full(X, y, fold_models=fold_models)
    valid_feats = feats[~in_train]
    valid_scored = valid_feats[["s1_id", "cand_id", "label"]].assign(
        prob=predict(model, valid_feats[feature_columns(feats)]))
    return {"model": model, "oof_scored": oof_scored, "folds": folds, "valid_scored": valid_scored}


def run_p3a_one_to_one(ctx: dict, trained: dict, truth: dict[str, list[str]]) -> dict:
    """P3-A: one-to-one ON vs OFF, threshold retuned (via the existing
    ``decide.tune_threshold`` mechanism) separately for each, on the one shared OOF.

    For each of ``one_to_one in (True, False)``: tunes ``(tau, singleton_tau)`` on
    the training OOF with that flag (``decide.tune_threshold``'s own sweep already
    applies one-to-one internally when asked, so this is production's exact tuning
    mechanism, not a new one), applies the matching flag to the held-out validation
    predictions (``decide.apply_threshold``, unmodified), and reports OOF/validation
    macro F0.5, precision/recall, singleton/non-singleton and per-country F0.5,
    predicted/true match counts, per-fold OOF stability, and -- for the ON row --
    how many matches (and how many of them true) one-to-one removed relative to the
    same tau/singleton_tau with one-to-one off.

    Returns ``{"rows": [...], "tuned": {True: {...}, False: {...}}}``; P3-B/C/D reuse
    ``tuned[True]``'s ``(tau, singleton_tau)`` as "the current production tuning".
    """
    train_ids, valid_ids = ctx["train_ids"], ctx["valid_ids"]
    s1_country = ctx["s1_country"]
    oof_scored, valid_scored = trained["oof_scored"], trained["valid_scored"]
    label_lookup_valid = _label_lookup(valid_scored)
    rows: list[dict] = []
    tuned: dict[bool, dict] = {}
    for one_to_one in (True, False):
        tau, stau, oof_f05, _grid = tune_threshold(oof_scored, truth, train_ids,
                                                    one_to_one=one_to_one)
        tuned[one_to_one] = {"tau": tau, "singleton_tau": stau}
        pred = apply_threshold(valid_scored, tau, valid_ids, stau, one_to_one=one_to_one)
        report = score_report(pred, truth, valid_ids, groups=s1_country)
        stability = _fold_stability(oof_scored, trained["folds"], tau, stau, truth,
                                    one_to_one=one_to_one)
        n_pred = sum(len(v) for v in pred.values())
        n_true = sum(len(set(truth.get(s, ()))) for s in valid_ids)
        if one_to_one:
            pred_off = apply_threshold(valid_scored, tau, valid_ids, stau, one_to_one=False)
            removal = _removal_counts(pred_off, pred, label_lookup_valid)
        else:
            removal = {"n_predicted_pre": n_pred, "n_predicted_post": n_pred,
                      "n_removed_by_step": 0, "n_true_matches_removed": 0,
                      "n_false_matches_removed": 0}
        rows.append({
            "experiment": f"P3A_one_to_one_{'on' if one_to_one else 'off'}",
            "one_to_one": one_to_one, "tau": tau, "singleton_tau": stau,
            "oof_macro_f05": oof_f05,
            **{f"valid_{k}": v for k, v in report.items()},
            "n_predicted_matches": n_pred, "n_true_matches": n_true,
            **removal,
            "fold_f05": ";".join(f"{v:.4f}" for v in stability["fold_f05"]),
            "fold_f05_mean": stability["fold_f05_mean"],
            "fold_f05_std": stability["fold_f05_std"],
        })
        _log(f"P3-A one_to_one={one_to_one}: OOF F0.5={oof_f05:.4f} "
            f"valid F0.5={report['macro_f05']:.4f} tau={tau} singleton_tau={stau}")
    return {"rows": rows, "tuned": tuned}


def run_p3b_decision_order(ctx: dict, trained: dict, truth: dict[str, list[str]],
                           tau: float, singleton_tau: float | None) -> list[dict]:
    """P3-B: decision order (threshold-then-one-to-one vs one-to-one-then-threshold),
    at the ONE ``(tau, singleton_tau)`` P3-A's one-to-one-ON tuning selected -- the
    model is not retrained and the threshold is not retuned between orders, so only
    ordering can explain a difference (task requirement).
    """
    train_ids, valid_ids = ctx["train_ids"], ctx["valid_ids"]
    s1_country = ctx["s1_country"]
    oof_scored, valid_scored = trained["oof_scored"], trained["valid_scored"]
    label_lookup_valid = _label_lookup(valid_scored)
    pred_threshold_only = apply_threshold(valid_scored, tau, valid_ids, singleton_tau,
                                          one_to_one=False)
    rows: list[dict] = []
    for name, decide_fn in _DECISION_ORDERS.items():
        oof_pred = decide_fn(oof_scored, tau, train_ids, singleton_tau)
        valid_pred = decide_fn(valid_scored, tau, valid_ids, singleton_tau)
        oof_f05 = score_report(oof_pred, truth, train_ids)["macro_f05"]
        report = score_report(valid_pred, truth, valid_ids, groups=s1_country)
        stability = _fold_stability(oof_scored, trained["folds"], tau, singleton_tau, truth,
                                    decide_fn=decide_fn)
        removal = _removal_counts(pred_threshold_only, valid_pred, label_lookup_valid)
        rows.append({
            "experiment": f"P3B_{name}", "decision_order": name,
            "tau": tau, "singleton_tau": singleton_tau,
            "oof_macro_f05": oof_f05,
            **{f"valid_{k}": v for k, v in report.items()},
            "n_predicted_matches": sum(len(v) for v in valid_pred.values()),
            "n_true_matches": sum(len(set(truth.get(s, ()))) for s in valid_ids),
            **removal,
            "fold_f05": ";".join(f"{v:.4f}" for v in stability["fold_f05"]),
            "fold_f05_mean": stability["fold_f05_mean"],
            "fold_f05_std": stability["fold_f05_std"],
        })
        _log(f"P3-B {name}: OOF F0.5={oof_f05:.4f} valid F0.5={report['macro_f05']:.4f}")
    return rows


def run_p3c_threshold_generalization(ctx: dict, trained: dict, truth: dict[str, list[str]],
                                     tau: float, singleton_tau: float | None) -> list[dict]:
    """P3-C: score the tuned ``tau``'s neighborhood (task's example spacing, clipped
    to ``config.TAU_GRID``) on the held-out validation predictions, at the SAME
    ``singleton_tau`` and the SAME (already-trained) model/OOF -- no retuning, no new
    thresholding method, per task requirement.
    """
    valid_ids = ctx["valid_ids"]
    s1_country = ctx["s1_country"]
    valid_scored = trained["valid_scored"]
    rows: list[dict] = []
    for t in _tau_neighborhood(tau):
        pred = apply_threshold(valid_scored, t, valid_ids, singleton_tau, one_to_one=config.ONE_TO_ONE)
        report = score_report(pred, truth, valid_ids, groups=s1_country)
        rows.append({
            "experiment": f"P3C_tau_{t:.3f}", "tau": t,
            "delta_from_selected_tau": round(t - tau, 3),
            "singleton_tau": singleton_tau, "is_selected": t == tau,
            **{f"valid_{k}": v for k, v in report.items()},
        })
        _log(f"P3-C tau={t}: valid F0.5={report['macro_f05']:.4f}")
    return rows


def run_p3d_singleton_threshold(ctx: dict, trained: dict, truth: dict[str, list[str]],
                                tau: float, singleton_tau: float | None) -> list[dict]:
    """P3-D: singleton-threshold behaviour at the selected value plus its neighborhood
    (task's example spacing; ``None`` gets no numeric neighborhood -- see
    :func:`_singleton_neighborhood`), at the SAME ``tau`` and the SAME model/OOF.

    Each row additionally reports predicted-singleton (zero-match), true-singleton
    and false-singleton (predicted empty despite real matches existing) counts.
    """
    valid_ids = ctx["valid_ids"]
    s1_country = ctx["s1_country"]
    valid_scored = trained["valid_scored"]
    truth_by_s1 = {s: set(truth.get(s, ())) for s in valid_ids}
    n_true_singletons = sum(1 for s in valid_ids if not truth_by_s1[s])
    rows: list[dict] = []
    for stau in _singleton_neighborhood(singleton_tau):
        pred = apply_threshold(valid_scored, tau, valid_ids, stau, one_to_one=config.ONE_TO_ONE)
        report = score_report(pred, truth, valid_ids, groups=s1_country)
        n_pred_singletons = sum(1 for s in valid_ids if not pred.get(s))
        n_false_singletons = sum(1 for s in valid_ids if not pred.get(s) and truth_by_s1[s])
        label = "none" if stau is None else f"{stau:.3f}"
        rows.append({
            "experiment": f"P3D_singleton_tau_{label}", "tau": tau, "singleton_tau": stau,
            "is_selected": stau == singleton_tau,
            "n_predicted_singletons": n_pred_singletons, "zero_match_count": n_pred_singletons,
            "n_true_singletons": n_true_singletons, "n_false_singletons": n_false_singletons,
            **{f"valid_{k}": v for k, v in report.items()},
        })
        _log(f"P3-D singleton_tau={stau}: valid F0.5={report['macro_f05']:.4f} "
            f"n_false_singletons={n_false_singletons}")
    return rows


def run_decision_validation(sample: float = 0.0045, use_embeddings: bool = True) -> pd.DataFrame:
    """P3: decision-layer validation on ONE fixed candidate configuration
    (``P3_CANDIDATE_CONFIG``, the current "both40" candidate) and ONE trained
    LightGBM model/OOF -- P3-A (one-to-one ON/OFF), P3-B (decision order), P3-C
    (threshold neighborhood) and P3-D (singleton-threshold neighborhood) each vary
    only their own decision-layer knob, per the task's experiment-isolation
    requirement. Never retrains between sub-experiments, never touches ``output/``,
    ``run_pipeline.py`` or any ``config.py`` default, and never runs ``--mode test``.

    Writes one row per configuration (18 total: 2 for P3-A, 2 for P3-B, up to 5 for
    P3-C, up to 5 for P3-D) to
    ``artifacts/diagnostics/decision_validation_<timestamp>.tsv`` and a sidecar
    ``decision_validation_<timestamp>.json`` (full per-experiment detail, the
    candidate configuration, the selected tau/singleton_tau, block stats, git commit,
    dataset hashes and env info). Returns the same frame.
    """
    _log(f"P3 decision validation: sample={sample}, candidate config={P3_CANDIDATE_CONFIG}")
    with diag.ResourceTracker() as rt:
        ctx = _p3_prepare(sample, use_embeddings)
        trained = _p3_train(ctx)
        truth = ctx["truth"]

        a = run_p3a_one_to_one(ctx, trained, truth)
        tau, singleton_tau = a["tuned"][True]["tau"], a["tuned"][True]["singleton_tau"]

        b_rows = run_p3b_decision_order(ctx, trained, truth, tau, singleton_tau)
        c_rows = run_p3c_threshold_generalization(ctx, trained, truth, tau, singleton_tau)
        d_rows = run_p3d_singleton_threshold(ctx, trained, truth, tau, singleton_tau)

    all_rows = a["rows"] + b_rows + c_rows + d_rows
    for row in all_rows:
        row.setdefault("candidate_config", str(P3_CANDIDATE_CONFIG))
        row.setdefault("sample", sample)
    out = pd.DataFrame(all_rows)
    ts = time.strftime("%Y%m%d_%H%M%S")
    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DIAG_DIR / f"decision_validation_{ts}.tsv"
    out.to_csv(out_path, sep="\t", index=False)
    meta = {
        "kind": "decision_validation", "timestamp": ts, "sample": sample,
        "use_embeddings": use_embeddings, "candidate_config": P3_CANDIDATE_CONFIG,
        "n_s1": len(ctx["s1_ids"]), "n_train_s1": len(ctx["train_ids"]),
        "n_valid_s1": len(ctx["valid_ids"]), "n_other": ctx["n_other"],
        "selected_tau": tau, "selected_singleton_tau": singleton_tau,
        "block_stats": ctx["block_stats"],
        "p3a_one_to_one": a["rows"], "p3b_decision_order": b_rows,
        "p3c_threshold_generalization": c_rows, "p3d_singleton_threshold": d_rows,
        "git_commit": rp._git_hash(),
        "dataset_file_hashes": diag.dataset_file_hashes(config.TRAIN_FILES),
        "env": diag.env_info(),
        "runtime_seconds": rt.report["runtime_s"],
        "peak_rss_bytes": rt.report["peak_rss_bytes"],
        "current_rss_bytes": rt.report["current_rss_bytes"],
    }
    diag.save_json(meta, DIAG_DIR / f"decision_validation_{ts}.json")
    _log(f"P3 decision validation: {len(all_rows)} rows written to {out_path} "
        f"(selected tau={tau}, singleton_tau={singleton_tau})")
    return out


# --- LOCO generalization comparison ---------------------------------------------------------

#: The two candidate configurations compared: production's "baseline" 20/20 K and
#: the P1-downstream/P3-validated "both40" 40/40 K. Only tfidf/embed K differ --
#: rare/digit/address stay at their production defaults, matching every earlier
#: P1/P3 comparison of these two configurations.
LOCO_CONFIGS: list[dict] = [
    {"name": "baseline", "tfidf": 20, "embed": 20, "rare": 10, "digit": 10, "address": 10},
    {"name": "both40", "tfidf": 40, "embed": 40, "rare": 10, "digit": 10, "address": 10},
]

#: Documentation of how this diagnostic satisfies each item of the LOCO leakage
#: checklist -- written once and recorded verbatim in every run's JSON sidecar, not
#: derived from data, so a reader can audit the *design* independently of any one
#: run's numbers. Item 5 (feature statistics) is the one place this diagnostic
#: deliberately does NOT reuse ``run_pipeline.valid_run``'s own ``loco=True`` branch:
#: that branch calls ``build_features`` once on the full pooled (all-countries)
#: frame before splitting by country, so its ``features.FeatureContext`` TF-IDF
#: vectorizers see the held-out country's own vocabulary. This module's
#: :func:`_loco_fold` instead fits ``FeatureContext`` on the training side only and
#: passes that same context explicitly into both sides' ``build_features`` calls,
#: closing that gap without touching ``features.py`` or ``run_pipeline.py``.
LOCO_LEAKAGE_CHECKS: dict[str, str] = {
    "validation_not_in_training": (
        "The held-out country's S1 rows never appear in feats_tr -- _country_split "
        "partitions prep['s1']/prep['others'] by an exact country-equality mask before "
        "any feature/model code runs, so X/y for train_oof/train_full never include a "
        "held-out-country row."),
    "no_held_out_labels_used_in_training": (
        "label_pairs(feats_tr, truth) is computed only on feats_tr (training-country "
        "pairs). feats_ho's labels are computed too, but only for score_report's "
        "evaluation after prediction -- never referenced by train_oof/train_full."),
    "threshold_tuning_scope": (
        "tune_threshold(oof_scored, truth, tr_ids) restricts its macro-F0.5 sweep "
        "universe to tr_ids (training-country S1 IDs only); truth.get(s) for a "
        "held-out-country s1_id is never looked up during tuning."),
    "candidate_generation_scope": (
        "generate_candidates is called twice per fold, once on the train-only (s1, "
        "others) subset and once on the held-out-only subset -- neither call's query "
        "or index pool includes the other side's records, so blocking cannot use "
        "held-out records as candidates for training S1s or vice versa."),
    "feature_statistics_scope": (
        "FeatureContext.fit(tr['s1'], tr['others']) is fit ONLY on training-country "
        "records; the same fitted ctx is passed explicitly (build_features(..., "
        "ctx=ctx)) to build features for BOTH sides, so TF-IDF IDF weights never see "
        "held-out-country vocabulary. (This is the one respect in which "
        "run_pipeline.valid_run's own loco=True branch pools FeatureContext across "
        "all countries before splitting -- fixed here at the diagnostic layer only; "
        "features.py is unmodified.)"),
    "embeddings_are_unsupervised": (
        "Sentence embeddings come from a frozen, pretrained encoder applied per-record "
        "in run_pipeline.prepare(), before any train/held-out split exists -- no label "
        "ever reaches the encoder, satisfying the task's explicit exception for "
        "embeddings."),
    "group_s1_separation_maintained": (
        "S1 IDs are partitioned by country (never split) before any S1 ID reaches "
        "model.group_folds -- no S1 ID appears in both the training GroupKFold OOF "
        "and the held-out evaluation set, and no S1 ID appears in more than one "
        "country partition."),
}


def _country_split(prep: dict, country: str) -> tuple[dict, dict]:
    """Partition an already-``prepare``d (normalised, optionally embedded) dataset
    into ``(train, held_out)`` by exact country match on the S1 and S2/S3 frames
    independently -- ``held_out`` is every record (S1 and S2/S3 alike) whose country
    equals ``country``.

    Both sides' frames are reset to a fresh 0..n-1 index so embedding arrays (which
    are positionally, not label, aligned with the original frames) can be sliced
    with the same boolean mask used to select the frame rows. Never reads or touches
    truth/labels -- country is a normalised covariate on S1/S2/S3 already, present
    before any labels are consulted.
    """
    s1, others = prep["s1"], prep["others"]
    ho_s1_mask = (s1[config.COUNTRY_COL] == country).to_numpy()
    ho_o_mask = (others[config.COUNTRY_COL] == country).to_numpy()

    def _slice(s1_mask: np.ndarray, o_mask: np.ndarray) -> dict:
        emb = None
        if prep["emb"] is not None:
            emb = (np.asarray(prep["emb"][0])[s1_mask], np.asarray(prep["emb"][1])[o_mask])
        return {"s1": s1[s1_mask].reset_index(drop=True),
               "others": others[o_mask].reset_index(drop=True), "emb": emb}

    return _slice(~ho_s1_mask, ~ho_o_mask), _slice(ho_s1_mask, ho_o_mask)


def _loco_fold(prep: dict, truth: dict[str, list[str]], k_overrides: dict[str, int],
              held_out_country: str) -> dict:
    """One LOCO fold: train on every other country, evaluate on ``held_out_country``.

    Candidate generation and ``features.FeatureContext`` fitting are both strictly
    scoped to their own side of the split (see :data:`LOCO_LEAKAGE_CHECKS`); model
    fitting, threshold tuning and per-fold stability use only the existing,
    unmodified ``model``/``decide`` primitives run_pipeline.fit_and_tune itself
    calls, inlined here (rather than calling ``fit_and_tune`` directly) only so this
    function can also return the OOF frame/fold ids needed for
    :func:`_fold_stability` -- the training math is identical either way. Decision
    settings (one-to-one, decision order, tau/singleton_tau grid) are the untouched
    production defaults from ``decide.py``/``config.py`` throughout, matching P3's
    "keep one-to-one ON, keep existing decision order" conclusion.
    """
    tr, ho = _country_split(prep, held_out_country)
    tr_ids = list(tr["s1"][config.ID_COL])
    ho_ids = list(ho["s1"][config.ID_COL])
    ho_country_map = dict(zip(ho_ids, ho["s1"][config.COUNTRY_COL]))

    with diag.ResourceTracker() as rt:
        cands_tr = generate_candidates(tr["s1"], tr["others"], embeddings=tr["emb"],
                                       verbose=False, k_overrides=k_overrides)
        cands_ho = generate_candidates(ho["s1"], ho["others"], embeddings=ho["emb"],
                                       verbose=False, k_overrides=k_overrides)
        block_stats = report_blocking_stats(cands_ho, truth, ho_ids, len(ho["others"]),
                                            ho_country_map, verbose=False)

        # The leakage fix: fit TF-IDF context on the training side only, reuse it
        # (unmodified) for the held-out side -- see LOCO_LEAKAGE_CHECKS["feature_statistics_scope"].
        ctx = FeatureContext.fit(tr["s1"], tr["others"])
        feats_tr = build_features(cands_tr, tr["s1"], tr["others"], tr["emb"], ctx=ctx, verbose=False)
        feats_tr["label"] = label_pairs(feats_tr, truth)
        feats_ho = build_features(cands_ho, ho["s1"], ho["others"], ho["emb"], ctx=ctx, verbose=False)
        feats_ho["label"] = label_pairs(feats_ho, truth)

        cols = feature_columns(feats_tr)
        X, y, groups = feats_tr[cols], feats_tr["label"].to_numpy(), feats_tr["s1_id"]
        folds = group_folds(groups, config.N_FOLDS)
        oof, fold_models = train_oof(X, y, groups, verbose=False)
        oof_scored = feats_tr[["s1_id", "cand_id"]].assign(prob=oof)
        tau, stau, oof_f05, _grid = tune_threshold(oof_scored, truth, tr_ids)
        model = train_full(X, y, fold_models=fold_models)
        stability = _fold_stability(oof_scored, folds, tau, stau, truth)

        sc_ho = feats_ho[["s1_id", "cand_id"]].assign(prob=predict(model, feats_ho[cols]))
        pred = apply_threshold(sc_ho, tau, ho_ids, stau)
        report = score_report(pred, truth, ho_ids)

    return {
        "held_out_country": held_out_country,
        "n_train_s1": len(tr_ids), "n_valid_s1": len(ho_ids),
        "n_other_train": len(tr["others"]), "n_other_valid": len(ho["others"]),
        "n_pairs": block_stats["n_pairs"],
        "block_pair_recall": block_stats["pair_recall"],
        "block_s1_full_recall": block_stats["s1_full_recall"],
        "block_mean_candidates": block_stats["mean_candidates"],
        "block_max_candidates": block_stats["max_candidates"],
        "oof_macro_f05": oof_f05,
        "macro_f05": report["macro_f05"],
        "precision": report["pair_precision"], "recall": report["pair_recall"],
        "singleton_f05": report["singleton_f05"], "non_singleton_f05": report["non_singleton_f05"],
        "tau": tau, "singleton_tau": stau,
        "fold_f05": ";".join(f"{v:.4f}" for v in stability["fold_f05"]),
        "fold_f05_mean": stability["fold_f05_mean"], "fold_f05_std": stability["fold_f05_std"],
        "runtime_seconds": rt.report["runtime_s"],
        "peak_rss_bytes": rt.report["peak_rss_bytes"],
        "current_rss_bytes": rt.report["current_rss_bytes"],
    }


def run_loco_comparison(sample: float = 0.0045, use_embeddings: bool = True) -> pd.DataFrame:
    """LOCO country-generalization comparison: ``LOCO_CONFIGS``' "baseline" (20/20)
    vs "both40" (40/40) candidate K, each evaluated identically (same normalisation,
    embedding model, feature definitions, LightGBM hyperparameters, threshold/
    singleton-threshold tuning mechanism, one-to-one ON, decision order, sample and
    seed -- see :data:`LOCO_CONFIGS`'s docstring) across every country-held-out
    direction actually present in the sampled training data.

    France is never a training country (CLAUDE.md: train has only US/India, France
    is test-only), so it can never appear as a LOCO direction here -- this is
    verified against the loaded sample's own country set (not assumed) and recorded
    explicitly in the JSON sidecar's ``limitations`` list, per the task's requirement
    to document rather than fabricate a France result.

    Writes ``artifacts/diagnostics/loco_comparison_<timestamp>.tsv`` (one row per
    ``(configuration, held_out_country)``) and a sidecar
    ``loco_comparison_<timestamp>.json`` (the exact configurations, directions,
    skipped directions, limitations, :data:`LOCO_LEAKAGE_CHECKS`, git commit,
    dataset hashes, env info and the full row detail). Returns the same frame. Never
    touches ``output/``, never modifies ``config.py``/``run_pipeline.py``/
    ``blocking.py``/``model.py``/``features.py``/``decide.py``/``evaluate.py``, and
    never runs ``--mode test``.
    """
    _log(f"LOCO comparison: sample={sample}, configs={[c['name'] for c in LOCO_CONFIGS]}")
    s1, s2, s3, truth = rp._load_train(sample)
    prep = rp.prepare(s1, s2, s3, use_embeddings=use_embeddings, use_cache=True)
    countries = sorted(set(prep["s1"][config.COUNTRY_COL]))
    directions = [c for c in countries if (prep["s1"][config.COUNTRY_COL] != c).any()]
    skipped = [c for c in countries if c not in directions]

    limitations = [
        f"Countries present in this sample's training data: {countries}. LOCO "
        "directions were computed only for these -- France has no labeled training "
        "records in the train split at all (CLAUDE.md: train is US/India only, "
        "France is test-only), so it cannot be evaluated as a supervised LOCO "
        "holdout (there is no remaining-country model to train it against, and no "
        "France ground truth to score it with). No France LOCO result is reported "
        "or implied by this experiment.",
    ]
    if skipped:
        limitations.append(
            f"Skipped as LOCO holdouts (no other training country remained to fit "
            f"on): {skipped}.")

    rows: list[dict] = []
    for cfg in LOCO_CONFIGS:
        k_overrides = {k: v for k, v in cfg.items() if k != "name"}
        for c in directions:
            row = _loco_fold(prep, truth, k_overrides, c)
            row["configuration"] = cfg["name"]
            row.update({f"k_{k}": v for k, v in k_overrides.items()})
            row["sample"] = sample
            rows.append(row)
            _log(f"LOCO {cfg['name']} held-out={c}: macro F0.5={row['macro_f05']:.4f} "
                f"(OOF={row['oof_macro_f05']:.4f}, "
                f"block_pair_recall={row['block_pair_recall']:.4f})")

    out = pd.DataFrame(rows)
    ts = time.strftime("%Y%m%d_%H%M%S")
    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    out_path = DIAG_DIR / f"loco_comparison_{ts}.tsv"
    out.to_csv(out_path, sep="\t", index=False)
    meta = {
        "kind": "loco_comparison", "timestamp": ts, "sample": sample,
        "use_embeddings": use_embeddings,
        "configs": LOCO_CONFIGS, "directions": directions, "skipped_directions": skipped,
        "countries_in_sample": countries,
        "limitations": limitations,
        "leakage_checks": LOCO_LEAKAGE_CHECKS,
        "git_commit": rp._git_hash(),
        "dataset_file_hashes": diag.dataset_file_hashes(config.TRAIN_FILES),
        "env": diag.env_info(),
        "rows": rows,
    }
    diag.save_json(meta, DIAG_DIR / f"loco_comparison_{ts}.json")
    _log(f"LOCO comparison: {len(rows)} rows written to {out_path}")
    return out


# --- Final pre-lock diagnostic: feature audit + hard-negative overlap ----------------------

#: One-line, hand-authored meaning for every feature ``features.build_features`` can
#: produce, keyed by exact column name. Read once against ``features.py`` (see that
#: module's ``pair_features``/``blocking_features``/``context_features``) rather than
#: derived at runtime, so the audit describes what the code *means* to compute, not
#: just its numeric shape. A feature absent from this dict (e.g. after a future
#: features.py change) is still audited below, flagged as unannotated rather than
#: silently skipped or crashing.
FEATURE_MEANINGS: dict[str, str] = {
    "name_jw": "Jaro-Winkler similarity of name_core (prefix-weighted edit distance).",
    "name_ratio": "rapidfuzz simple ratio of name_core (overall edit-distance similarity).",
    "name_token_set": "Token-set ratio of name_core (robust to token reordering/duplication).",
    "name_token_sort": "Token-sort ratio of name_core (robust to token reordering only).",
    "name_partial": "Partial (substring) ratio of name_core.",
    "name_full_token_set": "Token-set ratio of the fuller name_norm (suffix/alias retained).",
    "name_compact_ratio": "Ratio of name_compact (whitespace/punctuation-stripped name).",
    "name_compact_partial": "Partial ratio of name_compact.",
    "name_tfidf_cos": "Char 3-4-gram TF-IDF cosine of name_core (vocabulary-weighted).",
    "name_core_exact": "1.0 iff name_core strings are byte-identical, else 0.0.",
    "name_compact_exact": "1.0 iff name_compact strings are byte-identical, else 0.0.",
    "suffix_equal": "1.0 iff the raw legal-suffix token is identical (0.0 if both present but differ, NaN handled via suffix_one_missing).",
    "suffix_both": "1.0 iff both sides carry a recognised legal-suffix family.",
    "suffix_conflict": "1.0 iff both sides have a suffix family and they disagree (e.g. LLC vs Inc).",
    "suffix_one_missing": "1.0 iff exactly one side has a legal suffix and the other has none.",
    "acronym_match": "1.0 iff one side's acronym equals the other side's compact name (either direction).",
    "name_token_jaccard": "Jaccard overlap of name_core's whitespace tokens (0.0, not NaN, when either side is empty).",
    "name_len_ratio": "min(len)/max(len) of name_core -- a length-symmetry proxy.",
    "first_token_equal": "1.0/0.0/NaN: whether the first name token matches (NaN if either side has none).",
    "cand_is_website": "1.0 iff the candidate's business_name was flagged as a bare URL/website string.",
    "cand_has_dba": "1.0 iff the candidate carries a parsed 'doing business as' alias.",
    "addr_empty_s1": "1.0 iff the S1 record's normalised address is empty.",
    "addr_empty_cand": "1.0 iff the candidate's normalised address is empty.",
    "addr_ratio": "rapidfuzz ratio of addr_norm; NaN if either side's address is empty.",
    "addr_token_set": "Token-set ratio of addr_norm; NaN if either side's address is empty.",
    "addr_token_sort": "Token-sort ratio of addr_norm; NaN if either side's address is empty.",
    "addr_partial": "Partial ratio of addr_norm; NaN if either side's address is empty.",
    "addr_tfidf_cos": "Char 3-4-gram TF-IDF cosine of addr_norm; NaN if either side is empty.",
    "addr_token_jaccard": "Jaccard overlap of addr_norm's tokens; NaN if either side has none.",
    "digit_jaccard": "Jaccard overlap of extracted digit tokens (house/PIN/ZIP numbers); NaN if either side has none.",
    "digit_shared": "Count of digit tokens shared between the two addresses.",
    "digit_conflict": "1.0 iff both sides have digit tokens and share none (NaN if either side has no digits at all).",
    "digit_cand_subset": "1.0 iff the candidate's digit tokens are a subset of the S1's (NaN if either side has no digits).",
    "house_equal": "1.0/0.0/NaN: whether the parsed house number matches.",
    "long_num_equal": "1.0 iff any long numeric token (e.g. a full PIN/ZIP) is shared; NaN if either side has none.",
    "landmark_jaccard": "Jaccard overlap of parsed landmark-phrase tokens ('near X', 'opp Y').",
    "region_equal": "1.0/0.0/NaN: whether a parsed region/state token matches.",
    "same_country": "1.0 iff the two records' raw country strings are identical (the only country-derived feature; CLAUDE.md §2.4).",
    "emb_cos": "Cosine similarity of the multilingual sentence-embedding vectors (name+address); NaN if embeddings were not computed.",
    "in_tfidf": "1.0 iff this pair was produced by the TF-IDF name-cosine blocking pass.",
    "in_embed": "1.0 iff this pair was produced by the dense-embedding blocking pass.",
    "in_rare": "1.0 iff this pair was produced by the rare (high-IDF) shared-token blocking pass.",
    "in_digit": "1.0 iff this pair was produced by the shared-postal/number-token blocking pass.",
    "in_address": "1.0 iff this pair was produced by the address-token blocking pass.",
    "block_tfidf": "That pass's own cosine/match score, if the pair passed through it (else NaN).",
    "block_embed": "That pass's own cosine/match score, if the pair passed through it (else NaN).",
    "block_rare": "That pass's own cosine/match score, if the pair passed through it (else NaN).",
    "block_digit": "That pass's own cosine/match score, if the pair passed through it (else NaN).",
    "block_address": "That pass's own cosine/match score, if the pair passed through it (else NaN).",
    "n_passes": "Count of distinct blocking passes that surfaced this pair (1-5).",
    "pair_sim": "Ranking score used for context features: mean of name/address token-set ratio (name-only if address is missing).",
    "rank_in_s1": "This candidate's rank (1=best) among all candidates blocked for its S1.",
    "gap_to_best_s1": "pair_sim gap between this candidate and the S1's single best-scoring candidate.",
    "n_cands_s1": "Total candidate count blocked for this S1 (neighbourhood size).",
    "rank_in_cand": "This S1's rank (1=best) among all S1s competing for the same candidate (reverse view).",
    "gap_to_best_cand": "pair_sim gap between this S1 and the candidate's single best-scoring competing S1.",
    "n_s1_for_cand": "Count of distinct S1s that blocked this same candidate (orphan/contention signal).",
    "is_s3": "1.0 iff the candidate is from Source 3, 0.0 if Source 2.",
}

#: Feature -> family, read against CLAUDE.md §6.3's grouping plus the task's Part 5
#: buckets. Assigned by hand against ``features.py`` rather than inferred from name
#: prefixes at runtime, since a couple of features (``name_tfidf_cos``,
#: ``addr_tfidf_cos``, ``digit_*``/``house_equal``/``long_num_equal``) belong to a
#: more specific family (TF-IDF, NUMERIC/DIGIT) than a naive "name_"/"addr_" prefix
#: split would give them.
FEATURE_FAMILIES: dict[str, str] = {
    **{f: "TF-IDF" for f in ("name_tfidf_cos", "addr_tfidf_cos")},
    **{f: "EMBEDDING" for f in ("emb_cos",)},
    **{f: "NUMERIC/DIGIT" for f in (
        "digit_jaccard", "digit_shared", "digit_conflict", "digit_cand_subset",
        "house_equal", "long_num_equal")},
    **{f: "COUNTRY" for f in ("same_country",)},
    **{f: "LENGTH/MISSINGNESS" for f in (
        "name_len_ratio", "addr_empty_s1", "addr_empty_cand")},
    **{f: "NAME" for f in (
        "name_jw", "name_ratio", "name_token_set", "name_token_sort", "name_partial",
        "name_full_token_set", "name_compact_ratio", "name_compact_partial",
        "name_core_exact", "name_compact_exact", "suffix_equal", "suffix_both",
        "suffix_conflict", "suffix_one_missing", "acronym_match", "name_token_jaccard",
        "first_token_equal", "cand_is_website", "cand_has_dba")},
    **{f: "ADDRESS" for f in (
        "addr_ratio", "addr_token_set", "addr_token_sort", "addr_partial",
        "addr_token_jaccard", "landmark_jaccard", "region_equal")},
    **{f: "OTHER" for f in (
        "in_tfidf", "in_embed", "in_rare", "in_digit", "in_address",
        "block_tfidf", "block_embed", "block_rare", "block_digit", "block_address",
        "n_passes", "pair_sim", "rank_in_s1", "gap_to_best_s1", "n_cands_s1",
        "rank_in_cand", "gap_to_best_cand", "n_s1_for_cand", "is_s3")},
}

#: Coarse value-shape per feature, for Part 1's "data type/range" column. Features not
#: listed here (e.g. a future features.py addition) are reported as "unknown (not yet
#: annotated)" rather than guessed.
_RANGE_UNIT_NAN = "float32 in [0, 1] (NaN if either side's field is empty)"
_RANGE_UNIT = "float32 in [0, 1]"
_RANGE_FLAG_NAN = "{0.0, 1.0} flag (NaN if not applicable to this pair)"
_RANGE_FLAG = "{0.0, 1.0} flag"
_RANGE_COUNT = "non-negative integer count (unbounded)"
_RANGE_RANK = "positive integer rank, 1 = best (unbounded)"
_RANGE_GAP = "float >= 0, unbounded (0 = tied for best)"
FEATURE_RANGES: dict[str, str] = {
    **{f: _RANGE_UNIT for f in (
        "name_jw", "name_ratio", "name_token_set", "name_token_sort", "name_partial",
        "name_full_token_set", "name_compact_ratio", "name_compact_partial",
        "name_tfidf_cos", "name_token_jaccard", "name_len_ratio", "addr_tfidf_cos")},
    **{f: _RANGE_UNIT_NAN for f in (
        "addr_ratio", "addr_token_set", "addr_token_sort", "addr_partial",
        "addr_token_jaccard", "digit_jaccard", "landmark_jaccard")},
    **{f: _RANGE_FLAG for f in (
        "name_core_exact", "name_compact_exact", "acronym_match", "cand_is_website",
        "cand_has_dba", "addr_empty_s1", "addr_empty_cand", "same_country", "is_s3",
        "suffix_both")},
    **{f: _RANGE_FLAG_NAN for f in (
        "suffix_equal", "suffix_conflict", "suffix_one_missing", "first_token_equal",
        "digit_conflict", "digit_cand_subset", "house_equal", "long_num_equal",
        "region_equal", "in_tfidf", "in_embed", "in_rare", "in_digit", "in_address")},
    **{f: _RANGE_COUNT for f in (
        "digit_shared", "n_passes", "n_cands_s1", "n_s1_for_cand")},
    **{f: _RANGE_RANK for f in ("rank_in_s1", "rank_in_cand")},
    **{f: _RANGE_GAP for f in ("gap_to_best_s1", "gap_to_best_cand")},
    "pair_sim": _RANGE_UNIT_NAN,
    "emb_cos": "float32, typically in [-1, 1] (cosine of unit-normalised embeddings); NaN if embeddings disabled",
    **{f: "float32 in [0, 1] (NaN if this pair never passed that blocking pass)"
       for f in ("block_tfidf", "block_embed", "block_rare", "block_digit", "block_address")},
}

#: Feature-name groups whose members are near-duplicate similarity measures of the
#: *same* underlying string pair (documented by inspection of features.py, not
#: inferred from a correlation threshold) -- i.e. a-priori redundancy candidates the
#: model may not need all of. Membership here does NOT by itself mean a feature is
#: useless (LightGBM's gain importance in Part 5 is the actual evidence); it only
#: flags where redundancy is architecturally plausible so the report can cross-check
#: it against measured correlation and importance.
_REDUNDANCY_GROUPS: list[tuple[str, ...]] = [
    ("name_jw", "name_ratio", "name_token_set", "name_token_sort", "name_partial",
     "name_full_token_set", "name_compact_ratio", "name_compact_partial", "name_tfidf_cos"),
    ("name_core_exact", "name_compact_exact"),
    ("addr_ratio", "addr_token_set", "addr_token_sort", "addr_partial", "addr_tfidf_cos",
     "addr_token_jaccard"),
    ("digit_jaccard", "digit_shared", "digit_cand_subset"),
    ("rank_in_s1", "gap_to_best_s1"), ("rank_in_cand", "gap_to_best_cand"),
]


def _quantiles(x: np.ndarray) -> dict:
    """p50/p75/p90/p95/p99/max/mean/n of a 1-D array; all-null summary if empty."""
    x = np.asarray(x, dtype=float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return {"n": 0, "p50": None, "p75": None, "p90": None, "p95": None,
                "p99": None, "max": None, "mean": None}
    return {
        "n": int(len(x)),
        "p50": float(np.quantile(x, 0.50)), "p75": float(np.quantile(x, 0.75)),
        "p90": float(np.quantile(x, 0.90)), "p95": float(np.quantile(x, 0.95)),
        "p99": float(np.quantile(x, 0.99)), "max": float(x.max()), "mean": float(x.mean()),
    }


def build_feature_audit(feats: pd.DataFrame, model) -> pd.DataFrame:
    """Part 1 + part of Part 5: one row per model input feature.

    Combines the hand-authored meaning/family/range annotations above with two
    numbers actually measured from this run: each feature's Pearson correlation
    with the binary label (a cheap, model-free discrimination signal -- separate
    from, and a cross-check on, the LightGBM gain importance computed later) and
    its maximum absolute correlation with any *other* feature (the empirical
    redundancy check, cross-referenced against ``_REDUNDANCY_GROUPS``).
    """
    cols = feature_columns(feats)
    label = feats["label"].to_numpy(dtype=float)
    num = feats[cols].to_numpy(dtype=float)
    label_corr: dict[str, float] = {}
    with np.errstate(invalid="ignore", divide="ignore"):
        for i, c in enumerate(cols):
            col = num[:, i]
            mask = ~np.isnan(col)
            if mask.sum() < 2 or np.nanstd(col) == 0:
                label_corr[c] = float("nan")
                continue
            corr = np.corrcoef(col[mask], label[mask])[0, 1]
            label_corr[c] = float(corr) if np.isfinite(corr) else float("nan")
        corr_matrix = pd.DataFrame(num, columns=cols).corr().to_numpy()
    max_other_corr: dict[str, float] = {}
    for i, c in enumerate(cols):
        row = np.delete(corr_matrix[i], i)
        row = row[~np.isnan(row)]
        max_other_corr[c] = float(np.abs(row).max()) if len(row) else float("nan")

    redundancy_group = {}
    for group in _REDUNDANCY_GROUPS:
        for f in group:
            redundancy_group.setdefault(f, ";".join(g for g in group))

    rows = []
    for c in cols:
        rows.append({
            "feature": c,
            "family": FEATURE_FAMILIES.get(c, "OTHER"),
            "meaning": FEATURE_MEANINGS.get(c, "(not yet annotated -- new column, see features.py)"),
            "dtype_range": FEATURE_RANGES.get(c, "unknown (not yet annotated)"),
            "derivation": "raw" if c in ("same_country", "is_s3") else "derived",
            "label_correlation": label_corr.get(c),
            "max_abs_correlation_with_other_feature": max_other_corr.get(c),
            "a_priori_redundancy_group": redundancy_group.get(c, ""),
            "empirically_redundant": bool(max_other_corr.get(c) is not None
                                          and not np.isnan(max_other_corr.get(c, np.nan))
                                          and max_other_corr[c] >= 0.90),
        })
    audit = pd.DataFrame(rows)
    imp = feature_importance(model).set_index("feature")
    audit["gain_importance"] = audit["feature"].map(imp["gain"]).fillna(0.0)
    audit["split_importance"] = audit["feature"].map(imp["split"]).fillna(0.0)
    return audit.sort_values("gain_importance", ascending=False).reset_index(drop=True)


def _family_summary(audit: pd.DataFrame) -> pd.DataFrame:
    """Part 5: total/share of gain importance per feature family."""
    g = audit.groupby("family")["gain_importance"].sum().sort_values(ascending=False)
    total = g.sum()
    return pd.DataFrame({
        "family": g.index, "gain_importance": g.to_numpy(),
        "gain_share": (g / total).to_numpy() if total else np.zeros(len(g)),
    }).reset_index(drop=True)


def run_feature_hard_negative_audit(sample: float = 0.0045,
                                    use_embeddings: bool = True) -> dict:
    """Final pre-lock diagnostic: feature discrimination audit + hard-negative score
    overlap, run once on the "both40" (K_TFIDF_NAME=K_EMBEDDING=40) candidate
    configuration -- the leading candidate the task names -- against the existing
    validation split and existing pipeline functions only (measurement-only; changes
    no config default, no feature, no model, and no decision logic).

    Reuses, unmodified: ``blocking.generate_candidates`` (both40 k_overrides, same
    pattern as :func:`run_blocking_k_downstream`), ``features.build_features``/
    ``label_pairs``, :func:`_fit_and_tune_with_oof` (GroupKFold OOF fit + threshold
    tuning, train-side only), ``decide.apply_threshold`` (production one-to-one ON),
    ``evaluate.score_report``, and ``error_decomposition``'s
    ``classify_true_pairs``/``false_positive_report``/``slice_report`` -- the same
    stage classification :func:`run_error_decomposition` already uses to separate
    blocking failures from matcher failures, which is exactly Part 3/6's question.

    Writes ``artifacts/diagnostics/feature_hard_negative_audit_<ts>.json`` (full
    report incl. the feature audit table, score quantiles, gate booleans and the
    A/B recommendation) and a companion ``..._<ts>.tsv`` (the feature audit table,
    for spreadsheet inspection). Returns the JSON-serialisable report dict. Never
    touches ``output/``, never runs ``--mode test``, never changes a ``config.py``
    default.
    """
    ts = time.strftime("%Y%m%d_%H%M%S")
    _log(f"feature/hard-negative audit: sample={sample}, config=both40")
    s1, s2, s3, truth = rp._load_train(sample)
    prep = rp.prepare(s1, s2, s3, use_embeddings=use_embeddings, use_cache=True)
    s1_ids = list(prep["s1"][config.ID_COL])
    s1_country = dict(zip(prep["s1"][config.ID_COL], prep["s1"][config.COUNTRY_COL]))
    n_other = len(prep["others"])
    train_ids, valid_ids = rp.split_s1(s1_ids, truth)

    both40 = next(c for c in blocking_k_downstream_configs() if c["name"] == "both40")
    k_overrides = {k: v for k, v in both40.items() if k != "name"}
    cands = generate_candidates(prep["s1"], prep["others"], embeddings=prep["emb"],
                                verbose=False, k_overrides=k_overrides)
    block_stats = report_blocking_stats(cands, truth, s1_ids, n_other, s1_country, verbose=False)

    feats = build_features(cands, prep["s1"], prep["others"], prep["emb"])
    feats["label"] = label_pairs(feats, truth)
    in_train = feats["s1_id"].isin(set(train_ids)).to_numpy()
    cols = feature_columns(feats)
    fit = _fit_and_tune_with_oof(feats[in_train], truth, train_ids)
    tau, singleton_tau = fit["tau"], fit["singleton_tau"]

    valid_feats = feats[~in_train].reset_index(drop=True)
    valid_scored = valid_feats[["s1_id", "cand_id", "label"]].assign(
        prob=predict(fit["model"], valid_feats[cols]))
    valid_truth = {s: truth.get(s, []) for s in valid_ids}
    pred = apply_threshold(valid_scored, tau, valid_ids, singleton_tau)
    valid_report = score_report(pred, valid_truth, valid_ids, groups=s1_country)

    # --- Part 1: feature audit (measured against the training-side fit) ------------
    audit = build_feature_audit(feats[in_train], fit["model"])
    family_summary = _family_summary(audit)

    # --- Part 2/3/4: reuse the existing error-decomposition primitives, exactly as
    # run_error_decomposition does, but on this both40-configured validation split.
    classified = edecomp.classify_true_pairs(
        prep["s1"], prep["others"], cands, valid_scored, pred, valid_truth, tau)
    fp = edecomp.false_positive_report(valid_scored, pred, valid_truth)
    slices = edecomp.slice_report(classified, prep["s1"], prep["others"], cands, s1_country)

    merged = valid_scored.merge(valid_feats[["s1_id", "cand_id", *cols]],
                                on=["s1_id", "cand_id"], how="left")
    pos = merged[merged["label"] == 1]
    neg = merged[merged["label"] == 0]

    # --- Part 2: hard-negative score overlap ----------------------------------------
    hard_neg_cuts = {"score_ge_0.5": 0.5, "score_ge_0.7": 0.7,
                     "score_ge_tau": tau, "score_ge_0.9": 0.9}
    hard_negative_stats = {
        name: {"threshold": thr, **_quantiles(neg.loc[neg["prob"] >= thr, "prob"].to_numpy())}
        for name, thr in hard_neg_cuts.items()
    }
    score_overlap = {
        "true_positive_scores": _quantiles(pos["prob"].to_numpy()),
        "true_negative_scores": _quantiles(neg["prob"].to_numpy()),
        "hard_negatives": hard_negative_stats,
        "negatives_at_or_above_tau": int((neg["prob"] >= tau).sum()),
        "positives_at_or_above_tau": int((pos["prob"] >= tau).sum()),
        "negatives_within_0.05_of_tau_below": int(
            ((neg["prob"] >= tau - 0.05) & (neg["prob"] < tau)).sum()),
        "positives_within_0.05_of_tau_above": int(
            ((pos["prob"] < tau + 0.05) & (pos["prob"] >= tau)).sum()),
    }

    # --- Part 3: true matches that are hard to score --------------------------------
    miss_cuts = {"below_0.5": 0.5, "below_0.7": 0.7, "below_tau": tau}
    missed_positive_stats = {}
    for name, thr in miss_cuts.items():
        missed = pos[pos["prob"] < thr]
        by_country = (missed.assign(country=missed["s1_id"].map(s1_country))
                     .groupby("country").size().to_dict())
        by_source = missed.assign(
            source=np.where(missed["cand_id"].str.startswith("S3-"), "S3", "S2")
        ).groupby("source").size().to_dict()
        missed_positive_stats[name] = {
            "n": int(len(missed)),
            "share_of_true_positives_in_candidates": (
                float(len(missed) / len(pos)) if len(pos) else None),
            "by_country": by_country, "by_source": by_source,
            "mean_n_cands_s1": float(missed["n_cands_s1"].mean()) if len(missed) else None,
            "mean_addr_empty_s1": float(missed["addr_empty_s1"].mean()) if len(missed) else None,
            "mean_addr_empty_cand": float(missed["addr_empty_cand"].mean()) if len(missed) else None,
            "mean_name_core_exact": float(missed["name_core_exact"].mean()) if len(missed) else None,
        }
    # Rejected-by-matcher despite being a candidate: exactly edecomp's own
    # "matcher_false_negative" stage -- true matches that entered the candidate set
    # (so blocking succeeded) but scored below tau (so the matcher, not blocking, is
    # responsible). This is the one number Part 6's gate condition 4 hinges on.
    stage_counts = (classified["stage"].value_counts().to_dict() if len(classified) else {})

    # --- Part 4: false positives at tau ---------------------------------------------
    fp_merged = fp.merge(valid_feats[["s1_id", "cand_id", *cols]], on=["s1_id", "cand_id"],
                        how="left") if len(fp) else fp.assign(**{c: [] for c in cols})
    fp_stats = {
        "n_false_positives": int(len(fp)),
        "n_top_wrong": int(fp["top_wrong"].sum()) if len(fp) else 0,
        "by_source": (np.where(fp_merged["cand_id"].str.startswith("S3-"), "S3", "S2")
                      if len(fp_merged) else np.array([])),
        "mean_name_token_set": float(fp_merged["name_token_set"].mean()) if len(fp_merged) else None,
        "mean_addr_token_set": float(fp_merged["addr_token_set"].mean()) if len(fp_merged) else None,
        "mean_digit_shared": float(fp_merged["digit_shared"].mean()) if len(fp_merged) else None,
        "mean_n_cands_s1": float(fp_merged["n_cands_s1"].mean()) if len(fp_merged) else None,
        "mean_same_country": float(fp_merged["same_country"].mean()) if len(fp_merged) else None,
    }
    fp_stats["by_source"] = (pd.Series(fp_stats["by_source"]).value_counts().to_dict()
                             if len(fp_merged) else {})
    fp_country = (fp_merged.assign(country=fp_merged["s1_id"].map(s1_country))
                 .groupby("country").size().to_dict()) if len(fp_merged) else {}

    # --- Part 6: hard-negative gate ---------------------------------------------------
    n_true_pairs = len(classified)
    n_in_candidates = int(sum(v for k, v in stage_counts.items()
                              if k in ("correct_match", "matcher_false_negative",
                                       "decision_false_negative")))
    n_matcher_fn = int(stage_counts.get("matcher_false_negative", 0))
    n_blocking_loss = int(stage_counts.get("blocking_false_negative", 0)
                          + stage_counts.get("representation_failure", 0))
    n_hard_neg_at_tau = int(hard_negative_stats["score_ge_tau"]["n"])

    gate = {
        "condition_1_material_true_matches_in_candidates": {
            "value": n_in_candidates >= 20,
            "detail": f"{n_in_candidates} of {n_true_pairs} true pairs entered the "
                      f"both40 candidate set (blocking succeeded)."},
        "condition_2_systematically_rejected_by_matcher": {
            "value": n_matcher_fn >= 10 and (n_matcher_fn / n_in_candidates >= 0.05
                                             if n_in_candidates else False),
            "detail": f"{n_matcher_fn} true pairs entered the candidate set but scored "
                      f"below tau={tau:.3f} ('matcher_false_negative')."},
        "condition_3_hard_negatives_overlap_near_threshold": {
            "value": n_hard_neg_at_tau >= 10,
            "detail": f"{n_hard_neg_at_tau} negative pairs scored at or above the "
                      f"selected tau={tau:.3f}."},
        "condition_4_weakness_is_matcher_not_blocking": {
            "value": n_matcher_fn > n_blocking_loss,
            "detail": f"matcher_false_negative={n_matcher_fn} vs. "
                      f"blocking-caused misses (blocking_false_negative+"
                      f"representation_failure)={n_blocking_loss}."},
    }
    all_conditions_hold = all(v["value"] for v in gate.values())
    recommendation = "B" if all_conditions_hold else "A"
    conclusion = (
        "Remaining true-match losses are still dominated by blocking (blocking_false_negative"
        f"+representation_failure={n_blocking_loss} vs. matcher_false_negative={n_matcher_fn})."
        if not all_conditions_hold else
        "True matches are entering the candidate set in material numbers and being "
        "systematically rejected by the matcher, with hard negatives overlapping the "
        "positive score range near tau -- a targeted matcher/feature change is justified."
    )

    report = {
        "kind": "feature_hard_negative_audit", "timestamp": ts, "sample": sample,
        "use_embeddings": use_embeddings, "git_commit": rp._git_hash(),
        "dataset_file_hashes": diag.dataset_file_hashes(config.TRAIN_FILES),
        "env": diag.env_info(),
        "blocking_configuration": {"name": "both40", **k_overrides},
        "threshold_configuration": {"tau": tau, "singleton_tau": singleton_tau,
                                    "source": "tuned on OOF predictions, train-side only "
                                              "(decide.tune_threshold via _fit_and_tune_with_oof), "
                                              "not hardcoded"},
        "n_s1": len(s1_ids), "n_other": n_other,
        "n_train_s1": len(train_ids), "n_valid_s1": len(valid_ids),
        "block_stats": block_stats,
        "n_candidate_pairs": int(len(cands)),
        "n_positive_pairs_valid": int(len(pos)), "n_negative_pairs_valid": int(len(neg)),
        "oof_macro_f05": fit["oof_f05"], "valid_report": valid_report,
        "score_overlap": score_overlap,
        "missed_positive_stats": missed_positive_stats,
        "stage_counts": stage_counts,
        "false_positive_stats": {**fp_stats, "by_country": fp_country},
        "feature_family_summary": family_summary.to_dict(orient="records"),
        "gate": gate,
        "recommendation": recommendation,
        "conclusion": conclusion,
    }

    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    json_path = DIAG_DIR / f"feature_hard_negative_audit_{ts}.json"
    tsv_path = DIAG_DIR / f"feature_hard_negative_audit_{ts}.tsv"
    diag.save_json(report, json_path)
    audit.to_csv(tsv_path, sep="\t", index=False)
    _log(f"feature/hard-negative audit written to {json_path} and {tsv_path}")
    _log(f"gate: {gate}")
    _log(f"RECOMMENDATION: {recommendation} -- {conclusion}")
    report["_audit_table_path"] = str(tsv_path)
    report["_json_path"] = str(json_path)
    return report


# --- blocking-FN pass attribution ---------------------------------------------------------

#: The five production blocking passes, in the order the CLI reports them.
_PASS_NAMES = ("tfidf", "embed", "rare", "digit", "address")

#: Every ``blocker_status`` value the report can emit, in output order.
_STATUS_ORDER = ("NONE", "TFIDF_ONLY", "EMBED_ONLY", "RARE_ONLY", "DIGIT_ONLY",
                 "ADDRESS_ONLY", "MULTIPLE")


def _blocking_passes_by_country(
    s1: pd.DataFrame, others: pd.DataFrame,
    embeddings: tuple[np.ndarray, np.ndarray] | None,
    within_country: bool = config.BLOCK_WITHIN_COUNTRY,
    use_address_pass: bool = config.USE_ADDRESS_PASS,
) -> dict[str, dict[str, set[str]]]:
    """Run every individual production blocking pass, per country group, at its
    production ``config.K_*`` default -- i.e. exactly the per-group ``runs`` dict
    ``blocking.generate_candidates`` itself builds and calls, unmodified, except that
    each pass's own output is kept separate here instead of being folded into the
    bitmask union ``generate_candidates`` returns. Uses the same
    ``blocking.country_groups`` grouping production blocking uses, so which S1/
    candidate pairs are even eligible for a given pass is identical to production.

    Returns ``{pass_name: {s1_id: {cand_id, ...}}}`` -- ``"embed"`` is omitted
    entirely when ``embeddings`` is ``None`` and ``"address"`` when
    ``use_address_pass`` is ``False``, exactly mirroring which passes
    ``generate_candidates`` itself would have run.
    """
    s1 = s1.reset_index(drop=True)
    others = others.reset_index(drop=True)
    s1_ids_all = s1[config.ID_COL].to_numpy()
    cand_ids_all = others[config.ID_COL].to_numpy()
    out: dict[str, dict[str, set[str]]] = {
        name: {} for name in _PASS_NAMES
        if (name != "embed" or embeddings is not None) and (name != "address" or use_address_pass)
    }
    for _country, q_idx, x_idx in country_groups(s1, others, within_country):
        if len(x_idx) == 0:
            continue
        q = s1.iloc[q_idx].reset_index(drop=True)
        x = others.iloc[x_idx].reset_index(drop=True)
        runs = {
            "tfidf": lambda: tfidf_name_pass(q["name_core"], x["name_core"], k=config.K_TFIDF_NAME),
            "rare": lambda: rare_token_pass(q, x, k=config.K_RARE_TOKEN),
            "digit": lambda: digit_token_pass(q, x, k=config.K_POSTAL_TOKEN),
        }
        if embeddings is not None:
            runs["embed"] = lambda: embedding_pass(
                np.asarray(embeddings[0][q_idx]), np.asarray(embeddings[1][x_idx]),
                k=config.K_EMBEDDING)
        if use_address_pass:
            runs["address"] = lambda: address_pass(q, x, k=config.K_ADDRESS)
        for name, run in runs.items():
            res = run()
            if len(res) == 0:
                continue
            s1g = s1_ids_all[q_idx[res["q"].to_numpy()]]
            candg = cand_ids_all[x_idx[res["x"].to_numpy()]]
            bucket = out[name]
            for sid, cid in zip(s1g, candg):
                bucket.setdefault(sid, set()).add(cid)
    return out


def _clean_blocking_fn_pairs(
    sample: float, use_embeddings: bool
) -> tuple[pd.DataFrame, pd.DataFrame, dict, dict[str, str]]:
    """The exact "clean" blocking-false-negative pair set both
    ``blocking-fn-attribution`` and ``blocking-fn-forensics`` inspect: every true
    (S1, candidate) pair ``error_decomposition.classify_true_pairs`` stages as
    ``blocking_false_negative`` (absent from the production candidate union) with
    every one of ``error_decomposition.LOSS_KEYS``'s six representation-loss flags
    False.

    Factored out of ``run_blocking_fn_attribution`` (this is exactly the computation
    it used to do inline, unchanged) so ``run_blocking_fn_forensics`` inspects
    literally the same 97-pair set that command reports on a given ``--sample``, from
    the same deterministic sample and the same production-default classification
    path, rather than an independently regenerated approximation of it.

    Returns ``(clean, cands, prep, s1_country)``: ``clean`` has ``s1_id``/``cand_id``
    plus every ``LOSS_KEYS`` column (all False by construction) and ``stage`` (always
    ``"blocking_false_negative"``); ``cands`` is the production-default candidate
    union used for that classification; ``prep`` is
    ``run_pipeline.prepare``'s output; ``s1_country`` is ``{s1_id: country}``.
    """
    s1, s2, s3, truth = rp._load_train(sample)
    ctx = run_valid_for_diagnostics(s1, s2, s3, truth, use_embeddings)
    prep, cands, scored, pred, tau = ctx["prep"], ctx["cands"], ctx["scored"], ctx["pred"], ctx["tau"]
    valid_truth = {s: truth.get(s, []) for s in ctx["valid_ids"]}
    s1_country = dict(zip(prep["s1"][config.ID_COL], prep["s1"][config.COUNTRY_COL]))

    classified = edecomp.classify_true_pairs(
        prep["s1"], prep["others"], cands, scored, pred, valid_truth, tau)

    loss_cols = list(edecomp.LOSS_KEYS)
    is_bfn = classified["stage"] == "blocking_false_negative"
    no_loss = ~classified[loss_cols].any(axis=1) if len(classified) else pd.Series(dtype=bool)
    clean = classified[is_bfn & no_loss].reset_index(drop=True)
    # Sanity check 4: every selected pair must have all six representation-loss
    # flags False -- guaranteed by the filter above; asserted explicitly so a future
    # edit to that filter can never silently include a representation-failure row.
    assert not clean[loss_cols].any().any(), \
        "clean blocking FN selection must have every representation-loss flag False"

    # Sanity check 1: every clean FN pair must be absent from the exact candidate
    # union `classify_true_pairs` itself checked against to assign the
    # "blocking_false_negative" stage in the first place.
    cand_key = set(zip(cands["s1_id"], cands["cand_id"]))
    still_in_union = any((s, c) in cand_key for s, c in zip(clean["s1_id"], clean["cand_id"]))
    assert not still_in_union, \
        "clean blocking FN pairs must be absent from the final baseline candidate union"

    return clean, cands, prep, s1_country


def run_blocking_fn_attribution(sample: float = 0.0045,
                                use_embeddings: bool = True) -> dict:
    """Measurement-only diagnostic: attribute each "clean" blocking false negative --
    a true (S1, candidate) pair that ``error_decomposition.classify_true_pairs``
    stages as ``blocking_false_negative`` (absent from the final production candidate
    union) with every one of :data:`error_decomposition.LOSS_KEYS`'s six
    representation-loss flags False -- to whichever individual production blocking
    pass(es), if any, would retrieve it on their own, at production default K.

    A true pair the production candidate union missed is, by construction, missed by
    *every* individual pass at that same production K: the union IS the union of the
    five passes' own outputs (``blocking.generate_candidates``'s ``runs`` dict), so
    this diagnostic is not expected to surface true-match recall production silently
    discarded. Its actual purpose is a consistency check on the union construction
    itself -- if a "clean" blocking false negative here IS retrieved by an individual
    pass re-run, that is a genuine discrepancy between the individual pass and the
    production union path (sanity check 2 below), worth flagging, not something this
    function papers over or asserts away.

    Reuses, unmodified: ``run_pipeline._load_train``/:func:`run_valid_for_diagnostics`
    (the same deterministic sample, normalisation, embeddings, train/valid split,
    production-default blocking union and OOF-tuned decision the ``errors``/
    ``feature-hard-negative`` diagnostics already use),
    ``error_decomposition.classify_true_pairs`` (the same stage classification and
    representation-loss flags), and every one of ``blocking``'s five pass functions
    plus ``blocking.country_groups`` (via :func:`_blocking_passes_by_country`), each
    called at its production ``config.K_*`` default -- never approximated or
    reimplemented.

    Note on the five representation-loss keys named in the task vs.
    ``error_decomposition.LOSS_KEYS``: the task's five named flags
    (``postal_or_long_num_lost``, ``house_number_lost``, ``name_token_lost``,
    ``address_token_lost``, ``other_normalization_loss``) omit
    ``script_info_lost``, but its own sanity check asks this function to assert
    "all six representation-loss flags False" -- ``LOSS_KEYS`` has exactly six
    entries. This function follows the six-flag (stricter, superset) selection
    so both statements in the task are simultaneously satisfiable: every row this
    function selects has every flag in ``LOSS_KEYS``, ``script_info_lost``
    included, False.

    Writes ``clean_blocking_fn_attribution.tsv`` (one row per clean blocking FN) and
    ``summary.tsv`` under ``artifacts/diagnostics/blocking_fn_attribution_<ts>/``
    (plus a ``meta.json`` sidecar, mirroring this module's other diagnostics). Never
    touches ``output/``, never changes a ``config.py`` default, and never re-tunes or
    re-fits anything beyond what :func:`run_valid_for_diagnostics` already does.
    """
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = DIAG_DIR / f"blocking_fn_attribution_{ts}"
    _log(f"blocking FN attribution: sample={sample}")

    clean, cands, prep, s1_country = _clean_blocking_fn_pairs(sample, use_embeddings)
    _log(f"blocking FN attribution: {len(clean)} clean blocking FNs "
        f"(stage=blocking_false_negative, all {len(edecomp.LOSS_KEYS)} "
        f"representation-loss flags False)")

    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    detail_path = out_dir / "clean_blocking_fn_attribution.tsv"
    summary_path = out_dir / "summary.tsv"

    if len(clean) == 0:
        _log("blocking FN attribution: no clean blocking FNs in this sample -- nothing to attribute")
        detail = pd.DataFrame(columns=[
            "s1_id", "cand_id", "country", "retrieved_tfidf", "retrieved_embed",
            "retrieved_rare", "retrieved_digit", "retrieved_address",
            "n_blockers_retrieved", "blocker_status"])
        summary = pd.DataFrame([
            {"metric": "total_clean_blocking_fns", "value": 0},
            {"metric": "retrieved_by_at_least_one_blocker", "value": 0},
            {"metric": "retrieved_by_no_blocker", "value": 0},
        ])
        detail.to_csv(detail_path, sep="\t", index=False)
        summary.to_csv(summary_path, sep="\t", index=False)
        return {"detail": detail, "summary": summary, "out_dir": str(out_dir)}

    pass_maps = _blocking_passes_by_country(
        prep["s1"], prep["others"], prep["emb"],
        within_country=config.BLOCK_WITHIN_COUNTRY, use_address_pass=config.USE_ADDRESS_PASS)

    rows = []
    for s1_id, cand_id in zip(clean["s1_id"], clean["cand_id"]):
        flags = {p: bool(cand_id in pass_maps.get(p, {}).get(s1_id, ())) for p in _PASS_NAMES}
        n_ret = sum(flags.values())
        if n_ret == 0:
            status = "NONE"
        elif n_ret == 1:
            status = f"{next(p for p in _PASS_NAMES if flags[p]).upper()}_ONLY"
        else:
            status = "MULTIPLE"
        rows.append({
            "s1_id": s1_id, "cand_id": cand_id, "country": s1_country.get(s1_id, ""),
            **{f"retrieved_{p}": flags[p] for p in _PASS_NAMES},
            "n_blockers_retrieved": n_ret, "blocker_status": status,
        })
    detail = pd.DataFrame(rows)

    # Sanity check 2: a pair an individual pass retrieved should be explainable by the
    # production union logic -- but sanity check 1 already proved every clean FN pair
    # is absent from that union, so any row with n_blockers_retrieved > 0 here is a
    # genuine discrepancy between an individual pass re-run and the production union
    # construction, not something to silently accept.
    inconsistent = detail[detail["n_blockers_retrieved"] > 0]
    if len(inconsistent):
        _log(f"WARNING: {len(inconsistent)} clean blocking FN(s) were retrieved by an "
            f"individual pass re-run despite being absent from the production candidate "
            f"union -- this is a discrepancy between individual-pass and union-construction "
            f"logic in blocking.generate_candidates worth investigating, not an expected "
            f"finding. Affected s1_id/cand_id pairs are in {detail_path.name}.")

    total = len(detail)
    n_any = int((detail["n_blockers_retrieved"] > 0).sum())
    n_none = total - n_any
    pass_counts = {p: int(detail[f"retrieved_{p}"].sum()) for p in _PASS_NAMES}
    status_counts = detail["blocker_status"].value_counts().to_dict()

    def _pct(n: int, d: int) -> float:
        """Percentage of ``n`` over ``d``, 0.0 when ``d`` is 0."""
        return (n / d * 100.0) if d else 0.0

    summary_rows = [
        {"metric": "total_clean_blocking_fns", "value": total},
        {"metric": "retrieved_by_at_least_one_blocker", "value": n_any},
        {"metric": "retrieved_by_no_blocker", "value": n_none},
        {"metric": "retrieved_by_at_least_one_blocker_pct", "value": _pct(n_any, total)},
        {"metric": "retrieved_by_no_blocker_pct", "value": _pct(n_none, total)},
    ]
    for p in _PASS_NAMES:
        summary_rows.append({"metric": f"{p}_count", "value": pass_counts[p]})
        summary_rows.append({"metric": f"{p}_pct", "value": _pct(pass_counts[p], total)})
    for status in _STATUS_ORDER:
        cnt = int(status_counts.get(status, 0))
        summary_rows.append({"metric": f"status_{status}_count", "value": cnt})
        summary_rows.append({"metric": f"status_{status}_pct", "value": _pct(cnt, total)})
    for country, g in detail.groupby("country"):
        n_c = len(g)
        n_c_any = int((g["n_blockers_retrieved"] > 0).sum())
        summary_rows.append({"metric": f"country_{country}_n", "value": n_c})
        summary_rows.append({"metric": f"country_{country}_retrieved_by_any", "value": n_c_any})
        summary_rows.append({"metric": f"country_{country}_retrieved_by_any_pct",
                            "value": _pct(n_c_any, n_c)})
    summary = pd.DataFrame(summary_rows)

    detail.to_csv(detail_path, sep="\t", index=False)
    summary.to_csv(summary_path, sep="\t", index=False)
    meta = {
        "kind": "blocking_fn_attribution", "timestamp": ts, "sample": sample,
        "use_embeddings": use_embeddings, "git_commit": rp._git_hash(),
        "dataset_file_hashes": diag.dataset_file_hashes(config.TRAIN_FILES),
        "env": diag.env_info(), "n_clean_blocking_fns": total,
        "n_inconsistent_with_union": int(len(inconsistent)),
    }
    diag.save_json(meta, out_dir / "meta.json")
    _log(f"blocking FN attribution written under {out_dir}")

    print(f"\nClean blocking FNs: {total}\n")
    for p in _PASS_NAMES:
        print(f"{p.upper():<12}{pass_counts[p]} / {total} ({_pct(pass_counts[p], total):.1f}%)")
    print(f"\nRetrieved by >=1 blocker: {n_any} / {total} ({_pct(n_any, total):.1f}%)")
    print(f"Retrieved by none:         {n_none} / {total} ({_pct(n_none, total):.1f}%)")
    print("\nStatus:")
    for status in _STATUS_ORDER:
        cnt = int(status_counts.get(status, 0))
        print(f"  {status:<12}{cnt} / {total} ({_pct(cnt, total):.1f}%)")
    print("\nBy country:")
    for country, g in detail.groupby("country"):
        n_c = len(g)
        n_c_any = int((g["n_blockers_retrieved"] > 0).sum())
        print(f"  {country}: n={n_c}, retrieved_by_any={n_c_any} ({_pct(n_c_any, n_c):.1f}%)")

    return {"detail": detail, "summary": summary, "out_dir": str(out_dir)}


# --- blocking-FN pair-level forensics ------------------------------------------------------

#: Name/address raw+normalised fields carried through to the detail TSV, per side.
_FORENSICS_NAME_FIELDS = ("name_norm", "name_core", "first_token")
_FORENSICS_ADDR_FIELDS = ("addr_norm",)
_FORENSICS_NUM_FIELDS = ("digits", "house_no", "region")

#: Deterministic forensic categories, checked in this order (first match wins).
_FORENSICS_CATEGORIES = ("MULTI_FIELD_SIGNAL", "NAME_SIGNAL", "ADDRESS_SIGNAL",
                         "NUMERIC_SIGNAL", "VERY_LOW_SIGNAL")

#: Thresholds behind ``strong_name_signal``/``strong_address_signal`` below. Deliberately
#: conservative, round similarity cutoffs -- not fit or tuned on this data -- so the
#: classification stays a transparent, inspectable rule rather than a black-box score.
_STRONG_JACCARD = 0.5
_STRONG_JW = 0.90
_STRONG_TOKEN_SORT = 0.85
_STRONG_OVERLAP = 2


def _token_set(text: str) -> frozenset[str]:
    """Space-separated tokens of one normalised field, as a frozenset."""
    return frozenset(str(text).split())


def run_blocking_fn_forensics(sample: float = 0.0045, use_embeddings: bool = True) -> dict:
    """Measurement-only diagnostic: pair-level forensic measurements for the exact
    "clean" blocking false negatives ``blocking-fn-attribution`` reports (true pairs
    staged ``blocking_false_negative`` with every representation-loss flag False, so
    every individual blocking pass provably missed them and no mechanical raw-vs-
    normalised signal loss explains it either) -- investigating what these pairs
    actually look like, field by field, so that possible new blocking signals can be
    considered later from evidence rather than guesswork.

    Reuses, unmodified: :func:`_clean_blocking_fn_pairs` (the exact same 97-pair set,
    from the exact same deterministic sample and production-default classification
    path, that ``blocking-fn-attribution`` reports -- never a newly regenerated
    approximation of it) and ``features.pair_features`` (the exact production
    pairwise-similarity implementation -- Jaro-Winkler, rapidfuzz ratio/partial/
    token-sort, TF-IDF cosine, token Jaccard, digit/house/region equality -- fitted
    via ``features.FeatureContext.fit`` on this sample's own S1+others exactly as
    ``features.build_features`` does when given no context). The only similarity
    numbers computed outside ``pair_features`` here are ones it does not already
    expose: exact ``name_norm`` equality, address Jaro-Winkler (``pair_features`` only
    has rapidfuzz ratio/token-set/token-sort/partial for address, not Jaro-Winkler),
    token *overlap counts* (``pair_features`` has Jaccard but not the raw intersection
    size) and the actual shared-token lists -- each a direct, undisputed
    set/rapidfuzz computation on the same normalised fields, never a new formula.

    Classifies each pair into exactly one of :data:`_FORENSICS_CATEGORIES`
    (``MULTI_FIELD_SIGNAL`` > ``NAME_SIGNAL`` > ``ADDRESS_SIGNAL`` > ``NUMERIC_SIGNAL``
    > ``VERY_LOW_SIGNAL``, first match wins) via three transparent boolean gates
    (``strong_name_signal``/``strong_address_signal``/``strong_numeric_signal``,
    each kept as its own column in the detail TSV) built from fixed, round similarity
    thresholds (:data:`_STRONG_JACCARD`/:data:`_STRONG_JW`/:data:`_STRONG_TOKEN_SORT`/
    :data:`_STRONG_OVERLAP`) -- never a learned or fitted score. These forensic
    similarity numbers are diagnostic-only: this function never feeds them to
    ``blocking.generate_candidates``, never adds them as a new blocking pass, and
    changes no ``config.py``/``blocking.py``/``features.py``/``normalize.py``/
    ``model.py`` default, threshold or K value.

    Writes ``clean_fn_forensics.tsv`` (one row per clean blocking FN) and
    ``summary.tsv`` (aggregate + per-country counts) under
    ``artifacts/diagnostics/blocking_fn_forensics_<ts>/`` (plus a ``meta.json``
    sidecar, mirroring this module's other diagnostics). Never touches ``output/``.
    """
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_dir = DIAG_DIR / f"blocking_fn_forensics_{ts}"
    _log(f"blocking FN forensics: sample={sample}")

    clean, _cands, prep, s1_country = _clean_blocking_fn_pairs(sample, use_embeddings)
    _log(f"blocking FN forensics: {len(clean)} clean blocking FNs "
        f"(stage=blocking_false_negative, all {len(edecomp.LOSS_KEYS)} "
        f"representation-loss flags False)")

    DIAG_DIR.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    detail_path = out_dir / "clean_fn_forensics.tsv"
    summary_path = out_dir / "summary.tsv"

    if len(clean) == 0:
        _log("blocking FN forensics: no clean blocking FNs in this sample -- nothing to inspect")
        detail = pd.DataFrame()
        summary = pd.DataFrame([{"metric": "total_clean_fns", "value": 0}])
        detail.to_csv(detail_path, sep="\t", index=False)
        summary.to_csv(summary_path, sep="\t", index=False)
        return {"detail": detail, "summary": summary, "out_dir": str(out_dir)}

    s1_frame, others_frame = prep["s1"], prep["others"]
    s1_pos = pd.Index(s1_frame[config.ID_COL]).get_indexer(clean["s1_id"])
    o_pos = pd.Index(others_frame[config.ID_COL]).get_indexer(clean["cand_id"])
    if (s1_pos < 0).any() or (o_pos < 0).any():
        raise ValueError("clean blocking FN IDs missing from the normalised S1/others frames")
    r1 = s1_frame.iloc[s1_pos].reset_index(drop=True)
    r2 = others_frame.iloc[o_pos].reset_index(drop=True)

    # Production similarity implementation, unmodified: the same FeatureContext.fit
    # + pair_features path build_features itself uses when given no context.
    ctx = FeatureContext.fit(s1_frame, others_frame)
    pf = pair_features(r1, r2, ctx, emb=None)

    name_core_sets = [_token_set(t) for t in r1["name_core"]], [_token_set(t) for t in r2["name_core"]]
    addr_norm_sets = [_token_set(t) for t in r1["addr_norm"]], [_token_set(t) for t in r2["addr_norm"]]
    name_shared = [a & b for a, b in zip(*name_core_sets)]
    addr_shared = [a & b for a, b in zip(*addr_norm_sets)]

    addr1, addr2 = r1["addr_norm"].tolist(), r2["addr_norm"].tolist()
    addr_jw = np.array([JaroWinkler.normalized_similarity(a, b) if a and b else np.nan
                       for a, b in zip(addr1, addr2)], dtype=np.float32)
    name_exact_norm = (r1["name_norm"].to_numpy() == r2["name_norm"].to_numpy())

    rows = []
    for i in range(len(clean)):
        row: dict = {
            "s1_id": clean["s1_id"].iat[i], "cand_id": clean["cand_id"].iat[i],
            "country": s1_country.get(clean["s1_id"].iat[i], ""),
        }
        for field in _FORENSICS_NAME_FIELDS:
            row[f"s1_{field}"] = r1[field].iat[i]
            row[f"cand_{field}"] = r2[field].iat[i]
        for field in _FORENSICS_ADDR_FIELDS:
            row[f"s1_{field}"] = r1[field].iat[i]
            row[f"cand_{field}"] = r2[field].iat[i]
        for field, out_name in (("digits", "digits"), ("house_no", "house_number"), ("region", "region")):
            row[f"s1_{out_name}"] = r1[field].iat[i]
            row[f"cand_{out_name}"] = r2[field].iat[i]

        row["name_exact_norm"] = bool(name_exact_norm[i])
        row["name_exact_core"] = bool(pf["name_core_exact"].iat[i])
        row["name_token_jaccard"] = float(pf["name_token_jaccard"].iat[i])
        row["name_token_overlap_count"] = len(name_shared[i])
        row["name_jaro_winkler"] = float(pf["name_jw"].iat[i])
        row["name_edit_ratio"] = float(pf["name_ratio"].iat[i])
        row["name_partial_ratio"] = float(pf["name_partial"].iat[i])
        row["name_token_sort_ratio"] = float(pf["name_token_sort"].iat[i])

        addr_jac = pf["addr_token_jaccard"].iat[i]
        row["addr_token_jaccard"] = float(addr_jac) if not np.isnan(addr_jac) else np.nan
        row["addr_token_overlap_count"] = len(addr_shared[i])
        row["addr_jaro_winkler"] = float(addr_jw[i]) if not np.isnan(addr_jw[i]) else np.nan
        for out_name, col in (("addr_edit_ratio", "addr_ratio"),
                              ("addr_partial_ratio", "addr_partial"),
                              ("addr_token_sort_ratio", "addr_token_sort")):
            v = pf[col].iat[i]
            row[out_name] = float(v) if not np.isnan(v) else np.nan

        row["digit_shared_count"] = float(pf["digit_shared"].iat[i])
        dj = pf["digit_jaccard"].iat[i]
        row["digit_jaccard"] = float(dj) if not np.isnan(dj) else np.nan
        he = pf["house_equal"].iat[i]
        row["house_equal"] = (bool(he) if not np.isnan(he) else np.nan)
        re_ = pf["region_equal"].iat[i]
        row["region_equal"] = (bool(re_) if not np.isnan(re_) else np.nan)

        row["shared_name_tokens"] = ",".join(sorted(name_shared[i]))
        row["shared_address_tokens"] = ",".join(sorted(addr_shared[i]))

        # --- transparent, deterministic forensic category gates ---------------------
        strong_name = bool(
            row["name_exact_norm"] or row["name_exact_core"]
            or row["name_token_jaccard"] >= _STRONG_JACCARD
            or row["name_jaro_winkler"] >= _STRONG_JW
            or row["name_token_sort_ratio"] >= _STRONG_TOKEN_SORT)
        strong_addr = bool(
            (not np.isnan(row["addr_token_jaccard"]) and row["addr_token_jaccard"] >= _STRONG_JACCARD)
            or (not np.isnan(row["addr_jaro_winkler"]) and row["addr_jaro_winkler"] >= _STRONG_JW)
            or row["addr_token_sort_ratio"] >= _STRONG_TOKEN_SORT
            or row["addr_token_overlap_count"] >= _STRONG_OVERLAP)
        strong_numeric = bool(
            (row["house_equal"] is True) or (row["region_equal"] is True)
            or row["digit_shared_count"] >= 1)
        row["strong_name_signal"] = strong_name
        row["strong_address_signal"] = strong_addr
        row["strong_numeric_signal"] = strong_numeric
        n_fields = int(strong_name) + int(strong_addr) + int(strong_numeric)
        row["signal_field_count"] = n_fields
        if n_fields >= 2:
            row["forensic_category"] = "MULTI_FIELD_SIGNAL"
        elif strong_name:
            row["forensic_category"] = "NAME_SIGNAL"
        elif strong_addr:
            row["forensic_category"] = "ADDRESS_SIGNAL"
        elif strong_numeric:
            row["forensic_category"] = "NUMERIC_SIGNAL"
        else:
            row["forensic_category"] = "VERY_LOW_SIGNAL"
        rows.append(row)

    detail = pd.DataFrame(rows)
    total = len(detail)

    def _pct(n: int, d: int) -> float:
        """Percentage of ``n`` over ``d``, 0.0 when ``d`` is 0."""
        return (n / d * 100.0) if d else 0.0

    def _count(mask: pd.Series) -> int:
        """Count of True values in a boolean-or-NaN-safe mask."""
        return int(mask.fillna(False).astype(bool).sum())

    cat_counts = detail["forensic_category"].value_counts().to_dict()
    agg = {
        "total_clean_fns": total,
        "name_exact_norm_count": _count(detail["name_exact_norm"]),
        "name_exact_core_count": _count(detail["name_exact_core"]),
        "name_token_overlap_count": _count(detail["name_token_overlap_count"] > 0),
        "strong_name_similarity_count": _count(detail["strong_name_signal"]),
        "address_token_overlap_count": _count(detail["addr_token_overlap_count"] > 0),
        "strong_address_similarity_count": _count(detail["strong_address_signal"]),
        "digit_overlap_count": _count(detail["digit_shared_count"] > 0),
        "house_match_count": _count(detail["house_equal"] == True),  # noqa: E712 -- NaN-safe tri-state
        "region_match_count": _count(detail["region_equal"] == True),  # noqa: E712
        "multi_field_signal_count": int(cat_counts.get("MULTI_FIELD_SIGNAL", 0)),
        "very_low_signal_count": int(cat_counts.get("VERY_LOW_SIGNAL", 0)),
    }
    summary_rows = [{"metric": k, "value": v} for k, v in agg.items()]
    for cat in _FORENSICS_CATEGORIES:
        summary_rows.append({"metric": f"category_{cat}_count", "value": int(cat_counts.get(cat, 0))})
        summary_rows.append({"metric": f"category_{cat}_pct",
                            "value": _pct(int(cat_counts.get(cat, 0)), total)})

    for country, g in detail.groupby("country"):
        n_c = len(g)
        g_cat_counts = g["forensic_category"].value_counts().to_dict()
        prefix = f"country_{country}_"
        summary_rows.append({"metric": f"{prefix}total_clean_fns", "value": n_c})
        summary_rows.append({"metric": f"{prefix}name_exact_norm_count",
                            "value": _count(g["name_exact_norm"])})
        summary_rows.append({"metric": f"{prefix}name_exact_core_count",
                            "value": _count(g["name_exact_core"])})
        summary_rows.append({"metric": f"{prefix}strong_name_similarity_count",
                            "value": _count(g["strong_name_signal"])})
        summary_rows.append({"metric": f"{prefix}strong_address_similarity_count",
                            "value": _count(g["strong_address_signal"])})
        summary_rows.append({"metric": f"{prefix}digit_overlap_count",
                            "value": _count(g["digit_shared_count"] > 0)})
        summary_rows.append({"metric": f"{prefix}house_match_count",
                            "value": _count(g["house_equal"] == True)})  # noqa: E712
        summary_rows.append({"metric": f"{prefix}region_match_count",
                            "value": _count(g["region_equal"] == True)})  # noqa: E712
        for cat in _FORENSICS_CATEGORIES:
            summary_rows.append({"metric": f"{prefix}category_{cat}_count",
                                "value": int(g_cat_counts.get(cat, 0))})
    summary = pd.DataFrame(summary_rows)

    detail.to_csv(detail_path, sep="\t", index=False)
    summary.to_csv(summary_path, sep="\t", index=False)
    meta = {
        "kind": "blocking_fn_forensics", "timestamp": ts, "sample": sample,
        "use_embeddings": use_embeddings, "git_commit": rp._git_hash(),
        "dataset_file_hashes": diag.dataset_file_hashes(config.TRAIN_FILES),
        "env": diag.env_info(), "n_clean_fns": total,
        "thresholds": {"strong_jaccard": _STRONG_JACCARD, "strong_jaro_winkler": _STRONG_JW,
                      "strong_token_sort": _STRONG_TOKEN_SORT, "strong_overlap": _STRONG_OVERLAP},
    }
    diag.save_json(meta, out_dir / "meta.json")
    _log(f"blocking FN forensics written under {out_dir}")

    print(f"\nClean blocking FNs inspected: {total}\n")
    print(f"name_exact_norm:              {agg['name_exact_norm_count']} / {total} "
        f"({_pct(agg['name_exact_norm_count'], total):.1f}%)")
    print(f"name_exact_core:              {agg['name_exact_core_count']} / {total} "
        f"({_pct(agg['name_exact_core_count'], total):.1f}%)")
    print(f"strong_name_similarity:       {agg['strong_name_similarity_count']} / {total} "
        f"({_pct(agg['strong_name_similarity_count'], total):.1f}%)")
    print(f"strong_address_similarity:    {agg['strong_address_similarity_count']} / {total} "
        f"({_pct(agg['strong_address_similarity_count'], total):.1f}%)")
    print(f"digit_overlap:                {agg['digit_overlap_count']} / {total} "
        f"({_pct(agg['digit_overlap_count'], total):.1f}%)")
    print(f"house_match:                  {agg['house_match_count']} / {total} "
        f"({_pct(agg['house_match_count'], total):.1f}%)")
    print(f"region_match:                 {agg['region_match_count']} / {total} "
        f"({_pct(agg['region_match_count'], total):.1f}%)")
    print("\nForensic category:")
    for cat in _FORENSICS_CATEGORIES:
        cnt = int(cat_counts.get(cat, 0))
        print(f"  {cat:<20}{cnt} / {total} ({_pct(cnt, total):.1f}%)")
    print("\nBy country:")
    for country, g in detail.groupby("country"):
        n_c = len(g)
        print(f"  {country}: n={n_c}")
        g_cat_counts = g["forensic_category"].value_counts().to_dict()
        for cat in _FORENSICS_CATEGORIES:
            cnt = int(g_cat_counts.get(cat, 0))
            if cnt:
                print(f"    {cat:<20}{cnt} / {n_c} ({_pct(cnt, n_c):.1f}%)")

    return {"detail": detail, "summary": summary, "out_dir": str(out_dir)}


# --- CLI ---------------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the diagnose CLI's subcommands and arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    b = sub.add_parser("baseline", help="E0a/E0b/E0c: baseline reproduction + LOCO")
    b.add_argument("--sample", type=float, default=1.0)
    b.add_argument("--loco", action="store_true")
    b.add_argument("--no-embeddings", action="store_true")

    d = sub.add_parser("drift", help="E-drift: train/test distribution comparison")
    d.add_argument("--sample", type=float, default=1.0)
    d.add_argument("--with-candidates", action="store_true")
    d.add_argument("--no-embeddings", action="store_true")

    c = sub.add_parser("convergence", help="E-convergence: validation sample-size study")
    c.add_argument("--sizes", type=int, nargs="+", default=[2000, 5000, 10000, 25000])
    c.add_argument("--loco", action="store_true")
    c.add_argument("--no-embeddings", action="store_true")

    r = sub.add_parser("resource-stage", help="E2: staged resource qualification")
    r.add_argument("--stage", choices=[*_STAGES, "all"], default="all")

    e = sub.add_parser("errors", help="E1: error decomposition report")
    e.add_argument("--sample", type=float, default=1.0)
    e.add_argument("--no-embeddings", action="store_true")

    k = sub.add_parser(
        "blocking-ablation",
        help="P1: cheap blocking-only K ablation (one-factor-at-a-time sweep)")
    k.add_argument("--sample", type=float, default=0.0045,
                   help="S1 sample fraction (default: 0.0045, the reproducible baseline scale)")
    k.add_argument("--no-embeddings", action="store_true",
                   help="skip the embedding pass entirely (embed-K rows become no-ops)")

    kd = sub.add_parser(
        "blocking-k-downstream",
        help="P1-downstream: full blocking->model->decision OOF validation for "
             "P1's 4 Pareto-competitive blocking-K configurations")
    kd.add_argument("--sample", type=float, default=0.0045,
                    help="S1 sample fraction (default: 0.0045, the reproducible baseline scale)")
    kd.add_argument("--no-embeddings", action="store_true",
                    help="skip the embedding pass entirely")

    ce = sub.add_parser(
        "chunk-equality",
        help="E3: exact dense-retrieval candidate-ID equality across chunk sizes")
    ce.add_argument("--sample", type=float, default=0.0045,
                    help="S1 sample fraction (default: 0.0045, the reproducible baseline scale)")

    fp = sub.add_parser(
        "fp16-retrieval",
        help="E4: true-match retrieval loss from float16 dense-retrieval similarity")
    fp.add_argument("--sample", type=float, default=0.0045,
                    help="S1 sample fraction (default: 0.0045, the reproducible baseline scale)")

    dv = sub.add_parser(
        "decision-validation",
        help="P3: decision-layer validation (one-to-one ON/OFF, decision order, "
             "threshold and singleton-threshold neighborhoods)")
    dv.add_argument("--sample", type=float, default=0.0045,
                    help="S1 sample fraction (default: 0.0045, the reproducible baseline scale)")
    dv.add_argument("--no-embeddings", action="store_true")

    lc = sub.add_parser(
        "loco",
        help="LOCO country-generalization comparison: baseline (20/20) vs both40 "
             "(40/40) candidate K, evaluated per country-held-out direction")
    lc.add_argument("--sample", type=float, default=0.0045,
                    help="S1 sample fraction (default: 0.0045, the reproducible baseline scale)")
    lc.add_argument("--no-embeddings", action="store_true")

    fhn = sub.add_parser(
        "feature-hard-negative",
        help="Final pre-lock diagnostic: feature discrimination audit + hard-negative "
             "score overlap on the both40 candidate configuration")
    fhn.add_argument("--sample", type=float, default=0.0045,
                     help="S1 sample fraction (default: 0.0045, the reproducible baseline scale)")
    fhn.add_argument("--no-embeddings", action="store_true")

    bfa = sub.add_parser(
        "blocking-fn-attribution",
        help="Attribute clean (no representation-loss) blocking false negatives to "
             "individual production blocking passes")
    bfa.add_argument("--sample", type=float, default=0.0045,
                     help="S1 sample fraction (default: 0.0045, the reproducible baseline scale)")
    bfa.add_argument("--no-embeddings", action="store_true")

    bff = sub.add_parser(
        "blocking-fn-forensics",
        help="Pair-level forensic name/address/numeric similarity measurements for "
             "the clean (no representation-loss) blocking false negatives")
    bff.add_argument("--sample", type=float, default=0.0045,
                     help="S1 sample fraction (default: 0.0045, the reproducible baseline scale)")
    bff.add_argument("--no-embeddings", action="store_true")

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Dispatch to the requested diagnostic subcommand."""
    args = parse_args(argv)
    if args.command == "baseline":
        run_baseline(args.sample, args.loco, not args.no_embeddings)
    elif args.command == "drift":
        run_drift(args.sample, args.with_candidates, not args.no_embeddings)
    elif args.command == "convergence":
        run_convergence(args.sizes, not args.no_embeddings, args.loco)
    elif args.command == "resource-stage":
        run_resource_stage(args.stage)
    elif args.command == "errors":
        run_error_decomposition(args.sample, not args.no_embeddings)
    elif args.command == "blocking-ablation":
        run_blocking_ablation(args.sample, not args.no_embeddings)
    elif args.command == "blocking-k-downstream":
        run_blocking_k_downstream(args.sample, not args.no_embeddings)
    elif args.command == "chunk-equality":
        run_chunk_equality(args.sample)
    elif args.command == "fp16-retrieval":
        run_fp16_retrieval(args.sample)
    elif args.command == "decision-validation":
        run_decision_validation(args.sample, not args.no_embeddings)
    elif args.command == "loco":
        run_loco_comparison(args.sample, not args.no_embeddings)
    elif args.command == "feature-hard-negative":
        run_feature_hard_negative_audit(args.sample, not args.no_embeddings)
    elif args.command == "blocking-fn-attribution":
        run_blocking_fn_attribution(args.sample, not args.no_embeddings)
    elif args.command == "blocking-fn-forensics":
        run_blocking_fn_forensics(args.sample, not args.no_embeddings)


if __name__ == "__main__":
    main()
