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

from . import config
from . import diagnostics as diag
from . import drift as drift_mod
from . import error_decomposition as edecomp
from . import run_pipeline as rp
from .blocking import generate_candidates, report_blocking_stats
from .decide import apply_threshold, tune_threshold
from .evaluate import score_report
from .features import build_features, feature_columns, label_pairs
from .io_utils import load_source, load_split
from .model import group_folds, predict, train_full, train_oof

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
    artifacts = list((config.ARTIFACTS_DIR).glob("emb_*.npy"))
    return diag.run_stage("embed", fn, artifacts, DIAG_DIR)


def _stage_block() -> dict:
    """Stage 3: full-scale blocking/dense retrieval (requires stages 1-2's caches)."""
    _warn_if_missing("norm_*.parquet", "block", "normalize")
    _warn_if_missing("emb_*.npy", "block", "embed")

    def fn():
        prep = rp.prepare_from_files(config.TRAIN_FILES, use_embeddings=True, use_cache=True)
        rp.block(prep, use_cache=True)
    artifacts = list((config.ARTIFACTS_DIR).glob("cands_*.parquet"))
    return diag.run_stage("block", fn, artifacts, DIAG_DIR)


def _stage_features() -> dict:
    """Stage 4: full-scale feature generation (requires stages 1-3's caches)."""
    _warn_if_missing("norm_*.parquet", "features", "normalize")
    _warn_if_missing("emb_*.npy", "features", "embed")
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
                    singleton_tau: float | None, truth: dict[str, list[str]]) -> dict:
    """Per-fold macro F0.5 from OOF predictions, using the one tuned threshold pair.

    Each GroupKFold fold holds a disjoint set of S1 groups (``model.group_folds``'
    contract), so ``scored`` rows in fold ``f`` are exactly the out-of-fold
    predictions for the S1s validated in that fold -- unbiased, the same way the
    headline OOF macro F0.5 is. The single ``(tau, singleton_tau)`` from the full OOF
    sweep is reused for every fold (task requirement 10: no per-fold threshold
    re-tuning), so this measures how stable the *chosen* operating point is across
    folds, not how well each fold could have done with its own threshold.
    """
    s1_col = scored["s1_id"].to_numpy()
    per_fold: list[float] = []
    for f in sorted(set(int(v) for v in folds) - {-1}):
        mask = folds == f
        fold_s1 = sorted(set(s1_col[mask]))
        if not fold_s1:
            continue
        pred = apply_threshold(scored[mask], tau, fold_s1, singleton_tau)
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


if __name__ == "__main__":
    main()
