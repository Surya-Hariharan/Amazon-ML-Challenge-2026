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
from .blocking import country_groups, dense_topk, generate_candidates, report_blocking_stats
from .decide import apply_threshold, tune_threshold
from .evaluate import blocking_recall, score_report
from .features import build_features, feature_columns, label_pairs
from .io_utils import load_source, load_split
from .model import group_folds, predict, train_full, train_oof

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


if __name__ == "__main__":
    main()
