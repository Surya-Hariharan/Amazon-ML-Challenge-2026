"""CLI entry point: data -> normalise -> blocking -> features -> model -> decide -> output.

Usage (from code/business_entity_resolution/)::

    python -m src.run_pipeline --mode valid [--stage blocking|features|model|all]
                               [--sample 0.1] [--no-embeddings] [--loco]
    python -m src.run_pipeline --mode test [--sample 0.5] [--no-embeddings]

Paths come from ``config.py`` (``BER_DATA_DIR`` -> ``dataset/`` ->
``student_resource/dataset/``). Every run appends a row to ``experiments.csv``
(CLAUDE.md §5).

* ``valid``: block + featurise the (optionally sub-sampled) training data. Split S1s
  80/20 (stratified on singleton vs not). Tune tau on GroupKFold OOF predictions of
  the 80% part, train on it, and score the held-out 20% with macro F0.5. Also dumps
  the 50 worst false positives/negatives to ``artifacts/errors_*.tsv``. ``--loco``
  adds leave-one-country-out scores (CLAUDE.md §6.6), our only proxy for France.
* ``test``: train on the training data (tau from OOF), then block, featurise and
  predict the test split. Writes both output files via ``io_utils.write_submission``.
  ``candidate_pairs.tsv`` is exactly the pair set the model scored.

The core functions (:func:`valid_run`, :func:`test_run`) take DataFrames, so the
whole pipeline is unit-testable on synthetic data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from . import config
from .blocking import candidates_to_lists, compute_embeddings, generate_candidates, report_blocking_stats
from .decide import apply_threshold, tune_threshold
from .evaluate import score_report
from .features import FeatureContext, build_features, feature_columns, label_pairs
from .io_utils import load_split, parse_id_list, read_tsv, write_submission, write_tsv
from .model import feature_importance, predict, save_model, train_full, train_oof
from .normalize import normalize_frame

Encoder = Callable[[list[str]], np.ndarray]


def _log(msg: str) -> None:
    """Print a timestamped progress line."""
    print(f"[pipeline {time.strftime('%H:%M:%S')}] {msg}", flush=True)


# --- data preparation ----------------------------------------------------------------

def truth_from_frame(gt: pd.DataFrame) -> dict[str, list[str]]:
    """``{source1_entity_id: [matched ids]}`` from the ground-truth TSV frame."""
    return {s: parse_id_list(m) for s, m in zip(gt["source1_entity_id"], gt["matched_entity_ids"])}


def subsample_train(s1: pd.DataFrame, s2: pd.DataFrame, s3: pd.DataFrame,
                    truth: dict[str, list[str]], frac: float, seed: int = config.SEED):
    """Keep ``frac`` of S1s, their true matches, and ``frac`` of the orphan S2/S3 records.

    Matched records whose S1 was dropped are removed too, so candidates per S1 and the
    orphan share (~25%) keep their full-data proportions. Returns
    ``(s1, s2, s3, truth)``.
    """
    if frac >= 1.0:
        return s1, s2, s3, truth
    keep_s1 = s1.sample(frac=frac, random_state=seed)
    keep_ids = set(keep_s1[config.ID_COL])
    matched_all = {m for ms in truth.values() for m in ms}
    matched_keep = {m for s in keep_ids for m in truth.get(s, ())}

    def pick(df: pd.DataFrame) -> pd.DataFrame:
        """Kept matches plus a ``frac`` sample of orphans."""
        ids = df[config.ID_COL]
        orphans = df[~ids.isin(matched_all)].sample(frac=frac, random_state=seed)
        return pd.concat([df[ids.isin(matched_keep)], orphans]).sort_index()

    return (keep_s1.sort_index(), pick(s2), pick(s3),
            {s: truth.get(s, []) for s in keep_s1[config.ID_COL]})


def split_s1(s1_ids: list[str], truth: dict[str, list[str]],
             valid_frac: float = config.VALID_FRACTION, seed: int = config.SEED
             ) -> tuple[list[str], list[str]]:
    """Split S1 IDs into (train, valid), stratified on singleton vs non-singleton."""
    rng = np.random.default_rng(seed)
    train, valid = [], []
    for is_single in (True, False):
        ids = np.array(sorted(s for s in s1_ids if (not truth.get(s)) == is_single), dtype=object)
        rng.shuffle(ids)
        n_valid = int(round(len(ids) * valid_frac))
        valid.extend(ids[:n_valid])
        train.extend(ids[n_valid:])
    return sorted(train), sorted(valid)


def _frame_key(*parts) -> str:
    """Stable short hash of frames / values for artifact cache keys."""
    h = hashlib.sha1()
    for p in parts:
        if isinstance(p, pd.DataFrame):
            h.update(pd.util.hash_pandas_object(p, index=False).values.tobytes())
        else:
            h.update(repr(p).encode())
    return h.hexdigest()[:16]


def _cached(name: str, key: str, fn: Callable[[], pd.DataFrame], use_cache: bool) -> pd.DataFrame:
    """Load ``artifacts/<name>_<key>.parquet`` or compute and store it (CLAUDE.md §7)."""
    path = config.ARTIFACTS_DIR / f"{name}_{key}.parquet"
    if use_cache and path.exists():
        _log(f"cache hit {path.name}")
        return pd.read_parquet(path)
    df = fn()
    if use_cache:
        config.ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path, index=False)
    return df


def _normalize_and_cache(name: str, raw: pd.DataFrame, use_cache: bool) -> pd.DataFrame:
    """Cache-aware normalisation of one already-loaded raw source frame.

    Factored out of :func:`prepare` so :func:`_normalize_one` can share the
    exact same cache keys/paths (``norm_<name>_<hash>.parquet``) when it loads
    a source from disk instead of receiving it already in memory.
    """
    return _cached(f"norm_{name}", _frame_key(raw), lambda: normalize_frame(raw), use_cache)


def _normalize_one(name: str, path: Path, use_cache: bool) -> pd.DataFrame:
    """Load one raw source TSV from ``path``, then normalise and cache it.

    The raw frame read here is local to this call and is not part of the
    return value, so once this function returns nothing keeps it alive —
    letting :func:`prepare_from_files` normalise S1, S2 and S3 one at a time
    instead of requiring all three raw frames to already be materialised
    before it can even be called, the way :func:`prepare`'s ``s1``/``s2``/
    ``s3`` parameters do.
    """
    return _normalize_and_cache(name, read_tsv(path), use_cache)


def prepare(s1: pd.DataFrame, s2: pd.DataFrame, s3: pd.DataFrame,
            use_embeddings: bool = config.USE_EMBEDDINGS, encoder: Encoder | None = None,
            use_cache: bool = True) -> dict:
    """Normalise S1 and S2+S3 and (optionally) embed them.

    S2 and S3 are normalised independently, then concatenated, rather than
    concatenating the raw frames first — at full dataset scale this avoids ever
    materialising a ~10M-row raw combined frame, which roughly halves peak RAM
    during this step (measured: ~12.9 GiB combined-then-normalised vs. ~8.4 GiB
    for the larger of the two normalised independently). Row order, schema and
    content are unaffected: ``normalize_frame`` is a pure per-row transform with
    no cross-row statistics, so normalising then concatenating is equivalent to
    concatenating then normalising.

    This still requires the caller to already hold ``s1``, ``s2`` and ``s3``
    in memory simultaneously, since they are its parameters — see
    :func:`prepare_from_files` for the full-scale path that avoids that.

    Returns ``{"s1", "others", "emb"}``; ``emb`` is ``(s1_emb, others_emb)`` or None.
    """
    _log(f"normalising {len(s1):,} S1 + {len(s2) + len(s3):,} S2/S3 records")
    s1n = _normalize_and_cache("s1", s1, use_cache)
    s2n = _normalize_and_cache("s2", s2, use_cache)
    s3n = _normalize_and_cache("s3", s3, use_cache)
    on = pd.concat([s2n, s3n], ignore_index=True)
    # s2n/s3n are local to this call (not part of the return value) and pd.concat
    # copies their content into `on` rather than reusing it, so once `on` exists
    # they are dead weight — drop them now instead of at function exit so their
    # memory (as large as `on` itself, at full dataset scale) is freed before the
    # embedding pass below runs.
    del s2n, s3n
    emb = None
    if use_embeddings:
        _log("embedding records")
        emb = (compute_embeddings(s1n, encoder, cache=use_cache),
               compute_embeddings(on, encoder, cache=use_cache))
    return {"s1": s1n, "others": on, "emb": emb}


def prepare_from_files(files: dict[str, Path], use_embeddings: bool = config.USE_EMBEDDINGS,
                       encoder: Encoder | None = None, use_cache: bool = True) -> dict:
    """Sequential-loading equivalent of :func:`prepare`, for full-scale training.

    ``files`` maps ``"s1"``/``"s2"``/``"s3"`` to TSV paths (e.g.
    ``config.TRAIN_FILES``). Each source is loaded, normalised and cached by
    :func:`_normalize_one` — and its raw frame released — before the next
    source is even read, so at most one raw source (plus the much smaller,
    growing normalised set) is resident at a time. :func:`prepare`, by
    contrast, requires its ``s1``/``s2``/``s3`` arguments to already be
    materialised simultaneously by the caller before it can be called at all;
    this closes off the suspected full-scale bottleneck behind the overnight
    full-train run never producing a normalised S3 cache.

    Uses the exact same cache keys as :func:`prepare` (see
    :func:`_normalize_and_cache`), so a normalisation cache built by either
    function is reused by the other. Returns the same ``{"s1", "others",
    "emb"}`` shape as :func:`prepare`, with identical content and row order.
    """
    _log("normalising sources sequentially (one raw source resident at a time)")
    s1n = _normalize_one("s1", files["s1"], use_cache)
    s2n = _normalize_one("s2", files["s2"], use_cache)
    s3n = _normalize_one("s3", files["s3"], use_cache)
    on = pd.concat([s2n, s3n], ignore_index=True)
    del s2n, s3n  # see prepare()'s identical del: on already holds a full copy
    emb = None
    if use_embeddings:
        _log("embedding records")
        emb = (compute_embeddings(s1n, encoder, cache=use_cache),
               compute_embeddings(on, encoder, cache=use_cache))
    return {"s1": s1n, "others": on, "emb": emb}


def block(prep: dict, use_cache: bool = True) -> pd.DataFrame:
    """Blocking with an artifact cache keyed by inputs + blocking config."""
    knobs = [getattr(config, k) for k in config.TUNABLES if k not in
             ("MATCH_THRESHOLD", "SINGLETON_THRESHOLD", "ONE_TO_ONE", "TRAIN_SAMPLE_FRAC")]
    key = _frame_key(prep["s1"][[config.ID_COL]], prep["others"][[config.ID_COL]],
                     prep["emb"] is not None, knobs)
    return _cached("cands", key, lambda: generate_candidates(
        prep["s1"], prep["others"], embeddings=prep["emb"]), use_cache)


# --- modelling -----------------------------------------------------------------------

def fit_and_tune(feats: pd.DataFrame, truth: dict[str, list[str]], s1_ids: list[str],
                 verbose: bool = True) -> dict:
    """GroupKFold OOF -> tau sweep (macro F0.5 incl. singletons) -> full model.

    ``feats`` must carry a ``label`` column. ``s1_ids`` is the S1 universe for the
    sweep, including S1s with no candidates. Returns ``model``, ``tau``,
    ``singleton_tau``, ``oof_f05``, ``grid`` and ``fold_models``.
    """
    cols = feature_columns(feats)
    X, y = feats[cols], feats["label"].to_numpy()
    oof, fold_models = train_oof(X, y, feats["s1_id"], verbose=verbose)
    scored = feats[["s1_id", "cand_id"]].assign(prob=oof)
    tau, stau, f05, grid = tune_threshold(scored, truth, s1_ids)
    if verbose:
        _log(f"OOF macro F0.5 = {f05:.4f} at tau={tau} singleton_tau={stau}")
    model = train_full(X, y, fold_models=fold_models)
    return {"model": model, "tau": tau, "singleton_tau": stau, "oof_f05": f05,
            "grid": grid, "fold_models": fold_models}


def dump_errors(scored: pd.DataFrame, pred: dict[str, list[str]], prep: dict,
                n: int = 50, out_dir: Path | None = None) -> None:
    """Write the ``n`` worst false positives and false negatives (CLAUDE.md §7).

    FPs: predicted pairs with label 0, highest probability first. FNs: true pairs
    among the candidates that were not predicted, lowest probability first.
    """
    out_dir = config.ARTIFACTS_DIR if out_dir is None else out_dir
    predicted = {(s, c) for s, cs in pred.items() for c in cs}
    is_pred = np.fromiter((p in predicted for p in zip(scored["s1_id"], scored["cand_id"])),
                          dtype=bool, count=len(scored))
    fp = scored[is_pred & (scored["label"] == 0)].nlargest(n, "prob")
    fn = scored[~is_pred & (scored["label"] == 1)].nsmallest(n, "prob")
    cols = [config.ID_COL, config.NAME_COL, config.ADDRESS_COL, config.COUNTRY_COL]
    left = prep["s1"][cols].add_prefix("s1_").rename(columns={"s1_entity_id": "s1_id"})
    right = prep["others"][cols].add_prefix("cand_").rename(columns={"cand_entity_id": "cand_id"})
    for name, df in (("fp", fp), ("fn", fn)):
        df = df[["s1_id", "cand_id", "prob"]].merge(left, on="s1_id").merge(right, on="cand_id")
        write_tsv(df, out_dir / f"errors_{name}.tsv")


def valid_run(s1: pd.DataFrame, s2: pd.DataFrame, s3: pd.DataFrame,
              truth: dict[str, list[str]], stage: str = "all", loco: bool = False,
              use_embeddings: bool = config.USE_EMBEDDINGS, encoder: Encoder | None = None,
              use_cache: bool = True) -> dict:
    """Validation run on labelled data. Returns a flat metrics dict."""
    prep = prepare(s1, s2, s3, use_embeddings, encoder, use_cache)
    s1_ids = list(prep["s1"][config.ID_COL])
    s1_country = dict(zip(prep["s1"][config.ID_COL], prep["s1"][config.COUNTRY_COL]))
    cands = block(prep, use_cache)
    metrics = {f"block_{k}": v for k, v in report_blocking_stats(
        cands, truth, s1_ids, len(prep["others"]), s1_country).items()}
    if stage == "blocking":
        return metrics
    feats = build_features(cands, prep["s1"], prep["others"], prep["emb"])
    feats["label"] = label_pairs(feats, truth)
    metrics.update(n_feature_rows=len(feats), positive_rate=float(feats["label"].mean()))
    if stage == "features":
        _log(f"{len(feats):,} pairs, {len(feature_columns(feats))} features, "
             f"positive rate {metrics['positive_rate']:.4f}")
        return metrics

    train_ids, valid_ids = split_s1(s1_ids, truth)
    in_train = feats["s1_id"].isin(set(train_ids)).to_numpy()
    fitted = fit_and_tune(feats[in_train], truth, train_ids)
    valid_feats = feats[~in_train]
    scored = valid_feats[["s1_id", "cand_id", "label"]].assign(
        prob=predict(fitted["model"], valid_feats[feature_columns(feats)]))
    pred = apply_threshold(scored, fitted["tau"], valid_ids, fitted["singleton_tau"])
    report = score_report(pred, truth, valid_ids, groups=s1_country)
    metrics.update({f"valid_{k}": v for k, v in report.items()})
    metrics.update(tau=fitted["tau"], singleton_tau=fitted["singleton_tau"],
                   oof_f05=fitted["oof_f05"],
                   all_empty_baseline=float(np.mean([not truth.get(s) for s in valid_ids])))
    _log(f"VALID macro F0.5 = {report['macro_f05']:.4f} "
         f"(P={report['pair_precision']:.4f} R={report['pair_recall']:.4f})")
    if use_cache:
        dump_errors(scored, pred, prep)
        write_tsv(feature_importance(fitted["model"]), config.ARTIFACTS_DIR / "importance_valid.tsv")

    if loco:
        countries = sorted(set(s1_country.values()))
        row_country = feats["s1_id"].map(s1_country).to_numpy()
        for c in countries:
            tr_ids = [s for s in s1_ids if s1_country[s] != c]
            te_ids = [s for s in s1_ids if s1_country[s] == c]
            if not tr_ids:
                continue
            fit_c = fit_and_tune(feats[row_country != c], truth, tr_ids, verbose=False)
            te = feats[row_country == c]
            sc = te[["s1_id", "cand_id"]].assign(
                prob=predict(fit_c["model"], te[feature_columns(feats)]))
            p = apply_threshold(sc, fit_c["tau"], te_ids, fit_c["singleton_tau"])
            metrics[f"loco_f05_{c}"] = score_report(p, truth, te_ids)["macro_f05"]
            _log(f"LOCO held-out {c}: macro F0.5 = {metrics[f'loco_f05_{c}']:.4f}")
    return metrics


def _fit_on_prepared(prep: dict, truth: dict[str, list[str]],
                     use_cache: bool = True) -> tuple[dict, dict]:
    """Block, featurise and fit on an already-``prepare``d (normalised) split.

    Factored out of :func:`_train_model` so :func:`_train_model_from_files`
    — which normalises its sources sequentially via :func:`prepare_from_files`
    instead of :func:`prepare` — shares the same blocking/feature/fit logic
    rather than duplicating it.
    """
    tr_ids = list(prep["s1"][config.ID_COL])
    cands_tr = block(prep, use_cache)
    metrics = {f"block_{k}": v for k, v in report_blocking_stats(
        cands_tr, truth, tr_ids, len(prep["others"]),
        dict(zip(tr_ids, prep["s1"][config.COUNTRY_COL]))).items()}
    feats_tr = build_features(cands_tr, prep["s1"], prep["others"], prep["emb"])
    feats_tr["label"] = label_pairs(feats_tr, truth)
    fitted = fit_and_tune(feats_tr, truth, tr_ids)
    metrics.update(tau=fitted["tau"], singleton_tau=fitted["singleton_tau"],
                   oof_f05=fitted["oof_f05"])
    return fitted, metrics


def _train_model(s1: pd.DataFrame, s2: pd.DataFrame, s3: pd.DataFrame,
                 truth: dict[str, list[str]], use_embeddings: bool = config.USE_EMBEDDINGS,
                 encoder: Encoder | None = None, use_cache: bool = True) -> tuple[dict, dict]:
    """Prepare, block, featurise and fit on the training split. Returns ``(fitted, metrics)``.

    Raw ``s1``/``s2``/``s3`` and every intermediate (``prep_tr``, ``cands_tr``,
    ``feats_tr``) are local to this call and never appear in ``fitted``/``metrics``,
    so once it returns none of them are reachable any more — unlike a ``del``
    inside a function, which cannot free an argument the *caller* still holds a
    reference to for the duration of the call. ``run_test`` relies on exactly this:
    it calls this function to completion (freeing train's raw ~12M rows) *before*
    loading the test split, instead of holding both raw datasets at once.

    Requires the caller to already hold ``s1``, ``s2`` and ``s3`` simultaneously
    (they are its parameters); see :func:`_train_model_from_files` for the
    full-scale path that never materialises all three raw sources at once.
    """
    prep_tr = prepare(s1, s2, s3, use_embeddings, encoder, use_cache)
    return _fit_on_prepared(prep_tr, truth, use_cache)


def _train_model_from_files(files: dict[str, Path], truth: dict[str, list[str]],
                            use_embeddings: bool = config.USE_EMBEDDINGS,
                            encoder: Encoder | None = None,
                            use_cache: bool = True) -> tuple[dict, dict]:
    """Full-scale counterpart of :func:`_train_model`.

    Loads and normalises S1, S2 and S3 sequentially straight from ``files``
    via :func:`prepare_from_files`, instead of requiring the caller to already
    hold all three raw frames in memory the way :func:`_train_model`'s
    ``s1``/``s2``/``s3`` parameters do. Used by :func:`run_test` for the
    un-subsampled (``sample >= 1.0``) full-scale path.
    """
    prep_tr = prepare_from_files(files, use_embeddings, encoder, use_cache)
    return _fit_on_prepared(prep_tr, truth, use_cache)


def _predict_test(fitted: dict, t1: pd.DataFrame, t2: pd.DataFrame, t3: pd.DataFrame,
                  out_dir: Path = config.OUTPUT_DIR,
                  use_embeddings: bool = config.USE_EMBEDDINGS, encoder: Encoder | None = None,
                  use_cache: bool = True) -> dict:
    """Block, featurise and score the test split with an already-fitted model.

    Writes both output files and returns the test-side metrics dict.
    """
    prep_te = prepare(t1, t2, t3, use_embeddings, encoder, use_cache)
    te_ids = list(prep_te["s1"][config.ID_COL])
    cands_te = block(prep_te, use_cache)
    feats_te = build_features(cands_te, prep_te["s1"], prep_te["others"], prep_te["emb"],
                              ctx=FeatureContext.fit(prep_te["s1"], prep_te["others"]))
    scored = feats_te[["s1_id", "cand_id"]].assign(
        prob=predict(fitted["model"], feats_te[feature_columns(feats_te)]))
    matches = apply_threshold(scored, fitted["tau"], te_ids, fitted["singleton_tau"])
    candidates = candidates_to_lists(scored)  # exactly the set the model scored
    valid_ids = set(t2[config.ID_COL]) | set(t3[config.ID_COL])
    m_path, c_path = write_submission(matches, candidates, te_ids, valid_ids, out_dir)
    metrics = dict(test_n_s1=len(te_ids), test_n_pairs=len(scored),
                   test_mean_candidates=len(scored) / max(len(te_ids), 1),
                   test_mean_matches=sum(map(len, matches.values())) / max(len(te_ids), 1),
                   test_empty_share=float(np.mean([not v for v in matches.values()])))
    if use_cache:
        save_model(fitted["model"], config.ARTIFACTS_DIR / "model_final.txt")
        write_tsv(feature_importance(fitted["model"]), config.ARTIFACTS_DIR / "importance_final.tsv")
    _log(f"wrote {m_path} and {c_path}. Now run utils/validate_submission.py; it must PASS.")
    return metrics


def test_run(train: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict],
             test: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
             out_dir: Path = config.OUTPUT_DIR,
             use_embeddings: bool = config.USE_EMBEDDINGS, encoder: Encoder | None = None,
             use_cache: bool = True) -> dict:
    """Train on labelled data, predict the test split and write both output files.

    A thin wrapper over :func:`_train_model` and :func:`_predict_test`, kept so the
    whole pipeline stays testable end-to-end on synthetic DataFrames in one call
    (CLAUDE.md's testability requirement). At full dataset scale, ``run_test``
    calls the two halves directly as separate calls instead of going through this
    function, because a single call to this function requires the caller to pass
    ``train`` and ``test`` together, and Python keeps a reference to both on the
    caller's stack for this call's *entire* duration regardless of anything this
    function itself deletes internally — only returning from a call actually frees
    what only the caller referenced.
    """
    s1, s2, s3, truth = train
    fitted, metrics = _train_model(s1, s2, s3, truth, use_embeddings, encoder, use_cache)
    t1, t2, t3 = test
    metrics.update(_predict_test(fitted, t1, t2, t3, out_dir, use_embeddings, encoder, use_cache))
    return metrics


# --- logging / CLI -------------------------------------------------------------------

def _git_hash() -> str:
    """Short commit hash (``+dirty`` if uncommitted changes), or ``nogit``."""
    try:
        h = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=config.CODE_DIR,
                           capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=config.CODE_DIR,
                               capture_output=True, text=True).stdout.strip()
        return h + ("+dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "nogit"


def log_experiment(metrics: dict, mode: str, stage: str, path: Path | None = None) -> None:
    """Append timestamp, git hash, config snapshot and metrics to experiments.csv."""
    path = config.EXPERIMENTS_CSV if path is None else path
    row = {"timestamp": time.strftime("%Y-%m-%d %H:%M:%S"), "git": _git_hash(),
           "mode": mode, "stage": stage,
           "config": json.dumps({k: getattr(config, k) for k in config.TUNABLES}, default=str)}
    row.update(metrics)
    new = pd.DataFrame([row])
    if Path(path).exists():
        new = pd.concat([pd.read_csv(path), new], ignore_index=True)
    new.to_csv(path, index=False)
    _log(f"logged run to {path}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--mode", choices=["valid", "test"], required=True)
    parser.add_argument("--stage", choices=["blocking", "features", "model", "all"],
                        default="all", help="valid mode: stop after this stage")
    parser.add_argument("--sample", type=float, default=config.TRAIN_SAMPLE_FRAC,
                        help="fraction of train S1s to use (orphan density preserved)")
    parser.add_argument("--no-embeddings", action="store_true",
                        help="skip the embedding pass/feature (e.g. no GPU)")
    parser.add_argument("--loco", action="store_true",
                        help="valid mode: add leave-one-country-out scores")
    parser.add_argument("--no-cache", action="store_true", help="do not read/write artifacts/")
    return parser.parse_args(argv)


def _load_train(sample: float):
    """Load (and optionally sub-sample) the training split from config paths.

    Used for sub-sampled runs (``sample < 1.0``), where ``subsample_train``
    must cross-reference S1, S2 and S3 together to preserve orphan density —
    that inherently needs all three raw frames in memory at once, so there is
    nothing to gain from loading them sequentially here. The un-subsampled,
    full-scale path (``run_test`` at the default ``sample >= 1.0``) instead
    uses ``_train_model_from_files``, which never materialises all three raw
    sources simultaneously — see its docstring and :func:`prepare_from_files`.
    """
    data = load_split("train")
    truth = truth_from_frame(data["ground_truth"])
    return subsample_train(data["s1"], data["s2"], data["s3"], truth, sample)


def _load_train_truth() -> dict[str, list[str]]:
    """Load just the training ground truth, without touching S1/S2/S3.

    Used by :func:`run_test`'s full-scale path, which loads and normalises
    S1/S2/S3 sequentially via :func:`_train_model_from_files` instead.
    """
    return truth_from_frame(read_tsv(config.TRAIN_FILES["ground_truth"]))


def _load_test():
    """Load the test split's three source frames as a tuple, from config paths."""
    data = load_split("test")
    return data["s1"], data["s2"], data["s3"]


def run_valid(stage: str, sample: float = config.TRAIN_SAMPLE_FRAC, loco: bool = False,
              use_embeddings: bool = config.USE_EMBEDDINGS, use_cache: bool = True) -> dict:
    """Validation mode on the training files."""
    s1, s2, s3, truth = _load_train(sample)
    metrics = valid_run(s1, s2, s3, truth, stage, loco, use_embeddings, use_cache=use_cache)
    log_experiment({"sample": sample, **metrics}, "valid", stage)
    return metrics


def run_test(sample: float = config.TRAIN_SAMPLE_FRAC,
             use_embeddings: bool = config.USE_EMBEDDINGS, use_cache: bool = True) -> dict:
    """Test mode: train on the training files, write output/ for the test files.

    At ``sample >= 1.0`` (the default — the full-scale, no-subsampling path),
    training loads and normalises S1, S2 and S3 sequentially straight from
    ``config.TRAIN_FILES`` via ``_train_model_from_files``/``prepare_from_files``,
    so at most one raw source is resident at a time. This replaces the old
    ``_load_train`` -> ``_train_model`` path, whose ``load_split`` call
    materialised S1, S2, S3 *and* ground_truth simultaneously before
    normalisation even began — the full-scale bottleneck this fixes (S1+S2+S3
    total ~12.5M rows on a 15 GiB instance with no swap). Sub-sampled runs
    (``sample < 1.0``) still need S1/S2/S3 cross-referenced together to
    preserve orphan density (see ``subsample_train``), so they fall back to
    the old ``_load_train``/``_train_model`` raw-frame path; that path is for
    smaller, iteration-scale runs regardless of this change.

    Either way, the training half and ``_predict_test`` are called as two
    separate, sequential top-level calls (bypassing ``test_run``, see its
    docstring): the first fully returns -- releasing every training-time
    intermediate, since nothing here binds them to a name either -- *before*
    the test split is even loaded.
    """
    if sample >= 1.0:
        fitted, metrics = _train_model_from_files(config.TRAIN_FILES, _load_train_truth(),
                                                  use_embeddings, use_cache=use_cache)
    else:
        fitted, metrics = _train_model(*_load_train(sample), use_embeddings, use_cache=use_cache)
    metrics.update(_predict_test(fitted, *_load_test(), config.OUTPUT_DIR,
                                 use_embeddings, use_cache=use_cache))
    log_experiment({"sample": sample, **metrics}, "test", "all")
    return metrics


def main(argv: list[str] | None = None) -> None:
    """Dispatch to the requested mode."""
    args = parse_args(argv)
    use_emb = config.USE_EMBEDDINGS and not args.no_embeddings
    if args.mode == "valid":
        run_valid(args.stage, args.sample, args.loco, use_emb, not args.no_cache)
    else:
        run_test(args.sample, use_emb, not args.no_cache)


if __name__ == "__main__":
    main()
