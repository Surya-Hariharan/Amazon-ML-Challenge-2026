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
from .io_utils import load_split, parse_id_list, write_submission, write_tsv
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

    Returns ``{"s1", "others", "emb"}``; ``emb`` is ``(s1_emb, others_emb)`` or None.
    """
    _log(f"normalising {len(s1):,} S1 + {len(s2) + len(s3):,} S2/S3 records")
    s1n = _cached("norm_s1", _frame_key(s1), lambda: normalize_frame(s1), use_cache)
    s2n = _cached("norm_s2", _frame_key(s2), lambda: normalize_frame(s2), use_cache)
    s3n = _cached("norm_s3", _frame_key(s3), lambda: normalize_frame(s3), use_cache)
    on = pd.concat([s2n, s3n], ignore_index=True)
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


def test_run(train: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict],
             test: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame],
             out_dir: Path = config.OUTPUT_DIR,
             use_embeddings: bool = config.USE_EMBEDDINGS, encoder: Encoder | None = None,
             use_cache: bool = True) -> dict:
    """Train on labelled data, predict the test split and write both output files."""
    s1, s2, s3, truth = train
    prep_tr = prepare(s1, s2, s3, use_embeddings, encoder, use_cache)
    tr_ids = list(prep_tr["s1"][config.ID_COL])
    cands_tr = block(prep_tr, use_cache)
    metrics = {f"block_{k}": v for k, v in report_blocking_stats(
        cands_tr, truth, tr_ids, len(prep_tr["others"]),
        dict(zip(tr_ids, prep_tr["s1"][config.COUNTRY_COL]))).items()}
    feats_tr = build_features(cands_tr, prep_tr["s1"], prep_tr["others"], prep_tr["emb"])
    feats_tr["label"] = label_pairs(feats_tr, truth)
    fitted = fit_and_tune(feats_tr, truth, tr_ids)
    metrics.update(tau=fitted["tau"], singleton_tau=fitted["singleton_tau"],
                   oof_f05=fitted["oof_f05"])
    del feats_tr, cands_tr, prep_tr

    t1, t2, t3 = test
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
    metrics.update(test_n_s1=len(te_ids), test_n_pairs=len(scored),
                   test_mean_candidates=len(scored) / max(len(te_ids), 1),
                   test_mean_matches=sum(map(len, matches.values())) / max(len(te_ids), 1),
                   test_empty_share=float(np.mean([not v for v in matches.values()])))
    if use_cache:
        save_model(fitted["model"], config.ARTIFACTS_DIR / "model_final.txt")
        write_tsv(feature_importance(fitted["model"]), config.ARTIFACTS_DIR / "importance_final.tsv")
    _log(f"wrote {m_path} and {c_path}. Now run utils/validate_submission.py; it must PASS.")
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
    """Load (and optionally sub-sample) the training split from config paths."""
    data = load_split("train")
    truth = truth_from_frame(data["ground_truth"])
    return subsample_train(data["s1"], data["s2"], data["s3"], truth, sample)


def run_valid(stage: str, sample: float = config.TRAIN_SAMPLE_FRAC, loco: bool = False,
              use_embeddings: bool = config.USE_EMBEDDINGS, use_cache: bool = True) -> dict:
    """Validation mode on the training files."""
    s1, s2, s3, truth = _load_train(sample)
    metrics = valid_run(s1, s2, s3, truth, stage, loco, use_embeddings, use_cache=use_cache)
    log_experiment({"sample": sample, **metrics}, "valid", stage)
    return metrics


def run_test(sample: float = config.TRAIN_SAMPLE_FRAC,
             use_embeddings: bool = config.USE_EMBEDDINGS, use_cache: bool = True) -> dict:
    """Test mode: train on the training files, write output/ for the test files."""
    train = _load_train(sample)
    te = load_split("test")
    metrics = test_run(train, (te["s1"], te["s2"], te["s3"]), config.OUTPUT_DIR,
                       use_embeddings, use_cache=use_cache)
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
