"""Evaluation metrics: macro F0.5 per S1 entity, blocking recall, reduction ratio.

Scoring rule (CLAUDE.md §1): for every S1 entity compute F-beta (beta = 0.5) between
its predicted and true S2/S3 ID sets, then average over *all* S1 entities, singletons
included:

* truth empty, prediction empty      -> 1.0
* truth empty, prediction non-empty  -> 0.0
* truth non-empty, prediction empty  -> 0.0
* otherwise F_beta = (1 + b^2) * P * R / (b^2 * P + R)

Example: pred {47, 193, 812} vs truth {47, 812} -> P = 2/3, R = 1 -> F0.5 = 0.714.
"""

from collections.abc import Iterable, Mapping

from . import config


def fbeta_single(pred: Iterable[str], truth: Iterable[str], beta: float = config.BETA) -> float:
    """Return F-beta for one S1 entity, applying the singleton rules.

    Args:
        pred: predicted matched IDs (duplicates are ignored).
        truth: true matched IDs (duplicates are ignored).
        beta: recall weight; 0.5 weights precision 2x over recall.

    Returns:
        Score in [0, 1].
    """
    pred_set, truth_set = set(pred), set(truth)
    if not truth_set:
        return 1.0 if not pred_set else 0.0
    tp = len(pred_set & truth_set)
    if tp == 0:
        return 0.0
    precision = tp / len(pred_set)
    recall = tp / len(truth_set)
    b2 = beta * beta
    return (1 + b2) * precision * recall / (b2 * precision + recall)


def per_entity_scores(
    pred: Mapping[str, Iterable[str]],
    truth: Mapping[str, Iterable[str]],
    s1_ids: Iterable[str] | None = None,
    beta: float = config.BETA,
) -> dict[str, float]:
    """Return {s1_id: F-beta} over the evaluated S1 universe.

    Args:
        pred: S1 ID -> predicted IDs. Missing S1 IDs count as an empty prediction.
        truth: S1 ID -> true IDs. Missing S1 IDs count as singletons (no matches).
        s1_ids: S1 IDs to score. Defaults to the keys of ``truth``; pass the full S1
            list when ``truth`` only holds S1s that have matches.
        beta: F-beta parameter.
    """
    universe = list(dict.fromkeys(truth.keys() if s1_ids is None else s1_ids))
    return {
        s1: fbeta_single(pred.get(s1, ()), truth.get(s1, ()), beta) for s1 in universe
    }


def macro_fbeta(
    pred: Mapping[str, Iterable[str]],
    truth: Mapping[str, Iterable[str]],
    s1_ids: Iterable[str] | None = None,
    beta: float = config.BETA,
) -> float:
    """Return macro-averaged F-beta over S1 entities (the competition metric).

    See :func:`per_entity_scores` for argument semantics. Returns 0.0 for an empty
    universe.
    """
    scores = per_entity_scores(pred, truth, s1_ids, beta)
    return sum(scores.values()) / len(scores) if scores else 0.0


def score_report(
    pred: Mapping[str, Iterable[str]],
    truth: Mapping[str, Iterable[str]],
    s1_ids: Iterable[str] | None = None,
    groups: Mapping[str, str] | None = None,
    beta: float = config.BETA,
) -> dict[str, float]:
    """Return the headline metrics logged to experiments.csv.

    Keys: ``macro_f05``, ``pair_precision``, ``pair_recall``, ``singleton_f05``,
    ``non_singleton_f05``, ``n_s1`` and, if ``groups`` (S1 ID -> group label such as
    country) is given, ``f05_<group>`` for each group.
    """
    scores = per_entity_scores(pred, truth, s1_ids, beta)
    report: dict[str, float] = {"n_s1": float(len(scores))}
    report["macro_f05"] = sum(scores.values()) / len(scores) if scores else 0.0

    tp = n_pred = n_true = 0
    single, multi = [], []
    for s1, score in scores.items():
        p, t = set(pred.get(s1, ())), set(truth.get(s1, ()))
        tp += len(p & t)
        n_pred += len(p)
        n_true += len(t)
        (multi if t else single).append(score)
    report["pair_precision"] = tp / n_pred if n_pred else 0.0
    report["pair_recall"] = tp / n_true if n_true else 0.0
    report["singleton_f05"] = sum(single) / len(single) if single else float("nan")
    report["non_singleton_f05"] = sum(multi) / len(multi) if multi else float("nan")

    if groups is not None:
        by_group: dict[str, list[float]] = {}
        for s1, score in scores.items():
            by_group.setdefault(str(groups.get(s1, "")), []).append(score)
        for g, vals in sorted(by_group.items()):
            report[f"f05_{g}"] = sum(vals) / len(vals)
    return report


def blocking_recall(
    candidates: Mapping[str, Iterable[str]], truth: Mapping[str, Iterable[str]]
) -> dict[str, float]:
    """Return how many true pairs survive blocking.

    Returns:
        ``pair_recall``: share of true (S1, match) pairs present in the candidates;
        ``s1_full_recall``: share of non-singleton S1s whose every true match is a
        candidate; ``mean_candidates``: mean candidate-list length over S1s in
        ``candidates``. Recalls are 1.0 when there are no true pairs.
    """
    hit = total = full = n_multi = 0
    for s1, t in truth.items():
        t_set = set(t)
        if not t_set:
            continue
        c_set = set(candidates.get(s1, ()))
        found = len(t_set & c_set)
        hit += found
        total += len(t_set)
        n_multi += 1
        full += found == len(t_set)
    lengths = [len(set(c)) for c in candidates.values()]
    return {
        "pair_recall": hit / total if total else 1.0,
        "s1_full_recall": full / n_multi if n_multi else 1.0,
        "mean_candidates": sum(lengths) / len(lengths) if lengths else 0.0,
    }


def reduction_ratio(n_candidate_pairs: int, n_s1: int, n_other: int) -> float:
    """Return 1 - candidate pairs / all possible (S1, S2/S3) pairs.

    Args:
        n_candidate_pairs: total number of (S1, candidate) pairs after blocking.
        n_s1: number of S1 records.
        n_other: number of S2 + S3 records.
    """
    full = n_s1 * n_other
    return 1.0 - n_candidate_pairs / full if full else 0.0
