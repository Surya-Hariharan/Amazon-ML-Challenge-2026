"""Supplementary diagnostic: blocking recall split by S1->S2 vs S1->S3 true matches.
report_blocking_stats() does not split by target source, so this recomputes it
directly from generate_candidates() output on the same N=10,000 sample (same seed),
blocking-stage only (no features/model), to keep runtime small. No source modified.
"""
import json
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
CODE_DIR = REPO_ROOT / "code"
sys.path.insert(0, str(CODE_DIR))

from src import config
from src.io_utils import load_split
from src.run_pipeline import subsample_train, truth_from_frame, prepare, block

N = 10000
data = load_split("train")
s1, s2, s3, gt = data["s1"], data["s2"], data["s3"], data["ground_truth"]
truth_full = truth_from_frame(gt)
frac = min(1.0, N / len(s1))
s1_s, s2_s, s3_s, truth = subsample_train(s1, s2, s3, truth_full, frac, seed=config.SEED)

prep = prepare(s1_s, s2_s, s3_s, config.USE_EMBEDDINGS, None, use_cache=False)
cands = block(prep, use_cache=False)

s2_ids = set(s2_s[config.ID_COL])
s3_ids = set(s3_s[config.ID_COL])
cand_pairs = set(zip(cands["s1_id"], cands["cand_id"]))

true_to_s2 = [(s, m) for s, ms in truth.items() for m in ms if m in s2_ids]
true_to_s3 = [(s, m) for s, ms in truth.items() for m in ms if m in s3_ids]

hit_s2 = sum(1 for p in true_to_s2 if p in cand_pairs)
hit_s3 = sum(1 for p in true_to_s3 if p in cand_pairs)

result = {
    "n_true_s1_to_s2": len(true_to_s2), "hit_s1_to_s2": hit_s2,
    "recall_s1_to_s2": hit_s2 / max(len(true_to_s2), 1),
    "n_true_s1_to_s3": len(true_to_s3), "hit_s1_to_s3": hit_s3,
    "recall_s1_to_s3": hit_s3 / max(len(true_to_s3), 1),
}
print(json.dumps(result, indent=2))
with open(REPO_ROOT / "experiments" / "results" / "n10000_source_recall.json", "w") as f:
    json.dump(result, f, indent=2)
