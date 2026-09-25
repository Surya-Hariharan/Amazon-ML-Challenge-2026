"""Generate report figures from experiments/results/*.json (diagnostic only, no source
modification). Run: python make_figures.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
RES = ROOT / "experiments" / "results"
FIG = ROOT / "experiments" / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

TAGS = ["n500", "n2000", "n5000", "n10000"]
DATA = {t: json.load(open(RES / f"{t}.json")) for t in TAGS}


def savefig(name):
    plt.tight_layout()
    plt.savefig(FIG / name, dpi=130)
    plt.close()


# 1. candidate-count distribution per S1 (n10000, largest sample)
d = DATA["n10000"]
vol = d["stage_B_candidate_volume"]
plt.figure(figsize=(6, 4))
xs = ["mean", "median", "p95", "p99", "max"]
ys = [vol["mean"], vol["median"], vol["p95"], vol["p99"], vol["max"]]
plt.bar(xs, ys, color="#4C72B0")
plt.ylabel("candidates per S1")
plt.title("Candidate-count distribution per S1 (N=10,000 train sample)")
for x, y in zip(xs, ys):
    plt.text(x, y, f"{y:.1f}", ha="center", va="bottom", fontsize=9)
savefig("candidate_count_distribution.png")

# 2. positive vs negative score distribution (OOF probs, n10000)
pos = d["stage_E_score_analysis"]["positive_prob_percentiles"]
neg = d["stage_E_score_analysis"]["negative_prob_percentiles"]
qs = ["p01", "p05", "p10", "p25", "p50", "p75", "p90", "p95", "p99"]
plt.figure(figsize=(7, 4.5))
plt.plot(qs, [pos[q] for q in qs], marker="o", label="positive pairs (true matches)", color="#2ca02c")
plt.plot(qs, [neg[q] for q in qs], marker="o", label="negative pairs (non-matches)", color="#d62728")
plt.ylabel("LightGBM predicted probability (OOF)")
plt.xlabel("percentile")
plt.title("Positive vs negative OOF score distribution (N=10,000)")
plt.legend()
savefig("score_distribution_pos_vs_neg.png")

# 3. threshold vs F0.5 (macro, from decide.py's own sweep grid) for n10000
sweep = d["stage_F_threshold_sweep"]
th = [r["threshold"] for r in sweep]
macro = [r["macro_f05"] for r in sweep]
sel_tau = d["stage_F_selected_by_existing_tuning"]["tau"]
plt.figure(figsize=(7, 4.5))
plt.plot(th, macro, marker=".", color="#4C72B0")
plt.axvline(sel_tau, color="red", linestyle="--", label=f"tau selected by tune_threshold = {sel_tau}")
plt.xlabel("threshold tau")
plt.ylabel("macro F0.5 (train-fold OOF, incl. singletons)")
plt.title("Threshold vs macro F0.5 (N=10,000, OOF sweep)")
plt.legend()
savefig("threshold_vs_macro_f05.png")

# 4. threshold vs precision/recall (pair-level) for n10000
prec = [r["pair_precision"] for r in sweep]
rec = [r["pair_recall"] for r in sweep]
plt.figure(figsize=(7, 4.5))
plt.plot(th, prec, marker=".", label="pair precision", color="#1f77b4")
plt.plot(th, rec, marker=".", label="pair recall", color="#ff7f0e")
plt.xlabel("threshold tau")
plt.ylabel("value")
plt.title("Threshold vs pair-level precision/recall (N=10,000, OOF sweep)")
plt.legend()
savefig("threshold_vs_precision_recall.png")

# 5. blocking recall by country, across sample sizes
plt.figure(figsize=(7, 4.5))
ns = [DATA[t]["actual_n_s1"] for t in TAGS]
india = [DATA[t]["stage_B_blocking"].get("recall_India", None) for t in TAGS]
us = [DATA[t]["stage_B_blocking"].get("recall_US", None) for t in TAGS]
overall = [DATA[t]["stage_B_blocking"]["pair_recall"] for t in TAGS]
plt.plot(ns, india, marker="o", label="India")
plt.plot(ns, us, marker="o", label="US")
plt.plot(ns, overall, marker="s", linestyle="--", color="gray", label="overall")
plt.xscale("log")
plt.xlabel("sample size (N S1 entities, log scale)")
plt.ylabel("blocking pair recall")
plt.title("Blocking pair recall by country across sample sizes\n(France absent from train — no train data exists to measure it)")
plt.legend()
savefig("blocking_recall_by_country.png")

# 6. blocking recall by S2/S3 source pass usefulness (per-pass recall, n10000)
b = d["stage_B_blocking"]
passes = ["tfidf", "embed", "rare", "digit", "address"]
recalls = [b[f"recall_{p}"] for p in passes]
plt.figure(figsize=(6, 4.5))
plt.bar(passes, recalls, color="#55A868")
plt.ylabel("recall contributed by this pass alone")
plt.title("Blocking recall by pass (N=10,000)")
for x, y in zip(passes, recalls):
    plt.text(x, y, f"{y:.3f}", ha="center", va="bottom", fontsize=9)
savefig("blocking_recall_by_pass.png")

print("wrote figures:", [p.name for p in FIG.glob("*.png")])
