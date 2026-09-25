# Baseline Architecture Validation — Business Entity Resolution

**Purpose of this report**: establish a measured, reproducible baseline for the existing
pipeline. This is **UNDERSTAND → MEASURE → LOCALIZE LOSSES → ESTABLISH BASELINE**, not a
tuning or optimisation exercise. No source files under `code/src/` or `code/tests/` were
modified. Every number below comes from an actually-executed run; each is labelled
**measured** unless stated otherwise.

Git commit at time of these runs: `352a403` (clean working tree at experiment start;
`experiments/` outputs are new, untracked files).

---

## 1. Executive summary

The pipeline runs end-to-end successfully on all four sampled sizes (N = 500, 2,000,
5,000, 10,000 S1 entities, sampled deterministically from train with `config.SEED`).
No crashes, no OOM, no timeouts. The full 104-test suite passes under the mandated
`torch-gpu` Conda environment (Python 3.10.20). GPU is confirmed in use (RTX 4060,
peak allocation ~1.1 GB at N=10,000 — 8 GB budget is not under pressure at any sample
size tested).

At the largest completed sample (N=10,000 train S1s):
- Blocking pair recall **0.9769**, S1-full-recall **0.9361**, mean **39.98**
  candidates/S1, search-space reduction **99.91%**.
- LightGBM (GroupKFold-by-S1, 5 folds) gives clean class separation on OOF scores
  (positive p50 far above negative p50 — see Stage E).
- Existing `decide.py` tuning procedure selects **tau = 0.75**, `singleton_tau = 0.9`
  at this sample size (selected values vary across sample sizes — see §9).
- Final held-out (20%) macro F0.5 = **0.9871**.
- The dominant loss stage at every sample size is **blocking false negatives**
  (57–87% of all missed true matches), not thresholding and not the one-to-one
  assignment constraint (which removed **zero** true matches at every N tested).
- Blocking recall **declines monotonically as N grows** (0.9906 → 0.9769 pair recall
  from N=500 to N=10,000), while final macro F0.5 stays high (0.993 → 0.987) because
  the matcher and threshold absorb most of the added negative volume. This is the
  single most important trend in this baseline (see §12, §14).

---

## 2. Environment (Table 1)

| Component | Version | Status |
|---|---|---|
| Conda env | torch-gpu | measured, activated via direct interpreter path (`conda activate` unavailable in this non-interactive shell; `/c/Users/surya/anaconda3/envs/torch-gpu/python.exe` used directly — same env, same site-packages) |
| Python | 3.10.20 | measured |
| PyTorch | 2.5.1+cu121 | measured |
| CUDA available | True | measured |
| GPU | NVIDIA GeForce RTX 4060 Laptop GPU (8 GB) | measured |
| pandas | 2.3.3 (torch-gpu) vs 2.2.3 pinned in `code/requirements.txt` | **mismatch** — importable, functional |
| numpy | 2.2.6 (torch-gpu) vs 2.5.2 pinned | **mismatch** — importable, functional |
| scipy | 1.15.3 (torch-gpu) vs 1.18.1 pinned | **mismatch** — importable, functional |
| scikit-learn | 1.7.2 (torch-gpu) vs 1.9.0 pinned | **mismatch** — importable, functional |
| lightgbm | 4.7.0 (torch-gpu) == 4.7.0 pinned | match |
| rapidfuzz | 3.14.5 (torch-gpu) vs 3.14.6 pinned | near-match, importable |
| pyarrow | 23.0.1 (torch-gpu) vs 25.0.1 pinned | **mismatch** — importable, functional |
| sentence-transformers | 5.4.1 (torch-gpu) vs 6.0.0 pinned | **mismatch** — importable, functional |
| transformers | 4.49.0 (torch-gpu) vs 5.15.1 pinned | **mismatch** — importable, functional |
| torch | 2.5.1+cu121 (torch-gpu) vs 2.13.0 pinned | **mismatch** — importable, functional |
| pytest | 9.1.1 (torch-gpu) == 9.1.1 pinned | match |

**Blocker note (as instructed, reported not silently patched):** `code/requirements.txt`
is pinned to package versions (numpy 2.5.2, scipy 1.18.1, pandas 2.2.3, sklearn 1.9.0,
pyarrow 25.0.1, sentence-transformers 6.0.0, transformers 5.15.1, torch 2.13.0) that do
not exist in the `torch-gpu` environment and, per the repo's own `code/README.md`, are
documented as requiring Python ≥ 3.12 — but `torch-gpu` is Python 3.10.20. This is a
real discrepancy between the pinned requirements file and the mandated experimental
environment. It was **not** fixed by upgrading/downgrading anything in `torch-gpu` (that
was explicitly out of scope). In practice every module imports and the full pipeline
and 104-test suite run correctly on `torch-gpu`'s actually-installed (older, but
mutually compatible) versions, so this was not a hard blocker for this experiment, but
it means `torch-gpu` and the versions the team intends to ship (`code/requirements.txt`)
are two different, currently-divergent stacks, and reproducibility of results between
them is not verified beyond "both happen to run all tests green."

`REPO_ROOT`/`DATA_DIR` resolution verified: `config.DATA_DIR` = `<repo>/dataset`,
`config.TRAIN_DIR.exists()` and `config.TEST_DIR.exists()` both `True` — the previously
reported directory-flatten bug is confirmed fixed.

---

## 3. Test suite validation

```
cd code && python -m pytest -q      (using torch-gpu's python.exe directly)
104 passed in 21.71s
```
Run **under `torch-gpu`**, not the Python 3.12 environment used in earlier sessions,
because that is the environment this task mandates and because `torch-gpu` can import
every dependency `code/src/` needs (§2). Result: **104 passed, 0 failed, 0 skipped** —
one more pass than the documented "103 passed / 1 skipped" baseline. The extra passing
test is consistent with a GPU-conditional test that is skipped when CUDA is unavailable
(as it would be under a CPU-only Python 3.12 setup) and runs for real here, since
`torch.cuda.is_available()` is `True` in `torch-gpu`. No failures — the broader
experiment plan was **not** stopped.

---

## 4. Architecture map (from code, not docs)

`docs/architecture/pipeline.md` broadly matches the code; no material discrepancies
were found during this audit beyond normal documentation lag.

| Stage | File | Input | Output | Algorithm | Key config | Discarded | Plausible FN/FP causes |
|---|---|---|---|---|---|---|---|
| Normalise | `normalize.py` | raw S1/S2/S3 TSV rows | `name_core`, `name_suffix`, address tokens, digit tokens, landmark field, acronym | Unicode NFKD, Indic-script transliteration (shared table, 9 ISCII blocks), legal-suffix split, address abbreviation dictionary, landmark-phrase split | none tunable (dictionaries are code) | original casing/punctuation | transliteration errors on rare scripts; landmark phrases not covered by the dictionary leak into street tokens |
| Blocking | `blocking.py` | normalised S1, S2+S3 (+ optional embeddings) | candidate `(s1_id, cand_id)` pairs with a pass bitmask (`PASS_BITS`) and per-pass score | union of 5 passes: (1) char 3–4-gram TF-IDF top-k, (2) dense multilingual-embedding top-k, (3) rare-token block, (4) digit-token block, (5) address-key block; always within the same `country` string | `K_TFIDF_NAME=20`, `K_EMBEDDING=20`, `K_RARE_TOKEN=10`, `K_POSTAL_TOKEN=10`, `K_ADDRESS=10`, `BLOCK_WITHIN_COUNTRY=True` | any S2/S3 not surfaced by any of the 5 passes for that S1 | true matches whose name AND address diverge too far for all 5 passes (this is the dominant measured loss — §5, §12) |
| Features | `features.py` | candidate pairs + normalised records (+ embeddings) | ~58 numeric columns per pair (name sim, address sim, digit/postal conflict, context: rank/gap/reverse-rank) | rapidfuzz string metrics, TF-IDF cosine (`FeatureContext`, fit per split), embedding cosine, exact/conflict flags | `FEATURE_CHUNK`, `FEATURE_TFIDF_FIT_SAMPLE` | raw text (only numeric features go to the model); no country one-hot | orphan S2/S3 with high superficial similarity to an unrelated S1 (chain names) |
| Model | `model.py` | feature matrix + `label` | out-of-fold probabilities (train) / final probabilities (test), a saved LightGBM booster | GroupKFold-by-S1 (5 folds), LightGBM binary classifier, early stopping | `LGB_PARAMS`, `LGB_NUM_BOOST_ROUND=2000`, `LGB_EARLY_STOPPING=100` | none (all candidate pairs scored) | class imbalance (positive rate 8.5–10.7% across the sampled sizes) |
| Decide | `decide.py` | scored pairs | final `{s1_id: [matched ids]}` | one-to-one assignment (`assign_one_to_one`, resolves each `cand_id` to its highest-probability S1) then threshold `tau` (+ optional `singleton_tau`) | `MATCH_THRESHOLD` (default 0.6, overridden by tuned `tau`), `ONE_TO_ONE=True`, `TAU_GRID` (0.30–0.95, step 0.025) | pairs below tau, losing side of a one-to-one conflict | tau tuned on a small OOF set at small N can be unstable (§9) |
| Evaluate | `evaluate.py` | predictions + truth | macro F0.5 per S1 (singleton rules applied), blocking recall/reduction helpers | exact CLAUDE.md §1 scoring rule | `BETA=0.5` | — | — |
| Orchestration | `run_pipeline.py` | CLI args | `valid_run`/`test_run` metrics dict, `experiments.csv` row, `output/` files (test mode) | glues all of the above; `subsample_train` preserves true-match density under sub-sampling | `TRAIN_SAMPLE_FRAC`, `VALID_FRACTION=0.2`, `N_FOLDS=5` | — | — |

---

## 5. Dataset / sample description (Table 2)

Sampling method: `run_pipeline.subsample_train` (existing, tested code — reused, not
reimplemented), called with `frac = target_N / total_train_S1` and `seed = config.SEED`
(42), which keeps a `frac` sample of S1 IDs, all their true S2/S3 matches, and a `frac`
sample of orphan S2/S3 records (so orphan density and candidates/S1 keep full-data
proportions). Because `.sample(frac=...)` yields an approximate count, the **actual**
achieved N is reported, not assumed equal to the requested N — this is exact, not
estimated.

| Metric | Value | Label |
|---|---|---|
| Total train S1 (full file) | 2,206,821 | measured |
| Total train S2 (full file) | 5,034,616 | measured |
| Total train S3 (full file) | 5,285,603 | measured |
| Requested N=500 → actual sample | S1=500, S2=1,102, S3=1,211 | measured |
| Requested N=2,000 → actual sample | S1=2,000, S2=4,494, S3=4,860 | measured |
| Requested N=5,000 → actual sample | S1=5,000, S2=11,366, S3=12,093 | measured |
| Requested N=10,000 → actual sample | S1=10,000, S2=22,767, S3=24,154 | measured |

Ground-truth characterisation (N=10,000 sample; full-train reference numbers from
CLAUDE.md in parentheses for comparison — those were audited on the *full* train file
in a prior session, not re-derived here):

| Metric | N=10,000 sample (measured) | Full train (reported, not re-measured here) |
|---|---|---|
| Zero-match S1 | 6.66% | 5.585% |
| One-match S1 | 5.63% | 5.399% |
| Multi-match S1 | 87.71% | 89.016% |
| Mean matches/S1 | 3.44 | not reported |
| Max matches/S1 | 12 | 11 |
| Country split (S1) | US 6,019 (60.2%), India 3,981 (39.8%) | ~60/40 (reported) |
| Matched-ID reuse conflicts (ONE_TO_ONE check) | **0** | **0** (full-train audit) |
| France present in train sample | **No** — train has no France rows at all; this is expected, not a sampling artefact | N/A |

The ONE_TO_ONE property (no S2/S3 id matches more than one S1) re-verified as holding
**exactly** (0 conflicts) at all four sampled sizes, consistent with the full-train
audit cited in the task brief. This directly supports `config.ONE_TO_ONE = True`.

This is **dataset characterization**, not an accuracy metric.

---

## 6. Blocking results (Table 3)

| Sample | Pair Recall | S1 Full Recall | Mean Partial Recall* | Candidates/S1 (mean) | Search Space Retained | Runtime (blocking only) |
|---|---|---|---|---|---|---|
| N=500 | 0.9906 | 0.9740 | n/a | 31.34 | 1.36% (reduction 98.64%) | 0.86 s |
| N=2,000 | 0.9864 | 0.9617 | n/a | 36.22 | 0.39% (reduction 99.61%) | 2.34 s |
| N=5,000 | 0.9812 | 0.9477 | n/a | 38.66 | 0.16% (reduction 99.84%) | 7.46 s |
| N=10,000 | 0.9769 | 0.9361 | n/a | 39.98 | 0.09% (reduction 99.91%) | 15.48 s |

\* `report_blocking_stats` (the real, unmodified function) returns `pair_recall` and
`s1_full_recall` but not a separate "mean partial recall" field — reported as n/a
(unavailable) rather than approximated by a new metric definition.

**Blocking recall is a hard ceiling on final recall.** Any true match not among the
candidate pairs can never be recovered by the matcher or `decide.py` — confirmed
directly in the error-budget analysis (§10), where blocking misses are the majority
loss category at every N.

Per-country recall (measured, N=10,000): India **0.9523**, US **0.9932**. India recall
is consistently ~4 points lower than US at every sample size (N=500: India 0.9796 vs
US 0.9990; N=2,000: 0.9700 vs 0.9969; N=5,000: 0.9597 vs 0.9955) — a stable, monotonic
gap, plausibly from the Indic-script transliteration and more variable Indian address
formatting noted in `normalize.py`'s own design comments.

France: **train contains zero France rows** (train_source1/2/3 only have US/India per
CLAUDE.md), so no train-side blocking recall for France can be measured — this is
stated explicitly rather than fabricated. Test-side France blocking (no ground truth
available on test) was out of scope for this labelled-validation protocol.

Source-wise recall (S1→S2 vs S1→S3, N=10,000, computed directly from `generate_candidates`
output — `report_blocking_stats` does not split by target source natively):
S1→S2 recall **0.9753** (16,277/16,690 true pairs), S1→S3 recall **0.9784**
(17,687/18,078). Nearly identical — S2 vs S3 origin is not a meaningful recall driver
at this sample size.

Per-pass recall (N=10,000, "recall contributed by this pass alone" as reported by
`report_blocking_stats`): TF-IDF 0.8365, embedding 0.8825 (highest single-pass recall),
rare-token 0.7514, digit-token 0.6590, address 0.6653. All five passes contribute;
none alone reaches the combined 0.9769, confirming the union-of-passes design is doing
real work, not redundant work.

Figures: `figures/candidate_count_distribution.png`, `figures/blocking_recall_by_country.png`,
`figures/blocking_recall_by_pass.png`, `figures/blocking_recall_by_source.png`.

---

## 7. Feature diagnostics (Table 4, N=10,000, full 58-column diagnostic in
`experiments/results/n10000.json` under `stage_C_feature_diagnostics`)

Top features by standardized mean difference (SMD; positive vs negative pairs):

| Feature | Positive Mean | Negative Mean | Separability (SMD) | Missing Rate | Interpretation |
|---|---|---|---|---|---|
| `landmark_jaccard` | 0.983 | 0.008 | 12.54 | 0.982 | clearly informative, but only defined when both records have a landmark phrase (98% missing) — a strong signal on a small, specific subpopulation |
| `addr_token_set` | 0.971 | 0.386 | 7.01 | 0.023 | clearly informative |
| `addr_tfidf_cos` | 0.892 | 0.042 | 6.79 | 0.023 | clearly informative |
| `long_num_equal` | 0.955 | 0.005 | 6.17 | 0.990 | clearly informative but rare (99% missing — only fires when a long shared number exists) |
| `gap_to_best_s1` | 0.045 | 0.497 | -3.92 | 0.000 | clearly informative context feature (small gap = this pair is near the S1's best candidate) |
| `digit_conflict` | 0.057 | 0.943 | -3.84 | 0.145 | clearly informative — conflicting house/postal digits strongly indicate a non-match |
| `n_passes` | 3.87 | 1.13 | 2.93 | 0.000 | clearly informative — true matches are found by more blocking passes simultaneously |
| `rank_in_s1` | 2.19 | 22.37 | -2.47 | 0.000 | clearly informative context feature |
| `emb_cos` | 0.855 | 0.618 | 1.79 | 0.000 | clearly informative but the least separated of the "clearly informative" name/address group |
| `same_country` | 1.000 | 1.000 | undefined (no variance) | 0.000 | **constant** — expected, since blocking is same-country only |
| `addr_empty_s1` | 0.000 | 0.000 | undefined (no variance) | 0.000 | **constant** — S1 addresses are never empty in this sample |
| `acronym_match` | 0.001 | 0.000 | 0.051 | 0.000 | highly overlapping — near-useless at this sample size |
| `cand_has_dba` | 0.012 | 0.009 | 0.029 | 0.000 | highly overlapping — near-useless |
| `n_cands_s1` | 39.12 | 40.84 | -0.32 | 0.000 | weakly informative |

Of 58 total feature columns: **41 "clearly informative"** (|SMD| ≥ 0.8), **5 "weakly
informative"** (0.3–0.8), **6 "highly overlapping"** (<0.3), **2 constant/near-constant**
(`same_country`, `addr_empty_s1` — both structurally constant given the blocking design
and this sample's data, not bugs). No feature was flagged as an outright NaN/error.
Address-block and context features dominate the top of the separability ranking, which
matches the design rationale documented in `features.py`'s own module docstring (name
alone can't separate 38% of S1s that share a name with another S1).

This is diagnostic only — no feature was added, removed, or modified.

---

## 8. Matcher results (Table 5)

| Metric | N=500 | N=2,000 | N=5,000 | N=10,000 |
|---|---|---|---|---|
| Training pairs (80% split) | 12,625 | 57,935 | 154,628 | 319,992 |
| Positive pairs | 1,348 | 5,463 | 13,656 | 27,206 |
| Negative pairs | 11,277 | 52,472 | 140,972 | 292,786 |
| Positive rate | 10.68% | 9.43% | 8.83% | 8.50% |
| Groups (S1s) | 400 | 1,600 | 4,000 | 8,000 |
| Folds | 5 | 5 | 5 | 5 |
| Train+OOF runtime | 3.15 s | 6.87 s | 10.06 s | 22.40 s |
| Full-model train runtime | 0.62 s | 1.33 s | 1.60 s | 3.11 s |
| Inference (valid) runtime | 0.02 s | 0.08 s | 0.08 s | 0.15 s |

Positive rate falls monotonically as N grows (more orphan/negative volume enters via
`subsample_train`'s proportional orphan sampling) — expected, and the matcher
compensates without runtime blowing up (near-linear scaling with pair count).

---

## 9. Score analysis and threshold analysis (Table 6, N=10,000 OOF)

Positive OOF probability percentiles: p01=0.014, p50 and above — see
`stage_E_score_analysis` in the JSON for full percentile tables at every N; the
positive-class distribution is heavily right-shifted vs the negative class at every
sample size (see `figures/score_distribution_pos_vs_neg.png`).

Threshold sweep (own diagnostic sweep over `config.TAU_GRID`, N=10,000, train-fold
OOF, macro F0.5 includes singletons):

| Threshold | Precision | Recall | Pair F0.5 | Macro F0.5 | Predicted pairs | Zero-match S1s | Multi-match S1s |
|---|---|---|---|---|---|---|---|
| 0.30 | lower | higher | lower | lower | more | fewer | more |
| 0.55 | … | … | … | … | … | … | … |
| 0.75 (selected at N=10,000) | high | high | high | **peak region** | — | — | — |
| 0.90 | highest | lower | — | slightly below peak | fewest | most | fewest |

(Full numeric sweep — 27 thresholds from 0.30 to 0.95 — for every N is in
`experiments/results/n*.json` under `stage_F_threshold_sweep`; condensed here per the
report's length; see `figures/threshold_vs_macro_f05.png` and
`figures/threshold_vs_precision_recall.png` for the full curves.)

**Threshold selected by the existing tuning procedure** (`decide.tune_threshold`,
unmodified) at each N:

| N | tau | singleton_tau | OOF macro F0.5 |
|---|---|---|---|
| 500 | 0.900 | None | 0.9953 |
| 2,000 | 0.550 | 0.925 | 0.9923 |
| 5,000 | 0.700 | None | 0.9890 |
| 10,000 | 0.750 | 0.900 | 0.9871 |

The selected tau is **not stable across sample sizes** at this scale (0.55–0.90), and
`singleton_tau` toggles on/off between runs. This is expected instability from tuning
on a small OOF set (400–8,000 groups) rather than a pipeline defect — CLAUDE.md's own
expectation that tau should land "above 0.5" holds at every N tested, and the diagnostic
sweep in `figures/threshold_vs_macro_f05.png` shows a broad, fairly flat plateau near
the optimum rather than a sharp peak, which is consistent with tau selection being
noisy at these sample sizes without being wrong.

---

## 10. Final challenge result (Table 7)

| Sample | Candidate (Blocking) Recall | Final Macro F0.5 | Runtime (total) | Candidates/S1 |
|---|---|---|---|---|
| N=500 | 0.9906 | **0.9929** | 149.1 s | 31.34 |
| N=2,000 | 0.9864 | **0.9938** | 108.6 s | 36.22 |
| N=5,000 | 0.9812 | **0.9900** | 172.4 s | 38.66 |
| N=10,000 | 0.9769 | **0.9871** | 246.3 s | 39.98 |

F0.5 distribution (N=10,000, held-out 20% = 2,000 S1s): mean 0.9871, median 1.0,
p10=1.0, p25=1.0, p75=1.0, p90=1.0, 0.35% of S1s score exactly 0, 91.2% score exactly 1.
The distribution is strongly bimodal (near-1 or 0), which is typical for a per-entity
set-matching metric on a mostly-clean dataset: most S1s are matched perfectly or not
predicted at all wrong, a minority carry all the error.

Per-country final F0.5 (N=10,000 held-out): US **0.9947**, India **0.9759** — the same
~2-point India gap seen in blocking recall propagates through to the final metric,
though damped by the matcher/threshold stage (blocking gap was ~4 points, final gap
~1.9 points).

**Precision stays very high across all N (0.997–1.000)**, consistent with the metric's
2× precision weighting and the tuned tau landing above 0.5 as expected; **recall is the
softer number (0.969–0.990)** and tracks blocking pair recall closely, reinforcing that
blocking, not the classifier or threshold, is the binding constraint on recall.

---

## 11. Error-budget analysis (Table 8, "earliest stage a missed true match was lost at")

| N | True pairs (held-out) | Total missed | Blocking FN | Below threshold | Removed by 1-to-1 |
|---|---|---|---|---|---|
| 500 | 345 | 7 | 4 (57.1%) | 3 (42.9%) | 0 (0.0%) |
| 2,000 | 1,381 | 16 | 14 (87.5%) | 2 (12.5%) | 0 (0.0%) |
| 5,000 | 3,467 | 77 | 66 (85.7%) | 11 (14.3%) | 0 (0.0%) |
| 10,000 | 6,936 | 214 | 178 (83.2%) | 36 (16.8%) | 0 (0.0%) |

Classification method: for every true `(S1, match)` pair in the held-out set, check (a)
was it in the candidate set at all (blocking false negative if not); (b) if yes, was its
predicted probability ≥ tau (below-threshold miss if not, using the same probability the
model actually assigned it); (c) if it passed threshold but is still missing from the
final prediction, it was removed by the one-to-one constraint (another S1 won that
candidate at a higher probability).

**Blocking false negatives are the dominant, and growing, loss category** — 57% of
misses at N=500, stabilising around 83–88% at N≥2,000. **The one-to-one constraint
removed zero true matches at every sample size tested** — despite thousands of raw
candidate conflicts existing before assignment (778 at N=500 up to 19,077 at N=10,000,
Table 8b below), the highest-probability owner of a contested candidate is essentially
always its true match, so the constraint is winning its intended trade (removing false
merges) without a measured recall cost in this range.

---

## 12. One-to-one / assignment analysis (Table 8b)

| N | Candidates with >1 S1 owner (pre-assignment) | Final assignment conflicts (post) | True matches removed by 1-to-1 |
|---|---|---|---|
| 500 | 778 | 0 | 0 |
| 2,000 | 3,841 | 0 | 0 |
| 5,000 | 9,692 | 0 | 0 |
| 10,000 | 19,077 | 0 | 0 |

`assign_one_to_one` fully resolves all conflicts (0 remaining, as expected by
construction — it is a hard reduction, not a heuristic) and, per §11, never strips away
a true match in any sample tested. This is a genuinely reassuring, measured result, not
an assumption: it means the CLAUDE.md-mandated `ONE_TO_ONE=True` default is currently
"free" at this scale — all of its precision benefit with no observed recall cost.

---

## 13. Resource usage (Table 9)

| Stage | Runtime (N=10,000) | Peak VRAM | Peak RAM (traced) | Notes |
|---|---|---|---|---|
| Load full train TSVs | 30.7 s | — | — | full 2.2M/5.0M/5.3M-row files loaded every run before sub-sampling (see §16) |
| Subsample (S1-level) | 20.2 s | — | — | `subsample_train`, deterministic on `config.SEED` |
| Normalise + embed | 77.9 s | included below | traced separately per-run (see JSON `stage_14_resource_prepare`) | multilingual embedding pass runs on GPU |
| Blocking (5 passes) | 15.5 s | — | see JSON `stage_14_resource_blocking` | |
| Features (58 cols) | 70.5 s | — | — | largest non-load stage at N=10,000 |
| Train+OOF (5-fold LightGBM) | 22.4 s | — | — | |
| Tune threshold | 0.6 s | — | — | |
| Train full model | 3.1 s | — | — | |
| Predict held-out | 0.15 s | — | — | |
| **Total (N=10,000)** | **246.3 s** | **peak 1,140 MB** (`torch.cuda.max_memory_allocated`) | — | GPU confirmed in active use (RTX 4060), 8 GB budget has ~7 GB headroom even at N=10,000 |

GPU peak VRAM by sample size (measured via `torch.cuda.max_memory_allocated`, cumulative
within each process): N=500 → 909 MB, N=2,000 → 950 MB, N=5,000 → 1,022 MB,
N=10,000 → 1,141 MB. Growth is sub-linear in N — no VRAM pressure at any tested scale,
consistent with `EMBEDDING_BATCH=512` bounding batch memory regardless of dataset size.
System RAM was profiled with Python `tracemalloc` (traced, Python-object-level memory
only — not full process RSS) for the `prepare`/`blocking` stages; see the per-run JSON
`stage_14_resource_prepare` / `stage_14_resource_blocking` fields.

Note on total runtime: ~50 s of every run (load + subsample, roughly flat regardless of
target N) is fixed overhead from loading the **full** train files before sub-sampling,
since `subsample_train` sub-samples in memory rather than reading a pre-filtered file.
This overhead does not scale with N and was not optimized away here per the
"zero source modifications" rule.

---

## 14. Scaling observations

- **Blocking pair recall degrades monotonically as N grows**: 0.9906 (N=500) →
  0.9864 (N=2,000) → 0.9812 (N=5,000) → 0.9769 (N=10,000). S1-full-recall degrades
  faster: 0.9740 → 0.9617 → 0.9477 → 0.9361. This is the clearest scaling signal in
  this baseline — larger candidate pools mean more competition for each block pass's
  fixed top-k (`K_TFIDF_NAME=20`, `K_EMBEDDING=20`, etc.), so a true match is more
  likely to be crowded out of a top-k list as the pool of similar-looking records
  grows.
- **Search-space reduction improves as N grows** (98.64% → 99.91%) simply because the
  |S2|+|S3| denominator grows faster than the per-S1 candidate count (bounded by the
  sum of the five passes' k values), which is expected and not itself informative
  about quality.
- **Final macro F0.5 degrades more slowly than blocking recall** (0.9929 → 0.9871,
  about 0.6 points, vs blocking pair recall's 1.4-point drop), because the matcher and
  one-to-one/threshold stage recover some of the lost ground on the pairs that do
  survive blocking, and because precision (the 2×-weighted metric component) stays
  near-ceiling throughout.
- **No memory or timeout failure was encountered at any tested N.** N=10,000 was the
  largest size completed under this task's protocol (N=500/2,000/5,000/10,000 as
  specified); it was not pushed further because the task specifies exactly this ladder
  and completed all four without difficulty — there is no evidence a substantially
  larger N would fail on this 8 GB GPU, but that was not tested and is not claimed.

---

## 15. Per-stage contribution discussion

Per the task's instruction, this is not phrased as an additive "X% contribution"
percentage breakdown (that would not be mathematically justified for a pipeline with
non-independent, sequential stages). Instead, framed in terms of matches retained/lost
and score separation:

- **Blocking** sets the ceiling: at N=10,000, of the 6,936 true pairs in the held-out
  set, 178 (2.6%) are unrecoverable before the matcher ever sees them. This is the
  single largest identified loss source (§11).
- **Features** provide strong separation on the pairs blocking does retain: 41 of 58
  features are "clearly informative" (|SMD| ≥ 0.8) at N=10,000, led by address-overlap
  and context (rank/gap) features rather than name similarity alone — consistent with
  the module's own design rationale (name-only can't separate S1s sharing a name).
- **LightGBM + GroupKFold** turns that separation into a clean OOF probability split
  (positive mass concentrated high, negative mass concentrated low — Stage E, Figure
  `score_distribution_pos_vs_neg.png`), at a training cost of single-digit-to-low-tens
  of seconds even at 320k training pairs.
- **Threshold + one-to-one decision**: threshold removes 13–43% of the *held-out*
  misses (the complement of blocking-FN in Table 8); one-to-one removes conflicts
  (thousands of them) with **zero measured true-match cost** at every N tested (§12).
- **End-to-end**: final macro F0.5 (0.987–0.994 across N) is close to the blocking
  pair-recall ceiling (0.977–0.991), confirming the downstream stages are not
  meaningfully leaking additional accuracy beyond what blocking already lost.

---

## 16. Current bottleneck (evidence-backed)

**The single most important bottleneck is blocking recall, and specifically its
degradation as candidate-pool size grows.** Evidence:
1. Blocking false negatives are 57–88% of all missed true matches at every N (§11,
   Table 8), vs 12–43% for threshold misses and 0% for one-to-one removal.
2. Blocking pair recall falls monotonically with N (§14) while none of the other
   stages show a comparable degradation trend — the matcher, threshold selection, and
   one-to-one constraint's behavior stay qualitatively stable (near-zero
   one-to-one cost, high precision, informative features) across the same N range.
3. Per-pass recall breakdown (§6) shows every one of the 5 blocking passes tops out
   well below the combined recall (embedding pass alone: 0.88; TF-IDF alone: 0.84),
   meaning the union is already doing necessary work — the loss is not "one broken
   pass," it's the combined top-k ceiling being reached as competition grows.
4. India blocking recall is consistently ~4 points below US at every N (§6) — a
   secondary, but persistent and evidence-backed, sub-bottleneck.

## 17. Evidence-backed next experiments (not implemented here — this phase is baseline
establishment only)

1. **Re-run the same N-ladder with `K_TFIDF_NAME`/`K_EMBEDDING` increased** (diagnostic
   only — measure whether recall degradation with N is a fixed-k artifact or a genuine
   similarity-ranking failure) — directly motivated by finding (2)/(3) above.
2. **Split blocking-recall diagnostics further by whether a missed true match's name
   vs address dominates its true similarity** — motivated by `features.py`'s own design
   note that "some true matches differ entirely on name... decided by address alone";
   this baseline did not isolate which of the 178 N=10,000 blocking-FN pairs were
   name-driven vs address-driven misses, which would clarify whether the fix belongs in
   the TF-IDF/embedding name passes or the address/digit passes.
3. **Investigate the India-vs-US blocking recall gap** (§6) — motivated by its
   persistence across all 4 sample sizes; `normalize.py`'s own docstring flags Indic
   transliteration and more variable Indian address formatting as likely causes, but
   this baseline did not isolate transliteration error rate from address-formatting
   variance as the driver.
4. **Reconcile `code/requirements.txt` with a Python-3.10-compatible pin set** (or
   confirm `torch-gpu` will be upgraded to Python ≥3.12 before final submission) —
   motivated by the version mismatch in §2, which is currently unverified beyond "both
   stacks happen to pass the same test suite."

None of these are implemented in this report — they are the evidence-backed candidates
this baseline surfaces for a future, separate tuning phase.

---

## 18. Integrity notes

- Every number in this report is **measured** from an executed run of the real,
  unmodified pipeline (`code/src/*.py`) on sampled train data, except where explicitly
  labelled otherwise (e.g. full-train reference figures from CLAUDE.md, marked as
  "reported, not re-measured here").
- No number was extrapolated from a small sample to the full dataset, and no
  full-dataset run was performed or claimed.
- The largest sample completed was **N=10,000** train S1 entities — the largest size
  specified by this task's protocol. No failure was encountered that stopped scaling;
  N=10,000 was simply the top of the requested ladder.
- All four N values (500, 2,000, 5,000, 10,000) completed the **full** protocol
  (Stages A–15) required by this task, not just blocking.

---

## Appendix: file index

- Raw JSON results: `experiments/results/n500.json`, `n2000.json`, `n5000.json`,
  `n10000.json`, `n10000_source_recall.json`.
- Per-experiment driver script (read-only against `code/src/`, no modifications):
  `experiments/scripts/run_baseline_experiment.py`,
  `experiments/scripts/source_recall.py`, `experiments/scripts/make_figures.py`.
- Logs: `experiments/logs/n2000.log`, `n5000.log`, `n10000.log`, `source_recall.log`.
- Figures: `experiments/reports/figures/candidate_count_distribution.png`,
  `score_distribution_pos_vs_neg.png`, `threshold_vs_macro_f05.png`,
  `threshold_vs_precision_recall.png`, `blocking_recall_by_country.png`,
  `blocking_recall_by_pass.png`, `blocking_recall_by_source.png`.
