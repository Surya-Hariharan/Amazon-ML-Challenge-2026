# ML Development Roadmap & Challenge Compliance

```
Purpose: single, permanent synthesis of everything this repo has actually measured,
         audited and verified so far, cross-referenced against the two official
         challenge documents. This document does not introduce new measurements — every
         number here is cited from an existing report/JSON/test run. Where no evidence
         exists, the item is marked NOT TESTED rather than guessed.
Sources of truth: docs/challenge/problem_statement.md, docs/challenge/guidelines.md
                  (official requirements) and the repository itself (what has actually
                  been built/tested).
Date written: 2026-09-25.
```

Status legend used throughout: `[x]` COMPLETED (real repo evidence cited), `[~]`
PARTIALLY TESTED, `[ ]` NOT TESTED, `[!]` BLOCKED.

---

## 1. Project Status

**Test suite (re-run for this document):**

```
cd code && /c/Users/surya/anaconda3/envs/torch-gpu/python.exe -m pytest -q
138 passed in 23.35s
```

Environment: `torch-gpu` conda env, Python 3.10.20, `torch.cuda.is_available() == True`
(RTX 4060 Laptop GPU). **This is 138 passed / 0 failed / 0 skipped**, which differs from
both numbers cited in the task brief (104/0/0 and 137/0/1). This is not a regression —
it is explained directly by prior sessions' own findings: the baseline validation report
(`experiments/reports/baseline_architecture_validation.md` §3) already documented that
the 104→104-passed-0-skipped jump (vs. an earlier "103 passed / 1 skipped" figure)
happens specifically because one test is GPU-conditional and only skips when CUDA is
unavailable; the SageMaker storage implementation report
(`experiments/reports/sagemaker_storage_implementation.md`, "Tests added and results")
recorded **137 passed, 1 skipped** after adding `test_s3_sync.py`'s 33 tests, but that
run was under `py -3.12` (a different, CPU-only environment, per that report's own
command line). Under `torch-gpu` (CUDA available), that same skip-conditional test runs
for real instead of skipping: 104 (pre-s3_sync baseline) + 33 (s3_sync tests) + 1
(previously-skipped GPU test, now running) = 138, 0 skipped. Both historical numbers and
this session's number are mutually consistent once the environment difference is
accounted for — no code was modified to produce this result.

**Where the project actually stands:**
- Pipeline mechanics (normalise → block → features → model → decide → evaluate) are
  measured-correct up to N=10,000 train S1 entities (§5, §7 below) and data-correctness-
  audited on a real N=5,000 run plus full-file raw audits (§12 below).
- `--mode test` wiring is statically verified correct end-to-end (§3 below) but **has
  never been executed at real test scale** (1,732,544 S1 entities). No
  `output/matching_results.tsv` or `output/candidate_pairs.tsv` exists anywhere in the
  repo as of this writing — `output/` holds only `.gitkeep` and `README.md`.
- `docs/methodology/Documentation_template.md` is still the placeholder text ("replace
  with the official `Documentation_template.md`... and fill it in at CP12") — **not**
  filled in.
- `docs/planning/plan.md`'s checkpoint tracker shows every checkpoint (CP0 through CP13)
  still `[ ]` unchecked except sub-items of CP0; the "Submissions used" counter reads
  `Day 1 0/5 · Day 2 0/5 · Day 3 0/5` — **zero leaderboard submissions have been made**.
- No trained model artefact, OOF prediction file, or experiment-tracking CSV row exists
  from a full-scale run — everything measured so far is from N ≤ 10,000 sampled runs.

---

## 2. Official Challenge Requirements

Transcribed directly from `docs/challenge/problem_statement.md` and
`docs/challenge/guidelines.md` (the two authoritative sources — nothing below is
inferred from project docs).

### 2.1 Task
Given S1 (deduplicated reference), S2, S3 business records, output for every S1 test
entity the list (possibly empty) of S2/S3 entities describing the same real-world
business. Train countries: US, India. **Test adds France, unseen in training** — country
must be treated as an open set (problem_statement.md, "Data Description").

### 2.2 Required output files (leaderboard-scored vs. package-required vs. internal)

| File | Required by | Scored on leaderboard? | Required in final zip? |
|---|---|---|---|
| `output/matching_results.tsv` | problem_statement.md "Output Format" | **Yes — the only file scored** | Yes |
| `output/candidate_pairs.tsv` | problem_statement.md "Output Format" | No — used only to audit blocking recall/reduction ratio and pipeline correctness | Yes |
| `code/business_entity_resolution/src/` (all source) | problem_statement.md "Final Submission Package" | No | Yes |
| `code/business_entity_resolution/README.md` | problem_statement.md "Final Submission Package" | No | Yes |
| `code/business_entity_resolution/requirements.txt` | problem_statement.md "Final Submission Package" | No | Yes |
| `Documentation_template.md` (filled in) | problem_statement.md "Final Submission Package" + "Submission Requirements" §3 | No | Yes, at zip root |
| `MODELS.md` | Not named by the official docs — this is a **project-invented** compliance artefact for CLAUDE.md's model-licence rule, not an official Amazon requirement | No | Not named by the official structure diagram, but harmless to include under `code/business_entity_resolution/` |
| `experiments/`, `artifacts/`, `dataset/` | Not required — `docs/planning/plan.md`'s CP12 explicitly instructs excluding `dataset/`, `code/artifacts/`, `scratch/`, `docs/`, `experiments/` from the zip | No | **No — must be excluded** |

**Exact required zip structure** (problem_statement.md, "Final Submission Package"):
```
<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── code/
│   └── business_entity_resolution/
│       ├── src/
│       ├── README.md
│       └── requirements.txt
└── Documentation_template.md
```
Note the nested `code/business_entity_resolution/` name is required by the official
docs even though the working repo (this repo) keeps a flat `code/` — `docs/planning/
plan.md`'s CP12 already documents the required rename-on-zip step.

### 2.3 Output format rules (exact, from problem_statement.md)
1. Tab-separated files; columns `source1_entity_id`, `matched_entity_ids` (or
   `candidate_entity_ids`).
2. Exactly one row per test-set S1 entity.
3. Empty string for `matched_entity_ids`/`candidate_entity_ids` when there are no
   matches/candidates (not an omitted row).
4. No duplicate entity IDs within a single ID list.
5. ID lists comma-separated, no quoting.
6. `matched_entity_ids` must only reference S2/S3 IDs that exist in the **test** set — no
   self-matches to S1, no IDs absent from the test files.
7. No duplicate `source1_entity_id` rows.
8. **Matched IDs must be a subset of that S1's candidate list** — "a matched ID that
   never appeared as a candidate signals a pipeline bug."
9. `candidate_pairs.tsv` must be **the exact candidate set the final matching model
   scored for inference** — not an earlier/looser blocking pass later filtered further.

### 2.4 Constraints (problem_statement.md, "Constraints", numbered 1–5)
1. Format exactly as above; malformed submissions are not evaluated.
2. `matched_entity_ids` only S2/S3, existing in test set.
3. Every S1 test entity must appear.
4. No duplicate IDs in a list or duplicate S1 rows.
5. **"Final model should be an MIT/Apache 2.0 License model and up to 8 Billion
   parameters."** (This is the exact official wording; CLAUDE.md's stricter phrasing
   "must be MIT or Apache-2.0" is a project restatement, not a separate rule.)

### 2.5 Evaluation
Macro-averaged F_0.5 per S1 entity (formula in problem_statement.md), singletons
included (empty-list correct prediction = 1.0, any prediction on a true singleton =
0.0). **This is never "accuracy"** — every reference in this document to the pipeline's
score uses "macro F0.5," not "accuracy."

### 2.6 Submission mechanics (guidelines.md + problem_statement.md)
- Challenge window: 25–27 Sep 2026 (IST). Max **5 submissions/day for 3 days** (15
  total) — `matching_results.tsv` uploaded to the Portal each time.
- Public leaderboard during the challenge (subset of test), private leaderboard after
  (remaining portion) — **final ranking uses the private leaderboard**.
- Top-100 teams additionally submit: methodology, candidate generation/blocking
  strategy, model architecture and feature engineering.
- guidelines.md says the best-solution artefact is a "1-2 page document"; problem_
  statement.md's `Documentation_template.md` requirement says "no page limit." Both are
  transcribed verbatim in `docs/challenge/guidelines.md`'s own note; `docs/planning/
  plan.md`'s working resolution (lead with a 1-2 page executive summary, then the full
  doc) is a project decision, not an official rule — flagged here as an **unresolved
  ambiguity in the official docs themselves**, not a repo defect.
- **External data lookup is strictly prohibited** (entity-resolution APIs, business
  registries, geocoding APIs, web scraping, any internet augmentation) —
  disqualification on evidence of use.

---

## 3. Current Submission Compliance

**No submission has ever been produced.** Every row below is therefore judged either
against the static code path (`io_utils.write_submission`, traced in
`experiments/reports/leaderboard_submission_readiness.md`) or marked NOT VERIFIED
because no real output bytes exist to check.

| Requirement (problem_statement.md) | Enforced by `write_submission`? | Status | Evidence |
|---|---|---|---|
| Tab-separated, `\n` line endings | `_write_two_col`, `open(path, "w", newline="\n")` | PASS (code-level) | `leaderboard_submission_readiness.md` §4 |
| Exact headers | `config.MATCHING_HEADER`/`CANDIDATE_HEADER` match verbatim | PASS (code-level) | same §4 |
| IDs as strings, no float coercion | `read_tsv(dtype=str)`, `_clean_list` asserts `isinstance(raw, str)` | PASS (code-level) | `data_correctness_audit.md` §12 |
| One row per test S1, no dup S1 rows | `write_submission` raises `ValueError` on any duplicate `s1_ids` | PASS (code-level, and exercised by `test_cli_reads_files_from_config_paths`) | `leaderboard_submission_readiness.md` §2G |
| Empty string for no-match/no-candidate | `",".join([])` → `""` | PASS (code-level) | same §4 |
| No dup IDs within a list | `_clean_list`'s `seen` set | PASS (code-level) | same §4 |
| S2-/S3- only, no S1 self-match | `_clean_list` requires `MATCH_PREFIXES` | PASS (code-level) | same §4 |
| ID must exist in that split's S2/S3 files | `rid not in valid_ids` check | PASS (code-level) | same §4 |
| Matched ⊆ candidates | `missing = set(match) - set(cand)` raises if non-empty | PASS (code-level, hard guarantee — write_submission is fail-closed) | same §4 |
| `candidate_pairs.tsv` == exact set scored by the model | `candidates = candidates_to_lists(scored)` built from the same `feats_te` frame the model scores, not an earlier `cands_te` | PASS (code-level, train-side set-equality **measured** 193,324==193,324; test-side is code-identical but **INFERRED**, not independently re-run at scale) | `data_correctness_audit.md` §4 |
| Actual `matching_results.tsv` / `candidate_pairs.tsv` exist and are correct at real test scale | — | **NOT VERIFIED** — files do not exist yet | `leaderboard_submission_readiness.md` §5, confirmed again this session (`output/` still holds only `.gitkeep`/`README.md`) |
| `python3 utils/validate_submission.py ... → PASS` | — | **NOT VERIFIED** — nothing to validate | same |
| Model licence/size constraint (MIT/Apache-2.0, ≤8B params) | — | PASS (both models in the pipeline logged) | `code/MODELS.md` (§4 below) |
| No external data lookup | — | PASS (by design — no network/API calls found anywhere in `code/src/`, confirmed by every audit's grep passes) | `data_correctness_audit.md` §12, `sagemaker_storage_adaptability_audit.md` §5/§11 |

**`utils/validate_submission.py` vs. official requirements vs. what `write_submission`
already guarantees** (per `utils/README.md` and the problem statement):
- The validator checks: genuine tab-separation, exact headers, one row per required S1
  (no missing/no duplicate), no in-list duplicate IDs, S2-/S3- prefix only, and (with
  `--check-ids`, off by default) that every ID actually exists in the test files. It also
  **warns (never fails)** if a matched ID is missing from that S1's own candidate list.
- Everything the validator checks is **also already enforced, fail-closed, at write time**
  by `write_submission` (per the table above) — so for output produced by this pipeline
  specifically, the validator is a second, independent confirmation rather than the only
  gate. This matters because the validator is organizer-authored and must not be
  modified; the pipeline's own writer independently duplicating those guarantees is a
  defence-in-depth property, not a requirement.
- One asymmetry: the validator's matched-⊆-candidates check is a **warning**, while
  `write_submission` makes it a **hard `ValueError`** (refuses to write anything). This
  means the pipeline's own guarantee is strictly stronger than what the validator alone
  would catch — a looser third-party pipeline could pass the validator with a warning it
  ignores, but this codebase cannot produce such a file at all.
- No requirement in the official docs is left unchecked by both together, based on the
  cross-reference performed in `leaderboard_submission_readiness.md` §4.

---

## 4. ML Pipeline Artifact Inventory

Classification key: **(A)** officially required by Amazon, **(B)** required by this
pipeline's own logic, **(C)** required for reproducible ML development, **(D)**
recommended for debugging/experimentation, **(E)** optional/nice-to-have.

| Artifact | Where produced | Class | Currently exists? |
|---|---|---|---|
| `train_source1/2/3.tsv`, `train_ground_truth.tsv`, `test_source1/2/3.tsv` | `dataset/` (S3 download, gitignored) | (A) input data, not an output requirement, but the whole task depends on it | Yes, present locally (row counts verified §5 below) |
| Normalised S1 / S2+S3 frame cache (`artifacts/*.parquet`) | `run_pipeline._cached` via `normalize.py` | (D) — purely a recomputation-avoidance cache; pipeline is correct without it | Only at N≤10,000 sample scale so far |
| Sentence-embedding cache (`artifacts/emb_<hash>.npy`) | `blocking.compute_embeddings` | (B) — blocking pass 2 needs the embeddings to exist for a run, but the **cache file** itself is (D), recomputable | Only at sample scale |
| Blocking candidate cache (`artifacts/cands_<hash>.parquet`) | `run_pipeline.block` | (D) | Only at sample scale |
| Trained LightGBM model (`artifacts/model_final.txt`) | `model.py` (test mode only) | (C) — needed to reproduce a specific submission without retraining, but not officially required to be shipped as a file (the code that produces it is what's required) | **Does not exist** — no test-mode run has completed |
| OOF predictions | in-memory only (`fit_and_tune`); not persisted to disk anywhere in the code | (C) for threshold-tuning provenance | Not persisted at all — would need to be added if desired |
| `artifacts/errors_fp.tsv` / `errors_fn.tsv` (50 worst FP/FN) | `run_pipeline.dump_errors` | (D) per CLAUDE.md §6's explicit error-analysis mandate | Only at sample scale (see §12) |
| `artifacts/importance_valid.tsv` / `importance_final.tsv` | `run_pipeline.py` | (D) | Only at sample scale |
| `code/experiments.csv` | `run_pipeline.log_experiment()` | (C) — the run-level metric ledger CLAUDE.md §4 mandates | Regenerated per run, gitignored; not committed at any point (no full-scale row exists) |
| `experiments/reports/*.md` + `experiments/results/*.json` | Hand-written / driver-script output (this session's predecessors) | (C)/(D) — human-curated evidence, not pipeline output | Yes, 7 reports + 8 JSON files (inventoried in this document) |
| `code/artifacts/indexes/` | Nothing writes here | N/A | **Confirmed unused** — `blocking.dense_topk` is an exact brute-force chunked top-k search (optionally GPU-accelerated via torch), not an ANN index; no FAISS or any index object is ever built or persisted anywhere in `code/src/`. The directory exists only as `.gitkeep` scaffolding for a design possibility that was never implemented. Do not treat this as a missing artefact — it is correctly empty. |
| `output/matching_results.tsv`, `output/candidate_pairs.tsv` | `write_submission` (test mode) | **(A) — the only officially required, leaderboard-scored deliverables** | **Do not exist** |
| `docs/methodology/Documentation_template.md` (filled in) | Manual | (A) | **Placeholder text only, not filled in** |
| `code/MODELS.md` | Manual, per CLAUDE.md §2.2 | (C) — project-mandated compliance record, not itself named by the official docs, but directly supports official Constraint 5 | Yes, filled in for both models used (§ below) |
| `code/src/s3_sync.py` + `code/tests/test_s3_sync.py` | Added in a prior session | (D)/(E) — opt-in persistence convenience for the team's SageMaker workflow; **not part of the ML algorithm** and not referenced by `run_pipeline.py` | Yes, present, 33 tests passing |

---

## 5. Current Baseline

The only real, executed baseline is the four-point N-ladder (N=500/2,000/5,000/10,000
train S1 entities) from `experiments/reports/baseline_architecture_validation.md`,
re-confirmed by the independent data-correctness audit's N=5,000 re-run
(`data_correctness_audit.md` §3, which reproduced pair recall 0.98125 vs. the baseline's
0.9812 — a genuine independent-reproducibility check, not a copy).

**Environment for every number below**: `torch-gpu` conda env, Python 3.10.20, PyTorch
2.5.1+cu121, CUDA available, NVIDIA GeForce RTX 4060 Laptop GPU (8 GB). Git commit at
baseline-report time: `352a403`.

### 5.1 Dataset scale (full files, measured)

| Split | S1 | S2 | S3 | Ground truth |
|---|---|---|---|---|
| Train | 2,206,821 | 5,034,616 | 5,285,603 | 2,206,821 |
| Test | 1,732,544 | 4,887,273 | 5,082,316 | n/a (none provided) |

Train: US 1,323,633 / India 883,188 (S1), **zero France rows anywhere in train** —
verified by full-file count, not just by omission (`data_correctness_audit.md` §1).
Test: US 663,106 / India 809,986 / **France 259,452** (S1) — France is a real, populated
third bucket at full test scale, not an edge case.

### 5.2 Baseline results table (N=500/2,000/5,000/10,000, all VERIFIED/measured)

| Sample | Pair Recall | S1 Full Recall | Cands/S1 (mean) | Search-Space Reduction | Final Macro F0.5 (held-out 20%) | tau selected | Total runtime |
|---|---|---|---|---|---|---|---|
| N=500 | 0.9906 | 0.9740 | 31.34 | 98.64% | **0.9929** | 0.900 | 149.1 s |
| N=2,000 | 0.9864 | 0.9617 | 36.22 | 99.61% | **0.9938** | 0.550 (singleton_tau=0.925) | 108.6 s |
| N=5,000 | 0.9812 | 0.9477 | 38.66 | 99.84% | **0.9900** | 0.700 | 172.4 s |
| N=10,000 | 0.9769 | 0.9361 | 39.98 | 99.91% | **0.9871** | 0.750 (singleton_tau=0.900) | 246.3 s |

**Critical caveat, repeated because it matters for every downstream decision in this
document: these are TRAIN-SIDE held-out validation numbers on small S1 samples (≤10,000
of 2,206,821 train S1s), not a leaderboard score.** No test-set score exists anywhere in
this repo — the test set carries no ground truth, by design of the challenge. Treat
0.987–0.994 as an optimistic, small-sample signal, **not** a predictor of the private
leaderboard result.

### 5.3 Per-country and per-source breakdown (N=10,000, measured)
- Blocking recall: India **0.9523**, US **0.9932** (a ~4-point gap, stable across all
  four N values — India recall at N=500/2,000/5,000 was 0.9796/0.9700/0.9597 vs. US
  0.9990/0.9969/0.9955).
- Final F0.5 (held-out): India **0.9759**, US **0.9947** — the gap survives downstream
  (damped from ~4 points at blocking to ~1.9 points at the final metric).
- Source-wise blocking recall: S1→S2 **0.9753**, S1→S3 **0.9784** — not a material
  recall driver.
- Per-pass blocking recall (N=10,000): TF-IDF 0.8365, embedding **0.8825** (best single
  pass), rare-token 0.7514, digit-token 0.6590, address 0.6653 — none alone reaches the
  combined 0.9769, confirming the 5-pass union does real, non-redundant work.

### 5.4 Error budget (N=10,000 held-out, "earliest stage a missed true match was lost
at", measured)

| N | True pairs (held-out) | Total missed | Blocking FN | Below threshold | Removed by 1-to-1 |
|---|---|---|---|---|---|
| 500 | 345 | 7 | 4 (57.1%) | 3 (42.9%) | 0 (0.0%) |
| 2,000 | 1,381 | 16 | 14 (87.5%) | 2 (12.5%) | 0 (0.0%) |
| 5,000 | 3,467 | 77 | 66 (85.7%) | 11 (14.3%) | 0 (0.0%) |
| 10,000 | 6,936 | 214 | 178 (83.2%) | 36 (16.8%) | 0 (0.0%) |

**Blocking false negatives dominate at 57–88% of all misses** — this is the single most
important, repeatedly-confirmed finding in the repo (see §8, §18).

### 5.5 One-to-one assignment (measured, all N)

| N | Candidates with >1 S1 owner (pre-assignment) | True matches removed by 1-to-1 |
|---|---|---|
| 500 | 778 | 0 |
| 2,000 | 3,841 | 0 |
| 5,000 | 9,692 | 0 |
| 10,000 | 19,077 | 0 |

**Zero true matches removed by one-to-one assignment at every sample size tested** —
this "free" property (`ONE_TO_ONE=True` costs nothing measured so far) is confirmed
independently twice: once in the baseline report, once by the data-correctness audit's
walkthrough of real S1 IDs (`data_correctness_audit.md` §8).

---

## 6. Development Checklist

Reflects real repo state, not aspirational status.

- [x] Repo skeleton matches CLAUDE.md §3/§4 layout (`code/src/*.py` all present).
- [x] `evaluate.py` macro F0.5 implemented and unit-tested (part of the 138-test suite).
- [x] `io_utils.write_submission()` implemented and enforces every format rule
      (fail-closed, tested — `test_io_utils.py`).
- [x] Blocking, features, model, decide implemented and exercised end-to-end up to
      N=10,000 train S1 samples.
- [x] France (unseen-country) path exercised end-to-end on synthetic data
      (`test_test_run_writes_valid_submission_with_unseen_country`).
- [x] Data correctness audited stage-by-stage on a real N=5,000 run plus full-file raw
      audits (§12 below).
- [x] `--mode test` code path statically traced and found correct (§3 above).
- [x] Opt-in S3 sync utility added (`s3_sync.py`), zero impact on `run_pipeline.py`.
- [ ] **Full-scale `--mode test` run has never been executed.**
- [ ] `output/matching_results.tsv` / `output/candidate_pairs.tsv` do not exist.
- [ ] `utils/validate_submission.py` has never been run against real output (nothing to
      validate yet).
- [ ] `docs/methodology/Documentation_template.md` is still a placeholder.
- [ ] Zero leaderboard submissions made (`plan.md` submission log: 0/5, 0/5, 0/5).
- [ ] `code/requirements.txt` / `torch-gpu` environment mismatch unreconciled (known
      gap, carried forward across three audits — see §16).
- [ ] LOCO (leave-one-country-out) validation, mentioned throughout CLAUDE.md and
      `code/README.md`'s CLI (`--loco` flag exists), has **no evidence of having been
      run** anywhere in `experiments/reports/` or `experiments/results/` — flagged as
      NOT TESTED, not assumed done just because the flag exists in the CLI.
- [ ] Error-analysis dumps (`artifacts/errors_fp.tsv`/`errors_fn.tsv`) exist only at
      sample scale (N≤10,000); no full-scale error analysis has been performed (there is
      no ground truth on the real test set to analyse against anyway — this would need
      to happen on the train-side held-out split at larger/full scale).

---

## 7. Embedding Experiments

**Model actually used** (per `code/MODELS.md` and `config.py`, cross-checked, not
guessed): `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2`, revision
`e8f8c211226b894fcb81acc59f3b34ba3efd5f42`, **Apache-2.0**, **117,654,272 parameters
(≈118M, well under the 8B constraint)**, verified via the HF model API
(`cardData.license`, `safetensors.total`) on 2026-09-25. Used in `blocking.py`'s
embedding pass (pass 2 of 5) and `features.py`'s embedding-cosine feature. This is
class (A)-adjacent: not itself named by the official docs, but is the model whose
licence/size the official Constraint 5 governs, and `MODELS.md` (a project-invented
compliance artefact, class C) records it correctly.

- Batch size: `EMBEDDING_BATCH=512`. Dense top-k: `K_EMBEDDING=20`, chunked exact
  brute-force search (`DENSE_QUERY_CHUNK=4096`, `DENSE_INDEX_CHUNK=262,144`), runs on
  GPU when CUDA is available (confirmed used in every baseline run — GPU peak VRAM
  909 MB → 1,141 MB across N=500→10,000).
- Cache: content-hash-keyed (`SHA-1` of model name + `EMBEDDING_REVISION` + text
  content) at `artifacts/emb_<hash>.npy`, float16, L2-normalised — safe to reuse across
  runs/instances on identical inputs (`sagemaker_storage_adaptability_audit.md` §7-8).

**Checklist:**
- [x] Embedding pass runs correctly and produces the single highest-recall individual
      blocking pass measured (0.8825 at N=10,000, vs. TF-IDF's 0.8365) — §5.3.
- [x] Embeddings are content-addressed and cached, safe under multi-instance reuse.
- [ ] Embedding quality/coverage at full test scale (11.7M+ S2/S3 records, ~9 GB
      estimated float16 disk footprint per
      `sagemaker_storage_adaptability_audit.md` §14) is **UNKNOWN, extrapolated only**
      — no full-scale embedding run has been performed.
- [ ] No ablation exists comparing blocking recall with vs. without the embedding pass
      at any N — the 0.8825 figure is the pass's own standalone recall (measured), not
      a with/without-this-pass delta on the union. Worth running as a future experiment
      (see §19).
- [ ] No alternative embedding model has been evaluated (e.g. a different MIT/Apache
      multilingual model) — only one model has ever been tried, per `MODELS.md`'s single
      row.

---

## 8. Blocking Experiments

This is the highest-priority area by evidence: **blocking false negatives are the
dominant loss category (57–88% of all misses) at every sample size tested**, and
blocking pair recall **degrades monotonically as the candidate pool grows** (§5.2).

### 8.1 Measured checklist (values cited exactly, N=10,000 unless noted)

- [x] Pair recall: **0.9769** (N=10,000); trend across N: 0.9906 → 0.9864 → 0.9812 →
      0.9769 (monotonic decline as N grows).
- [x] S1 full recall: **0.9361** (N=10,000); trend: 0.9740 → 0.9617 → 0.9477 → 0.9361
      (declines faster than pair recall).
- [x] Candidates/S1 (mean): **39.98** (N=10,000); trend: 31.34 → 36.22 → 38.66 → 39.98.
      Candidate cardinality distribution (N=5,000, `data_correctness_audit.md` §3): min
      20, median 38, mean 38.66, p90 45, p95 47, p99 52, max 63 — tight and bounded.
- [x] Search-space reduction: **99.91%** (N=10,000); trend: 98.64% → 99.61% → 99.84% →
      99.91% (improves with N simply because the S2+S3 denominator grows faster than the
      bounded per-S1 candidate count — not itself a quality signal).
- [x] Per-country recall gap: India 0.9523 vs. US 0.9932 (N=10,000) — persistent and
      stable at every N (§5.3).
- [x] Per-source recall: S1→S2 0.9753, S1→S3 0.9784 (N=10,000) — not a material driver.
- [x] Per-pass recall (N=10,000): TF-IDF 0.8365, embedding 0.8825, rare-token 0.7514,
      digit-token 0.6590, address 0.6653 — all five contribute; union recall (0.9769)
      exceeds every individual pass, confirming genuine complementarity.
- [x] Candidate ID existence / duplicate-pair integrity: **0** generated candidate IDs
      absent from the blocked-against S2+S3 frame; **0** duplicate `(s1_id, cand_id)`
      rows in raw blocking output (N=5,000, real run — `data_correctness_audit.md` §3).
- [x] S1 coverage: **0%** of S1s got zero candidates at N=5,000 (every S1 has ≥1
      same-country S2/S3 record).
- [x] Recall by S1 group: multi-match 0.9812, singleton-match 0.9848 (N=5,000) — not
      meaningfully worse for the harder multi-match case.
- [x] **India-gap root cause: CONFIRMED**, not merely correlated. 12/15 sampled India
      blocking false-negative pairs show a dual mechanism, evidenced with real example
      text (`data_correctness_audit.md` §3): (a) the shared ISCII transliteration table
      (`normalize.py`'s `transliterate_indic`) produces a phonetically-related but
      weakly-overlapping string vs. the English S1 spelling (e.g. `"lakshmi"` →
      `"lakasami"`, `"constructions"` → `"kansatrakasanaj"`), hurting all three
      name-based passes simultaneously on the same pairs; (b) independently, Indian S2/S3
      addresses are more often truncated or digit-altered (e.g. `"233/6/52F"` vs. `"Hn
      439 233/6/52F"`; `"Flat No.A-312"` vs `"Flat No.a-12"`), which breaks the
      exact-key digit/address passes (`blocking.py`'s `key_pass` requires an *exact*
      shared key, no fuzzy fallback at blocking time) on the same pairs. India pairs
      therefore have fewer independent passes able to "rescue" them than US pairs. This
      is a joint, evidence-backed mechanism, not a restatement of correlation.
- [x] France blocking: **train has zero France rows, so no train-side France blocking
      recall can ever be measured** (this is stated as fact, not a gap to be filled by
      guessing). Test-side France blocking has no ground truth to score against (no test
      labels exist at all, for any country). The only France-specific verification that
      exists is functional/structural: normalisation and blocking apply identically to
      France (no country-literal branches found by grep — `data_correctness_audit.md`
      §2/§3/§12), and the synthetic France test in `test_pipeline.py` exercises
      non-trivial French predictions end-to-end.
- [ ] No k-value ablation exists (e.g. does raising `K_TFIDF_NAME`/`K_EMBEDDING` recover
      recall at high N, or is the ceiling a genuine similarity-ranking limit?) — this was
      explicitly flagged as an evidence-backed next step by the baseline report itself
      (§17 point 1) and has **not** been run.
- [ ] No decomposition of the 178 N=10,000 blocking-FN pairs by "name-driven vs.
      address-driven miss" exists — flagged by the baseline report (§17 point 2), not
      done.
- [ ] Full test-scale (1.73M S1 × 11.7M S2+S3) blocking has never been run; runtime,
      memory and recall at that scale are **UNKNOWN**, not merely unmeasured-but-assumed-
      fine (`sagemaker_storage_adaptability_audit.md` §14 explicitly flags candidate-pair
      and feature-matrix disk/RAM footprint at full scale as UNKNOWN).

### 8.2 `BLOCK_WITHIN_COUNTRY=True` justification
Confirmed on two independent bases: (1) full-train-ground-truth audit (cited in
CLAUDE.md, "100.0000% same-country matches, 0 cross-country") — this document does not
re-derive that full-population number, it is carried forward as previously established;
(2) code-level confirmation that no country string literal appears anywhere in
`code/src/*.py` (grep, `data_correctness_audit.md` §12), so the country-grouping
mechanism is generic and requires no change for France.

---

## 9. Feature Experiments

`features.py` produces **58 numeric columns** per candidate pair (measured, N=10,000
and independently re-confirmed at N=5,000).

- [x] 0 `inf` values anywhere; 0 columns 100% NaN (N=5,000, real run).
- [x] Feature separability diagnostics computed for all 58 columns at N=10,000: **41
      "clearly informative" (|SMD| ≥ 0.8)**, 5 "weakly informative" (0.3–0.8), 6 "highly
      overlapping" (<0.3), 2 constant/near-constant.
- [x] Top separators (SMD, N=10,000): `landmark_jaccard` 12.54 (but 98.2% missing — only
      fires on a small subpopulation with landmark phrases), `addr_token_set` 7.01,
      `addr_tfidf_cos` 6.79, `long_num_equal` 6.17 (99.0% missing), `gap_to_best_s1`
      −3.92, `digit_conflict` −3.84, `n_passes` 2.93, `rank_in_s1` −2.47, `emb_cos` 1.79.
- [x] Two exactly-constant features, both confirmed **structural, not bugs**, and
      confirmed twice independently (baseline report + data-correctness audit, and the
      latter additionally verified at full-population scale):
  - `same_country` — constant 1.0, because `BLOCK_WITHIN_COUNTRY=True` means every
    candidate pair is by construction same-country.
  - `addr_empty_s1` — constant 0.0, because `train_source1.tsv`'s `business_address`
    column is **0.0% empty across all 2,206,821 rows**, and `test_source1.tsv` likewise
    **0.0% across 1,732,544 rows** (full-file measurement, not sampled) — S1, the
    deduplicated reference source, simply never has a missing address in either split.
- [x] Manual trace of representative real pairs confirms the intended behaviour: a true
      match scores `name_jw=1.0`/`addr_tfidf_cos=1.0`/`digit_conflict=0.0`; a random
      non-match for the same S1 scores substantially lower on every axis; a
      deliberately-hard "same-name-different-place" negative (`name_token_set=0.909`
      but `digit_conflict=1.0`) is correctly separated by the address-conflict features
      — exactly the case `features.py`'s own design note says these features exist to
      catch (`data_correctness_audit.md` §5).
- [x] Label join is ID-based (not positional), exact-count-consistent with measured
      blocking recall (17,057 positive labels present in `feats` == 17,383 × 0.981246
      pair recall, to the row), 0 duplicate `(s1_id, cand_id)` label rows, 200/200
      sampled positive labels verified correct (`data_correctness_audit.md` §6).
- [ ] `FeatureContext` is refit separately per split (train vs. test), so
      `name_tfidf_cos`/`addr_tfidf_cos` are computed in different vector spaces per
      split — intentional per the module's own docstring (captures French n-grams that
      never occur in train), **confirmed as a design choice, not verified as
      calibrated** (no labeled France data exists to check whether this helps or hurts).
      Flagged as a design note, not an issue (`data_correctness_audit.md` §11, item 3).
- [ ] No feature ablation study exists (removing individual "highly overlapping"
      features like `acronym_match`/`cand_has_dba` and measuring F0.5 impact) — not run.
- [ ] No full-scale (test) feature-matrix run has been performed; peak RAM at full scale
      is UNKNOWN (estimated ~16 GB if fully materialised at once, per
      `sagemaker_storage_adaptability_audit.md` §14, but `build_features` is chunked
      (`FEATURE_CHUNK=1,000,000`) and features are never written to disk, so this is
      transient RAM pressure per chunk, not measured disk usage).

---

## 10. Model Experiments

LightGBM 4.7.0 (MIT), binary classifier, `GroupKFold(5)` by S1 ID, `LGB_PARAMS` per
`config.py` (num_leaves=127, learning_rate=0.05, min_child_samples=50, feature_fraction/
bagging_fraction=0.8, lambda_l2=1.0, seed=42, deterministic=True), early stopping at 100
rounds, max 2000 rounds.

- [x] Class (B)/(C) — trained from scratch on provided data only; libraries-not-a-model
      per CLAUDE.md §2.2, correctly logged in `MODELS.md` under "Models trained from
      scratch."
- [x] Training pair volume and positive rate (measured, all N): N=500 → 12,625 pairs,
      10.68% positive; N=2,000 → 57,935 pairs, 9.43%; N=5,000 → 154,628 pairs, 8.83%;
      N=10,000 → 319,992 pairs, 8.50%. Positive rate falls monotonically with N (more
      orphan/negative volume enters proportionally via `subsample_train`).
- [x] Runtime scales near-linearly with pair count (Train+OOF: 3.15 s → 22.40 s across
      N=500→10,000; Full-model train: 0.62 s → 3.11 s) — no blowup observed at tested
      scale.
- [x] GroupKFold leakage re-checked at the data level (not just via the unit test): **0
      S1s span more than one fold** on a real 4,000-S1 training subset
      (`data_correctness_audit.md` §7).
- [x] Prediction sanity: finite everywhere, min 3.97e-6, max 0.99996 (never exactly
      0/1 — consistent with a real sigmoid, not a degenerate constant).
- [x] OOF positive-class distribution is heavily right-shifted vs. negative class at
      every N (clean separation, `figures/score_distribution_pos_vs_neg.png`).
- [ ] No hyperparameter tuning pass exists (CLAUDE.md §6's suggested "num_leaves,
      min_child_samples, feature fraction, 3-seed average" tuning has not been run) —
      the current `LGB_PARAMS` are the original defaults chosen at pipeline design time,
      never validated against alternatives.
- [ ] No separate singleton classifier (CLAUDE.md §6.4's optional upgrade) has been
      built or evaluated.
- [ ] No cross-encoder feature (CLAUDE.md §6.4's optional upgrade, gated on beating
      LightGBM CV) has been attempted.
- [ ] No full-scale (all 2,206,821 train S1) training run has ever completed — the
      largest training run measured is N=10,000 (0.46% of full train).
- [ ] No trained model artefact (`artifacts/model_final.txt`) exists anywhere in the
      repo currently — only in-memory models from sampled runs have ever existed.

---

## 11. Threshold & Decision Experiments

`decide.tune_threshold` sweeps `config.TAU_GRID` (0.30 to 0.95, step 0.025, 27 values)
on OOF predictions, maximising macro F0.5 including singletons; `decide.
assign_one_to_one` runs before thresholding.

- [x] Selected tau is **not stable across sample sizes**: N=500 → tau=0.900,
      N=2,000 → tau=0.550 (singleton_tau=0.925), N=5,000 → tau=0.700, N=10,000 →
      tau=0.750 (singleton_tau=0.900). This instability is explicitly attributed to
      tuning on a small OOF set (400–8,000 groups), not a pipeline defect — the
      diagnostic sweep shows a broad, fairly flat plateau near the optimum rather than a
      sharp peak (`baseline_architecture_validation.md` §9).
- [x] Every selected tau is above 0.5, matching CLAUDE.md's own expectation (driven by
      ~25% of S2/S3 records being orphans, the main false-positive source).
- [x] Precision stays near-ceiling at every N (0.997–1.000, pair-level; held-out
      country-level: US 1.000, India 0.9835 at N=500) — consistent with the metric's
      2× precision weighting and tau landing above 0.5.
- [x] Recall is the softer number (0.969–0.990 pair-level) and tracks blocking pair
      recall closely — reinforcing blocking, not thresholding, as the binding recall
      constraint.
- [x] One-to-one assignment: **zero true matches removed at every N tested** (§5.5),
      despite resolving thousands of raw candidate conflicts (778 at N=500 → 19,077 at
      N=10,000) — confirmed twice independently (baseline report + a real-S1 walkthrough
      in the data-correctness audit, including a zero-match S1, a singleton, and a
      multi-match-both-sources S1, all producing exactly the expected prediction).
- [x] Threshold removes 12.5%–42.9% of held-out misses (the complement of blocking-FN in
      the error budget table, §5.4).
- [ ] No LOCO-based threshold re-tuning has been performed (train-on-one-country,
      threshold-tune, score-on-the-other) — the `--loco` CLI flag exists but no evidence
      of it having been run was found anywhere in `experiments/`.
- [ ] `SINGLETON_THRESHOLD` toggles on/off between N values in an unexplained pattern
      (None at N=500/5,000, set at N=2,000/10,000) — not investigated further; flagged
      as an open question, not resolved.
- [ ] No threshold behaviour has been measured at real test scale, where the S2/S3 pool
      composition (and thus orphan rate) may differ from the sampled train subsets.

---

## 12. Error Analysis

- [x] Real, executed error-budget classification at every N (§5.4) — the strongest
      error-analysis evidence in the repo: for every true `(S1, match)` pair in the
      held-out set, classified as blocking-FN / below-threshold / removed-by-1-to-1 /
      correctly recovered.
- [x] 15 real India blocking false-negative pairs manually inspected with actual raw
      text (§8.1) — the strongest single piece of root-cause evidence in the repo.
- [x] One anomalous US false negative flagged and left unresolved as-is rather than
      force-explained: `S1-176336021`/`S3-893893792`, names share no resemblance
      (`"Jones Newhold"` vs. `"Arcorbibelo (ID: 81803)"`), matched by address alone —
      presented as either a deliberately hard synthetic case or ground-truth noise,
      neither confirmed nor denied (`data_correctness_audit.md` §3).
- [x] Manual feature trace on 3 representative real pairs (true match, random negative,
      hard negative) — §9 above.
- [ ] The mandated `artifacts/errors_fp.tsv`/`errors_fn.tsv` (50 worst FP/FN dump,
      CLAUDE.md §6) has been produced only at sample scale during the baseline/audit
      runs, transiently — **no such file currently exists in the repo** (it is
      gitignored and regenerated per-run; no run's output was preserved as a committed
      artefact). This is expected behaviour per the gitignore design, not a gap, but it
      means there is no persistent, inspectable error-dump file to point to today.
- [ ] No systematic grouping of errors into CLAUDE.md §6's suggested causes ("suffix
      confusion, chains/branches of the same brand, landmark addresses, transliteration,
      typos") beyond the transliteration/address-truncation root cause already
      identified for India (§8.1) has been performed — CP6 ("Error analysis round") in
      `plan.md` is still unchecked.

---

## 13. Evaluation

`evaluate.py`'s macro F0.5 scorer (BETA=0.5, singleton rules per CLAUDE.md §1) — unit
tested against the exact problem_statement.md worked example (pred `[47,193,812]` vs.
truth `[47,812]` → 0.714) plus singleton edge cases (empty/empty → 1.0, any/empty → 0.0,
empty/non-empty → 0.0), per `plan.md`'s CP0 record; this is part of the 138-test suite.

- [x] F0.5 distribution characterised at N=10,000 (held-out 20% = 2,000 S1s): mean
      0.9871, median 1.0, 91.2% score exactly 1.0, 0.35% score exactly 0.0 — strongly
      bimodal, typical for a per-entity set-matching metric on mostly-clean data.
- [x] Per-country F0.5 measured (US 0.9947, India 0.9759 at N=10,000).
- [x] Singleton vs. non-singleton F0.5 tracked at every N (e.g. N=500: singleton_f05
      1.0, non_singleton_f05 0.9923).
- [x] Blocking recall / reduction ratio reporting (`report_blocking_stats`) exercised
      and cited throughout §5/§8.
- [ ] No score has ever been computed on the real test set (impossible without ground
      truth — by design of the challenge, this can only ever be a leaderboard score, not
      a local one). All "final" numbers in this document are train-side held-out
      numbers, repeated here for clarity per the documentation rules.
- [ ] No LOCO evaluation numbers exist (§11).

---

## 14. Experiment Tracking

### 14.1 Reusable template

```
### EXP-XXX: <short title>
- Status: [x] COMPLETED / [~] PARTIALLY TESTED / [ ] NOT TESTED / [!] BLOCKED
- Date:
- Area: <normalize|blocking|features|model|decide|evaluate|infra>
- Hypothesis:
- Why: <motivation, what evidence/finding prompted this>
- Configuration: <exact config.py values or diffs used>
- Dataset: <train sample / full train / valid split / France LOCO / etc.>
- Sample Size: <N S1 entities, actual achieved N if sampled>
- Code Version: <git commit hash>
- Metrics:
  | Metric | Value |
  |---|---|
- Results: <what was measured, in plain language>
- Error Analysis: <if applicable, FP/FN patterns found>
- Interpretation: <what this means for the pipeline>
- Decision: KEEP / REJECT / INVESTIGATE / BLOCKED
- Artifacts Saved: <paths, if any>
- Next Experiment: <what this motivates>
```

### 14.2 Retroactive write-up of real completed experiments

#### EXP-001: Baseline architecture validation, N=500
- Status: [x] COMPLETED
- Date: 2026-09-25 (session predating this document)
- Area: full pipeline (normalize → block → features → model → decide → evaluate)
- Hypothesis: the existing, unmodified pipeline runs end-to-end correctly and produces a
  measurable, reproducible baseline at small scale.
- Why: establish ground truth before any tuning/optimisation work.
- Configuration: all `config.py` defaults, unmodified.
- Dataset: train, S1-level deterministic sample (`subsample_train`, `seed=42`).
- Sample Size: requested N=500 → actual S1=500, S2=1,102, S3=1,211.
- Code Version: git commit `352a403`.
- Metrics:
  | Metric | Value |
  |---|---|
  | Pair recall | 0.9906 |
  | S1 full recall | 0.9740 |
  | Mean candidates/S1 | 31.34 |
  | Selected tau | 0.900 |
  | Held-out macro F0.5 | 0.9929 |
  | Blocking FN share of misses | 57.1% |
- Results: pipeline completed without crash/OOM/timeout; GPU peak VRAM 909 MB.
- Error Analysis: 7 missed true pairs in held-out set; 4 blocking-FN, 3 below-threshold,
  0 removed by one-to-one.
- Interpretation: pipeline mechanics are sound at this scale; blocking is already the
  dominant loss source even at the smallest N.
- Decision: KEEP (this is the established baseline, not a candidate to reject/replace).
- Artifacts Saved: `experiments/results/n500.json`, `experiments/reports/n500.md`.
- Next Experiment: repeat at N=2,000/5,000/10,000 to observe scaling behaviour → EXP-002/003/004.

#### EXP-002: Baseline architecture validation, N=2,000
- Status: [x] COMPLETED
- Date/Code Version: same session, `352a403`.
- Configuration/Dataset: identical to EXP-001, N=2,000.
- Sample Size: actual S1=2,000, S2=4,494, S3=4,860.
- Metrics: pair recall 0.9864, S1 full recall 0.9617, cands/S1 36.22, tau 0.550
  (singleton_tau=0.925), held-out macro F0.5 **0.9938** (highest of the N-ladder),
  blocking-FN share of misses 87.5%.
- Interpretation: recall already declining vs. N=500 despite F0.5 rising slightly —
  matcher/threshold absorb the added negative volume at this scale.
- Decision: KEEP (baseline data point).
- Artifacts Saved: `experiments/results/n2000.json`, `experiments/reports/n2000.md`.
- Next Experiment: EXP-003.

#### EXP-003: Baseline architecture validation, N=5,000
- Status: [x] COMPLETED
- Date/Code Version: same session, `352a403`.
- Sample Size: actual S1=5,000, S2=11,366, S3=12,093.
- Metrics: pair recall 0.9812, S1 full recall 0.9477, cands/S1 38.66, tau 0.700,
  held-out macro F0.5 0.9900, blocking-FN share of misses 85.7%.
- Interpretation: recall decline continues monotonically; F0.5 drops back below the
  N=500 figure, consistent with growing blocking-recall pressure.
- Decision: KEEP (baseline data point).
- Artifacts Saved: `experiments/results/n5000.json`, `experiments/reports/n5000.md`.
- Next Experiment: EXP-004.

#### EXP-004: Baseline architecture validation, N=10,000
- Status: [x] COMPLETED
- Date/Code Version: same session, `352a403`.
- Sample Size: actual S1=10,000, S2=22,767, S3=24,154.
- Metrics: pair recall **0.9769** (lowest of the ladder), S1 full recall **0.9361**,
  cands/S1 39.98, tau 0.750 (singleton_tau=0.900), held-out macro F0.5 **0.9871**
  (lowest of the ladder), blocking-FN share of misses 83.2%, India recall 0.9523 vs. US
  0.9932.
- Interpretation: this is the largest sample completed and confirms the trend seen at
  every step: blocking pair recall degrades monotonically with N while final F0.5
  degrades more slowly (matcher/threshold partially compensate). This N is the clearest
  evidence base for "blocking is the bottleneck" (§8, §18).
- Decision: KEEP (this is the current reference baseline for all planning in this
  document).
- Artifacts Saved: `experiments/results/n10000.json`,
  `experiments/results/n10000_source_recall.json`, `experiments/reports/n10000.md`,
  `experiments/reports/baseline_architecture_validation.md` (the synthesis report),
  7 figures under `experiments/reports/figures/`.
- Next Experiment: the baseline report's own §17 lists 4 evidence-backed candidates
  (k-value ablation, name-vs-address FN decomposition, India-gap root-causing,
  requirements.txt reconciliation) — the India-gap investigation was subsequently
  completed as EXP-005; the other three remain NOT TESTED (see §19).

#### EXP-005: Data correctness and lineage audit (incl. India-gap root cause)
- Status: [x] COMPLETED
- Date: 2026-09-25, git HEAD `352a403`.
- Area: cross-cutting (raw data → normalisation → blocking → features → labels → model
  → decide → submission-writer logic).
- Hypothesis: every stage transition produces exactly the shape/cardinality/ID semantics
  the next stage expects; the India blocking-recall gap has a findable root cause beyond
  correlation.
- Configuration: `config.py` defaults, unmodified; deterministic N=5,000 train sample
  (`seed=42`) plus full-file raw audits (no sampling) for structural checks.
- Dataset: train (full files for raw-integrity checks; N=5,000 S1 sample for stage-by-
  stage functional checks).
- Sample Size: 5,000 S1 (23,459 S2+S3 records) for functional checks; 2,206,821 S1 /
  5,034,616 S2 / 5,285,603 S3 / 2,206,821 GT for raw audits; 1,732,544 / 4,887,273 /
  5,082,316 for test-side raw audits.
- Code Version: git commit `352a403`.
- Metrics:
  | Metric | Value |
  |---|---|
  | Pair recall (N=5,000 reproduction) | 0.98125 (vs. baseline's 0.9812 — reproducibility confirmed) |
  | Candidate/model-input set equality (train) | 193,324 == 193,324 |
  | Positive labels present / total true pairs | 17,057 / 17,383 (== recall exactly) |
  | India FN rate vs. US FN rate | ~6× (279 India FN vs. 47 US FN, out of 6,924/10,459 true pairs respectively) |
  | Sampled India FN pairs showing dual root-cause mechanism | 12 / 15 |
- Results: no confirmed data-corruption bug anywhere in the traced pipeline. Two
  low-severity, already-known risks carried forward (tau instability, requirements.txt
  mismatch). One new finding: the India gap has a concrete, dual, evidence-backed
  mechanism (transliteration divergence + address truncation, acting jointly).
- Error Analysis: 15 real India blocking-FN text pairs manually inspected (§8.1).
- Interpretation: the pipeline's data plumbing is trustworthy; the India gap is a real,
  named, two-part mechanism rather than an unexplained correlation, which narrows future
  fix candidates to (a) a phonetic block key on the transliterated form, or (b) a
  fuzzy/single-edit-tolerant digit-pass key — neither implemented.
- Decision: KEEP (findings incorporated into planning; no code changed).
- Artifacts Saved: `experiments/reports/data_correctness_audit.md`,
  `experiments/results/data_correctness_raw_audit.json`,
  `experiments/results/data_correctness_n5000_summary.json`,
  `experiments/results/data_correctness_blocking_fn_examples.json`.
- Next Experiment: implement and measure one of the two India-gap fix candidates named
  above (§19, P1).

#### EXP-006: Leaderboard submission readiness audit (static trace)
- Status: [x] COMPLETED (as a static-trace audit — explicitly NOT a full-scale run)
- Date: 2026-09-25, git HEAD `352a403`.
- Area: infra / `run_pipeline.py` `--mode test` wiring.
- Hypothesis: `--mode test` is wired correctly and would, if run, produce a
  submission-format-valid output at real test scale.
- Configuration: none executed — read-only code trace plus citation of existing test
  coverage (`test_test_run_writes_valid_submission_with_unseen_country`,
  `test_cli_reads_files_from_config_paths`).
- Dataset: none run; dataset presence confirmed by row count only (§5.1 numbers).
- Sample Size: N/A (no run).
- Code Version: git commit `352a403`.
- Metrics: N/A — this experiment produced a code-path verdict, not a numeric metric.
- Results: **READY FOR FULL TEST INFERENCE**, with the explicit caveat that the real
  1,732,544-row run has never been executed and runtime/RAM at that scale are
  estimated (order of "hours," extrapolated from the N=10,000 timing table), not
  measured.
- Interpretation: no blocker exists in the code; the only way to close the remaining
  uncertainty is to actually run `--mode test`.
- Decision: INVESTIGATE (the recommended next action is to run it, not implemented in
  this audit or in this document per its own scope constraints).
- Artifacts Saved: `experiments/reports/leaderboard_submission_readiness.md`.
- Next Experiment: EXP-00X (unassigned) — a real, monitored `--mode test` run. **Not yet
  run as of this document.**

#### EXP-007: SageMaker + S3 storage adaptability audit
- Status: [x] COMPLETED (audit)
- Date: 2026-09-25, git HEAD `352a403`.
- Area: infra (paths, environment, storage boundaries).
- Hypothesis: the pipeline can run unchanged on a SageMaker Linux instance given an
  S3-synced dataset, with GitHub/S3/local-disk boundaries kept clean.
- Metrics/Results: **READY WITH MINOR CHANGES** — zero P0 blockers; 2 P1 items (sync
  tooling must explicitly pick up 3 files living outside `experiments/`;
  `.gitignore` needed an `output/*.tsv` rule); 3 P2 recommendations (requirements.txt
  reconciliation, configurable artifact/output/experiment roots, forward-looking cache-
  key pattern note).
- Interpretation: no source code was required to change for correctness; only
  operational tooling and one `.gitignore` line.
- Decision: KEEP (verdict), implemented as EXP-008.
- Artifacts Saved: `experiments/reports/sagemaker_storage_adaptability_audit.md`.
- Next Experiment: implement the P1 items → EXP-008.

#### EXP-008: SageMaker storage implementation (s3_sync.py)
- Status: [x] COMPLETED
- Date: 2026-09-25, git HEAD `352a403` (implementation commit not yet made at audit
  time — files present as untracked per this session's `git status`).
- Area: infra — new opt-in module, zero ML algorithm changes.
- Hypothesis: an isolated, opt-in S3 sync layer can satisfy EXP-007's P1 items without
  touching any `src/` file governing pipeline behaviour.
- Metrics:
  | Metric | Value |
  |---|---|
  | New tests added | 33 |
  | Test suite before | 104 passed, 0 failed, 0 skipped |
  | Test suite after (py -3.12, CPU) | 137 passed, 0 failed, 1 skipped |
  | Test suite after (torch-gpu, CUDA) — re-confirmed this session | 138 passed, 0 failed, 0 skipped |
  | Forbidden files touched (`normalize/blocking/features/model/decide/evaluate/
    run_pipeline.py`) | 0 (confirmed via empty `git diff --stat`) |
- Results: `.gitignore` gained `output/*.tsv`; `code/src/s3_sync.py` (new, lazy-boto3-
  import, `BER_S3_ROOT` env var, non-destructive by default) and
  `code/tests/test_s3_sync.py` (new) added; `docs/architecture/pipeline.md` gained a
  short storage-model section.
- Interpretation: `run_pipeline.py` behaviour is provably unchanged (not imported by
  it); this closes both EXP-007 P1 items.
- Decision: KEEP.
- Artifacts Saved: `code/src/s3_sync.py`, `code/tests/test_s3_sync.py`,
  `experiments/reports/sagemaker_storage_implementation.md`.
- Next Experiment: none infra-side; ML-side next experiments are §19.

---

## 15. Experiment Reproducibility Checklist

- [x] Every pipeline stage is seeded (`config.SEED = 42`), used consistently by
      `subsample_train`, LightGBM (`LGB_PARAMS["seed"]`), and GroupKFold.
- [x] `run_pipeline.log_experiment()` records git commit hash + config snapshot + metrics
      per run (mechanism exists and is tested), though no full-scale row has ever been
      committed/preserved (it's gitignored by design — see §16).
- [x] Data-correctness audit independently reproduced the N=5,000 baseline numbers to 4
      decimal places from a second, separate run (§5, §14 EXP-005) — a genuine
      reproducibility check, not a restatement.
- [x] Caching is content-hash-keyed (SHA-1 of input + relevant `config.TUNABLES`) for
      normalisation/embedding/blocking caches — safe to reuse across machines/instances
      on identical inputs.
- [ ] `code/requirements.txt` has never actually been installed and tested end-to-end —
      every verified test/experiment run in this repo used the `torch-gpu` conda
      environment's actually-installed (older) versions, not the pinned file (§16).
- [ ] No experiment has been run twice on the exact same N with the same code to confirm
      byte-identical output (`plan.md`'s CP9 "reproducibility dry run" checkpoint is
      still unchecked).
- [ ] No experiment has been run on a genuinely separate machine (e.g. an actual
      SageMaker instance, as opposed to the local Windows/torch-gpu environment every
      audit in this repo has used so far) to confirm cross-machine reproducibility.

---

## 16. Artifact Storage Checklist

- [x] `dataset/` gitignored, documented, regenerable from S3 (`dataset/README.md`).
- [x] `code/artifacts/{cache,embeddings,models,indexes}/` gitignored with `.gitkeep`
      scaffolding; `indexes/` confirmed genuinely unused (no ANN index ever built).
- [x] `output/*.tsv` now gitignored (added in EXP-008) — closes the latent GitHub/S3
      boundary gap flagged by EXP-007.
- [x] `experiments/{logs,results}/` gitignored; `experiments/{configs,reports}/`
      intentionally tracked (small, human-curated).
- [x] `code/experiments.csv` gitignored.
- [ ] **Known unreconciled gap**: `code/requirements.txt` pins versions implying Python
      ≥3.12 (`numpy==2.5.2`, `scipy==1.18.1`, `torch==2.13.0` — the latter not a real
      released PyTorch version at time of writing) while every actually-executed
      test/experiment in this repo ran on `torch-gpu`: Python 3.10.20, torch
      2.5.1+cu121, numpy 2.2.6, scipy 1.15.3, pandas 2.3.3, scikit-learn 1.7.2, pyarrow
      23.0.1, sentence-transformers 5.4.1, transformers 4.49.0 (lightgbm 4.7.0 and
      pytest 9.1.1 are exact matches). Classification: no dependency is *definitely*
      incompatible (the 138-test suite passes on the actual versions; no code was found
      calling a pinned-but-not-installed-version-only API), but this is a genuine,
      three-times-flagged (baseline report, data-correctness audit, SageMaker audit)
      documentation/reproducibility gap, explicitly deferred every time, still unfixed.
- [ ] No S3 upload has actually been performed with `s3_sync.py` in a live AWS
      environment (all 33 tests mock `_get_client`/inject a fake boto3 module — none
      constructs a real client or makes a network call, by design, but this means the
      *real* AWS path is untested end-to-end).

---

## 17. SageMaker + S3 Workflow

Per `docs/architecture/pipeline.md`'s "Storage" section and the two SageMaker reports:

```
GitHub (code/docs/configs — source of truth)
    │ git clone
    ▼
SageMaker instance (or local machine) — ephemeral local disk
    1. aws s3 sync s3://tensortrio/.../dataset/ dataset/   (BEFORE `import config`)
    2. pip install -r code/requirements.txt (or the verified torch-gpu stack — see §16)
    3. cd code && python -m src.run_pipeline --mode test
         reads  dataset/{train,test}/*.tsv
         writes code/artifacts/**  (caches — recomputable)
         writes code/experiments.csv
         writes output/*.tsv       (the two deliverables)
    4. python3 utils/validate_submission.py ... → must print PASS
    5. Explicit, opt-in sync UP (never mid-run):
         s3_sync.upload_artifacts()    -> s3://.../artifacts/
         s3_sync.upload_experiments()  -> s3://.../experiments/ (incl. the 3 files that
                                           live outside experiments/ on disk)
         s3_sync.upload_submissions()  -> s3://.../submissions/<tag>/
```

Key properties, all VERIFIED by code trace (not run against real AWS): no pandas/boto3
call anywhere reads S3 directly; every sync is an explicit function call the team
chooses to make; `run_pipeline.py` never imports `s3_sync` and its behaviour is
unaffected by this layer's existence. `BER_S3_ROOT` env var required, no hardcoded
bucket default — an absent `BER_S3_ROOT` raises immediately rather than silently
defaulting to the wrong bucket.

---

## 18. Experiment Dependency Graph

```
DATA
  │  (raw TSVs — verified structurally sound, §5.1, §12)
  ▼
NORMALIZATION
  │  (deterministic, null-safe, ID-preserving, country-agnostic — VERIFIED)
  ▼
EMBEDDINGS
  │  (Apache-2.0, 118M params, single-pass recall 0.8825 at N=10,000 — best individual
  │   pass, but not independently ablated — §7)
  ▼
BLOCKING  ◄── THE MEASURED BOTTLENECK (§8)
  │  Sets the RECALL CEILING for everything downstream. Evidence: 57–88% of all missed
  │  true matches at every tested N are blocking false negatives, not classifier or
  │  threshold misses (§5.4). A true match blocking never surfaces can NEVER be
  │  recovered by any later stage — this is a hard mathematical ceiling, not a tuning
  │  problem. CONCRETE EXAMPLE FROM THIS REPO: do not spend effort tuning LightGBM
  │  hyperparameters or the decision threshold to fix the India recall gap — 12/15
  │  sampled India misses never reached the feature/model stage at all (§8.1); the fix
  │  has to happen in normalize.py/blocking.py, not model.py/decide.py.
  ▼
FEATURES
  │  (41/58 columns "clearly informative"; strong separation on pairs that DO survive
  │   blocking — §9)
  ▼
MODEL
  │  (LightGBM, GroupKFold-by-S1, no leakage confirmed at data level — §10)
  ▼
THRESHOLD
  │  (tau tuned on OOF, unstable across small-N samples but consistently >0.5 — §11)
  ▼
DECISION (one-to-one assignment)
  │  (removed 0 true matches at every N tested — currently "free" — §5.5, §11)
  ▼
ERROR ANALYSIS
  │  (error-budget classification + India root-cause investigation — §12)
  ▼
NEW EXPERIMENT
  │  (§19 priorities — currently the largest unexplored area is blocking, per the
  │   dependency-graph logic above: fixing blocking recall raises the ceiling every
  │   downstream stage operates under, while fixing the model/threshold only
  │   redistributes error within an already-capped ceiling)
  ▼
FINAL PIPELINE
  │  (never run at full scale — §3, §6)
  ▼
SUBMISSION
     (never produced — §3, §6)
```

---

## 19. Experiment Priorities

Engineering priority list based on **evidenced weakness**, not a "best model" ranking.
Any model candidate mentioned below must satisfy problem_statement.md's Constraint 5
(MIT/Apache-2.0, ≤8B parameters) if one is ever proposed — none of the items below
propose a new model.

### P0 — blocks a valid, credible submission
1. **Run a real, monitored `--mode test` full-scale execution.** Nothing downstream of
   this (validator run, leaderboard upload, zip packaging) can happen without it. Static
   trace says READY (EXP-006); actual scale (~173–220× the largest tested N) has never
   been attempted. Risk: unverified RAM/runtime behaviour at scale (estimated "hours,"
   not measured) could surface a real constraint a code trace cannot catch. Dependency:
   none — this is the critical-path item.
2. **Fill in `docs/methodology/Documentation_template.md`.** Currently a placeholder.
   Officially required in the final zip (§2.2); cannot be done meaningfully until the
   real pipeline numbers (from item 1) exist to report.

### P1 — highest evidenced ML-quality weakness, addressable now
1. **Blocking recall decline with N.** Directly evidenced (§5.2, §8.1): pair recall
   0.9906→0.9769 as N grows 500→10,000. Since blocking sets a hard ceiling (§18), any
   gain here raises the ceiling for every downstream stage. Candidate experiment (named,
   not implemented, by the baseline report itself): re-run the N-ladder with
   `K_TFIDF_NAME`/`K_EMBEDDING` raised, to distinguish a fixed-k artifact from a genuine
   similarity-ranking limit. Computational cost: low (just re-running blocking, not the
   full pipeline). Risk: low (config-only change).
2. **India blocking-recall gap, now root-caused (§8.1).** Two named, independent
   mechanisms (transliteration divergence, address truncation) — either is now a
   concrete, scoped engineering target rather than a vague "India is worse" finding.
   Candidate experiments: (a) a phonetic/Soundex-style block key computed from the
   transliterated name form; (b) loosening the digit-pass exact-key join to tolerate a
   single-digit edit. Potential F0.5 impact: meaningful given India is ~40% of train S1
   volume and the gap is consistent at every N. Computational cost: moderate (new
   blocking pass or key-matching logic). Risk: moderate — a fuzzier key could raise
   candidate volume and hurt precision if not bounded carefully; must be measured against
   the reduction-ratio/precision tradeoff, not assumed to be free like one-to-one turned
   out to be (§5.5).
3. **`code/requirements.txt` / execution-environment reconciliation.** Three separate
   audits have flagged this and deferred it. Low computational cost, low risk (pure
   documentation/pinning fix), but blocks confident reproducibility claims in the final
   submission package — a reviewer trying to reproduce results from the pinned
   `requirements.txt` alone is in an untested configuration.

### P2 — valuable but not currently evidenced as urgent
1. Name-vs-address decomposition of the remaining (non-India-specific) blocking false
   negatives (baseline report §17 point 2) — would clarify whether further blocking
   investment belongs in name passes or address passes generally, not just for India.
2. LightGBM hyperparameter tuning / 3-seed averaging (CLAUDE.md §6.4) — the model stage
   is not currently evidenced as the bottleneck (§18), so this has lower expected F0.5
   impact per unit effort than the blocking items above, though it is cheap to run.
3. LOCO (leave-one-country-out) validation — never run despite the CLI flag existing.
   Directly relevant to France generalisation (the one country with zero train
   representation), and CLAUDE.md explicitly calls this out as the mechanism for
   "estimating the damage" of no French training data. Should be run before trusting any
   France-specific claim beyond the synthetic-test-level checks already done.
4. Feature ablation on the "highly overlapping" (<0.3 SMD) columns — low expected impact
   (LightGBM already handles uninformative features gracefully by simply not splitting
   on them), but would formally confirm they're safe to drop for a leaner feature set.
5. A real (non-mocked) end-to-end S3 upload test of `s3_sync.py` against actual AWS
   credentials — operational hardening, not ML-quality-relevant.

---

## 20. Master Experiment Table

Only real, evidenced experiments/audits are listed. No future/unrun item appears here
(they are tracked in §19 instead).

| ID | Area | Experiment | Status | Baseline | Result | Decision | Next |
|---|---|---|---|---|---|---|---|
| EXP-001 | Full pipeline | Baseline validation, N=500 | COMPLETED | n/a (first data point) | pair recall 0.9906, held-out F0.5 0.9929 | KEEP | EXP-002 |
| EXP-002 | Full pipeline | Baseline validation, N=2,000 | COMPLETED | EXP-001 | pair recall 0.9864, held-out F0.5 0.9938 | KEEP | EXP-003 |
| EXP-003 | Full pipeline | Baseline validation, N=5,000 | COMPLETED | EXP-002 | pair recall 0.9812, held-out F0.5 0.9900 | KEEP | EXP-004 |
| EXP-004 | Full pipeline | Baseline validation, N=10,000 | COMPLETED | EXP-003 | pair recall 0.9769, held-out F0.5 0.9871, India gap 4pts | KEEP (current reference baseline) | EXP-005, §19 P1 |
| EXP-005 | Data integrity + blocking | Data correctness & lineage audit (incl. India root cause) | COMPLETED | EXP-004's flagged India gap | root cause CONFIRMED (transliteration + address truncation, joint); no data-corruption bug found | KEEP | §19 P1.2 |
| EXP-006 | Infra (`--mode test`) | Leaderboard submission readiness audit | COMPLETED (static trace) | n/a | READY FOR FULL TEST INFERENCE, with caveat: never actually run | INVESTIGATE (run it) | §19 P0.1 |
| EXP-007 | Infra (SageMaker/S3) | Storage adaptability audit | COMPLETED (audit) | n/a | READY WITH MINOR CHANGES, 0 P0, 2 P1, 3 P2 | KEEP (implement P1s) | EXP-008 |
| EXP-008 | Infra (SageMaker/S3) | s3_sync.py implementation | COMPLETED | EXP-007's P1 items | 33 new tests, 0 forbidden-file changes, `.gitignore` fix applied | KEEP | none pending, infra-side |

---

## 21. Master Compliance + Artifact Matrix

| Item | Officially Required? | Pipeline Required? | Currently Produced? | Correct? | Persist? | Status |
|---|---|---|---|---|---|---|
| `output/matching_results.tsv` | **Yes** (only leaderboard-scored file) | Yes | **No** | N/A — not produced | Yes, must be in final zip | `[ ]` NOT TESTED |
| `output/candidate_pairs.tsv` | **Yes** (final zip, not leaderboard-scored) | Yes | **No** | N/A — not produced | Yes, must be in final zip | `[ ]` NOT TESTED |
| Final submission zip (`<team>_submission.zip`) | **Yes** | No (packaging step) | No | N/A | N/A | `[ ]` NOT TESTED |
| `train_source1/2/3.tsv`, `train_ground_truth.tsv` | Provided by organizers, not "produced" | Yes (training input) | Yes, locally | Yes — full-file row counts/integrity audited (§12) | No (gitignored, per §3) | `[x]` COMPLETED (as input verification) |
| `test_source1/2/3.tsv` | Provided by organizers | Yes (inference input) | Yes, locally | Yes — full-file row counts/integrity audited | No | `[x]` COMPLETED (as input verification) |
| `train_ground_truth.tsv` | Provided (train only, not test) | Yes (training/tuning) | Yes, locally | Yes, audited | No | `[x]` COMPLETED |
| Sentence-embedding artefacts | Not officially named | Yes (blocking pass 2) | Only at sample scale | Yes, at sample scale | No (gitignored cache) | `[~]` PARTIALLY TESTED |
| Embedding metadata (`MODELS.md`) | Not officially named, but supports Constraint 5 | Yes (compliance record) | Yes | Yes, verified against HF model API | Yes (tracked in git) | `[x]` COMPLETED |
| Blocking configuration (`config.py` k-values, `BLOCK_WITHIN_COUNTRY`) | Not officially named | Yes | Yes | Yes, measured at N≤10,000 | Yes (tracked in git) | `[x]` COMPLETED (at sample scale) |
| Candidate diagnostics (recall, reduction ratio) | Not officially named, but candidate_pairs.tsv is | Yes | Yes, at sample scale (§5, §8) | Yes | Yes (reports, tracked) | `[~]` PARTIALLY TESTED (sample-scale only) |
| Feature configuration | Not officially named | Yes | Yes | Yes, 58 columns diagnosed at N≤10,000 | Yes (code, tracked) | `[x]` COMPLETED (at sample scale) |
| Trained model artefact | Not officially named (code that produces it is required) | Yes for reproducing a specific submission | **No** — no persisted model exists | N/A | No (gitignored) | `[ ]` NOT TESTED |
| Model metadata (`MODELS.md` LightGBM row) | Supports Constraint 5 | Yes | Yes | Yes | Yes | `[x]` COMPLETED |
| Threshold (tau) | Not officially named | Yes | Yes, at sample scale, unstable across N | Partially — plausible but not full-scale-verified | No (in-memory only, not persisted as an artefact today) | `[~]` PARTIALLY TESTED |
| Experiment configuration (`config.TUNABLES` snapshot) | Not officially named | Yes (CLAUDE.md §4) | Mechanism exists, tested | Yes | No (gitignored `experiments.csv`) | `[~]` PARTIALLY TESTED (mechanism verified, no full-scale row exists) |
| Evaluation metrics (macro F0.5 etc.) | Yes, this is the scoring metric | Yes | Yes, at sample scale only | Yes | Yes (reports) | `[~]` PARTIALLY TESTED (train-side only, no leaderboard score) |
| Error analysis dumps | Not officially required | Yes (CLAUDE.md §6) | Only transiently, at sample scale | Yes, when produced | No (gitignored, none currently on disk) | `[~]` PARTIALLY TESTED |
| Runtime/hardware metadata | Not officially required | No | Yes, recorded per report (GPU, VRAM, timings) | Yes, at sample scale | Yes (in reports) | `[~]` PARTIALLY TESTED (sample scale only; full-scale is an estimate, not measured) |
| `Documentation_template.md` (filled) | **Yes** | No | **No — placeholder only** | N/A | Yes, required at zip root | `[ ]` NOT TESTED |
| `code/README.md`, `requirements.txt` | **Yes** (in `code/business_entity_resolution/`) | No | Yes, both exist | Partially — README accurate; requirements.txt has the known version-pin gap (§16) | Yes | `[~]` PARTIALLY TESTED |

---

## 22. What We Know

- Pipeline mechanics are correct end-to-end up to N=10,000 train S1 entities, on real
  executed runs, cross-checked by two independent audits (baseline + data-correctness).
- Blocking is the measured, dominant bottleneck (57–88% of misses at every N), and this
  is a mathematical ceiling — no downstream stage can recover a blocking-missed pair.
- Blocking recall degrades monotonically as candidate-pool size grows (0.9906 at N=500
  → 0.9769 at N=10,000) — a real scaling trend, not noise.
- The India-vs-US blocking gap (~4 points) has a confirmed, dual, evidence-backed root
  cause: ISCII transliteration divergence + address truncation/renumbering, acting
  jointly on the same pairs.
- One-to-one assignment removes zero true matches at every N tested — currently a "free"
  precision gain.
- Precision stays near-ceiling (0.997–1.000) at every N; recall is the softer, blocking-
  gated number.
- No data-corruption bug exists anywhere in the traced pipeline (normalisation through
  submission-writing) — confirmed by a dedicated, real-run audit.
- `--mode test`'s code path is statically correct by trace and by existing synthetic-data
  test coverage (including a France-unseen-country scenario).
- The pipeline can run unchanged on a SageMaker Linux instance given a synced dataset —
  no code changes needed, confirmed by trace (not by an actual Linux run).
- Two models are used, both compliant with Constraint 5 (Apache-2.0 118M-param embedding
  model; MIT-licensed LightGBM trained from scratch).
- No external data lookup exists anywhere in the code (confirmed by repeated, independent
  greps across every audit).

## 23. What We Don't Know Yet

- Whether the pipeline completes, and how long it takes, at real test scale
  (1,732,544 S1 × ~9.97M S2/S3) — only estimated (order of hours), never measured.
- What the actual private/public leaderboard F0.5 will be — no test-set score exists or
  can exist locally (no test ground truth was ever provided, by design).
- Whether blocking recall/precision at full scale resembles the N≤10,000 trend
  (continued decline) or plateaus — untested beyond N=10,000.
- Peak RAM and disk usage during full-scale feature computation — estimated (~16 GB if
  fully materialised) but the actual chunked, never-persisted-to-disk behaviour has
  never been measured at scale.
- Whether France-specific blocking/matching performs acceptably — no labelled France
  data exists anywhere (train has none); only structural/functional checks (no country-
  literal branches, synthetic end-to-end test) exist, not a quality measurement.
- Whether LOCO validation would reveal a larger France-generalisation risk than the
  India-vs-US gap already suggests — never run.
- Whether raising blocking k-values recovers the N-driven recall decline, or whether it
  is a genuine similarity-ranking ceiling — named as a next step by the baseline report,
  never executed.
- Whether `code/requirements.txt`'s pinned versions (numpy 2.5.2, torch 2.13.0, etc.)
  actually install and run correctly on a fresh Python ≥3.12 environment — never tested;
  only the older `torch-gpu` versions have ever been exercised.
- Whether a full training run (all 2,206,821 train S1, not a ≤10,000 sample) changes any
  of the qualitative findings above (e.g. does the India gap widen/narrow, does tau
  stabilise with more OOF data).

## 24. What We Should Try Next

In dependency order, consistent with §18's graph and §19's priorities:
1. Run the real `--mode test` full-scale inference (§19 P0.1) — this is the single
   highest-value next action; every other open question about full-scale behaviour
   depends on this actually happening.
2. Immediately after, run `utils/validate_submission.py` and confirm `PASS` before
   treating anything as submission-ready (CLAUDE.md §2 rule 6 — never skip this).
3. In parallel (does not block item 1): re-run the N-ladder with raised
   `K_TFIDF_NAME`/`K_EMBEDDING` to test whether the recall decline with N is a fixed-k
   artifact (§19 P1.1) — cheap, blocking-only re-run.
4. Implement and measure one of the two named India-gap fix candidates (§19 P1.2),
   validated on the same N=5,000/10,000 samples already established as reference points,
   so the before/after comparison is apples-to-apples with existing evidence.
5. Run LOCO validation (train-on-one-country, score-on-the-other) using the existing
   `--loco` CLI flag, to get the first real evidence-based signal on France
   generalisation risk beyond the synthetic unit test.
6. Fill in `docs/methodology/Documentation_template.md` once real full-scale numbers
   exist from item 1 — writing it before then would mean documenting numbers that don't
   yet exist.
7. Reconcile `code/requirements.txt` with either the actually-verified `torch-gpu` stack
   or a freshly-tested Python≥3.12 stack (§19 P1.3) — low cost, closes a three-times-
   flagged gap, should happen before the final zip is assembled.

## 25. Final Submission Readiness Checklist

| Item | Status | Evidence |
|---|---|---|
| Data verified | `[x]` COMPLETED | Full-file raw-integrity audit, all 7 files, `data_correctness_audit.md` §1 |
| Data paths verified | `[x]` COMPLETED | `config.py` path resolution traced and confirmed portable (Linux-safe, no hardcoded paths), `sagemaker_storage_adaptability_audit.md` §5/§11 |
| Normalization verified | `[x]` COMPLETED | Deterministic, null-safe, ID-preserving, country-agnostic, verified on real records including Devanagari/Bengali/Telugu/French examples, `data_correctness_audit.md` §2 |
| Embeddings verified | `[~]` PARTIALLY TESTED | Model licence/size verified (`MODELS.md`); embedding pass recall measured at N≤10,000 only; full-scale never run |
| Blocking verified | `[~]` PARTIALLY TESTED | Extensively measured at N≤10,000 (§5, §8); India root cause confirmed; full-scale (1.73M S1) never run |
| Features verified | `[~]` PARTIALLY TESTED | 58 columns diagnosed, 0 inf/all-NaN, at N≤10,000; full-scale feature matrix never materialised |
| Model verified | `[~]` PARTIALLY TESTED | GroupKFold no-leak confirmed at data level, OOF separation clean, at N≤10,000; no full-train model exists |
| Threshold verified | `[~]` PARTIALLY TESTED | Tau selection mechanism works and lands >0.5 at every N, but is unstable across small-N samples; never tuned on full OOF |
| Decision logic verified | `[x]` COMPLETED | One-to-one assignment removes 0 true matches at every N tested, confirmed twice independently including a real-S1 walkthrough |
| Evaluation verified | `[~]` PARTIALLY TESTED | Macro F0.5 scorer unit-tested against the exact official worked example + singleton edge cases; only ever run on train-side held-out data, never a real leaderboard score |
| Full test inference completed | `[ ]` NOT TESTED | No `--mode test` run at real scale has ever occurred; `output/` contains no generated TSVs (`leaderboard_submission_readiness.md` §5, re-confirmed this session) |
| `matching_results.tsv` generated | `[ ]` NOT TESTED | Does not exist |
| `candidate_pairs.tsv` generated | `[ ]` NOT TESTED | Does not exist |
| Official validator PASS | `[ ]` NOT TESTED | Nothing to validate yet |
| Submission package assembled | `[ ]` NOT TESTED | No zip has been built; `plan.md` CP12 unchecked |
| Documentation complete | `[ ]` NOT TESTED | `docs/methodology/Documentation_template.md` is still placeholder text |
| Final submission archived in S3 | `[ ]` NOT TESTED | No submission exists to archive; `plan.md` CP12's S3 upload step unchecked |

**Overall verdict: the pipeline is mechanically validated at small scale and believed
ready by static trace for full-scale execution, but zero real submission artefacts
exist yet.** The critical path to a first leaderboard submission is §24 items 1–2 (run
`--mode test`, then validate) — everything else in this document is either already done
at sample scale or is a quality-improvement item that should follow, not precede, a
first real submission per `docs/planning/plan.md`'s own checkpoint ordering (CP2's
"first submission" is explicitly a format/calibration check, not a wait-for-perfection
gate).
