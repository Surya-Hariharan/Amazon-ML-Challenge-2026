# CLAUDE.md — Amazon ML Challenge 2026: Business Entity Resolution

This file is read by Claude Code at the start of every session. It defines what we are
building, the rules we must never break, how the code is organised, and how to work.
Read `plan.md` next — it holds the current checkpoint and what to do now.

---

## 1. The task in one paragraph

Three sources of business records (`S1-`, `S2-`, `S3-` ID prefixes). Source 1 is
**deduplicated**. For every S1 record, output the list of S2/S3 records that refer to the
same real-world business (zero, one or many). Scored by **macro-averaged F0.5 per S1
entity**, singletons included: an S1 with no true matches scores 1.0 for an empty list and
0.0 for any prediction. Precision is weighted 2× over recall. False merges are the enemy.

Columns in every source file: `entity_id, business_name, business_address, country`.
Train countries: US, India. **Test adds France (unseen in training).**

---

## 2. Hard rules — never break these (disqualification or rejection)

1. **No external data lookup.** No entity-resolution APIs/services (including AWS Entity
   Resolution), no business registries, no geocoding APIs (including Amazon Location
   Service, Google Maps, Nominatim), no web scraping, no internet data augmentation.
   Only the provided train/test files. Hand-written normalisation dictionaries (e.g.
   "rd → road", "sarl" as a legal suffix) are code, not data lookup, and are allowed.
2. **Model licence and size:** any ML model in the final pipeline must be **MIT or
   Apache-2.0 licensed and ≤ 8B parameters**. Before adding any pretrained model, record
   its licence and parameter count in `MODELS.md`. No hosted/proprietary LLM calls.
   Known-OK examples (verify the model card before use):
   `sentence-transformers/all-MiniLM-L6-v2` (Apache-2.0),
   `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (Apache-2.0),
   `intfloat/multilingual-e5-small` (MIT). LightGBM (MIT), rapidfuzz (MIT),
   scikit-learn (BSD — library, not a model; fine).
3. **Always read/write TSV with `sep="\t"`.** Read IDs as strings:
   `pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)`.
4. **Country is an open set.** Never hard-code `{US, India}`, never filter to known
   countries, never one-hot country. France must flow through every stage. Features must
   be country-agnostic (a `same_country` flag is fine).
5. **Output format** (both files in `output/`):
   - `matching_results.tsv`: header `source1_entity_id\tmatched_entity_ids`
   - `candidate_pairs.tsv`: header `source1_entity_id\tcandidate_entity_ids`
   - Exactly one row per test S1 ID, no duplicate rows, no duplicate IDs in a list,
     comma-separated with no spaces or quotes, empty string for no matches,
     only S2-/S3- IDs that exist in the test files.
   - Every matched ID must also be in that S1's candidate list.
   - `candidate_pairs.tsv` = the exact set the final model scored, not an earlier pass.
6. **Validate before every submission:**
   ```bash
   python3 utils/validate_submission.py \
     --matching output/matching_results.tsv \
     --candidate output/candidate_pairs.tsv \
     --test-dir dataset/test
   ```
   Must print `PASS`. Never hand the user a file that has not passed.
7. **Submission budget: 5 per day, 3 days.** Never suggest a submission without a local
   validation score that justifies it (see `plan.md` submission log).
8. **Every function gets a docstring.** Commented source code is a required artefact.

---

## 3. Data location — S3 is the handoff point, not the working copy

The dataset was uploaded (as a zip) by the team to **`s3://tensortrio`** (region
`ap-south-1`). This bucket is the *only* thing shared between your laptop and all three
SageMaker notebook instances — each instance is a separate machine with its own empty
filesystem and cannot see your laptop's files or another instance's disk.

Flow: laptop → S3 (already done) → **each SageMaker instance downloads its own local
copy** → all local work happens on that copy from then on.

- List the bucket first to find the exact key — don't assume a path:
  `aws s3 ls s3://tensortrio/ --recursive --region ap-south-1`
- Download and unzip into a local `dataset/` folder on the instance. This folder is
  **gitignored and never committed** — it is not part of the code deliverable, and must
  not end up inside the final submission zip either (see §4 layout).
- Every teammate's instance repeats this download independently — it's a one-time setup
  step per instance, not per run.
- Verify after unzip: `dataset/train/{train_source1,train_source2,train_source3,
  train_ground_truth}.tsv` and `dataset/test/{test_source1,test_source2,
  test_source3}.tsv` exist and load correctly with `sep="\t"`.
- Outputs go back up to S3 too, but to a **different prefix** (e.g.
  `s3://tensortrio/submissions/<tag>/...`), never overwriting the raw dataset objects,
  and only after `validate_submission.py` prints PASS locally.

---

## 4. Repository layout

This section describes the **final submission zip** layout (`<team>_submission.zip`).
The working repo you are in now has a few extra top-level directories the zip does not
need — `docs/`, `resources/`, `experiments/`, `dataset/`, `utils/`, `scratch/` — see the
note after the tree for how zip assembly maps the working repo onto this layout.

```
<team>_submission.zip/
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── code/business_entity_resolution/
│   ├── src/
│   │   ├── config.py          # all paths, seeds, thresholds, k values (single source of truth)
│   │   ├── io_utils.py        # TSV load/save, ID-list formatting, submission writer
│   │   ├── normalize.py       # name/address normalisation, token + number extraction
│   │   ├── blocking.py        # candidate generation (multi-pass union)
│   │   ├── features.py        # pairwise features for (S1, candidate) pairs
│   │   ├── model.py           # training, OOF predictions, model save/load
│   │   ├── decide.py          # thresholding + one-to-one assignment → final lists
│   │   ├── evaluate.py        # macro F0.5, blocking recall, reduction ratio
│   │   └── run_pipeline.py    # CLI entry point: data → blocking → matching → output
│   ├── README.md              # exact reproduction steps
│   ├── requirements.txt       # pinned versions
│   └── MODELS.md              # licence + param count of every pretrained model used
└── Documentation_template.md  # filled-in methodology
```

**Where each zip entry comes from in this repo (as of the 2026-09-25 reorganisation):**
`output/` and `code/business_entity_resolution/` map 1:1 onto the same paths at repo
root. `Documentation_template.md` at the **zip root** is copied from this repo's working
copy at `docs/methodology/Documentation_template.md` — that file no longer lives at the
repo root; the packaging step (`plan.md` CP12) copies it to the zip root under its
original bare filename. The organizer-provided raw materials (`dataset/`, `utils/`) are
NOT nested inside a `student_resource/` folder in this repo — they are separate
top-level directories (`dataset/`, `utils/`) alongside `code/`; only `dataset/` is
excluded from the zip (it is regenerated by every teammate from
`resources/student_resource.zip` or S3, per §3, and is gitignored). `utils/` (needed to
run `validate_submission.py`) is not part of the required zip structure either — the
organizers already have their own copy.

Artefacts that are not code (cached embeddings, trained models, OOF preds) go in
`artifacts/` and are gitignored. `dataset/` (downloaded from S3 per §3) is also
gitignored and excluded from the submission zip.

---

## 5. Commands

```bash
# local validation run (train split into train/valid by S1 ID, scored with our F0.5)
python -m src.run_pipeline --mode valid

# full run: train on all training data, predict test, write output/
python -m src.run_pipeline --mode test

# individual stages while iterating
python -m src.run_pipeline --mode valid --stage blocking   # prints recall + reduction ratio
python -m src.run_pipeline --mode valid --stage features
python -m src.run_pipeline --mode valid --stage model
```

Every run prints and appends to `experiments.csv`: timestamp, git commit hash, config
diff, blocking recall, mean candidates per S1, validation macro F0.5, precision, recall,
F0.5 on singletons vs non-singletons, and per-country F0.5.

---

## 6. Pipeline architecture (target design)

### 6.1 Normalisation (`normalize.py`)
- Unicode NFKD + strip accents (needed for French: `é → e`), lowercase, `& → and`,
  collapse punctuation and whitespace.
- Legal-suffix canonicalisation to a separate field (don't just delete it):
  corp/corporation, inc/incorporated, llc, ltd/limited, pvt/private, co/company, plc,
  llp, and French forms sarl, sas, sasu, sa, eurl, sci, snc. Keep `name_core` (suffix
  removed) and `name_suffix`.
- Address abbreviations: rd/road, st/street, ave/av/avenue, blvd/bd/boulevard, ln/lane,
  nr/near, opp/opposite, and French rue, pl/place, chem/chemin, imp/impasse, all/allee.
- Landmark phrases ("near sbi atm", "opp", "behind", "next to", "pres de") → split into a
  `landmark` field so they don't pollute street-token similarity.
- Extract digit tokens generically (house numbers, 5-digit ZIP/CP, 6-digit PIN) without
  country-specific regexes.
- Acronym of `name_core` (e.g. "international business machines" → "ibm").

### 6.2 Blocking (`blocking.py`) — recall ceiling lives here
Union of several passes, each giving top-k S2/S3 neighbours per S1:
1. Char 3–4-gram TF-IDF on `name_core`, cosine top-k.
2. Dense embeddings of `name + address` with a small multilingual MIT/Apache model,
   cosine top-k (FAISS or sklearn brute force).
3. Shared rare name token (high IDF) blocks.
4. Shared postal-code/number token + shared first name token.
Default: block within the same country string, but measure on train whether cross-country
matches exist and make it a config flag. Tune k per pass to reach **≥ 98% pair recall on
validation** with the smallest mean candidate count. Always report recall and reduction
ratio.

### 6.3 Features (`features.py`)
Name: Jaro-Winkler, token_set_ratio, token_sort_ratio, partial_ratio (rapidfuzz),
TF-IDF cosine, embedding cosine, `name_core` exact match, suffix match/conflict,
acronym match either direction, token Jaccard, length ratio.
Address: token Jaccard, TF-IDF cosine, digit-token overlap, postal-code equal/conflict,
house-number equal/conflict, landmark overlap, missing-field flags.
Context (very strong for ER): rank of this candidate within the S1's list, score gap to
the S1's best candidate, reverse rank (rank of this S1 among all S1s that list this
candidate), number of candidates, source of candidate (S2 vs S3).
No country one-hots, no raw ID features.

### 6.4 Model (`model.py`)
LightGBM binary classifier on (S1, candidate) pairs. Cross-validation with
`GroupKFold` grouped by S1 ID — never let the same S1 appear in train and valid folds.
Keep out-of-fold predictions for threshold tuning. Optional later upgrade: fine-tuned
cross-encoder (MIT/Apache, ≤ 8B) as an extra feature, only if it beats LightGBM on CV.

### 6.5 Decision (`decide.py`)
- Because S1 is deduplicated, each S2/S3 record should belong to at most one S1. Confirm
  this on train ground truth, then enforce it: assign each candidate only to the S1 where
  its probability is highest.
- Keep pairs with probability ≥ τ. Tune τ on OOF predictions to maximise **macro F0.5
  including singletons**, not pair-level F1. Expect τ above 0.5.
- Optional second threshold for "is this S1 a singleton" using the S1's top score.

### 6.6 Generalising to France
No French training data exists. Estimate the damage with leave-one-country-out
validation (train on US → score on India, and the reverse). Prefer features that survive
that test. Language-agnostic normalisation matters more than model complexity here.

---

## 7. How to work

- Before starting a task, read the current checkpoint in `plan.md`. After finishing, tick
  it off and write the result numbers there.
- Small, testable steps. Run the validation pipeline after every meaningful change and
  compare against the last row of `experiments.csv`. A change that doesn't improve
  validation macro F0.5 gets reverted or put behind a config flag.
- Seed everything (`config.SEED`). Results must be reproducible from a clean checkout.
- Cache expensive work (embeddings, TF-IDF matrices) keyed by input hash in `artifacts/`.
- Commit after each checkpoint. Tag each leaderboard submission `sub-d{day}-{n}`.
- Pin every dependency added in `requirements.txt` immediately.
- If a decision trades precision for recall, state the expected effect on F0.5.
- Don't write long notebooks. Code lives in `src/`; throwaway analysis goes in
  `scratch/` (gitignored).
- Error analysis: after each model run, dump the 50 worst false positives and 50 worst
  false negatives from validation to `artifacts/errors_*.tsv` and look at them before
  inventing new features.

## 8. Suggested sub-agent split inside Claude Code

When a task is large, delegate along these lines so work runs in parallel without
touching the same files:
- **blocking agent** — owns `normalize.py`, `blocking.py`; target metric: blocking recall
  and candidates per S1.
- **matching agent** — owns `features.py`, `model.py`, `decide.py`; target metric:
  validation macro F0.5.
- **evaluation/packaging agent** — owns `evaluate.py`, `io_utils.py`, README,
  requirements, documentation and running the validator.
