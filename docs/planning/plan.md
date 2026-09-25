# plan.md — Checkpoint plan for Amazon ML Challenge 2026

Window: **25 Sep 2026 00:00 IST → 27 Sep 2026 23:59 IST**. 5 leaderboard submissions
per day (15 total). Final ranking uses the **private** leaderboard, so trust local
validation over small public-LB wiggles.

How to use this file (Claude Code and humans):
- Work top to bottom. Don't start a checkpoint until the previous gate is met.
- When a checkpoint is done, change `[ ]` to `[x]` and fill in the **Result** line.
- If a gate fails, write why under **Notes** and follow the fallback.

---

## Open conflicts / questions (resolve early, don't guess)

- **Methodology doc length conflict:** `../docs/challenge/guidelines.md`'s Key
  Instructions say the shared artefact for the top-100 stage is a "1-2-page document,"
  but `../docs/challenge/problem_statement.md`'s Submission Requirements §3 says
  `Documentation_template.md` has "no page limit — prioritise clarity and technical
  depth over brevity." **Ask via the Google Form** which applies. Until answered, write
  the full detailed doc but lead it with a 1-2 page executive summary (methodology,
  blocking recall, model, key numbers) so either reading is satisfied.

---

## Current status

- Current checkpoint: **CP0 (in progress — confirm remaining items below)**
- Environment: S3 dataset bucket set up at `s3://tensortrio` (ap-south-1) with team
  access, SageMaker notebook instances created for all 3 members, GitHub repo created
  and cloned into each instance. **Not yet confirmed:** dataset actually downloaded from
  S3 onto each instance's local disk, repo skeleton per CLAUDE.md §4, `evaluate.py` F0.5
  unit test, `io_utils.write_submission()`, `requirements.txt` installing cleanly.
  Do these before CP1 — CP2's first submission depends on them.
- Best validation macro F0.5: —
- Best public LB F0.5: —
- Submissions used: Day 1 0/5 · Day 2 0/5 · Day 3 0/5

---

## Day 0 — Thu 24 Sep (before data release)

### [ ] CP0 · Environment, data pull, and skeleton (time box: 3 h)
- [x] Compute set up: S3 bucket (`s3://tensortrio`, ap-south-1) with dataset zip and
  team access, SageMaker notebook instances for all 3 members, GitHub repo cloned into
  each instance.
- [ ] **Pull the dataset from S3 onto each instance's local disk** — the instance
  filesystem is separate from any team member's laptop, so this download is required
  per instance even though the data already exists on someone's laptop and in S3:
  ```bash
  aws s3 ls s3://tensortrio/ --recursive --region ap-south-1   # confirm the key first
  aws s3 cp s3://tensortrio/<path-to-zip> ./dataset.zip --region ap-south-1
  unzip dataset.zip -d dataset/
  ```
  Verify `dataset/train/*.tsv` and `dataset/test/*.tsv` load with `sep="\t"` and don't
  collapse into one column.
- [x] Create repo in the final zip layout (see CLAUDE.md §4); add `.gitignore` for
  `artifacts/`, `scratch/`, `dataset/` (the local S3 pull above must never be committed).
- [x] Stub every `src/` module with docstrings and empty functions.
- [x] Write `evaluate.py` macro F0.5 now and unit-test it on the PDF example
  (pred [47,193,812] vs truth [47,812] → 0.714) plus singleton cases (empty/empty → 1.0,
  any/empty → 0.0, empty/non-empty → 0.0).
- [x] Write `io_utils.write_submission()` that enforces every format rule itself.
- [ ] Confirm GPU quota / instance type is sufficient if using one.
- **Gate:** `pytest` passes for the F0.5 scorer; repo installs cleanly from
  `requirements.txt`; dataset present locally on every instance.
- Result: skeleton + evaluate.py + io_utils.py done; pytest 27/27 pass locally
  (Python 3.12, pandas 2.2.3). Still open: dataset pull on each instance, GPU check,
  clean `pip install -r requirements.txt` on SageMaker.

---

## Day 1 — Fri 25 Sep

**Staggering note:** this is written as a continuous 00:00→23:30 block, but 3 people
running it back to back with no sleep is a real execution risk. Split it: e.g. one
person covers CP1–CP2 overnight while the other two sleep, then all three are fresh for
CP3–CP5 during the day. Don't let CP1–CP5 timestamps force everyone awake at once.

### [ ] CP1 · Data audit (00:00–02:00)
Answer and record each of these; they drive later decisions:
- Row counts per source, train and test; country counts per source (confirm France
  appears only in test).
- % of S1 that are singletons in train → this is the **all-empty baseline score**.
- Distribution of matches per S1; share from S2 vs S3.
- Does any S2/S3 ID appear under two different S1s? (Decides one-to-one assignment.)
- Are there cross-country matches? (Decides country blocking.)
- Missing-value rates for name/address; typical noise examples per country.
- **Gate:** audit numbers written below.
- Result:

### [ ] CP2 · Validation harness + first submission (02:00–05:00)
- Split train by S1 ID into 80/20 (stratify on singleton vs not); also build the
  leave-one-country-out splits.
- Rule baseline: normalised exact `name_core` match within same country, plus
  rapidfuzz token_set_ratio ≥ 95 with address overlap.
- Write both output files, run the validator.
- **Submission #1 (D1):** rule baseline. Purpose: confirm format + calibrate local vs
  public score.
- **Gate:** validator PASS, status SCORED; local vs public gap recorded.
- Result:

### [ ] CP3 · Blocking v1 (05:00–10:00)
- Implement passes 1–4 from CLAUDE.md §6.2; union; dedupe.
- Grid-search k per pass. Report pair recall, S1-level full-recall %, mean candidates/S1.
- **Gate:** pair recall ≥ 98% on validation with mean candidates/S1 ≤ 50.
- Fallback: if recall stalls, inspect the missed pairs and add a targeted pass.
- Result:

### [ ] CP4 · Features + LightGBM v1 (10:00–18:00)
- Feature set from CLAUDE.md §6.3 (skip context features for now).
- GroupKFold(5) by S1; OOF predictions.
- `decide.py`: threshold sweep on OOF for macro F0.5; one-to-one assignment if CP1
  confirmed it.
- **Gate:** validation F0.5 clearly above CP2 rule baseline and above the all-empty
  baseline.
- **Submission #2 (D1):** LightGBM v1.
- Result:

### [ ] CP5 · Context features (18:00–23:30)
- Add rank, gap-to-best, reverse rank, candidate count.
- **Submission #3 (D1)** only if validation improves ≥ 0.005.
- Hold submissions #4–5 unless there's a real validated gain. Unused submissions don't
  roll over, so use one on the best-so-far variant late in the day if spare.
- Result:

---

## Day 2 — Sat 26 Sep

### [ ] CP6 · Error analysis round (00:00–04:00)
- Read the 50 worst FPs and FNs. Group them into causes (suffix confusion, chains/branches
  of the same brand, landmark addresses, transliteration, typos).
- One targeted fix per top cause.
- Result:

### [ ] CP7 · France robustness (04:00–10:00)
- Leave-one-country-out scores for current model. Record the drop.
- Add French legal-suffix and street normalisation; accent stripping check.
- Drop or regularise features that collapse under the LOCO test.
- Consider multilingual embedding similarity as a feature (MIT/Apache model only; log in
  MODELS.md).
- **Gate:** LOCO drop no worse than before; main validation not worse.
- Result:

### [ ] CP8 · Model upgrades (10:00–20:00)
Try in order, keep only what beats CV:
1. LightGBM tuning (num_leaves, min_child_samples, feature fraction), 3-seed average.
2. Separate singleton classifier on S1-level aggregates (top score, gap, count).
3. Optional: fine-tuned small cross-encoder on hard pairs, used as an extra feature.
- **Submissions D2:** up to 4, each tied to a validated gain. Keep ≥ 1 in reserve.
- Result:

### [ ] CP9 · Reproducibility dry run (20:00–23:59)
- Fresh environment → `pip install -r requirements.txt` → `run_pipeline --mode test`.
- Outputs must be byte-identical to the last run (seeded).
- Result:

---

## Day 3 — Sun 27 Sep

### [ ] CP10 · Final improvements (00:00–14:00)
- Small, safe changes only. Threshold re-tune on full OOF.
- Ensemble of top 2–3 validated configs by averaging probabilities.
- Result:

### [ ] CP11 · Code freeze (14:00)
- Choose the final model by **local validation + LOCO**, not by the single best public
  LB number.
- Tag commit `final`.

### [ ] CP12 · Package (14:00–20:00)
- Fill `docs/methodology/Documentation_template.md` (moved here from the repo root on
  2026-09-25 — see `CLAUDE.md` §4): lead with a 1-2 page executive summary (see "Open
  conflicts" above), then full methodology, blocking strategy (with recall and
  reduction ratio numbers), model architecture, feature list, experiments table,
  conclusions. The blank official template, if needed for reference, can be
  re-extracted from `resources/student_resource.zip` (see `resources/README.md`).
- README: exact commands data → blocking → matching → output.
- Verify: validator PASS; every match ⊂ candidates; `candidate_pairs.tsv` is the set the
  model actually scored; MODELS.md lists licences.
- **Build the zip with explicit exclusions** — `dataset/`, `artifacts/`, `scratch/`,
  `docs/`, `resources/`, and `experiments/` must NOT be inside
  `<team_name>_submission.zip`, and `Documentation_template.md` must be copied to the
  **zip root** (it no longer lives there in the working repo):
  ```bash
  mkdir -p /tmp/<team_name>_submission
  cp -r output code/business_entity_resolution /tmp/<team_name>_submission/
  cp docs/methodology/Documentation_template.md /tmp/<team_name>_submission/Documentation_template.md
  rm -rf /tmp/<team_name>_submission/code/business_entity_resolution/artifacts \
         /tmp/<team_name>_submission/code/business_entity_resolution/scratch
  cd /tmp/<team_name>_submission && zip -r ../<team_name>_submission.zip . && cd -
  ```
  Adjust paths to match your actual tree; the point is confirm nothing outside the
  required structure (§4 of CLAUDE.md) sneaks in, and that `Documentation_template.md`
  sits directly at the zip root (not under `docs/methodology/`).
- Build `<team_name>_submission.zip`, unzip it into a temp folder, re-run from it.
- **Upload the final outputs and the zip to S3** as a backup, under a submissions
  prefix — never overwrite the raw dataset objects:
  ```bash
  aws s3 cp output/matching_results.tsv \
    s3://tensortrio/submissions/final/matching_results.tsv --region ap-south-1
  aws s3 cp output/candidate_pairs.tsv \
    s3://tensortrio/submissions/final/candidate_pairs.tsv --region ap-south-1
  aws s3 cp <team_name>_submission.zip \
    s3://tensortrio/submissions/final/ --region ap-south-1
  ```
- Result:

### [ ] CP13 · Final submissions (by 22:00, one-hour buffer)
- Upload final `matching_results.tsv` to the Portal.
- Submit the zip.
- Result:

---

## Submission log

| # | Day | Time (IST) | Commit/tag | What changed | Local F0.5 | Public F0.5 |
|---|-----|-----------|------------|--------------|-----------|-------------|
| 1 | D1 |  |  | rule baseline |  |  |

## Experiment log (summary — full log in experiments.csv)

| Date/time | Change | Block recall | Cands/S1 | Val F0.5 | Singleton F0.5 | LOCO F0.5 | Keep? |
|-----------|--------|--------------|----------|----------|----------------|-----------|-------|

## Notes / decisions

-
