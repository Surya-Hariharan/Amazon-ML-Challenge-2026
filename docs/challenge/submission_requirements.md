# ML Challenge 2026 — Submission Requirements

> Source: verbatim excerpt from the organizer-provided `student_resource/README.md`
> (preserved in full at `resources/student_resource.zip`). Reproduced here unedited; do
> not paraphrase away from it. See also [`problem_statement.md`](problem_statement.md)
> and [`guidelines.md`](guidelines.md).

### Output Format

Your solution produces **two** tab-separated files, both placed in the `output/` folder
of your final submission package (see *Final Submission Package* below):

1. **`matching_results.tsv`** — your final entity matches. **This is the only file
   scored on the leaderboard** — it is what you upload to the Portal during the
   challenge.
2. **`candidate_pairs.tsv`** — the candidate set your blocking / candidate-generation
   stage produced, before your final matching model narrowed it down.

#### matching_results.tsv

Your final entity matches:

| Column | Description |
| --- | --- |
| source1_entity_id | The `entity_id` of a Source 1 record |
| matched_entity_ids | Comma-separated list of matching `entity_id`s from Source 2 and/or Source 3 |

**Example** (columns separated by a single tab, ID lists separated by commas with no quoting):

```
source1_entity_id	matched_entity_ids
S1-00001	S2-00047,S2-00193,S3-00812
S1-00002	S3-00004
S1-00003	
```

**Important:**

- Every Source 1 entity in the test set must have exactly one row
- Leave `matched_entity_ids` empty for entities with no matches (singletons)
- No duplicate entity IDs within a single ID list
- ID lists must only contain Source 2 or Source 3 IDs that exist in the test set

#### candidate_pairs.tsv

The candidate set from your blocking stage — every Source 2 / Source 3 record you
considered a plausible match for each Source 1 entity, *before* your final matching
model narrowed it down. This is the **exact set of records you feed into your matching
model for inference** — the final candidate list *just before* the ML model scores
them, not the raw output of an early blocking pass you later filter further. If your
pipeline has several blocking/filtering stages, `candidate_pairs.tsv` is the *last* one:
whatever your model actually runs inference over. Every ID in `matching_results.tsv`
should therefore appear here.

It is **not scored on the leaderboard**; we use it to analyse blocking quality (recall
ceiling, reduction ratio) and to verify your pipeline.

| Column | Description |
| --- | --- |
| source1_entity_id | The `entity_id` of a Source 1 record |
| candidate_entity_ids | Comma-separated list of candidate `entity_id`s from Source 2 and/or Source 3 |

**Example:**

```
source1_entity_id	candidate_entity_ids
S1-00001	S2-00047,S2-00193,S3-00812,S3-00999
S1-00002	S3-00004
S1-00003	
```

Same rules as `matching_results.tsv`: one row per Source 1 entity, `candidate_entity_ids`
empty when blocking found no candidates, S2-/S3- IDs only, no duplicates within a list.
Your final matches should be a **subset** of your candidates (a matched ID that never
appeared as a candidate signals a pipeline bug — the validator warns about it).

**Validate before submitting:** a helper script `utils/validate_submission.py` (stdlib
only, no dependencies) checks both files against every rule above so you can catch a
rejection locally instead of spending a submission on it. Run it from the repository
root (this project's copy of the organizer's `student_resource/` working directory):

```bash
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

It prints `PASS` (exit 0) when the files are safe to submit, or a numbered list of
issues to fix (exit 1). It only reads your output files and the test source files; it
does not compute your score.

### Final Submission Package

In addition to your live leaderboard uploads, **every team submits a single zip
archive** with your code and outputs. We use it to reproduce your results, audit your
blocking, and check the fair-play and model-license rules — the top teams' packages are
reviewed in detail before the final rankings are confirmed.

Structure:

```
<team_name>_submission.zip
├── output/
│   ├── matching_results.tsv        # final matches (same file you upload to the leaderboard)
│   └── candidate_pairs.tsv         # your blocking candidate set
├── code/
│   └── business_entity_resolution/
│       ├── src/                    # all your source code
│       ├── README.md               # how to reproduce end-to-end (data → blocking → matching → output)
│       └── requirements.txt        # pinned dependencies / environment
└── Documentation_template.md       # your methodology write-up (this filled-in template)
```

- **`output/`** — the two TSV files described above: `matching_results.tsv` and
  `candidate_pairs.tsv`.
- **`code/business_entity_resolution/`** — a self-contained, runnable copy of your
  pipeline. Put all source under `src/`, and include a `README.md` with exact run
  instructions plus a `requirements.txt` (or equivalent environment file) pinning
  versions. Anyone should be able to regenerate both output files from the
  training/test data using only what is in this folder.
- **Methodology document** — fill in the provided `Documentation_template.md` and drop
  it straight into the zip (the filled-in `.md` is fine; a `.pdf` export works too). No
  need to rename it.

  > **This project's working copy of `Documentation_template.md` now lives at
  > `docs/methodology/Documentation_template.md`** (moved there as part of the repo
  > reorganisation on 2026-09-25). It must still land at the **root** of the final zip
  > as `Documentation_template.md` — see the packaging commands in
  > `code/business_entity_resolution/README.md`, `CLAUDE.md` §4, and `plan.md` CP12 for
  > the exact `cp`/`zip` invocation that does this.

### Constraints

1. Format your output exactly as described above. Submissions that fail validation will
   not be evaluated. You should see a `SCORED` status with your F_0.5 score if the
   output is correctly formatted.
2. `matched_entity_ids` must only reference entities from Source 2 or Source 3.
   Self-matches to Source 1, and IDs that do not exist in the test set, will be
   rejected.
3. Every Source 1 entity must appear in your submission. Missing entities will cause
   rejection.
4. Duplicate entity IDs in any ID list will cause rejection, as will duplicate
   `source1_entity_id` rows.
5. Final model should be a MIT/Apache 2.0 License model and up to 8 Billion parameters.

### Submission Requirements

1. **Leaderboard (during the challenge):** upload `matching_results.tsv` in the Portal —
   tab-separated, with the exact column names described above. This is what drives the
   public and private leaderboards.
2. **Final submission package:** submit the single zip described in *Final Submission
   Package* above — `output/` with **both** `matching_results.tsv` (final matches) and
   `candidate_pairs.tsv` (your candidate-generation / blocking set fed to the model),
   `code/business_entity_resolution/` (runnable pipeline), and your methodology
   document. All teams must submit it; the top teams' packages are reviewed before the
   final rankings are confirmed.
3. Your methodology document must describe:
   - Methodology used
   - Candidate generation / blocking strategy
   - Model architecture and feature engineering
   - Any other relevant information about the approach

   A template for this documentation is provided in `Documentation_template.md`. There
   is no page limit — prioritise clarity and technical depth over brevity.
