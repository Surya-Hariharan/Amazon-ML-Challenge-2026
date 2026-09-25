# dataset/

This directory holds the organizer-provided train/test data for the Amazon ML Challenge
2026 Business Entity Resolution task. It is **populated locally by downloading from the
team's S3 bucket** (per `CLAUDE.md` §3) and is **gitignored** — never committed, and
excluded from the final submission zip. This file (`README.md`) is the one exception
checked into git, so the expected layout is documented even when the data itself is
absent from a fresh clone.

> The organizer's zip bundle that this dataset was originally extracted from is no
> longer kept in this repository (it was removed during the 2026-09-25 flatten of
> `code/`) — regenerate from S3 (Option B below) if you need the data again.

## Do not modify

Every file here is exactly as provided by the organizers. Nothing in `code/src/` writes
into this directory; all pipeline outputs go to `output/` (final results) or
`code/artifacts/` (caches, intermediate model/embedding artefacts).

## Layout

```
dataset/
├── train/
│   ├── train_source1.tsv       # Source 1: deduplicated reference records
│   ├── train_source2.tsv       # Source 2 records
│   ├── train_source3.tsv       # Source 3 records
│   └── train_ground_truth.tsv  # source1_entity_id -> matched_entity_ids (train only)
└── test/
    ├── test_source1.tsv        # Source 1 test records — generate matches for every row
    ├── test_source2.tsv        # Source 2 test records
    └── test_source3.tsv        # Source 3 test records (no ground truth provided)
```

## Format

All files are **tab-separated** (`.tsv`); read with `sep="\t"` and `dtype=str,
keep_default_na=False` so IDs and blank fields survive intact (CLAUDE.md §2 rule 3).

Source files (`*_source1/2/3.tsv`) share the same four columns:

| Column | Description |
| --- | --- |
| `entity_id` | Unique ID; prefix (`S1-`/`S2-`/`S3-`) identifies the source |
| `business_name` | Business name (abbreviations, legal suffixes, typos, transliterations) |
| `business_address` | Address (partial, landmark-based, format variations) |
| `country` | Open string label — see below |

`train_ground_truth.tsv` has two columns: `source1_entity_id`,
`matched_entity_ids` (comma-separated `S2-`/`S3-` IDs, empty for singletons).

Full field-level documentation is in
[`../docs/challenge/problem_statement.md`](../docs/challenge/problem_statement.md).

## Scale (measured on the currently downloaded copy, 2026-09-25)

Row counts below exclude the header row; re-run `wc -l` yourself after any re-download
to catch drift rather than trusting these numbers blindly.

| File | Records |
| --- | --- |
| `train/train_source1.tsv` | 2,206,821 |
| `train/train_source2.tsv` | 5,034,616 |
| `train/train_source3.tsv` | 5,285,603 |
| `train/train_ground_truth.tsv` | 2,206,821 |
| `test/test_source1.tsv` | 1,732,544 |
| `test/test_source2.tsv` | 4,887,273 |
| `test/test_source3.tsv` | 5,082,316 |

## Country coverage

- **Train:** `US`, `India` only.
- **Test:** adds `France`, which never appears in training. `country` must be treated as
  an open set everywhere in the pipeline (CLAUDE.md §2 rule 4) — never hard-code,
  filter, or one-hot to `{US, India}`.

## Regenerating this directory

```bash
# From the team's S3 bucket (see CLAUDE.md §3)
aws s3 ls s3://tensortrio/ --recursive --region ap-south-1
aws s3 cp s3://tensortrio/<path-to-zip> ./dataset.zip --region ap-south-1
unzip dataset.zip -d dataset/
```

Verify afterwards that `dataset/train/*.tsv` and `dataset/test/*.tsv` load with
`sep="\t"` and don't collapse into a single column.
