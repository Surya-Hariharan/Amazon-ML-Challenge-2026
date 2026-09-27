# dataset/

This directory holds the organizer-provided train and test data for the Amazon ML
Challenge 2026 Business Entity Resolution task. The data is **not versioned**: download
it locally (see [Getting the data](#getting-the-data)). Only this README is tracked, so
the expected layout is documented on a fresh clone.

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
keep_default_na=False` so IDs and blank fields survive intact.

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
[`../docs/problem-statement.md`](../docs/problem-statement.md).

## Scale

Row counts exclude the header row.

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
- **Test:** adds `France`, which never appears in training. `country` is treated as an
  open set everywhere in the pipeline — never hard-coded, filtered or one-hot encoded.

## Getting the data

```bash
# From the team's S3 bucket
aws s3 ls s3://tensortrio/ --recursive --region ap-south-1
aws s3 cp s3://tensortrio/<path-to-zip> ./dataset.zip --region ap-south-1
unzip dataset.zip -d dataset/
```

Verify afterwards that `dataset/train/*.tsv` and `dataset/test/*.tsv` load with
`sep="\t"` and don't collapse into a single column.
