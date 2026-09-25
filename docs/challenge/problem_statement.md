# ML Challenge 2026 — Problem Statement

> Source: verbatim excerpt from the organizer-provided `student_resource/README.md`
> (preserved in full at `resources/student_resource.zip` and, until this reorganisation,
> also at the repo root as `student_resource_README.md`). Reproduced here unedited as the
> authoritative problem statement; do not paraphrase away from it.

## Business Entity Resolution Challenge

In large-scale commercial platforms, business identity data arrives from multiple
independent sources — each contributing partial, noisy fragments of information about
the same real-world entities. These fragments share no common identifiers, and the
challenge of determining which records refer to the same business is known as Entity
Resolution (ER). Your challenge is to build an ML solution that, given business records
from 3 independent data sources with noisy and inconsistent fields, determines which
records across sources refer to the same real-world business entity.

Source 1 is the deduplicated reference source. Your task is to find all matching records
from Source 2 and Source 3 for each Source 1 entity. A Source 1 entity may match zero,
one, or many records from Source 2 and Source 3.

### File Format

**All files in this challenge are tab-separated (`.tsv`), and your submissions must be
tab-separated too.** Tabs are used because business addresses and the ID list columns
both contain commas. Read them with an explicit tab separator, for example:

```python
import pandas as pd
df = pd.read_csv("dataset/train/train_source1.tsv", sep="\t")
```

Reading a `.tsv` without `sep="\t"` will silently produce a single column containing the
whole line.

### Data Description

Each source file (`*_source1.tsv`, `*_source2.tsv`, `*_source3.tsv`) has the following
columns:

1. **entity_id:** Unique identifier for the record. The prefix indicates the source —
   `S1-`, `S2-`, or `S3-`.
2. **business_name:** Name of the business entity (may contain abbreviations, legal
   suffixes, typos, transliterations)
3. **business_address:** Address of the business (may contain partial addresses, format
   variations, missing components, landmark-based references)
4. **country:** Country label for the record. The **training** data covers `US` and
   `India`. The **test** set additionally contains a third country, `France`, that does
   **not** appear in the training data. Treat `country` as an open set of string labels:
   do **not** hard-code, filter, or one-hot your pipeline to only `{US, India}`, and
   remember that every test entity — `France` included — must appear in your submission.

There is no separate *source* column — a record's source is given by its `entity_id`
prefix (`S1-`/`S2-`/`S3-`) and by which file it appears in.

The ground truth file (`train_ground_truth.tsv`) has two columns:

1. **source1_entity_id:** The `entity_id` of a Source 1 record
2. **matched_entity_ids:** Comma-separated list of matching `entity_id`s from Source 2
   and/or Source 3 (empty when the entity has no matches)

**Noise Patterns to Expect:**

- **Name variations:** Abbreviations (Corp vs. Corporation, Pvt vs. Private, Ltd vs.
  Limited), legal suffix inconsistencies, DBA/trade names, punctuation differences (&
  vs. "and"), word-order transpositions, typos
- **Address variations:** Abbreviations (Rd vs. Road, St vs. Street), transliteration
  variants, missing components (no PIN code, no state), landmark-based references (Near
  SBI ATM), municipal numbering formats, component reordering

### Dataset Details

- **Training Dataset:** Business records across 3 sources with ground truth matching
  labels
- **Test Set:** Business records across 3 sources without matching labels

### File Descriptions

*Training files*

1. **dataset/train/train_source1.tsv:** Source 1 training records (the deduplicated
   reference source)
2. **dataset/train/train_source2.tsv:** Source 2 training records
3. **dataset/train/train_source3.tsv:** Source 3 training records
4. **dataset/train/train_ground_truth.tsv:** Ground truth matching labels for the
   training set

*Test files*

1. **dataset/test/test_source1.tsv:** Source 1 test records. Generate matches for every
   entity in this file.
2. **dataset/test/test_source2.tsv:** Source 2 test records
3. **dataset/test/test_source3.tsv:** Source 3 test records

No ground truth is provided for the test set. To measure your own performance, hold out
a validation split from the training data and score it yourself using the F_0.5 formula
given in [`guidelines.md`](guidelines.md).

For the exact output file formats, package structure and formatting constraints, see
[`submission_requirements.md`](submission_requirements.md). For the scoring formula,
leaderboard mechanics and fair-play rules, see [`guidelines.md`](guidelines.md).
