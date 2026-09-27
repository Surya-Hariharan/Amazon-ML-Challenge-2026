# Submission Guide

How to produce, validate and package the challenge deliverables. The authoritative
rules are in [problem-statement.md](problem-statement.md) and
[challenge-guidelines.md](challenge-guidelines.md).

## Budget

- 5 leaderboard submissions per day across the 3-day window. Unused submissions do not
  carry over.
- Final ranking uses the **private** leaderboard, so choose the final model by local
  validation and leave-one-country-out scores, not by small public-leaderboard gains.
- Only spend a submission on a change backed by a local validation improvement.
- Tag every submitted commit `sub-d{day}-{n}` (for example `sub-d2-3`).

## 1. Generate the outputs

```bash
cd code
python -m src.run_pipeline --mode test
```

This writes `output/matching_results.tsv` and `output/candidate_pairs.tsv`.

## 2. Validate

From the repository root:

```bash
python3 utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test
```

It must print `PASS`. Never upload a file that has not passed. Add `--check-ids` to also
confirm that every ID exists in the test files (memory-heavy).

## 3. Upload to the leaderboard

Upload `output/matching_results.tsv` in the challenge portal and record the result
against the commit tag.

## 4. Build the final package

The official archive layout is:

```text
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

Two names differ between this repository and the archive: the flat `code/` directory
becomes `code/business_entity_resolution/`, and [`docs/methodology.md`](methodology.md)
becomes `Documentation_template.md` at the archive root.

```bash
TEAM=<team_name>
STAGE=/tmp/${TEAM}_submission
PKG=$STAGE/code/business_entity_resolution

rm -rf "$STAGE" && mkdir -p "$PKG"
cp -r output "$STAGE/"
cp -r code/src code/tests code/README.md code/MODELS.md code/requirements.txt \
      code/conftest.py "$PKG/"
cp docs/methodology.md "$STAGE/Documentation_template.md"
find "$STAGE" -name "__pycache__" -type d -prune -exec rm -rf {} +
rm -f "$STAGE/output/.gitkeep" "$STAGE/output/README.md"

(cd "$STAGE" && zip -r "../${TEAM}_submission.zip" .)
```

`dataset/`, `code/artifacts/`, `docs/`, `experiments/`, `utils/` and `scratch/` must not
be in the archive.

## 5. Final checklist

- [ ] Validator prints `PASS` on the final `output/` files.
- [ ] Every matched ID appears in the same S1's candidate list.
- [ ] `candidate_pairs.tsv` is the exact set the final model scored.
- [ ] Every pretrained model is listed in `code/MODELS.md` with licence and size.
- [ ] `docs/methodology.md` is complete, with final numbers.
- [ ] The archive, unzipped into a clean directory, installs from `requirements.txt`
      and reproduces both output files.
- [ ] Outputs and archive backed up to S3 under `submissions/final/`, never over the raw
      dataset objects:

```bash
aws s3 cp output/ s3://tensortrio/submissions/final/ --recursive \
  --exclude "*" --include "*.tsv" --region ap-south-1
aws s3 cp /tmp/<team_name>_submission.zip s3://tensortrio/submissions/final/ \
  --region ap-south-1
```
