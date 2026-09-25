# utils/

Organizer-provided helper scripts. **Do not modify** — `validate_submission.py` is the
same script the organizers use to sanity-check submission format before scoring, so our
local run must match theirs exactly.

## `validate_submission.py`

Stdlib-only (Python 3.8+, no dependencies) validator for the two submission files. Run
it before every leaderboard upload (CLAUDE.md §2 rule 6):

```bash
python3 utils/validate_submission.py \
    --matching output/matching_results.tsv \
    --candidate output/candidate_pairs.tsv \
    --test-dir dataset/test
```

What it checks, reading only the output files and `dataset/test/test_source1.tsv` (plus
`test_source2/3.tsv` if `--check-ids` is passed):

- File is genuinely tab-separated (catches the #1 mistake: a comma-separated file saved
  with a `.tsv` extension).
- Header matches exactly (`source1_entity_id\tmatched_entity_ids` /
  `source1_entity_id\tcandidate_entity_ids`).
- Exactly one row per required Source 1 test entity — no missing entities, no
  duplicate `source1_entity_id` rows, no rows for an S1 ID outside the test set.
- No duplicate IDs inside a single ID list.
- Every listed ID starts with `S2-` or `S3-` (flags self-matches to `S1-` IDs and any
  ID with the wrong prefix).
- Optionally (`--check-ids`, off by default because it loads all test S2/S3 IDs into
  memory) that every matched/candidate ID actually exists in the test set.
- When both files are present: warns (never fails) if a matched ID is missing from
  that S1's candidate list — usually a pipeline bug.

Exit code `0` and a printed `PASS` means the files are safe to submit. Exit code `1`
prints a numbered list of issues to fix; it never computes or estimates your F0.5 score
— it only checks formatting.

## Arguments

| Flag | Default | Purpose |
| --- | --- | --- |
| `--matching` / `-m` | `output/matching_results.tsv` | Path to the final matches file |
| `--candidate` / `-c` | `output/candidate_pairs.tsv` (if present) | Path to the candidate set file (optional) |
| `--test-dir` / `-t` | `dataset/test` | Folder holding `test_source1/2/3.tsv` |
| `--check-ids` | off | Also verify matched/candidate IDs exist in the test S2/S3 files (memory-heavy on the full test set) |
