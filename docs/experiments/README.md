# Experiment tracking

This project tracks experiments two ways, per `CLAUDE.md` §5/§7:

1. **`experiments.csv`** (repo root of `code/business_entity_resolution/`, regenerated
   every run, gitignored) — every invocation of `run_pipeline.py` appends one row:
   timestamp, git commit hash, config diff, blocking recall, mean candidates per S1,
   validation macro F0.5, precision, recall, F0.5 on singletons vs non-singletons, and
   per-country F0.5. This is the machine-generated, exhaustive log.
2. **`plan.md`** submission log and experiment log tables (human-curated) — the subset
   of runs that mattered enough to tag as a checkpoint result or a leaderboard
   submission, with the reasoning for keeping/discarding a change.

The **`experiments/`** directory at the repo root (see
[`../../experiments/README.md`](../../experiments/README.md)) is a human-organised
complement to those two: a place for saved config snapshots, exported/summarised
reports, and result tables that are worth keeping around across sessions without
scrolling through the full `experiments.csv`. It does not replace either of the above —
`experiments.csv` remains the source of truth for what actually ran, and `plan.md`
remains the source of truth for checkpoint status.

## Workflow

- Before starting a task, read the current checkpoint in `plan.md`.
- After a meaningful change, run the validation pipeline (`python -m src.run_pipeline
  --mode valid`) and compare the new `experiments.csv` row against the previous one. A
  change that doesn't improve validation macro F0.5 gets reverted or put behind a
  config flag (`CLAUDE.md` §7).
- Error analysis after each model run: the 50 worst false positives and false negatives
  are dumped to `artifacts/errors_*.tsv` (see
  [`../../artifacts`](../../artifacts) — gitignored, not `experiments/`) and reviewed
  before inventing new features.
