# Pipeline architecture

> Factual restatement of the pipeline design already documented in `CLAUDE.md` §5/§6
> (target design) and `code/business_entity_resolution/README.md` (as-built stage
> table). No new design decisions are introduced here — this file exists purely to give
> the pipeline a home under `docs/` for readers who start from `docs/` rather than
> `CLAUDE.md`. Treat `CLAUDE.md` and `code/business_entity_resolution/README.md` as the
> sources of truth if this ever drifts from them.

## Stage flow

```
raw records ─▶ normalise ─▶ block (multi-pass union) ─▶ features ─▶ LightGBM ─▶ decide ─▶ submission
```

## Modules (one per stage, all under `code/business_entity_resolution/src/`)

| Stage | Module | What it does |
| --- | --- | --- |
| Config | `config.py` | Single source of truth for paths, seeds, thresholds and k values; every other module imports its constants from here |
| I/O | `io_utils.py` | TSV load/save, ID-list formatting, the submission writer that enforces every output format rule |
| Normalise | `normalize.py` | Script/accent normalisation, junk/honorific/DBA handling, legal-suffix split (EN/IN/FR into `name_core` + `name_suffix`), address abbreviation expansion (EN/FR), landmark split, digit-token extraction, acronym generation |
| Block | `blocking.py` | Union of multiple candidate-generation passes per country (char n-gram TF-IDF on `name_core`, multilingual dense embeddings, rare name tokens, digit/postal + first-token keys, address-only keys), chunked for memory-bounded processing at multi-million-record scale |
| Features | `features.py` | Pairwise (S1, candidate) features: name similarity (Jaro-Winkler, token ratios, TF-IDF/embedding cosine, suffix/acronym match, Jaccard), address similarity (token Jaccard, digit/postal/house-number overlap, landmark overlap), and context features (rank, score gap to best, reverse rank, candidate count, source) |
| Model | `model.py` | LightGBM binary pair classifier, cross-validated with `GroupKFold` grouped by S1 ID so no S1 leaks across folds; keeps out-of-fold predictions for threshold tuning |
| Decide | `decide.py` | One-to-one assignment (each S2/S3 record goes to at most one S1, matching the train ground-truth property), then a probability threshold tuned on OOF predictions to maximise macro F0.5 including singletons |
| Evaluate | `evaluate.py` | Macro F0.5 scorer (singleton rules included), blocking recall, reduction ratio, per-country breakdowns |
| Entry point | `run_pipeline.py` | CLI: `--mode valid` (train/valid split, local scoring) or `--mode test` (train on all training data, predict test, write `output/`); `--stage` to run one stage at a time |

See `CLAUDE.md` §5 (or §6, depending on which copy of `CLAUDE.md` you're reading — the
project-root copy in this repo is authoritative for this repo) for the full design
rationale behind each stage, including the specific similarity measures, blocking-pass
tuning targets (≥98% pair recall on validation), and the France/leave-one-country-out
generalisation strategy.

## Data flow between directories

```
dataset/{train,test}/*.tsv   (input, gitignored, see dataset/README.md)
        │
        ▼
code/business_entity_resolution/src/*.py   (pipeline code)
        │
        ├─▶ artifacts/   (gitignored caches: normalised frames, embeddings, candidates,
        │                 trained model, OOF predictions — keyed by input hash)
        │
        └─▶ output/matching_results.tsv, output/candidate_pairs.tsv   (final deliverable)
```

`experiments/` and `experiments.csv` (see
[`../experiments/README.md`](../experiments/README.md)) record run-level metrics
(blocking recall, mean candidates/S1, validation macro F0.5, precision/recall,
singleton vs non-singleton F0.5, per-country F0.5) rather than pipeline artefacts.
