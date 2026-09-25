# Pipeline architecture

> Factual restatement of the pipeline design already documented in `CLAUDE.md` §5/§6
> (target design) and `code/README.md` (as-built stage table). No new design decisions
> are introduced here — this file exists purely to give the pipeline a home under
> `docs/` for readers who start from `docs/` rather than `CLAUDE.md`. Treat `CLAUDE.md`
> and `code/README.md` as the sources of truth if this ever drifts from them.

## Stage flow

```
raw records ─▶ normalise ─▶ block (multi-pass union) ─▶ features ─▶ LightGBM ─▶ decide ─▶ submission
```

## Modules (one per stage, all under `code/src/`)

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
code/src/*.py   (pipeline code)
        │
        ├─▶ code/artifacts/   (gitignored caches: normalised frames, embeddings,
        │      candidates, trained model, OOF predictions — keyed by input hash)
        │
        └─▶ output/matching_results.tsv, output/candidate_pairs.tsv   (final deliverable)
```

`experiments/` and `experiments.csv` (see
[`../experiments/README.md`](../experiments/README.md)) record run-level metrics
(blocking recall, mean candidates/S1, validation macro F0.5, precision/recall,
singleton vs non-singleton F0.5, per-country F0.5) rather than pipeline artefacts.

## Storage: local disk vs. S3 vs. GitHub

Three separate things back this project, and the pipeline never conflates them:

* **GitHub is the source of truth for code** — everything under `code/src/`, docs,
  configs. Cloning the repo reproduces the pipeline, not the data.
* **SageMaker local disk (or your laptop's disk) is temporary compute storage.** Each
  SageMaker notebook instance is a separate machine with its own empty, ephemeral
  filesystem; `dataset/`, `code/artifacts/` and `output/` all live there and are
  gitignored. Normal local pipeline execution
  (`python -m src.run_pipeline --mode valid|test`) reads and writes only this local
  disk — it never touches S3, never requires AWS credentials, and never makes a
  network call.
* **S3 (`s3://tensortrio/...`) is persistent storage** — the only thing shared between
  a laptop and every SageMaker instance (CLAUDE.md §3). Moving data between local disk
  and S3 is always an **explicit, opt-in** action, never automatic: `run_pipeline.py`
  does not import or call anything in `s3_sync.py`.

`code/src/s3_sync.py` is that opt-in sync layer: `download_dataset()`,
`upload_artifacts()`, `upload_experiments()` and `upload_submissions()`, each
independently callable. It lazily imports `boto3` only when one of these functions is
actually called (boto3 is intentionally not a pinned pipeline dependency), and every
upload is additive by default — it never deletes existing S3 objects unless a caller
passes `delete_extra=True`. See the module docstring in `code/src/s3_sync.py` for the
exact S3 key-mapping scheme and `code/tests/test_s3_sync.py` for usage examples.
