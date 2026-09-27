# Business Entity Resolution — Reproduction Guide

For every Source 1 business record, this pipeline finds the Source 2 and Source 3 records
that describe the same real-world business, and writes the two challenge output files.
It is scored by macro F0.5 per Source 1 entity, singletons included.

This folder is self-contained: it regenerates both output files from the training and
test data. The licence and size of every model used are in [MODELS.md](MODELS.md).

## 1. Environment

Python **3.12 or newer** is required (the pinned NumPy and SciPy releases need it). A
CUDA GPU is strongly recommended for the full test run, which embeds ~13M records.

```bash
conda create -y -n ber python=3.12
conda activate ber
pip install -r requirements.txt
```

## 2. Data

Place the challenge data so that it looks like this, relative to the folder that
contains `code/`:

```text
dataset/
├── train/  train_source1.tsv  train_source2.tsv  train_source3.tsv  train_ground_truth.tsv
└── test/   test_source1.tsv   test_source2.tsv   test_source3.tsv
```

Or point the pipeline at any directory holding `train/` and `test/`:

```bash
export BER_DATA_DIR=/path/to/dataset
```

## 3. Reproduce the submission

Run from this folder (the one containing `src/`):

```bash
python -m src.run_pipeline --mode test
```

This trains on the full training data (threshold tuned on out-of-fold predictions),
then blocks, featurises and predicts the test split, and writes:

- `../output/matching_results.tsv` — final matches
- `../output/candidate_pairs.tsv` — the exact candidate set the model scored

Validate both files with the organizer's checker (from the folder containing `utils/`):

```bash
python3 utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test
```

## 4. Local validation

```bash
# 80/20 split of train S1s by ID, τ tuned on OOF predictions, macro F0.5 on the 20%
python -m src.run_pipeline --mode valid

python -m src.run_pipeline --mode valid --sample 0.05       # seeded 5% sample of S1s
python -m src.run_pipeline --mode valid --stage blocking    # recall, cands/S1, reduction ratio
python -m src.run_pipeline --mode valid --loco              # + leave-one-country-out scores
```

| Flag | Effect |
| --- | --- |
| `--sample F` | Use a fraction `F` of train S1 entities (orphan density preserved) |
| `--stage` | Stop after `blocking`, `features` or `model` |
| `--loco` | Add leave-one-country-out scores |
| `--no-embeddings` | Skip the embedding pass and feature (CPU-only machines) |
| `--no-cache` | Ignore the `artifacts/` cache |

Every run appends a row of metrics to `experiments.csv`. Validation runs also write the
50 worst false positives and false negatives to `artifacts/errors_{fp,fn}.tsv`.

## 5. Tests

```bash
python -m pytest -q
```

The suite uses synthetic fixtures only and needs no dataset.

## Pipeline

| Stage | Module | What it does |
| --- | --- | --- |
| Normalise | `src/normalize.py` | Indic transliteration, accent stripping, junk/honorific/DBA/website handling, legal-suffix split (English, Indian, French), address abbreviation expansion, landmark, house-number, region and digit extraction |
| Block | `src/blocking.py` | Union of 7 passes per country: name char TF-IDF, multilingual embeddings, rare name tokens, digit + first-token keys, address keys, street-word pairs, name + address char TF-IDF |
| Features | `src/features.py` | ~60 name, address, number, embedding, blocking and context features; missing information is `NaN` |
| Model | `src/model.py` | LightGBM, `GroupKFold(5)` by S1 ID, out-of-fold predictions |
| Decide | `src/decide.py` | One-to-one assignment, then τ (and optional singleton threshold) tuned for macro F0.5 with singletons |
| Output | `src/io_utils.py` | `write_submission()` enforces every format rule |
| Evaluate | `src/evaluate.py` | Macro F0.5, blocking recall, reduction ratio |
| Config | `src/config.py` | All paths, seeds, k values, thresholds and chunk sizes |

`src/diagnose.py` (with `diagnostics.py`, `drift.py`, `error_decomposition.py`) is a
measurement-only CLI for ablations and error attribution; it never writes to `output/`.
Run `python -m src.diagnose --help` for its subcommands.

## Scale and memory

The test split is ~1.7M Source 1 records against ~10M Source 2 + 3 records. Memory is
bounded by these settings in `src/config.py`:

| Setting | Bounds |
| --- | --- |
| `TFIDF_MAX_PRODUCT_NNZ` | Entries per sparse-product chunk (~12 bytes each) |
| `DENSE_QUERY_CHUNK` × `DENSE_INDEX_CHUNK` | Dense top-k working set (runs on GPU when available) |
| `KEY_PASS_CHUNK` | Source 1 rows per inverted-index join |
| `FEATURE_CHUNK` | Pairs per feature batch |

Normalised frames, embeddings and candidates are cached in `artifacts/`, keyed by an
input hash, so repeated runs skip the expensive stages. All randomness is seeded
(`config.SEED = 42`).
