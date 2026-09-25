# Business Entity Resolution — Amazon ML Challenge 2026

For every Source 1 business record, find the Source 2/3 records that describe the same
real-world business. Scored by macro F0.5 per S1 entity, singletons included.

- **[CLAUDE.md](../CLAUDE.md)** — task, hard rules, repo layout, pipeline design. Read first.
- **[plan.md](../docs/planning/plan.md)** — checkpoint plan and current status.
- **[MODELS.md](MODELS.md)** — licence + parameter count of every model used.

## Quick start on a fresh SageMaker instance

The pinned stack needs **Python >= 3.12** (numpy 2.5 / scipy 1.18). If the instance's
default kernel is older, create an environment first:

```bash
conda create -y -n ber python=3.12 && conda activate ber
```

```bash
git clone <repo-url> amazon_hack
cd amazon_hack
pip install -r code/requirements.txt

# dataset: one-time per instance, never committed (CLAUDE.md §3)
aws s3 ls s3://tensortrio/ --recursive --region ap-south-1        # find the zip key
aws s3 cp s3://tensortrio/<path-to-zip> ./dataset.zip --region ap-south-1
unzip dataset.zip -d dataset/
# If the zip nests folders (e.g. dataset/student_resource/dataset/train/...), point the
# pipeline at the folder that holds train/ and test/:
#   export BER_DATA_DIR=$PWD/dataset/student_resource/dataset

# tests (synthetic fixtures only, ~20 s)
cd code && python -m pytest -q && cd ..

# start Claude Code at the repo root (so it reads CLAUDE.md)
claude
```

## Reproduction (run from `code/`)

```bash
# local validation: 80/20 split of train S1s, tau tuned on OOF, macro F0.5 on the 20%
python -m src.run_pipeline --mode valid
python -m src.run_pipeline --mode valid --stage blocking   # CP3: recall, cands/S1, reduction ratio
python -m src.run_pipeline --mode valid --stage features
python -m src.run_pipeline --mode valid --loco             # + leave-one-country-out (France proxy)
python -m src.run_pipeline --mode valid --sample 0.1       # 10% of S1s, orphan density preserved

# full run: train on train, predict test, write ../../output/{matching_results,candidate_pairs}.tsv
python -m src.run_pipeline --mode test
```

Flags: `--no-embeddings` skips the embedding pass/feature (e.g. on a CPU-only
instance), and `--no-cache` ignores `artifacts/`. Every run appends a row to
`experiments.csv`. Valid runs also write `artifacts/errors_{fp,fn}.tsv` (50 worst
false positives/negatives) and `artifacts/importance_valid.tsv`.

Then validate before any submission (from the folder holding `utils/`):

```bash
python3 utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test
```

The data directory defaults to `dataset/` at the repo root (falling back to
`student_resource/dataset/`); override with `BER_DATA_DIR=/path/to/dataset`.

## Pipeline

| Stage | Module | What it does |
|-------|--------|--------------|
| Normalise | `src/normalize.py` | Indic-script transliteration, NFKD accent stripping, junk/honorific/DBA/website handling, legal-suffix split (EN/IN/FR), address abbreviation expansion (EN/FR), region codes, landmark split, digit tokens |
| Block | `src/blocking.py` | Union of 5 passes per country: name char-TF-IDF, multilingual embeddings, rare name tokens, digit + first-token keys, address-only keys. Chunked for the ~11.7M-record test scale |
| Features | `src/features.py` | 58 name / address / embedding / blocking / context features; missing info is NaN |
| Model | `src/model.py` | LightGBM, GroupKFold(5) by S1 ID, OOF predictions |
| Decide | `src/decide.py` | One-to-one assignment, then tau (and optional singleton threshold) tuned for macro F0.5 including singletons |
| Output | `src/io_utils.py` | `write_submission` enforces every format rule; `candidate_pairs.tsv` = the exact set scored |

## Scale and memory

Test is ~1.7M S1 against ~11.7M S2+S3. The knobs are in `src/config.py`:

- `TFIDF_MAX_PRODUCT_NNZ` caps each sparse-product chunk (about 12 bytes per entry).
- `DENSE_QUERY_CHUNK` × `DENSE_INDEX_CHUNK` bounds the dense top-k working set. It
  runs on the GPU when torch sees CUDA.
- `KEY_PASS_CHUNK` and `FEATURE_CHUNK` bound the joins and feature batches.

Embedding ~13M records needs a GPU. Normalised frames, embeddings and candidates are
cached in `artifacts/`, keyed by input hash.
