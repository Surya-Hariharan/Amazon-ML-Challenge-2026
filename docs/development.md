# Development Guide

How to set up an environment, get the data, run the pipeline and tests, and contribute
changes.

## Environment

- **Python 3.12 or newer** (the pinned NumPy and SciPy releases require it).
- A CUDA GPU is strongly recommended for full-scale runs: embedding ~13M records on a
  CPU is impractically slow. Use `--no-embeddings` on CPU-only machines.
- All dependencies are pinned in [`code/requirements.txt`](../code/requirements.txt).

```bash
conda create -y -n ber python=3.12
conda activate ber
pip install -r code/requirements.txt
```

The default PyPI `torch` wheel for Linux bundles CUDA. For a specific CUDA build, install
`torch` from the matching PyTorch index first.

## Data

The dataset is not part of the repository. The team copy lives in the S3 bucket
`s3://tensortrio` (region `ap-south-1`); each machine downloads its own local copy once:

```bash
aws s3 ls s3://tensortrio/ --recursive --region ap-south-1      # locate the archive
aws s3 cp s3://tensortrio/<path-to-zip> ./dataset.zip --region ap-south-1
unzip dataset.zip -d dataset/
```

The pipeline looks for `dataset/` at the repository root. If the archive unpacks to a
nested folder, point the pipeline at the directory that holds `train/` and `test/`:

```bash
export BER_DATA_DIR=/path/to/dataset
```

See [`dataset/README.md`](../dataset/README.md) for the expected layout and schema.

## Running the pipeline

Run every command from `code/`.

```bash
# Local validation: 80/20 split of train S1s, τ tuned on OOF, macro F0.5 on the 20%
python -m src.run_pipeline --mode valid

# Faster iteration on a seeded sample of S1 entities
python -m src.run_pipeline --mode valid --sample 0.05

# One stage at a time
python -m src.run_pipeline --mode valid --stage blocking   # recall, cands/S1, reduction ratio
python -m src.run_pipeline --mode valid --stage features
python -m src.run_pipeline --mode valid --stage model

# Leave-one-country-out scores (proxy for the unseen France split)
python -m src.run_pipeline --mode valid --loco

# Full run: train on all training data, predict test, write output/
python -m src.run_pipeline --mode test
```

| Flag | Effect |
| --- | --- |
| `--sample F` | Use a fraction `F` of train S1 entities (orphan density preserved) |
| `--stage` | Stop after `blocking`, `features` or `model` (`all` by default) |
| `--loco` | Add leave-one-country-out scores to a validation run |
| `--no-embeddings` | Skip the embedding pass and feature (CPU-only machines) |
| `--no-cache` | Ignore and do not write `code/artifacts/` caches |

Each run appends a row to `code/experiments.csv` with the git commit, the tunable
configuration, blocking recall, candidates per S1, macro F0.5, precision, recall,
singleton vs non-singleton F0.5 and per-country F0.5. Validation runs also write the
50 worst false positives and false negatives to `code/artifacts/errors_{fp,fn}.tsv` and
feature importances to `code/artifacts/importance_valid.tsv`.

## Diagnostics

`src.diagnose` is a measurement-only CLI. It never writes to `output/` and never
changes production behaviour; results go to `code/artifacts/diagnostics/`.

| Subcommand | Purpose |
| --- | --- |
| `baseline` | Reproduce the baseline, optionally with LOCO |
| `errors` | Attribute every lost true match to blocking, matcher or decision |
| `blocking-ablation` | One-factor-at-a-time sweeps over passes and k values |
| `blocking-k-downstream` | End-to-end F0.5 for selected k configurations |
| `blocking-fn-attribution`, `blocking-fn-forensics`, `blocking-address-forensics` | Why specific true pairs were not retrieved |
| `decision-validation` | Threshold and singleton-threshold behaviour |
| `feature-hard-negative` | Feature separability on hard negatives |
| `loco` | Leakage-safe leave-one-country-out evaluation |
| `drift` | Label-free train-vs-test distribution comparison |
| `convergence` | Metric stability across sample sizes |
| `resource-stage` | Time and memory per pipeline stage |
| `chunk-equality`, `fp16-retrieval` | Numerical-equivalence checks for chunking and precision |

```bash
python -m src.diagnose errors --sample 0.0045
python -m src.diagnose --help
```

## Tests

The test suite runs on synthetic fixtures only and needs no dataset.

```bash
cd code
python -m pytest -q
```

## Experiments

[`experiments/scripts/`](../experiments/scripts/) holds standalone drivers that call
the unmodified pipeline and write JSON results to `experiments/results/` (gitignored).
Headline results are summarised in [results.md](results.md).

## Cloud workflow (SageMaker + S3)

- **GitHub** is the source of truth for code.
- **Instance disks are ephemeral.** `dataset/`, `code/artifacts/` and `output/` live on
  local disk and are gitignored.
- **S3 is the only shared persistent storage.** Transfers are always explicit, via
  [`s3_sync.py`](../code/src/s3_sync.py). The S3 root must be set explicitly; there is
  no default bucket.

```python
# export BER_S3_ROOT=s3://tensortrio/amazon-ml-challenge-2026/
from src import s3_sync

s3_sync.download_dataset()           # s3://<root>/dataset/  -> dataset/
s3_sync.upload_artifacts()           # code/artifacts/        -> s3://<root>/artifacts/
s3_sync.upload_experiments()         # experiments/ + experiments.csv + error dumps
s3_sync.upload_submissions(tag="sub-d2-1")  # output/*.tsv   -> s3://<root>/submissions/<tag>/
```

Uploads are additive; nothing in S3 is deleted unless `delete_extra=True` is passed.
`boto3` is imported lazily and is not a pinned pipeline dependency.

## Conventions

### Challenge rules the code must respect

- **No external data.** No entity-resolution services, business registries, geocoding
  APIs, web scraping or internet augmentation. Hand-written normalisation dictionaries
  are allowed.
- **Model licences.** Any pretrained model must be MIT or Apache-2.0 licensed and have
  at most 8B parameters. Record it in [`code/MODELS.md`](../code/MODELS.md) before use.
- **Country is an open set.** Never hard-code, filter or one-hot countries; a
  `same_country` flag is the only country-derived feature.
- **TSV everywhere.** Read with
  `pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)`; write through
  `io_utils`.

### Code

- Every tunable lives in `config.py`; no other module hard-codes a path or a number.
- Every function has a docstring.
- Seed everything with `config.SEED`; cache expensive work in `code/artifacts/`, keyed
  by input hash.
- Pin every new dependency in `code/requirements.txt` as soon as it is used.
- Throwaway analysis goes in `scratch/` (gitignored), not in `src/`.

### Changes

- Run a validation pass after every meaningful change and compare with the previous row
  of `experiments.csv`. A change that does not improve validation macro F0.5 is
  reverted or placed behind a config flag.
- Review `errors_fp.tsv` / `errors_fn.tsv` before adding new features.
- If a change trades precision for recall, state the expected effect on F0.5.

### File naming

| Kind | Convention | Example |
| --- | --- | --- |
| Python modules and tests | `snake_case.py`, tests prefixed `test_` | `error_decomposition.py`, `test_decide.py` |
| Documentation in `docs/` | lowercase `kebab-case.md` | `problem-statement.md` |
| Standard repository files | upper case | `README.md`, `LICENSE`, `MODELS.md` |
| Git tags for leaderboard submissions | `sub-d{day}-{n}` | `sub-d2-3` |
