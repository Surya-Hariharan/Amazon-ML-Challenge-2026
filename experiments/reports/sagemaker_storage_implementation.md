# SageMaker storage implementation

Follows the SageMaker storage adaptability audit
(`experiments/reports/sagemaker_storage_adaptability_audit.md`, verdict PASS WITH
CHANGES, no P0 blockers): the ML pipeline was already SageMaker-compatible via local
disk execution and only needed an opt-in S3 sync utility for persistence. This is that
utility, added as a minimal, isolated addition — zero ML algorithm changes.

## Files changed, and why

| File | Change | Reason |
| --- | --- | --- |
| `.gitignore` | Added `output/*.tsv` (with an explanatory comment, matching the file's existing comment style) | Prevents generated submission TSVs from being accidentally committed, per the task spec. `output/.gitkeep` and `output/README.md` remain tracked (unaffected — `*.tsv` only matches the two generated files). |
| `code/src/s3_sync.py` | New file | The opt-in S3 persistence utility (see below). |
| `code/tests/test_s3_sync.py` | New file | Unit tests for the new module, following `code/tests/test_io_utils.py`'s house style (plain `pytest`, `tmp_path` fixtures, one behaviour per test, descriptive docstrings). |
| `docs/architecture/pipeline.md` | Added a short "Storage: local disk vs. S3 vs. GitHub" section at the end | Documents the local/S3/GitHub split and points at `s3_sync.py` as the interface, per the task's documentation requirement. No unrelated content touched. |

**Not changed, confirmed by `git diff --stat` (empty) against every forbidden file:**
`code/src/normalize.py`, `code/src/blocking.py`, `code/src/features.py`,
`code/src/model.py`, `code/src/decide.py`, `code/src/evaluate.py`. `code/src/run_pipeline.py`
was also not touched (confirmed by inspection and by `git status`/`git diff` — it does
not appear in the change set at all). `code/requirements.txt` was not touched (boto3 is
deliberately *not* added there — see below).

## S3 architecture

```
Laptop                    SageMaker instance A          SageMaker instance B
  │  (one-time upload,        │  download_dataset()          │  download_dataset()
  │   already done)           │  ───────────────▶ local disk │  ───────────────▶ local disk
  ▼                           │                               │
s3://tensortrio/<BER_S3_ROOT prefix>/     <── PERSISTENT, shared, survives instance teardown
  ├── dataset/            ◀── raw train/test TSVs (download only)
  ├── artifacts/          ◀── upload_artifacts()   (code/artifacts/: models, embeddings, OOF preds)
  ├── experiments/        ◀── upload_experiments() (experiments/ tree + the 3 "outside" files)
  └── submissions/        ◀── upload_submissions() (output/matching_results.tsv, candidate_pairs.tsv)

Each instance's local disk (dataset/, code/artifacts/, output/) is EPHEMERAL compute
storage — gitignored, not shared with other instances, lost when the instance is torn
down. GitHub remains the source of truth for code/src/*.py, docs, and configs; nothing
here changes that.
```

All sync operations are **opt-in function calls** — nothing in `run_pipeline.py` (or
any other pipeline module) imports or calls `s3_sync`. `python -m src.run_pipeline
--mode test` behaves exactly as before this change: 100% local, no AWS credentials, no
network calls.

## Local vs. persistent storage summary

| Storage | What lives there | Lifetime | Touched by normal pipeline run? |
| --- | --- | --- | --- |
| GitHub | `code/src/*.py`, docs, configs | Permanent, source of truth | N/A (this *is* the code) |
| Local disk (laptop or SageMaker instance) | `dataset/`, `code/artifacts/`, `output/` | Ephemeral — per-machine, gitignored | Yes — the only storage `run_pipeline.py` touches |
| S3 (`s3://tensortrio/...`) | Mirrors of the above, under `dataset/`, `artifacts/`, `experiments/`, `submissions/` prefixes | Persistent, shared across all instances | No — only touched by explicitly calling a `src.s3_sync` function |

## S3 key-mapping scheme (as implemented)

Root is `BER_S3_ROOT` (env var, e.g. `s3://tensortrio/amazon-ml-challenge-2026/`),
parsed by `s3_sync.parse_s3_root` into `(bucket, prefix)`. No hardcoded bucket/prefix
default exists — an absent `BER_S3_ROOT` raises `S3SyncError` the moment any operation
is invoked (never at import time, never guessed).

| Local source | S3 key (relative to `BER_S3_ROOT`) | Function |
| --- | --- | --- |
| `dataset/train/train_source1.tsv` (etc., any file under `config.DATA_DIR`) | `dataset/train/train_source1.tsv` | `download_dataset()` (download-only; structure/filenames preserved) |
| `code/artifacts/models/model_final.txt` (etc., any file under `config.ARTIFACTS_DIR`) | `artifacts/models/model_final.txt` | `upload_artifacts()` |
| `experiments/reports/n5000.md` (etc., any file under `experiments/`) | `experiments/reports/n5000.md` | `upload_experiments()` |
| `code/experiments.csv` (`config.EXPERIMENTS_CSV`) — lives outside `experiments/` | `experiments/experiments.csv` | `upload_experiments()` via `extra_experiment_files()` |
| `code/artifacts/errors_fp.tsv`, `errors_fn.tsv` (written by `run_pipeline.dump_errors`) — lives outside `experiments/` | `experiments/artifacts/errors_fp.tsv`, `experiments/artifacts/errors_fn.tsv` | `upload_experiments()` via `extra_experiment_files()` |
| `code/artifacts/importance_valid.tsv`, `importance_final.tsv` (written by `run_pipeline`) — lives outside `experiments/` | `experiments/artifacts/importance_valid.tsv`, `experiments/artifacts/importance_final.tsv` | `upload_experiments()` via `extra_experiment_files()` |
| `output/matching_results.tsv`, `output/candidate_pairs.tsv` | `submissions/matching_results.tsv`, `submissions/candidate_pairs.tsv` (or `submissions/<tag>/...` if a `tag` is passed, e.g. `sub-d1-1`, matching CLAUDE.md §7's tag scheme) | `upload_submissions()` |

The three "outside `experiments/`" paths were confirmed by reading `code/src/config.py`
(`EXPERIMENTS_CSV = CODE_DIR / "experiments.csv"`) and `code/src/run_pipeline.py`
(`write_tsv(df, out_dir / f"errors_{name}.tsv")` at line 197, and the
`importance_valid.tsv` / `importance_final.tsv` writes at lines 237 and 295) — not
guessed.

### Design details

* **boto3 is lazily imported.** `s3_sync.py` has no module-level `import boto3`; every
  boto3 call goes through `_get_client()`, which imports boto3 only when called and
  raises a clear `RuntimeError`/`S3SyncError` ("boto3 is required for S3 operations;
  install it or add it to requirements.txt") if it's missing. `code/requirements.txt`
  was deliberately **not** touched, per the task's constraints — boto3 is not a
  pipeline dependency.
* **Pure mapping vs. I/O separation.** `parse_s3_root`, `join_key`,
  `iter_local_files`, `relative_key`, `dataset_key_for_local_path`,
  `local_path_for_dataset_key`, `artifacts_key_for_local_path`,
  `experiments_key_for_local_path`, `extra_experiment_files`,
  `submission_key_for_local_path` are all plain functions with no boto3 dependency.
  The boto3-calling functions (`download_dataset`, `upload_artifacts`,
  `upload_experiments`, `upload_submissions`, and the shared `_upload_directory`
  helper) are thin wrappers that call these plus `_get_client()`.
* **Managed transfer.** Uses `client.download_file` / `client.upload_file`
  (boto3's managed `S3Transfer`, multipart-safe for large TSVs) rather than reading
  whole objects into memory, per the task spec.
* **Non-destructive by default.** `download_dataset(overwrite=False)` never clobbers an
  existing local file; `upload_artifacts`/`upload_experiments(delete_extra=False)`
  never delete existing S3 objects. Both defaults can be explicitly overridden by the
  caller, never implicitly.
* **Fails loudly.** Missing `BER_S3_ROOT`, a missing expected submission file, and any
  boto3 exception (e.g. permission errors) all propagate as errors — nothing is
  swallowed.
* **Zero import-time side effects** — confirmed (see Validation below).

## Tests added and results

`code/tests/test_s3_sync.py` — 33 new tests:

* **Pure logic (no boto3 involved at all):** `parse_s3_root` (valid/invalid URIs,
  prefix normalization), `get_s3_root` (env var read / missing-var error),
  `join_key`, `iter_local_files`, `relative_key`, the dataset key round trip and its
  malformed-key rejections, `artifacts_key_for_local_path`,
  `experiments_key_for_local_path`, `extra_experiment_files` (including the "skip
  files that don't exist yet" case and the "don't pick up unrelated files" case),
  `submission_key_for_local_path` (with/without tag).
* **Lazy-import path:** `_get_client` raises `S3SyncError` when boto3 is not
  importable (forced via `sys.modules` injection, so this passes identically whether
  or not boto3 happens to be installed in the dev environment — it is *not* installed
  here) and succeeds when a stand-in `boto3` module is injected.
* **boto3-calling functions, mocked:** `download_dataset` (BER_S3_ROOT-required
  check, new-file download preserving structure, skip-existing-without-overwrite,
  overwrite=True re-download), `upload_artifacts` (uploads every file,
  `delete_extra` opt-in only), `upload_experiments` (directory sync + the 3 extra
  files), `upload_submissions` (missing-file guard, tag-nested keys) — all via
  `monkeypatch.setattr(s3_sync, "_get_client", lambda: MagicMock(...))`, so no real
  boto3, network access, or AWS credentials are needed for any of these.
* Import-time side-effect check (`importlib.reload` with `BER_S3_ROOT` unset).

**Result:** `py -3.12 -m pytest -q` from `code/`:

```
137 passed, 1 skipped in ~15-22s
```

Before this change: **104 passed, 0 failed, 0 skipped** (verified by a fresh
`--collect-only` count of 104 test items and the git history's prior baseline). After:
**137 passed** (104 original + 33 new), **1 skipped** (`tests/test_blocking.py:83`,
"no CUDA device" — pre-existing, unrelated to this change, present before and after).
Zero regressions.

## Confirmations

1. **Forbidden files untouched.** `git diff --stat` against
   `normalize.py blocking.py features.py model.py decide.py evaluate.py` is empty.
   `run_pipeline.py` does not appear anywhere in `git status`/`git diff` either — not
   modified.
2. **Normal local pipeline behavior unchanged.** `run_pipeline.py` was not modified at
   all (confirmed above), and it contains no import of `s3_sync`, so
   `python -m src.run_pipeline --mode valid|test` runs exactly as it did before this
   change.
3. **Zero import-time side effects**, verified:
   ```
   py -3.12 -c "import sys; sys.path.insert(0,'code'); import src.s3_sync as s; print('imported cleanly')"
   ```
   succeeds with `BER_S3_ROOT` unset and boto3 not installed in this environment.
4. **No AWS credentials required for the test suite.** All 33 new tests either avoid
   boto3 entirely (pure-logic tests) or mock `_get_client`/inject a fake `boto3`
   module — none constructs a real `boto3.client("s3")` or makes a network call. The
   dev environment used to run the tests has no boto3 installed and no AWS
   credentials configured, and the suite still passes.
5. **`.gitignore` contains `output/*.tsv`** — confirmed present (added in this change,
   alongside the existing `dataset/*`, `code/artifacts/*`, `experiments/logs/*`,
   `experiments/results/*`, image, and `__pycache__/` rules, none of which were
   removed or altered).

## Deliberately deferred / out of scope

* `code/requirements.txt` was intentionally **not** modified — boto3 stays an optional,
  lazily-imported dependency for S3 operations only, per the task's explicit
  constraint. Installing boto3 (and pinning its version) is a separate task.
* No CLI wrapper was added for `s3_sync.py` — the task allowed one only "if trivial and
  wanted"; the primary interface is the four importable functions, which is sufficient
  for how these are actually invoked (from a notebook cell or a short script on the
  SageMaker instance).
* The known Python-version/torch-version mismatch in `code/requirements.txt`
  (mentioned in the task's constraints) remains out of scope, as instructed.
