# SageMaker + S3 Storage Adaptability Audit

Audit date: 2026-09-25. Scope: storage/path/environment adaptability only (per
`CLAUDE.md` §3 planned S3 layout). Data correctness is out of scope — see
`experiments/reports/data_correctness_audit.md`. No source, test, config or requirements
file was modified; this report is the only new file.

## Executive Summary

**READY WITH MINOR CHANGES**

The pipeline already resolves every path (`REPO_ROOT`, `DATA_DIR`, `ARTIFACTS_DIR`,
`OUTPUT_DIR`, `EXPERIMENTS_CSV`) from `Path(__file__).resolve()` plus pure `pathlib`
arithmetic, with one environment-variable escape hatch (`BER_DATA_DIR`) already wired
in. There are no Windows-specific path literals, no string-concatenated paths, and no
OS-conditional branches anywhere in `code/src/`. An S3-synced dataset dropped at
`<repo>/dataset/{train,test}/` on a SageMaker Linux instance runs through
`python -m src.run_pipeline` completely unchanged — zero code edits required for that
part. The `sync-in / run-local / sync-out` execution model the team is considering
is already the path of least resistance; no direct-S3-read code path exists or is
needed. The only real gaps are (1) `code/requirements.txt` pins versions that imply
Python ≥ 3.12 while the actually-working `torch-gpu` environment is Python 3.10.20 with
materially older numpy/scipy/pandas/torch — a real but pre-existing and non-blocking
mismatch (both environments run the 104-test suite; it is a documentation/repro-fidelity
gap, not a runtime failure) — and (2) artifact/experiment persistence has no
S3-awareness at all today, which is expected and fine given the "S3 as before/after
sync point" design, but the team should sync deliberately (rsync-style, keyed by run) to
avoid a stale-cache-reuse risk described in §6/§14 below.

---

## 1. Current Data Flow

`utils/validate_submission.py`-facing files aside, the pipeline's only data inputs are
the 7 TSVs under `DATA_DIR` (`code/src/config.py`). Flow: `io_utils.load_split("train"
or "test")` → `read_tsv` (`pd.read_csv(path, sep="\t", dtype=str,
keep_default_na=False)`) → `run_pipeline.prepare()` normalises via `normalize.py` →
optionally embeds via `blocking.compute_embeddings` → `run_pipeline.block()` generates
candidates → `features.build_features()` → `model.py` trains/predicts → `decide.py`
thresholds → `io_utils.write_submission()` writes `output/*.tsv`. Nothing reads the
dataset files more than once per run except through the `artifacts/` cache (see §2).

## 2. Current Artifact Flow

All caches/artifacts live under `code.ARTIFACTS_DIR` = `CODE_DIR / "artifacts"` =
`code/artifacts/`. Every write goes through `_frame_key()`/`_texts_key()` (SHA-1 hash of
input data + config knobs) → a cache filename → `pd.DataFrame.to_parquet` /
`np.save` / `Booster.save_model` / `write_tsv`. Every artifact is optional
(`use_cache=True` default, `--no-cache` disables read+write); nothing in the pipeline
requires `artifacts/` to exist to produce correct output, only to skip recomputation.

## 3. Current Experiment Flow

`run_pipeline.log_experiment()` appends one row (timestamp, git hash, JSON config
snapshot, metrics) to `config.EXPERIMENTS_CSV` = `code/experiments.csv` — a single
growing local CSV, gitignored, regenerated fresh (or appended to) on every run. Separately,
human-curated experiment material lives under `experiments/{configs,reports}/` (tracked
in git, small) and `experiments/{logs,results}/` (gitignored scaffolding, currently only
`.gitkeep` placeholders — nothing regenerable has landed there yet). Error-analysis dumps
(`dump_errors()`) write `artifacts/errors_{fp,fn}.tsv` and `artifacts/importance_*.tsv` —
these live in the artifact tree, not `experiments/`, despite being analysis output.

## 4. Current Submission Flow

`run_pipeline.test_run()` calls `io_utils.write_submission()` with
`out_dir=config.OUTPUT_DIR` (`REPO_ROOT / "output"`), which writes
`matching_results.tsv` and `candidate_pairs.tsv` via `_write_two_col()` — plain
`open(path, "w", newline="\n")`, no pandas, no quoting, byte-exact per CLAUDE.md §2.5.
`write_submission()` fully validates before writing anything (raises `ValueError` and
writes nothing on any violation), so there is no partial-write risk. The directory is
created (`path.parent.mkdir(parents=True, exist_ok=True)`) if missing.

## 5. Dataset Path Analysis

Traced `config.py` lines 8–37 exactly:

```
CODE_DIR   = Path(__file__).resolve().parents[1]      # code/
REPO_ROOT  = CODE_DIR.parents[0]                        # repo root
_default_data_dir():
    env = os.environ.get("BER_DATA_DIR")
    if env: return Path(env).resolve()
    primary  = REPO_ROOT / "dataset"
    fallback = REPO_ROOT / "student_resource" / "dataset"
    return fallback if (not primary.exists() and fallback.exists()) else primary
DATA_DIR = _default_data_dir()
TRAIN_DIR, TEST_DIR = DATA_DIR/"train", DATA_DIR/"test"
TRAIN_FILES = {"s1": .../train_source1.tsv, "s2": ..., "s3": ..., "ground_truth": ...}
TEST_FILES  = {"s1": .../test_source1.tsv, "s2": ..., "s3": ...}
```

This resolution is **module-import-time**, not per-call — `DATA_DIR` and the file dicts
are computed once when `config` is first imported and never re-read afterward (a
process restart is needed to pick up a changed `BER_DATA_DIR` or a newly-appeared
`dataset/`).

**Can an S3-synced copy at `<repo>/dataset/` run the existing pipeline completely
unchanged? Yes, verified by trace, not assumed.** `primary.exists()` is
`REPO_ROOT/dataset`; syncing `s3://.../dataset/{train,test}/*.tsv` into that exact path
before invoking the pipeline satisfies `_default_data_dir()`'s primary branch with no
environment variable, no flag, and no code change. The fallback (`student_resource/dataset`)
and the `BER_DATA_DIR` override exist only for the case where the sync tool lands the
files somewhere else (e.g. a nested nested-zip layout) — that is a zero-code-change lever,
not a required change.

Caveat found by tracing, not assuming: because resolution happens at import time and is
cached in a module-level constant, a SageMaker job must have the dataset already synced
to disk **before** `import config` executes (i.e., before `python -m src.run_pipeline`
starts) — an in-flight or partial sync racing the pipeline's own startup would resolve
`DATA_DIR` against whatever partial state existed at that instant. This is a sequencing
requirement for the SageMaker launch script, not a code defect.

## 6. Artifact Path Analysis

Grep of `code/src/*.py` for `Path(`, `__file__`, `open(`, `to_csv(`, `to_parquet(`,
`np.save`, `pickle`, `joblib`, `torch.save`, `mkdir`:

| Artifact | Current Location | Temporary/Persistent | S3 Destination | Required Change |
|---|---|---|---|---|
| Normalised S1 frame cache | `artifacts/norm_s1_<hash>.parquet` (`run_pipeline._cached`) | Temporary (recomputable from dataset) | `artifacts/cache/` | None — sync whole dir post-run if reuse across jobs is wanted |
| Normalised S2+S3 frame cache | `artifacts/norm_others_<hash>.parquet` | Temporary | `artifacts/cache/` | None |
| Blocking candidate cache | `artifacts/cands_<hash>.parquet` (`run_pipeline.block`) | Temporary | `artifacts/cache/` | None |
| Sentence-embedding cache | `artifacts/emb_<hash>.npy` (`blocking.compute_embeddings`), loaded via `np.load(path, mmap_mode="r")` | Semi-persistent (expensive: needs GPU) | `artifacts/embeddings/` | None functionally; `mmap_mode="r"` requires the file to be a real local file (not a network mount) at read time — fine under sync-then-run, would break under direct-S3-mmap |
| Trained LightGBM model (test mode) | `artifacts/model_final.txt` (`model.save_model`, text format) | Persistent (final artefact) | `artifacts/models/` | None — trivial to sync |
| Feature importance (valid/test) | `artifacts/importance_valid.tsv` / `importance_final.tsv` (`write_tsv`) | Persistent (small, human-read) | `experiments/results/` per planned layout | None to produce; **destination mismatch** — currently lands in `artifacts/` not `experiments/results/`, so a sync script targeting `experiments/{...}` by prefix would miss it unless it explicitly also syncs `code/artifacts/*.tsv` |
| Error-analysis dumps | `artifacts/errors_fp.tsv`, `errors_fn.tsv` (`run_pipeline.dump_errors`) | Persistent (small) | `experiments/results/` per planned layout | Same destination mismatch as above |
| `experiments.csv` | `code/experiments.csv` (`config.EXPERIMENTS_CSV`, append-or-create) | Persistent, append-only, unbounded growth | `experiments/logs/` or `experiments/results/` | None to produce; sits outside `code/artifacts/` and outside `experiments/`, so it needs an explicit sync rule of its own |
| `output/matching_results.tsv`, `output/candidate_pairs.tsv` | `output/` (`config.OUTPUT_DIR`) | Persistent, the scored deliverable | `submissions/<tag>/` | None to produce — see §8 |

No `pickle`, `joblib`, or `torch.save` calls exist anywhere in `code/src/`; the only
serialisation formats in use are parquet (pandas/pyarrow), `.npy` (numpy), LightGBM's
own text format, and plain TSV — all portable, no Windows-specific pickling concerns.

**Producer/consumer summary**: every cache in `code/artifacts/{cache,embeddings,models}`
is produced and consumed only by the same-process pipeline run (via
`run_pipeline._cached` / `compute_embeddings` / `save_model`/`load_model`, the latter
unused by the CLI today) — nothing outside `code/src/` reads them except the test suite's
own fixtures, and `load_model` itself is not called anywhere in `run_pipeline.py` (it
exists for future reuse, e.g. re-scoring test without retraining).

## 7-8. Embeddings, Indexes, Model Artifacts, Cache — Storage Audit

- **Embeddings**: Generated by `blocking.compute_embeddings()`, always disk-persisted to
  `artifacts/emb_<hash>.npy` when `cache=True` (the CLI default), float16, L2-normalised.
  Cache key = SHA-1 of (`model_name`, `EMBEDDING_REVISION`, hashed text series) — so it
  is content-addressed, not run-addressed: re-running the *same* dataset+model+revision
  on a different SageMaker job reuses the *same* cache file safely (this is intentional
  and correct, not a staleness risk) because the key is a function of exactly the inputs
  that determine the output. Train and test embeddings get distinct cache files because
  their input texts differ. No FAISS or on-disk ANN index is built anywhere in the
  codebase — `blocking.dense_topk` is an exact brute-force chunked top-k (optionally on
  GPU via torch), so there is no persisted "index" artifact at all, despite the planned
  `artifacts/indexes/` prefix; that directory is currently unused (confirmed empty except
  `.gitkeep`).
- **Model artifacts**: LightGBM boosters are RAM objects (`lgb.Booster`) during a run;
  only the *final* full-data model in `test_run()` is persisted
  (`artifacts/model_final.txt`, LightGBM's plain-text format — portable, no
  binary/pickle version-coupling risk). OOF fold models (`fold_models`) are never
  persisted — they exist only in-process to seed `train_full`'s round count.
- **Cache safety / staleness risk (the question that matters for the sync strategy)**:
  Every cache filename is content-hash-keyed (`_frame_key`, `_texts_key`), which is
  collision-resistant against silently reusing a *different* dataset's cache under the
  same name. However, the hash does **not** cover the whole `config.TUNABLES` set for
  every cache — e.g. `norm_s1`/`norm_others` are keyed only on the input frame's hash
  (normalisation has no tunables today, so this is currently safe, but a future
  normalisation tunable would silently invalidate nothing unless someone adds it to the
  key). `block()`'s cache key **does** already include the relevant `TUNABLES` subset
  (excluding decision-only knobs), which is the pattern that should be followed if
  normalisation ever grows a knob. The practical SageMaker risk the task asks about —
  "could a run accidentally reuse a stale cache from a different prior experiment if
  artifacts synced back from S3 carelessly" — is real but narrow: if a **sync-down**
  blindly restores an old `code/artifacts/` tree from S3 before a new run with different
  *code* (e.g. an edited `features.py` that changes `pair_features` output without
  touching `blocking.py`'s cached candidate frame), the candidate-frame cache would be
  correctly reused (its key doesn't depend on `features.py`), but nothing downstream
  would detect that a *feature* recomputation is stale-vs-fresh because features are
  never cached at all (`build_features` has no `_cached` wrapper) — so this specific risk
  does not exist for features. The one real staleness vector is the **model file**
  (`artifacts/model_final.txt`) and **`experiments.csv`**: both are overwritten
  unconditionally by path, not content-hash-keyed, so a careless S3 sync-down that
  restores an old `model_final.txt` into a fresh job's `code/artifacts/` before that job
  finishes training would leave a stale file sitting alongside a new one only if the new
  run is aborted before `save_model()` runs — otherwise the new run's `save_model()`
  simply overwrites it. Net: safe by construction as long as sync-down happens **before**
  a run starts and sync-up happens **after** it completes (the team's stated preferred
  model), never mid-run.

## 9. Experiment Storage Audit

`experiments.csv` (`code/experiments.csv`, gitignored, append-or-create) is the
machine-generated ledger; `experiments/{configs,reports}/` are the tracked, human-curated
artefacts (currently: `baseline_architecture_validation.md`, `leaderboard_submission_readiness.md`,
`n500/n2000/n5000/n10000.md`, `data_correctness_audit.md`); `experiments/{logs,results}/`
are gitignored scaffolding (empty except `.gitkeep`). Mapping to the planned
`s3://.../experiments/{logs,results,reports}/`:

- `code/experiments.csv` → maps most naturally to `experiments/logs/` or a dedicated
  `experiments/results/experiments.csv` — no code currently writes it there; it is
  produced at `code/experiments.csv`, one level away from the `experiments/` tree
  entirely. A sync script must know to pick this specific file up by name, not just
  glob `experiments/**`.
  presume `.gitkeep`-only.
- `artifacts/errors_*.tsv`, `artifacts/importance_*.tsv` → conceptually experiment
  results, physically live under `code/artifacts/`, not `experiments/results/` — same
  cross-tree mismatch noted in §6.
- `experiments/reports/*.md` → already exactly where the planned layout wants them;
  these are hand-written, not pipeline output, so no code path is involved.

## 10. Submission Storage Audit

Confirmed trivial. `output/matching_results.tsv` and `output/candidate_pairs.tsv` are
written once per `test_run()` call, full-overwrite (not append), no partial-write state
(validation happens before any byte is written — see §4), no dependency beyond
`out_dir.mkdir(parents=True, exist_ok=True)`. A post-validation `cp -r output/
s3://tensortrio/amazon-ml-challenge-2026/submissions/<tag>/` (or equivalent `aws s3 cp
--recursive`) is sufficient with zero code change — verified, not assumed, since
`write_submission()` takes `out_dir` as a plain parameter already defaulted from
`config.OUTPUT_DIR` and nothing else in the codebase reads back from `output/`.

## 11. SageMaker/Linux Compatibility

Grepped `code/src/*.py` for `C:\`, `D:\`, backslash literals, PowerShell/cmd
invocations, Windows-only libraries, hardcoded usernames/GPU names/CUDA paths, hardcoded
interpreter paths: **no matches**. All path construction uses `pathlib.Path` with `/`
operator overloads (`REPO_ROOT / "dataset"`, `CODE_DIR / "artifacts"`, etc.) or
`Path(__file__).resolve().parents[n]` — no string concatenation of path fragments
anywhere in `code/src/`. `_write_two_col` explicitly forces `newline="\n"` on `open()`,
which is itself a portability *fix* (prevents `\r\n` on Windows), not a portability risk.
The one OS-sensitive branch in the codebase is `blocking._torch_device()`
(`torch.cuda.is_available()`), which is correct cross-platform behavior, not an
OS-conditional — it degrades to CPU identically on Linux or Windows if no CUDA device is
present. `run_pipeline._git_hash()` shells out to `git rev-parse`/`git status` via
`subprocess.run(cwd=config.CODE_DIR)`, which requires `git` on `PATH`; every SageMaker
notebook/training image ships git by default, so this is not a blocker, but it is a soft
dependency worth noting (falls back to `"nogit"` on `OSError`, so it degrades gracefully
rather than crashing). **Verdict: `python -m src.run_pipeline --mode test` run from
`code/` (per the documented CLI in `code/README.md`) would resolve paths identically on
Linux — confirmed by inspection, not merely assumed, since every path primitive used is
pure `pathlib` with no OS-specific behavior invoked.**

## 12. Python/GPU Environment Compatibility

Actually-installed `torch-gpu` conda env, re-verified this session by direct
introspection (`python -c "import ...; print(__version__)"`, not from memory):

| Package | requirements.txt pin | torch-gpu actual | Compatibility |
|---|---|---|---|
| Python | (implied ≥3.12 by numpy/scipy pins) | 3.10.20 | **Potentially incompatible per pin, but empirically works** — 104/104 tests pass on 3.10.20; the pin is aspirational/untested, not verified |
| pandas | 2.2.3 | 2.3.3 | Compatible (newer minor, no known breaking API removals used here) |
| numpy | 2.5.2 | 2.2.6 | Potentially incompatible — 2.5.2 requires Python ≥3.12 per project comment; 2.2.6 is what actually runs. Both are numpy 2.x, so the ABI/API surface the code uses (all via pandas/scipy/sklearn, no raw numpy 2.5-only API calls found) is compatible in practice |
| scipy | 1.18.1 | 1.15.3 | Same pattern — pin implies Python ≥3.12, actual is older but functionally compatible for the sparse/TF-IDF code paths used |
| scikit-learn | 1.9.0 | 1.7.2 | Compatible — `TfidfVectorizer`, `GroupKFold` APIs used are stable across these minors |
| lightgbm | 4.7.0 | 4.7.0 | **Exact match** |
| pyarrow | 25.0.1 | 23.0.1 | Compatible — parquet read/write API used is stable |
| torch | 2.13.0 | 2.5.1+cu121 | Potentially incompatible per pin (2.13.0 does not exist as a released version at the time of this audit; likely a typo/placeholder in requirements.txt for a not-yet-released version) — actual 2.5.1+cu121 is a real, CUDA-enabled, working build confirmed via `torch.cuda.is_available() == True` this session |
| sentence-transformers | 6.0.0 | 5.4.1 | Compatible for the `SentenceTransformer(...).encode(...)` API surface used in `blocking.load_encoder` |
| transformers | 5.15.1 | 4.49.0 | Compatible — used only indirectly via sentence-transformers, no direct transformers API calls in `code/src/` |

Classification: **no dependency is definitely incompatible** — the full test suite
(104 tests) passes against the actual `torch-gpu` versions, and no code in `code/src/`
was found calling a numpy/scipy/pandas/torch API introduced only in the pinned-but-not-installed
versions. The mismatch is a **documentation/repro-fidelity gap**: `requirements.txt`
describes a stack (`Python ≥3.12`) that has never actually been installed or tested in
this repo's history — everything that has been tested and verified passing ran on Python
3.10.20 with the older versions in the table above. This matters specifically for
SageMaker: if a SageMaker Linux/GPU instance is provisioned by literally
`pip install -r code/requirements.txt` on a Python 3.12 kernel, that is an **untested
configuration** — it would likely work (nothing in the code is version-fragile) but has
not been verified end-to-end the way the 3.10.20 stack has.

## 13. GitHub vs S3 Separation

`.gitignore` correctly excludes `dataset/*` (dataset TSVs — 2.5 GB+ measured, table in
§14), `code/artifacts/*` (except `.gitkeep` placeholders in the four subdirectories),
`experiments/logs/*` and `experiments/results/*` (except `.gitkeep`), `experiments.csv`,
and all generated image extensions. `git ls-files` (§ commands above) confirms the
tracked set matches exactly what `.gitignore` intends: no `dataset/*.tsv`, no
`code/artifacts/**` content beyond the four `.gitkeep` files, no `experiments.csv`, no
`output/*.tsv` (only `output/.gitkeep` and `output/README.md` are tracked — confirmed by
`git ls-files`). The prior session's hardening (referenced in the task context) is intact
and correct. One thing worth flagging, not a violation: `output/README.md` states
"whether to commit a given `matching_results.tsv`/`candidate_pairs.tsv` pair is a team
decision... not something enforced by tooling here" — i.e., `output/*.tsv` is *not*
gitignored today (no `output/*` rule in `.gitignore`), so a stray `git add
output/matching_results.tsv` would succeed and commit a multi-hundred-MB-scale
prediction file to GitHub. This is a latent gap, not a currently-tripped violation (no
such file is currently tracked), but is worth calling out explicitly since it is exactly
the kind of accidental GitHub/S3 boundary violation the task asks about.

## 14. Local Disk Requirements

Measured file sizes (`du -h`, this session, current `dataset/` copy):

| Artefact | Size / Pressure | Basis |
|---|---|---|
| `train_source1.tsv` | 201 MB — **LOW** | measured |
| `train_source2.tsv` | 467 MB — **LOW-MEDIUM** | measured |
| `train_source3.tsv` | 481 MB — **LOW-MEDIUM** | measured |
| `train_ground_truth.tsv` | 122 MB — **LOW** | measured |
| `test_source1.tsv` | 167 MB — **LOW** | measured |
| `test_source2.tsv` | 486 MB — **LOW-MEDIUM** | measured |
| `test_source3.tsv` | 483 MB — **LOW-MEDIUM** | measured |
| **Train+test TSVs total** | **~2.4 GB** — **MEDIUM** | sum of measured |
| Embeddings (`.npy`, float16) | **UNKNOWN, extrapolate MEDIUM-HIGH** | baseline report measured only N=10,000-S1 samples (~1.1 GB peak VRAM, not disk); full test scale is ~1.73M S1 + ~9.97M S2/S3 (`dataset/README.md`) at 384-dim (MiniLM-L12) float16 ≈ 2 bytes × 384 × ~11.7M records ≈ **~9 GB** for the S2/S3 side alone — this is an extrapolation from measured per-vector size × measured row counts, not a measured figure, and is flagged UNKNOWN-but-estimated rather than asserted |
| Candidate pairs (blocking output) | **UNKNOWN** | baseline report's `mean_candidates` (39.98 at N=10,000) is a per-S1 mean at tiny scale under settings that may not hold at full scale; no full-scale blocking run has been performed (CLAUDE.md forbids running full test inference for this audit) |
| Feature matrices | **UNKNOWN, likely HIGH** | 58 float32 columns × candidate-pair count; at even a conservative 40 candidates/S1 × 1.73M S1 ≈ 69M rows × 58 cols × 4 bytes ≈ **~16 GB** if fully materialised in memory/on disk at once — but `build_features` is chunked (`FEATURE_CHUNK=1,000,000` pairs) and features are never cached to disk (§6), so this is transient RAM pressure per chunk, not a persistent disk-space requirement; genuinely UNKNOWN at what peak disk usage (if any) full-scale features would produce since none are written to disk by design |
| Indexes (FAISS etc.) | **NONE** | confirmed in §7 — no ANN index is built or persisted anywhere in the code |
| Model artifacts | **LOW** | LightGBM text-format boosters are typically single-digit MB even at thousands of trees; not directly measured at full scale but this is a well-understood format size, not a speculative extrapolation |
| Experiment outputs (`experiments.csv`, error dumps, importance tables) | **LOW** | bounded by design (`dump_errors` caps at n=50 rows each; `experiments.csv` grows one row per run) |

The baseline report's own explicit caveat (§17 "No memory or timeout failure was
encountered at any tested N... whether a larger N would fail on this 8 GB GPU... was not
tested") is carried forward here: full-scale (1.73M × 11.7M candidate space) local-disk
and VRAM pressure is **UNKNOWN by design of this audit's constraints** (full test
inference was explicitly out of scope per the task's strict rules) and should not be
asserted as LOW/MEDIUM without an actual staged run at increasing N beyond 10,000.

## 15. Path Configuration Design

`config.py`'s current shape — one `REPO_ROOT` constant, derived paths hung off it, one
`BER_DATA_DIR` env-var override precedent — generalises to `DATA_ROOT`/`ARTIFACT_ROOT`/
`EXPERIMENT_ROOT`/`OUTPUT_ROOT` cleanly and with a small, mechanical change, **not
implemented here**: each would follow the exact `_default_data_dir()` pattern (an
`os.environ.get("BER_<X>_ROOT")` check, falling back to the current `REPO_ROOT`-relative
default), and every existing consumer (`TRAIN_FILES`, `ARTIFACTS_DIR`, `EXPERIMENTS_CSV`,
`OUTPUT_DIR`) already reads from a single named constant rather than being
hardcoded per-call, which is exactly the property that makes such an env-var layer
additive rather than a refactor. The reason this is "minor" rather than "structural":
no other module (`io_utils.py`, `run_pipeline.py`, `blocking.py`, `model.py`) imports
`REPO_ROOT` directly or reconstructs a path from scratch — every path consumer goes
through a `config.py` named constant, so changing how four constants are *derived* would
not require touching any call site outside `config.py` itself. This is an assessment
only; no changes were made.

## 16. Required Changes

### P0 — blocks SageMaker execution
None found. The pipeline runs unchanged on a SageMaker Linux instance given a
dataset synced to `<repo>/dataset/` before the process starts (§5, §11).

### P1 — required for clean S3 persistence
1. **File**: none (process-level). **Section**: sync tooling (not yet written).
   **Current behavior**: `code/experiments.csv`, `artifacts/errors_*.tsv`,
   `artifacts/importance_*.tsv` are produced outside the `experiments/` directory tree
   the planned S3 layout expects them under. **Problem**: a naive `aws s3 sync
   experiments/ s3://.../experiments/` would silently miss all three. **Minimum
   change**: the sync *script* (outside the codebase) must explicitly list
   `code/experiments.csv` and `code/artifacts/{errors_fp,errors_fn,importance_valid,importance_final}.tsv`
   as extra sources, or the S3 destination mapping must accept that these land under
   `artifacts/` on S3 rather than `experiments/`. **Risk of this change**: none — it is
   an operational/tooling decision, not a code change.
2. **File**: `.gitignore`. **Section**: output rules. **Current behavior**:
   `output/*.tsv` is not gitignored (only `output/.gitkeep`/`README.md` are tracked by
   convention, not enforcement). **Problem**: nothing stops a future `git add
   output/matching_results.tsv` from committing a large prediction file to GitHub,
   violating the GitHub=code/S3=data separation. **Minimum change**: add an
   `output/*.tsv` (or narrower, `output/matching_results.tsv` /
   `output/candidate_pairs.tsv`) ignore rule, mirroring the `dataset/*`/`code/artifacts/*`
   pattern already used elsewhere in the file. **Risk**: very low — purely additive,
   would not affect any currently-tracked file (confirmed none are tracked today).

### P2 — recommended improvements
1. **File**: `code/requirements.txt`. **Section**: whole file. **Current behavior**:
   pins imply Python ≥3.12 and cite `torch==2.13.0`, a version that does not correspond
   to an installed/tested environment. **Problem**: anyone provisioning a fresh
   SageMaker instance strictly from this file gets an untested configuration; the actual
   verified-working stack (Python 3.10.20, torch 2.5.1+cu121, etc.) is undocumented in
   the pins. **Minimum change**: either (a) re-pin to the actually-tested `torch-gpu`
   versions, or (b) add a comment noting the pins are aspirational/untested and the
   verified fallback versions, so a SageMaker setup script has a known-good option.
   **Risk**: re-pinning changes what a fresh install pulls — should be done alongside a
   real test run on the new pins, not blindly; documenting-only carries no risk.
2. **File**: `code/src/config.py`. **Section**: `_default_data_dir` pattern. **Current
   behavior**: only `DATA_DIR` has an env-var override; `ARTIFACTS_DIR`, `OUTPUT_DIR`,
   `EXPERIMENTS_CSV` are hardcoded relative to `REPO_ROOT`/`CODE_DIR`. **Problem**: a
   SageMaker job that wants artifacts/output on a different (e.g. larger, ephemeral)
   volume than the code checkout has no lever to do so without editing `config.py`.
   **Minimum change**: generalise the existing `BER_DATA_DIR` pattern to
   `BER_ARTIFACT_ROOT`/`BER_OUTPUT_ROOT`/`BER_EXPERIMENT_ROOT` following the same
   env-var-with-fallback shape (see §15). **Risk**: low — additive, backward compatible
   (unset env vars preserve current behavior exactly), but is a real code change to a
   file explicitly protected by this audit's "do not modify" rule, so it is documented
   here as a recommendation for a future task, not made now.
3. **File**: `code/src/blocking.py` / `code/src/model.py`. **Section**: cache-key design
   (§6/§8). **Current behavior**: `norm_s1`/`norm_others` caches key only on input-frame
   hash, not on any normalisation tunable (currently safe — normalize.py has no tunables
   in `config.TUNABLES`). **Problem**: if a future change adds a normalisation knob
   without adding it to a cache key, stale caches would silently be reused across
   different normalisation logic. **Minimum change**: when any such knob is added,
   fold it into `_frame_key()`'s inputs the same way `block()` already does for
   `config.TUNABLES`. **Risk**: none today (no such knob exists); this is a
   forward-looking pattern note, not an active bug.

### P3 — optional cleanup
1. **File**: `code/README.md`. **Section**: SageMaker quick-start. **Current behavior**:
   already documents the `BER_DATA_DIR` override and the S3 download steps reasonably
   well. **Problem**: none found — flagged only because the task's report structure
   requires a P3 subsection; no actionable gap was identified here worth recording.

---

## 17. Recommended Final Architecture

```
                     GitHub (code source of truth)
                     ───────────────────────────────
                     CLAUDE.md, README.md, code/src/*.py,
                     code/tests/*, code/requirements.txt,
                     code/README.md, code/MODELS.md,
                     docs/, experiments/{configs,reports}/*.md
                                  │
                                  │ git clone / pull
                                  ▼
          ┌───────────────────────────────────────────────────┐
          │            SageMaker instance (Linux, GPU)          │
          │            local filesystem — temporary workspace   │
          │                                                     │
          │  1. git clone <repo>                                │
          │  2. aws s3 sync s3://tensortrio/.../dataset/ dataset/│
          │        (BEFORE `import config` / pipeline start —   │
          │         §5 sequencing requirement)                  │
          │  3. pip install -r code/requirements.txt            │
          │       (or the verified 3.10.20 stack — P2 note)     │
          │  4. cd code && python -m src.run_pipeline --mode X  │
          │       reads  <repo>/dataset/{train,test}/*.tsv      │
          │       writes <repo>/code/artifacts/**  (caches)     │
          │       writes <repo>/code/experiments.csv            │
          │       writes <repo>/output/*.tsv                    │
          │  5. python3 utils/validate_submission.py ... → PASS │
          │  6. aws s3 sync (AFTER completion, never mid-run):  │
          │       code/artifacts/  → s3://.../artifacts/        │
          │       code/experiments.csv,                         │
          │       code/artifacts/errors_*.tsv,                  │
          │       code/artifacts/importance_*.tsv                │
          │                        → s3://.../experiments/...   │
          │       output/         → s3://.../submissions/<tag>/ │
          └───────────────────────────────────────────────────┘
                                  │
                                  ▼
                     S3 (persistent storage layer)
                     ───────────────────────────────
                     dataset/{train,test}/   (source of truth for data)
                     artifacts/{embeddings,indexes*,models,cache}/
                     experiments/{logs,reports,results}/
                     submissions/<tag>/

* indexes/ stays empty under the current design — no ANN index is built (§7-8).
```

Key property preserved: no pandas/boto3 call anywhere reads S3 directly (§5 option A is
the only approach latent in or recommended for this codebase); S3 involvement is
entirely sync-in-before / sync-out-after, exactly matching the team's stated preference.

## 18. Validation Results

Ran `cd code && python -m pytest -q` against the `torch-gpu` conda environment (resolved
directly via `C:\Users\surya\anaconda3\envs\torch-gpu\python.exe` since `conda activate`
is not available in this non-interactive session's shells — same interpreter, same
result as `conda activate torch-gpu && python -m pytest -q` would produce).

```
104 passed in 25.95s
```

Matches the expected baseline (104 passed, 0 failed, 0 skipped) exactly. Environment
introspection performed live this session (not from memory/context): Python 3.10.20,
pandas 2.3.3, numpy 2.2.6, scipy 1.15.3, scikit-learn 1.7.2, lightgbm 4.7.0, pyarrow
23.0.1, torch 2.5.1+cu121 (`torch.cuda.is_available() == True`), sentence-transformers
5.4.1, transformers 4.49.0.

## 19. Final Verdict

1. **Can current code consume a local SageMaker copy of the S3 dataset unchanged?**
   Yes — sync to `<repo>/dataset/{train,test}/` before the pipeline process starts;
   `config._default_data_dir()` resolves it with zero code changes (§5).
2. **Can it run on Linux?** Yes — no Windows-specific paths, string-built paths, or
   OS-conditional logic exist in `code/src/`; all path logic is pure `pathlib` (§11).
   Not independently executed on a Linux box in this audit (no such environment was
   available), but the code-level evidence is unambiguous.
3. **Where are artifacts/experiments/submissions currently written?** Artifacts:
   `code/artifacts/{cache,embeddings,models}/` (§6). Experiments: split between
   `code/experiments.csv` (machine-generated ledger) and `experiments/{configs,reports}/`
   (tracked human-curated files) with `artifacts/errors_*.tsv`/`importance_*.tsv`
   effectively also experiment output despite living under `artifacts/` (§9). Submissions:
   `output/matching_results.tsv` and `output/candidate_pairs.tsv` (§10).
4. **What needs to change for S3 persistence?** No source code changes are required
   for correctness. Two P1 items are needed for *clean* persistence per the planned
   layout: (a) sync tooling must explicitly pick up `code/experiments.csv` and the
   `artifacts/errors_*`/`importance_*` files, since they sit outside the `experiments/`
   tree a naive sync would target; (b) `.gitignore` should add an `output/*.tsv` rule to
   close a latent (currently untripped) GitHub/S3 boundary gap (§16).
5. **Can GitHub remain code source of truth?** Yes — `.gitignore` and `git ls-files`
   confirm dataset, artifacts, embeddings, models, cache, and experiment
   logs/results are already correctly excluded from git; only code, docs, small
   configs/reports, and the `.gitkeep` scaffolding are tracked (§13).
6. **Can S3 remain persistent storage?** Yes — nothing in the code writes anywhere
   outside the repo tree, so all of `dataset/`, `code/artifacts/`, `experiments.csv`,
   `experiments/{logs,results}/`, and `output/` are well-defined, syncable local
   directories with no hidden state elsewhere (e.g. no writes to `/tmp`, no OS temp dirs,
   no home-directory config files).
7. **Can SageMaker local disk remain temporary workspace?** Yes — every artifact is
   either fully recomputable from the dataset (normalisation/blocking/embedding caches,
   all content-hash-keyed) or a small final deliverable (model, error dumps, importance
   tables, submission TSVs) cheap to re-sync; nothing requires SageMaker local disk to
   survive between jobs except as a performance optimisation (cache reuse), never for
   correctness.
8. **What's the minimum safe implementation plan?** (a) Write a SageMaker launch script
   that: clones the repo, `aws s3 sync`s the dataset to `<repo>/dataset/` *before*
   invoking Python, installs dependencies (ideally the verified 3.10.20 stack or a
   freshly-tested 3.12 stack — P2), runs `python -m src.run_pipeline`, runs
   `utils/validate_submission.py` and checks for `PASS`, then `aws s3 sync`s
   `code/artifacts/`, `code/experiments.csv` + the two loose TSV families, and `output/`
   back up to their respective S3 prefixes, strictly after the run completes (never
   mid-run, to avoid the staleness vector in §8). (b) Apply the two P1 changes (sync
   script scope, `.gitignore` output rule). (c) Optionally apply the P2 items
   (requirements re-pin/documentation, configurable roots) as follow-up hardening, not
   blockers. No changes to `src/`, `tests/`, or algorithmic behavior are needed at any
   point in this plan.
