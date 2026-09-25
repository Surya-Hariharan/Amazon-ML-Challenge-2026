# Leaderboard Submission Readiness Audit

```
Purpose: READ-ONLY diagnostic — is the pipeline's --mode test wiring correct and
         ready to produce a valid, scoreable submission at real test scale?
Scope:   Static code trace + existing test-suite evidence only. No full-scale test
         run was executed (see §8 for why). No code under code/src or code/tests
         was modified.
Date:    2026-09-25
```

---

## 1. Environment verification

Ran via the `torch-gpu` conda env's interpreter directly (`conda activate` is not
available in this non-interactive shell; same env, same site-packages as
`conda activate torch-gpu && python`):

```
C:\Users\surya\anaconda3\envs\torch-gpu\python.exe --version   -> Python 3.10.20
torch.__version__                                              -> 2.5.1+cu121
torch.cuda.is_available()                                      -> True
torch.cuda.get_device_name(0)                                  -> NVIDIA GeForce RTX 4060 Laptop GPU
```

Matches the previously-verified baseline exactly. The known, already-flagged
`requirements.txt` version mismatch (numpy 2.2.6 installed vs 2.5.2 pinned, etc.) was
not re-diagnosed per the task brief — everything imports and runs.

Dataset presence confirmed (row counts include header):

| File | Rows |
|---|---|
| `dataset/train/train_source1.tsv` | 2,206,822 |
| `dataset/train/train_source2.tsv` | 5,034,617 |
| `dataset/train/train_source3.tsv` | 5,285,604 |
| `dataset/train/train_ground_truth.tsv` | 2,206,822 |
| `dataset/test/test_source1.tsv` | 1,732,545 |
| `dataset/test/test_source2.tsv` | 4,887,274 |
| `dataset/test/test_source3.tsv` | 5,082,317 |

`output/` contains only `README.md` — no `matching_results.tsv` or
`candidate_pairs.tsv` exist yet.

---

## 2. Test-mode code path (file:line citations)

**A. Does `--mode test` exist in the CLI?**
Yes. `code/src/run_pipeline.py:331` — `parser.add_argument("--mode", choices=["valid", "test"], required=True)`; dispatched at `run_pipeline.py:371-378` (`main`) to `run_test()`.

**B. Does it load `test_source1/2/3.tsv`?**
Yes. `run_test()` (`run_pipeline.py:360-368`) calls `te = load_split("test")` (`run_pipeline.py:364`), which reads `config.TEST_FILES` (`config.py:45-49`, pointing at `test_source1.tsv`/`test_source2.tsv`/`test_source3.tsv`) via `io_utils.load_split` (`io_utils.py:28-34`), which uses `read_tsv` — `pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)` (`io_utils.py:16-18`), matching CLAUDE.md §2.3 exactly.

**C. Does it train on the complete training data (not a subsample) in this mode?**
Yes by default. `config.TRAIN_SAMPLE_FRAC = 1.0` (`config.py:84`); the CLI's `--sample` default is `config.TRAIN_SAMPLE_FRAC` (`run_pipeline.py:334-335`). `run_test(sample=config.TRAIN_SAMPLE_FRAC, ...)` (`run_pipeline.py:360`) calls `_load_train(sample)` → `subsample_train(..., frac=1.0)` (`run_pipeline.py:63-72`), which returns the input unchanged when `frac >= 1.0` (`run_pipeline.py:71-72`). So unmodified `--mode test` trains on all ~2.2M train S1 records and their full S2/S3 pool. (A caller could still pass `--sample <1.0` to shrink training data — the default does not.)

**D. Does it generate candidates for ALL test S1 records?**
Yes. `te_ids = list(prep_te["s1"][config.ID_COL])` (`run_pipeline.py:279`) is built from `prep_te = prepare(t1, t2, t3, ...)` where `t1` is the *unsampled* `test_source1` frame (`test_run` receives `test=(te["s1"], te["s2"], te["s3"])` from `run_test`, `run_pipeline.py:365` — no subsampling is ever applied to the test tuple). `cands_te = block(prep_te, use_cache)` (`run_pipeline.py:280`) blocks over all of `prep_te["s1"]`.

**E/F. Does it generate `candidate_pairs.tsv` and `matching_results.tsv`?**
Yes, both, in one call: `write_submission(matches, candidates, te_ids, valid_ids, out_dir)` (`run_pipeline.py:288`), which internally writes `config.MATCHING_FILE.name` and `config.CANDIDATE_FILE.name` (`io_utils.py:134-138`) — i.e. `matching_results.tsv` and `candidate_pairs.tsv` (`config.py:51-53`).

**G. Does it guarantee exactly one output row per TEST S1?**
Yes, by construction and by validation. `write_submission` iterates `s1_list = list(s1_ids)` (here `te_ids`, one entry per row) in that fixed order (`io_utils.py:110-132`), and **raises `ValueError`** if `s1_ids` contains duplicates (`io_utils.py:112-113`) or if `matches`/`candidates` reference an S1 ID outside `s1_ids` (`io_utils.py:117-120`) — nothing is written if any rule is violated (see docstring, `io_utils.py:107-108`). This guarantees one row per ID in `te_ids`; it does *not* independently re-verify that `test_source1.tsv` itself has no duplicate `entity_id` values (not checked here — would need a direct data scan, out of scope for a code trace).

**H. Is `candidate_pairs.tsv` the exact same set passed to the matcher?**
Yes. `scored = feats_te[["s1_id", "cand_id"]].assign(prob=predict(fitted["model"], feats_te[feature_columns(feats_te)]))` (`run_pipeline.py:283-284`) is the frame actually scored by the model; `candidates = candidates_to_lists(scored)` (`run_pipeline.py:286`) is built directly from that same `scored` frame — the inline comment even says `# exactly the set the model scored`. There is no earlier/looser blocking pass written out instead.

**I. Does it apply the tuned threshold from `decide.py`'s tuning procedure (trained on the train split)?**
Yes. `fitted = fit_and_tune(feats_tr, truth, tr_ids)` (`run_pipeline.py:272`) runs GroupKFold OOF + `tune_threshold` (`decide.py:132-153`) on the **training** data only, producing `fitted["tau"]`/`fitted["singleton_tau"]`. Those exact values are reused for test-set decisions: `matches = apply_threshold(scored, fitted["tau"], te_ids, fitted["singleton_tau"])` (`run_pipeline.py:285`) — no re-tuning on test.

**J. Does it apply the existing assignment/decision logic (one-to-one etc.)?**
Yes. `apply_threshold` (`decide.py:39-67`) defaults `one_to_one=config.ONE_TO_ONE` (`True`, `config.py:170`) and calls `assign_one_to_one` (`decide.py:26-36`) before thresholding — same code path used in `valid_run`, no special-casing for test mode.

**K. Does it enforce (by construction) that final matches ⊆ candidates?**
Yes, and it is a hard, code-level guarantee, not just a design intent. `write_submission` computes `missing = set(match) - set(cand)` per S1 and **raises `ValueError`** if non-empty (`io_utils.py:128-130`) — the function refuses to write either file if any matched ID is not also a candidate.

**L. Does it write the exact required TSV headers?**
Yes. `config.MATCHING_HEADER = ("source1_entity_id", "matched_entity_ids")` and `config.CANDIDATE_HEADER = ("source1_entity_id", "candidate_entity_ids")` (`config.py:69-70`) — verbatim match to `problem_statement.md`'s required columns — written via `_write_two_col` (`io_utils.py:69-76`), which writes `"\t".join(header)` as the first line, `\n` line endings, no quoting.

### Supporting evidence: existing test coverage of exactly this path

`code/tests/test_pipeline.py::test_test_run_writes_valid_submission_with_unseen_country` (lines 79-105) exercises `run_pipeline.test_run()` end-to-end on synthetic data that explicitly includes a **France** test split unseen during training, and asserts: correct headers, one row per S1 in original order, no duplicate/space/quote-containing IDs, matches ⊆ candidates, France entities receive real (non-trivial) predictions with `macro_fbeta > 0.6`. `test_cli_reads_files_from_config_paths` (lines 108-129) exercises the CLI entry point (`rp.main(["--mode", "test", ...])`) reading from `config.TRAIN_FILES`/`config.TEST_FILES` paths exactly as production would, and confirms `experiments.csv` gets a `mode == "test"` row and `matching_results.tsv` has one row per test S1. Both tests are part of the 104-test suite the baseline report already confirmed green under `torch-gpu`.

---

## 3. Output generation path

Traced above (§2, D–L). Summary: `run_test()` → `test_run()` (`run_pipeline.py:257-297`) does: train-side prepare → block → features → `fit_and_tune` (tau selection) → discard train intermediates (`del feats_tr, cands_tr, prep_tr`, line 275) → test-side prepare → block → features → score with the trained model and train-tuned tau → `apply_threshold` → `write_submission`. No stage is skipped or reordered for test mode relative to valid mode except that tau/singleton_tau come from train OOF rather than being re-tuned.

---

## 4. Schema verification (static, `io_utils.py`)

Checked directly against CLAUDE.md §2.5 / `problem_statement.md`'s Output Format section, all enforced in `write_submission`/`_clean_list`/`_write_two_col`:

| Rule | Enforced where | Result |
|---|---|---|
| Tab-separated | `_write_two_col`, `"\t".join(header)` (`io_utils.py:73`) | Yes |
| Exact headers | `config.MATCHING_HEADER`/`CANDIDATE_HEADER` (`config.py:69-70`) | Yes |
| IDs as strings | `read_tsv` uses `dtype=str` (`io_utils.py:18`); `_clean_list` requires `isinstance(raw, str)` (`io_utils.py:54-55`) | Yes |
| Comma-separated, no spaces, no quotes | `",".join(match)`/`",".join(cand)` (`io_utils.py:131-132`); `_clean_list` rejects IDs containing `",\t\n\r\"' "` (`io_utils.py:59-60`) | Yes |
| Empty string for no matches | Empty list → `",".join([])` → `""` | Yes |
| No duplicate S1 rows | `s1_ids contains duplicates` check (`io_utils.py:112-113`) | Yes (given unique `s1_ids` input) |
| No duplicate IDs inside a list | `_clean_list`'s `seen` set drops repeats (`io_utils.py:63-65`); `format_id_list` also dedupes via `dict.fromkeys` (`io_utils.py:44`, used elsewhere) | Yes |
| S2-/S3- IDs only (no S1 self-match) | `_clean_list` requires `rid.startswith(config.MATCH_PREFIXES)` = `("S2-", "S3-")` (`io_utils.py:57`, `config.py:67`) | Yes |
| ID must exist in that split's source files | `rid not in valid_ids` check (`io_utils.py:61-62`); in test mode `valid_ids = set(t2[ID_COL]) | set(t3[ID_COL])` built from the **test** S2/S3 frames (`run_pipeline.py:287`) | Yes |
| matched ⊆ candidates | `missing = set(match) - set(cand)` raises if non-empty (`io_utils.py:128-130`) | Yes |

No generated output files exist yet to cross-check against actual bytes (see §5/§6) — this section is verified from source code only.

---

## 5. Validator result

**Not run.** `output/matching_results.tsv` and `output/candidate_pairs.tsv` do not exist (`output/` contains only `README.md`, confirmed by directory listing). Per the task brief, no safe sampling mechanism exists to generate a bounded real-output smoke test in `--mode test` (see §8), so no output was produced in this session and the validator has nothing to check. Reporting this explicitly rather than guessing a `PASS`.

---

## 6. Completeness checks

Not applicable — no generated output exists (see §5).

---

## 7. Candidate/match consistency

Not applicable — no generated output exists (see §5). Note this is enforced at write-time regardless (§4, "matched ⊆ candidates" row): `write_submission` cannot produce a file where this is violated; it raises instead.

---

## 8. Resource considerations (estimate only, not measured)

No full- or even partial-scale `--mode test` run was attempted. Reasoning:

- `run_pipeline.py`'s only existing size-control flag is `--sample` (`run_pipeline.py:334-335`), which is wired through `_load_train(sample)` (`run_pipeline.py:344-348`) into `subsample_train` — **it only ever subsamples the training tuple** (`run_pipeline.py:360-368`, `run_test`). The test tuple `te = load_split("test")` (`run_pipeline.py:364`) is passed to `test_run` unmodified and is never subsampled anywhere in `test_run` (`run_pipeline.py:257-297`) — confirmed by reading the full function body, and by `subsample_train` only ever being called from `_load_train`. So even `--mode test --sample 0.01` would still run inference over the **entire** 1,732,544-row `test_source1.tsv` and its full S2/S3 pools; it would only shrink the training side.
- There is therefore no existing, safe way to bound the test-side population for a smoke run without either (a) inventing a new CLI flag (disallowed) or (b) pointing `BER_DATA_DIR` at a hand-made smaller copy of the test files (not an existing mechanism either, and would not be testing the real `--mode test` data path against the real files). Per the task brief, neither is in scope.
- Full-scale run size: test S1 is ~173× the baseline's largest tested N=10,000, and by default (`TRAIN_SAMPLE_FRAC=1.0`) the training side is ~2.2M S1 vs the baseline's 10,000 — roughly 220×. `test_run` does the training-side work (prepare/block/features/fit_and_tune) once at full scale, then repeats prepare/block/features at test scale on top.
- Extrapolating (labelled **estimate**, not measured) from the baseline's N=10,000 timings (`experiments/reports/baseline_architecture_validation.md` §13, Table 9: 246.3 s total, of which ~50 s is fixed train-file-load overhead, ~78 s normalize+embed, ~15.5 s blocking, ~70.5 s features, ~22.4 s train+OOF, ~3.1 s full-model train): even the sub-linear stages (blocking, per baseline's own "sub-linear... near-linear" language, §8/§14) would plausibly put a full `--mode test` run (full train + full test, not just N=10,000) in the range of **hours**, dominated by embedding + feature computation over tens of millions of pairs, not minutes. This is an order-of-magnitude estimate from a 173-220× scale jump, not a projection with error bars — treat it as a planning signal, not a runtime commitment.
- Memory: baseline GPU VRAM was flat/sub-linear (909 MB → 1,141 MB across N=500→10,000, `EMBEDDING_BATCH=512` bounds per-batch cost regardless of N), so VRAM is unlikely to be the constraint at full scale. RAM is a bigger open question — TF-IDF sparse products and the feature frame (58 columns × candidate-pairs count, potentially tens of millions of rows at full scale) were not profiled beyond N=10,000 in the baseline, so peak RAM at full scale is **unverified**, not just unmeasured-but-fine.
- Net: the baseline (§"Purpose" of that report) validated pipeline *mechanics* (blocking/features/model/decide/evaluate all function correctly and reproducibly) up to N=10,000; it did not validate `--mode test`'s wall-clock/memory behavior at real test scale, because it never ran that mode at that scale. This audit's static trace found the *wiring* to be correct (§2); it did not and could not confirm the wiring survives the ~200× scale jump without a real run.

---

## 9. Final verdict

**READY FOR FULL TEST INFERENCE** — with an explicit caveat.

Rationale: every wiring check A–L in §2 traces to source that is either self-evidently correct by inspection (headers, file loading, hard `ValueError` guarantees on subset/dedup/prefix rules) or is additionally exercised end-to-end by the existing test suite on synthetic data that specifically includes an unseen France split (`test_test_run_writes_valid_submission_with_unseen_country`), which is the one condition unique to the real test set that the N≤10,000 train-based baseline could not exercise (train has no France rows). No blocker was found in the `--mode test` code path itself.

Caveat (this is why the verdict is not unconditional): the real-scale run has **never actually been executed** — full test inference has not been attempted (no safe sample mechanism exists for the test side, §8), and no output files currently exist to validate (§5). Runtime and peak-RAM behavior at ~173-220× the largest tested scale are **estimated, not measured**, and could in principle surface a scale-specific issue (e.g. a memory ceiling in the TF-IDF/feature stage) that a code trace cannot rule out. "Ready" here means: the pipeline should be run for real next, not that a real-scale success has been confirmed.

---

## 10. Exact next command to run for full test inference

From `code/` (the package root containing `src/`), using the mandated `torch-gpu` interpreter:

```
C:\Users\surya\anaconda3\envs\torch-gpu\python.exe -m src.run_pipeline --mode test
```

This uses every current default (`TRAIN_SAMPLE_FRAC=1.0` → full training data, embeddings on, caching on) per `code/README.md`'s own documented reproduction command. Expect a long run per §8's estimate; monitor for memory pressure during the features stage in particular, since that stage's behavior at full scale is unverified. After it completes, run the mandated validator before treating anything as submission-ready:

```
C:\Users\surya\anaconda3\envs\torch-gpu\python.exe utils/validate_submission.py \
  --matching output/matching_results.tsv \
  --candidate output/candidate_pairs.tsv \
  --test-dir dataset/test
```
(run from the repository root, since `utils/` and `dataset/` are siblings of `code/`, not under it).
