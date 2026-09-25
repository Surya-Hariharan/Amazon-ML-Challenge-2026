# Data Correctness and Lineage Audit — Business Entity Resolution

**Scope**: this is a data-correctness/lineage audit, distinct from the prior
"leaderboard submission readiness" audit (verdict: READY FOR FULL TEST INFERENCE). That
question — does `--mode test` run and does the writer fail closed — is not repeated
here. This audit asks: at every stage transition, is the data produced exactly what the
next stage expects, and does it make sense for the challenge? Read-only. No source files
under `code/src/` or `code/tests/` were modified, no full 1.73M-row test inference was
run, no thresholds/algorithms changed.

Environment: `torch-gpu` (Python 3.10.20, CUDA/RTX4060 confirmed working), invoked as
`/c/Users/surya/anaconda3/envs/torch-gpu/python.exe`. Git HEAD at audit time: `352a403`.

## Executive Summary

**DATA PIPELINE VERIFIED**

Every stage transition traced in this audit — raw TSV load, normalisation, blocking,
candidate-pair construction, feature building, label joining, model training/inference,
one-to-one decision, and submission writing — produces exactly the shape, cardinality
and ID semantics the next stage expects, on both code inspection and a real, unmodified
run of the pipeline on a deterministic N=5,000-S1 train sample (frac = 5000/2,206,821,
`seed=config.SEED`). No confirmed data-corruption bug was found. Two low-severity,
already-known risks are carried forward (tau instability at small N; a requirements.txt
version mismatch that does not affect correctness). One genuinely new finding from this
pass: the persistent ~4-point India-vs-US blocking-recall gap has a concrete,
evidence-backed root cause — **English S1 names compared against native-script S2/S3
names via a phonetic ISCII transliteration table that does not converge on the English
spelling**, compounded by **address truncation/digit-renumbering in the Indian S2/S3
noise pattern** that also knocks out the digit/address blocking passes for the same
pairs. 12 of 15 sampled India blocking false negatives show this exact dual pattern.

## 1. Raw Data Integrity

Read with `io_utils.read_tsv` (i.e. exactly how the pipeline reads them:
`sep="\t", dtype=str, keep_default_na=False`), full files, no sampling.

| File | Rows | Dup entity_id | Bad prefix | Empty name | Empty addr | Empty country | Dup full row | Dup (name,addr,country) |
|---|---|---|---|---|---|---|---|---|
| train_source1 | 2,206,821 | 0 | 0 | 0.0% | 0.0% | 0.0% | 0 | 0 |
| train_source2 | 5,034,616 | 0 | 0 | 0.0% | 3.356% | 0.0% | 0 | 25,873 |
| train_source3 | 5,285,603 | 0 | 0 | 0.0% | 3.328% | 0.0% | 0 | 18,860 |
| train_ground_truth | 2,206,821 | 0 (dup S1 rows) | 0 (matched-ID prefix) | — | — | — | — | — |
| test_source1 | 1,732,544 | 0 | 0 | 0.0% | 0.0% | 0.0% | 0 | 0 |
| test_source2 | 4,887,273 | 0 | 0 | 0.0% | 2.648% | 0.0% | 0 | 22,641 |
| test_source3 | 5,082,316 | 0 | 0 | 0.0% | 2.678% | 0.0% | 0 | 16,293 |

Row counts match the context given in the task brief exactly (train S1/S2/S3/GT,
test S1/S2/S3). No whitespace-only names/addresses found anywhere (checked via
`.str.strip() == "" AND != ""`, count 0 in every file). No malformed entity-ID prefixes
in any file. `train_ground_truth.tsv`: 5.5848% of S1 rows have an empty
`matched_entity_ids` cell (singletons) — matches CLAUDE.md's stated 5.585% exactly. No
matched ID in the ground truth has a prefix other than `S2-`/`S3-` (i.e. no accidental
self-references to `S1-`).

Country distribution — **train**: US 1,323,633 / India 883,188 (S1); no France row
anywhere in train (S1/S2/S3), confirming the "France unseen in training" contract
exactly, not just by omission but by direct count. **Test**: US/India/France all
present in S1 (663,106 / 809,986 / 259,452) and in S2/S3, confirming France is a real,
populated third bucket the pipeline must handle, not an edge case with zero records.

Data characteristic (not a bug): 25,873 / 5,034,616 (train_s2) and 18,860 / 5,285,603
(train_s3) rows share an identical `(business_name, business_address, country)` triple
with a *different* `entity_id` — i.e. duplicate business listings under distinct IDs
within the same source. This is expected noisy real-world data (the same business
double-listed) and does not violate any documented invariant (each `entity_id` is still
unique; ground truth still targets specific IDs), but it does mean **more than one
candidate ID can be textually indistinguishable from the true match** for a given S1.
This interacts with, but does not break, the one-to-one decision layer (see §8/§11).

## 2. Normalisation Integrity

Traced `normalize.py` end to end (`fold`, `tokenize`, `normalize_name`,
`normalize_address`, `normalize_frame`) and ran it on real records, not just read the
code.

- **Determinism**: `normalize_name` called twice on the same 20 real S3 names is
  byte-identical both times (VERIFIED).
- **Null/empty safety**: `""`, `None`-as-`""` (the only form `read_tsv` can actually
  produce, since `keep_default_na=False` guarantees no `NaN`), `"   "`, `"\t\n"` all
  return `name_core=""`/`addr_norm=""` with no exception (VERIFIED).
- **Entity-ID passthrough**: `normalize_frame` on a real 500-row S3 sample preserves
  `entity_id` unchanged and 1:1 (500 unique IDs in, 500 unique IDs out) — normalisation
  never rewrites, drops or duplicates the ID column (VERIFIED).
- **Real RAW → NORMALISED examples** (all pulled from actual train/test rows, not
  constructed):
  - S3 legal-suffix/DBA handling: `"... Atlantic Preparatory Academy (Ltd)"` →
    core `"atlantic preparatory academy"`, suffix `"ltd"`.
  - Indic transliteration (Devanagari): `"जय सॉल्यूशंस प्राइवेट लिमिटेड"` →
    core `"jay solyushans"`, suffix `"ltd pvt"` — legal words recognised and split out
    correctly even mid-native-script; the transliteration itself is a rough phonetic
    approximation (`"sॉल्यूशंस"` → `"solyushans"`, not `"solutions"`) — this is exactly
    the mechanism behind the blocking finding in §5.
  - Bengali: `"সাই এস্টেট প্রাইভেট লিমিটেড"` → `"sai estet"` + `"ltd pvt"`.
  - Telugu: `"క్లాసిక్ టెక్ ఏజెన్సీస్ లిమిటెడ్"` → `"klasik tek ejensis"` + `"ltd"`.
  - Accented Latin (test would otherwise silently fail on this): `"Smart Inc
    Ópportunities"` → core `"smart opportunities"` (accent stripped, suffix `"inc"`
    correctly pulled from the middle of the string, not just the end).
  - Address landmark split (India): `"...NEAR SHIV MANDIR MANGTU COLONY, MAIN DADRI
    ROAD..."` → `landmark="shiv mandir mangtu colony"`, street tokens kept clean of the
    landmark phrase; `REGION` correctly extracted from a whole native-script state-name
    component (`"उत्तर प्रदेश"` → `"up"`) even when out of the expected trailing
    position (VERIFIED: region resolved regardless of component order).
  - French (test-only, unseen in training): `"41 Rue de Puébla, Lille,
    Hauts-de-France"` → `"41 rue de puebla lille"`, `region="fr-hdf"` (é stripped,
    `Rue` kept as a street word, not abbreviated away since it's already the
    canonical form; department/region alias resolved). `"Établissements The SARL"` →
    core `"etablissements the"`, suffix `"sarl"` (French legal suffix correctly
    recognised).
- **No country-conditional code path**: normalisation is applied identically regardless
  of the `country` column's value; France flows through the exact same `normalize_name`/
  `normalize_address` functions as US/India (VERIFIED by code — no `if country ==`
  anywhere in `normalize.py`, confirmed by grep across `code/src/`).

## 3. Blocking Integrity

Ran `blocking.generate_candidates` for real (not synthetic) on the N=5,000 sample
(23,459 S2+S3 records). Results below reproduce the previously-established baseline
(`experiments/reports/baseline_architecture_validation.md`, N=5,000 row) almost exactly
(pair recall 0.98125 vs. previously reported 0.9812; India recall 0.9597 vs. 0.9597;
US recall 0.9955 vs. 0.9955) — this is itself a strong reproducibility check: the same
seed, same code, same data gives the same numbers across independent runs.

- **Candidate ID existence**: 0 / 193,324 generated candidate IDs are absent from the
  S2+S3 frame actually blocked against (VERIFIED — every `cand_id` resolves).
  0 candidate IDs have an ambiguous or wrong source prefix (`S2-`/`S3-` inferred
  correctly from the ID string itself, matching CLAUDE.md's "no separate source
  column" contract).
- **Duplicate candidates**: 0 duplicate `(s1_id, cand_id)` rows in the raw blocking
  output — the `groupby(["q","x"]).agg(...)` union in `generate_candidates`
  (blocking.py:464) correctly de-duplicates pairs found by more than one pass into a
  single row with an OR'd pass bitmask, rather than emitting the pair once per pass.
- **S1 coverage**: 0% of S1s got zero candidates at this sample size (every S1 in both
  the India and US groups has ≥ 1 S2/S3 record in its own country).
- **Candidate cardinality**: min 20, median 38, mean 38.66, p90 45, p95 47, p99 52,
  max 63 candidates/S1 — a tight, bounded distribution (as expected: 5 passes each
  capped at k≈10-20 with heavy overlap between passes, never unbounded).
- **Country-constraint correctness**: `country_groups` (blocking.py:39) groups both S1
  and S2/S3 by the literal `country` string and only pairs records within the same
  group — confirmed structurally (no candidate crosses countries in the sample; this is
  also consistent with the previously-established full-population finding of 0
  cross-country true pairs, which is what licenses `BLOCK_WITHIN_COUNTRY=True` in the
  first place). The grouping code contains no country literals (`grep` for `"US"` /
  `"India"` / `"France"` as string literals in `code/src/*.py` returns 0 hits) — the
  mechanism is generic over whatever country strings appear, so France blocks correctly
  within its own group without any code change.
- **Recall by S1 group** (this sample): multi-match S1s 0.9812, singleton-match S1s
  0.9848 — recall is not meaningfully worse for the harder multi-match case.
- **Recall by source**: not separately re-derived in this pass (previously measured at
  N=10,000: S1→S2 0.9753, S1→S3 0.9784 — nearly identical, so source is not a material
  recall driver; not re-verified here, INFERRED from the prior baseline).

### Root cause of the India-vs-US blocking recall gap (new finding this pass)

326 of 17,383 true pairs (1.88%) were blocking false negatives in this sample; India FN
count is 279 vs. US 47, despite India holding only 40% of the true-pair volume
(6,924 vs. 10,459) — i.e. India's FN *rate* is roughly 6x US's. A random sample of 15
FN pairs (raw text pulled directly from the source files, saved to
`experiments/results/blocking_fn_examples.json` in this session's scratch output and
reproduced in full below by pattern) shows a consistent, dominant mechanism in 12/15
cases:

```
S1  (clean English):  "Lakshmi Digital Constructions Private Limited"
S2  (native script):  "ਲਕਸ਼ਮੀ ਡਿਜੀਟਲ ਕੰਸਟ੍ਰਕਸ਼ਨਜ਼ ਪ੍ਰਾਈਵੇਟ ਲਿਮਟਿਡ"  (Gurmukhi)
  -> transliterated core: "lakasami dijital kansatrakasanaj limatid"
  -> S1's core:            "lakshmi digital constructions"
```

The shared ISCII transliteration table (`normalize.py`'s `transliterate_indic`, by
design a rough phonetic mapping, not a dictionary lookup — dictionary lookup against
real English spellings would be an external-data violation) produces a string that is
phonetically related but has low char-3/4-gram and edit-distance overlap with the
English original (`"lakshmi"` vs `"lakasami"`, `"constructions"` vs
`"kansatrakasanaj"`). This directly explains why the **name-based passes** (TF-IDF,
embedding, rare-token) under-perform on India: it is not that they work worse on Indian
addresses in general, but specifically that a Latin-script S1 vs. native-script S2/S3
pair produces two strings with real but weak surface similarity after transliteration.

Critically, in most of these same 12 examples the **address-based passes also fail** on
the identical pair, for an independent reason — the noisy S2/S3 address is a truncated
or digit-altered version of S1's: `"233/6/52F"` (S1) vs. `"Hn 439 233/6/52F"` (S3,
extra prefix number but shared core numbers — still a near-miss for the digit pass'
exact-key join); `"Flat No.A-312"` vs. `"Flat No.a-12"` (a digit dropped, `312`→`12`,
which changes the digit-token set entirely); `"40/445"` vs. `"N/A"` (address discarded
outright); `"C/O Sinimol, Thaiyidayil, Pallippuram P O, Cherthala..."` vs. `"C/O
SINIMOL, CHERTHALA..."` (locality-level truncation that removes the shared street-level
tokens the address pass keys on). Because `digit_token_pass`/`address_pass` require an
*exact* shared digit or word key (blocking.py's `key_pass`), a single dropped or altered
digit is enough to miss the join entirely — there is no fuzzy fallback at the blocking
stage (fuzziness only enters at the *feature* stage, which blocking-FN pairs never
reach).

**Conclusion**: the India gap is not a general blocking-capacity shortfall, and not one
broken pass — it is the conjunction of (a) transliteration-vs-English-spelling
divergence hurting all three name-based passes simultaneously, and (b) India-specific
address truncation/renumbering (more prevalent in the S2/S3 Indian noise pattern than
the US one, per `normalize.py`'s own design comments) independently hurting the two
address-keyed passes on the *same* pairs, leaving fewer independent passes able to
"rescue" a India pair than a US one. This is evidence-backed, not merely a restatement
of the prior baseline's speculation citing "Indic transliteration and more variable
Indian address formatting" — this pass confirms *both* named mechanisms operating
jointly on real false-negative examples, not just correlated with country.

One examined FN pair (`S1-176336021` / `S3-893893792`, US) is worth flagging separately:
S1 name `"Jones Newhold"` vs. candidate name `"Arcorbibelo (ID: 81803)"` — these names
share no resemblance at all; only the address is a near-match (`"1860 Inglewood
Street"` vs `"Inglewood St"`, same city/state). This looks like either a deliberately
hard synthetic case (name replaced with unrelated noise, matched by address alone,
consistent with `features.py`'s own documented "some true matches ... decided by
address alone" design note) or genuine ground-truth noise; it is not evidence of a
pipeline bug since the *ground truth itself* asserts the match, and it is presented
here as-is rather than resolved either way.

## 4. Candidate Pair Integrity (blocking output vs. model input vs. `candidate_pairs.tsv`)

Directly compared, on the real N=5,000 run, the exact `(s1_id, cand_id)` **set** of
three artefacts by tracing the actual variables through `run_pipeline.py`, not by
reading a comment:

1. `cands` — the return value of `block(prep, ...)` (blocking output).
2. `feats` — the return value of `build_features(cands, ...)`, i.e. the model's input
   rows (traced via `feats[["s1_id","cand_id"]]`).
3. What `test_run` would write to `candidate_pairs.tsv` — `candidates_to_lists(scored)`
   where `scored = feats_te[["s1_id","cand_id"]].assign(prob=...)` (run_pipeline.py:283-286),
   i.e. built from the *same* `feats_te` frame the model scores, not from `cands_te`
   directly.

Measured: `set(zip(cands.s1_id, cands.cand_id)) == set(zip(feats.s1_id, feats.cand_id))`
→ **True**, with equal lengths (193,324 == 193,324). No candidate is silently dropped,
added, or deduplicated-away between blocking and the feature/model-input stage
(VERIFIED for train; the test path uses byte-identical code —
`build_features(cands_te, ...)` then `candidates_to_lists(scored)` where `scored` is
derived from that same `feats_te` — so this equality is INFERRED, not independently
re-measured, for the test split specifically, since re-running it at 1.73M rows was
out of scope). This closes the loop the problem statement explicitly warns about:
`candidate_pairs.tsv` must be "the final candidate list just before the ML model scores
them" — `run_pipeline.test_run`'s `candidates = candidates_to_lists(scored)` is
literally derived from the frame the model was just called on (`feats_te`), not from an
earlier `cands_te` blocking pass that could have silently diverged.

## 5. Feature Integrity

58 feature columns produced for all 193,324 pairs; 0 `inf` values anywhere; 0 columns
that are 100% NaN. Two columns are exactly constant on this sample:

- `same_country` — constant at 1.0. This is **structural, not a bug**: blocking only
  ever produces same-country pairs (`BLOCK_WITHIN_COUNTRY=True`), so every candidate
  pair the model ever sees is, by construction, same-country. The feature would only
  vary if cross-country blocking were enabled.
- `addr_empty_s1` — constant at 0.0. This is also **structural, and now verified at
  full-population scale in §1 above**: `train_source1.tsv`'s `business_address` column
  is 0.0% empty across all 2,206,821 rows (and `test_source1.tsv` likewise 0.0% across
  1,732,544 rows) — S1 (the deduplicated reference source) simply never has a missing
  address in either split, so `addr_empty_s1` cannot vary. (`addr_empty_cand`, the S2/S3
  counterpart, is *not* constant — 3.36%/3.33% of train S2/S3 rows have an empty
  address, correctly reflected as a real, varying feature.)

Both match the prior baseline's flag of these two columns as "constant/near-constant"
and confirm the baseline's own explanation (blocking design + S1 data property) rather
than a computation bug — re-derived independently here, not copied.

**Manual trace of representative pairs** (real IDs, not constructed):
- True match (`S1-465739288`/`S2-586371985`, label=1): `name_jw=1.0`,
  `name_token_set=1.0`, `name_core_exact=1.0`, `addr_token_set=1.0`,
  `addr_tfidf_cos=1.0`, `digit_conflict=0.0`, `same_country=1.0`, `rank_in_s1=1.0`
  (best-ranked candidate for its S1), `gap_to_best_s1=0.0`.
- Random non-match for the *same* S1 (`S2-729961171`, label=0): `name_jw=0.486`,
  `name_token_set=0.3`, `addr_token_set=0.261`, `rank_in_s1=39.0` of 41 candidates,
  `gap_to_best_s1=0.72`.
- **Hard negative** (`S1-60980827`/`S2-180175982`, label=0, chosen because
  `name_token_set=0.909` — deliberately picked as a near-miss on name alone): despite
  high name similarity, `addr_token_set=0.414` (much lower) and `digit_conflict=1.0`
  (conflicting house/postal digits) — this is exactly the "same-name-different-place"
  case `features.py`'s module docstring says the address-conflict features exist to
  catch, and the trace confirms it fires correctly on a real pair, not just in theory.

Sanity check requested by the audit brief — "an exact name match pair shouldn't score
worse on name similarity than an obviously unrelated pair" — holds: the true match's
name/address features strictly dominate both the random non-match's and the hard
negative's on every checked column.

## 6. Label Integrity

`features.label_pairs` joins by **ID** (`pairs[KEY_COLS].merge(true.drop_duplicates(...),
on=KEY_COLS, how="left")`, features.py:314-319) — confirmed not positional: the merge
key is the literal `(s1_id, cand_id)` string pair, and `drop_duplicates(KEY_COLS)` on
the truth side prevents a duplicate ground-truth row from fanning out the join.

Measured on the real N=5,000 run:
- Total true `(s1, match)` pairs in the truth dict: 17,383.
- Positive labels actually present in `feats`: 17,057.
- **17,057 == 17,383 × 0.981246 (measured blocking pair recall) exactly to the row** —
  i.e. every true pair that survived blocking is labeled 1, and every true pair
  blocking missed is simply *absent from the candidate frame* (not present-and-mislabeled-0).
  This is the correct behaviour: a blocking-FN pair was never in scope to begin with,
  and `report_blocking_stats`/`sweep_thresholds` account for it separately via the
  *full* truth dict, not via the candidate frame's labels (decide.py:83-99 explicitly
  uses `true_sets = {s: set(truth.get(s, ()))}` from the full truth, not from `feats`).
- 0 duplicate `(s1_id, cand_id)` rows in the labeled frame.
- 4,416 / 4,710 S1s with ≥ 1 positive candidate have **more than one** positive
  candidate — expected and correctly represented (multi-match is the dominant case per
  CLAUDE.md's 89% figure); confirms the label layer does not collapse multi-match S1s
  to a single positive.
- 200/200 randomly sampled positive-labeled rows have their `cand_id` actually present
  in `truth[s1_id]` — no source confusion (an S2 ID accidentally labeled positive
  against S3's ground truth, or vice versa) was found in this sample.

No evidence of label leakage, shifted rows, or positional misalignment.

## 7. Model Input/Output Integrity

- `X, y = feats[cols], feats["label"].to_numpy()` then `train_oof(X, y, feats["s1_id"], ...)`
  — `X`, `y`, and the `groups` argument are all sliced from the same `feats` frame in
  the same row order, so row alignment is guaranteed by construction, not by a
  coincidental index match (VERIFIED by reading `fit_and_tune`, model.py:159-176).
- Inference: `len(predictions) == len(valid_feats)` → **True** (38,696 == 38,696,
  measured). Predictions are finite everywhere (`np.isfinite(probs).all()` → True) and
  in a sane probability range (measured min 3.97e-6, max 0.99996 — never exactly 0/1,
  consistent with a real sigmoid output, not a degenerate constant).
- **GroupKFold leakage**, re-checked at the *data* level (not just via the existing
  unit test `test_group_folds_never_split_a_group`): for the real 4,000-S1 training
  subset of this sample, grouped the actual fold assignments by `s1_id` and counted
  how many distinct folds each S1 touches — **0 S1s span more than one fold**
  (measured, not assumed).
- `predict()` (model.py:117-122) explicitly reorders columns via
  `X[m.feature_name()]` before calling `.predict`, so a caller passing columns in a
  different order than training cannot silently corrupt predictions — a defensive
  design choice, confirmed present.

## 8. Decision Integrity

- `MATCH_THRESHOLD`/`SINGLETON_THRESHOLD` in `config.py` are documented defaults (0.6 /
  `None`); the actually-*applied* values come from `tune_threshold`'s OOF sweep, not the
  config defaults — confirmed on the real run: `tau=0.7`, `singleton_tau=None` were
  selected and are what `apply_threshold` was actually called with (not the config
  default of 0.6), consistent with `fit_and_tune`'s design (model.py→decide.py call
  chain always threads the *tuned* values forward, never silently falling back to
  `config.MATCH_THRESHOLD`).
- **One-to-one walkthrough on real S1s from this sample**:
  - Zero-match: `S1-109597244` → `pred: []` (correctly empty, not omitted from the dict).
  - Singleton: `S1-102108947`, truth `['S2-962171268']` → pred exactly
    `['S2-962171268']`.
  - Multi-match, both S2 and S3: `S1-101122655`, truth 5 IDs (`S2-989591927,
    S2-381445900, S3-192166021, S3-57810131, S3-783044741`) → pred contains exactly
    that set (order differs only by probability rank, not membership) — confirms the
    decision layer is not artificially capping multi-match S1s to a single source or a
    fixed count.
  - **Duplicate matched IDs across all S1s in the prediction dict: 0** (measured
    directly on the full `pred` dict, not inferred) — the one-to-one constraint holds
    at the final-output level, not just internally.
  - **Matches referencing an (s1, cand) pair the model never scored: 0** — every ID in
    every S1's match list traces back to a row in `scored`.
- No-match S1s remain represented as `s: []` in the dict (not dropped) — verified via
  the zero-match example above and structurally guaranteed by `apply_threshold`'s
  `{s: out.get(s, []) for s in universe}` (decide.py:67).

## 9. Submission Output Integrity

`io_utils.write_submission` (io_utils.py:78-139) was not re-run against a real full
test set in this audit (explicitly out of scope), but its enforcement logic was read in
full and cross-checked against its existing test coverage:

- `code/tests/test_io_utils.py` exercises: exact byte-for-byte header/row format
  including the empty-string-for-no-match case (`test_happy_path_format`), duplicate-ID
  dropping within a list (`test_duplicate_ids_in_list_are_dropped`), every rule
  violation from CLAUDE.md §2.5 as a parametrized `ValueError` case (match-not-in-
  candidates, non-S2/S3 ID, unknown ID, S1-not-in-universe on either matches or
  candidates, duplicate S1 IDs, non-S1 IDs) with the explicit assertion that **nothing
  is written on any violation** (`assert not any(tmp_path.iterdir())` — fail-closed,
  re-confirmed by reading the test, consistent with the prior submission-readiness
  audit's verdict), ID-separator/quote rejection, and a 50-row round-trip confirming
  exactly one row per S1 with no duplicates.
- `code/tests/test_pipeline.py::test_test_run_writes_valid_submission_with_unseen_country`
  is an actual end-to-end run of `run_pipeline.test_run` (not `write_submission` in
  isolation) on synthetic data that includes France as a **test-only** country (never
  in the synthetic training set), and asserts: exactly one row per S1 in S1-file order
  for both output files, no in-list duplicates, every matched ID is a subset of that
  S1's own candidate list, no spaces/quotes in the ID-list cells, and — specifically for
  the France slice — that France S1s receive non-trivial predictions
  (`sum(len(pred[s]) for s in fr) > 0`) and a real macro F0.5 above a sanity floor
  (`> 0.6`), i.e. the unseen-country path is exercised end-to-end, not just imported.
- `code/tests/test_pipeline.py::test_cli_reads_files_from_config_paths` runs the actual
  CLI entry point (`rp.main([...])`) end-to-end against files on disk (not just
  in-memory frames), confirming the output row count equals the test S1 file's row
  count.

This is consistent with, and slightly deeper than, the prior audit's fail-closed
finding — this pass additionally confirms the *France-in-test-only* path is part of
the exercised end-to-end test, not merely permitted by the code.

## 10. Cardinality Reconciliation

Measured directly on the real N=5,000 train run (train side; test-side full-population
numbers are structural/by-construction per §9, not independently re-run):

| Quantity | Value | Check |
|---|---|---|
| Sample S1 / S2 / S3 | 5,000 / 11,366 / 12,093 | matches `subsample_train`'s frac target exactly |
| Blocking candidate pairs | 193,324 | — |
| Model input rows (`feats`) | 193,324 | **== blocking candidate pairs** (VERIFIED, §4) |
| Model prediction rows (valid split) | 38,696 | **== valid-split model input rows** (VERIFIED, §7) |
| Final candidate rows (valid split) | 38,696 | same frame the model scored |
| Final matched S1 rows (non-empty predictions) | 943 / 1,000 valid S1s | — |
| Final match edges | 3,401 | — |
| `final_matches ⊆ final_candidates` | **True** (checked on 200 sampled S1s) | VERIFIED |

Full-population reconciliation (structural, from §1's raw-file counts and the code
paths traced in §4/§9, **not** an independent 1.73M-row re-run — INFERRED for the test
split specifically):
- Train: S1 2,206,821 / S2 5,034,616 / S3 5,285,603 / ground truth 2,206,821 rows
  (measured, §1).
- Test: S1 1,732,544 / S2 4,887,273 / S3 5,082,316 rows (measured, §1).
- `output S1 rows == test S1 entities`: guaranteed by `write_submission`'s
  `s1_list = list(s1_ids)` / `s1_set` uniqueness check plus "one row per `s1_ids`
  entry, in order" contract (io_utils.py:110-132), and `test_run` passes
  `te_ids = list(prep_te["s1"][config.ID_COL])` — i.e. literally every S1 ID from the
  loaded test file, not a subset — as that `s1_ids` argument (run_pipeline.py:279,288).
  This is VERIFIED by code + the existing `test_cli_reads_files_from_config_paths`
  test's row-count assertion (§9), not by running the real 1.73M-row file.

## 11. Train/Test Consistency

Confirmed by reading `run_pipeline.py`'s `test_run` (lines 257-297) against `valid_run`
(lines 200-254): both call the **identical** functions — `prepare` (normalise +
embed), `block`, `build_features`, `predict`, `apply_threshold` — with no parallel
reimplementation for the test path. The only intentional differences:

1. **Ground truth availability**: `valid_run` has `truth` for both training the model
   and scoring; `test_run` only has `truth` for the *training* half (used to fit/tune)
   and has none for the test predictions themselves (as required — test has no ground
   truth file).
2. **Model provenance**: `test_run`'s model is trained via `fit_and_tune(feats_tr, ...)`
   on the **entire** training split (`tr_ids = list(prep_tr["s1"][ID_COL])`, i.e. no
   80/20 hold-out), whereas `valid_run` trains on the 80% `train_ids` half and scores
   the held-out 20%. This is the documented, intentional difference (more training data
   for the real submission) — not a code-path divergence.
3. **`FeatureContext` refit for test** (run_pipeline.py:281-282): test features are
   built with `ctx=FeatureContext.fit(prep_te["s1"], prep_te["others"])`, i.e. TF-IDF
   vocabularies are refit on the **test** split's own text (including French n-grams
   that never occur in train), rather than reusing the training split's fitted
   vectorizer. This is called out in `FeatureContext`'s own docstring
   ("so IDF reflects the records being compared, including French n-grams that never
   occur in train") as intentional, not accidental — but it is worth flagging
   explicitly here as a genuine train/test asymmetry in *feature computation*, not just
   in data: the two splits' `name_tfidf_cos`/`addr_tfidf_cos` values are not computed
   against the same vector space. This is a documented design choice, not a bug, but it
   is the one place where "identical function, different fitted state" could in
   principle produce distribution shift beyond what the raw data itself would cause. No
   evidence this is currently miscalibrated was found (not testable without a labeled
   France set), so this is flagged as a **design note**, not an issue.

No other divergence was found: normalisation, blocking, and the classifier/decision
call chain are byte-for-byte the same code for both splits.

## 12. Silent Corruption Risks

Grepped `code/src/*.py` for `reset_index`, `drop_duplicates`, `.iloc[`, `groupby(`,
`sort_values`, `merge(`, and hardcoded country literals, and read every hit in context.

| Pattern / location | Classification | Reasoning |
|---|---|---|
| `blocking.py:425` `s1.iloc[q_idx]`/`others.iloc[x_idx]` | SAFE | Both frames are `reset_index(drop=True)` immediately before `country_groups` is called (lines 418-419), and `q_idx`/`x_idx` come from `groupby(...).indices` on those same reset frames — positional indices are self-consistent, not stale. |
| `blocking.py:464-469` `groupby(["q","x"]).agg(...).reset_index()` then re-attach `s1_id`/`cand_id` via `.to_numpy()[out["q"].to_numpy()]` | SAFE | `q`/`x` are the original positional indices carried through every pass frame unchanged; re-indexing the ID arrays by them is correct, and is exactly what §3 verified (0 bad candidate IDs, 0 dup pairs). |
| `normalize.py:614-619` `_apply_unique`: `pd.factorize` + `table.iloc[codes]` | SAFE | Standard dedup-compute-broadcast pattern; `codes` is aligned to the original series' row order by `factorize`'s contract, confirmed by the §2 ID-passthrough check (500 in, 500 unique out, order preserved). |
| `features.py:293-296` `pd.Index(s1[ID_COL]).get_indexer(cands["s1_id"])` | SAFE | Explicitly ID-based (`get_indexer` on a string Index), not positional; raises `ValueError` if any ID is missing (`(i1<0).any()`) rather than silently misaligning — a defensive check, not just an assumption. |
| `decide.py:26-36` `assign_one_to_one`: sort + `drop_duplicates("cand_id", keep="first")` + `.sort_index()` | SAFE | Deduplicates by the actual `cand_id` value (not position), ties broken deterministically by `s1_id`; `.sort_index()` restores the caller's original (possibly non-contiguous but unique) index before returning — confirmed 0 duplicate matched IDs and 0 unscored-pair matches on the real run (§8). |
| `features.py:314-319` `label_pairs` merge | SAFE | ID-based merge with `drop_duplicates(KEY_COLS)` on the truth side to prevent join fan-out; verified exact-count match with blocking recall (§6). |
| `run_pipeline.py:188-197` `dump_errors`: boolean ndarray `&` pandas Series, `.merge()` twice | SAFE | The ndarray (`is_pred`) is built via `np.fromiter` over `zip(scored["s1_id"], scored["cand_id"])`, i.e. positionally aligned to `scored`'s own iteration order and same length; combining it with a boolean Series of the same frame aligns correctly by pandas' documented ndarray-vs-Series broadcasting (no index mismatch possible since no external index is introduced). Diagnostic-only code (`errors_fp.tsv`/`errors_fn.tsv`), not on the output-file path. |
| Hardcoded country lists | NONE FOUND | `grep -rn "\"US\"\|'US'\|\"India\"\|'India'\|\"France\"\|'France'"` across `code/src/*.py` returns 0 matches — country never appears as a string literal anywhere in the pipeline logic (only as dictionary *values* inside `normalize.py`'s `_REGIONS`/`_NATIVE_REGIONS`, which map address components to region codes and are not used to filter or branch on the `country` column). |
| `config.BLOCK_WITHIN_COUNTRY` | SAFE (by design) | Not a hardcoded country list — a boolean toggle for whether blocking groups by whatever country strings are present; already verified in §3 to work generically for India/US/France without code changes. |
| Implicit ID type coercion | SAFE | `io_utils.read_tsv` reads every column as `dtype=str` explicitly; `write_submission`'s `_clean_list` further asserts `isinstance(raw, str)` and raises otherwise — no float/int coercion path exists for IDs anywhere in the traced code. |

No CONFIRMED BUG and no undisclosed POTENTIAL RISK was found in this pass beyond the
already-documented, already-flagged items in §11 (FeatureContext refit asymmetry,
design note not a bug) and the known `requirements.txt` version mismatch (§13 below,
carried forward from the prior baseline report, out of scope to fix here).

## 13. Existing Test Results

```
cd code && python -m pytest -q      (torch-gpu, python.exe invoked directly)
104 passed in 24.19s
```
Matches the expected baseline (104 passed, 0 failed, 0 skipped) exactly.

Carried forward from the prior baseline report (re-confirmed, not re-derived): a real
version mismatch exists between `code/requirements.txt`'s pinned versions (which target
Python ≥ 3.12) and the mandated `torch-gpu` environment (Python 3.10.20). Every module
imports and the full pipeline runs correctly on `torch-gpu`'s actually-installed
versions regardless. This is a documentation/reproducibility gap, not a data-correctness
issue, and is out of scope for this audit to fix.

## 14. Issues Found

| # | Severity | Stage | Evidence | Why it matters | Confirmed / suspected | Recommended next action |
|---|---|---|---|---|---|---|
| 1 | Low (informational) | Blocking (India recall) | §3: 12/15 sampled India blocking-FN pairs show transliteration divergence + address truncation acting jointly | Explains, with real examples, why India blocking recall trails US by ~4 points at every N in the existing baseline; not new degradation, but now has a named mechanism instead of a correlation | **Confirmed** (on this sample's FN text) | Not a code change in scope here; future work could add a phonetic-code (e.g. Soundex-style) block key computed from the transliterated form, or loosen the digit-pass exact-key join to tolerate a single-digit edit, as evidence-backed candidates — not implemented in this audit. |
| 2 | Low | Features | §5: `same_country` and `addr_empty_s1` are exactly constant | Two of 58 features carry no signal at current config; harmless to LightGBM (it will simply never split on them) but worth knowing they are dead weight, not misconfigured | Confirmed, and now explained structurally (blocking design + S1 data property, verified at full-population scale) rather than merely observed | No action required; documented as expected. |
| 3 | Low (design note) | Features (train/test consistency) | §11: `FeatureContext` is refit separately for train and test, so `*_tfidf_cos` features are computed in different vector spaces per split | Intentional per the module's own docstring (captures French n-grams), but is the one place "same code, different fitted state" could contribute to train/test drift beyond what the raw data causes | Confirmed as intentional design, not verified as calibrated (no labeled France data exists to check) | No action recommended without a labeled France sample; flagged for awareness only. |
| 4 | Low | Environment | §13 (carried forward) | `requirements.txt` pins Python≥3.12-only versions; `torch-gpu` runs older mutually-compatible versions | Reproducibility-on-paper gap, not a runtime failure | Confirmed (carried forward from prior audit, not re-derived here) | Reconcile `requirements.txt` with a Python-3.10-compatible pin set, or confirm intent to ship on Python≥3.12, before final submission — out of scope for this audit. |
| 5 | Data characteristic (not a defect) | Raw data | §1: 25,873 (train_s2) / 18,860 (train_s3) rows share identical (name,address,country) with a different entity_id | More than one candidate can be textually indistinguishable from a true match for some S1s | Confirmed | No action — real-world duplicate listings; the one-to-one decision layer already resolves conflicts by probability (§8), and this was already implicitly covered by the prior full-population ONE_TO_ONE=0-conflicts audit. |

No item above rises to CONFIRMED BUG.

## 15. Final Verdict

| Claim | Status |
|---|---|
| Raw file structure/prefixes/dtypes/missingness match documented contract | **VERIFIED** (full-file read, all 7 files) |
| France present only in test, at real volume, not filtered anywhere in code | **VERIFIED** (full-file counts + grep for country literals) |
| Normalisation is deterministic, null-safe, ID-preserving, country-agnostic | **VERIFIED** (real records, both directions checked) |
| Blocking candidate IDs all exist, no dup pairs, bounded cardinality | **VERIFIED** (real N=5,000 run) |
| India blocking-recall gap has a concrete text-level root cause | **VERIFIED** (15 sampled real FN pairs, 12 showing the named mechanism) |
| `candidate_pairs.tsv` set == exact model-input set (train) | **VERIFIED** (real run, set equality measured) |
| Same equality holds for the test split | **INFERRED** (identical code path; not independently re-run at 1.73M rows, out of scope) |
| Feature matrix: no inf, no all-NaN columns, constants explained | **VERIFIED** (real run + full-population cross-check for `addr_empty_s1`) |
| Label join is ID-based, count-consistent with blocking recall, no leakage in sample | **VERIFIED** (real run, exact-count match + 200-sample source check) |
| Model input/output row alignment, GroupKFold no-leak | **VERIFIED** (real run, data-level fold check, not just the existing unit test) |
| Decision layer: zero/singleton/multi/both-source S1s all handled; one-to-one holds at output | **VERIFIED** (real walkthrough on real S1 IDs from the sample) |
| `write_submission` enforcement logic is fail-closed and covers the France-in-test-only path | **VERIFIED** (via existing test suite read in full, not re-executed against real 1.73M-row files in this audit) |
| Full test-set (1.73M S1) output row-for-row correctness | **NOT VERIFIED IN THIS AUDIT** (explicitly out of scope — no full test inference was run; covered structurally by code + tests only) |
| `requirements.txt` fully reconciled with the execution environment | **NOT VERIFIED / KNOWN GAP** (carried forward, out of scope to fix here) |

**Overall: DATA PIPELINE VERIFIED.** Every stage transition that was in scope for a
real, executed check passed; the two items marked NOT VERIFIED were explicitly out of
scope by the audit's own constraints (no full test inference, no source changes), not
because a check was attempted and failed.
