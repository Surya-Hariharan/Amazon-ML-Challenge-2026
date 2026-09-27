# Architecture

This document describes how the pipeline turns three raw record sources into the two
submission files. Every stage is one module under [`code/src/`](../code/src/), and every
tunable lives in [`config.py`](../code/src/config.py).

```text
raw TSVs ─▶ normalise ─▶ block (7-pass union) ─▶ features ─▶ LightGBM ─▶ decide ─▶ output/
```

## Design principles

- **Precision first.** The metric is macro F0.5 per Source 1 (S1) entity, singletons
  included, so a false merge costs more than a missed link. Every decision stage is
  tuned against that exact metric.
- **Country is an open set.** Train covers US and India; test adds France. No stage
  branches on a country value, filters to known countries or one-hot encodes country.
  Blocking groups records by the raw country string, so an unseen country simply forms
  its own group.
- **Bounded memory at full scale.** Test is ~1.7M S1 records against ~10M S2/S3
  records. Every heavy step (sparse products, dense top-k, key joins, feature batches)
  runs in fixed-size chunks.
- **No external data.** Only the provided train/test files are used. Normalisation
  relies on hand-written dictionaries (abbreviations, legal suffixes), which are code,
  not data lookup.
- **Reproducible.** A single seed (`config.SEED`) drives sampling, folds and LightGBM;
  expensive intermediates are cached under `code/artifacts/`, keyed by input hash.

## Module map

| Stage | Module | Responsibility |
| --- | --- | --- |
| Configuration | [`config.py`](../code/src/config.py) | Paths, seed, k values, thresholds, chunk sizes, model parameters |
| I/O | [`io_utils.py`](../code/src/io_utils.py) | TSV read/write with string IDs; `write_submission()` enforces every output-format rule |
| Normalisation | [`normalize.py`](../code/src/normalize.py) | Canonical name/address fields, tokens, digits, acronyms |
| Blocking | [`blocking.py`](../code/src/blocking.py) | Multi-pass candidate generation and blocking metrics |
| Features | [`features.py`](../code/src/features.py) | Pairwise similarity, blocking and context features |
| Model | [`model.py`](../code/src/model.py) | LightGBM training, grouped cross-validation, out-of-fold predictions |
| Decision | [`decide.py`](../code/src/decide.py) | One-to-one assignment, threshold tuning, final match lists |
| Evaluation | [`evaluate.py`](../code/src/evaluate.py) | Macro F0.5, blocking recall, reduction ratio |
| Entry point | [`run_pipeline.py`](../code/src/run_pipeline.py) | `--mode valid` (local scoring) and `--mode test` (submission) |
| Diagnostics | [`diagnose.py`](../code/src/diagnose.py), [`diagnostics.py`](../code/src/diagnostics.py), [`drift.py`](../code/src/drift.py), [`error_decomposition.py`](../code/src/error_decomposition.py) | Measurement-only tooling: ablations, error attribution, drift, resource profiling |
| Storage | [`s3_sync.py`](../code/src/s3_sync.py) | Opt-in dataset download and artifact/submission upload to S3 |

## 1. Normalisation

Noise in the data is heavy and source-specific: S1 is clean title case, S2 is often
upper case with Indian place names in native Indic scripts, and S3 adds honorifics,
`X DBA Y` forms and shuffled address components. `normalize_frame()` produces the
following fields for every record:

| Field | Content |
| --- | --- |
| `name_norm` | Transliterated (all nine ISCII-derived Indic scripts share one table), NFKD accent-stripped, lower-cased, `&` → `and`, punctuation and junk prefixes removed |
| `name_core` / `name_suffix` | Legal form split off and canonicalised (English, Indian and French forms: `ltd`, `pvt`, `llc`, `inc`, `sarl`, `sas`, `eurl`, …) |
| `name_alias`, `is_website` | DBA / trade-name alias and website-style names (`dynamicsmarketing.com`) |
| `acronym`, `first_token` | Acronym of `name_core`; first core token |
| `addr_norm` | Address with abbreviations expanded (English and French: `rd`, `st`, `ave`, `bd`, `chem`, `imp`, …) |
| `landmark` | Landmark phrases (`near`, `opp`, `behind`, `pres de`, …) split out so they do not pollute street similarity |
| `digits`, `house_no`, `region` | Generic digit tokens, house number and region code, extracted without country-specific rules |

## 2. Blocking

Blocking sets the recall ceiling: a true match missed here can never be recovered. The
candidate set is the union of seven passes, each run within a country group:

| # | Pass | Key / similarity | Per-S1 limit |
| --- | --- | --- | --- |
| 1 | Name TF-IDF | Character 3–4-gram TF-IDF cosine on `name_core` | `K_TFIDF_NAME` = 20 |
| 2 | Dense embedding | Cosine on `paraphrase-multilingual-MiniLM-L12-v2` embeddings of name + address (fp32, GPU when available) | `K_EMBEDDING` = 20 |
| 3 | Rare name token | Shared high-IDF name token | `K_RARE_TOKEN` = 10 |
| 4 | Digit + first token | Shared digit run plus shared first name token | `K_POSTAL_TOKEN` = 10 |
| 5 | Address key | `digit@street-word` keys over the first four digit runs | `K_ADDRESS` = 10 |
| 6 | Street-word pairs | Unordered pairs of the rarest alphabetic address words (covers digit-free addresses) | `K_STREET` = 10 |
| 7 | Name + address TF-IDF | Character 3–4-gram TF-IDF on `name_core + addr_norm` | `K_TFIDF_ADDR` = 10 |

Each candidate records a bitmask of the passes that produced it plus each pass's score;
both become model features. Passes 5–7 were added after error attribution showed that
name-only retrieval misses transliterated names and degrades as the index grows, while
key-based passes stay stable (see [results.md](results.md)).

Memory is bounded by `TFIDF_MAX_PRODUCT_NNZ` (sparse-product chunk size),
`DENSE_QUERY_CHUNK` × `DENSE_INDEX_CHUNK` (dense top-k working set) and
`KEY_PASS_CHUNK` (inverted-index joins).

## 3. Features

About 60 numeric features per (S1, candidate) pair. Missing information is encoded as
`NaN`, never as zero similarity, so the model can tell "unknown" from "different".

| Family | Features |
| --- | --- |
| Name similarity | Jaro-Winkler, ratio, token-set, token-sort, partial ratio, compact (space-free) ratio/partial, TF-IDF cosine, token Jaccard, length ratio |
| Name identity | `name_core` exact, compact exact, acronym match (either direction), first-token equal, legal-suffix match/conflict, candidate is website, candidate has DBA alias |
| Address | TF-IDF cosine, token Jaccard, landmark Jaccard, empty-address flags |
| Numbers and region | Digit Jaccard, shared digit count, digit conflict, digit subset, house-number equal, long-number (postal) equal, region equal |
| Embedding | Cosine of the dense name + address embeddings |
| Blocking | Per-pass membership flags and scores, number of passes that found the pair |
| Context | Rank and gap-to-best within the S1's candidates, reverse rank and gap among S1s competing for the same candidate, candidate counts, candidate source (S2 vs S3) |
| Country | `same_country` only — no country identity features |

Context features are the main defence against the ~25% of S2/S3 records that have no
S1 parent: an orphan tends to be a weak, contested candidate everywhere it appears.

## 4. Model

A LightGBM binary classifier (MIT licence, trained from scratch) scores every candidate
pair. Cross-validation uses `GroupKFold(5)` grouped by S1 ID, so an S1 entity never
appears in both a training and a validation fold. Out-of-fold (OOF) predictions feed
threshold tuning; the final model is retrained on all training pairs. Parameters are in
`config.LGB_PARAMS` (127 leaves, learning rate 0.05, early stopping on 100 rounds,
deterministic mode).

## 5. Decision

1. **One-to-one assignment.** Train ground truth never assigns an S2/S3 record to more
   than one S1, so each candidate is kept only under the S1 where its probability is
   highest.
2. **Match threshold τ.** Pairs with probability ≥ τ are kept. τ is swept over
   0.30–0.95 on OOF predictions to maximise macro F0.5 *including singletons*.
3. **Singleton threshold.** Tuned jointly with τ (and dropped when it does not help):
   an S1 whose best probability falls below it receives an empty list, protecting the
   full 1.0 credit that singletons earn.

## 6. Output

`write_submission()` writes `output/matching_results.tsv` and
`output/candidate_pairs.tsv`, and refuses to write a file that would break a format
rule: one row per test S1, no duplicate IDs, S2-/S3- IDs only, and every match present in
that S1's candidate list. `candidate_pairs.tsv` is exactly the pair set the model scored.

## Data flow and storage

```text
dataset/{train,test}/*.tsv        input (gitignored; see dataset/README.md)
        │
        ▼
code/src/                         pipeline
        ├─▶ code/artifacts/       caches: normalised frames, embeddings, candidates,
        │                         trained model, error dumps (gitignored)
        ├─▶ code/experiments.csv  one row of metrics per run (gitignored)
        └─▶ output/*.tsv          submission files
```

Local runs never touch the network. Moving data to or from S3 is always an explicit
call into `s3_sync.py` (`download_dataset`, `upload_artifacts`, `upload_experiments`,
`upload_submissions`); `run_pipeline.py` does not import it. See
[development.md](development.md#cloud-workflow-sagemaker--s3).
