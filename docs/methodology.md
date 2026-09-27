# Methodology — Business Entity Resolution

Amazon ML Challenge 2026.

> This document is submitted as `Documentation_template.md` at the root of the final
> archive. Values marked *[final]* are filled in from the final full-scale run.

## Executive summary

For every Source 1 (S1) business record we retrieve a small set of plausible Source 2
and Source 3 records, score each pair with a gradient-boosted classifier, and keep only
confident, non-conflicting matches.

1. **Normalisation** turns noisy multilingual names and addresses into canonical
   fields: Indic scripts are transliterated to Latin, accents stripped, and legal
   suffixes, landmarks, house numbers and region codes separated out.
2. **Blocking** unions seven retrieval passes (character TF-IDF on name and on name +
   address, multilingual sentence embeddings, and four inverted-index key passes). On
   local validation it keeps **99.2% of true pairs** with about **46 candidates per
   S1** out of millions of records.
3. **Matching** uses a LightGBM classifier over about 60 name, address, number,
   embedding, blocking and context features, trained with folds grouped by S1 entity.
4. **Decision** enforces one-to-one assignment, then applies a match threshold (and,
   when it helps, a singleton threshold) tuned directly for macro F0.5 with singletons
   included.

Local validation macro F0.5: **0.990** (0.45% seeded sample of train, held-out 20%).
Leaderboard F0.5: *[final]*.

The pipeline uses only the provided data. Its two models are compliant:
`paraphrase-multilingual-MiniLM-L12-v2` (Apache-2.0, 118M parameters) and LightGBM
(MIT, trained from scratch).

## 1. Problem analysis

The task is to link each S1 record to every S2/S3 record describing the same business.
Macro F0.5 per S1 entity weights precision twice as heavily as recall and gives full
credit for correctly empty predictions, so false merges are the main risk.

Findings from the training data that shaped the design:

- **No cross-country matches and no shared S2/S3 records.** No true pair crosses
  countries and no S2/S3 record belongs to two S1 entities. This allows blocking within
  country groups and one-to-one assignment.
- **Many orphans.** About a quarter of S2/S3 records match no S1, so a candidate that
  looks similar is often still wrong.
- **Heavy name reuse.** 38% of S1 names are shared with another S1 (one name appears 253
  times), so the address must be as informative as the name.
- **Source-specific noise.** S2 is often upper case with native-script Indian place
  names; S3 adds honorifics, `X DBA Y` names, bracketed legal forms and shuffled address
  components. Both contain typos, digit/letter swaps and website-style names.
- **Unseen country at test time.** France (259,452 test S1 records) never appears in
  training, so every component must be language- and country-agnostic.

## 2. Methodology

### 2.1 Normalisation

All normalisation is deterministic string processing with hand-written dictionaries:

- Unicode NFKD with accent stripping (handles French accents), lower-casing, `&` → `and`,
  punctuation and junk-prefix removal.
- Transliteration of nine Indic scripts to Latin through one shared table, exploiting
  the parallel layout of the ISCII-derived Unicode blocks.
- Legal-form canonicalisation into a separate `name_suffix` field (English, Indian and
  French forms such as `pvt`, `ltd`, `llc`, `sarl`, `sas`, `eurl`), leaving `name_core`.
- Honorific, DBA-alias and website-name handling.
- Address abbreviation expansion (English and French), landmark-phrase extraction,
  digit-token, house-number and region-code extraction without country-specific rules.

### 2.2 Validation protocol

Train S1 entities are split 80/20, stratified on singleton vs non-singleton. The
threshold is tuned on five-fold out-of-fold predictions of the 80% part (folds grouped by
S1 ID) and the model is scored on the untouched 20%. Leave-one-country-out runs (train on
US, test on India and the reverse) estimate robustness to an unseen country.

## 3. Candidate generation (blocking)

Blocking bounds recall, so it received the most attention. Candidates are the union of
seven passes, each run within records sharing the same country string:

| # | Pass | Signal | Top-k |
| --- | --- | --- | --- |
| 1 | Name TF-IDF | Character 3–4-gram cosine on `name_core` | 20 |
| 2 | Dense embedding | Multilingual MiniLM cosine on name + address (exact fp32 search on GPU) | 20 |
| 3 | Rare name token | Shared high-IDF name token | 10 |
| 4 | Digit + first token | Shared digit run and first name token | 10 |
| 5 | Address key | `digit@street-word` keys over the first four digit runs | 10 |
| 6 | Street-word pairs | Pairs of the rarest address words (digit-free addresses) | 10 |
| 7 | Name + address TF-IDF | Character 3–4-gram cosine on name + address | 10 |

**How the pass set was chosen.** Error attribution on the original five passes showed
that lost matches were about five times more likely to be missing from the candidate set
than to be rejected by the model. They concentrated in Indian records, where
transliterations ("north constructions" vs "north kanstrakshans") defeat character
n-grams and alphanumeric house numbers ("24637B", "F-45D") lost their digits.
Counterfactual tests of new keys and representations led to passes 5–7 and to digit-run
extraction inside alphanumeric tokens:

| Configuration | Candidates / S1 | Pair recall | India recall |
| --- | --- | --- | --- |
| Original five passes | 40.0 | 97.7% | 95.2% |
| Current seven passes | 46.2 | 99.2% | 98.2% |

A density stress test showed that top-k similarity passes lose recall as the index grows
while key-based passes do not, which is why the new passes are mostly key-based or
address-aware. Raising k on similarity passes was about 100× less efficient per added
pair.

**Scale.** Every pass runs in bounded-memory chunks (adaptive sparse-product chunks,
blocked dense top-k, chunked key joins), so the same code handles the ~1.7M × ~10M test
problem. Full-scale blocking recall: *[final]*; candidate pairs: *[final]*.

The emitted `candidate_pairs.tsv` is exactly the set the model scores.

## 4. Model architecture and feature engineering

### 4.1 Features

About 60 features per pair. Missing information is `NaN`, never zero similarity.

- **Name similarity:** Jaro-Winkler, Levenshtein ratio, token-set, token-sort and
  partial ratios (rapidfuzz), space-free variants, character TF-IDF cosine, token
  Jaccard, length ratio.
- **Name identity:** exact `name_core` and compact-name match, acronym match in either
  direction, first-token equality, legal-suffix match or conflict, website and DBA flags.
- **Address:** TF-IDF cosine, token Jaccard, landmark Jaccard, empty-address flags.
- **Numbers and region:** digit Jaccard, shared-digit count, digit conflict, digit
  subset, house-number and long-number equality, region equality. Conflict features let
  the model reject same-name, different-place pairs.
- **Embedding:** cosine of multilingual name + address embeddings.
- **Blocking:** which passes found the pair, their scores and the number of passes.
- **Context:** rank and gap-to-best within the S1's candidates; reverse rank and gap
  among S1 entities competing for the same candidate; candidate counts; candidate source.
  These are the main defence against orphan records.
- **Country:** only `same_country`; no country identity features, so nothing is learnt
  that could not transfer to France.

### 4.2 Classifier

LightGBM binary classifier (127 leaves, learning rate 0.05, feature and bagging fraction
0.8, L2 = 1, up to 2,000 rounds with early stopping, deterministic mode, seed 42).
Five-fold `GroupKFold` by S1 ID produces out-of-fold predictions for threshold tuning;
the final model is retrained on all training pairs.

### 4.3 Decision

1. **One-to-one:** each S2/S3 record is kept only under its highest-probability S1. On
   validation this removed at most 2 of ~7,000 true matches while resolving thousands
   of candidates claimed by several S1 entities.
2. **Match threshold τ:** swept over 0.30–0.95 on OOF predictions to maximise macro
   F0.5 with singletons included. With the original five passes τ settled at 0.70–0.78;
   with the current richer candidate set and features the model is better separated and
   τ settles at 0.425.
3. **Singleton threshold:** tuned jointly with τ; S1 entities whose best probability is
   below it get an empty list. The current configuration selects no singleton threshold
   because τ alone performs best.

## 5. Results

| Metric (local validation, held-out 20%) | Value |
| --- | --- |
| Macro F0.5 | 0.990 |
| Pair precision / recall | 0.994 / 0.988 |
| Blocking pair recall | 0.992 |
| Candidates per S1 | 46.2 |
| Public leaderboard F0.5 | *[final]* |

Full experiment history: [results.md](results.md).

## 6. Other relevant information

### Generalisation to France

- No stage branches on country; blocking groups by the raw country string, so France
  forms its own group automatically.
- Normalisation covers French accents, legal forms (`sarl`, `sas`, `sasu`, `eurl`, `sci`,
  `snc`) and street words (`rue`, `chemin`, `impasse`, `allée`, `boulevard`).
- The multilingual embedding model covers French.
- Leave-one-country-out validation measures how much accuracy transfers to an unseen
  country: *[final]*.

### Compliance

- **No external data:** no APIs, registries, geocoding or web data; only the provided
  files and hand-written dictionaries.
- **Models:** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (Apache-2.0,
  117.7M parameters, pinned revision) and LightGBM 4.7.0 (MIT, trained from scratch).
  Details in `code/MODELS.md`.

### Reproducibility

A single seed controls sampling, folds and training; dependencies are pinned; the
embedding model revision is pinned. `code/README.md` gives the exact commands from raw
data to the two output files.
