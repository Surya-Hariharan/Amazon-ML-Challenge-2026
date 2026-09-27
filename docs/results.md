# Results

All numbers below are **local validation results on the training data**: a
deterministic, seeded sample of S1 entities, split 80/20 by S1 ID, with the threshold
tuned on out-of-fold predictions of the 80% and the score reported on the held-out 20%.
The test set has no labels, so these numbers are not leaderboard scores. Treat them as
the relative signal used to accept or reject changes.

## Dataset

| Split | Source 1 | Source 2 | Source 3 | Labels |
| --- | --- | --- | --- | --- |
| Train | 2,206,821 | 5,034,616 | 5,285,603 | 2,206,821 rows |
| Test | 1,732,544 | 4,887,273 | 5,082,316 | none |

- Train S1 by country: US 1,323,633 · India 883,188. No France records in train.
- Test S1 by country: US 663,106 · India 809,986 · **France 259,452**.
- 5.6% of train S1 entities are singletons (no true match); most have several matches
  (mean ≈ 3.5, max 10).
- No S2/S3 record is matched to more than one S1, and no true pair crosses countries.
  These two facts justify one-to-one assignment and within-country blocking.

## Current configuration

Measured on a 0.45% sample (9,931 S1 entities, 46,601 S2+S3 records, 7,012 true pairs
in the validation split).

| Metric | Value |
| --- | --- |
| Blocking pair recall | 0.9915 (US 0.9979 · India 0.9819) |
| Mean candidates per S1 | 46.2 |
| Validation macro F0.5 | **0.9901** (OOF 0.9920) |
| Pair precision / recall | 0.9940 / 0.9877 |
| Tuned thresholds | τ = 0.425, no singleton threshold |

## Blocking improvements

Error attribution showed that most lost matches were never retrieved (blocking false
negatives outnumbered matcher false negatives roughly 5 to 1), concentrated in Indian
records with transliterated names and alphanumeric house numbers. Each row below adds
one change to the original five-pass baseline, on the same sample:

| Variant | Cands/S1 | Pair recall | India recall | Valid F0.5 |
| --- | --- | --- | --- | --- |
| Five-pass baseline | 40.0 | 0.9769 | 0.9521 | 0.9866 |
| + address keys on all digit runs | 40.2 | 0.9810 | 0.9615 | 0.9879 |
| + digit runs inside alphanumeric tokens | 40.1 | 0.9782 | 0.9550 | 0.9875 |
| + street-word pair pass | 40.9 | 0.9864 | 0.9711 | 0.9907 |
| + name + address TF-IDF pass (k = 10) | 45.2 | 0.9903 | 0.9798 | 0.9912 |
| **All changes combined (k = 10), current** | 46.2 | **0.9915** | **0.9819** | 0.9901 |
| All changes combined (k = 20) | 54.1 | 0.9928 | 0.9848 | 0.9924 |

The k = 10 variant was chosen over k = 20 because candidate-pair memory is the binding
constraint at full test scale; the recall difference is 0.13 points.

## Baseline scaling study

The original five-pass pipeline on growing S1 samples:

| S1 sample | Pair recall | Cands/S1 | Reduction ratio | Valid F0.5 | τ |
| --- | --- | --- | --- | --- | --- |
| 500 | 0.9906 | 31.3 | 98.64% | 0.9929 | 0.900 |
| 2,000 | 0.9864 | 36.2 | 99.61% | 0.9938 | 0.550 |
| 5,000 | 0.9812 | 38.7 | 99.84% | 0.9900 | 0.700 |
| 10,000 | 0.9769 | 40.0 | 99.91% | 0.9871 | 0.750 |

Recall falls as the index grows because top-k similarity passes face more competitors.
A density stress test confirmed this: key-based passes stay flat as the index grows,
while name TF-IDF, rare-token and embedding recall decay. This motivated the key-based
and address-aware passes above.

## Error budget (baseline, 10,000 S1)

| True pairs (validation) | Missed | Lost at blocking | Below threshold | Removed by one-to-one |
| --- | --- | --- | --- | --- |
| 6,936 | 214 | 178 (83%) | 36 (17%) | 0 |

One-to-one assignment removed no true matches in this study and at most 2 of ~7,000 in
later runs, so it is an almost free precision gain.

## Key findings

- **Blocking is the bottleneck.** Most missed matches are never retrieved; increasing k
  on similarity passes is about 100× less efficient per added pair than adding new
  retrieval keys or richer representations.
- **Non-Latin scripts are the hardest slice.** Records with any non-Latin text hold 21%
  of true pairs but 59% of lost pairs. Transliterations such as "north constructions" vs
  "north kanstrakshans" defeat character n-grams, so address-aware passes do the work.
- **Indian PIN codes are almost absent** (6-digit numbers appear in 0.02% of India S2
  addresses), so postal keys cannot help there; house-number digit runs can.
- **Precision stays near its ceiling** (≈ 0.995) across every configuration; recall is
  the lever.
- **Singletons are the cost of recall.** The larger candidate set lowered singleton
  F0.5 from 0.991 to 0.956 on this sample while raising non-singleton accuracy; this is
  the main remaining target for decision-stage tuning.

## Open questions

- Full-scale behaviour: recall at the full ~10M-record index is extrapolated
  (≈ 97% with the current passes), not yet measured.
- France: no labelled French data exists. Leave-one-country-out validation
  (`--loco`) is the proxy for unseen-country generalisation.

## Reproducing these numbers

```bash
cd code
python -m src.run_pipeline --mode valid --sample 0.0045   # current configuration
python -m src.diagnose blocking-ablation --sample 0.0045  # per-pass and k sweeps
python -m src.diagnose errors --sample 0.0045             # error attribution
```

Every run appends a row to `code/experiments.csv`; diagnostic outputs are written to
`code/artifacts/diagnostics/`.
