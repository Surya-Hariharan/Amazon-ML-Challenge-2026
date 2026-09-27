# Business Entity Resolution

![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue)
![LightGBM](https://img.shields.io/badge/model-LightGBM-green)
![License: MIT](https://img.shields.io/badge/license-MIT-lightgrey)

A multilingual entity-resolution pipeline built for the **Amazon ML Challenge 2026**.
It links business records across three independent sources that share no identifiers,
at the scale of millions of records, across the US, India and an unseen third country,
France.

## The problem

Source 1 is a deduplicated reference list of businesses. For every Source 1 record, the
task is to find every Source 2 and Source 3 record describing the same real-world
business: zero, one or many. Names and addresses are noisy, with typos, abbreviations,
transliterated Indic scripts, legal-suffix variants, landmark-based addresses and
reordered fields.

Submissions are scored by **macro F0.5 per Source 1 entity**, singletons included:
precision counts twice as much as recall, and correctly predicting "no match" earns full
credit. Full details: [problem statement](docs/problem-statement.md).

## Approach

```text
raw records ─▶ normalise ─▶ block (7 passes) ─▶ features (~60) ─▶ LightGBM ─▶ decide ─▶ submission
```

| Stage | What it does |
| --- | --- |
| **Normalise** | Transliterates Indic scripts, strips accents, canonicalises legal suffixes and address abbreviations (English, Indian, French), and separates landmarks, house numbers and region codes |
| **Block** | Unions seven retrieval passes (name and name + address character TF-IDF, multilingual embeddings, and four inverted-index key passes), narrowing millions of possible records to about 46 candidates each |
| **Match** | Scores each pair with LightGBM over name, address, number, embedding, blocking and context features, cross-validated with folds grouped by entity |
| **Decide** | Enforces one-to-one assignment, then applies thresholds tuned directly for macro F0.5 with singletons |

Everything runs in bounded-memory chunks with GPU-accelerated exact top-k search, uses
only the provided data, and treats country as an open set, so the unseen French split
flows through every stage unchanged. See [architecture](docs/architecture.md) and
[methodology](docs/methodology.md).

## Results

Local validation on a seeded sample of the training data (held-out 20% of Source 1
entities):

| Metric | Value |
| --- | --- |
| Macro F0.5 | **0.990** |
| Pair precision / recall | 0.994 / 0.988 |
| Blocking recall (true pairs retained) | 0.992 |
| Candidates per Source 1 record | 46 |

Blocking changes driven by error analysis raised Indian-record recall from 95.2% to
98.2%. Experiment history: [results](docs/results.md).

## Quick start

```bash
git clone https://github.com/Surya-Hariharan/Amazon-ML-Challenge-2026.git
cd Amazon-ML-Challenge-2026
pip install -r code/requirements.txt       # Python >= 3.12

cd code
python -m pytest -q                        # synthetic-data test suite, no dataset needed
python -m src.run_pipeline --mode valid    # local validation (needs the dataset)
python -m src.run_pipeline --mode test     # writes output/*.tsv
```

The dataset is not included; see [dataset/README.md](dataset/README.md) and the
[development guide](docs/development.md) for setup.

## Repository layout

```text
.
├── code/                  Pipeline package (submitted as code/business_entity_resolution/)
│   ├── src/               normalize, blocking, features, model, decide, evaluate, run_pipeline
│   ├── tests/             Unit and end-to-end tests on synthetic data
│   ├── README.md          Reproduction instructions
│   ├── MODELS.md          Licence and size of every model used
│   └── requirements.txt   Pinned dependencies
├── docs/                  Architecture, methodology, results and guides
├── experiments/scripts/   Standalone experiment drivers
├── output/                Submission files (generated)
├── dataset/               Train and test data (not versioned)
└── utils/                 Organizer-provided submission validator
```

## Documentation

| Document | Contents |
| --- | --- |
| [Architecture](docs/architecture.md) | Stage-by-stage pipeline design |
| [Methodology](docs/methodology.md) | Full methodology write-up |
| [Results](docs/results.md) | Validation results and experiment findings |
| [Development guide](docs/development.md) | Setup, CLI reference, diagnostics, conventions |
| [Submission guide](docs/submission.md) | Validation and packaging |

## Compliance

- **No external data.** No entity-resolution services, registries, geocoding or web
  data; only the provided files and hand-written normalisation dictionaries.
- **Model licences.** `paraphrase-multilingual-MiniLM-L12-v2` (Apache-2.0, 118M
  parameters) and LightGBM (MIT, trained from scratch), both within the MIT/Apache-2.0,
  ≤ 8B-parameter rule. See [MODELS.md](code/MODELS.md).

## License

[MIT](LICENSE)
