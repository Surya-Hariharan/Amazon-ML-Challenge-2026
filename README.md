# Business Entity Resolution

> A multilingual entity-resolution system built for the **Amazon ML Challenge 2026**.

## Problem

Business records arrive from three independent sources that share no identifiers. Given a
deduplicated reference source, the task is to find every record in the other two sources
that describes the same real-world business. Names and addresses are noisy: typos,
transliteration, abbreviations, reordered fields and legal-suffix variants, across the
US, India and France (France appears only at test time). Scoring is a macro F0.5 per
reference entity, so precision counts double and correctly predicting "no match" earns full credit.

## Approach

```
raw records ─▶ normalise ─▶ block (5 passes) ─▶ features ─▶ LightGBM ─▶ decide ─▶ submission
```

- **Normalisation:** transliterates Indic scripts, strips accents, canonicalises legal
  suffixes and address abbreviations across English, Hindi/Indic and French, and splits
  landmarks from street text.
- **Blocking:** multi-pass candidate generation (character TF-IDF, multilingual
  embeddings, rare-token keys, digit-key matching, address-only matching) narrows
  ~11.7M records to a tractable candidate set per entity. Runs in bounded-memory chunks.
- **Matching:** a LightGBM classifier over ~58 similarity features (name, address,
  embedding and context signals), cross-validated with folds grouped by entity.
- **Decision:** one-to-one assignment, then a threshold tuned for the competition's
  macro F0.5 metric, with singletons counted.

## Tech stack

- Python 3.12, pandas, NumPy, SciPy
- LightGBM (pair classifier), scikit-learn (TF-IDF, cross-validation)
- rapidfuzz (string similarity)
- sentence-transformers and PyTorch (multilingual embeddings, GPU-accelerated top-k search)
- pyarrow (artifact caching), pytest

## Repository structure

```
README.md, LICENSE, CLAUDE.md, .gitignore, .gitattributes   — root: entry point, licence,
                                                                 AI/agent context, VCS config
code/
├── src/        normalize, blocking, features, model, decide, evaluate, run_pipeline
├── tests/      unit and end-to-end tests on synthetic data
├── artifacts/  cached embeddings, trained models, OOF preds — gitignored
├── README.md   full reproduction instructions
└── MODELS.md   licence and size of every pretrained model used
output/         matching_results.tsv and candidate_pairs.tsv (+ README.md)
docs/
├── challenge/     official problem statement + guidelines (verbatim from organizer PDFs)
├── methodology/   Documentation_template.md (the methodology write-up, filled in at CP12)
├── architecture/  pipeline.md — factual restatement of the pipeline design
├── experiments/   how experiment tracking works
└── planning/      plan.md — checkpoint plan and current status
dataset/        train/test TSVs, gitignored, pulled from S3 (see dataset/README.md)
utils/          validate_submission.py — organizer-provided, do not modify
experiments/    configs/, reports/ (tracked); logs/, results/ (gitignored)
```

> Note: the repo's `code/` directory is **flat** — `src/`, `tests/`, `artifacts/`,
> `README.md`, `MODELS.md`, and `requirements.txt` live directly under `code/`. The
> official submission zip still requires an internal `code/business_entity_resolution/`
> folder (see `docs/challenge/problem_statement.md`'s *Final Submission Package*
> section) — that folder name is applied only at packaging time, by copying `code/`
> into the zip under that name. See `CLAUDE.md` §4 and `docs/planning/plan.md` CP12 for
> the exact packaging commands.

`Documentation_template.md` is copied from `docs/methodology/` to the zip root when
assembling the final submission archive — see `CLAUDE.md` §4 and `docs/planning/plan.md`
CP12 for the exact packaging commands.

## Quick start

```bash
git clone https://github.com/Vishalspl-0903/tensortrio-.git && cd tensortrio-
pip install -r code/requirements.txt   # Python >= 3.12
cd code && python -m pytest -q          # synthetic-data tests
python -m src.run_pipeline --mode valid # needs the dataset
```

Data setup and every command are in
[code/README.md](code/README.md).

## Results

Results pending — validation in progress.

---

*Amazon ML Challenge 2026 · Business Entity Resolution*
