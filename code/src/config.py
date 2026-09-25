"""Single source of truth for paths, seeds, thresholds and k values.

Every other module imports its constants from here; nothing else hard-codes a path
or a tunable number. Paths are resolved relative to the repository root so the code
runs the same from a fresh clone on any machine.
"""

import os
from pathlib import Path

# --- Paths -------------------------------------------------------------------------

#: code/
CODE_DIR = Path(__file__).resolve().parents[1]
#: Repository root (<team>_submission/).
REPO_ROOT = CODE_DIR.parents[0]


def _default_data_dir() -> Path:
    """Return the local dataset directory.

    Honours the ``BER_DATA_DIR`` environment variable; otherwise uses ``dataset/`` at
    the repo root, falling back to ``student_resource/dataset/`` if only that exists.
    """
    env = os.environ.get("BER_DATA_DIR")
    if env:
        return Path(env).resolve()
    primary = REPO_ROOT / "dataset"
    fallback = REPO_ROOT / "student_resource" / "dataset"
    if not primary.exists() and fallback.exists():
        return fallback
    return primary


DATA_DIR = _default_data_dir()
TRAIN_DIR = DATA_DIR / "train"
TEST_DIR = DATA_DIR / "test"

TRAIN_FILES = {
    "s1": TRAIN_DIR / "train_source1.tsv",
    "s2": TRAIN_DIR / "train_source2.tsv",
    "s3": TRAIN_DIR / "train_source3.tsv",
    "ground_truth": TRAIN_DIR / "train_ground_truth.tsv",
}
TEST_FILES = {
    "s1": TEST_DIR / "test_source1.tsv",
    "s2": TEST_DIR / "test_source2.tsv",
    "s3": TEST_DIR / "test_source3.tsv",
}

OUTPUT_DIR = REPO_ROOT / "output"
MATCHING_FILE = OUTPUT_DIR / "matching_results.tsv"
CANDIDATE_FILE = OUTPUT_DIR / "candidate_pairs.tsv"

ARTIFACTS_DIR = CODE_DIR / "artifacts"
EXPERIMENTS_CSV = CODE_DIR / "experiments.csv"

# --- Data schema -------------------------------------------------------------------

ID_COL = "entity_id"
NAME_COL = "business_name"
ADDRESS_COL = "business_address"
COUNTRY_COL = "country"
SOURCE_COLUMNS = [ID_COL, NAME_COL, ADDRESS_COL, COUNTRY_COL]

S1_PREFIX = "S1-"
MATCH_PREFIXES = ("S2-", "S3-")

MATCHING_HEADER = ("source1_entity_id", "matched_entity_ids")
CANDIDATE_HEADER = ("source1_entity_id", "candidate_entity_ids")

# --- Reproducibility ---------------------------------------------------------------

SEED = 42

# --- Evaluation --------------------------------------------------------------------

BETA = 0.5
VALID_FRACTION = 0.2
N_FOLDS = 5

#: Fraction of train S1s used in a run (orphan density is preserved, see
#: run_pipeline.subsample_train). 1.0 = everything; lower it for fast iteration.
TRAIN_SAMPLE_FRAC = 1.0

# --- Blocking (tuned in CP3) -------------------------------------------------------

#: CP1 audit: zero cross-country true pairs in train, so blocking within the same
#: country string is safe. Country stays an open set (France flows through).
BLOCK_WITHIN_COUNTRY = True

#: Pass 1 — char 3-4-gram TF-IDF on name_core.
K_TFIDF_NAME = 20
TFIDF_NGRAM = (3, 4)
#: n-grams in more than this fraction of a group's records are dropped (bounds the
#: worst-case posting list). Only applied to groups of >= TFIDF_MAX_DF_MIN_DOCS
#: records; on small inputs a fractional cap would wipe out most of the vocabulary.
TFIDF_MAX_DF = 0.05
TFIDF_MAX_DF_MIN_DOCS = 100_000
TFIDF_MIN_DF = 2
#: Query pruning: each S1 keeps only its highest-weight (rarest) n-grams.
TFIDF_QUERY_TERMS = 12
#: Max stored entries of one chunk's sparse product. Chunks are sized adaptively from
#: posting-list lengths, so memory stays bounded at any scale (~12 bytes per entry).
TFIDF_MAX_PRODUCT_NNZ = 50_000_000

#: Pass 2 — dense multilingual embeddings of name + address (see MODELS.md).
USE_EMBEDDINGS = True
EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_REVISION = "e8f8c211226b894fcb81acc59f3b34ba3efd5f42"
EMBEDDING_BATCH = 512
K_EMBEDDING = 20
#: Query / index block sizes for the chunked exact top-k search.
DENSE_QUERY_CHUNK = 4096
DENSE_INDEX_CHUNK = 262_144

#: Pass 3 — shared rare name tokens.
K_RARE_TOKEN = 10
RARE_TOKENS_PER_RECORD = 3
#: A token shared by more than this many records (per country) is not a block key.
RARE_MAX_BLOCK = 200

#: Pass 4 — shared digit token + shared first name token.
K_POSTAL_TOKEN = 10
DIGIT_MAX_BLOCK = 200

#: Pass 5 — address-only key (digit sequence / digit + street word). Targets matches
#: whose names differ entirely (e.g. native-script names) and are decided by address.
USE_ADDRESS_PASS = True
K_ADDRESS = 10
ADDRESS_MAX_BLOCK = 100

#: S1 rows per chunk for the inverted-index passes (3-5).
KEY_PASS_CHUNK = 50_000

# --- Features (CP4) ----------------------------------------------------------------

#: Pairs per feature chunk (bounds peak memory of string features).
FEATURE_CHUNK = 1_000_000
#: Records sampled to fit the feature TF-IDF vocabularies / IDF.
FEATURE_TFIDF_FIT_SAMPLE = 2_000_000

# --- Model (CP4) -------------------------------------------------------------------

LGB_PARAMS = {
    "objective": "binary",
    "learning_rate": 0.05,
    "num_leaves": 127,
    "min_child_samples": 50,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambda_l2": 1.0,
    "verbose": -1,
    "seed": SEED,
    "deterministic": True,
    "num_threads": 0,
}
LGB_NUM_BOOST_ROUND = 2000
LGB_EARLY_STOPPING = 100

# --- Decision (tuned in CP4) -------------------------------------------------------

#: Default threshold until tuned on OOF. CLAUDE.md §6.5 expects tau > 0.5 because
#: ~25% of S2/S3 records are orphans (the main false-positive source).
MATCH_THRESHOLD = 0.6
#: Optional "S1 is a singleton" threshold on the S1's top probability (None = off).
SINGLETON_THRESHOLD = None
#: CP1 audit: no S2/S3 record matches more than one S1 in train -> enforce one-to-one.
ONE_TO_ONE = True
#: Threshold sweep grid used by decide.tune_threshold.
TAU_GRID = tuple(round(0.30 + 0.025 * i, 3) for i in range(27))  # 0.30 .. 0.95

#: Tunable values snapshotted into experiments.csv on every run.
TUNABLES = (
    "TRAIN_SAMPLE_FRAC", "BLOCK_WITHIN_COUNTRY", "K_TFIDF_NAME", "TFIDF_MAX_DF",
    "TFIDF_QUERY_TERMS",
    "USE_EMBEDDINGS", "K_EMBEDDING", "K_RARE_TOKEN", "RARE_MAX_BLOCK", "K_POSTAL_TOKEN",
    "DIGIT_MAX_BLOCK", "USE_ADDRESS_PASS", "K_ADDRESS", "ADDRESS_MAX_BLOCK",
    "MATCH_THRESHOLD", "SINGLETON_THRESHOLD", "ONE_TO_ONE",
)
