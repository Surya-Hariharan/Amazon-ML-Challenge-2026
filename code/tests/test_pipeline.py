"""End-to-end tests of src.run_pipeline on synthetic data (no real dataset needed)."""

import gc
import weakref

import pandas as pd
import pytest

from src import config
from src import run_pipeline as rp
from src.io_utils import read_tsv
from tests.synthetic import make_dataset, write_split
from tests.test_blocking import fake_encoder

FAST_LGB = {"objective": "binary", "learning_rate": 0.1, "num_leaves": 15,
            "min_child_samples": 5, "verbose": -1, "seed": 42, "deterministic": True,
            "num_threads": 1}


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    """Send artifacts/experiments to tmp and use a small, fast LightGBM config."""
    monkeypatch.setattr(config, "ARTIFACTS_DIR", tmp_path / "artifacts")
    monkeypatch.setattr(config, "EXPERIMENTS_CSV", tmp_path / "experiments.csv")
    monkeypatch.setattr(config, "LGB_PARAMS", FAST_LGB)
    monkeypatch.setattr(config, "LGB_NUM_BOOST_ROUND", 200)
    monkeypatch.setattr(config, "LGB_EARLY_STOPPING", 20)
    monkeypatch.setattr(config, "N_FOLDS", 3)
    return tmp_path


def test_prepare_normalises_s2_s3_independently_and_matches_combined(isolated):
    """prepare()'s per-source normalisation is equivalent to normalising the
    concatenated raw frame, and caches S2/S3 separately (not as one "others" blob)."""
    from src.normalize import normalize_frame

    s1, s2, s3, _ = make_dataset(n_s1=120, seed=11)
    prep = rp.prepare(s1, s2, s3, use_embeddings=False, use_cache=True)
    on = prep["others"]

    # Row count is preserved and matches the raw S2+S3 total.
    assert len(on) == len(s2) + len(s3)

    # Content is identical to the old "concat raw, then normalise" behaviour.
    expected = normalize_frame(pd.concat([s2, s3], ignore_index=True))
    pd.testing.assert_frame_equal(on.reset_index(drop=True), expected.reset_index(drop=True))

    # S2 and S3 were cached independently, not as a single combined "others" artifact.
    arts = isolated / "artifacts"
    cached = {p.name for p in arts.glob("norm_*.parquet")}
    assert any(n.startswith("norm_s2_") for n in cached)
    assert any(n.startswith("norm_s3_") for n in cached)
    assert not any(n.startswith("norm_others_") for n in cached)

    # A second call hits the S2/S3 caches independently rather than recomputing.
    calls = []
    real_normalize_frame = rp.normalize_frame

    def counting_normalize_frame(df):
        calls.append(len(df))
        return real_normalize_frame(df)

    rp.normalize_frame = counting_normalize_frame
    try:
        rp.prepare(s1, s2, s3, use_embeddings=False, use_cache=True)
    finally:
        rp.normalize_frame = real_normalize_frame
    assert calls == [], f"expected full cache hit, but normalize_frame was called: {calls}"


def test_prepare_drops_intermediate_s2n_s3n_after_concat(isolated, monkeypatch):
    """Regression test for the full-scale OOM fix: prepare() releases s2n/s3n (each
    as large as `others`) as soon as they are concatenated into it, instead of
    keeping both the pieces and the merged frame alive until the function returns."""
    s1, s2, s3, _ = make_dataset(n_s1=80, seed=30)
    live: list[weakref.ReferenceType] = []
    real_concat = pd.concat

    def tracking_concat(frames, **kwargs):
        result = real_concat(frames, **kwargs)
        live.extend(weakref.ref(f) for f in frames)
        return result

    monkeypatch.setattr(pd, "concat", tracking_concat)
    prep = rp.prepare(s1, s2, s3, use_embeddings=False, use_cache=True)
    gc.collect()
    assert all(r() is None for r in live), "s2n/s3n were not released after concat"
    assert len(prep["others"]) == len(s2) + len(s3)


def test_run_test_frees_raw_train_before_loading_test_split(isolated, monkeypatch):
    """Regression test for the full-scale OOM fix: run_test() must fully finish
    training on the train split -- releasing every raw/normalised training
    intermediate -- *before* it even loads the test split, rather than holding
    both splits in memory for the whole run (the actual full-scale OOM: both
    were loaded up front and test_run() itself cannot free one mid-call, since
    Python keeps a reference on the caller's stack for as long as a call is in
    flight, regardless of any `del` the callee runs -- see test_run's docstring).
    run_test() avoids that by calling the training half and _predict_test() as
    two separate, sequential top-level calls instead of going through a single
    test_run() call. At the default sample (>= 1.0), the training half is
    _train_model_from_files(), which loads S1/S2/S3 sequentially from disk
    rather than through _load_train() -- see the dedicated sequential-loading
    tests below for that part of the fix."""
    data = isolated / "dataset"
    s1, s2, s3, truth = make_dataset(n_s1=100, seed=40)
    write_split(data, "train", s1, s2, s3, truth)
    t1, t2, t3, _ = make_dataset(n_s1=70, seed=41)
    write_split(data, "test", t1, t2, t3)
    monkeypatch.setattr(config, "TRAIN_FILES", {
        "s1": data / "train/train_source1.tsv", "s2": data / "train/train_source2.tsv",
        "s3": data / "train/train_source3.tsv",
        "ground_truth": data / "train/train_ground_truth.tsv"})
    monkeypatch.setattr(config, "TEST_FILES", {
        "s1": data / "test/test_source1.tsv", "s2": data / "test/test_source2.tsv",
        "s3": data / "test/test_source3.tsv"})
    monkeypatch.setattr(config, "OUTPUT_DIR", isolated / "output")

    live: list[weakref.ReferenceType] = []
    real_read_tsv = rp.read_tsv

    def tracking_read_tsv(path):
        """Read as usual, but keep a weakref to every raw frame loaded from train/."""
        df = real_read_tsv(path)
        if "train" in str(path):
            live.append(weakref.ref(df))
        return df

    real_load_test = rp._load_test

    def tracking_load_test():
        """Before loading test, confirm every raw train frame is already collectible."""
        gc.collect()
        assert all(r() is None for r in live), "a raw train frame was still alive"
        return real_load_test()

    monkeypatch.setattr(rp, "read_tsv", tracking_read_tsv)
    monkeypatch.setattr(rp, "_load_test", tracking_load_test)
    rp.run_test(use_embeddings=False)


def test_prepare_from_files_never_holds_more_than_one_raw_source_at_once(isolated, monkeypatch):
    """Regression test for the full-scale OOM fix: prepare_from_files() must load,
    normalise and release S1, then S2, then S3 strictly one at a time -- never with
    more than one raw source resident -- instead of prepare()'s contract, which
    requires the caller to already hold all three raw frames simultaneously
    before it can even be called."""
    data = isolated / "dataset"
    s1, s2, s3, _ = make_dataset(n_s1=90, seed=50)
    write_split(data, "train", s1, s2, s3)
    files = {"s1": data / "train/train_source1.tsv", "s2": data / "train/train_source2.tsv",
            "s3": data / "train/train_source3.tsv"}

    live: dict[str, weakref.ReferenceType] = {}
    real_read_tsv = rp.read_tsv

    def tracking_read_tsv(path):
        """Before loading a new raw source, confirm every earlier one is already gone."""
        gc.collect()
        assert all(r() is None for r in live.values()), \
            f"raw source(s) still alive when loading {path}: {list(live)}"
        df = real_read_tsv(path)
        live[str(path)] = weakref.ref(df)
        return df

    monkeypatch.setattr(rp, "read_tsv", tracking_read_tsv)
    prep = rp.prepare_from_files(files, use_embeddings=False, use_cache=True)
    assert len(prep["others"]) == len(s2) + len(s3)
    assert len(prep["s1"]) == len(s1)


def test_prepare_from_files_matches_prepare_content(isolated):
    """prepare_from_files() (sequential, from disk) produces identical normalised
    content and row order to prepare() (in-memory), for the same underlying data."""
    data = isolated / "dataset"
    s1, s2, s3, _ = make_dataset(n_s1=70, seed=51)
    write_split(data, "train", s1, s2, s3)
    files = {"s1": data / "train/train_source1.tsv", "s2": data / "train/train_source2.tsv",
            "s3": data / "train/train_source3.tsv"}

    expected = rp.prepare(s1, s2, s3, use_embeddings=False, use_cache=False)
    got = rp.prepare_from_files(files, use_embeddings=False, use_cache=False)
    pd.testing.assert_frame_equal(got["s1"].reset_index(drop=True),
                                  expected["s1"].reset_index(drop=True))
    pd.testing.assert_frame_equal(got["others"].reset_index(drop=True),
                                  expected["others"].reset_index(drop=True))


def test_prepare_from_files_reuses_prepare_cache(isolated):
    """A normalisation cache built by prepare() is reused by prepare_from_files()
    (and vice versa): they share the same cache keys/paths, keyed on content."""
    data = isolated / "dataset"
    s1, s2, s3, _ = make_dataset(n_s1=60, seed=52)
    write_split(data, "train", s1, s2, s3)
    files = {"s1": data / "train/train_source1.tsv", "s2": data / "train/train_source2.tsv",
            "s3": data / "train/train_source3.tsv"}

    rp.prepare(s1, s2, s3, use_embeddings=False, use_cache=True)
    arts = isolated / "artifacts"
    cached_before = {p.name for p in arts.glob("norm_*.parquet")}
    assert cached_before, "prepare() should have written normalisation caches"

    calls = []
    real_normalize_frame = rp.normalize_frame

    def counting_normalize_frame(df):
        calls.append(len(df))
        return real_normalize_frame(df)

    rp.normalize_frame = counting_normalize_frame
    try:
        rp.prepare_from_files(files, use_embeddings=False, use_cache=True)
    finally:
        rp.normalize_frame = real_normalize_frame
    assert calls == [], f"expected a full cache hit, but normalize_frame was called: {calls}"
    assert {p.name for p in arts.glob("norm_*.parquet")} == cached_before


def test_normalize_one_loads_normalises_and_caches_a_single_source(isolated):
    """_normalize_one() reads one file, normalises it and caches it under the
    same key prepare() would use for that content -- the primitive
    prepare_from_files() is built from."""
    from src.normalize import normalize_frame

    data = isolated / "dataset"
    s1, _, _, _ = make_dataset(n_s1=40, seed=53)
    write_split(data, "train", s1, s1.iloc[:0], s1.iloc[:0])
    path = data / "train/train_source1.tsv"

    got = rp._normalize_one("s1", path, use_cache=True)
    expected = normalize_frame(rp.read_tsv(path))
    pd.testing.assert_frame_equal(got.reset_index(drop=True), expected.reset_index(drop=True))

    arts = isolated / "artifacts"
    cached = list(arts.glob("norm_s1_*.parquet"))
    assert len(cached) == 1

    # A second call hits the cache instead of recomputing.
    calls = []
    real_normalize_frame = rp.normalize_frame

    def counting_normalize_frame(df):
        calls.append(len(df))
        return real_normalize_frame(df)

    rp.normalize_frame = counting_normalize_frame
    try:
        rp._normalize_one("s1", path, use_cache=True)
    finally:
        rp.normalize_frame = real_normalize_frame
    assert calls == []


def test_split_s1_stratified_and_disjoint():
    """80/20 split keeps the singleton share and never overlaps."""
    truth = {f"S1-{i}": ([] if i % 10 == 0 else ["S2-x"]) for i in range(200)}
    tr, va = rp.split_s1(list(truth), truth)
    assert not set(tr) & set(va) and len(tr) + len(va) == 200
    assert len(va) == 40
    assert sum(not truth[s] for s in va) == 4  # 10% singletons preserved


def test_subsample_preserves_orphan_density():
    """Sub-sampling keeps matches of kept S1s and the same orphan share."""
    s1, s2, s3, truth = make_dataset(n_s1=400, seed=1)
    a1, a2, a3, at = rp.subsample_train(s1, s2, s3, truth, 0.5)
    assert len(a1) == 200 and set(at) == set(a1["entity_id"])
    others = pd.concat([a2, a3])
    matched = {m for ms in at.values() for m in ms}
    assert matched <= set(others["entity_id"])
    full_orphan = 1 - sum(map(len, truth.values())) / (len(s2) + len(s3))
    sub_orphan = 1 - len(matched) / len(others)
    assert sub_orphan == pytest.approx(full_orphan, abs=0.05)


def test_valid_run_end_to_end_with_loco(isolated):
    """Full valid mode beats the all-empty baseline; LOCO and error dumps produced."""
    s1, s2, s3, truth = make_dataset(n_s1=320, seed=2)
    m = rp.valid_run(s1, s2, s3, truth, stage="all", loco=True, use_embeddings=True,
                     encoder=fake_encoder)
    assert m["block_pair_recall"] >= 0.95
    assert m["valid_macro_f05"] > m["all_empty_baseline"] + 0.5
    assert m["valid_macro_f05"] > 0.85
    assert 0.3 <= m["tau"] <= 0.95
    assert {"loco_f05_US", "loco_f05_India", "valid_f05_US", "valid_f05_India"} <= set(m)
    arts = isolated / "artifacts"
    for f in ("errors_fp.tsv", "errors_fn.tsv", "importance_valid.tsv"):
        assert (arts / f).exists(), f
    # Second run reuses the cached normalisation / candidates / embeddings.
    m2 = rp.valid_run(s1, s2, s3, truth, stage="blocking", use_embeddings=True,
                      encoder=lambda t: pytest.fail("encoder called despite cache"))
    assert m2["block_pair_recall"] == m["block_pair_recall"]


def test_valid_run_stops_at_stage(isolated):
    """--stage blocking/features return early with only those metrics."""
    s1, s2, s3, truth = make_dataset(n_s1=120, seed=4)
    mb = rp.valid_run(s1, s2, s3, truth, stage="blocking", use_embeddings=False)
    assert "block_pair_recall" in mb and "valid_macro_f05" not in mb
    mf = rp.valid_run(s1, s2, s3, truth, stage="features", use_embeddings=False)
    assert mf["n_feature_rows"] > 0 and "valid_macro_f05" not in mf


def test_test_run_writes_valid_submission_with_unseen_country(isolated):
    """Train on US/India, predict a test split that adds France; outputs obey every rule."""
    train = make_dataset(n_s1=240, seed=6)
    t1, t2, t3, t_truth = make_dataset(n_s1=150, seed=7, countries=("US", "India", "France"))
    out = isolated / "output"
    m = rp.test_run(train, (t1, t2, t3), out, use_embeddings=True, encoder=fake_encoder)
    match, cand = read_tsv(out / "matching_results.tsv"), read_tsv(out / "candidate_pairs.tsv")
    assert list(match.columns) == ["source1_entity_id", "matched_entity_ids"]
    assert list(cand.columns) == ["source1_entity_id", "candidate_entity_ids"]
    assert match["source1_entity_id"].tolist() == t1["entity_id"].tolist()
    assert cand["source1_entity_id"].tolist() == t1["entity_id"].tolist()
    valid = set(t2["entity_id"]) | set(t3["entity_id"])
    cand_map = dict(zip(cand["source1_entity_id"], cand["candidate_entity_ids"]))
    for s, ms in zip(match["source1_entity_id"], match["matched_entity_ids"]):
        ids = [x for x in ms.split(",") if x]
        assert len(ids) == len(set(ids)) and set(ids) <= valid
        assert set(ids) <= set(cand_map[s].split(","))
        assert " " not in ms and '"' not in ms
    # France (never seen in training) gets real predictions and a sane score.
    pred = {s: [x for x in v.split(",") if x] for s, v in
            zip(match["source1_entity_id"], match["matched_entity_ids"])}
    from src.evaluate import macro_fbeta
    fr = [s for s, c in zip(t1["entity_id"], t1["country"]) if c == "France"]
    assert sum(len(pred[s]) for s in fr) > 0
    assert macro_fbeta(pred, t_truth, s1_ids=fr) > 0.6
    assert m["test_n_s1"] == len(t1)
    assert (isolated / "artifacts" / "model_final.txt").exists()


def test_cli_reads_files_from_config_paths(isolated, monkeypatch):
    """The CLI loads TSVs via config paths, runs, writes output/ and experiments.csv."""
    data = isolated / "dataset"
    s1, s2, s3, truth = make_dataset(n_s1=160, seed=8)
    write_split(data, "train", s1, s2, s3, truth)
    t1, t2, t3, _ = make_dataset(n_s1=80, seed=9, countries=("US", "India", "France"))
    write_split(data, "test", t1, t2, t3)
    monkeypatch.setattr(config, "TRAIN_FILES", {
        "s1": data / "train/train_source1.tsv", "s2": data / "train/train_source2.tsv",
        "s3": data / "train/train_source3.tsv",
        "ground_truth": data / "train/train_ground_truth.tsv"})
    monkeypatch.setattr(config, "TEST_FILES", {
        "s1": data / "test/test_source1.tsv", "s2": data / "test/test_source2.tsv",
        "s3": data / "test/test_source3.tsv"})
    monkeypatch.setattr(config, "OUTPUT_DIR", isolated / "output")
    rp.main(["--mode", "valid", "--stage", "blocking", "--no-embeddings", "--sample", "0.5"])
    rp.main(["--mode", "test", "--no-embeddings"])
    log = pd.read_csv(isolated / "experiments.csv")
    assert log["mode"].tolist() == ["valid", "test"]
    assert {"timestamp", "git", "config", "block_pair_recall"} <= set(log.columns)
    assert log["sample"].tolist() == [0.5, 1.0]
    assert len(read_tsv(isolated / "output" / "matching_results.tsv")) == len(t1)


def test_truth_from_frame():
    """Empty cells become singletons."""
    gt = pd.DataFrame({"source1_entity_id": ["S1-1", "S1-2"],
                       "matched_entity_ids": ["S2-1,S3-2", ""]})
    assert rp.truth_from_frame(gt) == {"S1-1": ["S2-1", "S3-2"], "S1-2": []}
