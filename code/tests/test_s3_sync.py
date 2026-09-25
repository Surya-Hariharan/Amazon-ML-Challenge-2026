"""Tests for src.s3_sync: the opt-in S3 persistence utility.

Split to match the module's own split (see s3_sync.py's docstring):

* Pure path/key-mapping logic is tested directly, with no boto3 involved at all —
  these tests run in any environment, boto3 installed or not.
* boto3-calling functions are tested by monkeypatching ``s3_sync._get_client`` to
  return a ``MagicMock``, so no real network access or AWS credentials are ever
  needed, and these tests do not care whether boto3 is actually installed either.
* The lazy-import error path (``_get_client`` when boto3 truly is not importable) is
  tested by forcing ``import boto3`` to fail via ``sys.modules`` injection, so it
  passes the same way regardless of whether this dev environment happens to have
  boto3 installed.

Nothing here requires live AWS credentials or touches the real network.
"""

from pathlib import Path
from unittest.mock import MagicMock, call

import pytest

from src import config, s3_sync


# --- Pure logic: S3 root / key parsing ------------------------------------------------

def test_parse_s3_root_bucket_and_prefix():
    """A root with a multi-segment prefix normalizes to a trailing-slash prefix."""
    root = s3_sync.parse_s3_root("s3://tensortrio/amazon-ml-challenge-2026/")
    assert root.bucket == "tensortrio"
    assert root.prefix == "amazon-ml-challenge-2026/"


def test_parse_s3_root_bucket_only_no_prefix():
    """A bare bucket URI (no prefix) parses to an empty prefix."""
    root = s3_sync.parse_s3_root("s3://tensortrio")
    assert root.bucket == "tensortrio"
    assert root.prefix == ""


def test_parse_s3_root_strips_extra_slashes():
    """Leading/trailing slashes in the prefix are normalized away and back."""
    root = s3_sync.parse_s3_root("s3://bucket//a/b//")
    assert root.bucket == "bucket"
    assert root.prefix == "a/b/"


@pytest.mark.parametrize("bad", ["", "not-s3", "http://bucket/x", "s3://", "s3:///no-bucket"])
def test_parse_s3_root_rejects_malformed(bad):
    """Anything not a well-formed s3://bucket[/prefix] URI raises S3SyncError."""
    with pytest.raises(s3_sync.S3SyncError):
        s3_sync.parse_s3_root(bad)


def test_get_s3_root_reads_env_var(monkeypatch):
    """BER_S3_ROOT (mirroring the BER_DATA_DIR pattern) drives the root."""
    monkeypatch.setenv("BER_S3_ROOT", "s3://tensortrio/prefix/")
    root = s3_sync.get_s3_root()
    assert root.bucket == "tensortrio"
    assert root.prefix == "prefix/"


def test_get_s3_root_missing_raises_clear_error(monkeypatch):
    """No BER_S3_ROOT -> a clear, actionable S3SyncError, never a guess."""
    monkeypatch.delenv("BER_S3_ROOT", raising=False)
    with pytest.raises(s3_sync.S3SyncError, match="BER_S3_ROOT"):
        s3_sync.get_s3_root()


def test_join_key_drops_empty_and_joins_with_slash():
    """join_key ignores empty segments and joins the rest with '/'."""
    assert s3_sync.join_key("prefix/", "", "dataset", "train/x.tsv") == "prefix/dataset/train/x.tsv"
    assert s3_sync.join_key("", "dataset") == "dataset"


# --- Pure logic: local path <-> S3 key mapping ----------------------------------------

def test_iter_local_files_recursive_sorted(tmp_path):
    """Every regular file under root is yielded, sorted, directories excluded."""
    (tmp_path / "a").mkdir()
    (tmp_path / "a" / "1.tsv").write_text("x")
    (tmp_path / "b.tsv").write_text("y")
    (tmp_path / "a" / "empty_dir").mkdir()
    files = list(s3_sync.iter_local_files(tmp_path))
    assert files == sorted(files)
    assert {f.name for f in files} == {"1.tsv", "b.tsv"}


def test_iter_local_files_missing_root_yields_nothing(tmp_path):
    """A root that doesn't exist yields no files (not an error) — callers decide."""
    assert list(s3_sync.iter_local_files(tmp_path / "does_not_exist")) == []


def test_dataset_key_round_trip(tmp_path):
    """dataset_key_for_local_path and local_path_for_dataset_key are inverses."""
    dataset_dir = tmp_path / "dataset"
    local = dataset_dir / "train" / "train_source1.tsv"
    local.parent.mkdir(parents=True)
    local.write_text("x")

    key = s3_sync.dataset_key_for_local_path(local, dataset_dir)
    assert key == "dataset/train/train_source1.tsv"

    back = s3_sync.local_path_for_dataset_key(key, dataset_dir)
    assert back == local


def test_local_path_for_dataset_key_rejects_wrong_prefix(tmp_path):
    """A key not under the expected dataset/ prefix fails loudly, never guesses."""
    with pytest.raises(s3_sync.S3SyncError):
        s3_sync.local_path_for_dataset_key("artifacts/x.tsv", tmp_path)


def test_local_path_for_dataset_key_rejects_directory_marker(tmp_path):
    """A key that names a 'directory' (trailing slash) is not a valid file target."""
    with pytest.raises(s3_sync.S3SyncError):
        s3_sync.local_path_for_dataset_key("dataset/train/", tmp_path)


def test_artifacts_key_for_local_path(tmp_path):
    """Artifacts map to keys under artifacts/, preserving subdirectory structure."""
    artifacts_dir = tmp_path / "artifacts"
    nested = artifacts_dir / "models" / "model_final.txt"
    nested.parent.mkdir(parents=True)
    nested.write_text("x")
    assert s3_sync.artifacts_key_for_local_path(nested, artifacts_dir) == "artifacts/models/model_final.txt"


def test_experiments_key_for_local_path(tmp_path):
    """Experiments files map to keys under experiments/, preserving structure."""
    experiments_dir = tmp_path / "experiments"
    nested = experiments_dir / "reports" / "n5000.md"
    nested.parent.mkdir(parents=True)
    nested.write_text("x")
    assert s3_sync.experiments_key_for_local_path(nested, experiments_dir) == "experiments/reports/n5000.md"


def test_extra_experiment_files_finds_the_three_known_paths(tmp_path, monkeypatch):
    """The 3 items outside experiments/ (audit finding) map to experiments/... keys.

    Covers config.EXPERIMENTS_CSV (code/experiments.csv) and code/artifacts/
    errors_*.tsv / importance_*.tsv, as identified in run_pipeline.py.
    """
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()
    experiments_csv = tmp_path / "experiments.csv"
    experiments_csv.write_text("run,f0.5\n")
    (artifacts_dir / "errors_fp.tsv").write_text("x")
    (artifacts_dir / "errors_fn.tsv").write_text("x")
    (artifacts_dir / "importance_valid.tsv").write_text("x")
    (artifacts_dir / "not_matched.tsv").write_text("x")  # must NOT be picked up

    monkeypatch.setattr(config, "EXPERIMENTS_CSV", experiments_csv)
    monkeypatch.setattr(config, "ARTIFACTS_DIR", artifacts_dir)

    mapping = s3_sync.extra_experiment_files()
    assert mapping[experiments_csv] == "experiments/experiments.csv"
    assert mapping[artifacts_dir / "errors_fp.tsv"] == "experiments/artifacts/errors_fp.tsv"
    assert mapping[artifacts_dir / "errors_fn.tsv"] == "experiments/artifacts/errors_fn.tsv"
    assert mapping[artifacts_dir / "importance_valid.tsv"] == "experiments/artifacts/importance_valid.tsv"
    assert (artifacts_dir / "not_matched.tsv") not in mapping
    assert len(mapping) == 4


def test_extra_experiment_files_skips_files_that_dont_exist(tmp_path, monkeypatch):
    """No experiments.csv on disk yet -> it's simply absent from the mapping."""
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()
    monkeypatch.setattr(config, "EXPERIMENTS_CSV", tmp_path / "no_such_file.csv")
    monkeypatch.setattr(config, "ARTIFACTS_DIR", artifacts_dir)
    assert s3_sync.extra_experiment_files() == {}


def test_submission_key_for_local_path_no_tag():
    """Without a tag, submissions map directly under submissions/."""
    path = Path("output/matching_results.tsv")
    assert s3_sync.submission_key_for_local_path(path) == "submissions/matching_results.tsv"


def test_submission_key_for_local_path_with_tag():
    """With a tag, submissions nest under submissions/<tag>/ (CLAUDE.md §7 tags)."""
    path = Path("output/candidate_pairs.tsv")
    key = s3_sync.submission_key_for_local_path(path, tag="sub-d1-1")
    assert key == "submissions/sub-d1-1/candidate_pairs.tsv"


# --- Import-time side effects ---------------------------------------------------------

def test_import_has_no_side_effects(monkeypatch):
    """Importing the module must never require BER_S3_ROOT or touch boto3.

    Reload-free check: the module is already imported above; this asserts that doing
    so did not, itself, require AWS credentials or boto3 (both fixtures here would be
    unset/absent in CI containers with neither).
    """
    monkeypatch.delenv("BER_S3_ROOT", raising=False)
    import importlib

    reloaded = importlib.reload(s3_sync)
    assert reloaded is not None  # reload succeeded with no BER_S3_ROOT and no boto3 call


# --- _get_client lazy import ------------------------------------------------------------

def test_get_client_raises_clear_error_when_boto3_missing(monkeypatch):
    """If boto3 cannot be imported, _get_client raises a clear, actionable S3SyncError.

    Forces the ImportError via sys.modules injection so this passes identically
    whether or not boto3 happens to be installed in the environment running the tests.
    """
    import sys

    monkeypatch.setitem(sys.modules, "boto3", None)  # `import boto3` -> ImportError
    with pytest.raises(s3_sync.S3SyncError, match="boto3"):
        s3_sync._get_client()


def test_get_client_returns_boto3_s3_client_when_available(monkeypatch):
    """When boto3 (or a stand-in) is importable, _get_client builds an s3 client."""
    import sys
    import types

    fake_boto3 = types.ModuleType("boto3")
    fake_client = MagicMock(name="s3_client")
    fake_boto3.client = MagicMock(return_value=fake_client)
    monkeypatch.setitem(sys.modules, "boto3", fake_boto3)

    client = s3_sync._get_client()
    assert client is fake_client
    fake_boto3.client.assert_called_once_with("s3")


# --- boto3-calling functions, with _get_client mocked ---------------------------------

@pytest.fixture
def s3_root_env(monkeypatch):
    """Set a fixed BER_S3_ROOT for the boto3-mocked tests."""
    monkeypatch.setenv("BER_S3_ROOT", "s3://tensortrio/amz/")
    return s3_sync.parse_s3_root("s3://tensortrio/amz/")


def _paginated_client(keys):
    """Build a MagicMock S3 client whose list_objects_v2 paginator yields `keys`."""
    client = MagicMock()
    paginator = MagicMock()
    client.get_paginator.return_value = paginator
    paginator.paginate.return_value = [{"Contents": [{"Key": k} for k in keys]}]
    return client


def test_download_dataset_requires_ber_s3_root(monkeypatch):
    """download_dataset fails loudly (before touching boto3) if BER_S3_ROOT is unset."""
    monkeypatch.delenv("BER_S3_ROOT", raising=False)
    with pytest.raises(s3_sync.S3SyncError, match="BER_S3_ROOT"):
        s3_sync.download_dataset()


def test_download_dataset_writes_new_files_preserving_structure(tmp_path, monkeypatch, s3_root_env):
    """New files are downloaded via download_file, preserving train//test structure."""
    keys = ["amz/dataset/train/train_source1.tsv", "amz/dataset/test/test_source1.tsv"]
    client = _paginated_client(keys)
    monkeypatch.setattr(s3_sync, "_get_client", lambda: client)

    dest = tmp_path / "dataset"
    written = s3_sync.download_dataset(dataset_dir=dest)

    assert sorted(str(p.relative_to(dest)).replace("\\", "/") for p in written) == [
        "test/test_source1.tsv", "train/train_source1.tsv",
    ]
    assert client.download_file.call_count == 2
    client.download_file.assert_any_call(
        "tensortrio", "amz/dataset/train/train_source1.tsv",
        str(dest / "train" / "train_source1.tsv"),
    )


def test_download_dataset_skips_existing_without_overwrite(tmp_path, monkeypatch, s3_root_env):
    """An existing local file is never clobbered unless overwrite=True."""
    dest = tmp_path / "dataset"
    existing = dest / "train" / "train_source1.tsv"
    existing.parent.mkdir(parents=True)
    existing.write_text("local copy")

    client = _paginated_client(["amz/dataset/train/train_source1.tsv"])
    monkeypatch.setattr(s3_sync, "_get_client", lambda: client)

    written = s3_sync.download_dataset(dataset_dir=dest)
    assert written == []
    client.download_file.assert_not_called()
    assert existing.read_text() == "local copy"  # untouched


def test_download_dataset_overwrite_true_redownloads(tmp_path, monkeypatch, s3_root_env):
    """overwrite=True lets a re-download proceed over an existing local file."""
    dest = tmp_path / "dataset"
    existing = dest / "train" / "train_source1.tsv"
    existing.parent.mkdir(parents=True)
    existing.write_text("stale")

    client = _paginated_client(["amz/dataset/train/train_source1.tsv"])
    monkeypatch.setattr(s3_sync, "_get_client", lambda: client)

    written = s3_sync.download_dataset(dataset_dir=dest, overwrite=True)
    assert written == [existing]
    client.download_file.assert_called_once()


def test_upload_artifacts_uploads_every_file(tmp_path, monkeypatch, s3_root_env):
    """upload_artifacts uploads each local file to artifacts/<relpath>."""
    artifacts_dir = tmp_path / "artifacts"
    (artifacts_dir / "models").mkdir(parents=True)
    (artifacts_dir / "models" / "model_final.txt").write_text("x")
    (artifacts_dir / "importance_valid.tsv").write_text("x")

    client = MagicMock()
    monkeypatch.setattr(s3_sync, "_get_client", lambda: client)

    keys = s3_sync.upload_artifacts(artifacts_dir=artifacts_dir)
    assert sorted(keys) == [
        "amz/artifacts/importance_valid.tsv", "amz/artifacts/models/model_final.txt",
    ]
    assert client.upload_file.call_count == 2
    client.delete_object.assert_not_called()  # delete_extra defaults to False


def test_upload_artifacts_delete_extra_is_opt_in(tmp_path, monkeypatch, s3_root_env):
    """delete_extra=True removes stale S3 objects; default False never does."""
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()
    (artifacts_dir / "keep.tsv").write_text("x")

    client = _paginated_client(["amz/artifacts/keep.tsv", "amz/artifacts/stale.tsv"])
    monkeypatch.setattr(s3_sync, "_get_client", lambda: client)

    s3_sync.upload_artifacts(artifacts_dir=artifacts_dir, delete_extra=True)
    client.delete_object.assert_called_once_with(Bucket="tensortrio", Key="amz/artifacts/stale.tsv")


def test_upload_experiments_includes_extra_files(tmp_path, monkeypatch, s3_root_env):
    """upload_experiments uploads the experiments/ tree plus the 3 extra items."""
    experiments_dir = tmp_path / "experiments"
    (experiments_dir / "reports").mkdir(parents=True)
    (experiments_dir / "reports" / "n500.md").write_text("x")

    experiments_csv = tmp_path / "experiments.csv"
    experiments_csv.write_text("x")
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir()
    (artifacts_dir / "errors_fp.tsv").write_text("x")
    monkeypatch.setattr(config, "EXPERIMENTS_CSV", experiments_csv)
    monkeypatch.setattr(config, "ARTIFACTS_DIR", artifacts_dir)

    client = MagicMock()
    monkeypatch.setattr(s3_sync, "_get_client", lambda: client)

    keys = s3_sync.upload_experiments(experiments_dir=experiments_dir)
    assert "amz/experiments/reports/n500.md" in keys
    assert "amz/experiments/experiments.csv" in keys
    assert "amz/experiments/artifacts/errors_fp.tsv" in keys
    assert client.upload_file.call_count == 3


def test_upload_submissions_requires_both_files(tmp_path, monkeypatch, s3_root_env):
    """Missing an expected output file raises rather than uploading a partial submission."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "matching_results.tsv").write_text("x")
    # candidate_pairs.tsv deliberately missing

    monkeypatch.setattr(s3_sync, "_get_client", lambda: MagicMock())
    with pytest.raises(s3_sync.S3SyncError, match="missing expected output"):
        s3_sync.upload_submissions(output_dir=output_dir)


def test_upload_submissions_uploads_both_with_tag(tmp_path, monkeypatch, s3_root_env):
    """Both submission files are uploaded, nested under submissions/<tag>/ when given."""
    output_dir = tmp_path / "output"
    output_dir.mkdir()
    (output_dir / "matching_results.tsv").write_text("x")
    (output_dir / "candidate_pairs.tsv").write_text("x")

    client = MagicMock()
    monkeypatch.setattr(s3_sync, "_get_client", lambda: client)

    keys = s3_sync.upload_submissions(output_dir=output_dir, tag="sub-d1-1")
    assert sorted(keys) == [
        "amz/submissions/sub-d1-1/candidate_pairs.tsv",
        "amz/submissions/sub-d1-1/matching_results.tsv",
    ]
    assert client.upload_file.call_count == 2
