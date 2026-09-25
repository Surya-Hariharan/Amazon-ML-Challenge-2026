"""Opt-in S3 persistence utility for the SageMaker workflow (CLAUDE.md §3).

Context: each SageMaker notebook instance is a separate machine with its own empty,
ephemeral local disk. ``s3://tensortrio`` (configured via the ``BER_S3_ROOT`` env var,
mirroring ``config._default_data_dir``'s ``BER_DATA_DIR`` pattern) is the only thing
shared between a laptop and every instance. This module is the *explicit, opt-in* sync
layer between that persistent bucket and the local working copy:

* :func:`download_dataset` — ``s3://<root>/dataset/`` -> local ``dataset/``.
* :func:`upload_artifacts` — local ``code/artifacts/`` -> ``s3://<root>/artifacts/``.
* :func:`upload_experiments` — local ``experiments/`` -> ``s3://<root>/experiments/``,
  plus the three items that live outside ``experiments/`` on disk
  (``config.EXPERIMENTS_CSV``, ``code/artifacts/errors_*.tsv``,
  ``code/artifacts/importance_*.tsv``) synced to sensible keys under
  ``experiments/`` (see :func:`extra_experiment_files` for the exact scheme).
* :func:`upload_submissions` — local ``output/matching_results.tsv`` and
  ``output/candidate_pairs.tsv`` -> ``s3://<root>/submissions/``.

Design rules this module obeys (do not relax without updating CLAUDE.md):

1. **No import-time side effects.** ``import s3_sync`` never touches the network, the
   filesystem beyond what Python itself does, or requires AWS credentials or boto3 to
   be installed. Every public function is independently callable and opt-in.
2. **boto3 is lazily imported**, only inside the functions that actually perform S3
   I/O (via :func:`_get_client`), because boto3 is *not* a pinned dependency of the
   normal ML pipeline (CLAUDE.md forbids adding it to ``code/requirements.txt`` in this
   change). If boto3 is not installed, calling an S3 operation raises a clear
   ``RuntimeError`` — never at import time.
3. **Pure logic is separated from I/O.** All local-path <-> S3-key mapping is done by
   plain functions with no boto3 dependency (``parse_s3_root``, ``join_key``,
   ``dataset_key_for_local_path``, ``iter_local_files``, ...), so it can be unit
   tested without boto3 installed at all. The boto3-calling functions are thin wrappers
   around this logic and are unit tested by mocking ``_get_client``.
4. **Never destructive by default.** Uploads never delete existing S3 objects unless a
   caller explicitly passes ``delete_extra=True`` (default ``False`` everywhere).
   Downloads never overwrite an existing local file unless the caller passes
   ``overwrite=True`` (default ``False``).
5. **Fails loudly.** Permission errors and missing files are never swallowed. A missing
   ``BER_S3_ROOT`` raises immediately when an operation is invoked, with a message
   telling the caller how to set it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from . import config

# --- Configuration -------------------------------------------------------------------

#: No hardcoded bucket default: an absent BER_S3_ROOT must fail loudly, not guess.
BER_S3_ROOT_ENV = "BER_S3_ROOT"


class S3SyncError(RuntimeError):
    """Raised for any S3-sync-specific configuration or operation failure."""


@dataclass(frozen=True)
class S3Root:
    """A parsed ``s3://bucket/prefix`` root."""

    bucket: str
    prefix: str  # normalized: no leading slash, either "" or ending in "/"


def parse_s3_root(root: str) -> S3Root:
    """Parse an ``s3://bucket[/prefix]`` URI into a bucket and normalized prefix.

    Pure function — no network, no boto3. Raises :class:`S3SyncError` for anything
    that is not a well-formed ``s3://`` URI with a non-empty bucket name.
    """
    if not root.startswith("s3://"):
        raise S3SyncError(f"BER_S3_ROOT must start with 's3://', got {root!r}")
    without_scheme = root[len("s3://"):]
    bucket, _, rest = without_scheme.partition("/")
    if not bucket:
        raise S3SyncError(f"BER_S3_ROOT is missing a bucket name: {root!r}")
    prefix = rest.strip("/")
    if prefix:
        prefix += "/"
    return S3Root(bucket=bucket, prefix=prefix)


def get_s3_root() -> S3Root:
    """Read and parse ``BER_S3_ROOT``; raise :class:`S3SyncError` if unset.

    Mirrors ``config._default_data_dir``'s use of ``BER_DATA_DIR`` for the local
    dataset, but has no fallback default — an S3 destination must never be guessed.
    """
    env = os.environ.get(BER_S3_ROOT_ENV)
    if not env:
        raise S3SyncError(
            f"{BER_S3_ROOT_ENV} is not set. Set it to your S3 root, e.g. "
            f"{BER_S3_ROOT_ENV}=s3://tensortrio/amazon-ml-challenge-2026/ "
            "before calling any src.s3_sync operation. This is never inferred or "
            "defaulted, to avoid silently syncing to the wrong bucket."
        )
    return parse_s3_root(env)


def join_key(prefix: str, *parts: str) -> str:
    """Join an S3 key prefix with path parts using forward slashes, no leading slash."""
    segments = [p.strip("/") for p in (prefix, *parts) if p.strip("/")]
    return "/".join(segments)


# --- Pure local-path <-> S3-key mapping (unit-testable without boto3) ----------------

def iter_local_files(root: Path) -> Iterator[Path]:
    """Yield every regular file under ``root``, recursively, in sorted order.

    Pure filesystem walk (no boto3). Used by every "upload a directory" operation so
    the set of files considered is deterministic and easy to unit test.
    """
    if not root.exists():
        return
    for path in sorted(root.rglob("*")):
        if path.is_file():
            yield path


def relative_key(root: Path, path: Path) -> str:
    """Return ``path``'s location relative to ``root`` as a forward-slash S3 key."""
    return path.resolve().relative_to(root.resolve()).as_posix()


def dataset_key_for_local_path(local_path: Path, dataset_dir: Path) -> str:
    """Map a local ``dataset/...`` file to its S3 key under ``<root>/dataset/``."""
    return join_key("dataset", relative_key(dataset_dir, local_path))


def local_path_for_dataset_key(key: str, dataset_dir: Path, s3_prefix: str = "dataset/") -> Path:
    """Map an S3 key under ``<root>/dataset/`` back to its local path under ``dataset_dir``.

    Inverse of :func:`dataset_key_for_local_path`. Rejects a key that escapes
    ``s3_prefix`` (defensive: a malformed/unexpected listing should fail loudly, not
    write outside ``dataset_dir``).
    """
    if not key.startswith(s3_prefix):
        raise S3SyncError(f"S3 key {key!r} is not under the expected prefix {s3_prefix!r}")
    rel = key[len(s3_prefix):]
    if not rel or rel.endswith("/"):
        raise S3SyncError(f"S3 key {key!r} does not name a file")
    return dataset_dir / rel


def artifacts_key_for_local_path(local_path: Path, artifacts_dir: Path) -> str:
    """Map a local ``code/artifacts/...`` file to its S3 key under ``<root>/artifacts/``."""
    return join_key("artifacts", relative_key(artifacts_dir, local_path))


def experiments_key_for_local_path(local_path: Path, experiments_dir: Path) -> str:
    """Map a local ``experiments/...`` file to its S3 key under ``<root>/experiments/``."""
    return join_key("experiments", relative_key(experiments_dir, local_path))


def extra_experiment_files() -> dict[Path, str]:
    """Return the files outside ``experiments/`` that :func:`upload_experiments` also syncs.

    The audit (``experiments/reports/sagemaker_storage_adaptability_audit.md``) found
    three items that a naive ``experiments/`` directory sync would miss because they
    physically live elsewhere on disk:

    * ``config.EXPERIMENTS_CSV`` (``code/experiments.csv``) -> key
      ``experiments/experiments.csv``.
    * ``code/artifacts/errors_*.tsv`` (written by
      ``run_pipeline.dump_errors``, e.g. ``errors_fp.tsv`` / ``errors_fn.tsv``) -> key
      ``experiments/artifacts/errors_*.tsv``.
    * ``code/artifacts/importance_*.tsv`` (written by ``run_pipeline`` for the ``valid``
      and ``test`` modes, e.g. ``importance_valid.tsv`` / ``importance_final.tsv``) ->
      key ``experiments/artifacts/importance_*.tsv``.

    Returns a mapping of existing local :class:`~pathlib.Path` -> destination S3 key.
    Only files that currently exist on disk are included (nothing is invented).
    """
    mapping: dict[Path, str] = {}
    if config.EXPERIMENTS_CSV.exists():
        mapping[config.EXPERIMENTS_CSV] = "experiments/experiments.csv"
    for pattern in ("errors_*.tsv", "importance_*.tsv"):
        for path in sorted(config.ARTIFACTS_DIR.glob(pattern)):
            mapping[path] = join_key("experiments/artifacts", path.name)
    return mapping


def submission_key_for_local_path(local_path: Path, tag: str | None = None) -> str:
    """Map a local ``output/*.tsv`` file to its S3 key under ``<root>/submissions/``.

    ``tag`` optionally nests the upload under ``submissions/<tag>/...`` so repeated
    submissions don't overwrite each other (CLAUDE.md §7's ``sub-d{day}-{n}`` tags).
    """
    if tag:
        return join_key("submissions", tag, local_path.name)
    return join_key("submissions", local_path.name)


# --- boto3 I/O (lazy import; every function here is opt-in and network-touching) -----

def _get_client():
    """Lazily import boto3 and return an S3 client.

    Raises :class:`S3SyncError` with an actionable message if boto3 is not installed.
    This is the *only* place boto3 is imported, and it only happens when an S3
    operation is actually invoked — never at module import time.
    """
    try:
        import boto3  # noqa: PLC0415 (intentionally lazy — see module docstring)
    except ImportError as exc:
        raise S3SyncError(
            "boto3 is required for S3 operations; install it "
            "(`pip install boto3`) or add it to code/requirements.txt. "
            "It is deliberately not a pipeline dependency: local pipeline runs "
            "(src.run_pipeline) never need it."
        ) from exc
    return boto3.client("s3")


def _list_keys(client, bucket: str, prefix: str) -> list[str]:
    """Return every object key under ``prefix`` in ``bucket`` (paginated listing)."""
    keys: list[str] = []
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            keys.append(obj["Key"])
    return keys


def download_dataset(dataset_dir: Path | None = None, overwrite: bool = False) -> list[Path]:
    """Download ``s3://<root>/dataset/`` to a local ``dataset/`` directory.

    Preserves the ``train/``/``test/`` structure and exact filenames, creating parent
    directories as needed. By default an existing local file is left untouched (not
    silently overwritten) unless ``overwrite=True``. Uses boto3's managed
    ``download_file`` transfer (multipart-safe for large TSVs) rather than reading
    whole objects into memory.

    Returns the list of local paths that were written (skipped files are not
    included). Raises :class:`S3SyncError` if ``BER_S3_ROOT`` is unset, and lets any
    boto3 exception (e.g. an access-denied error) propagate unchanged — S3 errors are
    never swallowed.
    """
    root = get_s3_root()
    dataset_dir = config.DATA_DIR if dataset_dir is None else Path(dataset_dir)
    client = _get_client()
    prefix = join_key(root.prefix, "dataset") + "/"
    keys = _list_keys(client, root.bucket, prefix)
    if not keys:
        print(f"s3_sync.download_dataset: no objects found under s3://{root.bucket}/{prefix}")
        return []

    written: list[Path] = []
    for key in keys:
        if key.endswith("/"):
            continue  # a directory marker, not a file
        rel = key[len(prefix):]
        local_path = dataset_dir / rel
        if local_path.exists() and not overwrite:
            print(f"s3_sync.download_dataset: skipping existing {local_path} (overwrite=False)")
            continue
        local_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"s3_sync.download_dataset: s3://{root.bucket}/{key} -> {local_path}")
        client.download_file(root.bucket, key, str(local_path))
        written.append(local_path)
    print(f"s3_sync.download_dataset: wrote {len(written)}/{len(keys)} file(s) to {dataset_dir}")
    return written


def _upload_directory(
    client, root: S3Root, local_dir: Path, key_prefix: str, delete_extra: bool,
) -> list[str]:
    """Upload every file under ``local_dir`` to ``<root.prefix><key_prefix>/...``.

    Shared implementation for :func:`upload_artifacts` and the directory portion of
    :func:`upload_experiments`. Never skips a file it was asked to upload; never
    deletes existing S3 objects unless ``delete_extra`` is True.
    """
    uploaded: list[str] = []
    local_files = list(iter_local_files(local_dir))
    for path in local_files:
        key = join_key(root.prefix, key_prefix, relative_key(local_dir, path))
        print(f"s3_sync.upload: {path} -> s3://{root.bucket}/{key}")
        client.upload_file(str(path), root.bucket, key)
        uploaded.append(key)

    if delete_extra:
        remote_prefix = join_key(root.prefix, key_prefix) + "/"
        existing = set(_list_keys(client, root.bucket, remote_prefix))
        keep = set(uploaded)
        stale = sorted(existing - keep)
        for key in stale:
            print(f"s3_sync.upload: delete_extra removing s3://{root.bucket}/{key}")
            client.delete_object(Bucket=root.bucket, Key=key)
    return uploaded


def upload_artifacts(artifacts_dir: Path | None = None, delete_extra: bool = False) -> list[str]:
    """Upload local ``code/artifacts/`` to ``s3://<root>/artifacts/``, preserving structure.

    ``delete_extra=True`` opts into removing S3 objects under ``artifacts/`` that no
    longer have a local counterpart; the default (``False``) never deletes anything.
    Returns the list of S3 keys written.
    """
    root = get_s3_root()
    artifacts_dir = config.ARTIFACTS_DIR if artifacts_dir is None else Path(artifacts_dir)
    client = _get_client()
    uploaded = _upload_directory(client, root, artifacts_dir, "artifacts", delete_extra)
    print(f"s3_sync.upload_artifacts: uploaded {len(uploaded)} file(s) from {artifacts_dir}")
    return uploaded


def upload_experiments(experiments_dir: Path | None = None, delete_extra: bool = False) -> list[str]:
    """Upload local ``experiments/`` to ``s3://<root>/experiments/``, preserving structure.

    Also explicitly uploads the three items the adaptability audit found live outside
    the ``experiments/`` tree (see :func:`extra_experiment_files` for the exact key
    scheme): ``code/experiments.csv``, ``code/artifacts/errors_*.tsv`` and
    ``code/artifacts/importance_*.tsv``.

    ``delete_extra=True`` (default ``False``) opts into removing S3 objects under
    ``experiments/`` that no longer have a local counterpart; note this only prunes
    the directory-synced portion, never the explicit extra files (those are additive
    by nature — deleting them from S3 because a fresh instance hasn't run the
    pipeline yet would destroy history).
    """
    root = get_s3_root()
    experiments_dir = config.REPO_ROOT / "experiments" if experiments_dir is None else Path(experiments_dir)
    client = _get_client()
    uploaded = _upload_directory(client, root, experiments_dir, "experiments", delete_extra)

    for local_path, rel_key in extra_experiment_files().items():
        key = join_key(root.prefix, rel_key)
        print(f"s3_sync.upload_experiments: {local_path} -> s3://{root.bucket}/{key}")
        client.upload_file(str(local_path), root.bucket, key)
        uploaded.append(key)

    print(f"s3_sync.upload_experiments: uploaded {len(uploaded)} file(s) total")
    return uploaded


def upload_submissions(output_dir: Path | None = None, tag: str | None = None) -> list[str]:
    """Upload ``output/matching_results.tsv`` and ``output/candidate_pairs.tsv`` to S3.

    Purely a copy-after-the-fact operation: it does not change where those files are
    produced locally (still ``config.OUTPUT_DIR`` / ``run_pipeline``'s ``--mode test``),
    and it never runs the pipeline itself. Only files that actually exist locally are
    uploaded; a missing expected file raises :class:`S3SyncError` rather than silently
    uploading a partial submission.

    ``tag`` optionally nests the upload under ``submissions/<tag>/...`` (see
    :func:`submission_key_for_local_path`), matching CLAUDE.md §7's
    ``sub-d{day}-{n}`` submission tags.
    """
    root = get_s3_root()
    output_dir = config.OUTPUT_DIR if output_dir is None else Path(output_dir)
    expected = [config.MATCHING_FILE.name, config.CANDIDATE_FILE.name]
    local_paths = [output_dir / name for name in expected]
    missing = [p for p in local_paths if not p.exists()]
    if missing:
        raise S3SyncError(
            "upload_submissions: missing expected output file(s) "
            f"{[str(p) for p in missing]}. Run the pipeline and validate "
            "(utils/validate_submission.py) before uploading."
        )

    client = _get_client()
    uploaded: list[str] = []
    for path in local_paths:
        key = join_key(root.prefix, submission_key_for_local_path(path, tag=tag))
        print(f"s3_sync.upload_submissions: {path} -> s3://{root.bucket}/{key}")
        client.upload_file(str(path), root.bucket, key)
        uploaded.append(key)
    return uploaded
