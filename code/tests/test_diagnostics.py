"""Tests for src.diagnostics (roadmap v3 Phase 0 instrumentation)."""

import json

from src import diagnostics as diag


def test_file_hash_deterministic_and_content_sensitive(tmp_path):
    """Same content -> same hash; different content -> different hash."""
    a = tmp_path / "a.txt"
    b = tmp_path / "b.txt"
    a.write_text("hello world", encoding="utf-8")
    b.write_text("hello world", encoding="utf-8")
    assert diag.file_hash(a) == diag.file_hash(b)
    b.write_text("hello worlds", encoding="utf-8")
    assert diag.file_hash(a) != diag.file_hash(b)


def test_dataset_file_hashes_flags_missing_files(tmp_path):
    """A missing path is reported as "missing", not an exception."""
    present = tmp_path / "present.tsv"
    present.write_text("x\ty\n1\t2\n", encoding="utf-8")
    out = diag.dataset_file_hashes({"present": present, "absent": tmp_path / "nope.tsv"})
    assert out["present"] == diag.file_hash(present)
    assert out["absent"] == "missing"


def test_env_info_has_expected_keys():
    """env_info always reports python/platform/package fields, best-effort."""
    info = diag.env_info()
    assert "python_version" in info and "platform" in info
    assert info["pkg_pandas"] != ""  # pandas is a hard dependency, must resolve
    assert "cuda_available" in info


def test_disk_usage_reports_positive_totals(tmp_path):
    """disk_usage returns total/used/free with total > 0."""
    out = diag.disk_usage(tmp_path)
    assert set(out) == {"total", "used", "free"}
    assert out["total"] > 0


def test_artifact_sizes_missing_vs_present(tmp_path):
    """A present file gets its byte size; a missing one gets None."""
    f = tmp_path / "art.bin"
    f.write_bytes(b"0123456789")
    out = diag.artifact_sizes([f, tmp_path / "missing.bin"])
    assert out[str(f)] == 10
    assert out[str(tmp_path / "missing.bin")] is None


def test_resource_tracker_records_runtime_and_best_effort_memory():
    """ResourceTracker always records runtime_s; memory fields are int or None."""
    with diag.ResourceTracker() as rt:
        sum(range(100_000))
    assert rt.report["runtime_s"] >= 0
    assert rt.report["peak_rss_bytes"] is None or isinstance(rt.report["peak_rss_bytes"], int)
    assert rt.report["peak_gpu_bytes"] is None or isinstance(rt.report["peak_gpu_bytes"], int)


def test_run_stage_success_records_report_and_artifacts(tmp_path):
    """A successful stage reports success=True and lists artifact sizes."""
    art = tmp_path / "out.bin"

    def fn():
        art.write_bytes(b"result")

    report = diag.run_stage("demo", fn, [art], log_dir=tmp_path / "logs")
    assert report["success"] is True
    assert report["artifact_sizes"][str(art)] == 6
    assert report["runtime_s"] >= 0
    logs = list((tmp_path / "logs").glob("stage_demo_*.json"))
    assert len(logs) == 1
    with open(logs[0]) as fh:
        loaded = json.load(fh)
    assert loaded["name"] == "demo" and loaded["success"] is True


def test_run_stage_failure_is_caught_and_reported(tmp_path):
    """A raising stage never propagates; it's reported as success=False with the error."""
    def fn():
        raise ValueError("boom")

    report = diag.run_stage("failing", fn, log_dir=tmp_path / "logs")
    assert report["success"] is False
    assert "boom" in report["error"]
    assert "ValueError" in report["traceback"]
    assert report["runtime_s"] is None


def test_save_json_round_trips(tmp_path):
    """save_json writes valid, re-loadable JSON, creating parent directories."""
    path = tmp_path / "nested" / "report.json"
    diag.save_json({"a": 1, "b": [1, 2, 3]}, path)
    with open(path) as fh:
        assert json.load(fh) == {"a": 1, "b": [1, 2, 3]}
