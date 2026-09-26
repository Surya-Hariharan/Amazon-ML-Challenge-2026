"""Environment, hashing and resource-usage instrumentation (roadmap v3 Phase 0).

Measurement infrastructure only. Nothing here changes normalisation, blocking,
feature, model, decision or submission behaviour -- every function is read-only
with respect to the ML pipeline and only observes/reports it.
"""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
import sys
import time
import traceback
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

_HASH_CHUNK = 1 << 20  # 1 MiB


def file_hash(path: str | Path, algo: str = "sha256") -> str:
    """Return the hex digest of a file's bytes, read in fixed-size chunks."""
    h = hashlib.new(algo)
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(_HASH_CHUNK), b""):
            h.update(chunk)
    return h.hexdigest()


def dataset_file_hashes(files: Mapping[str, str | Path]) -> dict[str, str]:
    """Return ``{name: sha256}`` for every path in ``files``.

    A missing file is reported as ``"missing"`` rather than raising, so a
    diagnostic run can still produce a report that flags what wasn't found.
    """
    out: dict[str, str] = {}
    for name, path in files.items():
        p = Path(path)
        out[name] = file_hash(p) if p.exists() else "missing"
    return out


def _package_version(name: str) -> str:
    """Return an installed package's version, or ``"not installed"``."""
    try:
        from importlib.metadata import version
        return version(name)
    except Exception:
        return "not installed"


def env_info() -> dict[str, Any]:
    """Return Python/platform/package-version/CUDA info for a diagnostic log.

    Best-effort: any field this process cannot determine (e.g. CUDA on a
    CPU-only machine) is reported as ``None`` rather than raising.
    """
    info: dict[str, Any] = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor(),
    }
    for pkg in ("pandas", "numpy", "scipy", "scikit-learn", "lightgbm", "torch",
                "sentence-transformers", "rapidfuzz", "pyarrow"):
        info[f"pkg_{pkg}"] = _package_version(pkg)
    try:
        import torch
        info["cuda_available"] = bool(torch.cuda.is_available())
        info["cuda_device_name"] = (torch.cuda.get_device_name(0)
                                    if torch.cuda.is_available() else None)
    except Exception:
        info["cuda_available"] = None
        info["cuda_device_name"] = None
    return info


def disk_usage(path: str | Path) -> dict[str, int]:
    """Return ``{total, used, free}`` bytes for the filesystem holding ``path``."""
    path = Path(path)
    path = path if path.exists() else path.parent
    total, used, free = shutil.disk_usage(path)
    return {"total": total, "used": used, "free": free}


def artifact_sizes(paths: Iterable[str | Path]) -> dict[str, int | None]:
    """Return ``{path: size_bytes}`` for each path, or ``None`` if it doesn't exist."""
    out: dict[str, int | None] = {}
    for p in paths:
        p = Path(p)
        out[str(p)] = p.stat().st_size if p.exists() else None
    return out


def _peak_rss_bytes() -> int | None:
    """Best-effort peak resident-set size of this process, in bytes.

    Uses ``resource.getrusage`` on POSIX (Linux reports KiB, other POSIX
    platforms bytes -- normalised to bytes here). Returns ``None`` on
    platforms without the ``resource`` module (e.g. Windows dev machines);
    trustworthy full-scale numbers must come from a run on the actual Linux
    SageMaker instance. Note this is a whole-process high-water mark since
    process start, not scoped to any particular call -- call it once per
    process/stage for a meaningful reading, not repeatedly within one process.
    """
    try:
        import resource
    except ImportError:
        return None
    ru_maxrss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return ru_maxrss * 1024 if platform.system() == "Linux" else ru_maxrss


def _peak_gpu_bytes() -> int | None:
    """Best-effort peak CUDA memory allocated since the last reset, in bytes."""
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    return int(torch.cuda.max_memory_allocated())


def _reset_gpu_peak() -> None:
    """Reset CUDA peak-memory stats, if CUDA is available."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass


class ResourceTracker:
    """Context manager recording wall time and best-effort peak RSS/GPU memory.

    ``self.report`` after ``__exit__`` holds ``runtime_s``, ``peak_rss_bytes``
    (``None`` off Linux) and ``peak_gpu_bytes`` (``None`` without CUDA).
    """

    def __init__(self) -> None:
        """Initialise an empty report."""
        self.report: dict[str, Any] = {}
        self._t0 = 0.0

    def __enter__(self) -> "ResourceTracker":
        """Start the timer and reset the GPU peak-memory counter."""
        _reset_gpu_peak()
        self._t0 = time.time()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        """Record runtime and best-effort peak RSS/GPU memory."""
        self.report["runtime_s"] = time.time() - self._t0
        self.report["peak_rss_bytes"] = _peak_rss_bytes()
        self.report["peak_gpu_bytes"] = _peak_gpu_bytes()


def run_stage(
    name: str,
    fn: Callable[[], Any],
    artifact_paths: Iterable[str | Path] = (),
    log_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run one resource-qualification stage; never raises.

    Times ``fn()``, tracks peak RSS/GPU memory and disk usage before/after,
    and catches any exception so a failing stage produces a clear failure
    report instead of crashing the qualification run (roadmap v3 O0: "if a
    stage fails, STOP and report the exact failure" -- the caller, not this
    function, decides whether to stop and move on).

    Returns a JSON-serialisable dict with ``name``, ``success``, ``start``,
    ``end``, ``runtime_s``, ``peak_rss_bytes``, ``peak_gpu_bytes``,
    ``disk_before``, ``disk_after``, ``artifact_sizes``, and, on failure,
    ``error`` and ``traceback``. If ``log_dir`` is given, the report is also
    written to ``<log_dir>/stage_<name>_<timestamp>.json``.
    """
    disk_before = disk_usage(Path.cwd())
    start = time.strftime("%Y-%m-%d %H:%M:%S")
    report: dict[str, Any] = {"name": name, "start": start}
    try:
        with ResourceTracker() as rt:
            fn()
        report.update(rt.report)
        report["success"] = True
    except Exception as exc:  # noqa: BLE001 -- must never crash the qualification driver
        report["success"] = False
        report["error"] = str(exc)
        report["traceback"] = traceback.format_exc()
        report["runtime_s"] = None
        report["peak_rss_bytes"] = None
        report["peak_gpu_bytes"] = None
    report["end"] = time.strftime("%Y-%m-%d %H:%M:%S")
    report["disk_before"] = disk_before
    report["disk_after"] = disk_usage(Path.cwd())
    report["artifact_sizes"] = artifact_sizes(artifact_paths)
    if log_dir is not None:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        save_json(report, log_dir / f"stage_{name}_{ts}.json")
    return report


def save_json(obj: Any, path: str | Path) -> None:
    """Write ``obj`` as indented JSON, creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=2, default=str)
