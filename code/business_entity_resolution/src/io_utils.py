"""TSV load/save, ID-list formatting and the submission writer.

All TSV I/O goes through here so the ``sep="\\t"`` / string-ID rules (CLAUDE.md §2.3)
are applied in exactly one place. :func:`write_submission` enforces every output-format
rule from CLAUDE.md §2.5 and refuses to write a file that would break one.
"""

from collections.abc import Collection, Iterable, Mapping, Sequence
from pathlib import Path

import pandas as pd

from . import config


def read_tsv(path: str | Path) -> pd.DataFrame:
    """Read a TSV with every column as a string and no NaN coercion."""
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)


def write_tsv(df: pd.DataFrame, path: str | Path) -> None:
    """Write a DataFrame as TSV (no index), creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, sep="\t", index=False)


def load_split(split: str) -> dict[str, pd.DataFrame]:
    """Load all source files of ``"train"`` or ``"test"`` as string DataFrames.

    Returns a dict keyed like ``config.TRAIN_FILES`` / ``config.TEST_FILES``.
    """
    files = {"train": config.TRAIN_FILES, "test": config.TEST_FILES}[split]
    return {key: read_tsv(path) for key, path in files.items()}


def parse_id_list(value: str) -> list[str]:
    """Parse a comma-separated ID cell into a list; empty cell -> []."""
    return [part.strip() for part in str(value).split(",") if part.strip()]


def format_id_list(ids: Iterable[str]) -> str:
    """Join IDs with commas (no spaces), dropping duplicates while keeping order."""
    return ",".join(dict.fromkeys(ids))


def _clean_list(
    s1: str, ids: Iterable[str], valid_ids: Collection[str], kind: str
) -> list[str]:
    """Validate and de-duplicate one S1's ID list; raise ValueError on a bad ID."""
    out: list[str] = []
    seen: set[str] = set()
    for raw in ids:
        if not isinstance(raw, str):
            raise ValueError(f"{kind} for {s1}: non-string ID {raw!r}")
        rid = raw.strip()
        if not rid.startswith(config.MATCH_PREFIXES):
            raise ValueError(f"{kind} for {s1}: {rid!r} is not an S2-/S3- ID")
        if any(ch in rid for ch in ",\t\n\r\"' "):
            raise ValueError(f"{kind} for {s1}: {rid!r} contains a separator or quote")
        if rid not in valid_ids:
            raise ValueError(f"{kind} for {s1}: {rid!r} is not in the source files")
        if rid not in seen:
            seen.add(rid)
            out.append(rid)
    return out


def _write_two_col(path: Path, header: Sequence[str], rows: list[tuple[str, str]]) -> None:
    """Write a two-column TSV byte-for-byte (no quoting, ``\\n`` line endings)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\t".join(header) + "\n")
        for s1, ids in rows:
            fh.write(f"{s1}\t{ids}\n")


def write_submission(
    matches: Mapping[str, Iterable[str]],
    candidates: Mapping[str, Iterable[str]],
    s1_ids: Sequence[str],
    valid_ids: Collection[str],
    out_dir: str | Path = config.OUTPUT_DIR,
) -> tuple[Path, Path]:
    """Write ``matching_results.tsv`` and ``candidate_pairs.tsv`` after validating them.

    Rules enforced (CLAUDE.md §2.5):

    * exactly one row per S1 ID, in ``s1_ids`` order; ``s1_ids`` must be unique S1- IDs;
    * S1 IDs missing from ``matches``/``candidates`` get an empty list; keys that are
      not in ``s1_ids`` are an error;
    * every listed ID is an S2-/S3- ID present in ``valid_ids``, with no separators or
      quotes; duplicate IDs inside a list are dropped;
    * every matched ID is also in that S1's candidate list;
    * lists are comma-separated with no spaces; empty string for no matches.

    Args:
        matches: S1 ID -> final matched IDs.
        candidates: S1 ID -> the candidate IDs the final model scored.
        s1_ids: every S1 ID of the split being predicted.
        valid_ids: every S2/S3 ID in that split's source files.
        out_dir: output directory.

    Returns:
        Paths of the matching and candidate files.

    Raises:
        ValueError: if any rule is violated; nothing is written in that case.
    """
    s1_list = list(s1_ids)
    s1_set = set(s1_list)
    if len(s1_set) != len(s1_list):
        raise ValueError("s1_ids contains duplicates")
    bad = [s for s in s1_list if not isinstance(s, str) or not s.startswith(config.S1_PREFIX)]
    if bad:
        raise ValueError(f"s1_ids contains non-S1 IDs, e.g. {bad[:3]}")
    for name, mapping in (("matches", matches), ("candidates", candidates)):
        extra = set(mapping) - s1_set
        if extra:
            raise ValueError(f"{name} has S1 IDs not in s1_ids, e.g. {sorted(extra)[:3]}")

    valid = valid_ids if isinstance(valid_ids, (set, frozenset)) else set(valid_ids)
    match_rows: list[tuple[str, str]] = []
    cand_rows: list[tuple[str, str]] = []
    for s1 in s1_list:
        cand = _clean_list(s1, candidates.get(s1, ()), valid, "candidates")
        match = _clean_list(s1, matches.get(s1, ()), valid, "matches")
        missing = set(match) - set(cand)
        if missing:
            raise ValueError(f"matches for {s1} not in its candidates: {sorted(missing)[:3]}")
        match_rows.append((s1, ",".join(match)))
        cand_rows.append((s1, ",".join(cand)))

    out_dir = Path(out_dir)
    match_path = out_dir / config.MATCHING_FILE.name
    cand_path = out_dir / config.CANDIDATE_FILE.name
    _write_two_col(match_path, config.MATCHING_HEADER, match_rows)
    _write_two_col(cand_path, config.CANDIDATE_HEADER, cand_rows)
    return match_path, cand_path
