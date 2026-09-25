"""Small deterministic synthetic ER dataset for tests (no real data is ever loaded).

Mimics the CP1 audit patterns at toy scale: clean S1 records; S2/S3 copies with caps,
abbreviations, dropped/reordered suffixes, typos, honorific prefixes, DBA forms, junk
prefixes and empty addresses; S1 "chain" names shared across different addresses;
orphan S2/S3 records; ~6% singletons; and a third country with French-style records.
"""

from __future__ import annotations

import random

import pandas as pd

_WORDS = ["green", "apex", "pioneer", "casa", "vision", "prime", "happy", "north",
          "swastik", "awa", "kasturi", "zephay", "marina", "brahma", "sterling", "vasai",
          "orion", "delta", "lotus", "summit", "harbor", "cedar", "maple", "royal",
          "golden", "silver", "blue", "crystal", "falcon", "tiger", "eagle", "river"]
_TRADES = ["traders", "seafood", "telecom", "health", "labs", "partners", "exports",
           "hospitality", "infotech", "pharmacie", "consultants", "retail", "builders"]
_STREETS = ["lincoln", "cotten", "newton", "tallgrass", "ellis", "colbert", "dieppe",
            "velachery", "gandhi", "nehru", "tondu", "roosevelt", "cherry", "friendly"]
_COUNTRIES = {
    "US": {"suffix": ["LLC", "Inc", "Corp", ""], "type": [("Road", "RD"), ("Street", "ST"),
           ("Avenue", "AVE"), ("Drive", "DR")], "cities": ["Tyler", "Peoria", "Austin"],
           "states": [("Texas", "TX"), ("Illinois", "IL"), ("Oregon", "OR")]},
    "India": {"suffix": ["Private Limited", "Pvt Ltd", "LLP", ""], "type": [
              ("Road", "RD"), ("Nagar", "NAGAR"), ("Street", "ST")],
              "cities": ["Chennai", "Mumbai", "Kanpur"],
              "states": [("Tamil Nadu", "TN"), ("Maharashtra", "MH"), ("Uttar Pradesh", "UP")]},
    "France": {"suffix": ["SARL", "SAS", "EURL", ""], "type": [("Rue", "R."),
               ("Avenue", "AVE"), ("Boulevard", "BD")], "cities": ["Bordeaux", "Lille", "Nantes"],
               "states": [("Nouvelle-Aquitaine", "Gironde"), ("Hauts-de-France", "Nord"),
                          ("Pays de la Loire", "Loire-Atlantique")]},
}


def _typo(rng: random.Random, word: str) -> str:
    """Swap two adjacent letters or substitute 0 for o."""
    if "o" in word and rng.random() < 0.3:
        return word.replace("o", "0", 1)
    if len(word) > 3:
        i = rng.randrange(1, len(word) - 2)
        return word[:i] + word[i + 1] + word[i] + word[i + 2:]
    return word


def _make_business(rng: random.Random, country: str, name: str | None = None) -> dict:
    """One clean business: title-case name with suffix, street-first address."""
    spec = _COUNTRIES[country]
    if name is None:
        core = f"{rng.choice(_WORDS).title()} {rng.choice(_TRADES).title()}"
        suffix = rng.choice(spec["suffix"])
        name = f"{core} {suffix}".strip()
    street, abbr = rng.choice(spec["type"])
    state = rng.choice(spec["states"])
    num = rng.randint(1, 9999)
    sname = rng.choice(_STREETS).title()
    return {"name": name, "num": num, "sname": sname, "stype": street, "sabbr": abbr,
            "city": rng.choice(spec["cities"]), "state": state, "country": country}


def _clean_address(b: dict) -> str:
    """S1-style address."""
    if b["stype"] == "Rue":
        return f"{b['num']} Rue {b['sname']}, {b['city']}, {b['state'][0]}"
    return f"{b['num']} {b['sname']} {b['stype']}, {b['city']}, {b['state'][0]}"


def _noisy_copy(rng: random.Random, b: dict, source: int) -> tuple[str, str]:
    """An S2 (source=2) or S3 (source=3) style noisy copy of business ``b``."""
    tokens = b["name"].split()
    if rng.random() < 0.3 and len(tokens) > 2:  # drop suffix
        tokens = tokens[:-1] if tokens[-1] not in ("Limited", "Ltd") else tokens[:-2]
    if rng.random() < 0.25:
        tokens[0] = _typo(rng, tokens[0])
    name = " ".join(tokens)
    if source == 2:
        if rng.random() < 0.4:
            name = name.upper()
        if rng.random() < 0.1:
            name = "-- " + name
        addr = f"{b['num']} {b['sname'].upper()} {b['sabbr']}, {b['city'].upper()}, {b['state'][0]}"
    else:
        if rng.random() < 0.2:
            name = rng.choice(["Smt ", "M/s ", "Dr ", ">> "]) + name
        if rng.random() < 0.1:
            name = f"Lyravera DBA {name}"
        addr = f"{b['num']} {b['sname']} {b['stype']}, {b['state'][1]}, {b['city']}"
    if rng.random() < 0.05:
        addr = ""
    return name, addr


def make_dataset(n_s1: int = 300, seed: int = 0, countries: tuple[str, ...] = ("US", "India"),
                 orphan_frac: float = 0.25, singleton_frac: float = 0.06):
    """Return ``(s1, s2, s3, truth)`` frames/dict with the real file schema.

    ``truth`` maps every S1 ID to its list of S2/S3 IDs (empty for singletons).
    """
    rng = random.Random(seed)
    s1_rows, s2_rows, s3_rows = [], [], []
    truth: dict[str, list[str]] = {}
    chain_names: list[str] = []
    next_id = [1000]

    def new_id(prefix: str) -> str:
        """Unique numeric-looking ID with a source prefix."""
        next_id[0] += rng.randint(1, 50)
        return f"{prefix}{next_id[0]}"

    for i in range(n_s1):
        country = countries[i % len(countries)]
        # ~1 in 8 S1s reuse a chain name (same name, different address).
        name = rng.choice(chain_names) if chain_names and rng.random() < 0.12 else None
        b = _make_business(rng, country, name)
        if name is None and rng.random() < 0.2:
            chain_names.append(b["name"])
        sid = new_id("S1-")
        s1_rows.append((sid, b["name"], _clean_address(b), country))
        matches = []
        if rng.random() >= singleton_frac:
            for _ in range(rng.choice([1, 2, 2, 3, 3, 4])):
                src = rng.choice([2, 3])
                cid = new_id(f"S{src}-")
                n, a = _noisy_copy(rng, b, src)
                (s2_rows if src == 2 else s3_rows).append((cid, n, a, country))
                matches.append(cid)
        truth[sid] = matches
    n_orphans = int(orphan_frac * (len(s2_rows) + len(s3_rows)) / (1 - orphan_frac))
    for j in range(n_orphans):
        country = countries[j % len(countries)]
        b = _make_business(rng, country)
        src = rng.choice([2, 3])
        n, a = _noisy_copy(rng, b, src)
        (s2_rows if src == 2 else s3_rows).append((new_id(f"S{src}-"), n, a, country))
    cols = ["entity_id", "business_name", "business_address", "country"]
    s1 = pd.DataFrame(s1_rows, columns=cols)
    s2 = pd.DataFrame(s2_rows, columns=cols).sample(frac=1, random_state=seed).reset_index(drop=True)
    s3 = pd.DataFrame(s3_rows, columns=cols).sample(frac=1, random_state=seed).reset_index(drop=True)
    return s1, s2, s3, truth


def write_split(root, split: str, s1, s2, s3, truth=None) -> None:
    """Write frames as the real TSV files under ``root/<split>/``."""
    d = root / split
    d.mkdir(parents=True, exist_ok=True)
    for i, df in enumerate((s1, s2, s3), start=1):
        df.to_csv(d / f"{split}_source{i}.tsv", sep="\t", index=False)
    if truth is not None:
        gt = pd.DataFrame({"source1_entity_id": list(truth),
                           "matched_entity_ids": [",".join(v) for v in truth.values()]})
        gt.to_csv(d / f"{split}_ground_truth.tsv", sep="\t", index=False)
