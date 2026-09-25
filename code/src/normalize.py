"""Name/address normalisation, token and number extraction (CLAUDE.md §6.1).

Design inputs (CP1 audit):

* S1 is clean title case. S2 is often ALL CAPS with abbreviated addresses and Indian
  states/cities (sometimes whole names) in native Indic scripts. S3 adds honorific
  prefixes (Smt, M/s, Dr, Sri), ``X DBA Y`` forms, ``[Limited]`` brackets, abbreviated
  states and shuffled address components.
* Both S2/S3: typos, digit/letter swaps (``SEAF0OD``), accents, ``&`` written as
  ``and``/``+``/``(&)``, reordered or dropped legal suffixes, website-style names
  (``dynamicsmarketing.com``), junk prefixes (``#``, ``--``, ``>>``), double spaces and
  house-number variants (``B3/97/11``, ``C-97/11``, ``NO.-971/2ND``).
* France appears only in test: accents, French legal forms and street words.

Everything here is language-agnostic string processing plus hand-written dictionaries
(allowed by CLAUDE.md §2.1). Country is never used to choose a code path, so an unseen
country flows through unchanged. Indic scripts are transliterated to Latin with one
table shared by all nine ISCII-derived Unicode blocks, which have parallel layouts.
"""

from __future__ import annotations

import re
import unicodedata
import pandas as pd

from . import config

# --- Indic transliteration -----------------------------------------------------------

_INDIC_LO, _INDIC_HI = 0x0900, 0x0D7F
_SOUTH_BASES = {0x0B80, 0x0C00, 0x0C80, 0x0D00}  # Tamil, Telugu, Kannada, Malayalam
_VIRAMA, _NUKTA = 0x4D, 0x3C

_CONSONANTS = {
    0x15: "k", 0x16: "kh", 0x17: "g", 0x18: "gh", 0x19: "ng", 0x1A: "ch", 0x1B: "chh",
    0x1C: "j", 0x1D: "jh", 0x1E: "ny", 0x1F: "t", 0x20: "th", 0x21: "d", 0x22: "dh",
    0x23: "n", 0x24: "t", 0x25: "th", 0x26: "d", 0x27: "dh", 0x28: "n", 0x29: "n",
    0x2A: "p", 0x2B: "ph", 0x2C: "b", 0x2D: "bh", 0x2E: "m", 0x2F: "y", 0x30: "r",
    0x31: "r", 0x32: "l", 0x33: "l", 0x34: "l", 0x35: "v", 0x36: "sh", 0x37: "sh",
    0x38: "s", 0x39: "h", 0x58: "k", 0x59: "kh", 0x5A: "g", 0x5B: "z", 0x5C: "r",
    0x5D: "r", 0x5E: "f", 0x5F: "y",
}
_VOWEL_SIGNS = {
    0x3E: "a", 0x3F: "i", 0x40: "i", 0x41: "u", 0x42: "u", 0x43: "ri", 0x44: "ri",
    0x45: "e", 0x46: "e", 0x47: "e", 0x48: "ai", 0x49: "o", 0x4A: "o", 0x4B: "o",
    0x4C: "au", 0x4E: "e", 0x56: "ai", 0x57: "au", 0x62: "l", 0x63: "l",
}
_INDEPENDENT = {
    0x04: "a", 0x05: "a", 0x06: "a", 0x07: "i", 0x08: "i", 0x09: "u", 0x0A: "u",
    0x0B: "ri", 0x0C: "l", 0x0D: "e", 0x0E: "e", 0x0F: "e", 0x10: "ai", 0x11: "o",
    0x12: "o", 0x13: "o", 0x14: "au", 0x60: "ri", 0x61: "l",
}
_MODIFIERS = {0x00: "n", 0x01: "n", 0x02: "n", 0x03: "h"}
#: Per-script exceptions to the shared table.
_SCRIPT_OVERRIDES = {
    0x0980: {0x4E: ("c", "t"), 0x70: ("c", "r"), 0x71: ("c", "w")},  # Bengali
    0x0A00: {0x70: ("m", "n"), 0x71: ("x", ""), 0x72: ("x", ""), 0x73: ("x", "")},
    0x0B00: {0x71: ("c", "w")},  # Oriya
    0x0D00: {0x54: ("x", "m"), 0x55: ("x", "y"), 0x56: ("x", "l"), 0x4E: ("x", "r"),
             0x7A: ("x", "n"), 0x7B: ("x", "n"), 0x7C: ("x", "r"), 0x7D: ("x", "l"),
             0x7E: ("x", "l"), 0x7F: ("x", "k")},  # Malayalam chillu letters
}


def transliterate_indic(text: str) -> str:
    """Transliterate Indic-script characters to rough lowercase Latin.

    Uses the shared ISCII offset layout of the Devanagari, Bengali, Gurmukhi, Gujarati,
    Oriya, Tamil, Telugu, Kannada and Malayalam blocks. Consonants carry an inherent
    ``a`` unless followed by a vowel sign or virama. The word-final inherent vowel is
    dropped for northern scripts ("राम" -> "ram") and kept for southern ones
    ("తెలంగాణ" -> "telangana"). Non-Indic characters pass through unchanged.
    """
    if not any(_INDIC_LO <= ord(ch) <= _INDIC_HI for ch in text):
        return text
    out: list[str] = []
    pending = False  # an inherent "a" is owed by the last consonant
    pending_south = False
    for ch in text:
        cp = ord(ch)
        if ch in "‌‍":
            continue
        if not (_INDIC_LO <= cp <= _INDIC_HI):
            if pending and pending_south:
                out.append("a")
            pending = False
            out.append(ch)
            continue
        base, off = cp & ~0x7F, cp & 0x7F
        kind_val = _SCRIPT_OVERRIDES.get(base, {}).get(off)
        if kind_val is None:
            if off in _CONSONANTS:
                kind_val = ("c", _CONSONANTS[off])
            elif off == _NUKTA:
                continue
            elif off == _VIRAMA:
                kind_val = ("v", "")
            elif off in _VOWEL_SIGNS:
                kind_val = ("s", _VOWEL_SIGNS[off])
            elif off in _MODIFIERS:
                kind_val = ("m", _MODIFIERS[off])
            elif off in _INDEPENDENT:
                kind_val = ("x", _INDEPENDENT[off])
            elif 0x66 <= off <= 0x6F:
                kind_val = ("x", str(off - 0x66))
            else:
                kind_val = ("x", "")
        kind, val = kind_val
        if kind == "c":
            if pending:
                out.append("a")
            out.append(val)
            pending, pending_south = True, base in _SOUTH_BASES
        elif kind in ("v", "s"):
            pending = False
            out.append(val)
        else:  # modifier / independent vowel / digit / other
            if pending:
                out.append("a")
            pending = False
            out.append(val)
    if pending and pending_south:
        out.append("a")
    return "".join(out)


# --- Generic folding -----------------------------------------------------------------

_LIGATURES = str.maketrans({
    "œ": "oe", "Œ": "oe", "æ": "ae", "Æ": "ae", "ß": "ss", "ø": "o", "Ø": "o",
    "ł": "l", "Ł": "l", "đ": "d", "Đ": "d", "ı": "i", "’": "'", "‘": "'", "`": "'",
    "´": "'", "–": "-", "—": "-", "“": '"', "”": '"',
})


def fold(text: str) -> str:
    """Transliterate Indic scripts, strip accents (NFKD), expand ligatures, lowercase.

    Combining marks are dropped only when they follow a Latin base letter, so other
    scripts are not mangled. ``"Troënes" -> "troenes"``, ``"TRÁDERS" -> "traders"``.
    """
    if not text:
        return ""
    text = transliterate_indic(unicodedata.normalize("NFC", str(text)))
    text = text.translate(_LIGATURES)
    out: list[str] = []
    prev_latin = False
    for ch in unicodedata.normalize("NFKD", text):
        if unicodedata.combining(ch):
            if prev_latin:
                continue
            out.append(ch)
            continue
        prev_latin = ord(ch) < 0x250
        out.append(ch)
    return "".join(out).lower()


_SPACE_RE = re.compile(r"\s+")
_NON_ALNUM_RE = re.compile(r"[^\w]+")
_UNDERSCORE_RE = re.compile(r"_+")


def collapse_ws(text: str) -> str:
    """Collapse whitespace runs to single spaces and strip."""
    return _SPACE_RE.sub(" ", text).strip()


def _join_single_letters(tokens: list[str]) -> list[str]:
    """Join runs of >= 2 single-letter tokens: ``l l c -> llc``, ``s a s -> sas``."""
    out: list[str] = []
    run: list[str] = []
    for tok in tokens + [""]:
        if len(tok) == 1 and tok.isalpha():
            run.append(tok)
            continue
        if len(run) >= 2:
            out.append("".join(run))
        else:
            out.extend(run)
        run = []
        if tok:
            out.append(tok)
    return out


def tokenize(text: str) -> list[str]:
    """Split folded text on non-alphanumerics and join single-letter runs."""
    text = _UNDERSCORE_RE.sub(" ", _NON_ALNUM_RE.sub(" ", text))
    return _join_single_letters(text.split())


def normalize_text(text: str) -> str:
    """Generic normalisation: fold, ``& -> and``, punctuation and whitespace collapse.

    ``"GREEN  TRÁDERS & Co."`` -> ``"green traders and co"``.
    """
    text = fold(text)
    text = re.sub(r"\s*&\s*", " and ", text)
    return " ".join(tokenize(text))


# --- Names ---------------------------------------------------------------------------

_JUNK_PREFIX_RE = re.compile(r"^[^\w(\[]+")
_JUNK_SUFFIX_RE = re.compile(r"[^\w)\].]+$")
_BRACKETS_RE = re.compile(r"[()\[\]{}<>]")
_PLUS_AND_RE = re.compile(r"\s\+\s|\s*&\s*")
_POSSESSIVE_RE = re.compile(r"'s\b")
_DIGIT_IN_WORD_RE = re.compile(r"(?<=[a-z])[01](?=[a-z])")
_DBA_RE = re.compile(
    r"\s(?:dba|d\s*/\s*b\s*/\s*a|d\.b\.a\.?|doing business as|trading as|t/a|aka|"
    r"a\.k\.a\.?)\s*:?\s+"
)
_WEBSITE_RE = re.compile(
    r"^(?:https?://)?(?:www\.)?([a-z0-9][a-z0-9-]*(?:\.[a-z0-9-]+)*?)"
    r"\.(?:com|net|org|in|co\.in|org\.in|fr|biz|info|co|us|io)/?$"
)
_URL_TOKEN_RE = re.compile(r"(?:https?://|www\.)\S+|\S+\.(?:com|net|org|in|fr)\b")

#: Honorific / filler prefixes S3 adds to names. Stripped symmetrically on all sources.
HONORIFICS = frozenset({"ms", "mrs", "mr", "smt", "shri", "sri", "dr", "the", "messrs"})

#: Legal forms -> canonical token. English, Indian and French forms.
_LEGAL_TOKENS = {
    "pvt": "pvt", "private": "pvt", "prvt": "pvt", "pte": "pvt",
    "ltd": "ltd", "limited": "ltd", "ltda": "ltd", "lmt": "ltd",
    "inc": "inc", "incorporated": "inc", "corp": "corp", "corporation": "corp",
    "co": "co", "company": "co", "cie": "co",
    "llc": "llc", "pllc": "llc", "llp": "llp", "lp": "lp", "lllp": "lp",
    "plc": "plc", "pc": "pc", "opc": "opc", "gmbh": "gmbh",
    "sarl": "sarl", "sas": "sas", "sasu": "sasu", "sa": "sa", "eurl": "eurl",
    "sci": "sci", "snc": "snc", "ei": "ei", "eirl": "eirl", "selarl": "selarl",
    "scp": "scp", "sca": "sca",
}
#: Multi-word legal phrases, replaced before token-level matching.
_LEGAL_PHRASES = [
    ("limited liability company", "llc"),
    ("limited liability partnership", "llp"),
    ("one person company", "opc"),
    ("societe a responsabilite limitee", "sarl"),
    ("societe par actions simplifiee", "sas"),
    ("societe anonyme", "sa"),
    ("and cie", "co"),
    ("and co", "co"),
]
#: Native-script legal words; their transliterations are added to _LEGAL_TOKENS below.
_NATIVE_LEGAL = {
    "pvt": ["प्राइवेट", "प्रा", "ప్రైవేట్", "பிரைவேட்", "ಪ್ರೈವೇಟ್", "প্রাইভেট", "પ્રાઇવેટ"],
    "ltd": ["लिमिटेड", "लि", "లిమిటెడ్", "லிமிடெட்", "ಲಿಮಿಟೆಡ್", "লিমিটেড", "લિમિટેડ"],
    "llp": ["एलएलपी"],
    "co": ["कंपनी"],
    "corp": ["कॉर्पोरेशन"],
}
for _canon, _words in _NATIVE_LEGAL.items():
    for _w in _words:
        for _tok in tokenize(fold(_w)):
            _LEGAL_TOKENS.setdefault(_tok, _canon)

#: Families used to decide whether two suffixes *conflict*. "co" is too weak a signal
#: to conflict with anything, so it has no family.
SUFFIX_FAMILY = {
    "pvt": "ltd", "ltd": "ltd", "opc": "ltd", "inc": "inc", "corp": "inc",
    "llc": "llc", "llp": "lp", "lp": "lp", "plc": "plc", "pc": "pc", "gmbh": "gmbh",
    "sarl": "sarl", "eurl": "sarl", "sas": "sas", "sasu": "sas", "sa": "sa",
    "sci": "sci", "snc": "snc", "ei": "ei", "eirl": "ei", "selarl": "selarl",
    "scp": "scp", "sca": "sca",
}


def _strip_junk(text: str) -> str:
    """Remove leading/trailing junk punctuation such as ``#``, ``--``, ``>>``, ``<<``."""
    return _JUNK_SUFFIX_RE.sub("", _JUNK_PREFIX_RE.sub("", text.strip()))


def detect_website(text: str) -> str:
    """Return the domain stem if ``text`` is a website-style name, else ``""``.

    ``"DYNAMICSMARKETING.COM" -> "dynamicsmarketing"``; ``"www.zircon.com" -> "zircon"``.
    Expects folded (lowercase) text.
    """
    m = _WEBSITE_RE.match(text.strip())
    if not m:
        return ""
    return re.sub(r"[^a-z0-9]", "", m.group(1).split(".")[-1] if "." in m.group(1)
                  else m.group(1))


def split_dba(text: str) -> tuple[str, str]:
    """Split ``"X DBA Y"`` into ``(Y, X)``: the trading name and the alias before it.

    Returns ``(text, "")`` when there is no DBA marker.
    """
    parts = _DBA_RE.split(f" {text} ", maxsplit=1)
    if len(parts) == 2 and parts[1].strip():
        return parts[1].strip(), parts[0].strip()
    return text, ""


def repair_digit_typos(token: str) -> str:
    """Replace 0/1 sandwiched between letters by o/l: ``seaf0od -> seafood``."""
    return _DIGIT_IN_WORD_RE.sub(lambda m: "o" if m.group(0) == "0" else "l", token)


def split_legal_suffix(tokens: list[str]) -> tuple[list[str], list[str]]:
    """Split name tokens into ``(core_tokens, canonical_suffix_tokens)``.

    Legal forms are removed wherever they occur, because S2/S3 reorder them
    (``"LLP Casa India"``, ``"Private Vasai It Ltd"``). If removing them would leave
    nothing, the tokens are kept as the core.
    """
    text = " ".join(tokens)
    for phrase, canon in _LEGAL_PHRASES:
        text = re.sub(rf"\b{phrase}\b", f"__{canon}__", text)
    core: list[str] = []
    suffix: list[str] = []
    for tok in text.split():
        if tok.startswith("__") and tok.endswith("__"):
            suffix.append(tok.strip("_"))
        elif tok in _LEGAL_TOKENS:
            suffix.append(_LEGAL_TOKENS[tok])
        else:
            core.append(tok)
    if not core:
        core = [t.strip("_") for t in text.split()]
    return core, sorted(set(suffix))


def acronym(name_core: str) -> str:
    """Initial-letter acronym of a multi-token core (``"international business
    machines" -> "ibm"``); ``""`` for single-token names."""
    toks = [t for t in name_core.split() if t not in ("and", "of", "de", "et")]
    return "".join(t[0] for t in toks) if len(toks) >= 2 else ""


def normalize_name(raw: str) -> dict:
    """Normalise one business name.

    Returns a dict with ``name_norm`` (full normalised name), ``name_core`` (legal
    forms and honorifics removed), ``name_suffix`` (sorted canonical legal forms),
    ``name_alias`` (text before a DBA marker), ``name_compact`` (core without spaces),
    ``is_website``, ``acronym`` and ``first_token``.
    """
    text = fold(raw)
    # "Zircon & Brothers Private Limited | www.zircon.com": keep the non-URL part.
    parts = [p.strip() for p in re.split(r"\s\|\s|\|", text) if p.strip()]
    website = ""
    if len(parts) > 1:
        non_url = [p for p in parts if not _URL_TOKEN_RE.search(p)]
        url = [p for p in parts if _URL_TOKEN_RE.search(p)]
        if non_url:
            text = " ".join(non_url)
            website = detect_website(url[0]) if url else ""
    text = _BRACKETS_RE.sub(" ", text)
    text = _strip_junk(text)
    text, alias = split_dba(text)
    # Honorific before a website name: "smt annapurnasafetyindia.com".
    words = text.split()
    while len(words) > 1 and re.sub(r"[^a-z]", "", words[0]) in HONORIFICS | {"m/s"}:
        words = words[1:]
    stem = detect_website(" ".join(words))
    is_website = bool(stem)
    if stem:
        text = stem
    text = _PLUS_AND_RE.sub(" and ", f" {text} ")
    text = _POSSESSIVE_RE.sub("s", text)
    tokens = [repair_digit_typos(t) for t in tokenize(text)]
    tokens = ["and" if t == "et" else t for t in tokens]
    while len(tokens) > 1 and tokens[0] in HONORIFICS:
        tokens = tokens[1:]
    # "(&)" and dangling "and" carry no identity once brackets are gone.
    while tokens and tokens[-1] == "and":
        tokens = tokens[:-1]
    core, suffix = split_legal_suffix(tokens)
    while len(core) > 1 and core[-1] == "and":
        core = core[:-1]
    name_core = " ".join(core)
    return {
        "name_norm": " ".join(tokens),
        "name_core": name_core,
        "name_suffix": " ".join(suffix),
        "name_alias": normalize_text(alias) if alias else "",
        "name_compact": name_core.replace(" ", ""),
        "is_website": is_website or bool(website),
        "acronym": acronym(name_core),
        "first_token": core[0] if core else "",
    }


# --- Addresses -----------------------------------------------------------------------

_ADDRESS_ABBREV = {
    "rd": "road", "st": "street", "str": "street", "ave": "avenue", "av": "avenue",
    "avn": "avenue", "blvd": "boulevard", "bd": "boulevard", "bvd": "boulevard",
    "ln": "lane", "dr": "drive", "ct": "court", "pl": "place", "hwy": "highway",
    "pkwy": "parkway", "cir": "circle", "ter": "terrace", "trl": "trail",
    "sq": "square", "mt": "mount", "ft": "fort", "apt": "apartment", "ste": "suite",
    "fl": "floor", "flr": "floor", "bldg": "building", "nr": "near", "opp": "opposite",
    "r": "rue", "chem": "chemin", "imp": "impasse", "all": "allee", "rte": "route",
    "fbg": "faubourg", "sec": "sector", "sect": "sector", "extn": "extension",
    "ext": "extension", "ph": "phase", "ind": "industrial", "indl": "industrial",
    "stn": "station", "mkt": "market", "n": "north", "s": "south", "e": "east",
    "w": "west", "ne": "northeast", "nw": "northwest", "se": "southeast",
    "sw": "southwest",
}
#: Number markers that carry no information once digits are extracted.
_NUMBER_MARKERS = frozenset({"no", "nos", "num", "number", "numero"})
_ORDINAL_RE = re.compile(r"^(\d+)(?:st|nd|rd|th|er|eme|e)$")
_LANDMARK_RE = re.compile(
    r"\b(?:near|opposite|behind|beside|besides|next to|in front of|adjacent to|adj|"
    r"adjoining|pres de|pres du|pres des|a cote de|a cote du|en face de|en face du|"
    r"derriere)\b"
)
_CARE_OF_RE = re.compile(r"\bc\s*/\s*o\b\.?")
_ADDR_JUNK_RE = re.compile(r"^[#*>\-~=_.,;:\s]+")

#: State / region names and codes -> canonical code. Only a *whole* comma-separated
#: component is matched, so street words are never touched.
_REGIONS: dict[str, list[str]] = {
    # US states + DC
    "al": ["alabama"], "ak": ["alaska"], "az": ["arizona"], "ar": ["arkansas",
    "arunachal pradesh"], "ca": ["california"], "co": ["colorado"], "ct": ["connecticut"],
    "de": ["delaware"], "fl": ["florida"], "ga": ["georgia", "goa"], "hi": ["hawaii"],
    "id": ["idaho"], "il": ["illinois"], "in": ["indiana"], "ia": ["iowa"],
    "ks": ["kansas"], "ky": ["kentucky"], "la": ["louisiana", "ladakh"],
    "me": ["maine"], "md": ["maryland"], "ma": ["massachusetts"], "mi": ["michigan"],
    "mn": ["minnesota", "manipur"], "ms": ["mississippi"], "mo": ["missouri"],
    "mt": ["montana"], "ne": ["nebraska"], "nv": ["nevada"], "nh": ["new hampshire"],
    "nj": ["new jersey"], "nm": ["new mexico"], "ny": ["new york"],
    "nc": ["north carolina"], "nd": ["north dakota"], "oh": ["ohio"],
    "ok": ["oklahoma"], "or": ["oregon"], "pa": ["pennsylvania"],
    "ri": ["rhode island"], "sc": ["south carolina"], "sd": ["south dakota"],
    "tn": ["tennessee", "tamil nadu", "tamilnadu", "tamilnatu"], "tx": ["texas"],
    "ut": ["utah", "uttarakhand", "uttaranchal"], "vt": ["vermont"],
    "va": ["virginia"], "wa": ["washington"], "wv": ["west virginia"],
    "wi": ["wisconsin"], "wy": ["wyoming"], "dc": ["district of columbia"],
    # India states / UTs (vehicle-style codes, as S3 uses)
    "an": ["andaman and nicobar islands", "andaman and nicobar"],
    "ap": ["andhra pradesh"], "as": ["assam"], "br": ["bihar"],
    "cg": ["chhattisgarh", "chattisgarh"], "ch": ["chandigarh"],
    "dl": ["delhi", "nct of delhi", "new delhi"], "gj": ["gujarat"],
    "hr": ["haryana"], "hp": ["himachal pradesh"],
    "jk": ["jammu and kashmir", "jammu kashmir"], "jh": ["jharkhand"],
    "ka": ["karnataka"], "kl": ["kerala"], "mp": ["madhya pradesh"],
    "mh": ["maharashtra"], "ml": ["meghalaya"], "mz": ["mizoram"], "nl": ["nagaland"],
    "od": ["odisha", "orissa"], "py": ["puducherry", "pondicherry"],
    "pb": ["punjab"], "rj": ["rajasthan"], "sk": ["sikkim"],
    "tg": ["telangana", "ts"], "tr": ["tripura"], "up": ["uttar pradesh"],
    "wb": ["west bengal"], "dn": ["dadra and nagar haveli and daman and diu",
    "dadra and nagar haveli"], "ld": ["lakshadweep"],
    # France: 13 metropolitan regions and their departements
    "fr-ara": ["auvergne rhone alpes", "ain", "allier", "ardeche", "cantal", "drome",
               "isere", "loire", "haute loire", "puy de dome", "rhone", "savoie",
               "haute savoie"],
    "fr-bfc": ["bourgogne franche comte", "cote d or", "doubs", "jura", "nievre",
               "haute saone", "saone et loire", "yonne", "territoire de belfort"],
    "fr-bre": ["bretagne", "cotes d armor", "finistere", "ille et vilaine", "morbihan"],
    "fr-cvl": ["centre val de loire", "cher", "eure et loir", "indre",
               "indre et loire", "loir et cher", "loiret"],
    "fr-cor": ["corse", "corse du sud", "haute corse"],
    "fr-ges": ["grand est", "ardennes", "aube", "marne", "haute marne",
               "meurthe et moselle", "meuse", "moselle", "bas rhin", "haut rhin",
               "vosges"],
    "fr-hdf": ["hauts de france", "aisne", "nord", "oise", "pas de calais", "somme"],
    "fr-idf": ["ile de france", "seine et marne", "yvelines", "essonne",
               "hauts de seine", "seine saint denis", "val de marne", "val d oise"],
    "fr-nor": ["normandie", "calvados", "eure", "manche", "orne", "seine maritime"],
    "fr-naq": ["nouvelle aquitaine", "charente", "charente maritime", "correze",
               "creuse", "dordogne", "gironde", "landes", "lot et garonne",
               "pyrenees atlantiques", "deux sevres", "vienne", "haute vienne"],
    "fr-occ": ["occitanie", "ariege", "aude", "aveyron", "gard", "haute garonne", "gers",
               "herault", "lot", "lozere", "hautes pyrenees", "pyrenees orientales",
               "tarn", "tarn et garonne"],
    "fr-pdl": ["pays de la loire", "loire atlantique", "maine et loire", "mayenne",
               "sarthe", "vendee"],
    "fr-pac": ["provence alpes cote d azur", "alpes de haute provence", "hautes alpes",
               "alpes maritimes", "bouches du rhone", "var", "vaucluse"],
}
#: Native-script state names seen in S2; transliterated at import time.
_NATIVE_REGIONS = {
    "up": ["उत्तर प्रदेश"], "mh": ["महाराष्ट्र"], "dl": ["दिल्ली", "नई दिल्ली"],
    "hr": ["हरियाणा"], "mp": ["मध्य प्रदेश"], "rj": ["राजस्थान"], "br": ["बिहार"],
    "gj": ["ગુજરાત", "गुजरात"], "ka": ["ಕರ್ನಾಟಕ"], "tn": ["தமிழ்நாடு", "तमिलनाडु"],
    "tg": ["తెలంగాణ"], "ap": ["ఆంధ్ర ప్రదేశ్"], "kl": ["കേരളം"], "wb": ["পশ্চিমবঙ্গ"],
    "pb": ["ਪੰਜਾਬ"], "od": ["ଓଡ଼ିଶା"], "jh": ["झारखंड"], "cg": ["छत्तीसगढ़"],
    "ut": ["उत्तराखंड"], "hp": ["हिमाचल प्रदेश"], "as": ["অসম"], "ga": ["गोवा"],
    "jk": ["जम्मू और कश्मीर"],
}
REGION_ALIASES: dict[str, str] = {}
for _code, _names in _REGIONS.items():
    REGION_ALIASES.setdefault(_code.replace("-", " "), _code)
    for _n in _names:
        REGION_ALIASES.setdefault(_n, _code)
for _code, _names in _NATIVE_REGIONS.items():
    for _n in _names:
        REGION_ALIASES.setdefault(" ".join(tokenize(fold(_n))), _code)
# Two-letter codes as written in addresses ("TX", "MH").
for _code in list(_REGIONS):
    if "-" not in _code:
        REGION_ALIASES.setdefault(_code, _code)


def _expand_address_tokens(tokens: list[str]) -> list[str]:
    """Expand street abbreviations, drop number markers, turn ordinals into digits.

    ``st``/``ste`` as the first token of a multi-token component mean saint/sainte
    (``"St Louis"``); elsewhere they mean street/suite.
    """
    out: list[str] = []
    for i, tok in enumerate(tokens):
        m = _ORDINAL_RE.match(tok)
        if m:
            out.append(m.group(1))
            continue
        if tok in _NUMBER_MARKERS:
            continue
        nxt = tokens[i + 1] if i + 1 < len(tokens) else ""
        if i == 0 and tok in ("st", "ste") and nxt.isalpha():
            out.append("saint" if tok == "st" else "sainte")
            continue
        if tok.isdigit():
            out.append(_strip_leading_zeros(tok))
            continue
        # Single-letter directions only before a word ("N Main"), not "E-11" blocks.
        if len(tok) == 1 and not nxt.isalpha():
            out.append(tok)
            continue
        out.append(_ADDRESS_ABBREV.get(tok, tok))
    return out


def _strip_leading_zeros(digits: str) -> str:
    """``"021744" -> "21744"``; ``"000" -> "0"``."""
    return digits.lstrip("0") or "0"


def extract_digit_tokens(text: str) -> list[str]:
    """Return all-digit tokens in order, leading zeros stripped, deduplicated.

    Generic (no country-specific regexes). Only whole numeric tokens count, so the
    letter prefixes S2/S3 add to house numbers do not create spurious numbers:
    ``"B3/97/11" -> ["97", "11"]`` (same as ``"C-97/11"``), ``"NO.-971/2ND"`` (after
    ordinal folding) ``-> ["971", "2"]``.
    """
    seen: dict[str, None] = {}
    for tok in re.split(r"[^0-9a-z]+", text.lower()):
        m = _ORDINAL_RE.match(tok)
        if m:
            tok = m.group(1)
        if tok.isdigit():
            seen.setdefault(_strip_leading_zeros(tok), None)
    return list(seen)


def normalize_address(raw: str) -> dict:
    """Normalise one address.

    Components are split on commas. Whole components that are a state/region name or
    code go to ``region``. Landmark phrases (``near``/``opp``/``behind``/``pres de``...)
    and ``C/O`` parts go to ``landmark`` so they do not pollute street similarity.

    Returns ``addr_norm`` (street + locality tokens), ``landmark``, ``region``
    (canonical codes, space-joined), ``digits`` (space-joined digit tokens outside
    landmarks), ``house_no`` (first digit token), ``long_nums`` (digit tokens with
    >= 5 digits: ZIP / PIN / CP-like) and ``addr_empty``.
    """
    text = fold(raw)
    text = _ADDR_JUNK_RE.sub("", text.strip())
    text = _CARE_OF_RE.sub(" near ", text)
    text = text.replace("&", " and ").replace("'", " ")
    kept: list[str] = []
    landmarks: list[str] = []
    regions: list[str] = []
    for comp in re.split(r"[,;|\n]", text):
        tokens = _expand_address_tokens(tokenize(comp))
        if not tokens:
            continue
        joined = " ".join(tokens)
        region = REGION_ALIASES.get(joined)
        if region is not None:
            regions.append(region)
            continue
        m = _LANDMARK_RE.search(joined)
        if m:
            before, after = joined[: m.start()].strip(), joined[m.end():].strip()
            if after:
                landmarks.append(after)
            if before:
                kept.append(before)
            continue
        kept.append(joined)
    addr_norm = " ".join(kept)
    digits = extract_digit_tokens(addr_norm)
    return {
        "addr_norm": addr_norm,
        "landmark": " ".join(landmarks),
        "region": " ".join(dict.fromkeys(regions)),
        "digits": " ".join(digits),
        "house_no": digits[0] if digits else "",
        "long_nums": " ".join(d for d in digits if len(d) >= 5),
        "addr_empty": not (addr_norm or landmarks),
    }


# --- Frames --------------------------------------------------------------------------

NAME_FIELDS = ("name_norm", "name_core", "name_suffix", "name_alias", "name_compact",
               "is_website", "acronym", "first_token")
ADDRESS_FIELDS = ("addr_norm", "landmark", "region", "digits", "house_no", "long_nums",
                  "addr_empty")


def _apply_unique(series: pd.Series, fn, fields: tuple[str, ...]) -> pd.DataFrame:
    """Apply ``fn`` once per distinct value of ``series`` and broadcast the result."""
    codes, uniques = pd.factorize(series, sort=False)
    rows = [fn(u) for u in uniques]
    table = pd.DataFrame(rows, columns=list(fields))
    return table.iloc[codes].reset_index(drop=True)


def normalize_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Return ``df`` plus every normalised name/address column.

    Each distinct name/address string is normalised once. Duplicates are common:
    38% of S1 names repeat. The index is reset; the original columns are kept.
    """
    df = df.reset_index(drop=True)
    names = _apply_unique(df[config.NAME_COL].fillna(""), normalize_name, NAME_FIELDS)
    addrs = _apply_unique(df[config.ADDRESS_COL].fillna(""), normalize_address,
                          ADDRESS_FIELDS)
    out = pd.concat([df, names, addrs], axis=1)
    out["is_website"] = out["is_website"].astype(bool)
    out["addr_empty"] = out["addr_empty"].astype(bool)
    return out
