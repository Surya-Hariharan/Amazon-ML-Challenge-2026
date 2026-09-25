"""Tests for src.normalize on hand-written strings mimicking the CP1 noise patterns."""

import pandas as pd
import pytest

from src.normalize import (
    acronym,
    detect_website,
    extract_digit_tokens,
    fold,
    normalize_address,
    normalize_frame,
    normalize_name,
    normalize_text,
    split_dba,
    transliterate_indic,
)


# --- generic folding -------------------------------------------------------------

@pytest.mark.parametrize(
    "raw, expected",
    [
        ("22 Avenue des Troënes", "22 avenue des troenes"),
        ("GREEN  TRÁDERS & Co.", "green traders and co"),
        ("Œuvre Française", "oeuvre francaise"),
        ("  multiple   spaces\there ", "multiple spaces here"),
    ],
)
def test_normalize_text(raw, expected):
    """Accents stripped, lowercase, & -> and, punctuation/whitespace collapsed."""
    assert normalize_text(raw) == expected


def test_fold_keeps_non_latin_marks():
    """Accent stripping only drops marks after Latin letters, not other scripts."""
    assert fold("É") == "e"
    assert "́" in fold("έ")  # Greek epsilon keeps its tonos


@pytest.mark.parametrize(
    "native, latin",
    [
        ("राम", "ram"),
        ("मार्केटिंग", "marketing"),
        ("తెలంగాణ", "telangana"),
        ("हरियाणा", "hariyana"),
        ("उत्तर प्रदेश", "uttar pradesh"),
        ("१२३", "123"),
    ],
)
def test_transliterate_indic(native, latin):
    """Shared ISCII-offset table across scripts; schwa rules north vs south."""
    assert transliterate_indic(native) == latin


# --- names -----------------------------------------------------------------------

def test_digit_letter_swap_and_caps():
    """'HAPPY  SEAF0OD' (all caps, double space, 0-for-O) == 'Happy Seafood'."""
    assert normalize_name("HAPPY  SEAF0OD")["name_core"] == "happy seafood"
    assert normalize_name("# 4 EH Inte1ligence LLC")["name_core"] == "4 eh intelligence"


def test_honorific_and_bracket_suffix():
    """'Smt Green Traders [Limited]' -> core 'green traders', suffix ltd."""
    n = normalize_name("Smt Green Traders [Limited]")
    assert n["name_core"] == "green traders"
    assert n["name_suffix"] == "ltd"
    assert normalize_name("M/s Awa Health")["name_core"] == "awa health"


def test_dba_extraction():
    """'X DBA Y' keeps Y as the name and X as the alias."""
    n = normalize_name("Lyravera DBA Green Traders Limited")
    assert n["name_core"] == "green traders"
    assert n["name_alias"] == "lyravera"
    assert split_dba("fluxarc dba: alzheimer union ei") == ("alzheimer union ei", "fluxarc")
    assert split_dba("no marker here") == ("no marker here", "")


@pytest.mark.parametrize(
    "variant", ["Kasturi & Co", "KASTURI AND  CO", "Kasturi + Co", "Kasturi (&)", "kasturi & co"]
)
def test_ampersand_variants_share_core(variant):
    """&, and, +, (&) all reduce to the same core."""
    assert normalize_name(variant)["name_core"] == "kasturi"


@pytest.mark.parametrize(
    "variant",
    ["Pioneer All Connection LLC", "Pioneer All Connection L.L.C.", "LLC Pioneer All Connection",
     "-- Pioneer All Connection, llc", ">> PIONEER ALL CONNECTION (LLC)"],
)
def test_suffix_forms_and_positions(variant):
    """Suffix recognised with dots, reordered to the front, and behind junk prefixes."""
    n = normalize_name(variant)
    assert n["name_core"] == "pioneer all connection"
    assert n["name_suffix"] == "llc"


def test_indian_and_french_suffixes():
    """Private Limited / Pvt Ltd / reordered; French SARL, SAS, EURL, SA, SASU, SCI."""
    for v in ["Vasai It Private Limited", "Private Vasai It Ltd", "VASAI IT PVT. LTD."]:
        n = normalize_name(v)
        assert n["name_core"] == "vasai it"
        assert n["name_suffix"] == "ltd pvt"
    for form, canon in [("SARL", "sarl"), ("SAS", "sas"), ("EURL", "eurl"), ("SA", "sa"),
                        ("SASU", "sasu"), ("SCI", "sci"), ("S.A.S.", "sas")]:
        n = normalize_name(f"Bordeaux Parents {form}")
        assert n["name_core"] == "bordeaux parents", form
        assert n["name_suffix"] == canon, form
    assert normalize_name("Locataires & Cie SARL")["name_suffix"] == "co sarl"


def test_native_script_name():
    """A Devanagari name transliterates and its legal words become suffixes."""
    n = normalize_name("राम मार्केटिंग प्राइवेट लिमिटेड")
    assert n["name_core"] == "ram marketing"
    assert n["name_suffix"] == "ltd pvt"


def test_website_names():
    """Website-style names are flagged and reduced to their domain stem."""
    assert detect_website("dynamicsmarketing.com") == "dynamicsmarketing"
    assert detect_website("www.zircon.co.in") == "zircon"
    assert detect_website("green traders") == ""
    n = normalize_name("DYNAMICSMARKETING.COM")
    assert n["is_website"] and n["name_core"] == "dynamicsmarketing"
    n = normalize_name("Smt annapurnasafetyindia.com")
    assert n["is_website"] and n["name_core"] == "annapurnasafetyindia"
    n = normalize_name("Zircon  & Brothers Private Limited | www.zircon.com")
    assert n["is_website"] and n["name_core"] == "zircon and brothers"
    n = normalize_name("c0mmissiononsanitation.com")
    assert n["name_core"] == "commissiononsanitation"


def test_acronym():
    """Acronym of a multi-token core; empty for single tokens."""
    assert acronym("international business machines") == "ibm"
    assert normalize_name("International Business Machines Corp")["acronym"] == "ibm"
    assert acronym("kasturi") == ""


def test_name_never_empty_core():
    """A name made only of legal words keeps them as the core."""
    assert normalize_name("LLC")["name_core"] == "llc"
    assert normalize_name("")["name_core"] == ""


# --- addresses -------------------------------------------------------------------

def test_french_address():
    """Accents stripped, region component split off."""
    a = normalize_address("22 Avenue des Troënes, Saint-Herblain, Pays de la Loire")
    assert a["addr_norm"] == "22 avenue des troenes saint herblain"
    assert a["region"] == "fr-pdl"
    assert a["house_no"] == "22"


def test_french_abbreviations_match_full_form():
    """'NO 37 R. COLBERT' == '37 Rue Colbert'; departement and region share a code."""
    a = normalize_address("NO 37 R. COLBERT, BORDEAUX, Gironde")
    b = normalize_address("37 Rue Colbert, Bordeaux, Nouvelle-Aquitaine")
    assert a["addr_norm"] == b["addr_norm"] == "37 rue colbert bordeaux"
    assert a["region"] == b["region"] == "fr-naq"
    c = normalize_address("12 BD DE LA PAIX, ST HERBLAIN")
    assert c["addr_norm"] == "12 boulevard de la paix saint herblain"


def test_us_abbreviations_and_junk():
    """'##1918 LINCOLN AVE' == '1918 Lincoln Avenue'; state name == state code."""
    a = normalize_address("##1918 LINCOLN AVE, PEORIA, IL")
    b = normalize_address("1918 Lincoln Avenue, Illinois, Peoria")
    assert a["addr_norm"] == "1918 lincoln avenue peoria"
    assert b["addr_norm"] == "1918 lincoln avenue peoria"
    assert a["region"] == b["region"] == "il"
    assert normalize_address("123 N Main St")["addr_norm"] == "123 north main street"


def test_indian_states_native_and_abbreviated():
    """Native-script, full and abbreviated state names map to one code."""
    for s in ["Tamil Nadu", "TN", "தமிழ்நாடு"]:
        assert normalize_address(f"No 9 Main Road, Chennai, {s}")["region"] == "tn", s
    for s in ["Uttar Pradesh", "UP", "उत्तर प्रदेश"]:
        assert normalize_address(f"Ganga Ganj, {s}")["region"] == "up", s
    assert normalize_address("P No 145, తెలంగాణ")["region"] == "tg"


def test_landmark_split():
    """Landmark phrases leave the street text and go to 'landmark'."""
    a = normalize_address("S.V. Road Vile Parle West Opp. Jain Mandir, Mumbai, Maharashtra")
    assert a["landmark"] == "jain mandir"
    assert "jain" not in a["addr_norm"] and "sv road" in a["addr_norm"]
    b = normalize_address("H. No.-971/2Nd Behind Novelty Cinema, North Delhi, Delhi")
    assert b["landmark"] == "novelty cinema"
    c = normalize_address("Near SBI ATM, 5 MG Road")
    assert c["landmark"] == "sbi atm" and c["addr_norm"] == "5 mg road"
    d = normalize_address("12 Rue X, pres de la gare")
    assert d["landmark"] == "la gare"
    e = normalize_address("C/O Bablu Singh, Nalanda, Bihar")
    assert e["landmark"] == "bablu singh" and e["region"] == "br"


def test_house_number_variants_agree():
    """'B3/97/11', 'C-97/11' and '97/11' give the same digit tokens."""
    for v in ["Sr No. B3/97/11, Rising Heights", "Sr No. C-97/11, Rising Heights",
              "SR NO. 97/11, RISING HEIGHTS"]:
        a = normalize_address(v)
        assert a["digits"] == "97 11", v
        assert a["house_no"] == "97", v
    assert extract_digit_tokens("h no 971 2nd") == ["971", "2"]
    assert extract_digit_tokens("021744 hustlers ridge") == ["21744"]


def test_long_numbers_generic():
    """5-6 digit tokens (ZIP / CP / PIN) are captured without country rules."""
    assert normalize_address("Lille 59000")["long_nums"] == "59000"
    assert normalize_address("Mumbai 400056")["long_nums"] == "400056"
    assert normalize_address("12 Main St")["long_nums"] == ""


def test_empty_address():
    """Empty or region-only addresses are flagged empty."""
    assert normalize_address("")["addr_empty"] is True
    assert normalize_address("HR")["addr_empty"] is True
    assert normalize_address("5 Main St")["addr_empty"] is False


# --- frames ----------------------------------------------------------------------

def test_normalize_frame_columns_and_open_country():
    """All fields added; an unseen country flows through untouched."""
    df = pd.DataFrame({
        "entity_id": ["S1-1", "S1-2", "S1-3"],
        "business_name": ["Green Traders Ltd", "Green Traders Ltd", "Bordeaux Parents SAS"],
        "business_address": ["No 9 Main Road, TN", "", "107 Rue du Tondu, Bordeaux"],
        "country": ["India", "India", "Atlantis"],
    })
    out = normalize_frame(df)
    assert len(out) == 3
    assert list(out["country"]) == ["India", "India", "Atlantis"]
    assert out.loc[0, "name_core"] == out.loc[1, "name_core"] == "green traders"
    assert out.loc[1, "addr_empty"]
    assert out.loc[2, "name_suffix"] == "sas"
    for col in ("name_core", "name_suffix", "addr_norm", "digits", "house_no", "landmark",
                "region", "is_website", "acronym", "first_token", "name_compact"):
        assert col in out.columns
