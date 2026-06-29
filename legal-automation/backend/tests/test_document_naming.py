"""Tests für die Dateibenennungs-Konvention YYYY-MM-DD_Mandant_Akte_Typ."""
from datetime import date

from app.services.document_naming import build_dirname, slugify


def test_slugify_umlauts():
    assert slugify("Müller") == "Mueller"
    assert slugify("Groß & Söhne") == "Gross-Soehne"


def test_slugify_empty_fallback():
    assert slugify("", "Mandant") == "Mandant"
    assert slugify("   ") == "Unbekannt"


def test_build_dirname_basic():
    name = build_dirname(date(2024, 3, 15), "Mustermann", "2024-0001-MUS", "Mandantengespräch")
    assert name == "2024-03-15_Mustermann_2024-0001-MUS_Mandantengespraech"


def test_build_dirname_sanitizes_slashes():
    # Aktenzeichen with slashes (court ref) must not break the path
    name = build_dirname(date(2024, 1, 5), "Schmidt", "12 O 345/24", "Verhandlung")
    assert "/" not in name
    assert name.startswith("2024-01-05_Schmidt_12-O-345-24_")


def test_build_dirname_missing_parts():
    name = build_dirname(date(2024, 1, 1), "", "", "")
    assert name == "2024-01-01_Mandant_ohne-Akte_Sonstiges"
