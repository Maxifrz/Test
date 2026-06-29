"""Tests für den Ladungs-Parser (E-Mail → Gerichtstermin-Vorschlag)."""
from datetime import date, time

from app.services.ladung_parser import parse_ladung


def test_numeric_date_with_time_and_az():
    subject = "Ladung zur mündlichen Verhandlung"
    body = (
        "In dem Rechtsstreit Mustermann ./. Schmidt, Az. 12 O 345/24, "
        "laden wir Sie zum Termin am 15.03.2024 um 10:30 Uhr in Saal 217."
    )
    r = parse_ladung(subject, body, sender="poststelle@lg-muenchen.justiz.bayern.de")
    assert r.found is True
    assert r.hearing_date == date(2024, 3, 15)
    assert r.hearing_time == time(10, 30)
    assert r.aktenzeichen == "12 O 345/24"
    assert r.room == "217"
    # gericht/justiz domain + time + AZ → high confidence
    assert r.confidence >= 0.9


def test_text_month_date():
    body = "Hauptverhandlung am 3. April 2024 um 9 Uhr."
    r = parse_ladung("Terminbestimmung", body, sender="")
    assert r.found is True
    assert r.hearing_date == date(2024, 4, 3)
    assert r.hearing_time == time(9, 0)


def test_no_keyword_means_not_found():
    body = "Anbei die Rechnung vom 15.03.2024 über 500 EUR."
    r = parse_ladung("Rechnung", body, sender="buchhaltung@firma.de")
    assert r.found is False


def test_keyword_but_no_date():
    body = "Der Termin zur Verhandlung wird Ihnen noch mitgeteilt."
    r = parse_ladung("Ladung folgt", body, sender="")
    assert r.found is False


def test_confidence_lower_without_court_domain():
    body = "Termin zur Güteverhandlung am 01.07.2024."
    r = parse_ladung("Güteverhandlung", body, sender="info@example.com")
    assert r.found is True
    assert r.hearing_date == date(2024, 7, 1)
    # no court domain, no time → moderate confidence
    assert r.confidence < 0.9


def test_invalid_date_ignored():
    body = "Verhandlung am 35.13.2024."  # invalid
    r = parse_ladung("Ladung", body, sender="")
    assert r.found is False
