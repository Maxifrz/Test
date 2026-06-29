"""
Manuell verifizierte Testfälle für den Fristen-Kalkulator.

RECHTLICH KRITISCH: Jeder Testfall basiert auf einem konkreten, von Hand
geprüften Datum. Vor Go-Live müssen diese Fälle vom verantwortlichen Anwalt
gegengeprüft werden.

Hinweis: Feiertage sind Ländersache. Diese Tests verwenden explizit Bayern
("BY"), damit sie unabhängig von der Laufzeit-Konfiguration reproduzierbar sind.
"""
from datetime import date

from app.services import deadline_calculator as dc

BY = "BY"


def test_is_business_day_weekend():
    # 2024-03-30 war ein Samstag, 2024-03-31 ein Sonntag
    assert dc.is_business_day(date(2024, 3, 30), BY) is False
    assert dc.is_business_day(date(2024, 3, 31), BY) is False
    # 2024-03-28 (Donnerstag) war ein normaler Werktag
    assert dc.is_business_day(date(2024, 3, 28), BY) is True


def test_is_business_day_holiday():
    # Karfreitag 2024 = 29.03.2024 (gesetzlicher Feiertag bundesweit)
    assert dc.is_business_day(date(2024, 3, 29), BY) is False
    # Tag der Deutschen Einheit 2024 = 03.10.2024 (Donnerstag, Feiertag)
    assert dc.is_business_day(date(2024, 10, 3), BY) is False


def test_zpo_339_einspruch_karfreitag_verschiebung():
    """
    PLAN-KERNFALL: Versäumnisurteil zugestellt am Freitag 15.03.2024.
    Einspruchsfrist §339 ZPO = 2 Wochen → rohes Fristende 29.03.2024.
    29.03.2024 ist Karfreitag, 30.03 Sa, 31.03 So (Ostersonntag),
    01.04 Ostermontag (Feiertag) → nächster Werktag 02.04.2024 (Dienstag).
    """
    result = dc.einspruch_versaeumnisurteil(date(2024, 3, 15), bundesland=BY)
    assert result.raw_deadline == date(2024, 3, 29)
    assert result.adjusted_for_holiday is True
    assert result.deadline == date(2024, 4, 2)
    assert "ZPO §339" in result.basis


def test_add_weeks_no_adjustment_needed():
    # Montag 04.03.2024 + 2 Wochen = Montag 18.03.2024 (Werktag, keine Verschiebung)
    result = dc.add_weeks(date(2024, 3, 4), 2, bundesland=BY)
    assert result.raw_deadline == date(2024, 3, 18)
    assert result.adjusted_for_holiday is False
    assert result.deadline == date(2024, 3, 18)


def test_berufung_einlegung_one_month():
    # ZPO §517: 1 Monat. Urteil zugestellt 15.01.2024 → 15.02.2024 (Donnerstag, Werktag)
    result = dc.berufung_einlegung(date(2024, 1, 15), bundesland=BY)
    assert result.raw_deadline == date(2024, 2, 15)
    assert result.deadline == date(2024, 2, 15)


def test_berufung_begruendung_two_months():
    # ZPO §520 II: 2 Monate. Zustellung 15.01.2024 → 15.03.2024 (Freitag, Werktag)
    result = dc.berufung_begruendung(date(2024, 1, 15), bundesland=BY)
    assert result.raw_deadline == date(2024, 3, 15)
    assert result.deadline == date(2024, 3, 15)


def test_months_nonexistent_target_day():
    """
    §188 III BGB: Zieltag existiert nicht. 31.01.2024 + 1 Monat → 29.02.2024
    (2024 ist Schaltjahr; letzter Tag des Februars).
    """
    result = dc.add_months(date(2024, 1, 31), 1, bundesland=BY)
    assert result.raw_deadline == date(2024, 2, 29)


def test_months_nonexistent_target_day_non_leap_year():
    # 31.01.2023 + 1 Monat → 28.02.2023 (kein Schaltjahr)
    result = dc.add_months(date(2023, 1, 31), 1, bundesland=BY)
    assert result.raw_deadline == date(2023, 2, 28)


def test_add_days_event_day_not_counted():
    """
    §187 I BGB: Ereignistag zählt nicht. Ereignis am 01.03.2024 + 10 Tage
    → 11.03.2024 (Montag, Werktag).
    """
    result = dc.add_days(date(2024, 3, 1), 10, bundesland=BY)
    assert result.raw_deadline == date(2024, 3, 11)
    assert result.deadline == date(2024, 3, 11)


def test_stpo_45_wiedereinsetzung_one_week():
    # StPO §45: 1 Woche. Wegfall Hindernis 06.05.2024 (Mo) → 13.05.2024 (Mo)
    result = dc.wiedereinsetzung_stpo(date(2024, 5, 6), bundesland=BY)
    assert result.raw_deadline == date(2024, 5, 13)
    assert result.deadline == date(2024, 5, 13)


def test_registry_contains_all_named_fristen():
    for key in [
        "einspruch_versaeumnisurteil",
        "berufung_einlegung",
        "berufung_begruendung",
        "klageerwiderung",
        "beschwerde_stpo",
        "wiedereinsetzung_stpo",
    ]:
        assert key in dc.FRIST_CALCULATORS
