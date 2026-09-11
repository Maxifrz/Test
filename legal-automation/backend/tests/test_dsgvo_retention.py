"""Tests für die Aufbewahrungs-/Löscheignungslogik (Art. 17 DSGVO)."""
from datetime import date

from app.services.dsgvo_retention import (
    MatterRetentionInfo,
    check_erasure_eligibility,
    is_retention_expired,
    retention_until,
)


def test_retention_until_starts_at_end_of_calendar_year():
    # § 50 Abs. 1 S. 2 BRAO / § 147 Abs. 4 AO: die Frist beginnt mit Ablauf des
    # Kalenderjahres. Eine am 15.03.2020 geschlossene Akte ist deshalb nicht ab
    # dem 15.03.2026 loeschbar, sondern erst ab dem 01.01.2027.
    assert retention_until(date(2020, 3, 15), 6) == date(2027, 1, 1)
    assert retention_until(date(2020, 12, 31), 6) == date(2027, 1, 1)
    assert retention_until(date(2020, 1, 1), 6) == date(2027, 1, 1)
    assert retention_until(None, 6) is None


def test_retention_until_independent_of_day_within_year():
    # Alle Tage desselben Jahres ergeben dasselbe Fristende — genau das ist der
    # Sinn des Fristbeginns "Schluss des Kalenderjahres".
    ends = {retention_until(date(2020, m, 1), 6) for m in range(1, 13)}
    assert ends == {date(2027, 1, 1)}


def test_retention_until_leap_day():
    # Der 29.02. braucht keine Sonderbehandlung mehr — es wird ohnehin auf den
    # Jahreswechsel gerechnet.
    assert retention_until(date(2020, 2, 29), 6) == date(2027, 1, 1)


def test_retention_tax_relevant_raises_to_ten_years():
    # § 147 Abs. 3 AO geht der kuerzeren BRAO-Frist vor
    assert retention_until(date(2020, 3, 15), 6, tax_relevant=True) == date(2031, 1, 1)
    # Ein bereits laengerer Wert an der Akte bleibt erhalten
    assert retention_until(date(2020, 3, 15), 12, tax_relevant=True) == date(2033, 1, 1)


def test_is_retention_expired():
    closed = date(2018, 5, 20)
    assert is_retention_expired(closed, 6, today=date(2024, 12, 31)) is False
    assert is_retention_expired(closed, 6, today=date(2025, 1, 1)) is True
    assert is_retention_expired(closed, 10, today=date(2025, 1, 1)) is False
    assert is_retention_expired(closed, 6, today=date(2025, 1, 1), tax_relevant=True) is False
    assert is_retention_expired(None, 6, today=date(2024, 6, 1)) is False


def test_eligibility_blocks_open_matter():
    matters = [MatterRetentionInfo("2024-0001-MUS", "active", None, 6)]
    e = check_erasure_eligibility(matters, today=date(2024, 6, 1))
    assert e.allowed is False
    assert "nicht abgeschlossen" in e.blocking_reasons[0]


def test_eligibility_blocks_running_retention():
    matters = [MatterRetentionInfo("2023-0009-ABC", "closed", date(2023, 1, 1), 6)]
    e = check_erasure_eligibility(matters, today=date(2024, 6, 1))
    assert e.allowed is False
    assert "Aufbewahrungsfrist" in e.blocking_reasons[0]
    assert "§ 50 BRAO" in e.blocking_reasons[0]
    # Fristende 31.12.2029, loeschbar ab 01.01.2030
    assert "2029-12-31" in e.blocking_reasons[0]
    assert "2030-01-01" in e.blocking_reasons[0]


def test_eligibility_names_ao_for_tax_relevant_matter():
    matters = [MatterRetentionInfo("2023-0009-ABC", "closed", date(2023, 1, 1), 6, tax_relevant=True)]
    e = check_erasure_eligibility(matters, today=date(2031, 6, 1))
    assert e.allowed is False
    assert "§ 147 AO" in e.blocking_reasons[0]


def test_eligibility_blocks_closed_matter_without_date():
    # Abgeschlossen, aber ohne closed_at: der Fristbeginn ist unbestimmbar.
    # Frueher galt eine solche Akte stillschweigend als loeschbar.
    matters = [MatterRetentionInfo("2020-0001-X", "closed", None, 6)]
    e = check_erasure_eligibility(matters, today=date(2030, 1, 1))
    assert e.allowed is False
    assert "Schlussdatum" in e.blocking_reasons[0]


def test_eligibility_allows_when_all_expired():
    matters = [
        MatterRetentionInfo("2015-0001-X", "closed", date(2015, 1, 1), 6),
        MatterRetentionInfo("2016-0002-Y", "archived", date(2016, 1, 1), 6),
    ]
    e = check_erasure_eligibility(matters, today=date(2024, 6, 1))  # Fristen bis Ende 2021/2022
    assert e.allowed is True
    assert e.blocking_reasons == []


def test_eligibility_no_matters_allowed():
    e = check_erasure_eligibility([], today=date(2024, 6, 1))
    assert e.allowed is True


def test_eligibility_mixed_collects_all_reasons():
    matters = [
        MatterRetentionInfo("A", "active", None, 6),
        MatterRetentionInfo("B", "closed", date(2023, 1, 1), 10),
        MatterRetentionInfo("C", "closed", date(2010, 1, 1), 6),  # expired, ok
    ]
    e = check_erasure_eligibility(matters, today=date(2024, 6, 1))
    assert e.allowed is False
    assert len(e.blocking_reasons) == 2
