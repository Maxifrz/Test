"""Tests für die Aufbewahrungs-/Löscheignungslogik (Art. 17 DSGVO)."""
from datetime import date

from app.services.dsgvo_retention import (
    MatterRetentionInfo,
    check_erasure_eligibility,
    is_retention_expired,
    retention_until,
)


def test_retention_until_basic():
    assert retention_until(date(2020, 3, 15), 6) == date(2026, 3, 15)
    assert retention_until(None, 6) is None


def test_retention_until_leap_day():
    assert retention_until(date(2020, 2, 29), 6) == date(2026, 2, 28)


def test_is_retention_expired():
    closed = date(2018, 1, 1)
    assert is_retention_expired(closed, 6, today=date(2024, 6, 1)) is True   # 2024 ≥ 2024
    assert is_retention_expired(closed, 10, today=date(2024, 6, 1)) is False  # bis 2028
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
    assert "2029-01-01" in e.blocking_reasons[0]


def test_eligibility_allows_when_all_expired():
    matters = [
        MatterRetentionInfo("2015-0001-X", "closed", date(2015, 1, 1), 6),
        MatterRetentionInfo("2016-0002-Y", "archived", date(2016, 1, 1), 6),
    ]
    e = check_erasure_eligibility(matters, today=date(2024, 6, 1))
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
