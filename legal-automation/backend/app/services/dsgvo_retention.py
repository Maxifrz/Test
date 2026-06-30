"""
Aufbewahrungs- und Löschlogik (DSGVO Art. 17) — reine Logik, unit-testbar.

RECHTLICH KRITISCH: Eine Löschung darf nicht erfolgen, solange gesetzliche
Aufbewahrungspflichten laufen:
- § 50 BRAO: Handakten 6 Jahre nach Mandatsende
- § 147 AO: steuerlich relevante Unterlagen 10 Jahre

Das Recht auf Löschung (Art. 17 DSGVO) ist daher kein 1-Klick-Vorgang, sondern
ein geprüfter Workflow. Diese Funktionen liefern die Entscheidungsgrundlage;
die eigentliche Anonymisierung erfolgt erst nach Freigabe.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

# Marker, mit dem PII-Felder bei der Löschung ersetzt werden
ERASURE_MARKER = "[GELÖSCHT gem. Art. 17 DSGVO]"

CLOSED_STATUSES = {"closed", "archived"}


def retention_until(closed_at: date | datetime | None, retention_years: int) -> date | None:
    """Datum, bis zu dem eine geschlossene Akte aufzubewahren ist (oder None)."""
    if closed_at is None:
        return None
    d = closed_at.date() if isinstance(closed_at, datetime) else closed_at
    # Aufbewahrung endet mit Ablauf des n-ten Jahrestags
    try:
        return d.replace(year=d.year + retention_years)
    except ValueError:
        # 29.02. → 28.02.
        return d.replace(year=d.year + retention_years, day=28)


def is_retention_expired(
    closed_at: date | datetime | None, retention_years: int, today: date | None = None
) -> bool:
    today = today or date.today()
    until = retention_until(closed_at, retention_years)
    if until is None:
        return False
    return today >= until


@dataclass
class MatterRetentionInfo:
    matter_number: str
    status: str
    closed_at: date | datetime | None
    retention_years: int


@dataclass
class ErasureEligibility:
    allowed: bool
    blocking_reasons: list[str] = field(default_factory=list)


def check_erasure_eligibility(
    matters: list[MatterRetentionInfo], today: date | None = None
) -> ErasureEligibility:
    """
    Prüft, ob ein Mandant gelöscht werden darf. Blockierend sind:
    - noch offene (nicht geschlossene) Akten
    - geschlossene Akten, deren Aufbewahrungsfrist noch läuft
    """
    today = today or date.today()
    reasons: list[str] = []
    for m in matters:
        if m.status not in CLOSED_STATUSES:
            reasons.append(f"Akte {m.matter_number} ist nicht abgeschlossen (Status: {m.status})")
            continue
        until = retention_until(m.closed_at, m.retention_years)
        if until is not None and today < until:
            reasons.append(
                f"Akte {m.matter_number}: Aufbewahrungsfrist läuft bis {until.isoformat()}"
            )
    return ErasureEligibility(allowed=len(reasons) == 0, blocking_reasons=reasons)
