"""
Aufbewahrungs- und Löschlogik (DSGVO Art. 17) — reine Logik, unit-testbar.

RECHTLICH KRITISCH: Eine Löschung darf nicht erfolgen, solange gesetzliche
Aufbewahrungspflichten laufen:
- § 50 Abs. 1 BRAO: Handakten 6 Jahre. Die Frist beginnt nach § 50 Abs. 1 S. 2
  BRAO mit dem **Ablauf des Kalenderjahres**, in dem der Auftrag beendet wurde.
- § 147 Abs. 3, 4 AO / § 257 Abs. 4, 5 HGB: steuerlich relevante Unterlagen
  10 Jahre, ebenfalls ab **Schluss des Kalenderjahres**.

WICHTIG (korrigiert): Früher wurde ab dem exakten Jahrestag des Aktenschlusses
gerechnet. Das lief allen drei Normen zuwider und gab Akten im Schnitt ein
halbes Jahr zu früh zur Löschung frei. Eine am 15.03.2020 geschlossene Akte ist
damit nicht ab dem 15.03.2026, sondern erst ab dem 01.01.2027 löschbar.

Das Recht auf Löschung (Art. 17 DSGVO) ist daher kein 1-Klick-Vorgang, sondern
ein geprüfter Workflow. Diese Funktionen liefern die Entscheidungsgrundlage;
die eigentliche Anonymisierung erfolgt erst nach Freigabe.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

# Marker, mit dem PII-Felder bei der Löschung ersetzt werden.
# Kanonisch definiert in dsgvo_erasure.py; hier re-exportiert, damit
# bestehende Importe weiter funktionieren.
from app.services.dsgvo_erasure import ERASURE_MARKER  # noqa: E402,F401

CLOSED_STATUSES = {"closed", "archived"}

# Gesetzliche Mindestfristen in Jahren
BRAO_RETENTION_YEARS = 6      # § 50 Abs. 1 BRAO — Handakten
AO_RETENTION_YEARS = 10       # § 147 Abs. 3 AO / § 257 Abs. 4 HGB — steuerrelevant


def retention_until(
    closed_at: date | datetime | None,
    retention_years: int,
    *,
    tax_relevant: bool = False,
) -> date | None:
    """
    Erster Tag, an dem eine geschlossene Akte gelöscht werden darf (oder None,
    wenn kein Schlussdatum vorliegt).

    Die Frist läuft ab dem **Schluss des Kalenderjahres** des Aktenschlusses
    (§ 50 Abs. 1 S. 2 BRAO, § 147 Abs. 4 AO, § 257 Abs. 5 HGB) und endet mit
    Ablauf des 31.12. des n-ten Folgejahres — löschbar ist also der 01.01.
    danach.

    `tax_relevant` hebt die Frist auf mindestens 10 Jahre an (§ 147 AO), auch
    wenn an der Akte ein kürzerer Wert hinterlegt ist.
    """
    if closed_at is None:
        return None
    d = closed_at.date() if isinstance(closed_at, datetime) else closed_at
    years = max(retention_years, AO_RETENTION_YEARS) if tax_relevant else retention_years
    # Fristende = 31.12. des Jahres (Schlussjahr + n); löschbar ab dem Folgetag.
    return date(d.year + years + 1, 1, 1)


def is_retention_expired(
    closed_at: date | datetime | None,
    retention_years: int,
    today: date | None = None,
    *,
    tax_relevant: bool = False,
) -> bool:
    today = today or date.today()
    until = retention_until(closed_at, retention_years, tax_relevant=tax_relevant)
    if until is None:
        return False
    return today >= until


@dataclass
class MatterRetentionInfo:
    matter_number: str
    status: str
    closed_at: date | datetime | None
    retention_years: int
    # Steuerrelevante Akten unterliegen § 147 AO (10 Jahre), unabhängig davon,
    # was an der Akte als Frist hinterlegt ist.
    tax_relevant: bool = False


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
        if m.closed_at is None:
            # Geschlossen ohne Schlussdatum: Fristbeginn unbestimmbar → blockieren,
            # statt stillschweigend freizugeben.
            reasons.append(
                f"Akte {m.matter_number}: als abgeschlossen markiert, aber ohne "
                "Schlussdatum — Fristbeginn nicht bestimmbar"
            )
            continue
        until = retention_until(m.closed_at, m.retention_years, tax_relevant=m.tax_relevant)
        if until is not None and today < until:
            norm = "§ 147 AO" if m.tax_relevant else "§ 50 BRAO"
            reasons.append(
                f"Akte {m.matter_number}: Aufbewahrungsfrist ({norm}) läuft bis "
                f"{(until - timedelta(days=1)).isoformat()}, löschbar ab {until.isoformat()}"
            )
    return ErasureEligibility(allowed=len(reasons) == 0, blocking_reasons=reasons)
