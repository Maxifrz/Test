"""
Fristen-Kalkulator für deutsche Verfahrensfristen.

RECHTLICH KRITISCH: Fehler hier können eine Berufspflichtverletzung darstellen.
Jede Regel ist mit dem zugrundeliegenden Paragraphen dokumentiert und durch
manuell verifizierte Tests abgesichert (siehe tests/test_deadline_calculator.py).

Grundlagen:
- §§ 187–193 BGB: Fristberechnung (Beginn, Ende, Feiertagsanpassung)
- ZPO §§ 221, 222: Verweis auf BGB-Fristberechnung im Zivilprozess
- §193 BGB: Fällt das Fristende auf Samstag, Sonntag oder gesetzlichen
  Feiertag, endet die Frist am nächsten Werktag.

Die gesetzlichen Feiertage richten sich nach dem Bundesland (Konfiguration
BUNDESLAND), da Feiertage in Deutschland Ländersache sind.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import holidays


@dataclass
class FristResult:
    """Ergebnis einer Fristberechnung mit vollständigem Nachvollziehbarkeits-Pfad."""

    deadline: date
    trigger_date: date
    basis: str
    raw_deadline: date  # vor §193-Anpassung
    adjusted_for_holiday: bool
    note: str


def _bundesland_holidays(year: int, bundesland: str | None = None) -> holidays.HolidayBase:
    """Gesetzliche Feiertage für ein Bundesland. Default aus Settings."""
    if bundesland is None:
        # Lazy import keeps this pure-logic module importable without the full
        # app/pydantic stack (e.g. in isolated unit tests).
        from app.core.config import get_settings

        bundesland = get_settings().BUNDESLAND
    # holidays lib uses subdivision codes like "BY", "BE", "NW"
    return holidays.Germany(years=year, subdiv=bundesland)


def is_business_day(d: date, bundesland: str | None = None) -> bool:
    """Werktag = Montag–Freitag und kein gesetzlicher Feiertag."""
    if d.weekday() >= 5:  # 5 = Samstag, 6 = Sonntag
        return False
    feiertage = _bundesland_holidays(d.year, bundesland)
    return d not in feiertage


def adjust_for_section_193(d: date, bundesland: str | None = None) -> tuple[date, bool]:
    """
    §193 BGB: Fällt das Fristende auf Sonntag, Samstag oder Feiertag, tritt an
    seine Stelle der nächste Werktag. Gibt (angepasstes Datum, wurde_verschoben).
    """
    original = d
    while not is_business_day(d, bundesland):
        d += timedelta(days=1)
    return d, d != original


def add_days(
    trigger_date: date,
    days: int,
    basis: str = "Tagesfrist",
    bundesland: str | None = None,
) -> FristResult:
    """
    Ereignisfrist nach §187 Abs. 1 BGB: Der Tag des Ereignisses zählt NICHT mit;
    die Frist beginnt am Folgetag. Fristende nach §188 Abs. 1 BGB. Anschließend
    §193-Feiertagsanpassung.
    """
    raw_deadline = trigger_date + timedelta(days=days)
    adjusted, moved = adjust_for_section_193(raw_deadline, bundesland)
    note = (
        f"{basis}: {days} Tage ab {trigger_date.isoformat()} "
        f"(§187 I BGB, Ereignistag zählt nicht). Fristende {raw_deadline.isoformat()}"
    )
    if moved:
        note += f"; wegen §193 BGB verschoben auf {adjusted.isoformat()}"
    return FristResult(
        deadline=adjusted,
        trigger_date=trigger_date,
        basis=basis,
        raw_deadline=raw_deadline,
        adjusted_for_holiday=moved,
        note=note,
    )


def add_weeks(
    trigger_date: date,
    weeks: int,
    basis: str = "Wochenfrist",
    bundesland: str | None = None,
) -> FristResult:
    """Wochenfrist nach §188 Abs. 2 BGB."""
    return add_days(trigger_date, weeks * 7, basis=basis, bundesland=bundesland)


def add_months(
    trigger_date: date,
    months: int,
    basis: str = "Monatsfrist",
    bundesland: str | None = None,
) -> FristResult:
    """
    Monatsfrist nach §188 Abs. 2, 3 BGB. Die Frist endet mit Ablauf des Tages
    des letzten Monats, der durch seine Zahl dem Tag entspricht, in den das
    Ereignis fällt. Fehlt dieser Tag (z.B. 31. → Februar), endet die Frist am
    letzten Tag des Monats (§188 Abs. 3 BGB).
    """
    # §187 I: Berechnung erfolgt vom Ereignistag aus; das Fristende ist der
    # entsprechende Tag (nicht Folgetag), da §188 II auf den Tag abstellt,
    # "welcher dem Tage vorhergeht, der durch seine Benennung dem Anfangstag
    # entspricht". Praktisch: gleicher Kalendertag X Monate später.
    month_index = trigger_date.month - 1 + months
    year = trigger_date.year + month_index // 12
    month = month_index % 12 + 1

    # Letzter gültiger Tag des Zielmonats bestimmen
    if month == 12:
        last_day = 31
    else:
        last_day = (date(year, month + 1, 1) - timedelta(days=1)).day
    day = min(trigger_date.day, last_day)
    raw_deadline = date(year, month, day)

    adjusted, moved = adjust_for_section_193(raw_deadline, bundesland)
    note = (
        f"{basis}: {months} Monate ab {trigger_date.isoformat()} "
        f"(§188 II/III BGB). Fristende {raw_deadline.isoformat()}"
    )
    if day < trigger_date.day:
        note += " (Zieltag existiert nicht, letzter Tag des Monats, §188 III BGB)"
    if moved:
        note += f"; wegen §193 BGB verschoben auf {adjusted.isoformat()}"

    return FristResult(
        deadline=adjusted,
        trigger_date=trigger_date,
        basis=basis,
        raw_deadline=raw_deadline,
        adjusted_for_holiday=moved,
        note=note,
    )


# --- Benannte Verfahrensfristen ---


def einspruch_versaeumnisurteil(zustellung: date, bundesland: str | None = None) -> FristResult:
    """ZPO §339 Abs. 1: Einspruchsfrist gegen Versäumnisurteil = 2 Wochen ab Zustellung."""
    return add_weeks(zustellung, 2, basis="ZPO §339 — Einspruch Versäumnisurteil", bundesland=bundesland)


def berufung_einlegung(zustellung: date, bundesland: str | None = None) -> FristResult:
    """ZPO §517: Berufungsfrist = 1 Monat ab Zustellung des Urteils."""
    return add_months(zustellung, 1, basis="ZPO §517 — Berufungseinlegung", bundesland=bundesland)


def berufung_begruendung(zustellung: date, bundesland: str | None = None) -> FristResult:
    """ZPO §520 Abs. 2: Berufungsbegründungsfrist = 2 Monate ab Zustellung des Urteils."""
    return add_months(zustellung, 2, basis="ZPO §520 II — Berufungsbegründung", bundesland=bundesland)


def klageerwiderung(zustellung: date, weeks: int = 2, bundesland: str | None = None) -> FristResult:
    """ZPO §276 Abs. 1 S. 2: Klageerwiderungsfrist, vom Gericht gesetzt (mind. 2 Wochen)."""
    return add_weeks(zustellung, weeks, basis="ZPO §276 — Klageerwiderung", bundesland=bundesland)


def beschwerde_stpo(zustellung: date, bundesland: str | None = None) -> FristResult:
    """StPO §311 Abs. 2: sofortige Beschwerde = 1 Woche ab Bekanntmachung."""
    return add_weeks(zustellung, 1, basis="StPO §311 — sofortige Beschwerde", bundesland=bundesland)


def wiedereinsetzung_stpo(wegfall_hindernis: date, bundesland: str | None = None) -> FristResult:
    """StPO §45 Abs. 1: Wiedereinsetzungsantrag = 1 Woche nach Wegfall des Hindernisses."""
    return add_weeks(
        wegfall_hindernis, 1, basis="StPO §45 — Wiedereinsetzung", bundesland=bundesland
    )


# Registry for the API: maps a frist_type string to a calculator function.
FRIST_CALCULATORS = {
    "einspruch_versaeumnisurteil": einspruch_versaeumnisurteil,
    "berufung_einlegung": berufung_einlegung,
    "berufung_begruendung": berufung_begruendung,
    "klageerwiderung": klageerwiderung,
    "beschwerde_stpo": beschwerde_stpo,
    "wiedereinsetzung_stpo": wiedereinsetzung_stpo,
}
