"""
Ladungs-Parser: extrahiert Termininformationen aus einer Gerichts-E-Mail
(Ladung / Terminbestimmung) heuristisch.

Reine Logik, keine externen Aufrufe → unit-testbar und DSGVO-konform
(kein Versand an Dritte). Die KI-gestützte Variante folgt in Phase 8; bis
dahin deckt ein robustes Regex-Set die gängigen Formate deutscher Gerichte ab.

WICHTIG: Das Ergebnis ist ein VORSCHLAG. Ein Mensch bestätigt den erzeugten
Kalendereintrag (Human-in-the-loop), bevor er verbindlich wird.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime, time

MONTHS = {
    "januar": 1, "februar": 2, "märz": 3, "maerz": 3, "april": 4, "mai": 5,
    "juni": 6, "juli": 7, "august": 8, "september": 9, "oktober": 10,
    "november": 11, "dezember": 12,
}

# 1) "15.03.2024" / "15.3.24"
_DATE_NUMERIC = re.compile(r"\b(\d{1,2})\.(\d{1,2})\.(\d{2,4})\b")
# 2) "15. März 2024"
_DATE_TEXT = re.compile(
    r"\b(\d{1,2})\.?\s*(" + "|".join(MONTHS) + r")\s+(\d{4})\b", re.IGNORECASE
)
# Uhrzeit: "10:30 Uhr", "10.30 Uhr", "9 Uhr"
_TIME = re.compile(r"\b(\d{1,2})(?:[:.](\d{2}))?\s*Uhr\b", re.IGNORECASE)
# Aktenzeichen: "12 O 345/24", "4 Ca 1234/23", "AZ: ..."
_AKTENZEICHEN = re.compile(r"\b(\d{1,3}\s?[A-Za-z]{1,4}\s?\d{1,5}/\d{2,4})\b")
# Saal / Raum
_ROOM = re.compile(r"\b(?:Saal|Sitzungssaal|Raum|Zimmer)\s+([A-Za-z0-9.\-/]+)", re.IGNORECASE)

_TERMIN_KEYWORDS = [
    "ladung", "termin", "verhandlung", "mündliche verhandlung",
    "sitzung", "hauptverhandlung", "verkündungstermin", "güteverhandlung",
]


@dataclass
class LadungResult:
    found: bool
    hearing_date: date | None = None
    hearing_time: time | None = None
    aktenzeichen: str | None = None
    room: str | None = None
    confidence: float = 0.0
    note: str = ""


def _parse_date(text: str) -> date | None:
    m = _DATE_NUMERIC.search(text)
    if m:
        day, month, year = int(m[1]), int(m[2]), int(m[3])
        if year < 100:
            year += 2000
        try:
            return date(year, month, day)
        except ValueError:
            pass
    m = _DATE_TEXT.search(text)
    if m:
        day = int(m[1])
        month = MONTHS[m[2].lower()]
        year = int(m[3])
        try:
            return date(year, month, day)
        except ValueError:
            pass
    return None


def _parse_time(text: str) -> time | None:
    m = _TIME.search(text)
    if m:
        hour = int(m[1])
        minute = int(m[2]) if m[2] else 0
        if 0 <= hour <= 23 and 0 <= minute <= 59:
            return time(hour, minute)
    return None


def parse_ladung(subject: str, body: str, sender: str = "") -> LadungResult:
    """
    Heuristik: Erkennt eine Ladung, wenn Termin-Keywords vorkommen und ein
    Datum extrahierbar ist. Confidence steigt mit Absender-Domain (Gericht),
    Aktenzeichen und Uhrzeit.
    """
    haystack = f"{subject}\n{body}".lower()

    has_keyword = any(k in haystack for k in _TERMIN_KEYWORDS)
    hearing_date = _parse_date(f"{subject}\n{body}")

    if not (has_keyword and hearing_date):
        return LadungResult(found=False, note="Keine Ladung erkannt (Keyword/Datum fehlt).")

    hearing_time = _parse_time(f"{subject}\n{body}")
    az_match = _AKTENZEICHEN.search(f"{subject}\n{body}")
    room_match = _ROOM.search(body)

    # Confidence scoring
    confidence = 0.4  # keyword + date present
    if ".de" in sender.lower() and ("gericht" in sender.lower() or "justiz" in sender.lower()):
        confidence += 0.3
    if hearing_time:
        confidence += 0.15
    if az_match:
        confidence += 0.15
    confidence = min(confidence, 1.0)

    parts = [f"Termin am {hearing_date.isoformat()}"]
    if hearing_time:
        parts.append(hearing_time.strftime("%H:%M Uhr"))
    if az_match:
        parts.append(f"Az. {az_match[1].strip()}")
    if room_match:
        parts.append(f"Saal {room_match[1]}")

    return LadungResult(
        found=True,
        hearing_date=hearing_date,
        hearing_time=hearing_time,
        aktenzeichen=az_match[1].strip() if az_match else None,
        room=room_match[1].rstrip(".,;") if room_match else None,
        confidence=round(confidence, 2),
        note="Vorschlag aus E-Mail erkannt: " + ", ".join(parts) + ". Bitte prüfen.",
    )
