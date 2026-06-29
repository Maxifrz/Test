"""
Einheitliche, nachvollziehbare Dateibenennung: YYYY-MM-DD_Mandant_Akte_Typ.
Reine Logik → unit-testbar. Sanitisiert Umlaute, Leerzeichen und unsichere
Pfad-Zeichen, damit der Ordnername filesystem- und audit-tauglich ist.
"""
from __future__ import annotations

import re
from datetime import date

_UMLAUTS = {
    "ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss",
    "Ä": "Ae", "Ö": "Oe", "Ü": "Ue",
}


def slugify(value: str, fallback: str = "Unbekannt") -> str:
    """ASCII-safe slug: Umlaute transliteriert, Sonderzeichen → '-', getrimmt."""
    if not value:
        return fallback
    for src, dst in _UMLAUTS.items():
        value = value.replace(src, dst)
    # Replace any run of non-alphanumeric chars with a single hyphen
    value = re.sub(r"[^A-Za-z0-9]+", "-", value)
    value = value.strip("-")
    return value or fallback


def build_dirname(meeting_date: date, mandant: str, aktenzeichen: str, typ: str) -> str:
    """
    Verzeichnisname nach Konvention YYYY-MM-DD_Mandant_Akte_Typ.
    Beispiel: 2024-03-15_Mustermann_2024-0001-MUS_Mandantengespraech
    """
    parts = [
        meeting_date.isoformat(),
        slugify(mandant, "Mandant"),
        slugify(aktenzeichen, "ohne-Akte"),
        slugify(typ, "Sonstiges"),
    ]
    return "_".join(parts)
