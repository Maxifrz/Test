"""
Zitations-Extraktion für den Wissensgraphen (GraphRAG V3.0).

Erkennt Verweise in Rechtstexten → Graphkanten:
- Normzitate (§/Art. + Gesetzeskürzel)
- Aktenzeichen (Muster wie in services/ladung_parser.py)
- ECLI (European Case Law Identifier)

Reine Logik → unit-testbar.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# § 133 InsO · § 133 Abs. 2 InsO · Art. 6 DSGVO
_NORM = re.compile(
    r"(?:§|Art\.?)\s*\d+[a-z]?\s*(?:Abs\.\s*\d+\s*)?(?:S\.\s*\d+\s*)?"
    r"(?P<gesetz>[A-ZÄÖÜ][A-Za-z0-9ÄÖÜäöü]{1,9})"
)
# Aktenzeichen: "12 O 345/24", "IX ZR 100/23"
_AZ = re.compile(r"\b([0-9IVXL]{1,4}\s?[A-Za-z]{1,4}\s?\d{1,5}/\d{2,4})\b")
# ECLI:DE:BGH:2023:...
_ECLI = re.compile(r"ECLI:[A-Z]{2}:[A-Z0-9]+:\d{4}:[A-Z0-9.]+", re.IGNORECASE)

# gängige Gesetzeskürzel, um Fehltreffer bei _NORM zu reduzieren
_KNOWN_GESETZE = {
    "InsO", "BGB", "ZPO", "StPO", "StGB", "HGB", "GmbHG", "AktG", "AO", "RVG",
    "InsVV", "StaRUG", "BRAO", "BORA", "DSGVO", "SGB", "EStG", "UStG", "GewO", "GG",
}


@dataclass(frozen=True)
class Citation:
    citation_type: str  # "norm" | "aktenzeichen" | "ecli"
    raw: str
    normalized: str


def _norm_norm(match: re.Match) -> str:
    return re.sub(r"\s+", " ", match.group(0)).strip()


def extract_citations(text: str) -> list[Citation]:
    """Extrahiert deduplizierte Zitate aus einem Textabschnitt."""
    found: dict[tuple[str, str], Citation] = {}

    for m in _NORM.finditer(text):
        gesetz = m.group("gesetz")
        # nur akzeptieren, wenn bekanntes Kürzel (reduziert Rauschen)
        if gesetz not in _KNOWN_GESETZE:
            continue
        norm = _norm_norm(m)
        found[("norm", norm)] = Citation("norm", raw=m.group(0), normalized=norm)

    for m in _ECLI.finditer(text):
        val = m.group(0)
        found[("ecli", val.upper())] = Citation("ecli", raw=val, normalized=val.upper())

    for m in _AZ.finditer(text):
        val = m.group(1).strip()
        # ECLI-Teile nicht doppelt als AZ erfassen
        if "ECLI" in val.upper():
            continue
        key = ("aktenzeichen", re.sub(r"\s+", " ", val))
        found.setdefault(key, Citation("aktenzeichen", raw=val, normalized=re.sub(r"\s+", " ", val)))

    return list(found.values())
