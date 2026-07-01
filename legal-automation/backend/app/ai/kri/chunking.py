"""
Struktur-bewusstes Chunking für Rechtsdokumente (GraphRAG V3.0).

Reine Logik, keine ML/DB-Abhängigkeit → unit-testbar. Ziel: präzise
zitierbare Chunks (Gesetze nach §/Absatz, Urteile nach Randnummern),
statt naivem Fixed-Window. Fällt auf Absatz-basiertes Chunking zurück.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# Gesetzesparagraph:  "§ 133", "§ 133a"
_SECTION = re.compile(r"§\s*\d+[a-z]?", re.IGNORECASE)
# Absatz am Segmentanfang:  "(1)", "(2) ..."
_ABSATZ = re.compile(r"(?m)^\s*\((\d+)\)\s*")
# Urteils-Randnummer:  "Rn. 12", "Randnummer 12", oder Zeilenanfang "12 "
_RANDNUMMER = re.compile(r"(?m)^\s*(?:Rn\.?|Randnummer)\s*(\d+)\b")

DEFAULT_MAX_CHARS = 1200
DEFAULT_OVERLAP = 150


@dataclass
class Chunk:
    ord: int
    heading: str | None
    text: str


def _clean(text: str) -> str:
    return re.sub(r"[ \t]+", " ", text).strip()


def chunk_gesetz(text: str) -> list[Chunk]:
    """Teilt Gesetzestext nach § und (soweit vorhanden) Absätzen."""
    matches = list(_SECTION.finditer(text))
    if not matches:
        return chunk_generic(text)

    chunks: list[Chunk] = []
    ordn = 0
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section_label = _clean(m.group(0)).replace(" ", " ")
        section_label = re.sub(r"§\s*", "§ ", section_label)
        body = text[m.end():end]

        abs_matches = list(_ABSATZ.finditer(body))
        if not abs_matches:
            t = _clean(text[start:end])
            if t:
                chunks.append(Chunk(ord=ordn, heading=section_label, text=t))
                ordn += 1
            continue
        for j, am in enumerate(abs_matches):
            a_start = am.start()
            a_end = abs_matches[j + 1].start() if j + 1 < len(abs_matches) else len(body)
            t = _clean(body[a_start:a_end])
            if t:
                chunks.append(
                    Chunk(ord=ordn, heading=f"{section_label} Abs. {am.group(1)}", text=t)
                )
                ordn += 1
    return chunks or chunk_generic(text)


def chunk_urteil(text: str) -> list[Chunk]:
    """Teilt Urteilstext nach Randnummern; sonst Absatz-Fallback."""
    matches = list(_RANDNUMMER.finditer(text))
    if not matches:
        return chunk_generic(text)
    chunks: list[Chunk] = []
    for i, m in enumerate(matches):
        start = m.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        t = _clean(text[start:end])
        if t:
            chunks.append(Chunk(ord=i, heading=f"Rn. {m.group(1)}", text=t))
    return chunks or chunk_generic(text)


def chunk_generic(text: str, max_chars: int = DEFAULT_MAX_CHARS, overlap: int = DEFAULT_OVERLAP) -> list[Chunk]:
    """Absatzweises Packen in Fenster ≤ max_chars mit Überlappung."""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks: list[Chunk] = []
    buf = ""
    ordn = 0
    for p in paras:
        if buf and len(buf) + len(p) + 1 > max_chars:
            chunks.append(Chunk(ord=ordn, heading=None, text=_clean(buf)))
            ordn += 1
            buf = (buf[-overlap:] + "\n" + p) if overlap else p
        else:
            buf = f"{buf}\n{p}".strip()
    if buf.strip():
        chunks.append(Chunk(ord=ordn, heading=None, text=_clean(buf)))
    return chunks


def chunk(text: str, source_type: str) -> list[Chunk]:
    """Dispatch nach Quelltyp."""
    if not text or not text.strip():
        return []
    if source_type == "gesetz":
        return chunk_gesetz(text)
    if source_type == "urteil":
        return chunk_urteil(text)
    return chunk_generic(text)
