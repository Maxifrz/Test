"""
Grounded-Answer-Helfer (V3.0): Prompt-Bau + Quellen-Referenz-Extraktion.

Reine Logik, LLM-frei → unit-testbar. Die Grounding-GARANTIE entsteht aus dem
Zusammenspiel: (1) Prompt erzwingt [S#]-Zitate und erlaubt nur den Kontext,
(2) extract_source_refs() liest die zitierten Marker aus der Antwort,
(3) retrieval.validate_grounded() prüft sie gegen die abgerufenen Chunks —
schlägt eines fehl, wird GROUNDING_REFUSAL geliefert statt der Antwort.
"""
from __future__ import annotations

import re

# [S1], [S2] … — auch Mehrfachnennungen wie [S1, S3]
_REF = re.compile(r"\[S(\d+)(?:\s*,\s*S?(\d+))*\]")
_REF_ALL_NUMS = re.compile(r"S?(\d+)")

SYSTEM_PROMPT = """Du bist ein juristischer Rechercheassistent einer deutschen Kanzlei \
(Schwerpunkt Insolvenzrecht). Beantworte die Frage AUSSCHLIESSLICH auf Basis der \
nummerierten Quellen im Kontext. Regeln:
1. Jede Aussage MUSS mit Quellenmarker(n) belegt sein, z. B. [S1] oder [S2, S3].
2. Erfinde NIEMALS Normen, Urteile oder Fundstellen. Nutze nur den Kontext.
3. Reicht der Kontext nicht aus, antworte exakt: KEINE_GRUNDLAGE
4. Antworte auf Deutsch, präzise und im Gutachtenstil, ohne Floskeln.
5. Dies ist ein ENTWURF für Berufsträger, keine Rechtsberatung."""


def build_prompt(question: str, context: str) -> str:
    """Baut den vollständigen Prompt aus System-Regeln, Kontext und Frage."""
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"=== KONTEXT (nummerierte Quellen) ===\n{context}\n\n"
        f"=== FRAGE ===\n{question}\n\n"
        f"=== ANTWORT (mit [S#]-Belegen) ==="
    )


def extract_source_refs(answer: str) -> list[int]:
    """Extrahiert alle zitierten Quellen-Nummern ([S1], [S2, S3]) dedupliziert,
    in Reihenfolge des ersten Auftretens."""
    seen: list[int] = []
    for m in _REF.finditer(answer):
        for num in _REF_ALL_NUMS.findall(m.group(0)):
            n = int(num)
            if n not in seen:
                seen.append(n)
    return seen


def is_refusal(answer: str) -> bool:
    """Hat das Modell mangels Grundlage abgelehnt?"""
    return "KEINE_GRUNDLAGE" in answer.upper().replace(" ", "_")


def refs_to_chunk_ids(refs: list[int], used_chunk_ids: list[int]) -> list[int]:
    """
    Mappt [S#]-Nummern (1-basiert, Reihenfolge aus build_context) auf chunk_ids.
    Unbekannte Nummern werden als -1 markiert → validate_grounded schlägt fehl.
    """
    out: list[int] = []
    for r in refs:
        idx = r - 1
        out.append(used_chunk_ids[idx] if 0 <= idx < len(used_chunk_ids) else -1)
    return out
