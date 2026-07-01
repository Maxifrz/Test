"""
Hybrid-Retrieval-Ranking + Grounding-Regeln (GraphRAG V3.0).

Reine Logik (keine DB/ML) → unit-testbar. Die eigentlichen pgvector-/tsvector-
Queries und die Graph-Expansion liefern Kandidaten mit Roh-Scores; hier werden
sie kombiniert, gewichtet, gerankt — und es wird entschieden, ob die Grundlage
für eine belegte Antwort ausreicht (sonst: Ablehnung statt Halluzination).
"""
from __future__ import annotations

from dataclasses import dataclass, field

# Antwort, wenn keine ausreichende Quellengrundlage gefunden wurde (Anti-Halluzination)
GROUNDING_REFUSAL = (
    "Zu dieser Frage wurde im hinterlegten Rechtsbestand keine ausreichend "
    "belegbare Grundlage gefunden. Bitte anwaltlich prüfen / Quelle ergänzen."
)


@dataclass
class Weights:
    vector: float = 0.6
    fts: float = 0.3
    graph: float = 0.1


@dataclass
class Candidate:
    chunk_id: int
    document_id: int
    vector_score: float = 0.0   # erwartet normalisiert [0,1]
    fts_score: float = 0.0      # erwartet normalisiert [0,1]
    graph_boost: float = 0.0    # [0,1], z.B. wenn über Zitationskante erreicht
    score: float = field(default=0.0)


def hybrid_score(c: Candidate, w: Weights) -> float:
    return round(w.vector * c.vector_score + w.fts * c.fts_score + w.graph * c.graph_boost, 6)


def apply_graph_boost(candidates: list[Candidate], cited_document_ids: set[int], boost: float = 1.0) -> None:
    """Hebt Kandidaten an, deren Dokument über eine Zitationskante referenziert ist."""
    for c in candidates:
        if c.document_id in cited_document_ids:
            c.graph_boost = max(c.graph_boost, boost)


def rank(candidates: list[Candidate], w: Weights | None = None, top_k: int = 8) -> list[Candidate]:
    w = w or Weights()
    for c in candidates:
        c.score = hybrid_score(c, w)
    # deterministisch: nach score desc, dann chunk_id asc
    ranked = sorted(candidates, key=lambda c: (-c.score, c.chunk_id))
    return ranked[:top_k]


def is_sufficient(ranked: list[Candidate], min_score: float = 0.35, min_count: int = 1) -> bool:
    """Genügend belegbare Grundlage vorhanden?"""
    strong = [c for c in ranked if c.score >= min_score]
    return len(strong) >= min_count


@dataclass
class ContextChunk:
    chunk_id: int
    document_id: int
    heading: str | None
    text: str


def build_context(chunks: list[ContextChunk], max_chars: int = 6000) -> tuple[str, list[int]]:
    """
    Baut den Kontextstring mit Quellen-Markern [S1], [S2], … und gibt die
    Reihenfolge der verwendeten chunk_ids zurück (für Quellenzuordnung).
    """
    parts: list[str] = []
    used: list[int] = []
    total = 0
    for i, ch in enumerate(chunks, start=1):
        tag = f"[S{i}]"
        head = f" {ch.heading}" if ch.heading else ""
        block = f"{tag}{head}\n{ch.text}"
        if total + len(block) > max_chars and used:
            break
        parts.append(block)
        used.append(ch.chunk_id)
        total += len(block)
    return "\n\n".join(parts), used


def validate_grounded(answer_source_ids: list[int], retrieved_ids: list[int]) -> bool:
    """
    Grounding-Garantie: Die Antwort darf nur Quellen zitieren, die auch
    abgerufen wurden. Fremde/erfundene Quellen → nicht grounded.
    """
    if not answer_source_ids:
        return False
    allowed = set(retrieved_ids)
    return all(sid in allowed for sid in answer_source_ids)
