"""
Hybrid-Retrieval-Ranking + Grounding-Regeln (GraphRAG V3.0).

Reine Logik (keine DB/ML) → unit-testbar. Die eigentlichen pgvector-/tsvector-
Queries und die Graph-Expansion liefern Kandidaten mit Roh-Scores; hier werden
sie kombiniert, gewichtet, gerankt — und es wird entschieden, ob die Grundlage
für eine belegte Antwort ausreicht (sonst: Ablehnung statt Halluzination).
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field

# Antwort, wenn keine ausreichende Quellengrundlage gefunden wurde (Anti-Halluzination)
GROUNDING_REFUSAL = (
    "Zu dieser Frage wurde im hinterlegten Rechtsbestand keine ausreichend "
    "belegbare Grundlage gefunden. Bitte anwaltlich prüfen / Quelle ergänzen."
)

# Antwort, wenn zwar Quellen zitiert wurden, der Text aber nicht durch sie
# gedeckt ist (Halluzination mit korrekt aussehendem Marker).
UNSUPPORTED_REFUSAL = (
    "Der Entwurf war nicht vollständig durch die abgerufenen Quellen gedeckt "
    "und wurde deshalb verworfen. Bitte Frage präzisieren oder Quelle ergänzen."
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
    rerank_score: float = 0.0   # [0,1], lexikalische Deckung mit der Frage
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


# Der Korpus wird aus öffentlichen Quellen bulk-ingestiert und ist damit
# NICHT vertrauenswürdig. Diese Muster würden im Kontext sonst wie
# Struktur-Elemente des Prompts wirken (indirekte Prompt-Injection):
#   - "[S3]" in einem Gesetzestext täuscht einen Quellenmarker vor
#   - "=== FRAGE ===" beendet scheinbar den Kontextblock
_MARKER_LIKE = re.compile(r"\[\s*[Ss]\s*\d+[^\]]*\]")
_DELIMITER_LIKE = re.compile(r"^\s*={2,}.*$", re.MULTILINE)


def neutralize_untrusted(text: str) -> str:
    """
    Entschärft Prompt-Struktur-Muster in nicht vertrauenswürdigem Text
    (Korpus-Chunks, Nutzerfragen), ohne den Inhalt zu verfälschen: Marker
    werden in runde Klammern überführt, Trennlinien entwertet.
    """
    out = _MARKER_LIKE.sub(lambda m: "(" + m.group(0)[1:-1] + ")", text)
    out = _DELIMITER_LIKE.sub(lambda m: m.group(0).replace("=", "-"), out)
    return out


def sanitize_question(question: str, max_chars: int = 2000) -> str:
    """
    Normalisiert und kürzt die Nutzerfrage. Steuerzeichen (inkl. der für
    Prompt-Injection beliebten Bidi-/Zero-Width-Zeichen) fliegen raus.
    """
    q = unicodedata.normalize("NFKC", question)
    q = "".join(c for c in q if c == "\n" or not unicodedata.category(c).startswith("C"))
    q = neutralize_untrusted(q).strip()
    return q[:max_chars]


def build_context(chunks: list[ContextChunk], max_chars: int = 6000) -> tuple[str, list[int]]:
    """
    Baut den Kontextstring mit Quellen-Markern [S1], [S2], … und gibt die
    Reihenfolge der verwendeten chunk_ids zurück (für Quellenzuordnung).

    Chunk-Text und Überschrift werden vorher entschärft — nur die von uns
    gesetzten Marker dürfen wie Marker aussehen.
    """
    parts: list[str] = []
    used: list[int] = []
    total = 0
    for i, ch in enumerate(chunks, start=1):
        tag = f"[S{i}]"
        heading = neutralize_untrusted(ch.heading) if ch.heading else ""
        head = f" {heading}" if heading else ""
        block = f"{tag}{head}\n{neutralize_untrusted(ch.text)}"
        if total + len(block) > max_chars and used:
            break
        parts.append(block)
        used.append(ch.chunk_id)
        total += len(block)
    return "\n\n".join(parts), used


def validate_grounded(answer_source_ids: list[int], retrieved_ids: list[int]) -> bool:
    """
    Formale Grounding-Prüfung: Die Antwort darf nur Quellen zitieren, die auch
    abgerufen wurden. Fremde/erfundene Quellen → nicht grounded.

    ACHTUNG: Das prüft nur die MARKER, nicht den Aussagegehalt. Die inhaltliche
    Deckung prüft `claim_support` / `validate_claim_support`.
    """
    if not answer_source_ids:
        return False
    allowed = set(retrieved_ids)
    return all(sid in allowed for sid in answer_source_ids)


# --- Inhaltliche Deckung (über die reine Marker-Prüfung hinaus) ---

_WORD = re.compile(r"[\wäöüßÄÖÜ]{4,}", re.UNICODE)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

# Juristische Allerweltswörter tragen keine Belegkraft und würden die
# Überlappung künstlich anheben.
_STOPWORDS = frozenset({
    "aber", "auch", "durch", "eine", "einem", "einen", "einer", "eines",
    "gemäß", "gemaess", "iarr", "kann", "muss", "nach", "nicht", "oder",
    "sich", "sind", "soweit", "über", "ueber", "unter", "wenn", "werden",
    "wird", "zwischen", "dabei", "dazu", "diese", "dieser", "dieses",
    "damit", "sowie", "jedoch", "somit", "daher", "dass",
})


def _tokens(text: str) -> set[str]:
    return {w.lower() for w in _WORD.findall(text)} - _STOPWORDS


def claim_support(sentence: str, source_text: str) -> float:
    """
    Anteil der inhaltstragenden Wörter eines Antwortsatzes, die auch in der
    zitierten Quelle vorkommen. 1.0 = vollständig gedeckt, 0.0 = gar nicht.

    Bewusst ein einfaches, erklärbares Maß: es soll frei erfundene Passagen
    auffangen, nicht Textgleichheit erzwingen. Ein Modell, das paraphrasiert,
    bleibt über der Schwelle; eines, das eine Fundstelle erfindet, nicht.
    """
    claim = _tokens(sentence)
    if not claim:
        return 1.0  # Satz ohne Inhaltswörter (z. B. "Im Ergebnis:") — neutral
    return len(claim & _tokens(source_text)) / len(claim)


def validate_claim_support(
    answer: str,
    cited_texts: list[str],
    min_support: float = 0.18,
) -> tuple[bool, list[str]]:
    """
    Prüft jeden belegten Antwortsatz gegen die Vereinigung der zitierten
    Quellen. Gibt (alles_gedeckt, ungedeckte_Sätze) zurück.

    min_support = 0 schaltet die Prüfung ab (Verhalten wie vor V3.1).
    """
    if min_support <= 0 or not cited_texts:
        return True, []
    corpus = "\n".join(cited_texts)
    unsupported: list[str] = []
    for raw in _SENTENCE_SPLIT.split(answer):
        sentence = raw.strip()
        # Nur Sätze prüfen, die überhaupt einen Beleg beanspruchen.
        if len(sentence) < 25 or "[S" not in sentence:
            continue
        if claim_support(sentence, corpus) < min_support:
            unsupported.append(sentence)
    return not unsupported, unsupported


# --- Zweite Stufe: Reranking gegen die Frage ---

def question_overlap(question: str, chunk_text: str, heading: str | None = None) -> float:
    """
    Lexikalische Deckung zwischen Frage und Chunk (0..1). Die Überschrift zählt
    doppelt: bei Gesetzestexten steht dort die Norm ("§ 133 InsO"), und genau
    danach wird gefragt.
    """
    q = _tokens(question)
    if not q:
        return 0.0
    body = _tokens(chunk_text)
    head = _tokens(heading) if heading else set()
    hits = len(q & body) + len(q & head)
    return min(1.0, hits / len(q))


def rerank(
    candidates: list[Candidate],
    question: str,
    texts: dict[int, tuple[str | None, str]],
    top_k: int = 8,
    weight: float = 0.35,
) -> list[Candidate]:
    """
    Bewertet die Kandidaten des Hybrid-Retrievals ein zweites Mal, diesmal
    direkt gegen den Fragetext. Das korrigiert den typischen Fehler reiner
    Vektorsuche: thematisch benachbarte, aber nicht einschlägige Chunks.

    `texts` bildet chunk_id -> (heading, text) ab. Kandidaten ohne Text
    behalten ihren Hybrid-Score (kein Nachteil durch fehlende Daten).
    Deterministisch: bei Score-Gleichstand entscheidet die chunk_id.
    """
    for c in candidates:
        entry = texts.get(c.chunk_id)
        if entry is None:
            continue
        heading, text = entry
        c.rerank_score = question_overlap(question, text, heading)
        c.score = round((1 - weight) * c.score + weight * c.rerank_score, 6)
    return sorted(candidates, key=lambda c: (-c.score, c.chunk_id))[:top_k]
