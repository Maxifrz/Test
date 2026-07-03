"""
KI-Rechtsrecherche-Service (V3.0 GraphRAG): Ingestion + Hybrid-Query.

Ingestion: Dokument → struktur-bewusstes Chunking → Embeddings (Ollama) →
Zitations-Kanten → Upsert mit Checksum-Dedup.

Query: Frage → Embedding → Hybrid-Retrieval (pgvector-Cosine + deutsche FTS)
→ Graph-Expansion über Zitationskanten → Ranking → Grounded Generation mit
[S#]-Belegen → Grounding-Validierung (sonst Ablehnung) → Audit in ki_queries.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

from sqlalchemy import select, text as sql_text
from sqlalchemy.ext.asyncio import AsyncSession

from app.ai.kri import answer as answer_mod
from app.ai.kri.chunking import chunk as chunk_text
from app.ai.kri.citations import extract_citations
from app.ai.kri.retrieval import (
    GROUNDING_REFUSAL,
    Candidate,
    ContextChunk,
    apply_graph_boost,
    build_context,
    is_sufficient,
    rank,
    validate_grounded,
)
from app.core.config import get_settings
from app.models.legal_knowledge import (
    KiQuery,
    LegalChunk,
    LegalCitation,
    LegalDocument,
)


def _checksum(source_type: str, external_id: str | None, text: str) -> str:
    h = hashlib.sha256()
    h.update(f"{source_type}|{external_id or ''}|".encode())
    h.update(text.encode())
    return h.hexdigest()


@dataclass
class IngestResult:
    document_id: int | None
    num_chunks: int
    duplicate: bool


async def ingest_document(
    db: AsyncSession,
    *,
    source_type: str,
    title: str,
    text: str,
    external_id: str | None = None,
    jurisdiction: str | None = None,
    url_or_ref: str | None = None,
    matter_id: int | None = None,
    embedder=None,  # async fn(text)->list[float]; None = ohne Embeddings (später nachziehbar)
) -> IngestResult:
    """Nimmt ein Dokument in den Wissensgraphen auf (idempotent via Checksum)."""
    checksum = _checksum(source_type, external_id, text)
    existing = await db.execute(select(LegalDocument.id).where(LegalDocument.checksum == checksum))
    if existing.scalar_one_or_none() is not None:
        return IngestResult(document_id=None, num_chunks=0, duplicate=True)

    doc = LegalDocument(
        source_type=source_type, external_id=external_id, title=title,
        jurisdiction=jurisdiction, url_or_ref=url_or_ref, matter_id=matter_id,
        checksum=checksum,
    )
    db.add(doc)
    await db.flush()

    chunks = chunk_text(text, source_type)
    for c in chunks:
        embedding = await embedder(c.text) if embedder else None
        chunk_row = LegalChunk(
            document_id=doc.id, ord=c.ord, heading=c.heading, text=c.text,
            embedding=embedding, token_count=len(c.text.split()),
        )
        db.add(chunk_row)
        await db.flush()
        for cit in extract_citations(c.text):
            db.add(
                LegalCitation(
                    chunk_id=chunk_row.id, document_id=doc.id,
                    citation_type=cit.citation_type, raw=cit.raw[:300],
                    normalized=cit.normalized[:300],
                )
            )
    await db.commit()
    return IngestResult(document_id=doc.id, num_chunks=len(chunks), duplicate=False)


async def resolve_citation_targets(db: AsyncSession) -> int:
    """Verknüpft Zitationskanten mit Ziel-Dokumenten im Korpus (external_id-Match)."""
    result = await db.execute(
        sql_text(
            """
            UPDATE legal_citations c
            SET target_document_id = d.id
            FROM legal_documents d
            WHERE c.target_document_id IS NULL
              AND d.external_id IS NOT NULL
              AND c.normalized ILIKE '%' || d.external_id || '%'
            """
        )
    )
    await db.commit()
    return result.rowcount or 0


# --- Query-Pipeline ---

async def _vector_candidates(db, query_embedding, allowed_matter_ids, limit: int):
    """pgvector-Cosine-Similarity; interne Dokumente nur mit Aktenzugriff."""
    query = (
        select(
            LegalChunk.id,
            LegalChunk.document_id,
            (1 - LegalChunk.embedding.cosine_distance(query_embedding)).label("sim"),
        )
        .join(LegalDocument, LegalDocument.id == LegalChunk.document_id)
        .where(LegalDocument.is_active == True, LegalChunk.embedding.isnot(None))  # noqa: E712
        .order_by(LegalChunk.embedding.cosine_distance(query_embedding))
        .limit(limit)
    )
    if allowed_matter_ids is not None:
        query = query.where(
            (LegalDocument.matter_id.is_(None))
            | (LegalDocument.matter_id.in_(allowed_matter_ids))
        )
    result = await db.execute(query)
    return {row.id: (row.document_id, max(0.0, float(row.sim))) for row in result}


async def _fts_candidates(db, question: str, allowed_matter_ids, limit: int):
    """Deutsche Volltextsuche mit ts_rank (auf [0,1] gedeckelt)."""
    sql = """
        SELECT c.id, c.document_id,
               LEAST(1.0, ts_rank(c.fts, plainto_tsquery('german', :q))) AS score
        FROM legal_chunks c
        JOIN legal_documents d ON d.id = c.document_id
        WHERE d.is_active AND c.fts @@ plainto_tsquery('german', :q)
    """
    params: dict = {"q": question, "lim": limit}
    if allowed_matter_ids is not None:
        sql += " AND (d.matter_id IS NULL OR d.matter_id = ANY(:ids))"
        params["ids"] = list(allowed_matter_ids)
    sql += " ORDER BY score DESC LIMIT :lim"
    result = await db.execute(sql_text(sql), params)
    return {row.id: (row.document_id, float(row.score)) for row in result}


async def _cited_document_ids(db, chunk_ids: list[int]) -> set[int]:
    """Graph-Expansion: Dokumente, die von den Kandidaten-Chunks zitiert werden."""
    if not chunk_ids:
        return set()
    result = await db.execute(
        select(LegalCitation.target_document_id).where(
            LegalCitation.chunk_id.in_(chunk_ids),
            LegalCitation.target_document_id.isnot(None),
        )
    )
    return set(result.scalars().all())


@dataclass
class QueryResult:
    answer: str
    grounded: bool
    sources: list[dict]
    model: str
    query_id: int | None = None


async def query_knowledge(
    db: AsyncSession,
    *,
    question: str,
    user_id: int,
    matter_id: int | None = None,
    allowed_matter_ids: set[int] | None = None,
    llm=None,  # OllamaClient-kompatibel; None → wird erzeugt
) -> QueryResult:
    settings = get_settings()
    if llm is None:
        from app.ai.llm.ollama_client import OllamaClient
        llm = OllamaClient()

    # 1) Hybrid-Retrieval
    query_embedding = await llm.embed(question)
    top_k = settings.KI_RETRIEVAL_TOP_K
    vec = await _vector_candidates(db, query_embedding, allowed_matter_ids, top_k * 3)
    fts = await _fts_candidates(db, question, allowed_matter_ids, top_k * 3)

    by_id: dict[int, Candidate] = {}
    for cid, (doc_id, sim) in vec.items():
        by_id[cid] = Candidate(chunk_id=cid, document_id=doc_id, vector_score=sim)
    for cid, (doc_id, score) in fts.items():
        if cid in by_id:
            by_id[cid].fts_score = score
        else:
            by_id[cid] = Candidate(chunk_id=cid, document_id=doc_id, fts_score=score)

    candidates = list(by_id.values())
    cited_docs = await _cited_document_ids(db, [c.chunk_id for c in candidates])
    apply_graph_boost(candidates, cited_docs)
    ranked = rank(candidates, top_k=top_k)

    async def _persist(answer_text: str, grounded: bool, sources: list[dict]) -> int:
        q = KiQuery(
            user_id=user_id, matter_id=matter_id, question=question,
            retrieved_chunk_ids=[c.chunk_id for c in ranked],
            answer=answer_text, sources=sources, model=settings.KI_LLM_MODEL,
            grounded=grounded,
        )
        db.add(q)
        await db.commit()
        await db.refresh(q)
        return q.id

    # 2) Suffizienz-Schwelle: ohne belegbare Grundlage keine Generierung
    if not is_sufficient(ranked, min_score=settings.KI_MIN_GROUNDING_SCORE):
        qid = await _persist(GROUNDING_REFUSAL, False, [])
        return QueryResult(GROUNDING_REFUSAL, False, [], settings.KI_LLM_MODEL, qid)

    # 3) Kontext + Generierung
    chunk_rows = (await db.execute(
        select(LegalChunk).where(LegalChunk.id.in_([c.chunk_id for c in ranked]))
    )).scalars().all()
    row_by_id = {r.id: r for r in chunk_rows}
    ordered = [row_by_id[c.chunk_id] for c in ranked if c.chunk_id in row_by_id]
    ctx_chunks = [
        ContextChunk(chunk_id=r.id, document_id=r.document_id, heading=r.heading, text=r.text)
        for r in ordered
    ]
    context, used_chunk_ids = build_context(ctx_chunks)
    raw_answer = await llm.generate(answer_mod.build_prompt(question, context))

    # 4) Grounding-Validierung: Antwort darf nur abgerufene Quellen zitieren
    if answer_mod.is_refusal(raw_answer):
        qid = await _persist(GROUNDING_REFUSAL, False, [])
        return QueryResult(GROUNDING_REFUSAL, False, [], settings.KI_LLM_MODEL, qid)

    refs = answer_mod.extract_source_refs(raw_answer)
    cited_chunk_ids = answer_mod.refs_to_chunk_ids(refs, used_chunk_ids)
    if not validate_grounded(cited_chunk_ids, used_chunk_ids):
        qid = await _persist(GROUNDING_REFUSAL, False, [])
        return QueryResult(GROUNDING_REFUSAL, False, [], settings.KI_LLM_MODEL, qid)

    # 5) Quellenliste für die zitierten Chunks
    doc_rows = (await db.execute(
        select(LegalDocument).where(
            LegalDocument.id.in_({row_by_id[cid].document_id for cid in cited_chunk_ids})
        )
    )).scalars().all()
    docs = {d.id: d for d in doc_rows}
    sources = []
    for ref, cid in zip(refs, cited_chunk_ids):
        r = row_by_id[cid]
        d = docs.get(r.document_id)
        sources.append({
            "marker": f"S{ref}",
            "chunk_id": cid,
            "heading": r.heading,
            "document_title": d.title if d else "?",
            "source_type": d.source_type if d else "?",
            "external_id": d.external_id if d else None,
            "url_or_ref": d.url_or_ref if d else None,
        })

    qid = await _persist(raw_answer, True, sources)
    return QueryResult(raw_answer, True, sources, settings.KI_LLM_MODEL, qid)
