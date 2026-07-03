"""
Wissensgraph der KI-Rechtsrecherche (V3.0 GraphRAG).

- legal_documents: Quelle (Gesetz/Urteil/intern) mit Checksum-Dedup
- legal_chunks:    zitierfähige Abschnitte mit Embedding (pgvector) + deutscher FTS
- legal_citations: Graphkanten aus der Zitats-Extraktion (app/ai/kri/citations.py)
- ki_queries:      Audit + Feedback jeder Recherche-Anfrage
- ingestion_jobs:  Protokoll der Ingestion-Läufe

Die Embedding-Dimension (Spalte `embedding`) ist in Migration 0010 fixiert und
muss zu Settings.KI_EMBED_DIM passen (Default 768, nomic-embed-text).
"""
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import Boolean, Date, DateTime, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin

SOURCE_TYPES = {
    "gesetz", "urteil", "eurlex",
    "intern_akte", "intern_schriftsatz", "intern_transkript",
}

EMBED_DIM = 768  # muss zur Migration + KI_EMBED_DIM passen


class LegalDocument(Base, TimestampMixin):
    __tablename__ = "legal_documents"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source_type: Mapped[str] = mapped_column(String(30), nullable=False, index=True)
    # Norm-Kürzel ("InsO"), ECLI, Az oder interne Referenz
    external_id: Mapped[str | None] = mapped_column(String(200), nullable=True, index=True)
    title: Mapped[str] = mapped_column(String(500), nullable=False)
    jurisdiction: Mapped[str | None] = mapped_column(String(50), nullable=True)
    doc_date: Mapped[datetime | None] = mapped_column(Date, nullable=True)
    url_or_ref: Mapped[str | None] = mapped_column(String(1000), nullable=True)
    # Nur bei internen Quellen gesetzt → erbt matter_access-RBAC
    matter_id: Mapped[int | None] = mapped_column(ForeignKey("matters.id"), nullable=True, index=True)
    checksum: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    is_active: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)

    chunks: Mapped[list["LegalChunk"]] = relationship(
        "LegalChunk", back_populates="document", cascade="all, delete-orphan"
    )


class LegalChunk(Base):
    __tablename__ = "legal_chunks"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("legal_documents.id"), nullable=False, index=True)
    ord: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    # z. B. "§ 133 Abs. 2" oder "Rn. 12" — macht den Chunk präzise zitierbar
    heading: Mapped[str | None] = mapped_column(String(200), nullable=True)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    embedding = mapped_column(Vector(EMBED_DIM), nullable=True)
    token_count: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # fts-tsvector wird in der Migration als generierte Spalte angelegt

    document: Mapped["LegalDocument"] = relationship("LegalDocument", back_populates="chunks")


class LegalCitation(Base):
    """Graphkante: Chunk → zitierte Norm/Urteil (roh + normalisiert)."""

    __tablename__ = "legal_citations"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    chunk_id: Mapped[int] = mapped_column(ForeignKey("legal_chunks.id"), nullable=False, index=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("legal_documents.id"), nullable=False, index=True)
    citation_type: Mapped[str] = mapped_column(String(20), nullable=False)  # norm|aktenzeichen|ecli
    raw: Mapped[str] = mapped_column(String(300), nullable=False)
    normalized: Mapped[str] = mapped_column(String(300), nullable=False, index=True)
    # aufgelöstes Ziel, falls das zitierte Dokument im Korpus existiert
    target_document_id: Mapped[int | None] = mapped_column(
        ForeignKey("legal_documents.id"), nullable=True, index=True
    )


class KiQuery(Base, TimestampMixin):
    """Audit + Feedback jeder Recherche-Anfrage (Qualitäts-Loop, DSGVO-Nachweis)."""

    __tablename__ = "ki_queries"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"), nullable=False, index=True)
    matter_id: Mapped[int | None] = mapped_column(ForeignKey("matters.id"), nullable=True)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    retrieved_chunk_ids: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    answer: Mapped[str | None] = mapped_column(Text, nullable=True)
    sources: Mapped[list] = mapped_column(JSONB, nullable=False, default=list)
    model: Mapped[str | None] = mapped_column(String(100), nullable=True)
    grounded: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    feedback: Mapped[str | None] = mapped_column(String(20), nullable=True)  # up|down
    feedback_note: Mapped[str | None] = mapped_column(Text, nullable=True)


class IngestionJob(Base, TimestampMixin):
    __tablename__ = "ingestion_jobs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source: Mapped[str] = mapped_column(String(100), nullable=False)
    status: Mapped[str] = mapped_column(String(20), nullable=False, default="pending")
    # pending | running | done | failed
    num_documents: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    num_chunks: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    num_duplicates: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_by_id: Mapped[int | None] = mapped_column(ForeignKey("users.id"), nullable=True)
