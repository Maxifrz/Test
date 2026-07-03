"""KI-Rechtsrecherche (V3.0 GraphRAG): Wissensgraph, pgvector, FTS, Audit

Revision ID: 0010
Revises: 0009
Create Date: 2025-01-01 00:00:00

Voraussetzung: pgvector-Extension im Postgres-Image (pgvector/pgvector:pg16
oder Extension-Paket). Embedding-Dimension 768 = nomic-embed-text; muss zu
Settings.KI_EMBED_DIM passen.
"""
from alembic import op
import sqlalchemy as sa
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import JSONB

revision = "0010"
down_revision = "0009"
branch_labels = None
depends_on = None

EMBED_DIM = 768


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "legal_documents",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("source_type", sa.String(30), nullable=False),
        sa.Column("external_id", sa.String(200), nullable=True),
        sa.Column("title", sa.String(500), nullable=False),
        sa.Column("jurisdiction", sa.String(50), nullable=True),
        sa.Column("doc_date", sa.Date, nullable=True),
        sa.Column("url_or_ref", sa.String(1000), nullable=True),
        sa.Column("matter_id", sa.Integer, sa.ForeignKey("matters.id"), nullable=True),
        sa.Column("checksum", sa.String(64), nullable=False),
        sa.Column("is_active", sa.Boolean, nullable=False, server_default=sa.true()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_legal_documents_source_type", "legal_documents", ["source_type"])
    op.create_index("ix_legal_documents_external_id", "legal_documents", ["external_id"])
    op.create_index("ix_legal_documents_matter_id", "legal_documents", ["matter_id"])
    op.create_unique_constraint("uq_legal_documents_checksum", "legal_documents", ["checksum"])

    op.create_table(
        "legal_chunks",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("document_id", sa.Integer, sa.ForeignKey("legal_documents.id"), nullable=False),
        sa.Column("ord", sa.Integer, nullable=False, server_default="0"),
        sa.Column("heading", sa.String(200), nullable=True),
        sa.Column("text", sa.Text, nullable=False),
        sa.Column("embedding", Vector(EMBED_DIM), nullable=True),
        sa.Column("token_count", sa.Integer, nullable=True),
    )
    op.create_index("ix_legal_chunks_document_id", "legal_chunks", ["document_id"])
    # Deutsche Volltextsuche als generierte Spalte + GIN
    op.execute(
        "ALTER TABLE legal_chunks ADD COLUMN fts tsvector "
        "GENERATED ALWAYS AS (to_tsvector('german', coalesce(heading, '') || ' ' || text)) STORED"
    )
    op.execute("CREATE INDEX ix_legal_chunks_fts ON legal_chunks USING GIN (fts)")
    # Vektor-Index (HNSW, Cosine). Bei sehr großen Korpora Parameter tunen.
    op.execute(
        "CREATE INDEX ix_legal_chunks_embedding ON legal_chunks "
        "USING hnsw (embedding vector_cosine_ops)"
    )

    op.create_table(
        "legal_citations",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("chunk_id", sa.Integer, sa.ForeignKey("legal_chunks.id"), nullable=False),
        sa.Column("document_id", sa.Integer, sa.ForeignKey("legal_documents.id"), nullable=False),
        sa.Column("citation_type", sa.String(20), nullable=False),
        sa.Column("raw", sa.String(300), nullable=False),
        sa.Column("normalized", sa.String(300), nullable=False),
        sa.Column("target_document_id", sa.Integer, sa.ForeignKey("legal_documents.id"), nullable=True),
    )
    op.create_index("ix_legal_citations_chunk_id", "legal_citations", ["chunk_id"])
    op.create_index("ix_legal_citations_document_id", "legal_citations", ["document_id"])
    op.create_index("ix_legal_citations_normalized", "legal_citations", ["normalized"])
    op.create_index("ix_legal_citations_target_document_id", "legal_citations", ["target_document_id"])

    op.create_table(
        "ki_queries",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("user_id", sa.Integer, sa.ForeignKey("users.id"), nullable=False),
        sa.Column("matter_id", sa.Integer, sa.ForeignKey("matters.id"), nullable=True),
        sa.Column("question", sa.Text, nullable=False),
        sa.Column("retrieved_chunk_ids", JSONB, nullable=False, server_default="[]"),
        sa.Column("answer", sa.Text, nullable=True),
        sa.Column("sources", JSONB, nullable=False, server_default="[]"),
        sa.Column("model", sa.String(100), nullable=True),
        sa.Column("grounded", sa.Boolean, nullable=False, server_default=sa.false()),
        sa.Column("feedback", sa.String(20), nullable=True),
        sa.Column("feedback_note", sa.Text, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_ki_queries_user_id", "ki_queries", ["user_id"])

    op.create_table(
        "ingestion_jobs",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("source", sa.String(100), nullable=False),
        sa.Column("status", sa.String(20), nullable=False, server_default="pending"),
        sa.Column("num_documents", sa.Integer, nullable=False, server_default="0"),
        sa.Column("num_chunks", sa.Integer, nullable=False, server_default="0"),
        sa.Column("num_duplicates", sa.Integer, nullable=False, server_default="0"),
        sa.Column("error", sa.Text, nullable=True),
        sa.Column("created_by_id", sa.Integer, sa.ForeignKey("users.id"), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
    )


def downgrade() -> None:
    op.drop_table("ingestion_jobs")
    op.drop_table("ki_queries")
    op.drop_table("legal_citations")
    op.drop_table("legal_chunks")
    op.drop_table("legal_documents")
