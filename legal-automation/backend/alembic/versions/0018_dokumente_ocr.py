"""Dokumentenverwaltung mit OCR und deutscher Volltextsuche

Eingescannte Dokumente waren bisher tote Anhaenge: ablegbar und abrufbar, aber
nicht durchsuchbar. Fuer eine Kanzlei ist der Volltext ueber die Handakte der
eigentliche Hebel.

Revision ID: 0018
Revises: 0017
"""
import sqlalchemy as sa
from alembic import op

revision = "0018"
down_revision = "0017"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "documents",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("matter_id", sa.Integer(), sa.ForeignKey("matters.id"), nullable=False),
        sa.Column("title", sa.String(500), nullable=False),
        sa.Column("kind", sa.String(30), nullable=False, server_default="sonstiges"),
        sa.Column("original_filename", sa.String(500), nullable=False),
        sa.Column("content_type", sa.String(255), nullable=True),
        sa.Column("size_bytes", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("storage_path", sa.String(1000), nullable=False),
        sa.Column("checksum", sa.String(64), nullable=False),
        sa.Column("document_date", sa.Date(), nullable=True),
        sa.Column("ocr_status", sa.String(20), nullable=False, server_default="pending"),
        sa.Column("ocr_error", sa.Text(), nullable=True),
        sa.Column("extracted_text", sa.Text(), nullable=True),
        sa.Column("page_count", sa.Integer(), nullable=True),
        sa.Column("text_from_ocr", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("uploaded_by_id", sa.Integer(), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("is_confidential", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("deleted_by_id", sa.Integer(), nullable=True),
    )
    op.create_index("ix_documents_matter_id", "documents", ["matter_id"])
    op.create_index("ix_documents_kind", "documents", ["kind"])
    op.create_index("ix_documents_checksum", "documents", ["checksum"])
    op.create_index("ix_documents_document_date", "documents", ["document_date"])
    op.create_index("ix_documents_ocr_status", "documents", ["ocr_status"])
    # Doppel-Uploads desselben Scans in dieselbe Akte verhindern
    op.create_index(
        "uq_documents_matter_checksum",
        "documents",
        ["matter_id", "checksum"],
        unique=True,
        postgresql_where=sa.text("deleted_at IS NULL"),
    )

    # Deutsche Volltextsuche ueber Titel + extrahierten Text. GIN-Index auf
    # einem Ausdruck: das haelt die Tabelle schmal und die Suche schnell.
    op.execute(
        """
        CREATE INDEX ix_documents_fts ON documents
        USING GIN (to_tsvector('german',
            coalesce(title, '') || ' ' || coalesce(extracted_text, '')))
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_documents_fts")
    op.drop_index("uq_documents_matter_checksum", table_name="documents")
    op.drop_index("ix_documents_ocr_status", table_name="documents")
    op.drop_index("ix_documents_document_date", table_name="documents")
    op.drop_index("ix_documents_checksum", table_name="documents")
    op.drop_index("ix_documents_kind", table_name="documents")
    op.drop_index("ix_documents_matter_id", table_name="documents")
    op.drop_table("documents")
