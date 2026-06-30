"""DSGVO-Tools: Verarbeitungsverzeichnis, Retention, Löschanträge, Datenexporte

Revision ID: 0008
Revises: 0007
Create Date: 2025-01-01 00:00:00
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision = "0008"
down_revision = "0007"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "processing_records",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("purpose", sa.Text, nullable=False),
        sa.Column("legal_basis", sa.String(255), nullable=False),
        sa.Column("data_categories", JSONB, nullable=False, server_default="[]"),
        sa.Column("data_subjects", sa.Text, nullable=True),
        sa.Column("recipients", sa.Text, nullable=True),
        sa.Column("retention", sa.Text, nullable=True),
        sa.Column("tom", sa.Text, nullable=True),
        sa.Column("is_active", sa.Boolean, nullable=False, server_default=sa.true()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
    )

    op.create_table(
        "data_retention_policies",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("matter_type", sa.String(50), nullable=True),
        sa.Column("retention_years", sa.Integer, nullable=False),
        sa.Column("legal_basis", sa.String(255), nullable=True),
        sa.Column("is_active", sa.Boolean, nullable=False, server_default=sa.true()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
    )

    op.create_table(
        "erasure_requests",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("client_id", sa.Integer, sa.ForeignKey("clients.id"), nullable=False),
        sa.Column("requested_by_id", sa.Integer, sa.ForeignKey("users.id"), nullable=False),
        sa.Column("requested_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("status", sa.String(20), nullable=False, server_default="open"),
        sa.Column("reason", sa.Text, nullable=True),
        sa.Column("blocking_reasons", JSONB, nullable=True, server_default="[]"),
        sa.Column("decided_by_id", sa.Integer, sa.ForeignKey("users.id"), nullable=True),
        sa.Column("decided_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("executed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("certificate_path", sa.String(1000), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_erasure_requests_client_id", "erasure_requests", ["client_id"])
    op.create_index("ix_erasure_requests_status", "erasure_requests", ["status"])

    op.create_table(
        "data_exports",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("client_id", sa.Integer, sa.ForeignKey("clients.id"), nullable=False),
        sa.Column("requested_by_id", sa.Integer, sa.ForeignKey("users.id"), nullable=False),
        sa.Column("status", sa.String(20), nullable=False, server_default="pending"),
        sa.Column("token", sa.String(64), nullable=False),
        sa.Column("file_path", sa.String(1000), nullable=True),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("downloaded_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_data_exports_client_id", "data_exports", ["client_id"])
    op.create_unique_constraint("uq_data_exports_token", "data_exports", ["token"])
    op.create_index("ix_data_exports_token", "data_exports", ["token"])


def downgrade() -> None:
    op.drop_table("data_exports")
    op.drop_table("erasure_requests")
    op.drop_table("data_retention_policies")
    op.drop_table("processing_records")
