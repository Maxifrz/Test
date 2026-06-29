"""Core data model: clients, matters, matter_access

Revision ID: 0002
Revises: 0001
Create Date: 2025-01-01 00:00:00
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision = "0002"
down_revision = "0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # --- clients ---
    op.create_table(
        "clients",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("client_number", sa.String(20), unique=True, nullable=False),
        sa.Column("first_name", sa.String(255), nullable=False),
        sa.Column("last_name", sa.String(255), nullable=False),
        sa.Column("company_name", sa.String(255), nullable=True),
        sa.Column("is_company", sa.Boolean, default=False, nullable=False),
        sa.Column("email", sa.String(255), nullable=True),
        sa.Column("phone", sa.String(50), nullable=True),
        sa.Column("address_line1", sa.String(255), nullable=True),
        sa.Column("address_line2", sa.String(255), nullable=True),
        sa.Column("postal_code", sa.String(20), nullable=True),
        sa.Column("city", sa.String(100), nullable=True),
        sa.Column("country", sa.String(2), default="DE", nullable=False),
        # PII fields — stored encrypted (EncryptedText TypeDecorator applies Fernet at ORM level)
        sa.Column("date_of_birth", sa.String(255), nullable=True),
        sa.Column("tax_id", sa.String(255), nullable=True),
        sa.Column("notes", sa.String(4096), nullable=True),
        # DSGVO
        sa.Column("dsgvo_consent_given_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("dsgvo_legal_basis", sa.String(50), nullable=True),
        sa.Column("created_by_id", sa.Integer, nullable=True),
        # SoftDeleteMixin + TimestampMixin
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("deleted_by_id", sa.Integer, nullable=True),
    )
    op.create_index("ix_clients_client_number", "clients", ["client_number"])
    op.create_index("ix_clients_email", "clients", ["email"])
    op.create_index("ix_clients_last_name", "clients", ["last_name"])

    # --- matters ---
    op.create_table(
        "matters",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("matter_number", sa.String(30), unique=True, nullable=False),
        sa.Column("title", sa.String(500), nullable=False),
        sa.Column("matter_type", sa.String(50), nullable=False),
        sa.Column("status", sa.String(30), nullable=False, server_default="open"),
        sa.Column("client_id", sa.Integer, sa.ForeignKey("clients.id"), nullable=False),
        sa.Column("lead_anwalt_id", sa.Integer, sa.ForeignKey("users.id"), nullable=False),
        sa.Column("created_by_id", sa.Integer, sa.ForeignKey("users.id"), nullable=False),
        sa.Column("court_file_ref", sa.String(100), nullable=True),
        sa.Column("court_name", sa.String(255), nullable=True),
        sa.Column("opposing_party", sa.String(255), nullable=True),
        sa.Column("opposing_counsel", sa.String(255), nullable=True),
        sa.Column("opened_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("closed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("statute_of_limitations", sa.DateTime(timezone=True), nullable=True),
        sa.Column("retention_years", sa.Integer, default=6, nullable=False),
        sa.Column("description", sa.Text, nullable=True),
        # SoftDeleteMixin + TimestampMixin
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("deleted_by_id", sa.Integer, nullable=True),
    )
    op.create_index("ix_matters_matter_number", "matters", ["matter_number"])
    op.create_index("ix_matters_client_id", "matters", ["client_id"])
    op.create_index("ix_matters_lead_anwalt_id", "matters", ["lead_anwalt_id"])
    op.create_index("ix_matters_status", "matters", ["status"])

    # --- matter_access ---
    op.create_table(
        "matter_access",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("user_id", sa.Integer, sa.ForeignKey("users.id"), nullable=False),
        sa.Column("matter_id", sa.Integer, sa.ForeignKey("matters.id"), nullable=False),
        sa.Column("matter_role", sa.String(20), nullable=False),
        sa.Column("granted_by_id", sa.Integer, sa.ForeignKey("users.id"), nullable=False),
        sa.Column("granted_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("revoked_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_matter_access_user_id", "matter_access", ["user_id"])
    op.create_index("ix_matter_access_matter_id", "matter_access", ["matter_id"])
    op.create_unique_constraint(
        "uq_matter_access_user_matter", "matter_access", ["user_id", "matter_id"]
    )


def downgrade() -> None:
    op.drop_table("matter_access")
    op.drop_table("matters")
    op.drop_table("clients")
