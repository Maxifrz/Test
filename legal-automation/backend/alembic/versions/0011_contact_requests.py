"""contact_requests — Kontaktformular der öffentlichen Website

Revision ID: 0011
Revises: 0010
Create Date: 2026-07-05
"""
from alembic import op
import sqlalchemy as sa

revision = "0011"
down_revision = "0010"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "contact_requests",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("email", sa.String(320), nullable=False),
        sa.Column("phone", sa.String(50), nullable=True),
        sa.Column("standort", sa.String(100), nullable=True),
        sa.Column("rolle", sa.String(100), nullable=True),
        sa.Column("message", sa.Text, nullable=False),
        sa.Column("consent_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("status", sa.String(20), nullable=False, server_default="neu"),
        sa.Column("processed_by_id", sa.Integer, sa.ForeignKey("users.id"), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_contact_requests_status", "contact_requests", ["status"])


def downgrade() -> None:
    op.drop_index("ix_contact_requests_status", table_name="contact_requests")
    op.drop_table("contact_requests")
