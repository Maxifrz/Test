"""Insolvenz: Forderungstabelle, Verteilungen, Gläubiger-Portal-Token

Revision ID: 0007
Revises: 0006
Create Date: 2025-01-01 00:00:00
"""
from alembic import op
import sqlalchemy as sa

revision = "0007"
down_revision = "0006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("matters", sa.Column("creditor_portal_token", sa.String(64), nullable=True))
    op.create_unique_constraint("uq_matters_creditor_portal_token", "matters", ["creditor_portal_token"])
    op.create_index("ix_matters_creditor_portal_token", "matters", ["creditor_portal_token"])

    op.create_table(
        "insolvency_claims",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("matter_id", sa.Integer, sa.ForeignKey("matters.id"), nullable=False),
        sa.Column("claim_number", sa.Integer, nullable=True),
        sa.Column("creditor_name", sa.String(255), nullable=False),
        sa.Column("creditor_email", sa.String(320), nullable=True),
        sa.Column("creditor_address", sa.Text, nullable=True),
        sa.Column("creditor_reference", sa.String(140), nullable=True),
        sa.Column("claim_amount", sa.Numeric(14, 2), nullable=False),
        sa.Column("established_amount", sa.Numeric(14, 2), nullable=True),
        sa.Column("claim_reason", sa.Text, nullable=True),
        sa.Column("rank", sa.String(30), nullable=False, server_default="insolvenz_38"),
        sa.Column("status", sa.String(20), nullable=False, server_default="angemeldet"),
        sa.Column("dispute_reason", sa.Text, nullable=True),
        sa.Column("source", sa.String(20), nullable=False, server_default="intern"),
        sa.Column("filed_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("created_by_id", sa.Integer, sa.ForeignKey("users.id"), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("deleted_by_id", sa.Integer, nullable=True),
    )
    op.create_index("ix_insolvency_claims_matter_id", "insolvency_claims", ["matter_id"])
    op.create_index("ix_insolvency_claims_rank", "insolvency_claims", ["rank"])
    op.create_index("ix_insolvency_claims_status", "insolvency_claims", ["status"])

    op.create_table(
        "distributions",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("matter_id", sa.Integer, sa.ForeignKey("matters.id"), nullable=False),
        sa.Column("distribution_type", sa.String(20), nullable=False, server_default="abschlag"),
        sa.Column("distributable_amount", sa.Numeric(14, 2), nullable=False),
        sa.Column("quote_38_pct", sa.Numeric(7, 4), nullable=False, server_default="0"),
        sa.Column("distributed_sum", sa.Numeric(14, 2), nullable=False, server_default="0"),
        sa.Column("remainder", sa.Numeric(14, 2), nullable=False, server_default="0"),
        sa.Column("created_by_id", sa.Integer, sa.ForeignKey("users.id"), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_distributions_matter_id", "distributions", ["matter_id"])

    op.create_table(
        "distribution_items",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("distribution_id", sa.Integer, sa.ForeignKey("distributions.id"), nullable=False),
        sa.Column("claim_id", sa.Integer, sa.ForeignKey("insolvency_claims.id"), nullable=False),
        sa.Column("established_amount", sa.Numeric(14, 2), nullable=False),
        sa.Column("amount", sa.Numeric(14, 2), nullable=False),
        sa.Column("quote_pct", sa.Numeric(7, 4), nullable=False, server_default="0"),
    )
    op.create_index("ix_distribution_items_distribution_id", "distribution_items", ["distribution_id"])


def downgrade() -> None:
    op.drop_table("distribution_items")
    op.drop_table("distributions")
    op.drop_table("insolvency_claims")
    op.drop_index("ix_matters_creditor_portal_token", table_name="matters")
    op.drop_constraint("uq_matters_creditor_portal_token", "matters", type_="unique")
    op.drop_column("matters", "creditor_portal_token")
