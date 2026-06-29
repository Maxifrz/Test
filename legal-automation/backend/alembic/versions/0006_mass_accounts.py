"""Insolvenz-Finanz: Massekonten, Massebuchungen, Importläufe, Zuordnungsregeln

Revision ID: 0006
Revises: 0005
Create Date: 2025-01-01 00:00:00
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision = "0006"
down_revision = "0005"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "mass_accounts",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("matter_id", sa.Integer, sa.ForeignKey("matters.id"), nullable=False),
        sa.Column("iban", sa.String(34), nullable=False),
        sa.Column("bic", sa.String(11), nullable=True),
        sa.Column("bank_name", sa.String(255), nullable=True),
        sa.Column("account_label", sa.String(255), nullable=True),
        sa.Column("account_type", sa.String(20), nullable=False, server_default="sonderkonto"),
        sa.Column("opening_balance", sa.Numeric(14, 2), nullable=False, server_default="0"),
        sa.Column("currency", sa.String(3), nullable=False, server_default="EUR"),
        sa.Column("is_active", sa.Boolean, nullable=False, server_default=sa.true()),
        sa.Column("created_by_id", sa.Integer, sa.ForeignKey("users.id"), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("deleted_by_id", sa.Integer, nullable=True),
    )
    op.create_index("ix_mass_accounts_matter_id", "mass_accounts", ["matter_id"])
    op.create_unique_constraint("uq_mass_accounts_iban", "mass_accounts", ["iban"])

    op.create_table(
        "import_batches",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("mass_account_id", sa.Integer, sa.ForeignKey("mass_accounts.id"), nullable=True),
        sa.Column("filename", sa.String(500), nullable=False),
        sa.Column("format", sa.String(10), nullable=False),
        sa.Column("imported_by_id", sa.Integer, sa.ForeignKey("users.id"), nullable=False),
        sa.Column("num_transactions", sa.Integer, nullable=False, server_default="0"),
        sa.Column("num_assigned", sa.Integer, nullable=False, server_default="0"),
        sa.Column("num_unassigned", sa.Integer, nullable=False, server_default="0"),
        sa.Column("num_duplicates", sa.Integer, nullable=False, server_default="0"),
        sa.Column("statement_opening_balance", sa.Numeric(14, 2), nullable=True),
        sa.Column("statement_closing_balance", sa.Numeric(14, 2), nullable=True),
        sa.Column("computed_closing_balance", sa.Numeric(14, 2), nullable=True),
        sa.Column("reconciled", sa.Boolean, nullable=False, server_default=sa.false()),
        sa.Column("status", sa.String(20), nullable=False, server_default="completed"),
        sa.Column("storage_path", sa.String(1000), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
    )

    op.create_table(
        "mass_transactions",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("mass_account_id", sa.Integer, sa.ForeignKey("mass_accounts.id"), nullable=False),
        sa.Column("matter_id", sa.Integer, sa.ForeignKey("matters.id"), nullable=False),
        sa.Column("import_batch_id", sa.Integer, sa.ForeignKey("import_batches.id"), nullable=True),
        sa.Column("booking_date", sa.Date, nullable=True),
        sa.Column("value_date", sa.Date, nullable=True),
        sa.Column("amount", sa.Numeric(14, 2), nullable=False),
        sa.Column("direction", sa.String(3), nullable=False),
        sa.Column("currency", sa.String(3), nullable=False, server_default="EUR"),
        sa.Column("purpose", sa.Text, nullable=True),
        sa.Column("counterparty_name", sa.String(255), nullable=True),
        sa.Column("counterparty_iban", sa.String(34), nullable=True),
        sa.Column("category", sa.String(30), nullable=False, server_default="unassigned"),
        sa.Column("end_to_end_id", sa.String(140), nullable=True),
        sa.Column("bank_reference", sa.String(140), nullable=True),
        sa.Column("dedup_key", sa.String(64), nullable=False),
        sa.Column("is_reconciled", sa.Boolean, nullable=False, server_default=sa.false()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_mass_transactions_mass_account_id", "mass_transactions", ["mass_account_id"])
    op.create_index("ix_mass_transactions_matter_id", "mass_transactions", ["matter_id"])
    op.create_index("ix_mass_transactions_category", "mass_transactions", ["category"])
    op.create_index("ix_mass_transactions_dedup_key", "mass_transactions", ["dedup_key"])
    op.create_unique_constraint("uq_mass_transactions_dedup", "mass_transactions", ["dedup_key"])

    op.create_table(
        "mass_assignment_rules",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("priority", sa.Integer, nullable=False, server_default="100"),
        sa.Column("conditions", JSONB, nullable=False, server_default="{}"),
        sa.Column("category", sa.String(30), nullable=False),
        sa.Column("is_active", sa.Boolean, nullable=False, server_default=sa.true()),
        sa.Column("created_by_id", sa.Integer, sa.ForeignKey("users.id"), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_mass_assignment_rules_priority", "mass_assignment_rules", ["priority"])


def downgrade() -> None:
    op.drop_table("mass_assignment_rules")
    op.drop_table("mass_transactions")
    op.drop_table("import_batches")
    op.drop_table("mass_accounts")
