"""Mehrere Postfaecher (email_accounts) statt eines globalen .env-Kontos

Bisher lag genau ein Postfach in der .env. Bei 19 Standorten ist das keine
Feinheit, sondern eine Grundvoraussetzung: jeder Standort fuehrt eigene
Korrespondenz. Eine Aenderung erforderte ausserdem einen Neustart des Stacks.

Zugangsdaten sind feldverschluesselt (Fernet) -- sonst stehen sie im Klartext
in der Datenbank und damit in jedem Backup.

Revision ID: 0019
Revises: 0018
"""
import sqlalchemy as sa
from alembic import op

revision = "0019"
down_revision = "0018"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "email_accounts",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("name", sa.String(200), nullable=False),
        sa.Column("standort", sa.String(100), nullable=True),
        sa.Column("imap_host", sa.String(255), nullable=False),
        sa.Column("imap_port", sa.Integer(), nullable=False, server_default="993"),
        sa.Column("imap_ssl", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("imap_username", sa.String(255), nullable=False),
        sa.Column("imap_password", sa.Text(), nullable=True),
        sa.Column("imap_folder", sa.String(255), nullable=False, server_default="INBOX"),
        sa.Column("smtp_host", sa.String(255), nullable=False),
        sa.Column("smtp_port", sa.Integer(), nullable=False, server_default="587"),
        sa.Column("smtp_tls", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("smtp_username", sa.String(255), nullable=True),
        sa.Column("smtp_password", sa.Text(), nullable=True),
        sa.Column("from_name", sa.String(200), nullable=False, server_default="Kanzlei"),
        sa.Column("from_email", sa.String(320), nullable=False),
        sa.Column("is_active", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("is_default", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column("last_sync_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_sync_error", sa.Text(), nullable=True),
        sa.Column("last_seen_uid", sa.Integer(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False, server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("deleted_by_id", sa.Integer(), nullable=True),
    )
    op.create_index("ix_email_accounts_standort", "email_accounts", ["standort"])
    op.create_index("ix_email_accounts_is_active", "email_accounts", ["is_active"])
    # Hoechstens EIN Standard-Postfach -- sonst waere der Versand nicht
    # deterministisch.
    op.create_index(
        "uq_email_accounts_single_default",
        "email_accounts",
        ["is_default"],
        unique=True,
        postgresql_where=sa.text("is_default = true AND deleted_at IS NULL"),
    )

    op.add_column(
        "email_messages",
        sa.Column("account_id", sa.Integer(), sa.ForeignKey("email_accounts.id"), nullable=True),
    )
    op.create_index("ix_email_messages_account_id", "email_messages", ["account_id"])
    # Bestandsnachrichten behalten account_id NULL: sie stammen aus der Zeit
    # des .env-Einzelpostfachs. Der Versand faellt fuer sie auf die .env
    # zurueck, solange kein Konto angelegt ist.


def downgrade() -> None:
    op.drop_index("ix_email_messages_account_id", table_name="email_messages")
    op.drop_column("email_messages", "account_id")
    op.drop_index("uq_email_accounts_single_default", table_name="email_accounts")
    op.drop_index("ix_email_accounts_is_active", table_name="email_accounts")
    op.drop_index("ix_email_accounts_standort", table_name="email_accounts")
    op.drop_table("email_accounts")
