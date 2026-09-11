"""E-Mail-Outbox: Zustellversuche, Fehlertext, BCC-Kennzeichen

Der Versand verschluckte SMTP-Fehler (except Exception: delivery_status =
"failed") ohne Logging und ohne Wiederholung. Der Endpunkt meldete 201, die
Mail war weg. In einer Kanzlei ist das ein Fristproblem.

Revision ID: 0016
Revises: 0015
"""
import sqlalchemy as sa
from alembic import op

revision = "0016"
down_revision = "0015"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "email_messages",
        sa.Column("delivery_attempts", sa.Integer(), nullable=False, server_default="0"),
    )
    op.add_column("email_messages", sa.Column("delivery_error", sa.Text(), nullable=True))
    op.add_column(
        "email_messages",
        sa.Column("use_bcc", sa.Boolean(), nullable=False, server_default=sa.false()),
    )
    # Index fuer den Outbox-Scan des Retry-Tasks
    op.create_index(
        "ix_email_messages_outbox",
        "email_messages",
        ["delivery_status", "delivery_attempts"],
        postgresql_where=sa.text("direction = 'outbound'"),
    )
    # Bestehende ausgehende Nachrichten ohne Status gelten als zugestellt --
    # sie stammen aus der Zeit vor der Outbox und wurden bereits versendet.
    op.execute(
        "UPDATE email_messages SET delivery_status = 'sent' "
        "WHERE direction = 'outbound' AND delivery_status IS NULL AND sent_at IS NOT NULL"
    )


def downgrade() -> None:
    op.drop_index("ix_email_messages_outbox", table_name="email_messages")
    op.drop_column("email_messages", "use_bcc")
    op.drop_column("email_messages", "delivery_error")
    op.drop_column("email_messages", "delivery_attempts")
