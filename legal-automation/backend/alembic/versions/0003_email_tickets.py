"""Email automation + ticket/task system

Revision ID: 0003
Revises: 0002
Create Date: 2025-01-01 00:00:00
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # --- email_rules (created before email_messages for FK) ---
    op.create_table(
        "email_rules",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("is_active", sa.Boolean, nullable=False, server_default=sa.true()),
        sa.Column("priority", sa.Integer, nullable=False, server_default="100"),
        sa.Column("conditions", JSONB, nullable=False, server_default="{}"),
        sa.Column("actions", JSONB, nullable=False, server_default="{}"),
        sa.Column("created_by_id", sa.Integer, sa.ForeignKey("users.id"), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_email_rules_priority", "email_rules", ["priority"])

    # --- email_templates ---
    op.create_table(
        "email_templates",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("category", sa.String(100), nullable=True),
        sa.Column("subject_template", sa.Text, nullable=False),
        sa.Column("body_template", sa.Text, nullable=False),
        sa.Column("variables_doc", JSONB, nullable=True, server_default="{}"),
        sa.Column("is_active", sa.Boolean, nullable=False, server_default=sa.true()),
        sa.Column("created_by_id", sa.Integer, sa.ForeignKey("users.id"), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("deleted_by_id", sa.Integer, nullable=True),
    )

    # --- email_messages ---
    op.create_table(
        "email_messages",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("message_id", sa.String(998), unique=True, nullable=False),
        sa.Column("direction", sa.String(10), nullable=False),
        sa.Column("from_address", sa.String(320), nullable=False),
        sa.Column("to_addresses", JSONB, nullable=False, server_default="[]"),
        sa.Column("cc_addresses", JSONB, nullable=True, server_default="[]"),
        sa.Column("subject", sa.String(998), nullable=True),
        sa.Column("body_text", sa.Text, nullable=True),
        sa.Column("body_html", sa.Text, nullable=True),
        sa.Column("in_reply_to", sa.String(998), nullable=True),
        sa.Column("references", sa.Text, nullable=True),
        sa.Column("thread_key", sa.String(998), nullable=True),
        sa.Column("matter_id", sa.Integer, sa.ForeignKey("matters.id"), nullable=True),
        sa.Column("client_id", sa.Integer, sa.ForeignKey("clients.id"), nullable=True),
        sa.Column("matched_rule_id", sa.Integer, sa.ForeignKey("email_rules.id"), nullable=True),
        sa.Column("is_read", sa.Boolean, nullable=False, server_default=sa.false()),
        sa.Column("needs_review", sa.Boolean, nullable=False, server_default=sa.false()),
        sa.Column("is_confidential", sa.Boolean, nullable=False, server_default=sa.false()),
        sa.Column("unknown_sender", sa.Boolean, nullable=False, server_default=sa.false()),
        sa.Column("sent_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("delivery_status", sa.String(30), nullable=True),
        sa.Column("sent_by_id", sa.Integer, sa.ForeignKey("users.id"), nullable=True),
        sa.Column("email_date", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("deleted_by_id", sa.Integer, nullable=True),
    )
    op.create_index("ix_email_messages_message_id", "email_messages", ["message_id"])
    op.create_index("ix_email_messages_from_address", "email_messages", ["from_address"])
    op.create_index("ix_email_messages_matter_id", "email_messages", ["matter_id"])
    op.create_index("ix_email_messages_thread_key", "email_messages", ["thread_key"])
    op.create_index("ix_email_messages_email_date", "email_messages", ["email_date"])
    op.create_index("ix_email_messages_in_reply_to", "email_messages", ["in_reply_to"])

    # --- email_attachments ---
    op.create_table(
        "email_attachments",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("email_id", sa.Integer, sa.ForeignKey("email_messages.id"), nullable=False),
        sa.Column("filename", sa.String(500), nullable=False),
        sa.Column("content_type", sa.String(255), nullable=True),
        sa.Column("size_bytes", sa.Integer, nullable=False, server_default="0"),
        sa.Column("storage_path", sa.String(1000), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_email_attachments_email_id", "email_attachments", ["email_id"])

    # --- sla_policies (before tickets for FK) ---
    op.create_table(
        "sla_policies",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("matter_type", sa.String(50), nullable=True),
        sa.Column("ticket_type", sa.String(30), nullable=True),
        sa.Column("target_hours", sa.Integer, nullable=False),
        sa.Column("is_active", sa.Boolean, nullable=False, server_default=sa.true()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
    )

    # --- tickets ---
    op.create_table(
        "tickets",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("title", sa.String(500), nullable=False),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column("ticket_type", sa.String(30), nullable=False, server_default="task"),
        sa.Column("status", sa.String(30), nullable=False, server_default="open"),
        sa.Column("priority", sa.String(20), nullable=False, server_default="normal"),
        sa.Column("due_date", sa.Date, nullable=True),
        sa.Column("frist_basis", sa.String(100), nullable=True),
        sa.Column("frist_trigger_date", sa.Date, nullable=True),
        sa.Column("frist_calculation_note", sa.Text, nullable=True),
        sa.Column("matter_id", sa.Integer, sa.ForeignKey("matters.id"), nullable=True),
        sa.Column("assignee_id", sa.Integer, sa.ForeignKey("users.id"), nullable=True),
        sa.Column("created_by_id", sa.Integer, sa.ForeignKey("users.id"), nullable=False),
        sa.Column("parent_id", sa.Integer, sa.ForeignKey("tickets.id"), nullable=True),
        sa.Column("sla_policy_id", sa.Integer, sa.ForeignKey("sla_policies.id"), nullable=True),
        sa.Column("sla_due_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("sla_breached", sa.Boolean, nullable=False, server_default=sa.false()),
        sa.Column("recurrence_rule", sa.String(255), nullable=True),
        sa.Column("closed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("deleted_by_id", sa.Integer, nullable=True),
    )
    op.create_index("ix_tickets_status", "tickets", ["status"])
    op.create_index("ix_tickets_due_date", "tickets", ["due_date"])
    op.create_index("ix_tickets_matter_id", "tickets", ["matter_id"])
    op.create_index("ix_tickets_assignee_id", "tickets", ["assignee_id"])
    op.create_index("ix_tickets_parent_id", "tickets", ["parent_id"])

    # --- ticket_comments ---
    op.create_table(
        "ticket_comments",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("ticket_id", sa.Integer, sa.ForeignKey("tickets.id"), nullable=False),
        sa.Column("author_id", sa.Integer, sa.ForeignKey("users.id"), nullable=False),
        sa.Column("body", sa.Text, nullable=False),
        sa.Column("is_internal", sa.Boolean, nullable=False, server_default=sa.true()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_ticket_comments_ticket_id", "ticket_comments", ["ticket_id"])

    # --- ticket_time_entries ---
    op.create_table(
        "ticket_time_entries",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("ticket_id", sa.Integer, sa.ForeignKey("tickets.id"), nullable=False),
        sa.Column("user_id", sa.Integer, sa.ForeignKey("users.id"), nullable=False),
        sa.Column("matter_id", sa.Integer, sa.ForeignKey("matters.id"), nullable=True),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("ended_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("duration_minutes", sa.Integer, nullable=True),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column("billable", sa.Boolean, nullable=False, server_default=sa.true()),
        sa.Column("rate_eur", sa.Numeric(10, 2), nullable=True),
        sa.Column("invoiced", sa.Boolean, nullable=False, server_default=sa.false()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_ticket_time_entries_ticket_id", "ticket_time_entries", ["ticket_id"])
    op.create_index("ix_ticket_time_entries_user_id", "ticket_time_entries", ["user_id"])
    op.create_index("ix_ticket_time_entries_matter_id", "ticket_time_entries", ["matter_id"])


def downgrade() -> None:
    op.drop_table("ticket_time_entries")
    op.drop_table("ticket_comments")
    op.drop_table("tickets")
    op.drop_table("sla_policies")
    op.drop_table("email_attachments")
    op.drop_table("email_messages")
    op.drop_table("email_templates")
    op.drop_table("email_rules")
