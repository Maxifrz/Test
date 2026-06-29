"""Calendar automation: events + attendees

Revision ID: 0004
Revises: 0003
Create Date: 2025-01-01 00:00:00
"""
from alembic import op
import sqlalchemy as sa

revision = "0004"
down_revision = "0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "calendar_events",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("title", sa.String(500), nullable=False),
        sa.Column("description", sa.Text, nullable=True),
        sa.Column("event_type", sa.String(30), nullable=False, server_default="internal_meeting"),
        sa.Column("start_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("end_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("all_day", sa.Boolean, nullable=False, server_default=sa.false()),
        sa.Column("location", sa.String(500), nullable=True),
        sa.Column("travel_buffer_minutes", sa.Integer, nullable=False, server_default="0"),
        sa.Column("matter_id", sa.Integer, sa.ForeignKey("matters.id"), nullable=True),
        sa.Column("organizer_id", sa.Integer, sa.ForeignKey("users.id"), nullable=False),
        sa.Column("created_by_id", sa.Integer, sa.ForeignKey("users.id"), nullable=False),
        sa.Column("ticket_id", sa.Integer, sa.ForeignKey("tickets.id"), nullable=True),
        sa.Column("status", sa.String(20), nullable=False, server_default="confirmed"),
        sa.Column("recurrence_rule", sa.String(255), nullable=True),
        sa.Column("source", sa.String(20), nullable=False, server_default="manual"),
        sa.Column("external_uid", sa.String(998), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("deleted_by_id", sa.Integer, nullable=True),
    )
    op.create_index("ix_calendar_events_start_at", "calendar_events", ["start_at"])
    op.create_index("ix_calendar_events_organizer_id", "calendar_events", ["organizer_id"])
    op.create_index("ix_calendar_events_matter_id", "calendar_events", ["matter_id"])
    op.create_index("ix_calendar_events_external_uid", "calendar_events", ["external_uid"])

    op.create_table(
        "calendar_attendees",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("event_id", sa.Integer, sa.ForeignKey("calendar_events.id"), nullable=False),
        sa.Column("user_id", sa.Integer, sa.ForeignKey("users.id"), nullable=True),
        sa.Column("external_name", sa.String(255), nullable=True),
        sa.Column("external_email", sa.String(320), nullable=True),
        sa.Column("response_status", sa.String(20), nullable=False, server_default="needs_action"),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_calendar_attendees_event_id", "calendar_attendees", ["event_id"])
    op.create_index("ix_calendar_attendees_user_id", "calendar_attendees", ["user_id"])


def downgrade() -> None:
    op.drop_table("calendar_attendees")
    op.drop_table("calendar_events")
