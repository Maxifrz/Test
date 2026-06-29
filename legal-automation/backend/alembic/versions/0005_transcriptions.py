"""Meeting transcriptions: transcriptions, segments, edits + German FTS

Revision ID: 0005
Revises: 0004
Create Date: 2025-01-01 00:00:00
"""
from alembic import op
import sqlalchemy as sa

revision = "0005"
down_revision = "0004"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "transcriptions",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("title", sa.String(500), nullable=False),
        sa.Column("meeting_type", sa.String(50), nullable=False, server_default="Besprechung"),
        sa.Column("meeting_date", sa.Date, nullable=False),
        sa.Column("matter_id", sa.Integer, sa.ForeignKey("matters.id"), nullable=True),
        sa.Column("created_by_id", sa.Integer, sa.ForeignKey("users.id"), nullable=False),
        sa.Column("status", sa.String(20), nullable=False, server_default="queued"),
        sa.Column("progress_stage", sa.String(50), nullable=True),
        sa.Column("error_message", sa.Text, nullable=True),
        sa.Column("language", sa.String(10), nullable=False, server_default="de"),
        sa.Column("model_used", sa.String(50), nullable=True),
        sa.Column("duration_seconds", sa.Float, nullable=True),
        sa.Column("original_filename", sa.String(500), nullable=True),
        sa.Column("storage_dir", sa.String(1000), nullable=True),
        sa.Column("full_text", sa.Text, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("deleted_by_id", sa.Integer, nullable=True),
    )
    op.create_index("ix_transcriptions_matter_id", "transcriptions", ["matter_id"])
    op.create_index("ix_transcriptions_status", "transcriptions", ["status"])

    # German full-text search: generated tsvector column + GIN index
    op.execute(
        "ALTER TABLE transcriptions ADD COLUMN fts tsvector "
        "GENERATED ALWAYS AS (to_tsvector('german', coalesce(full_text, ''))) STORED"
    )
    op.execute("CREATE INDEX ix_transcriptions_fts ON transcriptions USING GIN (fts)")

    op.create_table(
        "transcript_segments",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("transcription_id", sa.Integer, sa.ForeignKey("transcriptions.id"), nullable=False),
        sa.Column("segment_index", sa.Integer, nullable=False),
        sa.Column("speaker", sa.String(50), nullable=False, server_default="SPEAKER_00"),
        sa.Column("speaker_label", sa.String(255), nullable=True),
        sa.Column("start_seconds", sa.Float, nullable=False),
        sa.Column("end_seconds", sa.Float, nullable=False),
        sa.Column("text", sa.Text, nullable=False),
        sa.Column("confidence", sa.Float, nullable=True),
        sa.Column("edited", sa.Boolean, nullable=False, server_default=sa.false()),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_transcript_segments_transcription_id", "transcript_segments", ["transcription_id"])

    op.create_table(
        "transcript_edits",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("segment_id", sa.Integer, sa.ForeignKey("transcript_segments.id"), nullable=False),
        sa.Column("transcription_id", sa.Integer, sa.ForeignKey("transcriptions.id"), nullable=False),
        sa.Column("editor_id", sa.Integer, sa.ForeignKey("users.id"), nullable=False),
        sa.Column("field", sa.String(50), nullable=False),
        sa.Column("old_value", sa.Text, nullable=True),
        sa.Column("new_value", sa.Text, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_transcript_edits_segment_id", "transcript_edits", ["segment_id"])
    op.create_index("ix_transcript_edits_transcription_id", "transcript_edits", ["transcription_id"])


def downgrade() -> None:
    op.drop_table("transcript_edits")
    op.drop_table("transcript_segments")
    op.execute("DROP INDEX IF EXISTS ix_transcriptions_fts")
    op.drop_table("transcriptions")
