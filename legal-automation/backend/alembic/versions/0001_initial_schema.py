"""Initial schema: users, user_sessions, audit_log

Revision ID: 0001
Revises:
Create Date: 2025-01-01 00:00:00
"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects.postgresql import JSONB

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    # --- users ---
    op.create_table(
        "users",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("email", sa.String(255), unique=True, nullable=False),
        sa.Column("full_name", sa.String(255), nullable=False),
        sa.Column("role", sa.String(50), nullable=False),
        sa.Column("password_hash", sa.String(255), nullable=False),
        sa.Column("totp_secret", sa.String(255), nullable=True),
        sa.Column("totp_enabled", sa.Boolean, default=False, nullable=False),
        sa.Column("is_active", sa.Boolean, default=True, nullable=False),
        sa.Column("failed_login_count", sa.Integer, default=0, nullable=False),
        sa.Column("locked_until", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_login", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_login_ip", sa.String(45), nullable=True),
        sa.Column("phone", sa.String(50), nullable=True),
        sa.Column("signature_html", sa.Text, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("deleted_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("deleted_by_id", sa.Integer, nullable=True),
    )
    op.create_index("ix_users_email", "users", ["email"])

    # --- user_sessions ---
    op.create_table(
        "user_sessions",
        sa.Column("id", sa.Integer, primary_key=True),
        sa.Column("session_id", sa.String(64), unique=True, nullable=False),
        sa.Column("user_id", sa.Integer, nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("last_active", sa.DateTime(timezone=True), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("ip_address", sa.String(45), nullable=False),
        sa.Column("user_agent", sa.Text, nullable=True),
        sa.Column("is_revoked", sa.Boolean, default=False, nullable=False),
    )
    op.create_index("ix_user_sessions_session_id", "user_sessions", ["session_id"])
    op.create_index("ix_user_sessions_user_id", "user_sessions", ["user_id"])

    # --- audit_log ---
    op.create_table(
        "audit_log",
        sa.Column("id", sa.BigInteger, primary_key=True),
        sa.Column("event_time", sa.DateTime(timezone=True), server_default=sa.func.now(), nullable=False),
        sa.Column("user_id", sa.Integer, nullable=True),
        sa.Column("user_email", sa.String(255), nullable=False),
        sa.Column("session_id", sa.String(64), nullable=False),
        sa.Column("ip_address", sa.String(45), nullable=False),
        sa.Column("user_agent", sa.Text, nullable=True),
        sa.Column("action", sa.String(120), nullable=False),
        sa.Column("resource", sa.String(120), nullable=False),
        sa.Column("matter_id", sa.Integer, nullable=True),
        sa.Column("client_id", sa.Integer, nullable=True),
        sa.Column("details", JSONB, nullable=True),
        sa.Column("result", sa.String(20), nullable=False),
    )
    op.create_index("ix_audit_log_event_time", "audit_log", ["event_time"])
    op.create_index("ix_audit_log_user_id", "audit_log", ["user_id"])
    op.create_index("ix_audit_log_matter_id", "audit_log", ["matter_id"])
    op.create_index("ix_audit_log_action", "audit_log", ["action"])

    # Immutability trigger: prevent UPDATE and DELETE on audit_log
    op.execute("""
        CREATE OR REPLACE FUNCTION prevent_audit_log_modification()
        RETURNS TRIGGER AS $$
        BEGIN
            RAISE EXCEPTION 'Audit log is immutable — UPDATE and DELETE are not permitted (DSGVO Art. 5 Abs. 2)';
        END;
        $$ LANGUAGE plpgsql;
    """)
    op.execute("""
        CREATE TRIGGER audit_log_immutable
        BEFORE UPDATE OR DELETE ON audit_log
        FOR EACH ROW EXECUTE FUNCTION prevent_audit_log_modification();
    """)

    # Create initial admin user (password: Admin1234! — MUST be changed on first login)
    op.execute("""
        INSERT INTO users (email, full_name, role, password_hash, is_active,
                           totp_enabled, failed_login_count, created_at)
        VALUES (
            'admin@kanzlei.local',
            'Administrator',
            'admin',
            '$2b$12$qfbARWc79RSWBGmP5vocPew564Xb7D/532ydMQEO.wB5rZe3.QQbm',
            true,
            false,
            0,
            NOW()
        )
        ON CONFLICT (email) DO NOTHING;
    """)


def downgrade() -> None:
    op.execute("DROP TRIGGER IF EXISTS audit_log_immutable ON audit_log;")
    op.execute("DROP FUNCTION IF EXISTS prevent_audit_log_modification;")
    op.drop_table("audit_log")
    op.drop_table("user_sessions")
    op.drop_table("users")
