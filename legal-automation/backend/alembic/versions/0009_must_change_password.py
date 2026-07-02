"""users.must_change_password + Pflicht-Wechsel für Initial-Admin

Revision ID: 0009
Revises: 0008
Create Date: 2025-01-01 00:00:00
"""
from alembic import op
import sqlalchemy as sa

revision = "0009"
down_revision = "0008"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "users",
        sa.Column("must_change_password", sa.Boolean, nullable=False, server_default=sa.false()),
    )
    # Der in 0001 geseedete Initial-Admin (Admin1234!) MUSS das Passwort ändern
    op.execute(
        "UPDATE users SET must_change_password = true WHERE email = 'admin@kanzlei.local'"
    )


def downgrade() -> None:
    op.drop_column("users", "must_change_password")
