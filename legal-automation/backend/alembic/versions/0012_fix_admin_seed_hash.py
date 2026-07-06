"""Initial-Admin: Passwort-Hash reparieren

Der in 0001 geseedete bcrypt-Hash gehörte NICHT zum dokumentierten
Initial-Passwort "Admin1234!" (Copy-Paste-Beispielhash) — der Erst-Login
einer frischen Installation wäre unmöglich gewesen. 0001 ist für neue
Installationen korrigiert; diese Migration repariert bereits migrierte
Datenbanken. Sie fasst den Admin NUR an, solange er im Auslieferzustand
ist (must_change_password = true, d. h. Passwort wurde nie geändert).

Revision ID: 0012
Revises: 0011
Create Date: 2026-07-06
"""
from alembic import op

revision = "0012"
down_revision = "0011"
branch_labels = None
depends_on = None

# bcrypt-Hash von "Admin1234!" (verifiziert via app.core.security.verify_password)
_ADMIN_HASH = "$2b$12$qfbARWc79RSWBGmP5vocPew564Xb7D/532ydMQEO.wB5rZe3.QQbm"


def upgrade() -> None:
    op.execute(
        f"""
        UPDATE users
        SET password_hash = '{_ADMIN_HASH}'
        WHERE email = 'admin@kanzlei.local'
          AND must_change_password = true
        """
    )


def downgrade() -> None:
    # Kein sinnvolles Downgrade — der alte Hash war zu keinem Passwort gültig.
    pass
