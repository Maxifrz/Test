"""E-Mail-Normalisierung + case-insensitiver Unique-Index auf users.email

Bisher wurden Adressen so gespeichert, wie sie eingegeben wurden. Damit
konnten "Max@Kanzlei.de" und "max@kanzlei.de" als zwei Konten nebeneinander
existieren, und der Login-Lockout (Redis-Key aus der Roh-Eingabe) ließ sich
durch Groß-/Kleinschreibung umgehen.

Revision ID: 0013
Revises: 0012
"""
import sqlalchemy as sa
from alembic import op

revision = "0013"
down_revision = "0012"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # 1) Bestehende Adressen normalisieren. Kollidierende Duplikate (gleiche
    #    Adresse in unterschiedlicher Schreibweise) werden NICHT stillschweigend
    #    zusammengeführt — das wäre ein Datenverlust. Stattdessen bleibt der
    #    älteste Datensatz aktiv, die jüngeren werden deaktiviert und als
    #    gelöscht markiert, damit der Unique-Index greifen kann.
    op.execute(
        """
        WITH ranked AS (
            SELECT id,
                   lower(btrim(email)) AS norm,
                   row_number() OVER (
                       PARTITION BY lower(btrim(email)) ORDER BY id ASC
                   ) AS rn
            FROM users
            WHERE deleted_at IS NULL
        )
        UPDATE users u
        SET is_active = false,
            deleted_at = now(),
            email = ranked.norm || '.dup' || u.id
        FROM ranked
        WHERE u.id = ranked.id AND ranked.rn > 1
        """
    )
    op.execute("UPDATE users SET email = lower(btrim(email))")

    # 2) Case-insensitiver Unique-Index. Der bestehende Unique-Constraint auf
    #    users.email bleibt bestehen (er deckt den Exaktfall ab); dieser Index
    #    verhindert zusätzlich Varianten in anderer Schreibweise, falls doch
    #    einmal am Service vorbei geschrieben wird.
    op.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_users_email_lower "
        "ON users (lower(email)) WHERE deleted_at IS NULL"
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS uq_users_email_lower")
