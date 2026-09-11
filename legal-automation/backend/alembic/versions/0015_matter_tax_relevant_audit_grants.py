"""matters.tax_relevant + wirksame Audit-Log-Rechte

Zwei Dinge, die zusammengehoeren, weil beide die Aufbewahrung betreffen:

1. matters.tax_relevant -- steuerlich relevante Akten unterliegen § 147 Abs. 3
   AO (10 Jahre) statt der 6 Jahre aus § 50 BRAO. Ohne dieses Flag hat der
   Loeschworkflow solche Akten vier Jahre zu frueh freigegeben.

2. REVOKE UPDATE, DELETE ON audit_log fuer die Laufzeit-Rolle. init_db.sql
   kuendigte das an, setzte es aber nie um; ausserdem lief die Anwendung als
   Eigentuemer, an dem Grants wirkungslos sind. Jetzt wird es tatsaechlich
   entzogen -- sofern eine Laufzeit-Rolle existiert (siehe init_db.sql).

Revision ID: 0015
Revises: 0014
"""
import sqlalchemy as sa
from alembic import op

revision = "0015"
down_revision = "0014"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "matters",
        sa.Column("tax_relevant", sa.Boolean(), nullable=False, server_default=sa.false()),
    )
    # Insolvenz- und Sanierungsakten sind praktisch immer steuerrelevant
    # (Masseverwaltung, Steuererklaerungen des Schuldners). Bestand entsprechend
    # vorbelegen -- die kuerzere Frist waere hier der gefaehrlichere Default.
    op.execute(
        "UPDATE matters SET tax_relevant = true WHERE matter_type IN ('insolvenz', 'sanierung', 'tax')"
    )
    op.create_index(
        "ix_matters_retention_scan",
        "matters",
        ["status", "closed_at"],
        postgresql_where=sa.text("deleted_at IS NULL"),
    )

    # Audit-Log-Rechte. Laeuft nur, wenn eine Laufzeit-Rolle existiert; sonst
    # ist die Anwendung der Eigentuemer und der Trigger bleibt der einzige
    # Schutz (in init_db.sql dokumentiert).
    op.execute(
        """
        DO $$
        DECLARE
            v_user text;
        BEGIN
            SELECT rolname INTO v_user FROM pg_roles
            WHERE rolname IN ('legalapp_rt', current_setting('custom.runtime_user', true))
              AND rolcanlogin
            LIMIT 1;

            IF v_user IS NULL THEN
                RAISE NOTICE 'Keine Laufzeit-Rolle gefunden -- audit_log-Rechte unveraendert.';
                RETURN;
            END IF;

            EXECUTE format('REVOKE UPDATE, DELETE, TRUNCATE ON audit_log FROM %I', v_user);
            EXECUTE format('GRANT SELECT, INSERT ON audit_log TO %I', v_user);
            EXECUTE format(
                'ALTER DEFAULT PRIVILEGES IN SCHEMA public REVOKE UPDATE, DELETE ON TABLES FROM %I',
                v_user
            );
        END
        $$;
        """
    )
    op.execute(
        """
        DO $$
        BEGIN
            IF EXISTS (SELECT FROM pg_roles WHERE rolname = 'audit_reader') THEN
                GRANT SELECT ON audit_log TO audit_reader;
            END IF;
        END
        $$;
        """
    )


def downgrade() -> None:
    op.drop_index("ix_matters_retention_scan", table_name="matters")
    op.drop_column("matters", "tax_relevant")
    op.execute(
        """
        DO $$
        DECLARE
            v_user text;
        BEGIN
            SELECT rolname INTO v_user FROM pg_roles
            WHERE rolname IN ('legalapp_rt', current_setting('custom.runtime_user', true))
              AND rolcanlogin
            LIMIT 1;
            IF v_user IS NOT NULL THEN
                EXECUTE format('GRANT UPDATE, DELETE ON audit_log TO %I', v_user);
            END IF;
        END
        $$;
        """
    )
