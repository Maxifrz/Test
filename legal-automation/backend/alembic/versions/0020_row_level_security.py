"""Row-Level-Security als zweite Verteidigungslinie fuer die Aktentrennung

Die Aktentrennung wird bisher ausschliesslich anwendungsseitig durchgesetzt
(ensure_matter_access + Listen-Filter, 45 Aufrufstellen). Das ist sauber
umgesetzt -- aber genau EIN vergessener Aufruf in einem neuen Endpunkt reicht
fuer ein Berufsrechtsproblem. RLS zieht die Regel in die Datenbank: selbst eine
fehlerhafte Query kann dann keine fremde Akte liefern.

Die Policies werden hier ANGELEGT, RLS aber NICHT aktiviert. Gruende:

1. RLS wirkt nur gegen einen Nicht-Eigentuemer. Laeuft die Anwendung als
   POSTGRES_USER (Standard), umgeht sie die Policies ohnehin -- erst der
   Least-Privilege-Laufzeit-User aus init_db.sql macht sie wirksam.
2. Die Anwendung muss app.current_user_id je Transaktion setzen (macht
   app/core/deps.py bereits). Eine Bestandsinstallation, die vor diesem Update
   startet, wuerde sonst schlagartig nichts mehr sehen.

Aktivierung durch den Betreiber (docs/OPERATIONS.md):
    SELECT enable_matter_rls();

Revision ID: 0020
Revises: 0019
"""
from alembic import op

revision = "0020"
down_revision = "0019"
branch_labels = None
depends_on = None

# Tabellen mit direktem Aktenbezug (Spalte matter_id)
MATTER_SCOPED_TABLES = [
    "email_messages",
    "tickets",
    "calendar_events",
    "transcriptions",
    "insolvency_claims",
    "distributions",
    "documents",
    "mass_accounts",
    "ki_queries",
]


def upgrade() -> None:
    # Hilfsfunktion: welche Akten darf der aktuelle Nutzer sehen?
    # SECURITY DEFINER, damit sie matter_access auch dann lesen kann, wenn auf
    # dieser Tabelle selbst eine Policy liegt.
    op.execute(
        """
        CREATE OR REPLACE FUNCTION app_current_user_id() RETURNS integer AS $$
        BEGIN
            -- Kein gesetzter Wert (z. B. Migrationen, Worker) -> NULL.
            -- Die Policies lassen NULL bewusst durch: Hintergrundprozesse
            -- arbeiten aktenuebergreifend und laufen ohnehin nicht ueber die
            -- eingeschraenkte Rolle.
            RETURN nullif(current_setting('app.current_user_id', true), '')::integer;
        EXCEPTION WHEN others THEN
            RETURN NULL;
        END;
        $$ LANGUAGE plpgsql STABLE;
        """
    )
    op.execute(
        """
        CREATE OR REPLACE FUNCTION app_is_admin() RETURNS boolean AS $$
            SELECT EXISTS (
                SELECT 1 FROM users
                WHERE id = app_current_user_id()
                  AND role = 'admin'
                  AND is_active
                  AND deleted_at IS NULL
            );
        $$ LANGUAGE sql STABLE SECURITY DEFINER;
        """
    )
    op.execute(
        """
        CREATE OR REPLACE FUNCTION app_can_access_matter(p_matter_id integer)
        RETURNS boolean AS $$
            SELECT
                p_matter_id IS NULL          -- noch nicht zugeordnet (Review-Queue)
                OR app_current_user_id() IS NULL  -- Systemkontext (Worker, Migration)
                OR app_is_admin()
                OR EXISTS (
                    SELECT 1 FROM matter_access
                    WHERE user_id = app_current_user_id()
                      AND matter_id = p_matter_id
                      AND revoked_at IS NULL
                );
        $$ LANGUAGE sql STABLE SECURITY DEFINER;
        """
    )

    for table in MATTER_SCOPED_TABLES:
        op.execute(
            f"""
            DO $$
            BEGIN
                IF EXISTS (SELECT FROM information_schema.tables
                           WHERE table_name = '{table}' AND table_schema = 'public') THEN
                    EXECUTE 'DROP POLICY IF EXISTS matter_isolation ON {table}';
                    EXECUTE 'CREATE POLICY matter_isolation ON {table}
                             USING (app_can_access_matter(matter_id))
                             WITH CHECK (app_can_access_matter(matter_id))';
                END IF;
            END
            $$;
            """
        )

    # Ein-/Ausschalter fuer den Betreiber. FORCE, damit die Policies auch fuer
    # den Tabelleneigentuemer gelten -- ohne FORCE waere RLS bei einer
    # Installation ohne Laufzeit-Rolle wirkungslos.
    tables_sql = ", ".join(f"'{t}'" for t in MATTER_SCOPED_TABLES)
    op.execute(
        f"""
        CREATE OR REPLACE FUNCTION enable_matter_rls() RETURNS text AS $$
        DECLARE
            t text;
            n integer := 0;
        BEGIN
            FOREACH t IN ARRAY ARRAY[{tables_sql}] LOOP
                IF EXISTS (SELECT FROM information_schema.tables
                           WHERE table_name = t AND table_schema = 'public') THEN
                    EXECUTE format('ALTER TABLE %I ENABLE ROW LEVEL SECURITY', t);
                    EXECUTE format('ALTER TABLE %I FORCE ROW LEVEL SECURITY', t);
                    n := n + 1;
                END IF;
            END LOOP;
            RETURN format('Row-Level-Security auf %s Tabellen aktiviert', n);
        END;
        $$ LANGUAGE plpgsql;
        """
    )
    op.execute(
        f"""
        CREATE OR REPLACE FUNCTION disable_matter_rls() RETURNS text AS $$
        DECLARE
            t text;
            n integer := 0;
        BEGIN
            FOREACH t IN ARRAY ARRAY[{tables_sql}] LOOP
                IF EXISTS (SELECT FROM information_schema.tables
                           WHERE table_name = t AND table_schema = 'public') THEN
                    EXECUTE format('ALTER TABLE %I NO FORCE ROW LEVEL SECURITY', t);
                    EXECUTE format('ALTER TABLE %I DISABLE ROW LEVEL SECURITY', t);
                    n := n + 1;
                END IF;
            END LOOP;
            RETURN format('Row-Level-Security auf %s Tabellen deaktiviert', n);
        END;
        $$ LANGUAGE plpgsql;
        """
    )


def downgrade() -> None:
    op.execute("SELECT disable_matter_rls()")
    for table in MATTER_SCOPED_TABLES:
        op.execute(
            f"""
            DO $$
            BEGIN
                IF EXISTS (SELECT FROM information_schema.tables
                           WHERE table_name = '{table}' AND table_schema = 'public') THEN
                    EXECUTE 'DROP POLICY IF EXISTS matter_isolation ON {table}';
                END IF;
            END
            $$;
            """
        )
    op.execute("DROP FUNCTION IF EXISTS enable_matter_rls()")
    op.execute("DROP FUNCTION IF EXISTS disable_matter_rls()")
    op.execute("DROP FUNCTION IF EXISTS app_can_access_matter(integer)")
    op.execute("DROP FUNCTION IF EXISTS app_is_admin()")
    op.execute("DROP FUNCTION IF EXISTS app_current_user_id()")
