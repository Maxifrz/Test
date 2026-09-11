-- Laeuft einmalig beim ersten Start des Postgres-Containers
-- (docker-entrypoint-initdb.d), also VOR der ersten Alembic-Migration.
--
-- Zweck: einen echten Least-Privilege-Laufzeit-User anlegen, der das
-- Audit-Log nur BESCHREIBEN, aber nicht aendern oder loeschen kann.
--
-- Was hier frueher stand, war wirkungslos:
--   * GRANT ... ON ALL TABLES lief, bevor Alembic irgendeine Tabelle angelegt
--     hatte -- es betraf null Tabellen.
--   * Das im Kommentar angekuendigte REVOKE auf audit_log existierte nirgends.
--   * Die Rechte gingen an POSTGRES_USER, also den Eigentuemer der Datenbank --
--     an dem sind Grants ohnehin bedeutungslos.
--   * Die Rolle audit_reader wurde angelegt und nie verwendet.
--
-- Modell jetzt: zwei Rollen.
--   OWNER   (POSTGRES_USER)      -> Migrationen und DDL. Nutzt DATABASE_URL_SYNC.
--   RUNTIME (APP_RUNTIME_USER)   -> die Anwendung.       Nutzt DATABASE_URL.
--
-- Setzt der Betreiber APP_RUNTIME_PASSWORD nicht, wird die Rolle nicht angelegt
-- und die Anwendung laeuft wie bisher als Eigentuemer -- dann schuetzt das
-- Audit-Log allein der Immutability-Trigger aus Migration 0001. Diese Wahl ist
-- bewusst moeglich, damit ein Update bestehender Installationen nicht bricht.

\set runtime_user  `echo "${APP_RUNTIME_USER:-legalapp_rt}"`
\set runtime_pass  `echo "${APP_RUNTIME_PASSWORD:-}"`

-- Werte in Sitzungs-Settings heben, damit der DO-Block sie lesen kann.
SELECT set_config('custom.runtime_user', :'runtime_user', false);
SELECT set_config('custom.runtime_pass', :'runtime_pass', false);

-- Lesende Rolle fuer Audit-Auswertungen (Revision, Datenschutzbeauftragter).
-- Bekommt in Migration 0015 SELECT auf audit_log.
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'audit_reader') THEN
        CREATE ROLE audit_reader NOLOGIN;
    END IF;
END
$$;

-- Trigger-Funktion fuer die Unveraenderlichkeit des Audit-Logs. Der Trigger
-- selbst wird in Migration 0001 gesetzt (die Tabelle existiert hier noch
-- nicht); die Funktion vorab anzulegen ist idempotent.
CREATE OR REPLACE FUNCTION prevent_audit_log_modification()
RETURNS TRIGGER AS $$
BEGIN
    RAISE EXCEPTION 'Audit log is immutable. UPDATE and DELETE are not permitted.';
END;
$$ LANGUAGE plpgsql;

-- Laufzeit-Rolle anlegen und mit Default-Privilegien versehen.
-- ALTER DEFAULT PRIVILEGES ist der entscheidende Teil: es wirkt auf Tabellen,
-- die der Eigentuemer SPAETER anlegt -- also auf alles, was Alembic erzeugt.
DO $do$
DECLARE
    v_user text := current_setting('custom.runtime_user', true);
    v_pass text := current_setting('custom.runtime_pass', true);
BEGIN
    IF v_pass IS NULL OR v_pass = '' THEN
        RAISE NOTICE
            'APP_RUNTIME_PASSWORD nicht gesetzt -- Laufzeit-Rolle uebersprungen. '
            'Die Anwendung laeuft als Eigentuemer; das Audit-Log schuetzt dann '
            'nur der Trigger aus Migration 0001.';
        RETURN;
    END IF;
    IF v_user IS NULL OR v_user = '' THEN
        v_user := 'legalapp_rt';
    END IF;

    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = v_user) THEN
        EXECUTE format('CREATE ROLE %I LOGIN PASSWORD %L', v_user, v_pass);
    END IF;

    EXECUTE format('GRANT CONNECT ON DATABASE %I TO %I', current_database(), v_user);
    EXECUTE format('GRANT USAGE ON SCHEMA public TO %I', v_user);

    -- Bereits vorhandene Objekte (bei diesem Lauf noch keine, schadet nicht)
    EXECUTE format(
        'GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO %I', v_user);
    EXECUTE format(
        'GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO %I', v_user);

    -- Kuenftige Objekte des Eigentuemers -- hier liegt die eigentliche Wirkung
    EXECUTE format(
        'ALTER DEFAULT PRIVILEGES FOR ROLE %I IN SCHEMA public '
        'GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO %I', current_user, v_user);
    EXECUTE format(
        'ALTER DEFAULT PRIVILEGES FOR ROLE %I IN SCHEMA public '
        'GRANT USAGE, SELECT ON SEQUENCES TO %I', current_user, v_user);

    RAISE NOTICE 'Laufzeit-Rolle % angelegt. DATABASE_URL auf diese Rolle zeigen lassen; '
                 'DATABASE_URL_SYNC (Migrationen) bleibt beim Eigentuemer.', v_user;
END
$do$;
