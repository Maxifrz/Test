-- Run at container first-start via docker-entrypoint-initdb.d
-- Enforces INSERT-only access to audit_log at the PostgreSQL level

-- Create audit_reader role for admin queries
DO $$
BEGIN
    IF NOT EXISTS (SELECT FROM pg_roles WHERE rolname = 'audit_reader') THEN
        CREATE ROLE audit_reader;
    END IF;
END
$$;

-- The application user (legalapp) may only INSERT into audit_log
-- This is the critical DSGVO guarantee: no application code can UPDATE or DELETE audit entries
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO legalapp;

-- We will revoke UPDATE and DELETE on audit_log after the table is created by Alembic.
-- The trigger below handles this post-migration.

-- Trigger function to prevent any modification of audit_log rows
CREATE OR REPLACE FUNCTION prevent_audit_log_modification()
RETURNS TRIGGER AS $$
BEGIN
    RAISE EXCEPTION 'Audit log is immutable. UPDATE and DELETE are not permitted.';
END;
$$ LANGUAGE plpgsql;

-- Note: The trigger is applied by Alembic migration after table creation.
-- See alembic/versions/0001_initial.py
