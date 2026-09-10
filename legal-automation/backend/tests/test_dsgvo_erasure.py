"""
Tests der Loeschregistry (Art. 17 DSGVO).

Der wichtigste Test hier ist test_registry_covers_all_pii_tables: er haelt die
Registry gegen die tatsaechlichen Modelle. Genau das war der Ursprungsfehler --
die Loeschung fasste nur `clients` an, waehrend E-Mails, Transkripte,
Forderungsanmeldungen und Kontaktanfragen unberuehrt blieben und das Zertifikat
trotzdem behauptete, der Personenbezug sei entfallen.
"""
import dataclasses

import pytest

from app.services.dsgvo_erasure import (
    ERASURE_MARKER,
    ERASURE_REGISTRY,
    Scope,
    Strategy,
    build_erasure_plan,
    covered_tables,
)

# Tabellen, die personenbezogene Inhaltsdaten eines Mandanten tragen koennen.
# Wer eine neue solche Tabelle anlegt, muss sie hier UND in der Registry
# eintragen -- sonst schlaegt dieser Test fehl.
PII_BEARING_TABLES = {
    "clients",
    "matters",
    "email_messages",
    "email_attachments",
    "documents",
    "transcriptions",
    "transcript_segments",
    "transcript_edits",
    "insolvency_claims",
    "calendar_events",
    "calendar_attendees",
    "tickets",
    "contact_requests",
    "ki_queries",
    "legal_documents",
    "legal_chunks",
    "audit_log",
}


def test_registry_covers_all_pii_tables():
    missing = PII_BEARING_TABLES - covered_tables()
    assert not missing, (
        f"Tabellen mit Personenbezug ohne Loeschregel: {sorted(missing)}. "
        "Jede neue Tabelle mit Mandantendaten muss in ERASURE_REGISTRY eingetragen "
        "werden, sonst behauptet das Loeschzertifikat mehr, als geloescht wurde."
    )


def test_registry_has_no_unknown_tables():
    unknown = covered_tables() - PII_BEARING_TABLES
    assert not unknown, f"Registry kennt Tabellen, die der Test nicht kennt: {sorted(unknown)}"


def test_client_master_data_is_erased():
    rule = next(r for r in ERASURE_REGISTRY if r.table == "clients")
    alle = set(rule.pii_columns) | set(rule.encrypted_pii_columns)
    for field in ("first_name", "last_name", "email", "phone", "date_of_birth", "tax_id", "notes"):
        assert field in alle
    # Der Blind-Index muss mit weg, sonst bleibt die Adresse ueber ihn bestaetigbar
    assert "email_index" in rule.null_columns


def test_encrypted_columns_get_the_encrypted_marker():
    """
    Verschluesselte Spalten (EncryptedText) duerfen NICHT den Klartext-Marker
    bekommen: das naechste Lesen scheitert dann an der Entschluesselung und der
    anonymisierte Datensatz ist dauerhaft unlesbar. Gefunden bei der Abnahme
    gegen ein echtes Postgres, wo danach das DSGVO-Dashboard in einen Fehler lief.
    """
    rule = next(r for r in ERASURE_REGISTRY if r.table == "clients")
    # Die Fernet-verschluesselten Felder des Client-Modells
    for field in ("email", "phone", "address_line1", "postal_code", "date_of_birth",
                  "tax_id", "notes"):
        assert field in rule.encrypted_pii_columns, f"{field} muss verschluesselt gesetzt werden"
        assert field not in rule.pii_columns
    # Klartext-Spalten bleiben Klartext
    for field in ("first_name", "last_name", "city"):
        assert field in rule.pii_columns

    sql = next(s.sql for s in build_erasure_plan() if s.table == "clients")
    assert '"email" = CAST(:marker_enc AS text)' in sql
    assert '"last_name" = CAST(:marker AS text)' in sql


def test_encrypted_and_plaintext_columns_never_overlap():
    for rule in ERASURE_REGISTRY:
        overlap = set(rule.pii_columns) & set(rule.encrypted_pii_columns)
        assert not overlap, f"{rule.table}: {overlap} doppelt zugeordnet"


def test_transcript_content_is_erased():
    # Die woertliche Rede der Person ist der sensibelste Bestand ueberhaupt
    rule = next(r for r in ERASURE_REGISTRY if r.table == "transcript_segments")
    assert "text" in rule.pii_columns
    assert rule.strategy is Strategy.ANONYMIZE


def test_email_attachments_are_purged_from_disk():
    rule = next(r for r in ERASURE_REGISTRY if r.table == "email_attachments")
    assert rule.strategy is Strategy.PURGE_FILES
    assert rule.path_column == "storage_path"


def test_audit_log_is_explicitly_kept_with_reason():
    rule = next(r for r in ERASURE_REGISTRY if r.table == "audit_log")
    assert rule.strategy is Strategy.KEEP
    assert rule.reason  # Eine KEEP-Regel ohne Begruendung waere nicht pruefbar
    assert "Art. 5" in rule.reason


def test_every_keep_rule_has_a_reason():
    for rule in ERASURE_REGISTRY:
        if rule.strategy is Strategy.KEEP:
            assert rule.reason, f"KEEP ohne Begruendung: {rule.table}"


def test_every_anonymize_rule_has_columns_and_scope():
    for rule in ERASURE_REGISTRY:
        if rule.strategy is Strategy.ANONYMIZE:
            assert rule.pii_columns or rule.null_columns, f"{rule.table}: keine Spalten"
            assert rule.where_sql, f"{rule.table}: kein WHERE"
            assert rule.scope is not None, f"{rule.table}: kein Scope"


# --- Planung ---

def test_marker_parameter_is_explicitly_cast():
    """
    Eine Regel setzt denselben Parameter auf Spalten unterschiedlicher Typen
    (text und varchar). Ohne expliziten Cast leitet asyncpg widerspruechliche
    Typen fuer $1 ab und bricht mit AmbiguousParameterError ab -- gefunden bei
    der Abnahme gegen ein echtes Postgres, wo es JEDE Loeschung zerlegt haette.
    """
    for step in build_erasure_plan():
        if not step.sql:
            continue
        assert ":marker" not in step.sql or "CAST(:marker AS text)" in step.sql, (
            f"{step.table}: :marker ohne expliziten Cast"
        )
        # Kein nackter Parameter mehr uebrig
        assert "= :marker" not in step.sql


def test_not_null_list_columns_are_emptied_not_nulled():
    """
    email_messages.to_addresses ist NOT NULL. Ein NULL verletzt die Constraint
    und bricht die GESAMTE Loeschung ab -- gefunden bei der Abnahme gegen ein
    echtes Postgres. Solche Spalten bekommen ein leeres JSON-Array.
    """
    rule = next(r for r in ERASURE_REGISTRY if r.table == "email_messages")
    assert "to_addresses" in rule.empty_json_columns
    assert "to_addresses" not in rule.null_columns

    sql = next(s.sql for s in build_erasure_plan() if s.table == "email_messages")
    assert """"to_addresses" = '[]'::jsonb""" in sql
    assert '"to_addresses" = NULL' not in sql
    # Die nullable Spalten bleiben NULL
    assert '"cc_addresses" = NULL' in sql


def test_identifiers_are_quoted():
    """
    email_messages.references heisst wie ein reserviertes SQL-Schluesselwort.
    Ohne Anfuehrungszeichen ist das ein Syntaxfehler -- gefunden bei der
    Abnahme gegen ein echtes Postgres, wo es die Loeschung abgebrochen hat.
    """
    sql = next(s.sql for s in build_erasure_plan() if s.table == "email_messages")
    assert '"references" = NULL' in sql
    assert '"from_address" = ' in sql
    assert sql.startswith('UPDATE "email_messages"')


def test_all_generated_sql_quotes_identifiers():
    import re

    for step in build_erasure_plan():
        if not step.sql:
            continue
        # Jede SET-Zuweisung muss einen quotierten Spaltennamen haben
        set_part = step.sql.split(" SET ", 1)[1].split(" WHERE ", 1)[0]
        for assignment in set_part.split(", "):
            assert assignment.strip().startswith('"'), f"{step.table}: {assignment}"
        assert re.match(r'UPDATE "\w+" SET ', step.sql), step.sql


def test_quote_escapes_embedded_quotes():
    from app.services.dsgvo_erasure import _quote

    assert _quote("references") == '"references"'
    assert _quote('od"d') == '"od""d"'


def test_plan_generates_parameterised_sql():
    plan = build_erasure_plan()
    for step in plan:
        if step.sql is None:
            continue
        # Werte duerfen NIE eingesetzt sein -- nur benannte Parameter
        assert ERASURE_MARKER not in step.sql
        assert ":marker" in step.sql or ":marker_enc" in step.sql or "NULL" in step.sql
        assert step.sql.startswith("UPDATE ")


def test_plan_order_puts_children_before_parents():
    tables = [s.table for s in build_erasure_plan()]
    # Kindtabellen zuerst: ein Abbruch mittendrin hinterlaesst keine Zeile,
    # deren Elternzeile schon anonymisiert ist
    assert tables.index("transcript_segments") < tables.index("transcriptions")
    assert tables.index("email_attachments") < tables.index("email_messages")
    assert tables.index("calendar_attendees") < tables.index("calendar_events")
    assert tables.index("legal_chunks") < tables.index("legal_documents")
    # clients zuletzt -- vorher werden client_id/E-Mail noch gebraucht
    assert tables.index("matters") < tables.index("clients")


def test_plan_covers_registry_completely():
    assert len(build_erasure_plan()) == len(ERASURE_REGISTRY)


def test_keep_rules_produce_no_sql():
    for step in build_erasure_plan():
        if step.strategy is Strategy.KEEP:
            assert step.sql is None
            assert step.reason


def test_plan_uses_only_known_parameters():
    allowed = {":marker", ":marker_enc", ":client_id", ":matter_ids", ":client_email", ":min_len"}
    import re

    for step in build_erasure_plan():
        if not step.sql:
            continue
        # (?<!:) schliesst Postgres-Typcasts aus ("'[]'::jsonb" ist kein Parameter)
        for param in re.findall(r"(?<!:):\w+", step.sql):
            assert param in allowed, f"Unbekannter Parameter {param} in {step.table}"


def test_custom_marker_is_honoured():
    plan = build_erasure_plan(marker="X")
    assert plan  # Der Marker geht als Parameter rein, nicht in den SQL-Text
    assert all(":marker" in s.sql for s in plan if s.sql)


def test_path_steps_expose_where_clause():
    for step in build_erasure_plan():
        if step.path_column:
            assert step.path_where, f"{step.table}: Pfadspalte ohne WHERE"


def test_registry_is_immutable():
    # frozen dataclass -- verhindert versehentliches Umkonfigurieren zur Laufzeit
    rule = ERASURE_REGISTRY[0]
    with pytest.raises(dataclasses.FrozenInstanceError):
        rule.table = "andere_tabelle"


def test_scope_values_are_documented():
    for rule in ERASURE_REGISTRY:
        if rule.scope is not None:
            assert isinstance(rule.scope, Scope)
