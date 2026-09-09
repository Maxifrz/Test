"""
Tests der Loeschregistry (Art. 17 DSGVO).

Der wichtigste Test hier ist test_registry_covers_all_pii_tables: er haelt die
Registry gegen die tatsaechlichen Modelle. Genau das war der Ursprungsfehler --
die Loeschung fasste nur `clients` an, waehrend E-Mails, Transkripte,
Forderungsanmeldungen und Kontaktanfragen unberuehrt blieben und das Zertifikat
trotzdem behauptete, der Personenbezug sei entfallen.
"""
import pytest

from app.services.dsgvo_erasure import (
    ERASURE_MARKER,
    ERASURE_REGISTRY,
    ErasureRule,
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
    for field in ("first_name", "last_name", "email", "phone", "date_of_birth", "tax_id", "notes"):
        assert field in rule.pii_columns
    # Der Blind-Index muss mit weg, sonst bleibt die Adresse ueber ihn bestaetigbar
    assert "email_index" in rule.null_columns


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

def test_plan_generates_parameterised_sql():
    plan = build_erasure_plan()
    for step in plan:
        if step.sql is None:
            continue
        # Werte duerfen NIE eingesetzt sein -- nur benannte Parameter
        assert ERASURE_MARKER not in step.sql
        assert ":marker" in step.sql or "NULL" in step.sql
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
    allowed = {":marker", ":client_id", ":matter_ids", ":client_email", ":min_len"}
    import re

    for step in build_erasure_plan():
        if not step.sql:
            continue
        for param in re.findall(r":\w+", step.sql):
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
    with pytest.raises(Exception):
        rule.table = "andere_tabelle"


def test_scope_values_are_documented():
    for rule in ERASURE_REGISTRY:
        if rule.scope is not None:
            assert isinstance(rule.scope, Scope)
