"""
Löschregistry für Art. 17 DSGVO.

Warum eine Registry statt Ad-hoc-Code: die bisherige Umsetzung anonymisierte
ausschließlich die Tabelle `clients`. E-Mails (samt Anhängen auf der Platte),
Transkripte (die wörtliche Rede der Person!), Forderungsanmeldungen,
Kontaktanfragen, Kalendereinträge und KI-Anfragen blieben unberührt — das
ausgestellte Löschzertifikat behauptete trotzdem, der Personenbezug sei
entfallen. Jede neue Tabelle vergrößerte diese Lücke stillschweigend.

Jetzt ist die Löschung deklarativ beschrieben und wird über einen
Abdeckungstest (tests/test_dsgvo_erasure.py) gegen die Modelle geprüft: eine
neue Tabelle mit Personenbezug MUSS hier eingetragen werden, sonst schlägt
der Test fehl.

Strategien:
- ANONYMIZE: Zeile bleibt, PII-Spalten werden durch den Marker ersetzt.
  Für alles, was aus Integritäts- oder Buchführungsgründen bestehen bleiben
  muss (Forderungstabelle, Verteilungsquoten, Fristen).
- PURGE_FILES: verschlüsselte Nutzdateien werden gelöscht (Anhänge, Audio).
- KEEP: bewusst unangetastet, mit Begründung (audit_log — die Zeilen sind
  per Trigger unveränderlich und ihre Beweiskraft ist der Zweck).
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class Strategy(StrEnum):
    ANONYMIZE = "anonymize"
    PURGE_FILES = "purge_files"
    KEEP = "keep"


class Scope(StrEnum):
    """Wie die Zeilen einer Tabelle dem Mandanten zugeordnet werden."""

    CLIENT = "client"                # Spalte client_id
    MATTER = "matter"                # Spalte matter_id, über die Akten des Mandanten
    MATTER_CHILD = "matter_child"    # über eine Elterntabelle, die matter_id trägt
    EMAIL_MATCH = "email_match"      # nur über die E-Mail-Adresse verknüpfbar
    USER = "user"                    # Anfragen eines Nutzers (KI-Historie)


@dataclass(frozen=True)
class ErasureRule:
    table: str
    label: str
    strategy: Strategy
    scope: Scope | None = None
    # Spalten, die durch den Marker ersetzt werden
    pii_columns: tuple[str, ...] = ()
    # Spalten, die auf NULL gesetzt werden (z. B. JSONB-Adresslisten)
    null_columns: tuple[str, ...] = ()
    # Spalte mit einem Dateipfad, dessen Inhalt gelöscht wird
    path_column: str | None = None
    # SQL-Fragment, das die betroffenen Zeilen auswählt. :client_id und
    # :matter_ids stehen als Parameter zur Verfügung.
    where_sql: str = ""
    reason: str = ""


ERASURE_MARKER = "[GELÖSCHT gem. Art. 17 DSGVO]"

# Reihenfolge = Ausführungsreihenfolge. Kindtabellen vor Elterntabellen, damit
# ein Abbruch mittendrin keine verwaisten Verweise hinterlässt.
ERASURE_REGISTRY: tuple[ErasureRule, ...] = (
    ErasureRule(
        table="transcript_segments",
        label="Transkript-Segmente (wörtliche Rede)",
        strategy=Strategy.ANONYMIZE,
        scope=Scope.MATTER_CHILD,
        pii_columns=("text", "speaker_label"),
        where_sql=(
            "transcription_id IN (SELECT id FROM transcriptions "
            "WHERE matter_id = ANY(:matter_ids))"
        ),
    ),
    ErasureRule(
        table="transcript_edits",
        label="Transkript-Änderungshistorie",
        strategy=Strategy.ANONYMIZE,
        scope=Scope.MATTER_CHILD,
        pii_columns=("old_value", "new_value"),
        where_sql=(
            "transcription_id IN (SELECT id FROM transcriptions "
            "WHERE matter_id = ANY(:matter_ids))"
        ),
    ),
    ErasureRule(
        table="transcriptions",
        label="Transkripte (Volltext, Originalaudio)",
        strategy=Strategy.ANONYMIZE,
        scope=Scope.MATTER,
        pii_columns=("title", "full_text", "original_filename"),
        path_column="storage_dir",
        where_sql="matter_id = ANY(:matter_ids)",
    ),
    ErasureRule(
        table="documents",
        label="Dokumente der Handakte (Scans, Schriftsätze) inkl. Volltext",
        strategy=Strategy.ANONYMIZE,
        scope=Scope.MATTER,
        pii_columns=("title", "original_filename", "extracted_text"),
        path_column="storage_path",
        where_sql="matter_id = ANY(:matter_ids)",
    ),
    ErasureRule(
        table="email_attachments",
        label="E-Mail-Anhänge (verschlüsselte Dateien)",
        strategy=Strategy.PURGE_FILES,
        scope=Scope.MATTER_CHILD,
        pii_columns=("filename",),
        path_column="storage_path",
        where_sql=(
            "email_id IN (SELECT id FROM email_messages "
            "WHERE client_id = :client_id OR matter_id = ANY(:matter_ids))"
        ),
    ),
    ErasureRule(
        table="email_messages",
        label="E-Mails (Absender, Betreff, Inhalt)",
        strategy=Strategy.ANONYMIZE,
        scope=Scope.CLIENT,
        pii_columns=("from_address", "subject", "body_text", "body_html"),
        null_columns=("to_addresses", "cc_addresses", "references"),
        where_sql="client_id = :client_id OR matter_id = ANY(:matter_ids)",
    ),
    ErasureRule(
        table="insolvency_claims",
        label="Forderungsanmeldungen (Gläubigerdaten)",
        strategy=Strategy.ANONYMIZE,
        scope=Scope.MATTER,
        pii_columns=(
            "creditor_name", "creditor_email", "creditor_address",
            "creditor_reference", "claim_reason", "dispute_reason",
        ),
        where_sql="matter_id = ANY(:matter_ids)",
        reason="Beträge, Rang und Status bleiben — sie tragen die Verteilungsrechnung.",
    ),
    ErasureRule(
        table="calendar_attendees",
        label="Externe Termin-Teilnehmer",
        strategy=Strategy.ANONYMIZE,
        scope=Scope.MATTER_CHILD,
        pii_columns=("external_name", "external_email"),
        where_sql=(
            "event_id IN (SELECT id FROM calendar_events WHERE matter_id = ANY(:matter_ids))"
        ),
    ),
    ErasureRule(
        table="calendar_events",
        label="Kalendereinträge",
        strategy=Strategy.ANONYMIZE,
        scope=Scope.MATTER,
        pii_columns=("title", "description", "location"),
        where_sql="matter_id = ANY(:matter_ids)",
    ),
    ErasureRule(
        table="tickets",
        label="Aufgaben und Fristen",
        strategy=Strategy.ANONYMIZE,
        scope=Scope.MATTER,
        pii_columns=("title", "description"),
        where_sql="matter_id = ANY(:matter_ids)",
        reason="Fristdaten bleiben — sie belegen die Einhaltung der Berufspflichten.",
    ),
    ErasureRule(
        table="ki_queries",
        label="KI-Recherchen (Frage und Entwurf)",
        strategy=Strategy.ANONYMIZE,
        scope=Scope.MATTER,
        pii_columns=("question", "answer", "feedback_note"),
        where_sql="matter_id = ANY(:matter_ids)",
    ),
    ErasureRule(
        table="legal_chunks",
        label="Textabschnitte interner KI-Dokumente",
        strategy=Strategy.ANONYMIZE,
        scope=Scope.MATTER_CHILD,
        pii_columns=("text", "heading"),
        where_sql=(
            "document_id IN (SELECT id FROM legal_documents WHERE matter_id = ANY(:matter_ids))"
        ),
    ),
    ErasureRule(
        table="legal_documents",
        label="Interne Dokumente im KI-Korpus",
        strategy=Strategy.ANONYMIZE,
        scope=Scope.MATTER,
        pii_columns=("title",),
        where_sql="matter_id = ANY(:matter_ids)",
    ),
    ErasureRule(
        table="matters",
        label="Akten (Rubrum, Gegenseite, Beschreibung)",
        strategy=Strategy.ANONYMIZE,
        scope=Scope.CLIENT,
        pii_columns=(
            "title", "description", "opposing_party", "opposing_counsel", "court_file_ref",
        ),
        where_sql="client_id = :client_id",
        reason="matter_number, Fristen und Status bleiben (Aktenführung, § 50 BRAO).",
    ),
    ErasureRule(
        table="contact_requests",
        label="Kontaktanfragen von der Website",
        strategy=Strategy.ANONYMIZE,
        scope=Scope.EMAIL_MATCH,
        pii_columns=("name", "email", "phone", "message", "standort", "rolle"),
        where_sql="lower(email) = :client_email",
        reason="Nur über die Adresse verknüpfbar — es gibt keine Mandanten-Beziehung.",
    ),
    ErasureRule(
        table="clients",
        label="Mandanten-Stammdaten",
        strategy=Strategy.ANONYMIZE,
        scope=Scope.CLIENT,
        pii_columns=(
            "first_name", "last_name", "company_name", "email", "phone",
            "address_line1", "address_line2", "postal_code", "city",
            "date_of_birth", "tax_id", "notes",
        ),
        null_columns=("email_index",),
        where_sql="id = :client_id",
    ),
    ErasureRule(
        table="audit_log",
        label="Audit-Log",
        strategy=Strategy.KEEP,
        reason=(
            "Bewusst unangetastet: Die Tabelle ist per DB-Trigger unveränderlich, "
            "ihre Beweiskraft ist der Zweck (Art. 5 Abs. 2 DSGVO, Rechenschaftspflicht). "
            "Sie enthält Aktions-Metadaten, keine Inhaltsdaten. Der Personenbezug "
            "der referenzierten Stammdaten entfällt durch deren Anonymisierung."
        ),
    ),
)


# --- Planung (rein, ohne DB — deshalb testbar) ---

@dataclass
class ErasureStep:
    table: str
    label: str
    strategy: Strategy
    sql: str | None
    path_column: str | None
    path_where: str | None
    reason: str = ""


def build_erasure_plan(
    registry: tuple[ErasureRule, ...] = ERASURE_REGISTRY,
    marker: str = ERASURE_MARKER,
) -> list[ErasureStep]:
    """
    Übersetzt die Registry in konkrete UPDATE-Anweisungen. Erzeugt reines SQL
    mit benannten Parametern (:client_id, :matter_ids, :client_email, :marker)
    — die Werte kommen erst bei der Ausführung dazu.
    """
    steps: list[ErasureStep] = []
    for rule in registry:
        if rule.strategy is Strategy.KEEP:
            steps.append(
                ErasureStep(rule.table, rule.label, rule.strategy, None, None, None, rule.reason)
            )
            continue

        assignments = [f"{col} = :marker" for col in rule.pii_columns]
        assignments += [f"{col} = NULL" for col in rule.null_columns]

        sql = None
        if assignments:
            # Tabellen- und Spaltennamen stammen ausschliesslich aus der
            # Registry oben (Konstanten im Quelltext), nie aus Eingaben; die
            # WERTE gehen als benannte Parameter rein (:marker, :client_id, ...).
            sql = f"UPDATE {rule.table} SET {', '.join(assignments)} WHERE {rule.where_sql}"  # noqa: S608

        steps.append(
            ErasureStep(
                table=rule.table,
                label=rule.label,
                strategy=rule.strategy,
                sql=sql,
                path_column=rule.path_column,
                path_where=rule.where_sql if rule.path_column else None,
                reason=rule.reason,
            )
        )
    return steps


def covered_tables(registry: tuple[ErasureRule, ...] = ERASURE_REGISTRY) -> set[str]:
    return {r.table for r in registry}
