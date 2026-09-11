"""Kontaktdaten der Mandanten verschluesseln + Blind-Index

Bisher lagen E-Mail, Telefon und Postanschrift im Klartext, obwohl README und
TOM "Stammdaten (PII feldverschluesselt, Fernet)" behaupteten. Diese Migration
schliesst die Luecke fuer die Kontaktdaten und legt fuer die E-Mail einen
deterministischen Blind-Index an, damit die Zuordnung eingehender Nachrichten
weiter funktioniert.

Namensfelder (first_name, last_name, company_name, city) bleiben bewusst im
Klartext -- Namenssuche und alphabetische Sortierung sind der zentrale
Arbeitsablauf; die Abwaegung ist in docs/dsgvo/TOM.md dokumentiert.

Ausserdem: EncryptedText-Spalten werden von VARCHAR(n) auf TEXT umgestellt.
Fernet vergroessert den Wert um ~33 % + 57 Byte, sodass in notes VARCHAR(4096)
nur rund 3.000 Zeichen Klartext passten -- darueber brach der Commit ab.

Revision ID: 0014
Revises: 0013
"""
import sqlalchemy as sa
from alembic import op

revision = "0014"
down_revision = "0013"
branch_labels = None
depends_on = None

# Spalten, die bereits verschluesselt waren und nur den Typ wechseln
_ALREADY_ENCRYPTED = ["date_of_birth", "tax_id", "notes"]
# Spalten, die neu verschluesselt werden (Bestand muss migriert werden)
_NEWLY_ENCRYPTED = ["email", "phone", "address_line1", "address_line2", "postal_code"]


def _encrypt_existing_rows(conn) -> None:
    """
    Verschluesselt die Bestandswerte in Python. Fernet ist nicht in Postgres
    verfuegbar, also muss das ueber die Anwendung laufen -- mit demselben
    ENCRYPTION_KEYS, mit dem die Anwendung spaeter liest.
    """
    from app.core.encryption import blind_index, encrypt

    rows = conn.execute(
        sa.text(
            "SELECT id, email, phone, address_line1, address_line2, postal_code FROM clients"
        )
    ).fetchall()

    for row in rows:
        values = {"id": row.id}
        for col in _NEWLY_ENCRYPTED:
            raw = getattr(row, col)
            values[col] = encrypt(raw) if raw else None
        values["email_index"] = blind_index(row.email)
        conn.execute(
            sa.text(
                """
                UPDATE clients SET
                    email = :email,
                    email_index = :email_index,
                    phone = :phone,
                    address_line1 = :address_line1,
                    address_line2 = :address_line2,
                    postal_code = :postal_code
                WHERE id = :id
                """
            ),
            values,
        )


def upgrade() -> None:
    conn = op.get_bind()

    # 1) Blind-Index-Spalte anlegen (noch leer)
    op.add_column("clients", sa.Column("email_index", sa.String(64), nullable=True))

    # 2) Typen auf TEXT weiten -- MUSS vor dem Verschluesseln passieren, sonst
    #    sprengt der laengere Ciphertext die alten VARCHAR-Grenzen.
    for col in _ALREADY_ENCRYPTED + _NEWLY_ENCRYPTED:
        op.alter_column("clients", col, type_=sa.Text(), existing_nullable=True)

    # 3) Der alte Klartext-Index auf email ist wertlos (und irrefuehrend),
    #    sobald die Spalte verschluesselt ist.
    op.execute("DROP INDEX IF EXISTS ix_clients_email")

    # 4) Bestandsdaten verschluesseln + Index fuellen
    _encrypt_existing_rows(conn)

    # 5) Index fuer die exakte Suche
    op.create_index("ix_clients_email_index", "clients", ["email_index"])


def downgrade() -> None:
    """
    Entschluesselt die Kontaktdaten zurueck in den Klartext. Die
    Laengenbegrenzungen werden NICHT wiederhergestellt: ein Wert, der
    zwischenzeitlich laenger als das alte VARCHAR-Limit geworden ist, ginge
    dabei verloren.
    """
    from app.core.encryption import decrypt

    conn = op.get_bind()
    op.drop_index("ix_clients_email_index", table_name="clients")

    rows = conn.execute(
        sa.text(
            "SELECT id, email, phone, address_line1, address_line2, postal_code FROM clients"
        )
    ).fetchall()
    for row in rows:
        values = {"id": row.id}
        for col in _NEWLY_ENCRYPTED:
            raw = getattr(row, col)
            values[col] = decrypt(raw) if raw else None
        conn.execute(
            sa.text(
                """
                UPDATE clients SET
                    email = :email, phone = :phone, address_line1 = :address_line1,
                    address_line2 = :address_line2, postal_code = :postal_code
                WHERE id = :id
                """
            ),
            values,
        )

    op.drop_column("clients", "email_index")
    op.create_index("ix_clients_email", "clients", ["email"])
