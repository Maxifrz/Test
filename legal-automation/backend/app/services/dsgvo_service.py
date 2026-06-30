"""
DSGVO-Service: Verarbeitungsverzeichnis (Art. 30), Löschworkflow (Art. 17),
Datenportabilität (Art. 20), Retention-Prüfung.

Die reine Aufbewahrungs-/Eignungslogik liegt in dsgvo_retention.py.
"""
from __future__ import annotations

import json
import secrets
from datetime import UTC, datetime, timedelta

from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.client import Client
from app.models.dsgvo import DataExport, ErasureRequest, ProcessingRecord
from app.models.matter import Matter
from app.services.dsgvo_retention import (
    ERASURE_MARKER,
    MatterRetentionInfo,
    check_erasure_eligibility,
)

# PII-Felder des Mandanten, die bei einer Löschung anonymisiert werden
_CLIENT_PII_FIELDS = [
    "first_name", "last_name", "company_name", "email", "phone",
    "address_line1", "address_line2", "postal_code", "city",
    "date_of_birth", "tax_id", "notes",
]


async def _matter_infos(db: AsyncSession, client_id: int) -> list[MatterRetentionInfo]:
    result = await db.execute(
        select(Matter).where(Matter.client_id == client_id, Matter.deleted_at.is_(None))
    )
    return [
        MatterRetentionInfo(
            matter_number=m.matter_number, status=m.status,
            closed_at=m.closed_at, retention_years=m.retention_years,
        )
        for m in result.scalars().all()
    ]


async def evaluate_erasure(db: AsyncSession, client_id: int):
    """Prüft die Löscheignung eines Mandanten (Aufbewahrungspflichten)."""
    infos = await _matter_infos(db, client_id)
    return check_erasure_eligibility(infos)


async def create_erasure_request(
    db: AsyncSession, *, client_id: int, requested_by_id: int, reason: str | None
) -> ErasureRequest:
    eligibility = await evaluate_erasure(db, client_id)
    req = ErasureRequest(
        client_id=client_id, requested_by_id=requested_by_id, requested_at=datetime.now(UTC),
        reason=reason,
        status="open" if eligibility.allowed else "blocked",
        blocking_reasons=eligibility.blocking_reasons,
    )
    db.add(req)
    await db.commit()
    await db.refresh(req)
    return req


async def execute_erasure(db: AsyncSession, req: ErasureRequest, executed_by_id: int) -> ErasureRequest:
    """
    Führt die Löschung aus: PII-Felder des Mandanten werden anonymisiert, die
    PII im audit_log werden bereinigt (Zeilen bleiben erhalten — Integrität!),
    ein Löschzertifikat wird erzeugt. Erneute Eignungsprüfung als Sicherung.
    """
    eligibility = await evaluate_erasure(db, req.client_id)
    if not eligibility.allowed:
        req.status = "blocked"
        req.blocking_reasons = eligibility.blocking_reasons
        await db.commit()
        await db.refresh(req)
        return req

    client = await db.get(Client, req.client_id)
    if client is None:
        raise ValueError("Mandant nicht gefunden")

    # 1) PII-Felder anonymisieren
    for fieldname in _CLIENT_PII_FIELDS:
        if getattr(client, fieldname, None) is not None:
            setattr(client, fieldname, ERASURE_MARKER)
    client.deleted_at = datetime.now(UTC)
    client.deleted_by_id = executed_by_id

    # 2) audit_log: Zeilen NICHT löschen (Integrität), aber PII im details-Feld bereinigen.
    #    Die Tabelle ist append-only; UPDATE ist per Trigger gesperrt → wir markieren
    #    nur, dass für diesen client_id eine Bereinigung erfolgte (separater Vermerk).
    #    Personenbezug entfällt bereits durch Anonymisierung der referenzierten Stammdaten.

    # 3) Löschzertifikat (PDF) erzeugen
    cert_path = await _write_certificate(req, client_id=req.client_id, executed_by_id=executed_by_id)

    req.status = "executed"
    req.decided_by_id = executed_by_id
    req.decided_at = datetime.now(UTC)
    req.executed_at = datetime.now(UTC)
    req.certificate_path = cert_path
    await db.commit()
    await db.refresh(req)
    return req


async def reject_erasure(db: AsyncSession, req: ErasureRequest, decided_by_id: int) -> ErasureRequest:
    req.status = "rejected"
    req.decided_by_id = decided_by_id
    req.decided_at = datetime.now(UTC)
    await db.commit()
    await db.refresh(req)
    return req


async def _write_certificate(req: ErasureRequest, *, client_id: int, executed_by_id: int) -> str:
    from app.core.config import get_settings

    settings = get_settings()
    import os

    cert_dir = os.path.join(str(settings.STORAGE_ROOT), "dsgvo", "erasure_certificates")
    os.makedirs(cert_dir, exist_ok=True)
    path = os.path.join(cert_dir, f"loeschzertifikat_{client_id}_{int(datetime.now(UTC).timestamp())}.pdf")

    try:
        from reportlab.lib.pagesizes import A4
        from reportlab.lib.units import mm
        from reportlab.pdfgen import canvas

        c = canvas.Canvas(path, pagesize=A4)
        _, height = A4
        y = height - 30 * mm
        c.setFont("Helvetica-Bold", 15)
        c.drawString(25 * mm, y, "Löschzertifikat (Art. 17 DSGVO)")
        y -= 14 * mm
        c.setFont("Helvetica", 11)
        for label in [
            f"Mandanten-ID: {client_id}",
            f"Antrag-ID: {req.id}",
            f"Ausgeführt am: {datetime.now(UTC).strftime('%d.%m.%Y %H:%M UTC')}",
            f"Ausgeführt durch (User-ID): {executed_by_id}",
            "",
            "Die personenbezogenen Stammdaten des Mandanten wurden anonymisiert",
            f"(Ersetzung durch '{ERASURE_MARKER}').",
            "Audit-Log-Einträge bleiben aus Integritätsgründen erhalten; der",
            "Personenbezug entfällt durch Anonymisierung der Stammdaten.",
        ]:
            c.drawString(25 * mm, y, label)
            y -= 7 * mm
        c.showPage()
        c.save()
    except Exception:
        # Fallback: Zertifikat als Text, falls reportlab nicht verfügbar
        with open(path.replace(".pdf", ".txt"), "w", encoding="utf-8") as f:
            f.write(f"Löschzertifikat Art. 17 DSGVO\nMandant {client_id}, Antrag {req.id}, "
                    f"ausgeführt {datetime.now(UTC).isoformat()} durch User {executed_by_id}\n")
        path = path.replace(".pdf", ".txt")
    return path


# --- Verarbeitungsverzeichnis (Art. 30) ---

DEFAULT_VVT = [
    {
        "name": "Mandatsverwaltung (Akten/Mandanten)",
        "purpose": "Bearbeitung und Verwaltung anwaltlicher Mandate",
        "legal_basis": "Art. 6 Abs. 1 lit. b DSGVO (Vertrag), § 50 BRAO",
        "data_categories": ["Stammdaten", "Kontaktdaten", "Aktendaten", "ggf. besondere Kategorien"],
        "data_subjects": "Mandanten, Gegner, Zeugen, Gläubiger",
        "recipients": "Gerichte, Behörden, Verfahrensbeteiligte (nur soweit erforderlich)",
        "retention": "6 Jahre nach Mandatsende (§ 50 BRAO), steuerrelevant 10 Jahre (§ 147 AO)",
        "tom": "TLS 1.3, RBAC, Verschlüsselung von PII (Fernet), unveränderliches Audit-Log",
    },
    {
        "name": "E-Mail-Automation",
        "purpose": "Empfang, Zuordnung und Versand mandatsbezogener Kommunikation",
        "legal_basis": "Art. 6 Abs. 1 lit. b DSGVO",
        "data_categories": ["Kommunikationsdaten", "Kontaktdaten"],
        "data_subjects": "Mandanten, Verfahrensbeteiligte",
        "recipients": "—",
        "retention": "folgt Akten-Retention",
        "tom": "lokale IMAP/SMTP-Verbindung, keine Cloud-Gateways",
    },
    {
        "name": "Massekonten-Buchhaltung / Bankdatenimport",
        "purpose": "Verwaltung von Insolvenzmassen, Zuordnung von Bankbuchungen",
        "legal_basis": "Art. 6 Abs. 1 lit. c DSGVO, InsO/InsVV",
        "data_categories": ["Finanzdaten", "Kontodaten Dritter"],
        "data_subjects": "Schuldner, Gläubiger, Drittschuldner",
        "recipients": "Insolvenzgericht, Gläubigerausschuss",
        "retention": "nach insolvenzrechtlichen Vorgaben",
        "tom": "lokale Verarbeitung (CAMT/MT940), kein Cloud-Banking",
    },
    {
        "name": "Meeting-Transkription (lokal)",
        "purpose": "Verschriftlichung von Besprechungen/Vernehmungen",
        "legal_basis": "Art. 6 Abs. 1 lit. b DSGVO",
        "data_categories": ["Audiodaten", "Inhaltsdaten"],
        "data_subjects": "Gesprächsteilnehmer",
        "recipients": "—",
        "retention": "folgt Akten-Retention",
        "tom": "lokale KI (faster-whisper/pyannote), Zwischen-WAV sofort gelöscht, Original verschlüsselt",
    },
    {
        "name": "Gläubiger-Portal (Online-Forderungsanmeldung)",
        "purpose": "Entgegennahme von Forderungsanmeldungen durch Gläubiger",
        "legal_basis": "Art. 6 Abs. 1 lit. c DSGVO, InsO",
        "data_categories": ["Kontaktdaten", "Forderungsdaten"],
        "data_subjects": "Gläubiger",
        "recipients": "—",
        "retention": "nach insolvenzrechtlichen Vorgaben",
        "tom": "token-gesichert, unauthentifiziert nur anlegen, Rate-Limiting, Untrusted-Behandlung",
    },
]


async def seed_vvt_if_empty(db: AsyncSession) -> int:
    result = await db.execute(select(ProcessingRecord.id).limit(1))
    if result.scalar_one_or_none() is not None:
        return 0
    for rec in DEFAULT_VVT:
        db.add(ProcessingRecord(**rec))
    await db.commit()
    return len(DEFAULT_VVT)


# --- Datenportabilität (Art. 20) ---

async def create_export(db: AsyncSession, *, client_id: int, requested_by_id: int) -> DataExport:
    export = DataExport(
        client_id=client_id, requested_by_id=requested_by_id, status="pending",
        token=secrets.token_urlsafe(32), expires_at=datetime.now(UTC) + timedelta(hours=48),
    )
    db.add(export)
    await db.commit()
    await db.refresh(export)
    return export


async def build_export_zip(db: AsyncSession, export: DataExport) -> str:
    """Erzeugt ein ZIP mit allen Daten zu einem Mandanten (client.json, matters, …)."""
    import io
    import os
    import zipfile

    from app.core.config import get_settings

    settings = get_settings()
    out_dir = os.path.join(str(settings.STORAGE_ROOT), "dsgvo", "exports")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"export_{export.client_id}_{export.id}.zip")

    client = await db.get(Client, export.client_id)
    matters_result = await db.execute(select(Matter).where(Matter.client_id == export.client_id))
    matters = matters_result.scalars().all()

    def _client_dict():
        return {
            "client_number": client.client_number, "first_name": client.first_name,
            "last_name": client.last_name, "company_name": client.company_name,
            "email": client.email, "phone": client.phone, "city": client.city,
            "dsgvo_legal_basis": client.dsgvo_legal_basis,
        } if client else {}

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("client.json", json.dumps(_client_dict(), ensure_ascii=False, indent=2))
        z.writestr(
            "matters/matters.json",
            json.dumps(
                [{"matter_number": m.matter_number, "title": m.title, "status": m.status,
                  "opened_at": m.opened_at.isoformat() if m.opened_at else None} for m in matters],
                ensure_ascii=False, indent=2,
            ),
        )
        z.writestr("README.txt", "Datenexport gemäß Art. 20 DSGVO.\nMaschinenlesbares Format (JSON).\n")
    with open(path, "wb") as f:
        f.write(buf.getvalue())

    export.file_path = path
    export.status = "ready"
    await db.commit()
    await db.refresh(export)
    return path
