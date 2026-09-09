"""
E-Mail Service: IMAP-Abruf, Persistenz, Regelanwendung, SMTP-Versand,
Template-Rendering. Die reine Routing-Logik liegt in email_routing.py.
"""
from __future__ import annotations

import email
import hashlib
import logging
import re
from datetime import UTC, datetime
from email.header import decode_header, make_header
from email.utils import formataddr, getaddresses, make_msgid, parseaddr, parsedate_to_datetime
from typing import NamedTuple

from jinja2 import StrictUndefined, TemplateError
from jinja2.sandbox import SandboxedEnvironment
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.email import EmailAttachment, EmailMessage, EmailRule, EmailTemplate
from app.services.email_routing import (
    IncomingEmail,
    detect_confidential,
    evaluate_rules,
)

logger = logging.getLogger("app.email")

# SandboxedEnvironment statt Environment: Vorlagen kommen aus der Datenbank und
# werden mit from_string() ausgeführt. Ein unsandboxed Environment erlaubt über
# Attributzugriffe ({{ ''.__class__.__mro__ }}) den Ausbruch bis zur
# Shell — Templates sind damit faktisch Code, nicht Daten.
# autoescape=True: Vorlagen erzeugen auch HTML-Mails, Kontextwerte (Mandanten-
# namen, Aktenzeichen) müssen escaped werden. StrictUndefined macht fehlende
# Variablen laut, statt sie als Leerstring auszuliefern.
_jinja = SandboxedEnvironment(undefined=StrictUndefined, autoescape=True)


def _decode(value: str | None) -> str:
    if not value:
        return ""
    try:
        return str(make_header(decode_header(value)))
    except Exception:
        return value


# Anhänge größer als dieses Limit werden nicht extrahiert (bleiben im Postfach);
# Schutz gegen Storage-Flutung durch eine einzelne eingehende Mail.
MAX_ATTACHMENT_BYTES = 25 * 1024 * 1024

_FILENAME_KEEP = re.compile(r"[^\w.\-() äöüÄÖÜß]")


def safe_attachment_filename(name: str | None) -> str:
    """Sanitize an attachment filename for on-disk storage (no path traversal)."""
    base = (name or "").replace("\\", "/").rsplit("/", 1)[-1].strip()
    base = _FILENAME_KEEP.sub("_", base).strip(". ")
    if len(base) > 140:
        stem, dot, ext = base.rpartition(".")
        base = (stem[: 140 - len(ext) - 1] + dot + ext) if dot else base[:140]
    return base or "anhang.bin"


def parse_date_header(raw: str | None) -> datetime | None:
    """
    RFC-2822-Datum in ein timezone-bewusstes datetime uebersetzen.

    Wichtig fuer eine Kanzlei: aus dem Zugangszeitpunkt laufen Fristen. Vorher
    wurde der Header gar nicht ausgelesen und stattdessen der Ingest-Zeitpunkt
    gespeichert -- nach einem Sync-Ausfall trugen alle nachgeholten Mails
    dasselbe Datum.
    """
    if not raw:
        return None
    try:
        dt = parsedate_to_datetime(raw)
    except (TypeError, ValueError):
        return None
    if dt is None:
        return None
    # Mails ohne Zeitzone als UTC lesen, statt sie naiv zu speichern
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


def thread_root(references: str | None, in_reply_to: str | None, message_id: str) -> str:
    """
    Wurzel-Message-ID eines Threads.

    Frueher wurde in_reply_to als thread_key genutzt. Damit zeigt eine Antwort
    auf eine Antwort auf die Antwort statt auf den Ursprung -- derselbe Verlauf
    zerfiel ab der dritten Ebene in mehrere Threads. RFC 5322 fuehrt in
    References die vollstaendige Kette, aeltester Eintrag zuerst.
    """
    if references:
        ids = re.findall(r"<[^<>]+>", references)
        if ids:
            return ids[0]
    return in_reply_to or message_id


def parse_raw_email(raw_bytes: bytes) -> dict:
    """Parse a raw RFC822 message into a normalized dict (incl. attachments)."""
    msg = email.message_from_bytes(raw_bytes)

    from_name, from_addr = parseaddr(msg.get("From", ""))
    to_addrs = [addr for _, addr in getaddresses(msg.get_all("To", []))]
    cc_addrs = [addr for _, addr in getaddresses(msg.get_all("Cc", []))]

    body_text, body_html = "", ""
    attachments: list[dict] = []
    attachment_bytes = 0
    if msg.is_multipart():
        for part in msg.walk():
            if part.is_multipart():
                continue
            ctype = part.get_content_type()
            disp = str(part.get("Content-Disposition") or "").lower()
            filename = _decode(part.get_filename())
            # Anhang = explizite attachment-Disposition ODER benannter Nicht-Text-Part
            # (deckt inline-Bilder/PDFs ab, die viele Clients ohne "attachment" senden).
            if "attachment" in disp or (filename and ctype not in ("text/plain", "text/html")):
                try:
                    payload = part.get_payload(decode=True)
                except Exception:  # noqa: S112
                    # Ein unlesbarer Teil darf den Import der Mail nicht verhindern
                    continue
                if not payload or len(payload) > MAX_ATTACHMENT_BYTES:
                    continue
                # Grenzen je NACHRICHT, nicht nur je Anhang: 1.000 Anhaenge zu
                # je 24 MB waeren sonst 24 GB, komplett im Arbeitsspeicher.
                if len(attachments) >= _limits().EMAIL_MAX_ATTACHMENTS:
                    logger.warning("Anhangs-Obergrenze erreicht, weitere verworfen")
                    continue
                if attachment_bytes + len(payload) > _limits().EMAIL_MAX_TOTAL_ATTACHMENT_BYTES:
                    logger.warning("Gesamt-Anhangsgroesse ueberschritten, weitere verworfen")
                    continue
                attachment_bytes += len(payload)
                attachments.append(
                    {"filename": safe_attachment_filename(filename), "content_type": ctype, "payload": payload}
                )
                continue
            try:
                payload = part.get_payload(decode=True)
                if payload is None:
                    continue
                charset = part.get_content_charset() or "utf-8"
                decoded = payload.decode(charset, errors="replace")
            except Exception:  # noqa: S112
                # Ein unlesbarer Teil darf den Import der Mail nicht verhindern
                continue
            if ctype == "text/plain" and not body_text:
                body_text = decoded
            elif ctype == "text/html" and not body_html:
                body_html = decoded
    else:
        payload = msg.get_payload(decode=True)
        if payload:
            charset = msg.get_content_charset() or "utf-8"
            body_text = payload.decode(charset, errors="replace")

    message_id = (msg.get("Message-ID") or "").strip()
    if not message_id:
        # Synthesize a stable ID from headers + body for dedup
        digest = hashlib.sha256(raw_bytes).hexdigest()
        message_id = f"<synthetic-{digest}@local>"

    references = msg.get("References")
    in_reply_to = (msg.get("In-Reply-To") or "").strip() or None

    return {
        "message_id": message_id,
        "from_address": from_addr.lower(),
        "to_addresses": to_addrs,
        "cc_addresses": cc_addrs,
        "subject": _decode(msg.get("Subject")),
        "body_text": body_text,
        "body_html": body_html,
        "in_reply_to": in_reply_to,
        "references": references,
        # Echtes Sendedatum aus dem Header; None, wenn er fehlt oder unlesbar ist
        "email_date": parse_date_header(msg.get("Date")),
        "thread_key": thread_root(references, in_reply_to, message_id),
        "attachments": attachments,
    }


# Voreinstellungen der Anhangsgrenzen. parse_raw_email bleibt damit ohne
# Konfiguration importier- und testbar (reines Parsing); der Betreiber kann die
# Werte per .env anheben.
DEFAULT_MAX_ATTACHMENTS = 50
DEFAULT_MAX_TOTAL_ATTACHMENT_BYTES = 100 * 1024 * 1024


class _Limits(NamedTuple):
    EMAIL_MAX_ATTACHMENTS: int
    EMAIL_MAX_TOTAL_ATTACHMENT_BYTES: int


def _limits() -> _Limits:
    """Anhangsgrenzen aus den Settings, mit Rückfall auf die Voreinstellungen."""
    try:
        from app.core.config import get_settings

        settings = get_settings()
        return _Limits(
            settings.EMAIL_MAX_ATTACHMENTS,
            settings.EMAIL_MAX_TOTAL_ATTACHMENT_BYTES,
        )
    except Exception:
        # Ohne vollständige Umgebung (reine Parser-Tests) gelten die Defaults.
        return _Limits(DEFAULT_MAX_ATTACHMENTS, DEFAULT_MAX_TOTAL_ATTACHMENT_BYTES)


async def message_exists(db: AsyncSession, message_id: str) -> bool:
    result = await db.execute(
        select(EmailMessage.id).where(EmailMessage.message_id == message_id)
    )
    return result.scalar_one_or_none() is not None


async def _load_active_rules(db: AsyncSession) -> list[tuple[int, dict, dict, int]]:
    result = await db.execute(
        select(EmailRule)
        .where(EmailRule.is_active == True)  # noqa: E712
        .order_by(EmailRule.priority.asc())
    )
    return [(r.id, r.conditions or {}, r.actions or {}, r.priority) for r in result.scalars().all()]


async def _known_sender(db: AsyncSession, from_address: str) -> int | None:
    """
    Return client_id if the sender address matches a known client.

    Die Adresse ist verschlüsselt gespeichert; gesucht wird über den
    deterministischen Blind-Index. `.first()` statt `.scalar_one_or_none()`:
    theoretisch können zwei Mandanten dieselbe Adresse führen (Eheleute,
    Sammelpostfach) — das darf den Mail-Import nicht mit einem
    MultipleResultsFound abbrechen.
    """
    from app.core.encryption import blind_index
    from app.models.client import Client

    idx = blind_index(from_address)
    if idx is None:
        return None
    result = await db.execute(
        select(Client.id)
        .where(Client.email_index == idx, Client.deleted_at.is_(None))
        .order_by(Client.id.asc())
    )
    return result.scalars().first()


async def ingest_email(db: AsyncSession, parsed: dict) -> EmailMessage | None:
    """
    Persist a parsed inbound email with dedup, rule routing, and confidentiality
    detection. Returns None if the message was a duplicate.
    """
    if await message_exists(db, parsed["message_id"]):
        return None

    incoming = IncomingEmail(
        from_address=parsed["from_address"],
        subject=parsed.get("subject", ""),
        body_text=parsed.get("body_text", ""),
    )

    rules = await _load_active_rules(db)
    match = evaluate_rules(rules, incoming)

    client_id = await _known_sender(db, parsed["from_address"])

    matter_id = None
    confidential = detect_confidential(incoming)
    needs_review = False
    matched_rule_id = None

    if match:
        matched_rule_id = match.rule_id
        actions = match.actions
        if "file_to_matter" in actions:
            matter_id = actions["file_to_matter"]
        if actions.get("set_confidential"):
            confidential = True
        if actions.get("flag_review"):
            needs_review = True

    # Unknown sender + no rule match → route to review queue
    unknown_sender = client_id is None and matter_id is None
    if unknown_sender and not match:
        needs_review = True

    msg = EmailMessage(
        message_id=parsed["message_id"],
        direction="inbound",
        from_address=parsed["from_address"],
        to_addresses=parsed.get("to_addresses", []),
        cc_addresses=parsed.get("cc_addresses", []),
        subject=parsed.get("subject"),
        body_text=parsed.get("body_text"),
        body_html=parsed.get("body_html"),
        in_reply_to=parsed.get("in_reply_to"),
        references=parsed.get("references"),
        thread_key=parsed.get("thread_key") or parsed["message_id"],
        matter_id=matter_id,
        client_id=client_id,
        matched_rule_id=matched_rule_id,
        is_confidential=confidential,
        needs_review=needs_review,
        unknown_sender=unknown_sender,
        # Echtes Sendedatum; nur als Rueckfall der Ingest-Zeitpunkt.
        email_date=parsed.get("email_date") or datetime.now(UTC),
    )
    db.add(msg)
    await db.commit()
    await db.refresh(msg)

    if parsed.get("attachments"):
        await store_attachments(db, msg.id, parsed["attachments"])

    return msg


async def store_attachments(db: AsyncSession, email_id: int, attachments: list[dict]) -> list[EmailAttachment]:
    """
    Persist parsed attachments: Fernet-verschlüsselte Ablage unter
    STORAGE_ROOT/emails/{email_id}/ + EmailAttachment-Zeilen (Klartext liegt
    nie auf der Platte — DSGVO/TOM). Idempotent pro ingest (nur bei Neuanlage
    der Mail aufgerufen; Dedup verhindert Doppel-Ingest).
    """
    import asyncio

    from app.core.config import get_settings
    from app.core.encryption import encrypt_bytes

    settings = get_settings()
    target_dir = settings.STORAGE_ROOT / "emails" / str(email_id)
    target_dir.mkdir(parents=True, exist_ok=True)

    records: list[EmailAttachment] = []
    for i, att in enumerate(attachments):
        filename = safe_attachment_filename(att.get("filename"))
        payload: bytes = att["payload"]
        enc_path = target_dir / f"{i:02d}_{filename}.enc"
        token = await asyncio.to_thread(encrypt_bytes, payload)
        await asyncio.to_thread(enc_path.write_bytes, token)
        records.append(
            EmailAttachment(
                email_id=email_id,
                filename=filename,
                content_type=att.get("content_type"),
                size_bytes=len(payload),
                storage_path=str(enc_path),
            )
        )
    db.add_all(records)
    await db.commit()
    return records


async def read_attachment_bytes(attachment: EmailAttachment) -> bytes:
    """Decrypt an attachment from disk into memory (for authenticated download)."""
    import asyncio
    from pathlib import Path

    from app.core.encryption import decrypt_bytes

    token = await asyncio.to_thread(Path(attachment.storage_path).read_bytes)
    return await asyncio.to_thread(decrypt_bytes, token)


def render_template(template: EmailTemplate, context: dict) -> tuple[str, str]:
    """Render subject + body of an EmailTemplate with the given context."""
    try:
        subject = _jinja.from_string(template.subject_template).render(**context)
        body = _jinja.from_string(template.body_template).render(**context)
    except TemplateError as exc:
        raise ValueError(f"Template-Fehler: {exc}") from exc
    return subject, body


def chunk_recipients(addresses: list[str], max_per_message: int) -> list[list[str]]:
    """
    Zerlegt eine Empfaengerliste in versendbare Bloecke. Ein Gläubiger-
    rundschreiben kann hunderte Adressen haben; viele Mailserver lehnen
    ueberlange Empfaengerlisten ab.
    """
    if max_per_message < 1:
        raise ValueError("max_per_message muss mindestens 1 sein")
    return [
        addresses[i : i + max_per_message]
        for i in range(0, len(addresses), max_per_message)
    ]


def build_mime(
    *,
    from_name: str,
    from_email: str,
    to_addresses: list[str],
    subject: str,
    body_text: str,
    body_html: str | None,
    message_id: str,
    use_bcc: bool,
    in_reply_to: str | None = None,
    references: str | None = None,
):
    """
    Baut die MIME-Nachricht.

    use_bcc: Bei mehreren Empfaengern MUSS BCC verwendet werden. Vorher landeten
    alle Adressen im To-Header — bei einem Glaeubigerrundschreiben im
    Insolvenzverfahren erfaehrt damit jeder Glaeubiger die Adressen aller
    anderen (meldepflichtige Datenpanne nach Art. 33 DSGVO).
    """
    from email.message import EmailMessage as PyEmailMessage

    mime = PyEmailMessage()
    mime["From"] = formataddr((from_name, from_email))
    if use_bcc:
        # An sich selbst adressieren, alle echten Empfaenger ins Bcc
        mime["To"] = formataddr((from_name, from_email))
        mime["Bcc"] = ", ".join(to_addresses)
    else:
        mime["To"] = ", ".join(to_addresses)
    mime["Subject"] = subject
    # Die Message-ID wird gesetzt UND gespeichert. Vorher wurde eine
    # Timestamp-Bastelei nur in die DB geschrieben, waehrend der Mailserver der
    # tatsaechlichen Nachricht eine andere ID gab — der Datensatz war mit der
    # realen Mail nie korrelierbar, und zwei Sends in derselben Mikrosekunde
    # kollidierten auf der Unique-Spalte.
    mime["Message-ID"] = message_id
    if in_reply_to:
        mime["In-Reply-To"] = in_reply_to
    if references:
        mime["References"] = references
    mime.set_content(body_text)
    if body_html:
        mime.add_alternative(body_html, subtype="html")
    return mime


async def send_email(
    db: AsyncSession,
    *,
    to_addresses: list[str],
    subject: str,
    body_text: str,
    body_html: str | None,
    sent_by_id: int,
    matter_id: int | None = None,
    client_id: int | None = None,
    force_bcc: bool | None = None,
    in_reply_to: str | None = None,
) -> EmailMessage:
    """
    Stellt eine Nachricht in die Outbox und versucht sofort zuzustellen.

    Anders als frueher wird ein SMTP-Fehler NICHT verschluckt: die Nachricht
    bleibt mit delivery_status="queued" und einem Fehlertext liegen und wird
    vom Celery-Task `retry_pending_outbox` erneut versucht. In einer Kanzlei
    ist eine still verlorene Mail ein Fristproblem.

    Bei mehr als einem Empfaenger wird automatisch BCC verwendet
    (force_bcc ueberschreibt die Automatik).
    """
    from app.core.config import get_settings

    settings = get_settings()
    recipients = [a.strip() for a in to_addresses if a and a.strip()]
    if not recipients:
        raise ValueError("Mindestens ein Empfaenger erforderlich")

    use_bcc = force_bcc if force_bcc is not None else len(recipients) > 1
    from_email = settings.SMTP_FROM_EMAIL or "noreply@local"
    domain = from_email.rsplit("@", 1)[-1] if "@" in from_email else "local"

    record = EmailMessage(
        message_id=make_msgid(domain=domain),
        direction="outbound",
        from_address=from_email,
        to_addresses=recipients,
        subject=subject,
        body_text=body_text,
        body_html=body_html,
        matter_id=matter_id,
        client_id=client_id,
        sent_by_id=sent_by_id,
        in_reply_to=in_reply_to,
        thread_key=in_reply_to or None,
        delivery_status="queued",
        delivery_attempts=0,
        use_bcc=use_bcc,
        email_date=datetime.now(UTC),
    )
    db.add(record)
    await db.commit()
    await db.refresh(record)

    await attempt_delivery(db, record)
    return record


async def attempt_delivery(db: AsyncSession, record: EmailMessage) -> bool:
    """
    Ein Zustellversuch fuer einen Outbox-Eintrag. Gibt True bei Erfolg zurueck.
    Zaehlt Versuche und haelt den letzten Fehler fest; nach SMTP_MAX_ATTEMPTS
    gilt die Nachricht als endgueltig gescheitert und wird nicht weiter
    versucht (der Eintrag bleibt sichtbar, damit niemand von einem Versand
    ausgeht, der nie stattfand).
    """
    from app.core.config import get_settings

    settings = get_settings()

    if not settings.SMTP_HOST:
        record.delivery_status = "no_smtp_configured"
        record.delivery_error = "SMTP_HOST ist nicht konfiguriert"
        await db.commit()
        logger.warning("E-Mail %s nicht versendet: SMTP nicht konfiguriert", record.message_id)
        return False

    import aiosmtplib

    recipients = list(record.to_addresses or [])
    chunks = chunk_recipients(recipients, settings.EMAIL_MAX_RECIPIENTS_PER_MESSAGE)

    record.delivery_attempts = (record.delivery_attempts or 0) + 1
    try:
        for chunk in chunks:
            mime = build_mime(
                from_name=settings.SMTP_FROM_NAME,
                from_email=record.from_address,
                to_addresses=chunk,
                subject=record.subject or "",
                body_text=record.body_text or "",
                body_html=record.body_html,
                message_id=record.message_id,
                use_bcc=bool(record.use_bcc) or len(chunk) > 1,
                in_reply_to=record.in_reply_to,
            )
            await aiosmtplib.send(
                mime,
                hostname=settings.SMTP_HOST,
                port=settings.SMTP_PORT,
                username=settings.SMTP_USERNAME or None,
                password=settings.SMTP_PASSWORD or None,
                start_tls=settings.SMTP_TLS,
                timeout=settings.SMTP_TIMEOUT_SECONDS,
            )
    except Exception as exc:
        # Bewusst breit: aiosmtplib wirft je nach Fehlerbild sehr
        # unterschiedliche Typen, und keiner davon darf still verschwinden.
        record.delivery_error = f"{type(exc).__name__}: {exc}"[:1000]
        if record.delivery_attempts >= settings.SMTP_MAX_ATTEMPTS:
            record.delivery_status = "failed"
            logger.error(
                "E-Mail %s endgueltig gescheitert nach %d Versuchen: %s",
                record.message_id, record.delivery_attempts, record.delivery_error,
            )
        else:
            record.delivery_status = "queued"
            logger.warning(
                "Zustellversuch %d fuer %s fehlgeschlagen: %s",
                record.delivery_attempts, record.message_id, record.delivery_error,
            )
        await db.commit()
        return False

    record.delivery_status = "sent"
    record.delivery_error = None
    record.sent_at = datetime.now(UTC)
    await db.commit()
    logger.info("E-Mail %s zugestellt an %d Empfaenger", record.message_id, len(recipients))
    return True


async def pending_outbox(db: AsyncSession, limit: int = 50) -> list[EmailMessage]:
    """Outbox-Eintraege, die noch einen Zustellversuch verdienen."""
    from app.core.config import get_settings

    max_attempts = get_settings().SMTP_MAX_ATTEMPTS
    result = await db.execute(
        select(EmailMessage)
        .where(
            EmailMessage.direction == "outbound",
            EmailMessage.delivery_status == "queued",
            EmailMessage.delivery_attempts < max_attempts,
        )
        .order_by(EmailMessage.id.asc())
        .limit(limit)
    )
    return list(result.scalars().all())
