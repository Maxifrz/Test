"""
E-Mail Service: IMAP-Abruf, Persistenz, Regelanwendung, SMTP-Versand,
Template-Rendering. Die reine Routing-Logik liegt in email_routing.py.
"""
from __future__ import annotations

import email
import hashlib
from datetime import UTC, datetime
from email.header import decode_header, make_header
from email.utils import parseaddr, getaddresses

from jinja2 import Environment, StrictUndefined, TemplateError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.models.email import EmailMessage, EmailRule, EmailTemplate
from app.services.email_routing import (
    IncomingEmail,
    detect_confidential,
    evaluate_rules,
)

# Sandboxed Jinja env — autoescape off (templates are plain text/HTML authored
# by Kanzlei staff), StrictUndefined surfaces missing variables loudly.
_jinja = Environment(undefined=StrictUndefined, autoescape=True)


def _decode(value: str | None) -> str:
    if not value:
        return ""
    try:
        return str(make_header(decode_header(value)))
    except Exception:
        return value


def parse_raw_email(raw_bytes: bytes) -> dict:
    """Parse a raw RFC822 message into a normalized dict."""
    msg = email.message_from_bytes(raw_bytes)

    from_name, from_addr = parseaddr(msg.get("From", ""))
    to_addrs = [addr for _, addr in getaddresses(msg.get_all("To", []))]
    cc_addrs = [addr for _, addr in getaddresses(msg.get_all("Cc", []))]

    body_text, body_html = "", ""
    if msg.is_multipart():
        for part in msg.walk():
            ctype = part.get_content_type()
            disp = str(part.get("Content-Disposition") or "")
            if "attachment" in disp:
                continue
            try:
                payload = part.get_payload(decode=True)
                if payload is None:
                    continue
                charset = part.get_content_charset() or "utf-8"
                decoded = payload.decode(charset, errors="replace")
            except Exception:
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

    return {
        "message_id": message_id,
        "from_address": from_addr.lower(),
        "to_addresses": to_addrs,
        "cc_addresses": cc_addrs,
        "subject": _decode(msg.get("Subject")),
        "body_text": body_text,
        "body_html": body_html,
        "in_reply_to": (msg.get("In-Reply-To") or "").strip() or None,
        "references": msg.get("References"),
    }


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
    """Return client_id if the sender address matches a known client."""
    from app.models.client import Client

    result = await db.execute(
        select(Client.id).where(Client.email == from_address, Client.deleted_at.is_(None))
    )
    return result.scalar_one_or_none()


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
        thread_key=parsed.get("in_reply_to") or parsed["message_id"],
        matter_id=matter_id,
        client_id=client_id,
        matched_rule_id=matched_rule_id,
        is_confidential=confidential,
        needs_review=needs_review,
        unknown_sender=unknown_sender,
        email_date=datetime.now(UTC),
    )
    db.add(msg)
    await db.commit()
    await db.refresh(msg)
    return msg


def render_template(template: EmailTemplate, context: dict) -> tuple[str, str]:
    """Render subject + body of an EmailTemplate with the given context."""
    try:
        subject = _jinja.from_string(template.subject_template).render(**context)
        body = _jinja.from_string(template.body_template).render(**context)
    except TemplateError as exc:
        raise ValueError(f"Template-Fehler: {exc}") from exc
    return subject, body


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
) -> EmailMessage:
    """
    Send via SMTP and persist an outbound record with an audit-friendly trail.
    Each recipient gets its own delivery record handled by the caller for
    mass-send scenarios; this sends a single message to all recipients.
    """
    from app.core.config import get_settings

    settings = get_settings()
    sent_ok = False
    delivery_status = "failed"

    if settings.SMTP_HOST:
        import aiosmtplib
        from email.message import EmailMessage as PyEmailMessage

        mime = PyEmailMessage()
        mime["From"] = f"{settings.SMTP_FROM_NAME} <{settings.SMTP_FROM_EMAIL}>"
        mime["To"] = ", ".join(to_addresses)
        mime["Subject"] = subject
        mime.set_content(body_text)
        if body_html:
            mime.add_alternative(body_html, subtype="html")

        try:
            await aiosmtplib.send(
                mime,
                hostname=settings.SMTP_HOST,
                port=settings.SMTP_PORT,
                username=settings.SMTP_USERNAME or None,
                password=settings.SMTP_PASSWORD or None,
                start_tls=settings.SMTP_TLS,
            )
            sent_ok = True
            delivery_status = "sent"
        except Exception:
            delivery_status = "failed"

    record = EmailMessage(
        message_id=f"<outbound-{datetime.now(UTC).timestamp()}@local>",
        direction="outbound",
        from_address=settings.SMTP_FROM_EMAIL if settings.SMTP_HOST else "noreply@local",
        to_addresses=to_addresses,
        subject=subject,
        body_text=body_text,
        body_html=body_html,
        matter_id=matter_id,
        client_id=client_id,
        sent_by_id=sent_by_id,
        sent_at=datetime.now(UTC) if sent_ok else None,
        delivery_status=delivery_status,
        email_date=datetime.now(UTC),
    )
    db.add(record)
    await db.commit()
    await db.refresh(record)
    return record
