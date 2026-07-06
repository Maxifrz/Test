from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from sqlalchemy import func, select

from app.core.deps import DB, accessible_matter_ids, ensure_matter_access, require_permission
from app.models.email import EmailAttachment, EmailMessage, EmailRule, EmailTemplate
from app.schemas.email import (
    EmailAttachmentResponse,
    EmailDetail,
    EmailFileRequest,
    EmailListResponse,
    EmailPreviewRequest,
    EmailPreviewResponse,
    EmailRuleCreate,
    EmailRuleResponse,
    EmailRuleUpdate,
    EmailSendRequest,
    EmailTemplateCreate,
    EmailTemplateResponse,
    EmailTemplateUpdate,
)
from app.services import email_service

router = APIRouter(prefix="/emails", tags=["emails"])


@router.get("", response_model=EmailListResponse)
async def list_emails(
    db: DB,
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=100),
    direction: str | None = Query(None),
    matter_id: int | None = Query(None),
    needs_review: bool | None = Query(None),
    current_user=Depends(require_permission("email.read")),
):
    query = select(EmailMessage).where(EmailMessage.deleted_at.is_(None))
    if direction:
        query = query.where(EmailMessage.direction == direction)
    if matter_id:
        await ensure_matter_access(db, current_user, matter_id)
        query = query.where(EmailMessage.matter_id == matter_id)
    else:
        # Trennungsgebot: Nicht-Admins sehen nur E-Mails ihrer Akten
        # (plus die unzugeordnete Review-Queue).
        allowed = await accessible_matter_ids(db, current_user)
        if allowed is not None:
            query = query.where(
                (EmailMessage.matter_id.is_(None)) | (EmailMessage.matter_id.in_(allowed))
            )
    if needs_review is not None:
        query = query.where(EmailMessage.needs_review == needs_review)

    count_result = await db.execute(select(func.count()).select_from(query.subquery()))
    total = count_result.scalar_one()

    query = query.order_by(EmailMessage.email_date.desc().nullslast()).offset((page - 1) * page_size).limit(page_size)
    result = await db.execute(query)
    return EmailListResponse(items=result.scalars().all(), total=total, page=page, page_size=page_size)


@router.post("/send", response_model=EmailDetail, status_code=status.HTTP_201_CREATED)
async def send_email(
    data: EmailSendRequest,
    db: DB,
    current_user=Depends(require_permission("email.send")),
):
    return await email_service.send_email(
        db,
        to_addresses=data.to_addresses,
        subject=data.subject,
        body_text=data.body_text,
        body_html=data.body_html,
        sent_by_id=current_user.id,
        matter_id=data.matter_id,
        client_id=data.client_id,
    )


# --- Templates ---

@router.post("/templates", response_model=EmailTemplateResponse, status_code=status.HTTP_201_CREATED)
async def create_template(
    data: EmailTemplateCreate,
    db: DB,
    current_user=Depends(require_permission("email.rule.create")),
):
    tmpl = EmailTemplate(**data.model_dump(), created_by_id=current_user.id)
    db.add(tmpl)
    await db.commit()
    await db.refresh(tmpl)
    return tmpl


@router.get("/templates", response_model=list[EmailTemplateResponse])
async def list_templates(
    db: DB,
    current_user=Depends(require_permission("email.read")),
):
    result = await db.execute(
        select(EmailTemplate).where(EmailTemplate.deleted_at.is_(None), EmailTemplate.is_active == True)  # noqa: E712
    )
    return result.scalars().all()


@router.post("/templates/preview", response_model=EmailPreviewResponse)
async def preview_template(
    data: EmailPreviewRequest,
    db: DB,
    current_user=Depends(require_permission("email.read")),
):
    result = await db.execute(select(EmailTemplate).where(EmailTemplate.id == data.template_id))
    tmpl = result.scalar_one_or_none()
    if not tmpl:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Template not found")
    try:
        subject, body = email_service.render_template(tmpl, data.context)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc))
    return EmailPreviewResponse(subject=subject, body=body)


# --- Rules ---

@router.post("/rules", response_model=EmailRuleResponse, status_code=status.HTTP_201_CREATED)
async def create_rule(
    data: EmailRuleCreate,
    db: DB,
    current_user=Depends(require_permission("email.rule.create")),
):
    rule = EmailRule(**data.model_dump(), created_by_id=current_user.id)
    db.add(rule)
    await db.commit()
    await db.refresh(rule)
    return rule


@router.get("/rules", response_model=list[EmailRuleResponse])
async def list_rules(
    db: DB,
    current_user=Depends(require_permission("email.read")),
):
    result = await db.execute(select(EmailRule).order_by(EmailRule.priority.asc()))
    return result.scalars().all()


@router.patch("/rules/{rule_id}", response_model=EmailRuleResponse)
async def update_rule(
    rule_id: int,
    data: EmailRuleUpdate,
    db: DB,
    current_user=Depends(require_permission("email.rule.update")),
):
    result = await db.execute(select(EmailRule).where(EmailRule.id == rule_id))
    rule = result.scalar_one_or_none()
    if not rule:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Rule not found")
    for field, value in data.model_dump(exclude_unset=True).items():
        setattr(rule, field, value)
    await db.commit()
    await db.refresh(rule)
    return rule


@router.delete("/rules/{rule_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_rule(
    rule_id: int,
    db: DB,
    current_user=Depends(require_permission("email.rule.delete")),
):
    result = await db.execute(select(EmailRule).where(EmailRule.id == rule_id))
    rule = result.scalar_one_or_none()
    if not rule:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Rule not found")
    await db.delete(rule)
    await db.commit()


# --- Einzel-E-Mail (parametrische Routen MÜSSEN nach den statischen stehen,
#     sonst fängt /{email_id} z. B. GET /templates ab) ---

@router.get("/{email_id}", response_model=EmailDetail)
async def get_email(
    email_id: int,
    db: DB,
    current_user=Depends(require_permission("email.read")),
):
    result = await db.execute(
        select(EmailMessage).where(EmailMessage.id == email_id, EmailMessage.deleted_at.is_(None))
    )
    msg = result.scalar_one_or_none()
    if not msg:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Email not found")
    await ensure_matter_access(db, current_user, msg.matter_id)
    if not msg.is_read:
        msg.is_read = True
        await db.commit()
        await db.refresh(msg)
    return msg


async def _load_email_guarded(db, current_user, email_id: int) -> EmailMessage:
    result = await db.execute(
        select(EmailMessage).where(EmailMessage.id == email_id, EmailMessage.deleted_at.is_(None))
    )
    msg = result.scalar_one_or_none()
    if not msg:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Email not found")
    await ensure_matter_access(db, current_user, msg.matter_id)
    return msg


@router.get("/{email_id}/attachments", response_model=list[EmailAttachmentResponse])
async def list_attachments(
    email_id: int,
    db: DB,
    current_user=Depends(require_permission("email.read")),
):
    await _load_email_guarded(db, current_user, email_id)
    result = await db.execute(
        select(EmailAttachment).where(EmailAttachment.email_id == email_id).order_by(EmailAttachment.id)
    )
    return result.scalars().all()


@router.get("/{email_id}/attachments/{attachment_id}/download")
async def download_attachment(
    email_id: int,
    attachment_id: int,
    db: DB,
    current_user=Depends(require_permission("email.read")),
):
    await _load_email_guarded(db, current_user, email_id)
    result = await db.execute(
        select(EmailAttachment).where(
            EmailAttachment.id == attachment_id, EmailAttachment.email_id == email_id
        )
    )
    att = result.scalar_one_or_none()
    if not att:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Attachment not found")
    try:
        data = await email_service.read_attachment_bytes(att)
    except (FileNotFoundError, ValueError):
        raise HTTPException(status_code=status.HTTP_410_GONE, detail="Anhang nicht mehr verfügbar")
    # RFC 5987-Encoding für Umlaute im Dateinamen
    from urllib.parse import quote

    return Response(
        content=data,
        media_type=att.content_type or "application/octet-stream",
        headers={
            "Content-Disposition": f"attachment; filename*=UTF-8''{quote(att.filename)}",
            "X-Content-Type-Options": "nosniff",
        },
    )


@router.post("/{email_id}/file", response_model=EmailDetail)
async def file_email_to_matter(
    email_id: int,
    data: EmailFileRequest,
    db: DB,
    current_user=Depends(require_permission("email.read")),
):
    result = await db.execute(
        select(EmailMessage).where(EmailMessage.id == email_id, EmailMessage.deleted_at.is_(None))
    )
    msg = result.scalar_one_or_none()
    if not msg:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Email not found")
    # Zugriff auf Quelle (falls schon zugeordnet) UND Ziel-Akte erforderlich
    await ensure_matter_access(db, current_user, msg.matter_id)
    await ensure_matter_access(db, current_user, data.matter_id)
    msg.matter_id = data.matter_id
    msg.needs_review = False
    msg.unknown_sender = False
    await db.commit()
    await db.refresh(msg)
    return msg
