from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import func, select

from app.core.deps import DB, require_permission
from app.models.email import EmailMessage, EmailRule, EmailTemplate
from app.schemas.email import (
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
        query = query.where(EmailMessage.matter_id == matter_id)
    if needs_review is not None:
        query = query.where(EmailMessage.needs_review == needs_review)

    count_result = await db.execute(select(func.count()).select_from(query.subquery()))
    total = count_result.scalar_one()

    query = query.order_by(EmailMessage.email_date.desc().nullslast()).offset((page - 1) * page_size).limit(page_size)
    result = await db.execute(query)
    return EmailListResponse(items=result.scalars().all(), total=total, page=page, page_size=page_size)


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
    if not msg.is_read:
        msg.is_read = True
        await db.commit()
        await db.refresh(msg)
    return msg


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
    msg.matter_id = data.matter_id
    msg.needs_review = False
    msg.unknown_sender = False
    await db.commit()
    await db.refresh(msg)
    return msg


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
