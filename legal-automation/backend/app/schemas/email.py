from datetime import datetime

from pydantic import BaseModel


class EmailListItem(BaseModel):
    id: int
    direction: str
    from_address: str
    subject: str | None
    matter_id: int | None
    client_id: int | None
    is_read: bool
    needs_review: bool
    is_confidential: bool
    unknown_sender: bool
    email_date: datetime | None

    model_config = {"from_attributes": True}


class EmailDetail(EmailListItem):
    to_addresses: list
    cc_addresses: list | None
    body_text: str | None
    body_html: str | None
    in_reply_to: str | None
    thread_key: str | None
    delivery_status: str | None

    model_config = {"from_attributes": True}


class EmailAttachmentResponse(BaseModel):
    id: int
    filename: str
    content_type: str | None
    size_bytes: int

    model_config = {"from_attributes": True}


class EmailListResponse(BaseModel):
    items: list[EmailListItem]
    total: int
    page: int
    page_size: int


class EmailFileRequest(BaseModel):
    matter_id: int


# --- Rules ---

class EmailRuleCreate(BaseModel):
    name: str
    priority: int = 100
    conditions: dict
    actions: dict
    is_active: bool = True


class EmailRuleUpdate(BaseModel):
    name: str | None = None
    priority: int | None = None
    conditions: dict | None = None
    actions: dict | None = None
    is_active: bool | None = None


class EmailRuleResponse(BaseModel):
    id: int
    name: str
    priority: int
    conditions: dict
    actions: dict
    is_active: bool

    model_config = {"from_attributes": True}


# --- Templates ---

class EmailTemplateCreate(BaseModel):
    name: str
    category: str | None = None
    subject_template: str
    body_template: str
    variables_doc: dict | None = None
    is_active: bool = True


class EmailTemplateUpdate(BaseModel):
    name: str | None = None
    category: str | None = None
    subject_template: str | None = None
    body_template: str | None = None
    variables_doc: dict | None = None
    is_active: bool | None = None


class EmailTemplateResponse(BaseModel):
    id: int
    name: str
    category: str | None
    subject_template: str
    body_template: str
    variables_doc: dict | None
    is_active: bool

    model_config = {"from_attributes": True}


# --- Send / Preview ---

class EmailSendRequest(BaseModel):
    to_addresses: list[str]
    subject: str
    body_text: str
    body_html: str | None = None
    matter_id: int | None = None
    client_id: int | None = None


class EmailPreviewRequest(BaseModel):
    template_id: int
    context: dict


class EmailPreviewResponse(BaseModel):
    subject: str
    body: str
