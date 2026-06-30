from datetime import datetime

from pydantic import BaseModel


class ProcessingRecordResponse(BaseModel):
    id: int
    name: str
    purpose: str
    legal_basis: str
    data_categories: list
    data_subjects: str | None
    recipients: str | None
    retention: str | None
    tom: str | None
    is_active: bool

    model_config = {"from_attributes": True}


class ProcessingRecordCreate(BaseModel):
    name: str
    purpose: str
    legal_basis: str
    data_categories: list[str] = []
    data_subjects: str | None = None
    recipients: str | None = None
    retention: str | None = None
    tom: str | None = None


class RetentionPolicyResponse(BaseModel):
    id: int
    name: str
    matter_type: str | None
    retention_years: int
    legal_basis: str | None
    is_active: bool

    model_config = {"from_attributes": True}


class RetentionPolicyCreate(BaseModel):
    name: str
    matter_type: str | None = None
    retention_years: int
    legal_basis: str | None = None


class ErasureEligibilityResponse(BaseModel):
    allowed: bool
    blocking_reasons: list[str]


class ErasureRequestCreate(BaseModel):
    client_id: int
    reason: str | None = None


class ErasureRequestResponse(BaseModel):
    id: int
    client_id: int
    requested_by_id: int
    requested_at: datetime
    status: str
    reason: str | None
    blocking_reasons: list | None
    decided_at: datetime | None
    executed_at: datetime | None
    certificate_path: str | None

    model_config = {"from_attributes": True}


class DataExportResponse(BaseModel):
    id: int
    client_id: int
    status: str
    token: str
    expires_at: datetime | None
    download_path: str | None = None

    model_config = {"from_attributes": True}


class AdminOverviewResponse(BaseModel):
    active_sessions: int
    locked_users: int
    users_total: int
    users_with_2fa: int
    open_erasure_requests: int
    blocked_erasure_requests: int
    matters_past_retention: int
