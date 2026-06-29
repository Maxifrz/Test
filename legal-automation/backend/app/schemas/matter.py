from datetime import datetime

from pydantic import BaseModel, field_validator

from app.core.rbac import MatterRole

MATTER_TYPES = {"civil", "criminal", "family", "labor", "admin", "tax", "ip", "other"}
MATTER_STATUSES = {"open", "active", "pending_closing", "closed", "archived"}


class MatterCreate(BaseModel):
    title: str
    matter_type: str
    client_id: int
    lead_anwalt_id: int
    court_file_ref: str | None = None
    court_name: str | None = None
    opposing_party: str | None = None
    opposing_counsel: str | None = None
    statute_of_limitations: datetime | None = None
    retention_years: int = 6
    description: str | None = None

    @field_validator("matter_type")
    @classmethod
    def valid_matter_type(cls, v: str) -> str:
        if v not in MATTER_TYPES:
            raise ValueError(f"matter_type must be one of {sorted(MATTER_TYPES)}")
        return v

    @field_validator("retention_years")
    @classmethod
    def min_retention(cls, v: int) -> int:
        if v < 6:
            raise ValueError("Minimum retention is 6 years per BRAO §50")
        return v


class MatterUpdate(BaseModel):
    title: str | None = None
    matter_type: str | None = None
    status: str | None = None
    lead_anwalt_id: int | None = None
    court_file_ref: str | None = None
    court_name: str | None = None
    opposing_party: str | None = None
    opposing_counsel: str | None = None
    statute_of_limitations: datetime | None = None
    retention_years: int | None = None
    description: str | None = None

    @field_validator("status")
    @classmethod
    def valid_status(cls, v: str | None) -> str | None:
        if v is not None and v not in MATTER_STATUSES:
            raise ValueError(f"status must be one of {sorted(MATTER_STATUSES)}")
        return v

    @field_validator("retention_years")
    @classmethod
    def min_retention(cls, v: int | None) -> int | None:
        if v is not None and v < 6:
            raise ValueError("Minimum retention is 6 years per BRAO §50")
        return v


class MatterAccessGrant(BaseModel):
    user_id: int
    matter_role: MatterRole


class MatterAccessRevoke(BaseModel):
    user_id: int


class MatterAccessResponse(BaseModel):
    id: int
    user_id: int
    matter_id: int
    matter_role: str
    granted_by_id: int
    granted_at: datetime
    revoked_at: datetime | None

    model_config = {"from_attributes": True}


class MatterResponse(BaseModel):
    id: int
    matter_number: str
    title: str
    matter_type: str
    status: str
    client_id: int
    lead_anwalt_id: int
    created_by_id: int
    court_file_ref: str | None
    court_name: str | None
    opposing_party: str | None
    opposing_counsel: str | None
    opened_at: datetime
    closed_at: datetime | None
    statute_of_limitations: datetime | None
    retention_years: int
    description: str | None
    created_at: datetime

    model_config = {"from_attributes": True}


class MatterListItem(BaseModel):
    id: int
    matter_number: str
    title: str
    matter_type: str
    status: str
    client_id: int
    lead_anwalt_id: int
    opened_at: datetime
    created_at: datetime

    model_config = {"from_attributes": True}


class MatterListResponse(BaseModel):
    items: list[MatterListItem]
    total: int
    page: int
    page_size: int
