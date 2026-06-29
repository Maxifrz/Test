from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel, field_validator

RANKS = {"insolvenz_38", "nachrangig_39", "absonderung", "masseverbindlichkeit"}
STATUSES = {"angemeldet", "geprueft", "festgestellt", "bestritten"}
DIST_TYPES = {"abschlag", "schluss"}


class ClaimCreate(BaseModel):
    matter_id: int
    creditor_name: str
    claim_amount: Decimal
    rank: str = "insolvenz_38"
    creditor_email: str | None = None
    creditor_address: str | None = None
    creditor_reference: str | None = None
    claim_reason: str | None = None

    @field_validator("rank")
    @classmethod
    def valid_rank(cls, v: str) -> str:
        if v not in RANKS:
            raise ValueError(f"rank must be one of {sorted(RANKS)}")
        return v


class ClaimUpdate(BaseModel):
    status: str | None = None
    established_amount: Decimal | None = None
    rank: str | None = None
    dispute_reason: str | None = None
    claim_reason: str | None = None

    @field_validator("status")
    @classmethod
    def valid_status(cls, v: str | None) -> str | None:
        if v is not None and v not in STATUSES:
            raise ValueError(f"status must be one of {sorted(STATUSES)}")
        return v

    @field_validator("rank")
    @classmethod
    def valid_rank(cls, v: str | None) -> str | None:
        if v is not None and v not in RANKS:
            raise ValueError(f"rank must be one of {sorted(RANKS)}")
        return v


class ClaimResponse(BaseModel):
    id: int
    matter_id: int
    claim_number: int | None
    creditor_name: str
    creditor_email: str | None
    creditor_address: str | None
    creditor_reference: str | None
    claim_amount: Decimal
    established_amount: Decimal | None
    claim_reason: str | None
    rank: str
    status: str
    dispute_reason: str | None
    source: str
    filed_at: datetime

    model_config = {"from_attributes": True}


class ClaimTotals(BaseModel):
    count: int
    sum_angemeldet: Decimal
    sum_festgestellt: Decimal
    count_festgestellt: int
    count_bestritten: int


class ClaimTableResponse(BaseModel):
    items: list[ClaimResponse]
    totals: ClaimTotals


class DistributionRequest(BaseModel):
    matter_id: int
    distributable_amount: Decimal
    distribution_type: str = "abschlag"
    persist: bool = False

    @field_validator("distribution_type")
    @classmethod
    def valid_type(cls, v: str) -> str:
        if v not in DIST_TYPES:
            raise ValueError(f"distribution_type must be one of {sorted(DIST_TYPES)}")
        return v


class DistributionItemResponse(BaseModel):
    claim_id: int
    established_amount: Decimal
    amount: Decimal
    quote_pct: Decimal


class DistributionResponse(BaseModel):
    distribution_id: int | None = None
    matter_id: int
    distributable: Decimal
    total_38: Decimal
    total_39: Decimal
    quote_38_pct: Decimal
    distributed_sum: Decimal
    remainder: Decimal
    items: list[DistributionItemResponse]


class PortalEnableResponse(BaseModel):
    matter_id: int
    creditor_portal_token: str
    submit_path: str


class PublicClaimSubmit(BaseModel):
    creditor_name: str
    claim_amount: Decimal
    creditor_email: str | None = None
    creditor_address: str | None = None
    creditor_reference: str | None = None
    claim_reason: str | None = None
