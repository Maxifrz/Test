from datetime import date, datetime
from decimal import Decimal

from pydantic import BaseModel, field_validator

ACCOUNT_TYPES = {"sonderkonto", "anderkonto", "treuhand"}


class MassAccountCreate(BaseModel):
    matter_id: int
    iban: str
    bic: str | None = None
    bank_name: str | None = None
    account_label: str | None = None
    account_type: str = "sonderkonto"
    opening_balance: Decimal = Decimal("0")

    @field_validator("account_type")
    @classmethod
    def valid_type(cls, v: str) -> str:
        if v not in ACCOUNT_TYPES:
            raise ValueError(f"account_type must be one of {sorted(ACCOUNT_TYPES)}")
        return v


class MassAccountResponse(BaseModel):
    id: int
    matter_id: int
    iban: str
    bic: str | None
    bank_name: str | None
    account_label: str | None
    account_type: str
    opening_balance: Decimal
    currency: str
    is_active: bool
    created_at: datetime

    model_config = {"from_attributes": True}


class MassAccountBalance(BaseModel):
    account_id: int
    matter_id: int
    opening_balance: Decimal
    current_balance: Decimal
    currency: str


class TransactionResponse(BaseModel):
    id: int
    mass_account_id: int
    matter_id: int
    import_batch_id: int | None
    booking_date: date | None
    value_date: date | None
    amount: Decimal
    direction: str
    currency: str
    purpose: str | None
    counterparty_name: str | None
    counterparty_iban: str | None
    category: str
    is_reconciled: bool

    model_config = {"from_attributes": True}


class TransactionListResponse(BaseModel):
    items: list[TransactionResponse]
    total: int
    page: int
    page_size: int


class TransactionUpdate(BaseModel):
    category: str | None = None
    mass_account_id: int | None = None


class ImportReportResponse(BaseModel):
    batch_id: int
    num_transactions: int
    num_assigned: int
    num_unassigned: int
    num_duplicates: int
    reconciled: bool
    statement_closing: Decimal | None
    computed_closing: Decimal | None


class ImportBatchResponse(BaseModel):
    id: int
    mass_account_id: int | None
    filename: str
    format: str
    num_transactions: int
    num_assigned: int
    num_unassigned: int
    num_duplicates: int
    reconciled: bool
    statement_closing_balance: Decimal | None
    computed_closing_balance: Decimal | None
    created_at: datetime

    model_config = {"from_attributes": True}
