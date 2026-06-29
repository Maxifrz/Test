from datetime import datetime

from pydantic import BaseModel, EmailStr, field_validator


class ClientCreate(BaseModel):
    first_name: str
    last_name: str
    company_name: str | None = None
    is_company: bool = False
    email: str | None = None
    phone: str | None = None
    address_line1: str | None = None
    address_line2: str | None = None
    postal_code: str | None = None
    city: str | None = None
    country: str = "DE"
    date_of_birth: str | None = None
    tax_id: str | None = None
    notes: str | None = None
    dsgvo_legal_basis: str | None = "contract"

    @field_validator("country")
    @classmethod
    def country_uppercase(cls, v: str) -> str:
        return v.upper()


class ClientUpdate(BaseModel):
    first_name: str | None = None
    last_name: str | None = None
    company_name: str | None = None
    email: str | None = None
    phone: str | None = None
    address_line1: str | None = None
    address_line2: str | None = None
    postal_code: str | None = None
    city: str | None = None
    country: str | None = None
    date_of_birth: str | None = None
    tax_id: str | None = None
    notes: str | None = None
    dsgvo_legal_basis: str | None = None


class ClientResponse(BaseModel):
    id: int
    client_number: str
    first_name: str
    last_name: str
    company_name: str | None
    is_company: bool
    email: str | None
    phone: str | None
    address_line1: str | None
    address_line2: str | None
    postal_code: str | None
    city: str | None
    country: str
    date_of_birth: str | None
    tax_id: str | None
    notes: str | None
    dsgvo_consent_given_at: datetime | None
    dsgvo_legal_basis: str | None
    created_at: datetime
    display_name: str

    model_config = {"from_attributes": True}


class ClientListItem(BaseModel):
    id: int
    client_number: str
    display_name: str
    email: str | None
    phone: str | None
    city: str | None
    created_at: datetime

    model_config = {"from_attributes": True}


class ClientListResponse(BaseModel):
    items: list[ClientListItem]
    total: int
    page: int
    page_size: int
