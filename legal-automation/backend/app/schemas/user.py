"""Schemas der Benutzerverwaltung (Admin) und des eigenen Profils."""
from datetime import datetime

from pydantic import BaseModel, ConfigDict, EmailStr, field_validator

from app.core.rbac import Role

ROLES = {r.value for r in Role}


class UserResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    email: str
    full_name: str
    role: str
    is_active: bool
    totp_enabled: bool
    must_change_password: bool
    phone: str | None = None
    last_login: datetime | None = None
    created_at: datetime | None = None


class UserListResponse(BaseModel):
    items: list[UserResponse]
    total: int
    page: int
    page_size: int


class UserCreate(BaseModel):
    # EmailStr an der Nutzeranlage (anders als beim Login): hier soll eine
    # Tippfehler-Adresse gar nicht erst in die DB kommen.
    email: EmailStr
    full_name: str
    role: str
    phone: str | None = None

    @field_validator("role")
    @classmethod
    def valid_role(cls, v: str) -> str:
        if v not in ROLES:
            raise ValueError(f"role must be one of {sorted(ROLES)}")
        return v

    @field_validator("full_name")
    @classmethod
    def name_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("full_name darf nicht leer sein")
        return v.strip()


class UserCreateResponse(BaseModel):
    user: UserResponse
    # Einmal-Passwort; wird nur bei der Anlage zurückgegeben und nirgends
    # gespeichert. Der Nutzer muss es beim ersten Login wechseln.
    initial_password: str


class UserUpdate(BaseModel):
    full_name: str | None = None
    role: str | None = None
    phone: str | None = None
    is_active: bool | None = None
    signature_html: str | None = None

    @field_validator("role")
    @classmethod
    def valid_role(cls, v: str | None) -> str | None:
        if v is not None and v not in ROLES:
            raise ValueError(f"role must be one of {sorted(ROLES)}")
        return v


class SelfUpdate(BaseModel):
    """Was ein Nutzer an sich selbst ändern darf (nicht: Rolle, Aktiv-Status)."""

    full_name: str | None = None
    phone: str | None = None
    signature_html: str | None = None


class PasswordResetResponse(BaseModel):
    user_id: int
    initial_password: str
    sessions_revoked: int
