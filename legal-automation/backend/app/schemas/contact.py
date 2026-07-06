from datetime import datetime

from pydantic import BaseModel, EmailStr, Field


class PublicContactCreate(BaseModel):
    """Eingabe des öffentlichen Kontaktformulars (Untrusted)."""

    name: str = Field(min_length=1, max_length=200)
    email: EmailStr
    phone: str | None = Field(default=None, max_length=50)
    standort: str | None = Field(default=None, max_length=100)
    rolle: str | None = Field(default=None, max_length=100)
    message: str = Field(min_length=10, max_length=10_000)
    consent: bool  # DSGVO-Checkbox; ohne True wird abgelehnt (422 in der Route)
    # Honeypot: für Menschen unsichtbares Feld — gefüllt = Bot, Anfrage wird
    # kommentarlos verworfen (gleiche Antwort, kein Speichern).
    firma: str | None = None


class ContactRequestResponse(BaseModel):
    id: int
    name: str
    email: str
    phone: str | None
    standort: str | None
    rolle: str | None
    message: str
    consent_at: datetime
    status: str
    created_at: datetime

    model_config = {"from_attributes": True}


class ContactRequestUpdate(BaseModel):
    status: str = Field(pattern="^(neu|erledigt)$")
