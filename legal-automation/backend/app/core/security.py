import secrets
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

import pyotp
from jose import JWTError, jwt
from passlib.context import CryptContext

from app.core.config import get_settings

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto", bcrypt__rounds=12)

ALGORITHM = "RS256"


# --- Password ---

def hash_password(password: str) -> str:
    return pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    return pwd_context.verify(plain, hashed)


# Hash eines zufälligen Wertes, gegen den verifiziert wird, wenn der Nutzer
# nicht existiert. Einmal beim Import erzeugt, damit jeder Login denselben
# bcrypt-Aufwand hat — unabhängig davon, ob die Adresse bekannt ist.
_DUMMY_HASH = pwd_context.hash(secrets.token_urlsafe(32))


def verify_password_constant_time(plain: str, hashed: str | None) -> bool:
    """
    Wie verify_password, führt aber auch bei unbekanntem Nutzer eine echte
    bcrypt-Verifikation durch (gegen einen Dummy-Hash). Ohne das verrät die
    Antwortzeit, welche E-Mail-Adressen existieren (User-Enumeration).
    """
    if hashed is None:
        pwd_context.verify(plain, _DUMMY_HASH)
        return False
    try:
        return pwd_context.verify(plain, hashed)
    except ValueError:
        # Unlesbarer/abgeschnittener Hash in der DB darf keinen 500 auslösen.
        return False


# bcrypt verarbeitet nur die ersten 72 Byte. Ohne Obergrenze wären längere
# Passwörter stillschweigend gleichwertig — und sehr lange Eingaben ein
# billiger CPU-DoS über den Login-Endpunkt.
MAX_PASSWORD_BYTES = 72


def password_meets_policy(password: str) -> bool:
    """Minimum: 10 chars, 1 uppercase, 1 lowercase, 1 digit, 1 special char.
    Maximum: 72 Byte (bcrypt-Grenze)."""
    if len(password) < 10:
        return False
    if len(password.encode("utf-8")) > MAX_PASSWORD_BYTES:
        return False
    has_upper = any(c.isupper() for c in password)
    has_lower = any(c.islower() for c in password)
    has_digit = any(c.isdigit() for c in password)
    has_special = any(c in "!@#$%^&*()_+-=[]{}|;':\",./<>?" for c in password)
    return has_upper and has_lower and has_digit and has_special


# --- JWT ---

def create_access_token(subject: str | int, session_id: str, role: str, extra: dict | None = None) -> str:
    settings = get_settings()
    expire = datetime.now(UTC) + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    payload: dict[str, Any] = {
        "sub": str(subject),
        "sid": session_id,
        "role": role,
        "exp": expire,
        "iat": datetime.now(UTC),
        "type": "access",
    }
    if extra:
        payload.update(extra)
    return jwt.encode(payload, settings.jwt_private_key, algorithm=ALGORITHM)


def create_refresh_token(subject: str | int, session_id: str) -> str:
    settings = get_settings()
    expire = datetime.now(UTC) + timedelta(hours=settings.REFRESH_TOKEN_EXPIRE_HOURS)
    payload: dict[str, Any] = {
        "sub": str(subject),
        "sid": session_id,
        "exp": expire,
        "iat": datetime.now(UTC),
        "type": "refresh",
    }
    return jwt.encode(payload, settings.jwt_private_key, algorithm=ALGORITHM)


def decode_token(token: str) -> dict[str, Any]:
    settings = get_settings()
    try:
        return jwt.decode(token, settings.jwt_public_key, algorithms=[ALGORITHM])
    except JWTError as exc:
        raise ValueError(f"Invalid token: {exc}") from exc


# --- TOTP ---

def generate_totp_secret() -> str:
    return pyotp.random_base32()


def get_totp_uri(secret: str, user_email: str, issuer: str = "Legal Automation") -> str:
    totp = pyotp.TOTP(secret)
    return totp.provisioning_uri(name=user_email, issuer_name=issuer)


def verify_totp(secret: str, code: str) -> bool:
    totp = pyotp.TOTP(secret)
    return totp.verify(code, valid_window=1)


# --- Session tokens ---

def generate_session_id() -> str:
    return secrets.token_urlsafe(32)
