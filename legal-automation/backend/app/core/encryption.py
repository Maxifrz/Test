from base64 import urlsafe_b64encode
from typing import Any

from cryptography.fernet import Fernet, MultiFernet, InvalidToken
from sqlalchemy import String, TypeDecorator

from app.core.config import get_settings


def _build_fernet() -> MultiFernet:
    settings = get_settings()
    keys = [Fernet(k.encode()) for k in settings.fernet_keys]
    return MultiFernet(keys)


def encrypt(plaintext: str) -> str:
    """Encrypt a string. Returns base64-encoded ciphertext."""
    if not plaintext:
        return plaintext
    f = _build_fernet()
    return f.encrypt(plaintext.encode()).decode()


def decrypt(ciphertext: str) -> str:
    """Decrypt a previously encrypted string."""
    if not ciphertext:
        return ciphertext
    f = _build_fernet()
    try:
        return f.decrypt(ciphertext.encode()).decode()
    except InvalidToken as exc:
        raise ValueError("Decryption failed — key may have been rotated") from exc


class EncryptedText(TypeDecorator):
    """SQLAlchemy column type that transparently encrypts/decrypts string values."""

    impl = String
    cache_ok = True

    def process_bind_param(self, value: Any, dialect: Any) -> str | None:
        if value is None:
            return None
        return encrypt(str(value))

    def process_result_value(self, value: Any, dialect: Any) -> str | None:
        if value is None:
            return None
        return decrypt(value)
