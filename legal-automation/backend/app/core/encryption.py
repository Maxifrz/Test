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


def encrypt_file(plaintext_path: str, ciphertext_path: str) -> None:
    """Encrypt a file on disk (e.g. original meeting audio → original.enc)."""
    f = _build_fernet()
    with open(plaintext_path, "rb") as src:
        token = f.encrypt(src.read())
    with open(ciphertext_path, "wb") as dst:
        dst.write(token)


def decrypt_file(ciphertext_path: str, plaintext_path: str) -> None:
    """Decrypt a previously encrypted file back to plaintext on disk."""
    f = _build_fernet()
    with open(ciphertext_path, "rb") as src:
        try:
            data = f.decrypt(src.read())
        except InvalidToken as exc:
            raise ValueError("File decryption failed — key may have been rotated") from exc
    with open(plaintext_path, "wb") as dst:
        dst.write(data)


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
