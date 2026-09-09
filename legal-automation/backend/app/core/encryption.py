"""
Feldverschlüsselung (Fernet) + Blind-Index für durchsuchbare Felder.

Warum ein Blind-Index: Fernet ist probabilistisch — derselbe Klartext ergibt
jedes Mal einen anderen Ciphertext. Eine Gleichheitssuche (`WHERE email = ...`)
ist damit unmöglich. Der Blind-Index legt daneben einen deterministischen
HMAC des normalisierten Wertes ab; danach lässt sich exakt suchen, ohne dass
die Spalte den Klartext preisgibt.

Grenzen, bewusst und dokumentiert:
- Über einen Blind-Index ist nur EXAKTE Suche möglich, keine Teilstring-Suche.
- Wer den HMAC-Schlüssel und eine Kandidatenliste hat, kann Werte bestätigen
  (das ist bei jedem Blind-Index so). Er schützt gegen das Auslesen eines
  DB-Dumps, nicht gegen einen kompromittierten Anwendungsserver.
"""
from base64 import urlsafe_b64encode
from functools import lru_cache
import hashlib
import hmac
import unicodedata
from typing import Any

from cryptography.fernet import Fernet, MultiFernet, InvalidToken
from sqlalchemy import Text, TypeDecorator

from app.core.config import get_settings


@lru_cache(maxsize=1)
def _fernet_for(keys: tuple[str, ...]) -> MultiFernet:
    return MultiFernet([Fernet(k.encode()) for k in keys])


def _build_fernet() -> MultiFernet:
    # Über die Schlüssel als Cache-Key: eine Rotation (andere ENCRYPTION_KEYS)
    # erzeugt automatisch ein neues Objekt, statt das alte weiterzuverwenden.
    # Vorher wurde bei JEDEM encrypt/decrypt ein MultiFernet neu gebaut.
    return _fernet_for(tuple(get_settings().fernet_keys))


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


def encrypt_bytes(data: bytes) -> bytes:
    """Encrypt raw bytes (e.g. an email attachment before writing to disk)."""
    return _build_fernet().encrypt(data)


def decrypt_bytes(data: bytes) -> bytes:
    """Decrypt bytes previously produced by encrypt_bytes."""
    try:
        return _build_fernet().decrypt(data)
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


# --- Blind-Index ---

def normalize_for_index(value: str) -> str:
    """
    Normalisiert einen Wert vor der Indexbildung, damit Schreibvarianten
    denselben Index ergeben: Unicode-NFKC, Kleinschreibung, Whitespace weg.
    """
    return unicodedata.normalize("NFKC", value).strip().lower()


def blind_index(value: str | None) -> str | None:
    """
    Deterministischer HMAC-SHA256 (hex) des normalisierten Wertes — für exakte
    Suche auf verschlüsselten Spalten. None/leer bleibt None, damit fehlende
    Werte nicht alle denselben Index teilen.
    """
    if not value:
        return None
    key = get_settings().pii_index_key
    digest = hmac.new(key, normalize_for_index(value).encode("utf-8"), hashlib.sha256)
    return digest.hexdigest()


# --- SQLAlchemy-Typ ---

class EncryptedText(TypeDecorator):
    """
    Spaltentyp, der Werte transparent ver-/entschlüsselt.

    impl ist bewusst Text und nicht String(n): Fernet vergrößert den Wert um
    ~33 % plus 57 Byte Overhead. Eine VARCHAR(4096)-Spalte fasste damit nur
    rund 3.000 Zeichen Klartext — darüber brach der Commit mit
    "value too long for type character varying". Ein Längenargument wird aus
    Kompatibilität mit bestehenden Modelldefinitionen akzeptiert und für die
    fachliche Validierung aufbewahrt, aber nicht an die DDL weitergereicht.
    """

    impl = Text
    cache_ok = True

    def __init__(self, plaintext_length: int | None = None, *args: Any, **kwargs: Any) -> None:
        self.plaintext_length = plaintext_length
        super().__init__(*args, **kwargs)

    def process_bind_param(self, value: Any, dialect: Any) -> str | None:
        if value is None:
            return None
        return encrypt(str(value))

    def process_result_value(self, value: Any, dialect: Any) -> str | None:
        if value is None:
            return None
        return decrypt(value)
