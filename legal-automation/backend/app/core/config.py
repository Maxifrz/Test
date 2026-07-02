from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # App
    APP_NAME: str = "Legal Automation Platform"
    ENVIRONMENT: Literal["development", "production", "test"] = "production"
    LOG_LEVEL: str = "INFO"
    ALLOWED_HOSTS: list[str] = ["localhost"]
    BUNDESLAND: str = "BY"

    # Database — no default, fails on startup if missing
    DATABASE_URL: str
    DATABASE_URL_SYNC: str

    # Redis — no default
    REDIS_URL: str
    CELERY_BROKER_URL: str
    CELERY_RESULT_BACKEND: str

    # Security — no default, fails on startup if missing
    SECRET_KEY: str
    JWT_PRIVATE_KEY_PATH: Path
    JWT_PUBLIC_KEY_PATH: Path
    ENCRYPTION_KEYS: str  # comma-separated Fernet keys

    # Session
    SESSION_TIMEOUT_MINUTES: int = 30
    MAX_FAILED_LOGINS: int = 5
    LOCKOUT_MINUTES: int = 30
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 15
    REFRESH_TOKEN_EXPIRE_HOURS: int = 8

    # Storage
    STORAGE_ROOT: Path = Path("/storage")

    # Email
    IMAP_HOST: str = ""
    IMAP_PORT: int = 993
    IMAP_SSL: bool = True
    IMAP_USERNAME: str = ""
    IMAP_PASSWORD: str = ""
    SMTP_HOST: str = ""
    SMTP_PORT: int = 587
    SMTP_TLS: bool = True
    SMTP_USERNAME: str = ""
    SMTP_PASSWORD: str = ""
    SMTP_FROM_NAME: str = "Kanzlei"
    SMTP_FROM_EMAIL: str = ""

    # Transcription
    WHISPER_MODEL: str = "large-v3"
    WHISPER_DEVICE: str = "cpu"
    WHISPER_COMPUTE_TYPE: str = "int8"
    DIARIZATION_ENABLED: bool = True
    PYANNOTE_PIPELINE: str = "pyannote/speaker-diarization-3.1"

    # DSGVO
    DEFAULT_MATTER_RETENTION_YEARS: int = 10
    DSGVO_CONTACT_EMAIL: str = ""

    @field_validator("ALLOWED_HOSTS", mode="before")
    @classmethod
    def parse_allowed_hosts(cls, v: str | list) -> list[str]:
        if isinstance(v, str):
            return [h.strip() for h in v.split(",")]
        return v

    @field_validator("SECRET_KEY")
    @classmethod
    def secret_key_must_be_set(cls, v: str) -> str:
        if not v:
            raise ValueError("SECRET_KEY must be set — generate with: openssl rand -hex 32")
        return v

    @field_validator("ENCRYPTION_KEYS")
    @classmethod
    def encryption_keys_must_be_set(cls, v: str) -> str:
        if not v:
            raise ValueError(
                "ENCRYPTION_KEYS must be set — generate with: "
                "python -c \"from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())\""
            )
        return v

    # JWT-Key-Existenz wird LAZY geprüft (erst bei Zugriff), nicht beim Settings()-Init:
    # Celery-Worker/Beat brauchen keine JWT-Keys und dürfen ohne die Mounts starten.
    # Das Backend prüft beide Keys explizit im Lifespan (Fail-fast beim Boot).
    @property
    def jwt_private_key(self) -> str:
        if not self.JWT_PRIVATE_KEY_PATH.exists():
            raise ValueError(
                f"JWT private key not found at {self.JWT_PRIVATE_KEY_PATH}. "
                "Generate with: openssl genrsa -out backend/jwt_private.pem 4096"
            )
        return self.JWT_PRIVATE_KEY_PATH.read_text()

    @property
    def jwt_public_key(self) -> str:
        if not self.JWT_PUBLIC_KEY_PATH.exists():
            raise ValueError(
                f"JWT public key not found at {self.JWT_PUBLIC_KEY_PATH}. "
                "Generate with: openssl rsa -in backend/jwt_private.pem -pubout -out backend/jwt_public.pem"
            )
        return self.JWT_PUBLIC_KEY_PATH.read_text()

    @property
    def fernet_keys(self) -> list[str]:
        return [k.strip() for k in self.ENCRYPTION_KEYS.split(",") if k.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
