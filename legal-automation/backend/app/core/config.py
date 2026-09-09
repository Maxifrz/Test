from functools import lru_cache
from pathlib import Path
from typing import Annotated, Literal

from pydantic import field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # App
    APP_NAME: str = "Legal Automation Platform"
    ENVIRONMENT: Literal["development", "production", "test"] = "production"
    LOG_LEVEL: str = "INFO"
    # NoDecode: pydantic-settings würde Listen-Felder sonst als JSON parsen und
    # am Komma-Format aus .env ("localhost,127.0.0.1") beim Boot scheitern —
    # das Splitting übernimmt der field_validator unten.
    ALLOWED_HOSTS: Annotated[list[str], NoDecode] = ["localhost"]
    BUNDESLAND: str = "BY"
    # Netze, aus denen X-Forwarded-For geglaubt wird (Reverse-Proxy). Alles
    # andere darf seine IP im Audit-Log nicht selbst bestimmen.
    TRUSTED_PROXY_PREFIXES: Annotated[list[str], NoDecode] = ["172.20.", "127.0.0.1"]

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
    # HMAC-Schlüssel für Blind-Indizes auf verschlüsselten, durchsuchbaren
    # Feldern (z. B. clients.email). Leer = aus SECRET_KEY abgeleitet.
    PII_INDEX_KEY: str = ""

    # Redis
    REDIS_MAX_CONNECTIONS: int = 50

    # Session
    SESSION_TIMEOUT_MINUTES: int = 30
    # Wie oft user_sessions.last_active höchstens geschrieben wird. Vorher lief
    # bei JEDEM authentifizierten Request ein UPDATE über den Hot Path.
    SESSION_ACTIVITY_UPDATE_SECONDS: int = 60
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
    SMTP_TIMEOUT_SECONDS: int = 30
    # Outbox: Zustellversuche, bevor eine Mail als endgültig gescheitert gilt
    SMTP_MAX_ATTEMPTS: int = 5
    # Empfänger pro Nachricht; darüber wird in Teilnachrichten zerlegt
    EMAIL_MAX_RECIPIENTS_PER_MESSAGE: int = 200
    # Anhangsgrenzen je eingehender Mail (vorher nur pro Anhang begrenzt)
    EMAIL_MAX_ATTACHMENTS: int = 50
    EMAIL_MAX_TOTAL_ATTACHMENT_BYTES: int = 100 * 1024 * 1024

    # Transcription
    WHISPER_MODEL: str = "large-v3"
    WHISPER_DEVICE: str = "cpu"
    WHISPER_COMPUTE_TYPE: str = "int8"
    DIARIZATION_ENABLED: bool = True
    PYANNOTE_PIPELINE: str = "pyannote/speaker-diarization-3.1"

    # DSGVO
    DEFAULT_MATTER_RETENTION_YEARS: int = 10
    DSGVO_CONTACT_EMAIL: str = ""

    # KI-Rechtsrecherche (V3.0 GraphRAG) — vollständig lokal via Ollama
    KI_ENABLED: bool = False
    OLLAMA_BASE_URL: str = "http://ollama:11434"
    KI_LLM_MODEL: str = "phi3:mini"          # CPU-Profil; GPU: mistral / llama3.1
    KI_EMBED_MODEL: str = "nomic-embed-text"
    KI_EMBED_DIM: int = 768                   # muss zu Migration 0010 passen
    KI_RETRIEVAL_TOP_K: int = 8
    KI_MIN_GROUNDING_SCORE: float = 0.35
    KI_MAX_CONTEXT_CHARS: int = 6000
    # Zweite Stufe: Kandidaten werden nach dem Hybrid-Retrieval neu bewertet
    KI_RERANK_ENABLED: bool = True
    KI_RERANK_CANDIDATES: int = 24
    # Mindest-Überlappung zwischen Antwortsatz und zitierter Quelle, damit die
    # Antwort als belegt gilt (0 = nur Marker prüfen, wie bisher)
    KI_MIN_CLAIM_SUPPORT: float = 0.18

    # Dokumente / OCR
    OCR_ENABLED: bool = True
    OCR_LANGUAGES: str = "deu+eng"
    DOCUMENT_MAX_BYTES: int = 100 * 1024 * 1024

    # Betriebsmetriken (/api/metrics, Prometheus-Textformat)
    METRICS_ENABLED: bool = True
    # Leer = nur aus TRUSTED_PROXY_PREFIXES erreichbar; sonst Bearer-Token
    METRICS_TOKEN: str = ""

    @field_validator("ALLOWED_HOSTS", "TRUSTED_PROXY_PREFIXES", mode="before")
    @classmethod
    def parse_csv_list(cls, v: str | list) -> list[str]:
        if isinstance(v, str):
            return [h.strip() for h in v.split(",") if h.strip()]
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

    @property
    def trusted_proxy_prefixes(self) -> list[str]:
        return self.TRUSTED_PROXY_PREFIXES

    @property
    def pii_index_key(self) -> bytes:
        """
        Schlüssel für die Blind-Index-HMACs. Fällt auf einen aus SECRET_KEY
        abgeleiteten Wert zurück, damit Bestandsinstallationen ohne
        PII_INDEX_KEY weiterlaufen — für Neuinstallationen erzeugt setup.sh
        einen eigenen Schlüssel (Rotation des einen ohne den anderen).
        """
        raw = self.PII_INDEX_KEY or f"pii-index:{self.SECRET_KEY}"
        return raw.encode()


@lru_cache
def get_settings() -> Settings:
    return Settings()
