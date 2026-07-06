"""
Integrationstests gegen echtes PostgreSQL (mit pgvector) + Redis.

Aktivierung:  RUN_INTEGRATION=1 pytest tests/integration/
Ohne RUN_INTEGRATION=1 werden alle Tests übersprungen (Unit-Läufe bleiben schnell).

Voraussetzungen (lokal oder CI-Services):
  - PostgreSQL 16 mit pgvector, erreichbar über DATABASE_URL(_SYNC)
    (Default: postgresql://legal_test:legal_test@127.0.0.1:5432/legal_test;
    der Nutzer braucht Rechte für CREATE EXTENSION vector + DROP SCHEMA)
  - Redis über REDIS_URL (Default: redis://127.0.0.1:6379/9)

Das Schema wird pro Testlauf komplett neu aufgebaut (DROP SCHEMA + alembic
upgrade head) — damit ist jeder Lauf zugleich ein echter Migrationstest.
"""
import os
import pathlib
import subprocess
import tempfile

import pytest

RUN = os.environ.get("RUN_INTEGRATION") == "1"
BACKEND_DIR = pathlib.Path(__file__).resolve().parents[2]


def pytest_collection_modifyitems(config, items):
    if RUN:
        return
    skip = pytest.mark.skip(reason="RUN_INTEGRATION=1 + Postgres/Redis erforderlich")
    for item in items:
        item.add_marker(skip)


if RUN:
    # Umgebung MUSS vor dem ersten `from app...`-Import stehen (Settings lesen env).
    _tmp = pathlib.Path(tempfile.mkdtemp(prefix="la-integration-"))

    if "JWT_PRIVATE_KEY_PATH" not in os.environ:
        from cryptography.hazmat.primitives import serialization
        from cryptography.hazmat.primitives.asymmetric import rsa

        key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        (_tmp / "jwt_private.pem").write_bytes(
            key.private_bytes(
                serialization.Encoding.PEM,
                serialization.PrivateFormat.PKCS8,
                serialization.NoEncryption(),
            )
        )
        (_tmp / "jwt_public.pem").write_bytes(
            key.public_key().public_bytes(
                serialization.Encoding.PEM, serialization.PublicFormat.SubjectPublicKeyInfo
            )
        )
        os.environ["JWT_PRIVATE_KEY_PATH"] = str(_tmp / "jwt_private.pem")
        os.environ["JWT_PUBLIC_KEY_PATH"] = str(_tmp / "jwt_public.pem")

    if "ENCRYPTION_KEYS" not in os.environ:
        from cryptography.fernet import Fernet

        os.environ["ENCRYPTION_KEYS"] = Fernet.generate_key().decode()

    os.environ.setdefault(
        "DATABASE_URL", "postgresql+asyncpg://legal_test:legal_test@127.0.0.1:5432/legal_test"
    )
    os.environ.setdefault(
        "DATABASE_URL_SYNC", "postgresql://legal_test:legal_test@127.0.0.1:5432/legal_test"
    )
    os.environ.setdefault("REDIS_URL", "redis://127.0.0.1:6379/9")
    os.environ.setdefault("CELERY_BROKER_URL", os.environ["REDIS_URL"])
    os.environ.setdefault("CELERY_RESULT_BACKEND", os.environ["REDIS_URL"])
    os.environ.setdefault("SECRET_KEY", "integration-test-secret-key-0123456789")
    os.environ.setdefault("ENVIRONMENT", "test")
    os.environ.setdefault("ALLOWED_HOSTS", "localhost")
    os.environ.setdefault("STORAGE_ROOT", str(_tmp / "storage"))
    (pathlib.Path(os.environ["STORAGE_ROOT"])).mkdir(parents=True, exist_ok=True)


@pytest.fixture(scope="session")
def migrated_db():
    """Schema frisch aufbauen: DROP SCHEMA → alembic upgrade head → Redis-Testdb leeren."""
    from sqlalchemy import create_engine, text

    engine = create_engine(os.environ["DATABASE_URL_SYNC"], isolation_level="AUTOCOMMIT")
    with engine.connect() as conn:
        conn.execute(text("DROP SCHEMA public CASCADE"))
        conn.execute(text("CREATE SCHEMA public"))
    engine.dispose()

    subprocess.run(
        ["alembic", "upgrade", "head"], cwd=BACKEND_DIR, check=True, env=os.environ.copy()
    )

    import redis as redis_sync

    redis_sync.Redis.from_url(os.environ["REDIS_URL"]).flushdb()
    yield


import pytest_asyncio  # noqa: E402


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def client(migrated_db):
    from httpx import ASGITransport, AsyncClient

    from app.main import app

    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://localhost") as c:
        yield c

    # Pool sauber schließen, solange der Session-Loop noch lebt (sonst GC-Warnungen)
    from app.core.deps import engine

    await engine.dispose()


@pytest_asyncio.fixture(scope="session", loop_scope="session")
async def session_factory(migrated_db):
    """Direkter DB-Zugriff für Test-Setup (z. B. Zweitnutzer anlegen)."""
    from app.core.deps import AsyncSessionLocal

    return AsyncSessionLocal
