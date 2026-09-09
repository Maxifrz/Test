"""
Benutzerverwaltung: Anlage, Änderung, Passwort-/2FA-Reset, Deaktivierung.

Grundsätze:
- Passwörter werden nie im Klartext gespeichert und nur EINMAL (bei Anlage
  bzw. Reset) an den anlegenden Admin zurückgegeben.
- Jede Anlage/Zurücksetzung erzwingt Passwortwechsel beim nächsten Login
  (`must_change_password`); rollenabhängig zusätzlich 2FA-Einrichtung.
- Statusändernde Eingriffe (Deaktivierung, Passwort-Reset, 2FA-Reset)
  widerrufen alle offenen Sessions des Nutzers — sonst liefe ein gestohlenes
  Token weiter.
"""
from __future__ import annotations

import secrets
import string
from datetime import UTC, datetime

from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.rbac import Role
from app.core.security import hash_password, password_meets_policy
from app.models.user import User, UserSession

# Zeichenvorrat ohne verwechselbare Glyphen (0/O, 1/l/I) — die Passwörter
# werden vorgelesen oder abgetippt.
_ALPHABET_LOWER = "abcdefghijkmnopqrstuvwxyz"
_ALPHABET_UPPER = "ABCDEFGHJKLMNPQRSTUVWXYZ"
_DIGITS = "23456789"
_SPECIAL = "!@#$%*+-="


def generate_initial_password(length: int = 16) -> str:
    """Erzeugt ein Einmal-Passwort, das garantiert die Passwortrichtlinie erfüllt."""
    if length < 10:
        raise ValueError("length muss mindestens 10 sein (Passwortrichtlinie)")
    pools = [_ALPHABET_LOWER, _ALPHABET_UPPER, _DIGITS, _SPECIAL]
    # Je ein Zeichen aus jeder Klasse, Rest frei — danach mischen.
    chars = [secrets.choice(p) for p in pools]
    all_chars = "".join(pools)
    chars += [secrets.choice(all_chars) for _ in range(length - len(pools))]
    rng = secrets.SystemRandom()
    rng.shuffle(chars)
    pwd = "".join(chars)
    # Sicherheitsnetz: die Richtlinie ist die Autorität, nicht diese Konstruktion.
    if not password_meets_policy(pwd):  # pragma: no cover — konstruktiv unerreichbar
        return generate_initial_password(length)
    return pwd


def normalize_email(email: str) -> str:
    """E-Mail-Adressen werden case-insensitiv geführt (RFC-konform für den Domain-
    Teil, praktisch auch für den Local-Teil bei allen gängigen Providern)."""
    return email.strip().lower()


async def get_user(db: AsyncSession, user_id: int) -> User | None:
    result = await db.execute(
        select(User).where(User.id == user_id, User.deleted_at.is_(None))
    )
    return result.scalar_one_or_none()


async def get_by_email(db: AsyncSession, email: str) -> User | None:
    result = await db.execute(
        select(User).where(User.email == normalize_email(email), User.deleted_at.is_(None))
    )
    return result.scalar_one_or_none()


async def list_users(
    db: AsyncSession,
    *,
    role: str | None = None,
    is_active: bool | None = None,
    search: str | None = None,
    page: int = 1,
    page_size: int = 50,
) -> tuple[list[User], int]:
    query = select(User).where(User.deleted_at.is_(None))
    if role:
        query = query.where(User.role == role)
    if is_active is not None:
        query = query.where(User.is_active == is_active)
    if search:
        pattern = f"%{search.strip().lower()}%"
        query = query.where(
            func.lower(User.full_name).like(pattern) | func.lower(User.email).like(pattern)
        )

    total = (await db.execute(select(func.count()).select_from(query.subquery()))).scalar_one()
    query = query.order_by(User.full_name.asc()).offset((page - 1) * page_size).limit(page_size)
    rows = (await db.execute(query)).scalars().all()
    return list(rows), total


async def create_user(
    db: AsyncSession,
    *,
    email: str,
    full_name: str,
    role: str,
    phone: str | None = None,
) -> tuple[User, str]:
    """Legt einen Nutzer an und gibt (Nutzer, Einmal-Passwort) zurück."""
    email = normalize_email(email)
    existing = await db.execute(select(User.id).where(User.email == email))
    if existing.scalar_one_or_none() is not None:
        raise ValueError(f"E-Mail-Adresse {email} ist bereits vergeben")

    initial_password = generate_initial_password()
    user = User(
        email=email,
        full_name=full_name.strip(),
        role=role,
        password_hash=hash_password(initial_password),
        phone=phone,
        is_active=True,
        must_change_password=True,
        totp_enabled=False,
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return user, initial_password


async def revoke_all_sessions(db: AsyncSession, user_id: int) -> int:
    """Widerruft alle offenen Sessions eines Nutzers. Gibt die Anzahl zurück."""
    result = await db.execute(
        update(UserSession)
        .where(UserSession.user_id == user_id, UserSession.is_revoked == False)  # noqa: E712
        .values(is_revoked=True)
    )
    await db.commit()
    return result.rowcount or 0


async def update_user(db: AsyncSession, user: User, data: dict) -> User:
    """Wendet ein Teil-Update an. Deaktivierung widerruft alle Sessions."""
    deactivated = data.get("is_active") is False and user.is_active

    for field, value in data.items():
        if value is not None:
            setattr(user, field, value)
    await db.commit()
    await db.refresh(user)

    if deactivated:
        await revoke_all_sessions(db, user.id)
        await db.refresh(user)
    return user


async def reset_password(db: AsyncSession, user: User) -> tuple[str, int]:
    """Setzt ein neues Einmal-Passwort und widerruft alle Sessions."""
    initial_password = generate_initial_password()
    user.password_hash = hash_password(initial_password)
    user.must_change_password = True
    await db.commit()
    revoked = await revoke_all_sessions(db, user.id)
    return initial_password, revoked


async def reset_totp(db: AsyncSession, user: User) -> int:
    """
    Setzt 2FA zurück (Geräteverlust). Der Nutzer muss beim nächsten Login neu
    einrichten; alle Sessions werden widerrufen.
    """
    user.totp_secret = None
    user.totp_enabled = False
    await db.commit()
    return await revoke_all_sessions(db, user.id)


async def soft_delete_user(db: AsyncSession, user: User, deleted_by_id: int) -> None:
    """
    Deaktiviert und markiert den Nutzer als gelöscht. Die Zeile bleibt bestehen:
    Audit-Log und `matter_access` referenzieren sie, und die Nachvollziehbarkeit
    anwaltlicher Bearbeitung darf nicht durch eine Nutzerlöschung brechen.
    """
    user.deleted_at = datetime.now(UTC)
    user.deleted_by_id = deleted_by_id
    user.is_active = False
    await db.commit()
    await revoke_all_sessions(db, user.id)


async def count_active_admins(db: AsyncSession, *, exclude_user_id: int | None = None) -> int:
    query = select(func.count()).select_from(User).where(
        User.role == Role.ADMIN,
        User.is_active == True,  # noqa: E712
        User.deleted_at.is_(None),
    )
    if exclude_user_id is not None:
        query = query.where(User.id != exclude_user_id)
    return (await db.execute(query)).scalar_one()
