from collections.abc import AsyncGenerator
from typing import Annotated

from fastapi import Cookie, Depends, Header, HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import get_settings
from app.core.rbac import has_permission
from app.core.security import decode_token

settings = get_settings()

engine = create_async_engine(
    settings.DATABASE_URL,
    pool_size=10,
    max_overflow=20,
    pool_pre_ping=True,
    echo=settings.ENVIRONMENT == "development",
)
AsyncSessionLocal = async_sessionmaker(engine, expire_on_commit=False)


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    async with AsyncSessionLocal() as session:
        yield session


DB = Annotated[AsyncSession, Depends(get_db_session)]


async def _authenticate(
    request: Request,
    authorization: str | None,
    db: AsyncSession,
    allowed_scopes: frozenset[str] = frozenset(),
):
    """Validate JWT, check session validity, set request.state.user."""
    from datetime import UTC, datetime
    from sqlalchemy import select
    from app.models.user import User, UserSession

    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    if not authorization or not authorization.startswith("Bearer "):
        raise credentials_exception

    token = authorization.removeprefix("Bearer ").strip()
    try:
        payload = decode_token(token)
    except ValueError:
        raise credentials_exception

    if payload.get("type") != "access":
        raise credentials_exception

    # Eingeschränkte Token-Scopes: Setup-/Pflicht-Flows dürfen nur ihre
    # jeweiligen Endpunkte nutzen (2FA-Einrichtung bzw. Passwort-Wechsel).
    scope = payload.get("scope")
    if scope == "totp_setup" and "totp_setup" not in allowed_scopes:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="2FA-Einrichtung erforderlich, bevor die Anwendung genutzt werden kann",
            headers={"X-2FA-Setup-Required": "true"},
        )
    if scope == "pwd_change" and "pwd_change" not in allowed_scopes:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Passwort-Wechsel erforderlich, bevor die Anwendung genutzt werden kann",
            headers={"X-Password-Change-Required": "true"},
        )

    user_id = int(payload["sub"])
    session_id = payload["sid"]

    # Check session is still valid
    session_result = await db.execute(
        select(UserSession).where(
            UserSession.session_id == session_id,
            UserSession.is_revoked == False,
            UserSession.expires_at > datetime.now(UTC),
        )
    )
    session = session_result.scalar_one_or_none()
    if not session:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Session expired or revoked",
            headers={"X-Session-Expired": "true"},
        )

    # Update last_active
    session.last_active = datetime.now(UTC)
    await db.commit()

    user_result = await db.execute(
        select(User).where(User.id == user_id, User.is_active == True, User.deleted_at == None)
    )
    user = user_result.scalar_one_or_none()
    if not user:
        raise credentials_exception

    request.state.user = user
    request.state.session_id = session_id
    return user


async def get_current_user(
    request: Request,
    authorization: Annotated[str | None, Header()] = None,
    db: AsyncSession = Depends(get_db_session),
):
    return await _authenticate(request, authorization, db)


async def get_current_user_allow_totp_setup(
    request: Request,
    authorization: Annotated[str | None, Header()] = None,
    db: AsyncSession = Depends(get_db_session),
):
    """Nur für /auth/totp/setup|confirm: akzeptiert auch Setup-Scope-Tokens."""
    return await _authenticate(request, authorization, db, frozenset({"totp_setup"}))


async def get_current_user_allow_pwd_change(
    request: Request,
    authorization: Annotated[str | None, Header()] = None,
    db: AsyncSession = Depends(get_db_session),
):
    """Nur für /auth/change-password: akzeptiert auch Pflicht-Wechsel-Tokens."""
    return await _authenticate(request, authorization, db, frozenset({"pwd_change"}))


CurrentUser = Annotated[object, Depends(get_current_user)]


def get_matter_access_dependency(matter_id_param: str = "matter_id"):
    """
    Dependency factory: returns the matter + access grant for the current user.
    Raises 404 if matter not found, 403 if user has no active access (admins bypass).
    """
    async def _check(
        request: Request,
        db: AsyncSession = Depends(get_db_session),
        current_user=Depends(get_current_user),
    ):
        from sqlalchemy import select
        from app.core.rbac import Role
        from app.models.matter import Matter
        from app.models.matter_access import MatterAccess

        matter_id = request.path_params.get(matter_id_param)
        if matter_id is None:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="matter_id required")

        matter_id = int(matter_id)
        result = await db.execute(
            select(Matter).where(Matter.id == matter_id, Matter.deleted_at.is_(None))
        )
        matter = result.scalar_one_or_none()
        if not matter:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Matter not found")

        if current_user.role == Role.ADMIN:
            return matter, None

        access_result = await db.execute(
            select(MatterAccess).where(
                MatterAccess.user_id == current_user.id,
                MatterAccess.matter_id == matter_id,
                MatterAccess.revoked_at.is_(None),
            )
        )
        access = access_result.scalar_one_or_none()
        if not access:
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="No access to this matter")

        return matter, access

    return _check


async def ensure_matter_access(db: AsyncSession, user, matter_id: int | None) -> None:
    """
    Akten-Trennungsgebot (matter-level RBAC): 403, wenn der Nutzer keinen aktiven
    Zugriff auf die Akte hat. Admins passieren immer; `matter_id is None`
    (noch nicht zugeordnete Objekte, z. B. E-Mail-Review-Queue) ist bewusst
    erlaubt — die Sichtbarkeit regelt dort die Rollen-Permission.
    """
    from sqlalchemy import select
    from app.core.rbac import Role
    from app.models.matter_access import MatterAccess

    if matter_id is None or user.role == Role.ADMIN:
        return
    result = await db.execute(
        select(MatterAccess.id).where(
            MatterAccess.user_id == user.id,
            MatterAccess.matter_id == matter_id,
            MatterAccess.revoked_at.is_(None),
        )
    )
    if result.scalar_one_or_none() is None:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="No access to this matter")


async def accessible_matter_ids(db: AsyncSession, user) -> set[int] | None:
    """
    Für Listen-Filter: None = Admin (kein Filter nötig), sonst die Menge der
    Akten-IDs mit aktivem Zugriff des Nutzers.
    """
    from sqlalchemy import select
    from app.core.rbac import Role
    from app.models.matter_access import MatterAccess

    if user.role == Role.ADMIN:
        return None
    result = await db.execute(
        select(MatterAccess.matter_id).where(
            MatterAccess.user_id == user.id,
            MatterAccess.revoked_at.is_(None),
        )
    )
    return set(result.scalars().all())


def require_permission(action: str):
    """Dependency factory: raises 403 if the current user's role lacks the given permission."""
    async def _check(user=Depends(get_current_user)):
        if not has_permission(user.role, action):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Insufficient permissions")
        return user
    return _check
