from datetime import UTC, datetime, timedelta
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Request, Response, status
from pydantic import BaseModel, EmailStr
from redis.asyncio import Redis
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.core.deps import get_db_session, get_current_user, get_current_user_allow_totp_setup
from app.core.rbac import requires_2fa
from app.core.security import (
    create_access_token,
    create_refresh_token,
    decode_token,
    generate_session_id,
    generate_totp_secret,
    get_totp_uri,
    hash_password,
    password_meets_policy,
    verify_password,
    verify_totp,
)
from app.models.user import User, UserSession

router = APIRouter(prefix="/auth", tags=["auth"])
settings = get_settings()


def _get_redis() -> Redis:
    import redis.asyncio as aioredis
    return aioredis.from_url(settings.REDIS_URL, decode_responses=True)


LOCKOUT_KEY = "login_lock:{email}"
FAIL_COUNT_KEY = "login_fail:{email}"


async def _check_lockout(redis: Redis, email: str) -> None:
    locked = await redis.get(LOCKOUT_KEY.format(email=email))
    if locked:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Account temporarily locked due to too many failed login attempts",
        )


async def _record_failure(redis: Redis, email: str) -> None:
    key = FAIL_COUNT_KEY.format(email=email)
    count = await redis.incr(key)
    await redis.expire(key, settings.LOCKOUT_MINUTES * 60)
    if count >= settings.MAX_FAILED_LOGINS:
        await redis.set(
            LOCKOUT_KEY.format(email=email),
            "1",
            ex=settings.LOCKOUT_MINUTES * 60,
        )


async def _clear_failures(redis: Redis, email: str) -> None:
    await redis.delete(FAIL_COUNT_KEY.format(email=email))
    await redis.delete(LOCKOUT_KEY.format(email=email))


# --- Schemas ---

class LoginRequest(BaseModel):
    email: EmailStr
    password: str
    totp_code: str | None = None


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    requires_totp: bool = False
    totp_setup_required: bool = False


class RefreshRequest(BaseModel):
    pass  # Refresh token comes from httpOnly cookie


class TotpSetupResponse(BaseModel):
    secret: str
    qr_uri: str


# --- Endpoints ---

@router.post("/login", response_model=TokenResponse)
async def login(
    body: LoginRequest,
    request: Request,
    response: Response,
    db: AsyncSession = Depends(get_db_session),
):
    redis = _get_redis()
    await _check_lockout(redis, body.email)

    result = await db.execute(
        select(User).where(User.email == body.email, User.is_active == True, User.deleted_at == None)
    )
    user = result.scalar_one_or_none()

    if not user or not verify_password(body.password, user.password_hash):
        await _record_failure(redis, body.email)
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials")

    await _clear_failures(redis, body.email)

    # If 2FA required and not yet provided, return partial response
    totp_setup_required = False
    if requires_2fa(user.role):
        if not user.totp_enabled:
            # 2FA-Pflicht durchsetzen: eingeschränktes Setup-Token ausstellen,
            # das nur die TOTP-Einrichtung erlaubt (kein Refresh-Cookie).
            totp_setup_required = True
        elif not body.totp_code:
            return TokenResponse(access_token="", requires_totp=True)
        elif not verify_totp(user.totp_secret, body.totp_code):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid 2FA code")

    session_id = generate_session_id()
    now = datetime.now(UTC)
    expires_at = now + timedelta(hours=settings.REFRESH_TOKEN_EXPIRE_HOURS)

    db_session = UserSession(
        session_id=session_id,
        user_id=user.id,
        created_at=now,
        last_active=now,
        expires_at=expires_at,
        ip_address=request.client.host if request.client else "unknown",
        user_agent=request.headers.get("User-Agent"),
    )
    db.add(db_session)
    user.last_login = now
    user.last_login_ip = request.client.host if request.client else "unknown"
    await db.commit()

    if totp_setup_required:
        access_token = create_access_token(
            user.id, session_id, user.role, extra={"scope": "totp_setup"}
        )
        return TokenResponse(access_token=access_token, totp_setup_required=True)

    access_token = create_access_token(user.id, session_id, user.role)
    refresh_token = create_refresh_token(user.id, session_id)

    response.set_cookie(
        key="refresh_token",
        value=refresh_token,
        httponly=True,
        secure=True,
        samesite="strict",
        max_age=settings.REFRESH_TOKEN_EXPIRE_HOURS * 3600,
        path="/api/auth",
    )

    return TokenResponse(access_token=access_token)


@router.post("/refresh", response_model=TokenResponse)
async def refresh(
    request: Request,
    response: Response,
    db: AsyncSession = Depends(get_db_session),
):
    refresh_token = request.cookies.get("refresh_token")
    if not refresh_token:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="No refresh token")

    try:
        payload = decode_token(refresh_token)
    except ValueError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid refresh token")

    if payload.get("type") != "refresh":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Wrong token type")

    user_id = int(payload["sub"])
    session_id = payload["sid"]

    result = await db.execute(
        select(UserSession).where(
            UserSession.session_id == session_id,
            UserSession.is_revoked == False,
            UserSession.expires_at > datetime.now(UTC),
        )
    )
    db_session = result.scalar_one_or_none()
    if not db_session:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Session expired")

    user_result = await db.execute(
        select(User).where(User.id == user_id, User.is_active == True)
    )
    user = user_result.scalar_one_or_none()
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="User not found")

    access_token = create_access_token(user.id, session_id, user.role)
    return TokenResponse(access_token=access_token)


@router.post("/logout")
async def logout(
    request: Request,
    response: Response,
    db: AsyncSession = Depends(get_db_session),
    user=Depends(get_current_user),
):
    session_id = request.state.session_id
    result = await db.execute(
        select(UserSession).where(UserSession.session_id == session_id)
    )
    db_session = result.scalar_one_or_none()
    if db_session:
        db_session.is_revoked = True
        await db.commit()

    response.delete_cookie("refresh_token", path="/api/auth")
    return {"detail": "Logged out"}


@router.post("/totp/setup", response_model=TotpSetupResponse)
async def setup_totp(
    db: AsyncSession = Depends(get_db_session),
    user=Depends(get_current_user_allow_totp_setup),
):
    if user.totp_enabled:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="2FA already enabled")
    secret = generate_totp_secret()
    user.totp_secret = secret
    await db.commit()
    return TotpSetupResponse(secret=secret, qr_uri=get_totp_uri(secret, user.email))


@router.post("/totp/confirm", response_model=TokenResponse)
async def confirm_totp(
    body: dict,
    request: Request,
    db: AsyncSession = Depends(get_db_session),
    user=Depends(get_current_user_allow_totp_setup),
):
    code = body.get("code", "")
    if not user.totp_secret or not verify_totp(user.totp_secret, code):
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Invalid TOTP code")
    user.totp_enabled = True
    await db.commit()
    # Nach erfolgreicher Einrichtung: vollwertiges Access-Token ausstellen,
    # damit der Nutzer nahtlos weiterarbeiten kann (Session existiert bereits).
    access_token = create_access_token(user.id, request.state.session_id, user.role)
    return TokenResponse(access_token=access_token)
