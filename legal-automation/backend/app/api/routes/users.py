"""
Benutzerverwaltung (Admin) + eigenes Profil.

Ohne diese Endpunkte ist das RBAC-/`matter_access`-Modell unbenutzbar: es
gäbe nur den geseedeten Initial-Admin und keinen Weg, Anwälte, Sachbearbeiter
oder Sekretariat anzulegen, denen Akten zugewiesen werden können.

Sicherheitsregeln:
- Einmal-Passwörter werden erzeugt, gehasht gespeichert und genau einmal
  (in der Antwort auf Anlage/Reset) im Klartext ausgegeben.
- Der letzte aktive Admin kann weder deaktiviert, noch degradiert, noch
  gelöscht werden — sonst sperrt sich die Kanzlei selbst aus.
- Niemand darf die eigene Rolle oder den eigenen Aktiv-Status ändern.
"""
from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.core.deps import DB, get_current_user, require_permission
from app.core.rbac import Role
from app.schemas.user import (
    PasswordResetResponse,
    SelfUpdate,
    UserCreate,
    UserCreateResponse,
    UserListResponse,
    UserResponse,
    UserUpdate,
)
from app.services import user_service

router = APIRouter(prefix="/users", tags=["users"])


async def _load_target(db: DB, user_id: int):
    user = await user_service.get_user(db, user_id)
    if not user:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Nutzer nicht gefunden")
    return user


async def _guard_last_admin(db: DB, target, *, new_role: str | None = None, deactivating: bool = False) -> None:
    """Verhindert, dass der letzte aktive Admin seine Rechte oder seinen Zugang verliert."""
    if target.role != Role.ADMIN:
        return
    losing_admin = deactivating or (new_role is not None and new_role != Role.ADMIN)
    if not losing_admin:
        return
    if await user_service.count_active_admins(db, exclude_user_id=target.id) == 0:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Der letzte aktive Administrator kann nicht deaktiviert oder herabgestuft werden",
        )


# --- Eigenes Profil (jede Rolle) ---

@router.get("/me", response_model=UserResponse)
async def read_own_profile(current_user=Depends(get_current_user)):
    return current_user


@router.patch("/me", response_model=UserResponse)
async def update_own_profile(
    data: SelfUpdate,
    db: DB,
    current_user=Depends(get_current_user),
):
    # SelfUpdate enthält bewusst weder `role` noch `is_active`.
    return await user_service.update_user(db, current_user, data.model_dump(exclude_unset=True))


# --- Verwaltung (Admin) ---

@router.get("", response_model=UserListResponse)
async def list_users(
    db: DB,
    role: str | None = Query(None),
    is_active: bool | None = Query(None),
    search: str | None = Query(None, max_length=100),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    current_user=Depends(require_permission("user.read")),
):
    items, total = await user_service.list_users(
        db, role=role, is_active=is_active, search=search, page=page, page_size=page_size
    )
    return UserListResponse(
        items=[UserResponse.model_validate(u) for u in items],
        total=total,
        page=page,
        page_size=page_size,
    )


@router.get("/{user_id}", response_model=UserResponse)
async def read_user(user_id: int, db: DB, current_user=Depends(require_permission("user.read"))):
    return await _load_target(db, user_id)


@router.post("", response_model=UserCreateResponse, status_code=status.HTTP_201_CREATED)
async def create_user(
    data: UserCreate,
    db: DB,
    current_user=Depends(require_permission("user.create")),
):
    try:
        user, initial_password = await user_service.create_user(
            db, email=str(data.email), full_name=data.full_name, role=data.role, phone=data.phone
        )
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail=str(exc))
    return UserCreateResponse(
        user=UserResponse.model_validate(user), initial_password=initial_password
    )


@router.patch("/{user_id}", response_model=UserResponse)
async def update_user(
    user_id: int,
    data: UserUpdate,
    db: DB,
    current_user=Depends(require_permission("user.update")),
):
    target = await _load_target(db, user_id)
    payload = data.model_dump(exclude_unset=True)

    if target.id == current_user.id and ("role" in payload or "is_active" in payload):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Eigene Rolle und eigener Aktiv-Status können nicht selbst geändert werden",
        )
    await _guard_last_admin(
        db, target,
        new_role=payload.get("role"),
        deactivating=payload.get("is_active") is False,
    )
    return await user_service.update_user(db, target, payload)


@router.post("/{user_id}/reset-password", response_model=PasswordResetResponse)
async def reset_password(
    user_id: int,
    db: DB,
    current_user=Depends(require_permission("user.update")),
):
    """Neues Einmal-Passwort. Der Nutzer muss es beim nächsten Login wechseln;
    seine 2FA bleibt bestehen und wird beim Login weiterhin abgefragt."""
    target = await _load_target(db, user_id)
    initial_password, revoked = await user_service.reset_password(db, target)
    return PasswordResetResponse(
        user_id=target.id, initial_password=initial_password, sessions_revoked=revoked
    )


@router.post("/{user_id}/reset-2fa", status_code=status.HTTP_200_OK)
async def reset_2fa(
    user_id: int,
    db: DB,
    current_user=Depends(require_permission("user.update")),
):
    """2FA-Reset bei Geräteverlust. Erzwingt Neu-Einrichtung beim nächsten Login."""
    target = await _load_target(db, user_id)
    revoked = await user_service.reset_totp(db, target)
    return {"detail": "2FA zurückgesetzt", "sessions_revoked": revoked}


@router.delete("/{user_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_user(
    user_id: int,
    db: DB,
    current_user=Depends(require_permission("user.delete")),
):
    target = await _load_target(db, user_id)
    if target.id == current_user.id:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT, detail="Selbstlöschung ist nicht möglich"
        )
    await _guard_last_admin(db, target, deactivating=True)
    await user_service.soft_delete_user(db, target, deleted_by_id=current_user.id)
