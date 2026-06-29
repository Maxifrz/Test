from enum import StrEnum
from typing import Literal


class Role(StrEnum):
    ADMIN = "admin"
    ANWALT = "anwalt"
    SACHBEARBEITER = "sachbearbeiter"
    SEKRETARIAT = "sekretariat"


class MatterRole(StrEnum):
    LEAD = "lead"
    SUPPORT = "support"
    READONLY = "readonly"


# Global permission matrix: role → set of allowed actions
_PERMISSIONS: dict[str, set[str]] = {
    Role.ADMIN: {
        "user.create", "user.read", "user.update", "user.delete",
        "matter.create", "matter.read", "matter.update", "matter.delete",
        "matter.access.grant", "matter.access.revoke",
        "client.create", "client.read", "client.update", "client.delete",
        "email.read", "email.send", "email.rule.create", "email.rule.update", "email.rule.delete",
        "ticket.create", "ticket.read", "ticket.update", "ticket.delete",
        "calendar.create", "calendar.read", "calendar.update", "calendar.delete",
        "transcription.upload", "transcription.read", "transcription.edit", "transcription.delete",
        "finance.read", "finance.write",
        "dsgvo.export", "dsgvo.erasure", "dsgvo.vvt.edit",
        "audit.read",
        "admin.all",
    },
    Role.ANWALT: {
        "matter.create", "matter.read", "matter.update",
        "matter.access.grant",
        "client.create", "client.read", "client.update",
        "email.read", "email.send",
        "ticket.create", "ticket.read", "ticket.update",
        "calendar.create", "calendar.read", "calendar.update",
        "transcription.upload", "transcription.read", "transcription.edit",
        "finance.read",
    },
    Role.SACHBEARBEITER: {
        "matter.read", "matter.update",
        "client.read", "client.update",
        "email.read", "email.send",
        "ticket.create", "ticket.read", "ticket.update",
        "calendar.create", "calendar.read", "calendar.update",
        "transcription.upload", "transcription.read", "transcription.edit",
        "finance.read",
    },
    Role.SEKRETARIAT: {
        "matter.read",
        "client.read",
        "email.read",
        "ticket.create", "ticket.read",
        "calendar.read", "calendar.create",
        "transcription.upload", "transcription.read",
    },
}

# 2FA requirement per role
REQUIRES_2FA: dict[str, bool] = {
    Role.ADMIN: True,
    Role.ANWALT: True,
    Role.SACHBEARBEITER: False,
    Role.SEKRETARIAT: False,
}


def has_permission(role: str, action: str) -> bool:
    return action in _PERMISSIONS.get(role, set())


def requires_2fa(role: str) -> bool:
    return REQUIRES_2FA.get(role, False)
