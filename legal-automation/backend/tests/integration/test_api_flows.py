"""
End-to-End-Flows über die echte API (Postgres + Redis + Migrationen).

Die Tests bilden bewusst eine ERZÄHLREIHENFOLGE ab (Erst-Login-Kette →
Arbeiten mit vollem Token) und teilen Zustand über STATE — sie sind als
eine zusammenhängende Suite gedacht, nicht einzeln lauffähig.
"""
import pyotp
import pytest

pytestmark = pytest.mark.asyncio(loop_scope="session")

ADMIN_EMAIL = "admin@kanzlei.local"
INITIAL_PW = "Admin1234!"
NEW_PW = "Integr4tions!Passwort"

STATE: dict = {}


def _auth(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


# ---------------------------------------------------------------- Basis

async def test_health(client):
    r = await client.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["db"] == "connected"
    assert body["redis"] == "connected"


async def test_login_wrong_password_is_401(client):
    r = await client.post("/api/auth/login", json={"email": ADMIN_EMAIL, "password": "falsch!"})
    assert r.status_code == 401


# ------------------------------------------- Erst-Login-Kette (P0-Fixes)

async def test_initial_login_forces_password_change(client):
    """Der in 0001/0012 geseedete Admin kann sich mit Admin1234! anmelden
    und erhält NUR ein pwd_change-Token (Migrationstest für den Hash-Fix)."""
    r = await client.post("/api/auth/login", json={"email": ADMIN_EMAIL, "password": INITIAL_PW})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["password_change_required"] is True
    STATE["pwd_change_token"] = body["access_token"]

    # Scope-Beschränkung: normales Arbeiten ist mit diesem Token verboten
    r = await client.get("/api/clients", headers=_auth(body["access_token"]))
    assert r.status_code == 403
    assert r.headers.get("X-Password-Change-Required") == "true"


async def test_password_change_then_totp_setup_chain(client):
    r = await client.post(
        "/api/auth/change-password",
        headers=_auth(STATE["pwd_change_token"]),
        json={"current_password": INITIAL_PW, "new_password": NEW_PW},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    # Admin-Rolle verlangt 2FA → nächste Stufe der Kette
    assert body["totp_setup_required"] is True
    setup_token = body["access_token"]

    # Auch das Setup-Token erlaubt kein normales Arbeiten
    r = await client.get("/api/clients", headers=_auth(setup_token))
    assert r.status_code == 403
    assert r.headers.get("X-2FA-Setup-Required") == "true"

    r = await client.post("/api/auth/totp/setup", headers=_auth(setup_token))
    assert r.status_code == 200, r.text
    secret = r.json()["secret"]
    STATE["totp_secret"] = secret

    r = await client.post(
        "/api/auth/totp/confirm",
        headers=_auth(setup_token),
        json={"code": pyotp.TOTP(secret).now()},
    )
    assert r.status_code == 200, r.text
    STATE["admin_token"] = r.json()["access_token"]

    # Vollwertiges Token → normale Arbeit möglich
    r = await client.get("/api/clients", headers=_auth(STATE["admin_token"]))
    assert r.status_code == 200


async def test_relogin_requires_totp_code(client):
    r = await client.post("/api/auth/login", json={"email": ADMIN_EMAIL, "password": NEW_PW})
    assert r.status_code == 200
    assert r.json().get("requires_totp") is True

    r = await client.post(
        "/api/auth/login",
        json={
            "email": ADMIN_EMAIL,
            "password": NEW_PW,
            "totp_code": pyotp.TOTP(STATE["totp_secret"]).now(),
        },
    )
    assert r.status_code == 200
    assert r.json()["access_token"]


# ------------------------------------------------- Routen-Regressionen

async def test_emails_templates_not_shadowed_by_id_route(client):
    """Regression P0-3: /emails/templates darf nicht von /emails/{id} gefangen werden."""
    r = await client.get("/api/emails/templates", headers=_auth(STATE["admin_token"]))
    assert r.status_code == 200
    assert isinstance(r.json(), list)

    r = await client.get("/api/emails/rules", headers=_auth(STATE["admin_token"]))
    assert r.status_code == 200


# ------------------------------------------ CRUD + Trennungsgebot (RBAC)

async def test_client_and_matter_crud(client):
    r = await client.post(
        "/api/clients",
        headers=_auth(STATE["admin_token"]),
        json={"first_name": "Max", "last_name": "Mustermann", "email": "max@firma.example"},
    )
    assert r.status_code == 201, r.text
    STATE["client_id"] = r.json()["id"]

    r = await client.post(
        "/api/matters",
        headers=_auth(STATE["admin_token"]),
        json={
            "title": "Insolvenzverfahren Mustermann GmbH",
            "matter_type": "insolvenz",
            "client_id": STATE["client_id"],
            "lead_anwalt_id": 1,
        },
    )
    assert r.status_code == 201, r.text
    STATE["matter_id"] = r.json()["id"]

    r = await client.get(f"/api/matters/{STATE['matter_id']}", headers=_auth(STATE["admin_token"]))
    assert r.status_code == 200


async def test_matter_access_enforced_for_second_user(client, session_factory):
    """Trennungsgebot: Sachbearbeiter ohne matter_access sieht die Akte nicht."""
    from app.core.security import hash_password
    from app.models.user import User

    async with session_factory() as db:
        sb = User(
            email="sb@kanzlei.local",
            full_name="Sach Bearbeiter",
            role="sachbearbeiter",
            password_hash=hash_password("SbPasswort1!"),
            is_active=True,
        )
        db.add(sb)
        await db.commit()
        await db.refresh(sb)
        STATE["sb_id"] = sb.id

    r = await client.post(
        "/api/auth/login", json={"email": "sb@kanzlei.local", "password": "SbPasswort1!"}
    )
    assert r.status_code == 200, r.text
    STATE["sb_token"] = r.json()["access_token"]
    assert STATE["sb_token"]  # Sachbearbeiter: kein 2FA-Zwang, direkt vollwertig

    r = await client.get(f"/api/matters/{STATE['matter_id']}", headers=_auth(STATE["sb_token"]))
    assert r.status_code == 403

    r = await client.post(
        f"/api/matters/{STATE['matter_id']}/access",
        headers=_auth(STATE["admin_token"]),
        json={"user_id": STATE["sb_id"], "matter_role": "support"},
    )
    assert r.status_code == 201, r.text

    r = await client.get(f"/api/matters/{STATE['matter_id']}", headers=_auth(STATE["sb_token"]))
    assert r.status_code == 200


# ------------------------------------- E-Mail-Anhänge (neues Feature E2E)

async def test_email_attachment_roundtrip(client, session_factory):
    """Ingest einer Mail mit PDF-Anhang → verschlüsselte Ablage → Liste + Download
    über die API; Trennungsgebot greift auch auf Anhänge."""
    from email.message import EmailMessage as PyEmailMessage

    from app.services import email_service

    pdf = b"%PDF-1.4 integration " + b"q" * 256
    mime = PyEmailMessage()
    mime["From"] = "gericht@ag-hannover.example"
    mime["To"] = "kanzlei@eckert.example"
    mime["Subject"] = "Beschluss mit Anlage"
    mime["Message-ID"] = "<integration-attach-1@example>"
    mime.set_content("Anbei der Beschluss.")
    mime.add_attachment(pdf, maintype="application", subtype="pdf", filename="Beschluss.pdf")

    parsed = email_service.parse_raw_email(mime.as_bytes())
    async with session_factory() as db:
        msg = await email_service.ingest_email(db, parsed)
        assert msg is not None
        email_id = msg.id

    # Anhang auf eine zweite Akte legen, auf die der Sachbearbeiter KEINEN Zugriff hat
    r = await client.post(
        "/api/matters",
        headers=_auth(STATE["admin_token"]),
        json={
            "title": "Zweitverfahren", "matter_type": "insolvenz",
            "client_id": STATE["client_id"], "lead_anwalt_id": 1,
        },
    )
    matter2 = r.json()["id"]
    r = await client.post(
        f"/api/emails/{email_id}/file",
        headers=_auth(STATE["admin_token"]),
        json={"matter_id": matter2},
    )
    assert r.status_code == 200, r.text

    r = await client.get(f"/api/emails/{email_id}/attachments", headers=_auth(STATE["admin_token"]))
    assert r.status_code == 200
    atts = r.json()
    assert len(atts) == 1
    assert atts[0]["filename"] == "Beschluss.pdf"
    assert atts[0]["size_bytes"] == len(pdf)

    r = await client.get(
        f"/api/emails/{email_id}/attachments/{atts[0]['id']}/download",
        headers=_auth(STATE["admin_token"]),
    )
    assert r.status_code == 200
    assert r.content == pdf  # entschlüsselter Inhalt == Original
    assert r.headers["content-type"].startswith("application/pdf")

    # Sachbearbeiter hat keinen Zugriff auf matter2 → 403 auf Mail UND Anhänge
    r = await client.get(f"/api/emails/{email_id}", headers=_auth(STATE["sb_token"]))
    assert r.status_code == 403
    r = await client.get(f"/api/emails/{email_id}/attachments", headers=_auth(STATE["sb_token"]))
    assert r.status_code == 403


# ------------------------------------------- Kontaktformular (Website)

async def test_public_contact_stores_request(client, session_factory):
    from sqlalchemy import select

    from app.models.contact import ContactRequest

    r = await client.post(
        "/api/public/contact",
        json={
            "name": "Erika Beispiel",
            "email": "erika@firma.example",
            "message": "Wir benötigen kurzfristig eine Sanierungsberatung.",
            "standort": "Hannover",
            "rolle": "Unternehmen / Geschäftsführung",
            "consent": True,
        },
    )
    assert r.status_code == 200, r.text
    assert r.json() == {"status": "ok"}

    async with session_factory() as db:
        row = (
            await db.execute(select(ContactRequest).where(ContactRequest.email == "erika@firma.example"))
        ).scalar_one()
        assert row.status == "neu"
        assert row.consent_at is not None
        STATE["contact_id"] = row.id


async def test_public_contact_honeypot_discards_silently(client, session_factory):
    from sqlalchemy import select

    from app.models.contact import ContactRequest

    r = await client.post(
        "/api/public/contact",
        json={
            "name": "Bot", "email": "bot@spam.example",
            "message": "Buy backlinks now, ten chars.",
            "consent": True, "firma": "Spam GmbH",
        },
    )
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}  # identische Antwort — kein Bot-Oracle

    async with session_factory() as db:
        row = (
            await db.execute(select(ContactRequest).where(ContactRequest.email == "bot@spam.example"))
        ).scalar_one_or_none()
        assert row is None


async def test_contact_requests_admin_workflow(client):
    r = await client.get("/api/contact-requests", headers=_auth(STATE["admin_token"]))
    assert r.status_code == 200
    body = r.json()
    assert body["total"] >= 1
    assert any(i["id"] == STATE["contact_id"] for i in body["items"])

    r = await client.patch(
        f"/api/contact-requests/{STATE['contact_id']}",
        headers=_auth(STATE["admin_token"]),
        json={"status": "erledigt"},
    )
    assert r.status_code == 200
    assert r.json()["status"] == "erledigt"

    # Ohne Token kein Zugriff
    r = await client.get("/api/contact-requests")
    assert r.status_code in (401, 403)
