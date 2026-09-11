"""
Tests der Postfach-Aufloesung.

Der wichtigste Fall: eine Bestandsinstallation ohne angelegte Konten muss nach
dem Update ohne Umkonfiguration weiterlaufen -- das .env-Einzelpostfach bleibt
als Rueckfall gueltig.
"""

from app.services.email_account_service import MailboxConfig, _from_model


class _FakeAccount:
    """Minimales Stand-in fuer EmailAccount (kein DB-Zugriff noetig)."""

    def __init__(self, **kw):
        defaults = {
            "id": 1, "name": "Muenchen", "imap_host": "imap.example.de", "imap_port": 993,
            "imap_ssl": True, "imap_username": "kanzlei", "imap_password": "geheim",
            "imap_folder": "INBOX", "smtp_host": "smtp.example.de", "smtp_port": 587,
            "smtp_tls": True, "smtp_username": "kanzlei", "smtp_password": "geheim2",
            "from_name": "Kanzlei Muenchen", "from_email": "muenchen@example.de",
        }
        defaults.update(kw)
        for k, v in defaults.items():
            setattr(self, k, v)


def test_from_model_maps_all_fields():
    cfg = _from_model(_FakeAccount())
    assert isinstance(cfg, MailboxConfig)
    assert cfg.account_id == 1
    assert cfg.imap_host == "imap.example.de"
    assert cfg.smtp_port == 587
    assert cfg.from_email == "muenchen@example.de"


def test_from_model_handles_missing_passwords():
    # Ein Konto ohne SMTP-Passwort (offener Relay im internen Netz) darf nicht
    # mit None weiterlaufen -- aiosmtplib erwartet einen String oder None,
    # aber der Vergleich "if mailbox.smtp_username" braucht einen sauberen Wert.
    cfg = _from_model(_FakeAccount(smtp_password=None, smtp_username=None))
    assert cfg.smtp_password == ""
    assert cfg.smtp_username == ""


def test_from_settings_returns_none_without_config(monkeypatch):
    from app.services import email_account_service as mod

    class _S:
        IMAP_HOST = ""
        SMTP_HOST = ""

    monkeypatch.setattr("app.core.config.get_settings", lambda: _S())
    assert mod._from_settings() is None


def test_from_settings_builds_fallback(monkeypatch):
    from app.services import email_account_service as mod

    class _S:
        IMAP_HOST = "imap.alt.de"
        IMAP_PORT = 993
        IMAP_SSL = True
        IMAP_USERNAME = "alt"
        IMAP_PASSWORD = "pw"
        SMTP_HOST = "smtp.alt.de"
        SMTP_PORT = 587
        SMTP_TLS = True
        SMTP_USERNAME = "alt"
        SMTP_PASSWORD = "pw"
        SMTP_FROM_NAME = "Alt"
        SMTP_FROM_EMAIL = "alt@example.de"

    monkeypatch.setattr("app.core.config.get_settings", lambda: _S())
    cfg = mod._from_settings()
    assert cfg is not None
    # account_id None kennzeichnet den .env-Rueckfall: dafuer gibt es keine
    # DB-Zeile, an der ein Sync-Ergebnis vermerkt werden koennte.
    assert cfg.account_id is None
    assert cfg.imap_host == "imap.alt.de"
    assert cfg.from_email == "alt@example.de"


def test_fallback_used_when_only_smtp_configured(monkeypatch):
    # Versand ohne Abruf ist eine gueltige Konfiguration
    from app.services import email_account_service as mod

    class _S:
        IMAP_HOST = ""
        IMAP_PORT = 993
        IMAP_SSL = True
        IMAP_USERNAME = ""
        IMAP_PASSWORD = ""
        SMTP_HOST = "smtp.alt.de"
        SMTP_PORT = 587
        SMTP_TLS = True
        SMTP_USERNAME = ""
        SMTP_PASSWORD = ""
        SMTP_FROM_NAME = "Alt"
        SMTP_FROM_EMAIL = "alt@example.de"

    monkeypatch.setattr("app.core.config.get_settings", lambda: _S())
    cfg = mod._from_settings()
    assert cfg is not None and cfg.smtp_host == "smtp.alt.de"
