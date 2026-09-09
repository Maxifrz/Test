"""Tests der Benutzerverwaltung (reine Logik: Passworterzeugung, Normalisierung)."""
import pytest

from app.core.security import (
    MAX_PASSWORD_BYTES,
    password_meets_policy,
    verify_password_constant_time,
    hash_password,
)
from app.services.user_service import generate_initial_password, normalize_email


def test_generated_password_meets_policy():
    for _ in range(50):
        assert password_meets_policy(generate_initial_password())


def test_generated_passwords_are_unique():
    generated = {generate_initial_password() for _ in range(100)}
    assert len(generated) == 100


def test_generated_password_respects_length():
    assert len(generate_initial_password(24)) == 24
    with pytest.raises(ValueError):
        generate_initial_password(8)


def test_generated_password_avoids_confusable_glyphs():
    joined = "".join(generate_initial_password() for _ in range(50))
    for glyph in "0O1lI":
        assert glyph not in joined


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Admin@Kanzlei.local", "admin@kanzlei.local"),
        ("  MAX@Example.DE  ", "max@example.de"),
        ("already@lower.de", "already@lower.de"),
    ],
)
def test_normalize_email(raw, expected):
    assert normalize_email(raw) == expected


def test_password_policy_rejects_overlong():
    # bcrypt schneidet nach 72 Byte ab -> laengere Passwoerter waeren
    # stillschweigend aequivalent.
    ok = "Abcdefgh1!" + "x" * (MAX_PASSWORD_BYTES - 10)
    assert password_meets_policy(ok)
    assert not password_meets_policy(ok + "x")


def test_password_policy_counts_bytes_not_chars():
    # Umlaute belegen 2 Byte in UTF-8
    pwd = "Abcdefg1!" + "ä" * 32  # 9 + 64 = 73 Byte
    assert len(pwd) < MAX_PASSWORD_BYTES
    assert not password_meets_policy(pwd)


def test_constant_time_verify_handles_missing_user():
    # Unbekannter Nutzer -> False, aber ohne Exception (und mit bcrypt-Aufwand)
    assert verify_password_constant_time("irgendwas", None) is False


def test_constant_time_verify_matches_real_hash():
    hashed = hash_password("Korrekt1!xyz")
    assert verify_password_constant_time("Korrekt1!xyz", hashed) is True
    assert verify_password_constant_time("falsch", hashed) is False


def test_constant_time_verify_survives_corrupt_hash():
    assert verify_password_constant_time("egal", "kein-gueltiger-bcrypt-hash") is False
