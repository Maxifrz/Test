"""Unit tests for the pure email rule engine."""
from app.services.email_routing import (
    IncomingEmail,
    detect_confidential,
    evaluate_rules,
)


def _email(frm="kontakt@gericht.bayern.de", subj="Mahnung Az. 12 O 3/24", body="Sehr geehrte Damen"):
    return IncomingEmail(from_address=frm, subject=subj, body_text=body)


def test_from_domain_match():
    rules = [(1, {"from_domain": "gericht.bayern.de"}, {"file_to_matter": 5}, 10)]
    match = evaluate_rules(rules, _email())
    assert match is not None
    assert match.actions["file_to_matter"] == 5


def test_priority_first_match_wins():
    rules = [
        (1, {"from_domain": "gericht.bayern.de"}, {"file_to_matter": 1}, 10),
        (2, {"from_domain": "gericht.bayern.de"}, {"file_to_matter": 2}, 20),
    ]
    match = evaluate_rules(rules, _email())
    assert match.rule_id == 1  # lower priority number wins


def test_match_all_requires_every_condition():
    rules = [(
        1,
        {"match": "all", "from_domain": "gericht.bayern.de", "subject_contains": ["Berufung"]},
        {"flag_review": True},
        10,
    )]
    # subject has "Mahnung", not "Berufung" → no match
    assert evaluate_rules(rules, _email()) is None


def test_match_any_one_condition_suffices():
    rules = [(
        1,
        {"match": "any", "from_domain": "unbekannt.de", "subject_contains": ["Mahnung"]},
        {"flag_review": True},
        10,
    )]
    match = evaluate_rules(rules, _email())
    assert match is not None


def test_no_rules_returns_none():
    assert evaluate_rules([], _email()) is None


def test_empty_conditions_never_match():
    rules = [(1, {}, {"file_to_matter": 9}, 10)]
    assert evaluate_rules(rules, _email()) is None


def test_subject_keyword_case_insensitive():
    rules = [(1, {"subject_contains": ["mahnung"]}, {"flag_review": True}, 10)]
    assert evaluate_rules(rules, _email(subj="MAHNUNG dringend")) is not None


def test_detect_confidential():
    assert detect_confidential(_email(subj="Streng vertraulich", body="...")) is True
    assert detect_confidential(_email(subj="Terminbestätigung", body="Hallo")) is False
