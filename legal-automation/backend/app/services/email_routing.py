"""
E-Mail Rule Engine. Reine Logik — wertet Regelbedingungen gegen eine
eingehende Nachricht aus. Keine externen Aufrufe, damit unit-testbar.

Regelmodell (JSONB in email_rules):
  conditions = {
    "match": "all" | "any",          # Default "all"
    "from_domain": "gericht.bayern.de",
    "from_address": "kontakt@...",
    "subject_contains": ["Mahnung", "Frist"],
    "body_contains": ["Aktenzeichen"]
  }
  actions = {
    "file_to_matter": 42,
    "set_confidential": true,
    "flag_review": true,
    "create_ticket": {"title": "...", "ticket_type": "follow_up"}
  }

Die erste passende Regel (nach aufsteigender priority) gewinnt.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class IncomingEmail:
    from_address: str
    subject: str
    body_text: str


@dataclass
class RuleMatch:
    rule_id: int
    actions: dict
    priority: int


def _domain_of(address: str) -> str:
    return address.rsplit("@", 1)[-1].lower() if "@" in address else ""


def _check_condition(conditions: dict, email: IncomingEmail) -> bool:
    """True wenn die Regelbedingungen auf die E-Mail zutreffen."""
    checks: list[bool] = []

    if "from_domain" in conditions:
        checks.append(_domain_of(email.from_address) == conditions["from_domain"].lower())

    if "from_address" in conditions:
        checks.append(email.from_address.lower() == conditions["from_address"].lower())

    if "subject_contains" in conditions:
        subj = (email.subject or "").lower()
        keywords = conditions["subject_contains"]
        # any keyword in subject satisfies this single check
        checks.append(any(k.lower() in subj for k in keywords))

    if "body_contains" in conditions:
        body = (email.body_text or "").lower()
        keywords = conditions["body_contains"]
        checks.append(any(k.lower() in body for k in keywords))

    if not checks:
        return False  # leere Bedingung greift nie automatisch

    match_mode = conditions.get("match", "all")
    return all(checks) if match_mode == "all" else any(checks)


def evaluate_rules(rules: list[tuple[int, dict, dict, int]], email: IncomingEmail) -> RuleMatch | None:
    """
    rules: Liste von (rule_id, conditions, actions, priority), bereits nach
    priority aufsteigend sortiert. Gibt die erste passende Regel zurück.
    """
    for rule_id, conditions, actions, priority in rules:
        if _check_condition(conditions, email):
            return RuleMatch(rule_id=rule_id, actions=actions, priority=priority)
    return None


# Keywords, die auf vertrauliche Inhalte hindeuten (Default; pro Kanzlei erweiterbar)
CONFIDENTIAL_KEYWORDS = [
    "vertraulich", "streng vertraulich", "anwaltsgeheimnis",
    "ärztliches attest", "diagnose", "vergleichsangebot",
]


def detect_confidential(email: IncomingEmail) -> bool:
    haystack = f"{email.subject or ''} {email.body_text or ''}".lower()
    return any(k in haystack for k in CONFIDENTIAL_KEYWORDS)
