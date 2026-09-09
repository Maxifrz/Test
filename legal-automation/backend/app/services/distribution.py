"""
Verteilungsrechner (Insolvenzquote) — reine Logik, unit-testbar.

Rangfolge nach InsO:
  1. Massekosten und sonstige Masseverbindlichkeiten (§§ 53–55 InsO) —
     vorweg zu berichtigen, nehmen NICHT an der Quote teil.
  2. Insolvenzforderungen (§ 38 InsO) — quotal.
  3. Nachrangige Forderungen (§ 39 InsO) — erst bei vollständiger
     Befriedigung der § 38-Forderungen, und dort in der Reihenfolge des
     § 39 Abs. 1 Nr. 1–5: ein niedrigerer Rang erhält nur, was nach voller
     Befriedigung aller höheren Ränge übrig bleibt.

Nicht an der Quote beteiligt:
  - Aussonderung (§ 47 InsO): der Gegenstand gehört nicht zur Masse.
  - Absonderung (§§ 49–51 InsO): der Gläubiger wird aus dem Sicherungsgut
    befriedigt und nimmt nur mit seinem AUSFALL an der Quote teil
    (§ 52 InsO) — dafür `ClaimInput.secured_recovery`.

Bestrittene Forderungen (§ 189 InsO): Für sie ist der auf sie entfallende
Anteil ZURÜCKZUBEHALTEN, solange die Feststellung betrieben wird. Sie werden
deshalb bei der Quotenbildung mitgezählt, aber nicht ausgezahlt.

Cent-genau: Rundungsdifferenzen werden nach dem Verfahren größter Reste
(Hare/Niemeyer) verteilt, sodass die Summe der Auszahlungen exakt der
verteilbaren Masse entspricht und kein Gläubiger mehr als seine Forderung
erhält.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from decimal import ROUND_DOWN, ROUND_HALF_UP, Decimal

CENT = Decimal("0.01")

RANK_38 = "insolvenz_38"      # reguläre Insolvenzgläubiger
RANK_39 = "nachrangig_39"     # nachrangige Insolvenzgläubiger (Sammelrang)

# § 39 Abs. 1 InsO — Unterränge. Kleinere Zahl = höherer Rang.
# Ohne diese Aufteilung landeten alle Nachrangigen in einem Topf und wurden
# gleichmäßig bedient, obwohl Nr. 1 vor Nr. 5 geht.
RANK_39_1 = "nachrangig_39_1"   # nach Eröffnung laufende Zinsen
RANK_39_2 = "nachrangig_39_2"   # Kosten der Verfahrensteilnahme
RANK_39_3 = "nachrangig_39_3"   # Geldstrafen, Bußgelder, Zwangsgelder
RANK_39_4 = "nachrangig_39_4"   # unentgeltliche Leistungen des Schuldners
RANK_39_5 = "nachrangig_39_5"   # Gesellschafterdarlehen

# Reihenfolge der Befriedigung. RANK_39 (ohne Unterrang) wird wie Nr. 5
# behandelt — konservativ, da der niedrigste Rang.
SUBORDINATED_ORDER: tuple[str, ...] = (
    RANK_39_1, RANK_39_2, RANK_39_3, RANK_39_4, RANK_39_5, RANK_39,
)

RANK_LABELS = {
    RANK_38: "§ 38 InsO — Insolvenzforderung",
    RANK_39_1: "§ 39 I Nr. 1 InsO — Zinsen ab Eröffnung",
    RANK_39_2: "§ 39 I Nr. 2 InsO — Kosten der Verfahrensteilnahme",
    RANK_39_3: "§ 39 I Nr. 3 InsO — Geldstrafen/Bußgelder",
    RANK_39_4: "§ 39 I Nr. 4 InsO — unentgeltliche Leistungen",
    RANK_39_5: "§ 39 I Nr. 5 InsO — Gesellschafterdarlehen",
    RANK_39: "§ 39 InsO — nachrangig (ohne Unterrang)",
}


@dataclass
class ClaimInput:
    claim_id: int
    established_amount: Decimal   # festgestellter Betrag
    rank: str = RANK_38
    # § 52 InsO: Bei Absonderungsberechtigten nimmt nur der AUSFALL teil.
    # Betrag, der aus dem Sicherungsgut erlöst wurde (0 = kein Sicherungsgut).
    secured_recovery: Decimal = Decimal("0")
    # § 189 InsO: bestrittene Forderung — zählt für die Quote mit, wird aber
    # zurückbehalten statt ausgezahlt.
    disputed: bool = False

    @property
    def participating_amount(self) -> Decimal:
        """Betrag, mit dem die Forderung an der Verteilung teilnimmt (§ 52 InsO)."""
        return max(Decimal("0"), self.established_amount - self.secured_recovery)


@dataclass
class DistributionItem:
    claim_id: int
    established_amount: Decimal
    # Betrag, mit dem teilgenommen wurde (nach Abzug der Sicherheitenverwertung)
    participating_amount: Decimal
    amount: Decimal              # tatsächlich auszuzahlen
    quote_pct: Decimal
    rank: str = RANK_38
    withheld: bool = False       # zurückbehalten (§ 189 InsO)


@dataclass
class DistributionResult:
    # Rohmasse vor Abzug der Masseverbindlichkeiten
    gross_estate: Decimal
    mass_liabilities: Decimal
    distributable: Decimal              # nach Abzug §§ 53–55
    total_38: Decimal
    total_39: Decimal
    quote_38_pct: Decimal               # Prozentsatz für § 38 (0–100)
    items: list[DistributionItem] = field(default_factory=list)
    distributed_sum: Decimal = Decimal("0.00")
    withheld_sum: Decimal = Decimal("0.00")   # § 189 InsO Rückstellung
    remainder: Decimal = Decimal("0.00")      # nicht verteilte Restmasse
    rank_quotes: dict[str, Decimal] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def _money(v: Decimal) -> Decimal:
    return v.quantize(CENT, rounding=ROUND_HALF_UP)


def _distribute_prorata(
    amount_available: Decimal, claims: list[ClaimInput]
) -> tuple[dict[int, Decimal], Decimal]:
    """
    Verteilt amount_available quotal auf claims (nach participating_amount).

    Rundung nach dem Verfahren größter Reste (Hare/Niemeyer): jeder Gläubiger
    erhält zunächst den abgerundeten Anteil, die verbleibenden Cent gehen an
    die Forderungen mit dem größten Rundungsverlust. Vorher fiel die gesamte
    Rundungsdifferenz der betragsgrößten Forderung zu — willkürlich, und in
    Randfällen konnte sie dadurch mehr als ihren Anspruch erhalten.
    """
    total = sum((c.participating_amount for c in claims), Decimal("0"))
    if total <= 0 or amount_available <= 0:
        return {c.claim_id: Decimal("0.00") for c in claims}, Decimal("0.00")

    capped = _money(min(amount_available, total))
    quote = capped / total

    exact: dict[int, Decimal] = {}
    floors: dict[int, Decimal] = {}
    for c in claims:
        value = c.participating_amount * quote
        exact[c.claim_id] = value
        floors[c.claim_id] = value.quantize(CENT, rounding=ROUND_DOWN)

    distributed = sum(floors.values(), Decimal("0"))
    remaining_cents = int(((capped - distributed) / CENT).to_integral_value())

    if remaining_cents > 0:
        # Größte Reste zuerst; bei Gleichstand entscheidet die claim_id
        # (Determinismus ist hier wichtiger als Fairness in der 3. Nachkommastelle).
        by_remainder = sorted(
            claims,
            key=lambda c: (-(exact[c.claim_id] - floors[c.claim_id]), c.claim_id),
        )
        for c in by_remainder[:remaining_cents]:
            # Nie mehr als die Forderung selbst
            if floors[c.claim_id] + CENT <= c.participating_amount:
                floors[c.claim_id] += CENT

    result = {cid: _money(v) for cid, v in floors.items()}
    return result, _money(sum(result.values(), Decimal("0")))


def _quote_pct(paid: Decimal, base: Decimal) -> Decimal:
    return _money(paid / base * 100) if base > 0 else Decimal("0.00")


def compute_distribution(
    distributable: Decimal,
    claims: list[ClaimInput],
    *,
    mass_liabilities: Decimal = Decimal("0"),
) -> DistributionResult:
    """
    claims: NUR festgestellte oder bestrittene Forderungen (Filterung durch den
    Aufrufer). Aus-/Absonderungsrechte werden über `secured_recovery` abgebildet,
    nicht als eigener Rang.

    `distributable` ist die Rohmasse; `mass_liabilities` (§§ 53–55 InsO,
    einschließlich der Verwaltervergütung) wird vorweg abgezogen. Vorher musste
    der Aufrufer das selbst tun — eine stille Fehlerquelle, weil nichts daran
    erinnerte.
    """
    gross = _money(distributable)
    mass = _money(mass_liabilities)
    notes: list[str] = []

    available = gross - mass
    if available < 0:
        # Masseunzulänglichkeit (§ 208 InsO): für die Insolvenzgläubiger bleibt
        # nichts. Die Rangfolge INNERHALB der Masseverbindlichkeiten (§ 209
        # InsO) ist ein eigenes Verfahren und wird hier nicht gerechnet.
        notes.append(
            "Masseunzulänglichkeit: die Masseverbindlichkeiten übersteigen die "
            "Masse (§ 208 InsO). Keine Verteilung an Insolvenzgläubiger; die "
            "Rangfolge nach § 209 InsO ist gesondert zu ermitteln."
        )
        available = Decimal("0.00")

    claims_38 = [c for c in claims if c.rank == RANK_38]
    subordinated = [c for c in claims if c.rank != RANK_38]

    total_38 = sum((c.participating_amount for c in claims_38), Decimal("0"))
    total_39 = sum((c.participating_amount for c in subordinated), Decimal("0"))

    items: list[DistributionItem] = []
    distributed = Decimal("0.00")
    withheld = Decimal("0.00")
    rank_quotes: dict[str, Decimal] = {}

    def emit(claim: ClaimInput, amount: Decimal) -> None:
        nonlocal distributed, withheld
        # § 189 InsO: bestrittene Forderungen nehmen an der Quote teil, der
        # Betrag wird aber zurückbehalten, bis die Feststellung geklärt ist.
        if claim.disputed:
            withheld += amount
            payout = Decimal("0.00")
        else:
            distributed += amount
            payout = amount
        items.append(
            DistributionItem(
                claim_id=claim.claim_id,
                established_amount=claim.established_amount,
                participating_amount=claim.participating_amount,
                amount=payout,
                quote_pct=_quote_pct(amount, claim.participating_amount),
                rank=claim.rank,
                withheld=claim.disputed,
            )
        )

    # --- Stufe 1: § 38 InsO ---
    amounts_38, assigned_38 = _distribute_prorata(available, claims_38)
    for c in claims_38:
        emit(c, amounts_38.get(c.claim_id, Decimal("0.00")))

    if total_38 > 0:
        quote_38_pct = Decimal("100.00") if available >= total_38 else _quote_pct(available, total_38)
    else:
        quote_38_pct = Decimal("0.00")
    rank_quotes[RANK_38] = quote_38_pct

    rest = available - assigned_38

    # --- Stufe 2: § 39 InsO, Unterrang für Unterrang ---
    if rest > 0 and subordinated:
        for rank in SUBORDINATED_ORDER:
            group = [c for c in subordinated if c.rank == rank]
            if not group:
                continue
            if rest <= 0:
                for c in group:
                    emit(c, Decimal("0.00"))
                rank_quotes[rank] = Decimal("0.00")
                continue
            group_total = sum((c.participating_amount for c in group), Decimal("0"))
            amounts, assigned = _distribute_prorata(rest, group)
            for c in group:
                emit(c, amounts.get(c.claim_id, Decimal("0.00")))
            rank_quotes[rank] = (
                Decimal("100.00") if rest >= group_total else _quote_pct(rest, group_total)
            )
            rest -= assigned
        notes.append(
            "Nachrangige Forderungen wurden in der Reihenfolge des § 39 Abs. 1 "
            "InsO bedient; ein niedrigerer Rang erhält nur den nach voller "
            "Befriedigung der höheren Ränge verbleibenden Betrag."
        )
    else:
        for c in subordinated:
            emit(c, Decimal("0.00"))
            rank_quotes.setdefault(c.rank, Decimal("0.00"))

    if withheld > 0:
        notes.append(
            f"{_money(withheld)} € entfallen auf bestrittene Forderungen und sind "
            "nach § 189 InsO zurückzubehalten, bis die Feststellung geklärt ist."
        )
    if any(c.secured_recovery > 0 for c in claims):
        notes.append(
            "Absonderungsberechtigte nehmen nur mit ihrem Ausfall teil (§ 52 InsO); "
            "der aus dem Sicherungsgut erlöste Betrag ist abgezogen."
        )

    return DistributionResult(
        gross_estate=gross,
        mass_liabilities=mass,
        distributable=_money(available),
        total_38=_money(total_38),
        total_39=_money(total_39),
        quote_38_pct=quote_38_pct,
        items=items,
        distributed_sum=_money(distributed),
        withheld_sum=_money(withheld),
        remainder=_money(available - distributed - withheld),
        rank_quotes=rank_quotes,
        notes=notes,
    )
