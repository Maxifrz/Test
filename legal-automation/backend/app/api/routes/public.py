"""
Öffentliche, NICHT authentifizierte Endpunkte für das Gläubiger-Portal.

Sicherheit: Zugang ausschließlich über das unguessbare, verfahrensspezifische
Token (`matters.creditor_portal_token`). Es können nur Forderungen ANGEMELDET
werden (Status 'angemeldet', Quelle 'glaeubiger_portal') — keine Lese-/Änderungs-
rechte an bestehenden Daten. Rate-Limiting erfolgt über nginx. Inhalte gelten
als nicht vertrauenswürdig (Untrusted) und werden erst durch Sachbearbeiter
geprüft/festgestellt.
"""
from fastapi import APIRouter, HTTPException, status

from app.core.deps import DB
from app.models.insolvency import RANK_38
from app.schemas.insolvency import PublicClaimSubmit
from app.services import insolvency_service

router = APIRouter(prefix="/public", tags=["public"])


@router.get("/creditor-claims/{token}")
async def portal_info(token: str, db: DB):
    """Minimale Info, ob das Token gültig ist (für die Portal-Startseite)."""
    matter = await insolvency_service.matter_by_portal_token(db, token)
    if not matter:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Ungültiges oder abgelaufenes Token")
    # Keine sensiblen Verfahrensdaten preisgeben — nur Aktenzeichen zur Orientierung
    return {"matter_number": matter.matter_number, "title": matter.title}


@router.post("/creditor-claims/{token}", status_code=status.HTTP_201_CREATED)
async def submit_claim(token: str, data: PublicClaimSubmit, db: DB):
    matter = await insolvency_service.matter_by_portal_token(db, token)
    if not matter:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Ungültiges oder abgelaufenes Token")
    if data.claim_amount <= 0:
        raise HTTPException(status_code=422, detail="Forderungsbetrag muss positiv sein")

    claim = await insolvency_service.create_claim(
        db, matter_id=matter.id, creditor_name=data.creditor_name,
        claim_amount=data.claim_amount, rank=RANK_38, source="glaeubiger_portal",
        creditor_email=data.creditor_email, creditor_address=data.creditor_address,
        creditor_reference=data.creditor_reference, claim_reason=data.claim_reason,
        created_by_id=None,
    )
    return {
        "status": "eingereicht",
        "claim_number": claim.claim_number,
        "hinweis": "Ihre Forderung wurde angemeldet und wird durch die Kanzlei geprüft.",
    }
