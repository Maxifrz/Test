from decimal import Decimal

from fastapi import APIRouter, Depends, HTTPException, Query, status
from sqlalchemy import select

from app.core.deps import DB, ensure_matter_access, require_permission
from app.schemas.insolvency import (
    ClaimCreate,
    ClaimResponse,
    ClaimTableResponse,
    ClaimTotals,
    ClaimUpdate,
    DistributionItemResponse,
    DistributionRequest,
    DistributionResponse,
    PortalEnableResponse,
)
from app.services import insolvency_service
from app.services.matter_service import get_matter

router = APIRouter(prefix="/insolvency", tags=["insolvency"])


async def ensure_matter_access_and_load(db, current_user, matter_id: int):
    """Prueft den Aktenzugriff und liefert die Akte (fuer PDF-Kopfdaten)."""
    from app.models.matter import Matter

    await ensure_matter_access(db, current_user, matter_id)
    matter = (await db.execute(
        select(Matter).where(Matter.id == matter_id, Matter.deleted_at.is_(None))
    )).scalar_one_or_none()
    if matter is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Akte nicht gefunden")
    return matter


@router.post("/claims", response_model=ClaimResponse, status_code=status.HTTP_201_CREATED)
async def create_claim(
    data: ClaimCreate,
    db: DB,
    current_user=Depends(require_permission("finance.write")),
):
    await ensure_matter_access(db, current_user, data.matter_id)
    claim = await insolvency_service.create_claim(
        db, matter_id=data.matter_id, creditor_name=data.creditor_name,
        claim_amount=data.claim_amount, rank=data.rank, creditor_email=data.creditor_email,
        creditor_address=data.creditor_address, creditor_reference=data.creditor_reference,
        claim_reason=data.claim_reason, created_by_id=current_user.id,
    )
    return claim


@router.get("/claims", response_model=ClaimTableResponse)
async def list_claims(
    db: DB,
    matter_id: int = Query(...),
    current_user=Depends(require_permission("finance.read")),
):
    await ensure_matter_access(db, current_user, matter_id)
    items = await insolvency_service.list_claims(db, matter_id)
    totals = await insolvency_service.table_totals(db, matter_id)
    return ClaimTableResponse(items=items, totals=ClaimTotals(**totals))


@router.patch("/claims/{claim_id}", response_model=ClaimResponse)
async def update_claim(
    claim_id: int,
    data: ClaimUpdate,
    db: DB,
    current_user=Depends(require_permission("finance.write")),
):
    claim = await insolvency_service.get_claim(db, claim_id)
    if not claim:
        raise HTTPException(status_code=404, detail="Forderung nicht gefunden")
    await ensure_matter_access(db, current_user, claim.matter_id)
    return await insolvency_service.update_claim(db, claim, data.model_dump(exclude_unset=True))


@router.post("/distribution", response_model=DistributionResponse)
async def distribution(
    data: DistributionRequest,
    db: DB,
    current_user=Depends(require_permission("finance.write")),
):
    await ensure_matter_access(db, current_user, data.matter_id)

    # Die Kennzahlen (Summen, Ränge, Hinweise) kommen immer aus der
    # Berechnung; beim Persistieren wird zusätzlich gespeichert.
    preview = await insolvency_service.preview_distribution(
        db, data.matter_id, data.distributable_amount, data.mass_liabilities
    )

    def to_response(distribution_id: int | None) -> DistributionResponse:
        return DistributionResponse(
            distribution_id=distribution_id,
            matter_id=data.matter_id,
            gross_estate=preview.gross_estate,
            mass_liabilities=preview.mass_liabilities,
            distributable=preview.distributable,
            total_38=preview.total_38,
            total_39=preview.total_39,
            quote_38_pct=preview.quote_38_pct,
            distributed_sum=preview.distributed_sum,
            withheld_sum=preview.withheld_sum,
            remainder=preview.remainder,
            rank_quotes=preview.rank_quotes,
            notes=preview.notes,
            items=[
                DistributionItemResponse(
                    claim_id=i.claim_id,
                    established_amount=i.established_amount,
                    participating_amount=i.participating_amount,
                    amount=i.amount,
                    quote_pct=i.quote_pct,
                    rank=i.rank,
                    withheld=i.withheld,
                )
                for i in preview.items
            ],
        )

    if data.persist:
        dist = await insolvency_service.run_distribution(
            db, matter_id=data.matter_id, distributable=data.distributable_amount,
            distribution_type=data.distribution_type, created_by_id=current_user.id,
            mass_liabilities=data.mass_liabilities,
        )
        return to_response(dist.id)

    return to_response(None)


@router.post("/matters/{matter_id}/creditor-portal", response_model=PortalEnableResponse)
async def enable_portal(
    matter_id: int,
    db: DB,
    current_user=Depends(require_permission("finance.write")),
):
    matter = await get_matter(db, matter_id)
    if not matter:
        raise HTTPException(status_code=404, detail="Akte nicht gefunden")
    await ensure_matter_access(db, current_user, matter_id)
    token = await insolvency_service.enable_creditor_portal(db, matter)
    return PortalEnableResponse(
        matter_id=matter_id, creditor_portal_token=token,
        submit_path=f"/api/public/creditor-claims/{token}",
    )


@router.get("/matters/{matter_id}/tabellenauszug")
async def tabellenauszug(
    matter_id: int,
    db: DB,
    glaeubiger_name: str | None = Query(None, max_length=255),
    current_user=Depends(require_permission("finance.read")),
):
    """
    Tabellenauszug nach § 178 Abs. 3 InsO als PDF.

    Fuer festgestellte Forderungen wirkt die Eintragung wie ein rechtskraeftiges
    Urteil -- der Auszug ist ein Vollstreckungstitel, kein Beiblatt. Massgeblich
    bleibt die bei Gericht gefuehrte Tabelle; das PDF gibt den Stand zum
    Ausstellungszeitpunkt wieder.
    """
    from fastapi.responses import Response

    from app.services.tabellenauszug_pdf import (
        TabellenauszugContext,
        TabellenRow,
        render_pdf,
    )

    matter = await ensure_matter_access_and_load(db, current_user, matter_id)
    claims = await insolvency_service.list_claims(db, matter_id)
    if not claims:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Fuer dieses Verfahren sind keine Forderungen angemeldet",
        )

    rows = [
        TabellenRow(
            lfd_nr=c.claim_number or idx,
            creditor_name=c.creditor_name,
            claim_amount=Decimal(c.claim_amount),
            established_amount=Decimal(c.established_amount) if c.established_amount is not None else None,
            rank=c.rank,
            status=c.status,
            dispute_reason=c.dispute_reason,
        )
        for idx, c in enumerate(claims, start=1)
    ]

    pdf = render_pdf(
        rows,
        TabellenauszugContext(
            gericht=matter.court_name,
            aktenzeichen=matter.court_file_ref,
            schuldner=matter.title,
            verwalter=None,
            matter_number=matter.matter_number,
            pruefungstermin=None,
            glaeubiger_name=glaeubiger_name,
        ),
    )
    filename = f"tabellenauszug_{matter.matter_number or matter_id}.pdf"
    return Response(
        content=pdf,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
