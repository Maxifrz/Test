from decimal import Decimal

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.core.deps import DB, ensure_matter_access, require_permission
from app.models.insolvency import Distribution, DistributionItem
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
from sqlalchemy import select

router = APIRouter(prefix="/insolvency", tags=["insolvency"])


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
    if data.persist:
        dist = await insolvency_service.run_distribution(
            db, matter_id=data.matter_id, distributable=data.distributable_amount,
            distribution_type=data.distribution_type, created_by_id=current_user.id,
        )
        # reload items
        result = await db.execute(select(DistributionItem).where(DistributionItem.distribution_id == dist.id))
        items = result.scalars().all()
        # totals for response
        preview = await insolvency_service.preview_distribution(db, data.matter_id, data.distributable_amount)
        return DistributionResponse(
            distribution_id=dist.id, matter_id=data.matter_id,
            distributable=Decimal(dist.distributable_amount), total_38=preview.total_38,
            total_39=preview.total_39, quote_38_pct=Decimal(dist.quote_38_pct),
            distributed_sum=Decimal(dist.distributed_sum), remainder=Decimal(dist.remainder),
            items=[DistributionItemResponse(claim_id=i.claim_id, established_amount=Decimal(i.established_amount),
                                            amount=Decimal(i.amount), quote_pct=Decimal(i.quote_pct)) for i in items],
        )

    preview = await insolvency_service.preview_distribution(db, data.matter_id, data.distributable_amount)
    return DistributionResponse(
        matter_id=data.matter_id, distributable=preview.distributable, total_38=preview.total_38,
        total_39=preview.total_39, quote_38_pct=preview.quote_38_pct,
        distributed_sum=preview.distributed_sum, remainder=preview.remainder,
        items=[DistributionItemResponse(claim_id=i.claim_id, established_amount=i.established_amount,
                                        amount=i.amount, quote_pct=i.quote_pct) for i in preview.items],
    )


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
