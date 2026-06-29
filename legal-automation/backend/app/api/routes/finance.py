import os
from decimal import Decimal

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
from sqlalchemy import select

from app.core.config import get_settings
from app.core.deps import DB, require_permission
from app.models.finance import ImportBatch, MassTransaction
from app.schemas.finance import (
    ImportBatchResponse,
    ImportReportResponse,
    InsVVCalcRequest,
    InsVVCalcResponse,
    MassAccountBalance,
    MassAccountCreate,
    MassAccountResponse,
    RVGCalcRequest,
    RVGCalcResponse,
    TransactionListResponse,
    TransactionUpdate,
)
from app.schemas.finance import FeePosition
from app.services import bank_import_service, mass_account_service
from app.services import insvv_calculator, rvg_calculator

router = APIRouter(prefix="/finance", tags=["finance"])

ALLOWED_EXT = {".xml", ".sta", ".txt", ".mt940", ".940"}


@router.post("/mass-accounts", response_model=MassAccountResponse, status_code=status.HTTP_201_CREATED)
async def create_mass_account(
    data: MassAccountCreate,
    db: DB,
    current_user=Depends(require_permission("finance.write")),
):
    account = await mass_account_service.create_account(
        db, matter_id=data.matter_id, iban=data.iban, created_by_id=current_user.id,
        bic=data.bic, bank_name=data.bank_name, account_label=data.account_label,
        account_type=data.account_type, opening_balance=data.opening_balance,
    )
    return account


@router.get("/mass-accounts", response_model=list[MassAccountResponse])
async def list_mass_accounts(
    db: DB,
    matter_id: int | None = Query(None),
    current_user=Depends(require_permission("finance.read")),
):
    return await mass_account_service.list_accounts(db, matter_id=matter_id)


@router.get("/mass-accounts/{account_id}/balance", response_model=MassAccountBalance)
async def account_balance(
    account_id: int,
    db: DB,
    current_user=Depends(require_permission("finance.read")),
):
    acc = await mass_account_service.get_account(db, account_id)
    if not acc:
        raise HTTPException(status_code=404, detail="Massekonto nicht gefunden")
    balance = await mass_account_service.current_balance(db, account_id)
    return MassAccountBalance(
        account_id=acc.id, matter_id=acc.matter_id,
        opening_balance=Decimal(acc.opening_balance), current_balance=balance, currency=acc.currency,
    )


@router.post("/import", response_model=ImportReportResponse)
async def import_bank_statement(
    db: DB,
    file: UploadFile = File(...),
    account_id: int | None = Form(None),
    current_user=Depends(require_permission("finance.write")),
):
    ext = os.path.splitext(file.filename or "")[1].lower()
    if ext and ext not in ALLOWED_EXT:
        raise HTTPException(status_code=422, detail=f"Nicht unterstütztes Format: {ext}")
    content = await file.read()
    if not content:
        raise HTTPException(status_code=422, detail="Leere Datei")

    report = await bank_import_service.import_statement(
        db, filename=file.filename or "statement", content=content,
        imported_by_id=current_user.id, fallback_account_id=account_id,
    )
    if report.num_assigned == 0 and report.num_transactions > 0 and report.num_duplicates == 0:
        # Nothing could be assigned: likely IBAN not registered as a Massekonto
        raise HTTPException(
            status_code=422,
            detail="Keine Buchung zugeordnet — IBAN des Auszugs ist keinem Massekonto zugeordnet. "
                   "Konto anlegen oder account_id angeben.",
        )
    return ImportReportResponse(**report.__dict__)


@router.get("/transactions", response_model=TransactionListResponse)
async def list_transactions(
    db: DB,
    account_id: int | None = Query(None),
    matter_id: int | None = Query(None),
    category: str | None = Query(None),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=200),
    current_user=Depends(require_permission("finance.read")),
):
    items, total = await mass_account_service.list_transactions(
        db, account_id=account_id, matter_id=matter_id, category=category, page=page, page_size=page_size
    )
    return TransactionListResponse(items=items, total=total, page=page, page_size=page_size)


@router.patch("/transactions/{transaction_id}", response_model=TransactionUpdate)
async def update_transaction(
    transaction_id: int,
    data: TransactionUpdate,
    db: DB,
    current_user=Depends(require_permission("finance.write")),
):
    result = await db.execute(select(MassTransaction).where(MassTransaction.id == transaction_id))
    tx = result.scalar_one_or_none()
    if not tx:
        raise HTTPException(status_code=404, detail="Buchung nicht gefunden")
    if data.category is not None:
        tx.category = data.category
    if data.mass_account_id is not None:
        acc = await mass_account_service.get_account(db, data.mass_account_id)
        if not acc:
            raise HTTPException(status_code=404, detail="Zielkonto nicht gefunden")
        tx.mass_account_id = acc.id
        tx.matter_id = acc.matter_id
    await db.commit()
    return data


@router.get("/import-batches", response_model=list[ImportBatchResponse])
async def list_import_batches(
    db: DB,
    current_user=Depends(require_permission("finance.read")),
):
    result = await db.execute(select(ImportBatch).order_by(ImportBatch.id.desc()).limit(100))
    return result.scalars().all()


# --- Vergütungsrechner (reine Berechnung, finance.read) ---

@router.post("/insvv/calculate", response_model=InsVVCalcResponse)
async def calculate_insvv_endpoint(
    data: InsVVCalcRequest,
    current_user=Depends(require_permission("finance.read")),
):
    result = insvv_calculator.calculate_insvv(
        data.berechnungsgrundlage,
        zuschlaege=[(f.name, f.percent) for f in data.zuschlaege],
        abschlaege=[(f.name, f.percent) for f in data.abschlaege],
        anzahl_glaeubiger=data.anzahl_glaeubiger,
        auslagen=data.auslagen,
        vat_rate=data.vat_rate,
        mindestverguetung_override=data.mindestverguetung_override,
    )
    return InsVVCalcResponse(
        berechnungsgrundlage=result.berechnungsgrundlage,
        regelverguetung=result.regelverguetung,
        adjustments=[
            FeePosition(name=a.name, percent=a.percent, amount=a.amount) for a in result.adjustments
        ],
        verguetung_nach_anpassung=result.verguetung_nach_anpassung,
        mindestverguetung=result.mindestverguetung,
        mindestverguetung_angewandt=result.mindestverguetung_angewandt,
        auslagen=result.auslagen,
        netto=result.netto,
        umsatzsteuer=result.umsatzsteuer,
        brutto=result.brutto,
    )


@router.post("/rvg/calculate", response_model=RVGCalcResponse)
async def calculate_rvg_endpoint(
    data: RVGCalcRequest,
    current_user=Depends(require_permission("finance.read")),
):
    result = rvg_calculator.calculate_rvg(
        data.gegenstandswert,
        [(f.name, f.percent) for f in data.fees],
        add_auslagenpauschale=data.add_auslagenpauschale,
        vat_rate=data.vat_rate,
    )
    return RVGCalcResponse(
        gegenstandswert=result.gegenstandswert,
        wertgebuehr_1_0=result.wertgebuehr_1_0,
        positions=[
            FeePosition(name=p.name, factor=p.factor, amount=p.amount) for p in result.positions
        ],
        gebuehren_summe=result.gebuehren_summe,
        auslagenpauschale=result.auslagenpauschale,
        netto=result.netto,
        umsatzsteuer=result.umsatzsteuer,
        brutto=result.brutto,
    )
