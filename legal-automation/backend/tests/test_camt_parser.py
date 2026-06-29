"""Tests für den CAMT.053-Parser."""
from datetime import date
from decimal import Decimal

from app.services.camt_parser import parse_camt053

CAMT = """<?xml version="1.0" encoding="UTF-8"?>
<Document xmlns="urn:iso:std:iso:20022:tech:xsd:camt.053.001.02">
  <BkToCstmrStmt>
    <Stmt>
      <Acct><Id><IBAN>DE02100100109307118603</IBAN></Id></Acct>
      <Bal>
        <Tp><CdOrPrtry><Cd>OPBD</Cd></CdOrPrtry></Tp>
        <Amt Ccy="EUR">1000.00</Amt><CdtDbtInd>CRDT</CdtDbtInd>
      </Bal>
      <Bal>
        <Tp><CdOrPrtry><Cd>CLBD</Cd></CdOrPrtry></Tp>
        <Amt Ccy="EUR">1350.50</Amt><CdtDbtInd>CRDT</CdtDbtInd>
      </Bal>
      <Ntry>
        <Amt Ccy="EUR">500.00</Amt>
        <CdtDbtInd>CRDT</CdtDbtInd>
        <BookgDt><Dt>2024-03-15</Dt></BookgDt>
        <ValDt><Dt>2024-03-15</Dt></ValDt>
        <AcctSvcrRef>REF-001</AcctSvcrRef>
        <NtryDtls><TxDtls>
          <Refs><EndToEndId>E2E-001</EndToEndId></Refs>
          <RmtInf><Ustrd>Zahlung Schuldner Mueller</Ustrd></RmtInf>
          <RltdPties>
            <Dbtr><Nm>Mueller GmbH</Nm></Dbtr>
            <DbtrAcct><Id><IBAN>DE89370400440532013000</IBAN></Id></DbtrAcct>
          </RltdPties>
        </TxDtls></NtryDtls>
      </Ntry>
      <Ntry>
        <Amt Ccy="EUR">149.50</Amt>
        <CdtDbtInd>DBIT</CdtDbtInd>
        <BookgDt><Dt>2024-03-16</Dt></BookgDt>
        <NtryDtls><TxDtls>
          <RmtInf><Ustrd>Gerichtskosten AG Muenchen</Ustrd></RmtInf>
          <RltdPties>
            <Cdtr><Nm>Justizkasse Bayern</Nm></Cdtr>
          </RltdPties>
        </TxDtls></NtryDtls>
      </Ntry>
    </Stmt>
  </BkToCstmrStmt>
</Document>
"""


def test_camt_account_and_balances():
    s = parse_camt053(CAMT)
    assert s.account_iban == "DE02100100109307118603"
    assert s.opening_balance == Decimal("1000.00")
    assert s.closing_balance == Decimal("1350.50")
    assert len(s.transactions) == 2


def test_camt_credit_entry():
    s = parse_camt053(CAMT)
    t = s.transactions[0]
    assert t.direction == "in"
    assert t.amount == Decimal("500.00")
    assert t.booking_date == date(2024, 3, 15)
    assert t.purpose == "Zahlung Schuldner Mueller"
    assert t.counterparty_name == "Mueller GmbH"
    assert t.counterparty_iban == "DE89370400440532013000"
    assert t.end_to_end_id == "E2E-001"
    assert t.bank_reference == "REF-001"


def test_camt_debit_entry():
    s = parse_camt053(CAMT)
    t = s.transactions[1]
    assert t.direction == "out"
    assert t.amount == Decimal("149.50")
    assert t.counterparty_name == "Justizkasse Bayern"
    assert t.signed_amount == Decimal("-149.50")


def test_camt_balances_reconcile_with_entries():
    s = parse_camt053(CAMT)
    computed = s.opening_balance + sum(t.signed_amount for t in s.transactions)
    assert computed == s.closing_balance  # 1000 + 500 - 149.50 = 1350.50


def test_camt_empty_document():
    s = parse_camt053('<?xml version="1.0"?><Document xmlns="x"></Document>')
    assert s.account_iban is None
    assert s.transactions == []
