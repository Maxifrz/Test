import api from "./client";

export interface MassAccount {
  id: number;
  matter_id: number;
  iban: string;
  bic: string | null;
  bank_name: string | null;
  account_label: string | null;
  account_type: string;
  opening_balance: string;
  currency: string;
  is_active: boolean;
  created_at: string;
}

export interface MassAccountBalance {
  account_id: number;
  matter_id: number;
  opening_balance: string;
  current_balance: string;
  currency: string;
}

export interface Transaction {
  id: number;
  mass_account_id: number;
  matter_id: number;
  import_batch_id: number | null;
  booking_date: string | null;
  value_date: string | null;
  amount: string;
  direction: string;
  currency: string;
  purpose: string | null;
  counterparty_name: string | null;
  counterparty_iban: string | null;
  category: string;
  is_reconciled: boolean;
}

export interface TransactionList {
  items: Transaction[];
  total: number;
  page: number;
  page_size: number;
}

export interface ImportReport {
  batch_id: number;
  num_transactions: number;
  num_assigned: number;
  num_unassigned: number;
  num_duplicates: number;
  reconciled: boolean;
  statement_closing: string | null;
  computed_closing: string | null;
}

export interface FeePosition {
  name: string;
  percent?: string | null;
  factor?: string | null;
  amount: string;
}

export interface InsVVResult {
  berechnungsgrundlage: string;
  regelverguetung: string;
  adjustments: FeePosition[];
  verguetung_nach_anpassung: string;
  mindestverguetung: string;
  mindestverguetung_angewandt: boolean;
  auslagen: string;
  netto: string;
  umsatzsteuer: string;
  brutto: string;
}

export interface RVGResult {
  gegenstandswert: string;
  wertgebuehr_1_0: string;
  positions: FeePosition[];
  gebuehren_summe: string;
  auslagenpauschale: string;
  netto: string;
  umsatzsteuer: string;
  brutto: string;
}

export const financeApi = {
  listAccounts: (matter_id?: number) =>
    api.get<MassAccount[]>("/finance/mass-accounts", { params: { matter_id } }).then((r) => r.data),

  createAccount: (data: {
    matter_id: number;
    iban: string;
    bic?: string;
    bank_name?: string;
    account_label?: string;
    account_type?: string;
    opening_balance?: string;
  }) => api.post<MassAccount>("/finance/mass-accounts", data).then((r) => r.data),

  balance: (accountId: number) =>
    api.get<MassAccountBalance>(`/finance/mass-accounts/${accountId}/balance`).then((r) => r.data),

  importStatement: (file: File, accountId?: number) => {
    const form = new FormData();
    form.append("file", file);
    if (accountId != null) form.append("account_id", String(accountId));
    return api
      .post<ImportReport>("/finance/import", form, {
        headers: { "Content-Type": "multipart/form-data" },
      })
      .then((r) => r.data);
  },

  transactions: (params?: { account_id?: number; matter_id?: number; category?: string; page?: number }) =>
    api.get<TransactionList>("/finance/transactions", { params }).then((r) => r.data),

  updateTransaction: (id: number, data: { category?: string; mass_account_id?: number }) =>
    api.patch(`/finance/transactions/${id}`, data).then((r) => r.data),

  calcInsVV: (data: {
    berechnungsgrundlage: string;
    zuschlaege?: { name: string; percent: string }[];
    abschlaege?: { name: string; percent: string }[];
    anzahl_glaeubiger?: number;
    auslagen?: string;
    vat_rate?: string;
  }) => api.post<InsVVResult>("/finance/insvv/calculate", data).then((r) => r.data),

  calcRVG: (data: {
    gegenstandswert: string;
    fees: { name: string; percent: string }[];
    add_auslagenpauschale?: boolean;
    vat_rate?: string;
  }) => api.post<RVGResult>("/finance/rvg/calculate", data).then((r) => r.data),

  antragPdf: (data: InsVVAntragRequest) =>
    api.post("/finance/insvv/antrag-pdf", data, { responseType: "blob" }).then((r) => r.data as Blob),
};

export interface InsVVAntragRequest {
  berechnungsgrundlage: string;
  zuschlaege?: { name: string; percent: string }[];
  anzahl_glaeubiger?: number;
  auslagen?: string;
  gericht?: string;
  aktenzeichen?: string;
  schuldner?: string;
  verwalter?: string;
  matter_number?: string;
}

export interface Claim {
  id: number;
  matter_id: number;
  claim_number: number | null;
  creditor_name: string;
  creditor_email: string | null;
  creditor_address: string | null;
  creditor_reference: string | null;
  claim_amount: string;
  established_amount: string | null;
  claim_reason: string | null;
  rank: string;
  status: string;
  dispute_reason: string | null;
  source: string;
  filed_at: string;
}

export interface ClaimTotals {
  count: number;
  sum_angemeldet: string;
  sum_festgestellt: string;
  count_festgestellt: number;
  count_bestritten: number;
}

export interface ClaimTable {
  items: Claim[];
  totals: ClaimTotals;
}

export interface DistributionItem {
  claim_id: number;
  established_amount: string;
  amount: string;
  quote_pct: string;
}

export interface DistributionResult {
  distribution_id: number | null;
  matter_id: number;
  distributable: string;
  total_38: string;
  total_39: string;
  quote_38_pct: string;
  distributed_sum: string;
  remainder: string;
  items: DistributionItem[];
}

export const insolvencyApi = {
  listClaims: (matter_id: number) =>
    api.get<ClaimTable>("/insolvency/claims", { params: { matter_id } }).then((r) => r.data),

  createClaim: (data: {
    matter_id: number;
    creditor_name: string;
    claim_amount: string;
    rank?: string;
    creditor_email?: string;
    claim_reason?: string;
  }) => api.post<Claim>("/insolvency/claims", data).then((r) => r.data),

  updateClaim: (id: number, data: { status?: string; established_amount?: string; dispute_reason?: string; rank?: string }) =>
    api.patch<Claim>(`/insolvency/claims/${id}`, data).then((r) => r.data),

  distribution: (data: { matter_id: number; distributable_amount: string; distribution_type?: string; persist?: boolean }) =>
    api.post<DistributionResult>("/insolvency/distribution", data).then((r) => r.data),

  enablePortal: (matter_id: number) =>
    api.post<{ matter_id: number; creditor_portal_token: string; submit_path: string }>(
      `/insolvency/matters/${matter_id}/creditor-portal`, {}
    ).then((r) => r.data),
};
