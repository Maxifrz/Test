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
};
