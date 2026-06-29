import { useState, useRef } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { financeApi, MassAccount, ImportReport } from "../lib/api/finance";

const CATEGORY_LABELS: Record<string, string> = {
  massezufluss: "Massezufluss",
  masseverbindlichkeit: "Masseverbindlichkeit",
  gerichtskosten: "Gerichtskosten",
  verwaltverguetung: "Verwaltervergütung",
  sonstiges: "Sonstiges",
  unassigned: "Nicht zugeordnet",
};
const ACCOUNT_TYPES = ["sonderkonto", "anderkonto", "treuhand"];

function eur(v: string | number) {
  return new Intl.NumberFormat("de-DE", { style: "currency", currency: "EUR" }).format(Number(v));
}

export default function FinancePage() {
  const qc = useQueryClient();
  const [selectedAccount, setSelectedAccount] = useState<number | null>(null);
  const [showCreate, setShowCreate] = useState(false);
  const [report, setReport] = useState<ImportReport | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  const { data: accounts } = useQuery({
    queryKey: ["mass-accounts"],
    queryFn: () => financeApi.listAccounts(),
  });

  const { data: balance } = useQuery({
    queryKey: ["mass-balance", selectedAccount],
    queryFn: () => financeApi.balance(selectedAccount!),
    enabled: selectedAccount !== null,
  });

  const { data: txs } = useQuery({
    queryKey: ["mass-transactions", selectedAccount],
    queryFn: () => financeApi.transactions({ account_id: selectedAccount!, page: 1 }),
    enabled: selectedAccount !== null,
  });

  const importMut = useMutation({
    mutationFn: (file: File) => financeApi.importStatement(file, selectedAccount ?? undefined),
    onSuccess: (r) => {
      setReport(r);
      qc.invalidateQueries({ queryKey: ["mass-transactions"] });
      qc.invalidateQueries({ queryKey: ["mass-balance"] });
    },
    onError: (e: any) => alert(e?.response?.data?.detail ?? "Import fehlgeschlagen"),
  });

  return (
    <div className="p-6 max-w-6xl mx-auto">
      <div className="flex items-center justify-between mb-6">
        <div>
          <h1 className="text-2xl font-semibold text-gray-900">Finanzen — Massekonten</h1>
          <p className="text-sm text-gray-500">Bankauszug-Import (CAMT.053 / MT940), automatische Verfahrenszuordnung</p>
        </div>
        <div className="flex gap-2">
          <input ref={fileRef} type="file" accept=".xml,.sta,.txt,.mt940,.940" className="hidden"
            onChange={(e) => { const f = e.target.files?.[0]; if (f) importMut.mutate(f); e.target.value = ""; }} />
          <button onClick={() => fileRef.current?.click()} className="px-3 py-2 border border-gray-300 text-gray-700 rounded-md text-sm hover:bg-gray-50">
            Auszug importieren
          </button>
          <button onClick={() => setShowCreate(true)} className="px-4 py-2 bg-blue-600 text-white rounded-md text-sm font-medium hover:bg-blue-700">
            + Massekonto
          </button>
        </div>
      </div>

      {report && (
        <div className={`mb-4 p-3 rounded border text-sm ${report.reconciled ? "bg-green-50 border-green-200 text-green-800" : "bg-amber-50 border-amber-200 text-amber-800"}`}>
          Import: {report.num_assigned} zugeordnet, {report.num_duplicates} Duplikate, {report.num_unassigned} offen.{" "}
          {report.statement_closing != null && (
            <>Saldo-Abgleich: {report.reconciled ? "stimmt überein ✓" : `Abweichung (Auszug ${eur(report.statement_closing)} vs. berechnet ${eur(report.computed_closing ?? 0)})`}</>
          )}
          <button onClick={() => setReport(null)} className="ml-2 underline">schließen</button>
        </div>
      )}

      <div className="grid grid-cols-4 gap-4">
        <div className="col-span-1 shadow ring-1 ring-black ring-opacity-5 rounded-lg bg-white">
          <div className="p-3 text-xs font-medium text-gray-500 uppercase border-b">Konten</div>
          <ul className="divide-y divide-gray-200">
            {accounts?.map((a) => (
              <li key={a.id} onClick={() => setSelectedAccount(a.id)} className={`p-3 cursor-pointer hover:bg-gray-50 ${selectedAccount === a.id ? "bg-blue-50" : ""}`}>
                <div className="text-sm font-medium text-gray-800">{a.account_label ?? a.iban}</div>
                <div className="text-xs text-gray-500">Akte #{a.matter_id} · {a.account_type}</div>
              </li>
            ))}
            {accounts?.length === 0 && <li className="p-4 text-sm text-gray-400">Keine Konten.</li>}
          </ul>
        </div>

        <div className="col-span-3 shadow ring-1 ring-black ring-opacity-5 rounded-lg bg-white p-5">
          {selectedAccount === null ? (
            <div className="text-sm text-gray-400">Wähle ein Massekonto.</div>
          ) : (
            <>
              {balance && (
                <div className="mb-4 flex gap-6">
                  <div>
                    <div className="text-xs text-gray-500 uppercase">Massebestand</div>
                    <div className="text-2xl font-semibold text-gray-900">{eur(balance.current_balance)}</div>
                  </div>
                  <div>
                    <div className="text-xs text-gray-500 uppercase">Anfangssaldo</div>
                    <div className="text-lg text-gray-600">{eur(balance.opening_balance)}</div>
                  </div>
                </div>
              )}
              <table className="min-w-full divide-y divide-gray-200 text-sm">
                <thead>
                  <tr className="text-left text-xs text-gray-500 uppercase">
                    <th className="py-2">Datum</th><th>Verwendungszweck</th><th>Gegenpartei</th>
                    <th>Kategorie</th><th className="text-right">Betrag</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-100">
                  {txs?.items.map((t) => (
                    <tr key={t.id} className="hover:bg-gray-50">
                      <td className="py-2 text-gray-500">{t.booking_date ?? t.value_date ?? "—"}</td>
                      <td className="text-gray-800 max-w-xs truncate">{t.purpose}</td>
                      <td className="text-gray-600">{t.counterparty_name ?? "—"}</td>
                      <td><span className="text-xs px-2 py-0.5 rounded-full bg-gray-100 text-gray-700">{CATEGORY_LABELS[t.category] ?? t.category}</span></td>
                      <td className={`text-right font-medium ${t.direction === "in" ? "text-green-700" : "text-red-700"}`}>
                        {t.direction === "in" ? "+" : "−"}{eur(t.amount)}
                      </td>
                    </tr>
                  ))}
                  {txs?.items.length === 0 && <tr><td colSpan={5} className="py-8 text-center text-gray-400">Keine Buchungen.</td></tr>}
                </tbody>
              </table>
            </>
          )}
        </div>
      </div>

      {showCreate && <CreateAccountModal onClose={() => setShowCreate(false)} onDone={() => { setShowCreate(false); qc.invalidateQueries({ queryKey: ["mass-accounts"] }); }} />}
    </div>
  );
}

function CreateAccountModal({ onClose, onDone }: { onClose: () => void; onDone: () => void }) {
  const [form, setForm] = useState({ matter_id: "", iban: "", bank_name: "", account_label: "", account_type: "sonderkonto", opening_balance: "0" });
  const [error, setError] = useState<string | null>(null);

  const mut = useMutation({
    mutationFn: () => financeApi.createAccount({
      matter_id: parseInt(form.matter_id), iban: form.iban, bank_name: form.bank_name || undefined,
      account_label: form.account_label || undefined, account_type: form.account_type,
      opening_balance: form.opening_balance || "0",
    }),
    onSuccess: onDone,
    onError: (e: any) => setError(e?.response?.data?.detail ?? "Fehler"),
  });

  return (
    <div className="fixed inset-0 bg-black bg-opacity-40 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg shadow-xl w-full max-w-md p-6">
        <h2 className="text-lg font-semibold mb-4">Massekonto anlegen</h2>
        {error && <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded text-sm text-red-700">{error}</div>}
        <div className="space-y-3">
          <input placeholder="Akten-ID" type="number" value={form.matter_id} onChange={(e) => setForm(f => ({ ...f, matter_id: e.target.value }))} className="inp" />
          <input placeholder="IBAN" value={form.iban} onChange={(e) => setForm(f => ({ ...f, iban: e.target.value }))} className="inp" />
          <input placeholder="Bank" value={form.bank_name} onChange={(e) => setForm(f => ({ ...f, bank_name: e.target.value }))} className="inp" />
          <input placeholder="Bezeichnung (optional)" value={form.account_label} onChange={(e) => setForm(f => ({ ...f, account_label: e.target.value }))} className="inp" />
          <div className="grid grid-cols-2 gap-3">
            <select value={form.account_type} onChange={(e) => setForm(f => ({ ...f, account_type: e.target.value }))} className="inp">
              {ACCOUNT_TYPES.map((t) => <option key={t} value={t}>{t}</option>)}
            </select>
            <input placeholder="Anfangssaldo" type="number" step="0.01" value={form.opening_balance} onChange={(e) => setForm(f => ({ ...f, opening_balance: e.target.value }))} className="inp" />
          </div>
        </div>
        <div className="flex justify-end gap-3 pt-5">
          <button onClick={onClose} className="px-4 py-2 text-sm text-gray-700 border rounded-md hover:bg-gray-50">Abbrechen</button>
          <button onClick={() => { setError(null); mut.mutate(); }} disabled={mut.isPending || !form.matter_id || !form.iban} className="px-4 py-2 text-sm bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50">
            {mut.isPending ? "Speichern…" : "Anlegen"}
          </button>
        </div>
        <style>{`.inp{width:100%;padding:0.5rem 0.75rem;border:1px solid #d1d5db;border-radius:0.375rem;font-size:0.875rem}`}</style>
      </div>
    </div>
  );
}
