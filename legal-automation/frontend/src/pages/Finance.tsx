import { useState, useRef } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { financeApi, insolvencyApi, ImportReport, InsVVResult, Claim, DistributionResult } from "../lib/api/finance";

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
  const [tab, setTab] = useState<"konten" | "forderungen" | "rechner">("konten");
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
        {tab === "konten" && (
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
        )}
      </div>

      <div className="mb-5 flex gap-2 border-b border-gray-200">
        {([["konten", "Massekonten"], ["forderungen", "Forderungen & Verteilung"], ["rechner", "Vergütungsrechner"]] as const).map(([t, label]) => (
          <button key={t} onClick={() => setTab(t)}
            className={`px-4 py-2 text-sm font-medium -mb-px border-b-2 ${tab === t ? "border-blue-600 text-blue-700" : "border-transparent text-gray-500 hover:text-gray-700"}`}>
            {label}
          </button>
        ))}
      </div>

      {tab === "rechner" && <VerguetungsRechner />}
      {tab === "forderungen" && <ForderungenView />}

      {tab === "konten" && report && (
        <div className={`mb-4 p-3 rounded border text-sm ${report.reconciled ? "bg-green-50 border-green-200 text-green-800" : "bg-amber-50 border-amber-200 text-amber-800"}`}>
          Import: {report.num_assigned} zugeordnet, {report.num_duplicates} Duplikate, {report.num_unassigned} offen.{" "}
          {report.statement_closing != null && (
            <>Saldo-Abgleich: {report.reconciled ? "stimmt überein ✓" : `Abweichung (Auszug ${eur(report.statement_closing)} vs. berechnet ${eur(report.computed_closing ?? 0)})`}</>
          )}
          <button onClick={() => setReport(null)} className="ml-2 underline">schließen</button>
        </div>
      )}

      {tab === "konten" && (
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
      )}

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

const RANK_LABELS: Record<string, string> = {
  insolvenz_38: "§ 38 (regulär)",
  nachrangig_39: "§ 39 (nachrangig)",
  absonderung: "Absonderung",
  masseverbindlichkeit: "Masseverbindlichkeit",
};
const STATUS_LABELS: Record<string, string> = {
  angemeldet: "angemeldet",
  geprueft: "geprüft",
  festgestellt: "festgestellt",
  bestritten: "bestritten",
};

function ForderungenView() {
  const qc = useQueryClient();
  const [matterId, setMatterId] = useState("");
  const activeMatter = matterId ? parseInt(matterId) : null;

  const { data: table } = useQuery({
    queryKey: ["claims", activeMatter],
    queryFn: () => insolvencyApi.listClaims(activeMatter!),
    enabled: activeMatter !== null,
  });

  const [newClaim, setNewClaim] = useState({ creditor_name: "", claim_amount: "", rank: "insolvenz_38" });
  const addMut = useMutation({
    mutationFn: () => insolvencyApi.createClaim({ matter_id: activeMatter!, ...newClaim }),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ["claims"] }); setNewClaim({ creditor_name: "", claim_amount: "", rank: "insolvenz_38" }); },
  });

  const updateMut = useMutation({
    mutationFn: ({ id, data }: { id: number; data: any }) => insolvencyApi.updateClaim(id, data),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["claims"] }),
  });

  const portalMut = useMutation({
    mutationFn: () => insolvencyApi.enablePortal(activeMatter!),
  });

  const [distributable, setDistributable] = useState("");
  const [dist, setDist] = useState<DistributionResult | null>(null);
  const distMut = useMutation({
    mutationFn: () => insolvencyApi.distribution({ matter_id: activeMatter!, distributable_amount: distributable }),
    onSuccess: setDist,
  });

  function feststellen(c: Claim) {
    const amount = prompt("Festgestellter Betrag (€):", c.claim_amount);
    if (amount != null) updateMut.mutate({ id: c.id, data: { status: "festgestellt", established_amount: amount } });
  }
  function bestreiten(c: Claim) {
    const reason = prompt("Grund des Bestreitens:", "");
    if (reason != null) updateMut.mutate({ id: c.id, data: { status: "bestritten", dispute_reason: reason } });
  }

  return (
    <div className="space-y-5">
      <div className="flex items-end gap-3">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">Akten-ID</label>
          <input type="number" value={matterId} onChange={(e) => { setMatterId(e.target.value); setDist(null); }} className="inp3" />
        </div>
        {activeMatter && (
          <button onClick={() => portalMut.mutate()} className="px-3 py-2 border border-gray-300 text-gray-700 rounded-md text-sm hover:bg-gray-50">
            Gläubiger-Portal aktivieren
          </button>
        )}
      </div>

      {portalMut.data && (
        <div className="p-3 bg-blue-50 border border-blue-200 rounded text-sm text-blue-800">
          Portal aktiv. Öffentlicher Anmelde-Link für Gläubiger:
          <code className="block mt-1 break-all">{portalMut.data.submit_path}</code>
        </div>
      )}

      {activeMatter && table && (
        <>
          <div className="flex gap-6 text-sm">
            <span>Forderungen: <strong>{table.totals.count}</strong></span>
            <span>Summe angemeldet: <strong>{eur(table.totals.sum_angemeldet)}</strong></span>
            <span>Summe festgestellt: <strong>{eur(table.totals.sum_festgestellt)}</strong></span>
            <span className="text-red-600">bestritten: {table.totals.count_bestritten}</span>
          </div>

          <div className="overflow-hidden shadow ring-1 ring-black ring-opacity-5 rounded-lg bg-white">
            <table className="min-w-full divide-y divide-gray-200 text-sm">
              <thead className="bg-gray-50">
                <tr className="text-left text-xs text-gray-500 uppercase">
                  <th className="px-3 py-2">Nr.</th><th>Gläubiger</th><th>Rang</th>
                  <th className="text-right">angemeldet</th><th className="text-right">festgestellt</th>
                  <th>Status</th><th>Quelle</th><th></th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-100">
                {table.items.map((c) => (
                  <tr key={c.id} className="hover:bg-gray-50">
                    <td className="px-3 py-2 text-gray-500">{c.claim_number}</td>
                    <td className="text-gray-800">{c.creditor_name}</td>
                    <td className="text-gray-500">{RANK_LABELS[c.rank] ?? c.rank}</td>
                    <td className="text-right">{eur(c.claim_amount)}</td>
                    <td className="text-right">{c.established_amount ? eur(c.established_amount) : "—"}</td>
                    <td>
                      <span className={`text-xs px-2 py-0.5 rounded-full ${c.status === "festgestellt" ? "bg-green-100 text-green-700" : c.status === "bestritten" ? "bg-red-100 text-red-700" : "bg-gray-100 text-gray-600"}`}>
                        {STATUS_LABELS[c.status] ?? c.status}
                      </span>
                    </td>
                    <td className="text-xs text-gray-400">{c.source === "glaeubiger_portal" ? "Portal" : "intern"}</td>
                    <td className="text-right pr-3 whitespace-nowrap">
                      <button onClick={() => feststellen(c)} className="text-xs text-green-700 mr-2">feststellen</button>
                      <button onClick={() => bestreiten(c)} className="text-xs text-red-600">bestreiten</button>
                    </td>
                  </tr>
                ))}
                {table.items.length === 0 && <tr><td colSpan={8} className="py-6 text-center text-gray-400">Keine Forderungen.</td></tr>}
              </tbody>
            </table>
          </div>

          {/* Neue Forderung */}
          <div className="flex items-end gap-2 flex-wrap">
            <input placeholder="Gläubiger" value={newClaim.creditor_name} onChange={(e) => setNewClaim(s => ({ ...s, creditor_name: e.target.value }))} className="inp3" />
            <input placeholder="Betrag €" type="number" step="0.01" value={newClaim.claim_amount} onChange={(e) => setNewClaim(s => ({ ...s, claim_amount: e.target.value }))} className="inp3 w-32" />
            <select value={newClaim.rank} onChange={(e) => setNewClaim(s => ({ ...s, rank: e.target.value }))} className="inp3">
              {Object.entries(RANK_LABELS).map(([v, l]) => <option key={v} value={v}>{l}</option>)}
            </select>
            <button onClick={() => addMut.mutate()} disabled={!newClaim.creditor_name || !newClaim.claim_amount} className="px-3 py-2 bg-blue-600 text-white rounded-md text-sm disabled:opacity-50">
              + Forderung
            </button>
          </div>

          {/* Verteilungsrechner */}
          <div className="border-t pt-5">
            <h3 className="font-semibold text-gray-900 mb-2">Verteilungsrechner</h3>
            <div className="flex items-end gap-2 mb-3">
              <div>
                <label className="block text-sm text-gray-700 mb-1">Verteilbare Masse €</label>
                <input type="number" step="0.01" value={distributable} onChange={(e) => setDistributable(e.target.value)} className="inp3" />
              </div>
              <button onClick={() => distMut.mutate()} disabled={!distributable} className="px-3 py-2 bg-amber-600 text-white rounded-md text-sm disabled:opacity-50">
                Quote berechnen
              </button>
            </div>
            {dist && (
              <div>
                <div className="text-sm mb-2">
                  Quote § 38: <strong>{dist.quote_38_pct} %</strong> · verteilt {eur(dist.distributed_sum)} · Restmasse {eur(dist.remainder)}
                </div>
                <table className="min-w-full text-sm border rounded">
                  <thead className="bg-gray-50 text-xs text-gray-500 uppercase text-left">
                    <tr><th className="px-3 py-2">Forderung #</th><th className="text-right">festgestellt</th><th className="text-right">Quote</th><th className="text-right">Auszahlung</th></tr>
                  </thead>
                  <tbody className="divide-y divide-gray-100">
                    {dist.items.map((i) => (
                      <tr key={i.claim_id}>
                        <td className="px-3 py-2">{i.claim_id}</td>
                        <td className="text-right">{eur(i.established_amount)}</td>
                        <td className="text-right">{i.quote_pct} %</td>
                        <td className="text-right font-medium">{eur(i.amount)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </>
      )}
      <style>{`.inp3{padding:0.5rem 0.75rem;border:1px solid #d1d5db;border-radius:0.375rem;font-size:0.875rem}`}</style>
    </div>
  );
}

function VerguetungsRechner() {
  const [grundlage, setGrundlage] = useState("");
  const [glaeubiger, setGlaeubiger] = useState("1");
  const [betriebsfortfuehrung, setBetriebsfortfuehrung] = useState(false);
  const [auslagen, setAuslagen] = useState("0");
  const [result, setResult] = useState<InsVVResult | null>(null);
  const [error, setError] = useState<string | null>(null);

  const mut = useMutation({
    mutationFn: () =>
      financeApi.calcInsVV({
        berechnungsgrundlage: grundlage,
        zuschlaege: betriebsfortfuehrung ? [{ name: "Betriebsfortführung (§3)", percent: "0.5" }] : [],
        anzahl_glaeubiger: parseInt(glaeubiger) || 1,
        auslagen: auslagen || "0",
      }),
    onSuccess: setResult,
    onError: (e: any) => setError(e?.response?.data?.detail ?? "Berechnung fehlgeschlagen"),
  });

  async function downloadAntrag() {
    const blob = await financeApi.antragPdf({
      berechnungsgrundlage: grundlage,
      zuschlaege: betriebsfortfuehrung ? [{ name: "Betriebsfortführung (§3)", percent: "0.5" }] : [],
      anzahl_glaeubiger: parseInt(glaeubiger) || 1,
      auslagen: auslagen || "0",
    });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "verguetungsantrag.pdf";
    a.click();
    URL.revokeObjectURL(url);
  }

  return (
    <div className="grid grid-cols-2 gap-6">
      <div className="shadow ring-1 ring-black ring-opacity-5 rounded-lg bg-white p-5 space-y-4">
        <h2 className="text-lg font-semibold text-gray-900">InsVV-Vergütung berechnen</h2>
        <p className="text-xs text-gray-500">
          Regelvergütung § 2 InsVV (Staffel) zzgl. Zu-/Abschläge (§ 3), Mindestvergütung,
          Auslagen (§ 8) und USt. Rechtlich kritisch — Sätze vor Go-Live vom Verwalter freigeben.
        </p>
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">Berechnungsgrundlage (Masse) €</label>
          <input type="number" step="0.01" value={grundlage} onChange={(e) => setGrundlage(e.target.value)} className="inp2" />
        </div>
        <div className="grid grid-cols-2 gap-3">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Anzahl Gläubiger</label>
            <input type="number" value={glaeubiger} onChange={(e) => setGlaeubiger(e.target.value)} className="inp2" />
          </div>
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Auslagen € (§ 8)</label>
            <input type="number" step="0.01" value={auslagen} onChange={(e) => setAuslagen(e.target.value)} className="inp2" />
          </div>
        </div>
        <label className="flex items-center gap-2 text-sm text-gray-700">
          <input type="checkbox" checked={betriebsfortfuehrung} onChange={(e) => setBetriebsfortfuehrung(e.target.checked)} />
          Zuschlag Betriebsfortführung (+50 %)
        </label>
        {error && <div className="p-2 bg-red-50 border border-red-200 rounded text-sm text-red-700">{error}</div>}
        <button onClick={() => { setError(null); mut.mutate(); }} disabled={mut.isPending || !grundlage}
          className="px-4 py-2 bg-blue-600 text-white rounded-md text-sm font-medium hover:bg-blue-700 disabled:opacity-50">
          {mut.isPending ? "Berechne…" : "Berechnen"}
        </button>
      </div>

      <div className="shadow ring-1 ring-black ring-opacity-5 rounded-lg bg-white p-5">
        {!result ? (
          <div className="text-sm text-gray-400">Eingaben links → Berechnen.</div>
        ) : (
          <table className="min-w-full text-sm">
            <tbody className="divide-y divide-gray-100">
              <tr><td className="py-2 text-gray-600">Regelvergütung (§ 2)</td><td className="text-right font-medium">{eur(result.regelverguetung)}</td></tr>
              {result.adjustments.map((a, i) => (
                <tr key={i}><td className="py-2 text-gray-600">{a.name}</td><td className="text-right">{eur(a.amount)}</td></tr>
              ))}
              <tr><td className="py-2 text-gray-700">Vergütung nach Anpassung</td><td className="text-right font-medium">{eur(result.verguetung_nach_anpassung)}</td></tr>
              {result.mindestverguetung_angewandt && (
                <tr><td className="py-2 text-amber-700">Mindestvergütung angewandt</td><td className="text-right text-amber-700">{eur(result.mindestverguetung)}</td></tr>
              )}
              <tr><td className="py-2 text-gray-600">Auslagen (§ 8)</td><td className="text-right">{eur(result.auslagen)}</td></tr>
              <tr><td className="py-2 text-gray-700">Netto</td><td className="text-right font-medium">{eur(result.netto)}</td></tr>
              <tr><td className="py-2 text-gray-600">USt</td><td className="text-right">{eur(result.umsatzsteuer)}</td></tr>
              <tr className="border-t-2"><td className="py-2 font-semibold text-gray-900">Brutto</td><td className="text-right text-lg font-semibold">{eur(result.brutto)}</td></tr>
            </tbody>
          </table>
        )}
        {result && (
          <button onClick={downloadAntrag} className="mt-4 px-4 py-2 border border-gray-300 text-gray-700 rounded-md text-sm hover:bg-gray-50">
            Vergütungsantrag als PDF
          </button>
        )}
      </div>
      <style>{`.inp2{width:100%;padding:0.5rem 0.75rem;border:1px solid #d1d5db;border-radius:0.375rem;font-size:0.875rem}`}</style>
    </div>
  );
}
