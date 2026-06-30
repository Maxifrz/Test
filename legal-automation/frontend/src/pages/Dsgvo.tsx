import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { dsgvoApi } from "../lib/api/dsgvo";

export default function DsgvoPage() {
  const [tab, setTab] = useState<"dashboard" | "vvt" | "erasure" | "export">("dashboard");
  const TABS: [typeof tab, string][] = [
    ["dashboard", "Admin-Dashboard"],
    ["vvt", "Verarbeitungsverzeichnis"],
    ["erasure", "Löschanträge (Art. 17)"],
    ["export", "Datenexport (Art. 20)"],
  ];

  return (
    <div className="p-6 max-w-6xl mx-auto">
      <h1 className="text-2xl font-semibold text-gray-900 mb-1">DSGVO</h1>
      <p className="text-sm text-gray-500 mb-5">Datenschutz-Werkzeuge: Art. 30 / Art. 17 / Art. 20 + Admin-Übersicht</p>

      <div className="flex gap-1 border-b border-gray-200 mb-5">
        {TABS.map(([t, label]) => (
          <button key={t} onClick={() => setTab(t)}
            className={`px-4 py-2 text-sm font-medium -mb-px border-b-2 ${tab === t ? "border-blue-600 text-blue-700" : "border-transparent text-gray-500 hover:text-gray-700"}`}>
            {label}
          </button>
        ))}
      </div>

      {tab === "dashboard" && <Dashboard />}
      {tab === "vvt" && <Vvt />}
      {tab === "erasure" && <Erasure />}
      {tab === "export" && <Export />}
    </div>
  );
}

function Dashboard() {
  const { data } = useQuery({ queryKey: ["dsgvo-overview"], queryFn: dsgvoApi.overview });
  if (!data) return <div className="text-sm text-gray-400">Laden…</div>;
  const cards: [string, number, string][] = [
    ["Aktive Sessions", data.active_sessions, ""],
    ["Gesperrte Nutzer", data.locked_users, data.locked_users > 0 ? "text-red-600" : ""],
    ["2FA aktiv", data.users_with_2fa, ""],
    ["Nutzer gesamt", data.users_total, ""],
    ["Offene Löschanträge", data.open_erasure_requests, data.open_erasure_requests > 0 ? "text-amber-600" : ""],
    ["Blockierte Löschanträge", data.blocked_erasure_requests, ""],
    ["Akten über Aufbewahrungsfrist", data.matters_past_retention, data.matters_past_retention > 0 ? "text-amber-600" : ""],
  ];
  return (
    <div className="grid grid-cols-3 gap-4">
      {cards.map(([label, value, cls]) => (
        <div key={label} className="bg-white border border-gray-200 rounded-lg p-4">
          <div className="text-xs text-gray-500 uppercase">{label}</div>
          <div className={`text-2xl font-semibold ${cls}`}>{value}</div>
        </div>
      ))}
      <div className="col-span-3 text-xs text-gray-500">
        Hinweis: „Akten über Aufbewahrungsfrist" sind Löschkandidaten — die Löschung erfolgt ausschließlich
        über den geprüften Workflow (Art. 17), niemals automatisch.
      </div>
    </div>
  );
}

function Vvt() {
  const { data } = useQuery({ queryKey: ["dsgvo-vvt"], queryFn: dsgvoApi.vvt });
  if (!data) return <div className="text-sm text-gray-400">Laden…</div>;
  return (
    <div className="space-y-3">
      {data.map((r) => (
        <div key={r.id} className="bg-white border border-gray-200 rounded-lg p-4">
          <div className="font-medium text-gray-900">{r.name}</div>
          <div className="text-sm text-gray-600 mt-1">{r.purpose}</div>
          <div className="grid grid-cols-2 gap-x-6 gap-y-1 mt-3 text-xs text-gray-600">
            <div><span className="text-gray-400">Rechtsgrundlage:</span> {r.legal_basis}</div>
            <div><span className="text-gray-400">Betroffene:</span> {r.data_subjects ?? "—"}</div>
            <div><span className="text-gray-400">Empfänger:</span> {r.recipients ?? "—"}</div>
            <div><span className="text-gray-400">Aufbewahrung:</span> {r.retention ?? "—"}</div>
            <div className="col-span-2"><span className="text-gray-400">Datenkategorien:</span> {r.data_categories.join(", ")}</div>
            <div className="col-span-2"><span className="text-gray-400">TOM:</span> {r.tom ?? "—"}</div>
          </div>
        </div>
      ))}
    </div>
  );
}

function Erasure() {
  const qc = useQueryClient();
  const [clientId, setClientId] = useState("");
  const [elig, setElig] = useState<{ allowed: boolean; blocking_reasons: string[] } | null>(null);
  const { data: requests } = useQuery({ queryKey: ["dsgvo-erasure"], queryFn: dsgvoApi.listErasure });

  const checkMut = useMutation({
    mutationFn: () => dsgvoApi.eligibility(parseInt(clientId)),
    onSuccess: setElig,
  });
  const createMut = useMutation({
    mutationFn: () => dsgvoApi.createErasure(parseInt(clientId)),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ["dsgvo-erasure"] }); setElig(null); },
  });
  const execMut = useMutation({
    mutationFn: (id: number) => dsgvoApi.executeErasure(id),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["dsgvo-erasure"] }),
  });

  return (
    <div className="space-y-5">
      <div className="bg-white border border-gray-200 rounded-lg p-4">
        <div className="flex items-end gap-2">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Mandanten-ID</label>
            <input type="number" value={clientId} onChange={(e) => setClientId(e.target.value)} className="px-3 py-2 border rounded-md text-sm" />
          </div>
          <button onClick={() => checkMut.mutate()} disabled={!clientId} className="px-3 py-2 border border-gray-300 rounded-md text-sm hover:bg-gray-50">
            Löscheignung prüfen
          </button>
          <button onClick={() => createMut.mutate()} disabled={!clientId} className="px-3 py-2 bg-blue-600 text-white rounded-md text-sm disabled:opacity-50">
            Löschantrag anlegen
          </button>
        </div>
        {elig && (
          <div className={`mt-3 p-3 rounded text-sm ${elig.allowed ? "bg-green-50 border border-green-200 text-green-800" : "bg-amber-50 border border-amber-200 text-amber-800"}`}>
            {elig.allowed ? "Löschung zulässig — keine laufenden Aufbewahrungspflichten." : (
              <><b>Blockiert:</b><ul className="list-disc ml-5 mt-1">{elig.blocking_reasons.map((r, i) => <li key={i}>{r}</li>)}</ul></>
            )}
          </div>
        )}
      </div>

      <div className="overflow-hidden shadow ring-1 ring-black ring-opacity-5 rounded-lg bg-white">
        <table className="min-w-full divide-y divide-gray-200 text-sm">
          <thead className="bg-gray-50"><tr className="text-left text-xs text-gray-500 uppercase">
            <th className="px-3 py-2">ID</th><th>Mandant</th><th>Status</th><th>Begründung / Blockade</th><th></th>
          </tr></thead>
          <tbody className="divide-y divide-gray-100">
            {requests?.map((r) => (
              <tr key={r.id}>
                <td className="px-3 py-2">{r.id}</td>
                <td>#{r.client_id}</td>
                <td><span className={`text-xs px-2 py-0.5 rounded-full ${r.status === "executed" ? "bg-green-100 text-green-700" : r.status === "blocked" ? "bg-red-100 text-red-700" : "bg-gray-100 text-gray-600"}`}>{r.status}</span></td>
                <td className="text-xs text-gray-500">{(r.blocking_reasons ?? []).join("; ") || r.reason || "—"}</td>
                <td className="text-right pr-3">
                  {r.status === "open" && <button onClick={() => execMut.mutate(r.id)} className="text-xs text-red-700">ausführen</button>}
                  {r.status === "executed" && <span className="text-xs text-gray-400">Zertifikat erstellt</span>}
                </td>
              </tr>
            ))}
            {requests?.length === 0 && <tr><td colSpan={5} className="py-6 text-center text-gray-400">Keine Löschanträge.</td></tr>}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function Export() {
  const [clientId, setClientId] = useState("");
  const [link, setLink] = useState<string | null>(null);
  const mut = useMutation({
    mutationFn: () => dsgvoApi.createExport(parseInt(clientId)),
    onSuccess: (d) => setLink(d.download_path),
  });
  return (
    <div className="bg-white border border-gray-200 rounded-lg p-4 max-w-xl">
      <p className="text-sm text-gray-600 mb-3">
        Erstellt ein maschinenlesbares ZIP (client.json, Akten, …) gemäß Art. 20 DSGVO.
        Der Download-Link ist 48 Stunden gültig und nur einmal verwendbar.
      </p>
      <div className="flex items-end gap-2">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">Mandanten-ID</label>
          <input type="number" value={clientId} onChange={(e) => setClientId(e.target.value)} className="px-3 py-2 border rounded-md text-sm" />
        </div>
        <button onClick={() => mut.mutate()} disabled={!clientId} className="px-3 py-2 bg-blue-600 text-white rounded-md text-sm disabled:opacity-50">
          Export erstellen
        </button>
      </div>
      {link && (
        <div className="mt-3 p-3 bg-green-50 border border-green-200 rounded text-sm text-green-800">
          Export bereit: <a className="underline break-all" href={link}>{link}</a> (48 h, single-use)
        </div>
      )}
    </div>
  );
}
