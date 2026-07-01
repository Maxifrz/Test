import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { ticketsApi, TicketCreate, FristTicketCreate } from "../lib/api/tickets";

const STATUS_LABELS: Record<string, string> = {
  open: "Offen",
  in_progress: "In Bearbeitung",
  pending_review: "Review",
  closed: "Geschlossen",
  blocked: "Blockiert",
};
const PRIORITY_COLORS: Record<string, string> = {
  urgent: "bg-red-100 text-red-800",
  high: "bg-orange-100 text-orange-800",
  normal: "bg-blue-100 text-blue-700",
  low: "bg-gray-100 text-gray-500",
};
const TYPE_LABELS: Record<string, string> = {
  task: "Aufgabe",
  frist: "Frist",
  court_date: "Gerichtstermin",
  client_meeting: "Mandantengespräch",
  follow_up: "Wiedervorlage",
};

const FRIST_LABELS: Record<string, string> = {
  einspruch_versaeumnisurteil: "Einspruch Versäumnisurteil (ZPO §339, 2 Wo.)",
  berufung_einlegung: "Berufungseinlegung (ZPO §517, 1 Mon.)",
  berufung_begruendung: "Berufungsbegründung (ZPO §520, 2 Mon.)",
  klageerwiderung: "Klageerwiderung (ZPO §276, 2 Wo.)",
  beschwerde_stpo: "Sofortige Beschwerde (StPO §311, 1 Wo.)",
  wiedereinsetzung_stpo: "Wiedereinsetzung (StPO §45, 1 Wo.)",
};

export default function TicketsPage() {
  const qc = useQueryClient();
  const [statusFilter, setStatusFilter] = useState("");
  const [page, setPage] = useState(1);
  const [mode, setMode] = useState<"task" | "frist" | null>(null);

  const [task, setTask] = useState<TicketCreate>({ title: "", ticket_type: "task", priority: "normal" });
  const [frist, setFrist] = useState<FristTicketCreate>({ frist_type: "einspruch_versaeumnisurteil", trigger_date: "" });
  const [formError, setFormError] = useState<string | null>(null);

  const { data, isLoading } = useQuery({
    queryKey: ["tickets", page, statusFilter],
    queryFn: () => ticketsApi.list({ page, page_size: 20, status: statusFilter || undefined }),
  });

  const createTask = useMutation({
    mutationFn: () => ticketsApi.create(task),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ["tickets"] }); close(); },
    onError: (e: any) => setFormError(e?.response?.data?.detail ?? "Fehler"),
  });

  const createFrist = useMutation({
    mutationFn: () => ticketsApi.createFrist(frist),
    onSuccess: () => { qc.invalidateQueries({ queryKey: ["tickets"] }); close(); },
    onError: (e: any) => setFormError(e?.response?.data?.detail ?? "Fehler"),
  });

  function close() {
    setMode(null);
    setFormError(null);
    setTask({ title: "", ticket_type: "task", priority: "normal" });
    setFrist({ frist_type: "einspruch_versaeumnisurteil", trigger_date: "" });
  }

  const isOverdue = (d: string | null) =>
    d != null && new Date(d) < new Date(new Date().toDateString());

  return (
    <div className="p-6 max-w-6xl mx-auto">
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-2xl font-semibold text-gray-900">Aufgaben & Fristen</h1>
        <div className="flex gap-2">
          <button onClick={() => setMode("frist")} className="px-4 py-2 bg-amber-600 text-white rounded-md text-sm font-medium hover:bg-amber-700">
            + Frist berechnen
          </button>
          <button onClick={() => setMode("task")} className="px-4 py-2 bg-blue-600 text-white rounded-md text-sm font-medium hover:bg-blue-700">
            + Neue Aufgabe
          </button>
        </div>
      </div>

      <div className="mb-4 flex gap-2 flex-wrap">
        <button onClick={() => { setStatusFilter(""); setPage(1); }} className={`px-3 py-1 rounded-full text-sm border ${statusFilter === "" ? "bg-blue-600 text-white border-blue-600" : "border-gray-300 text-gray-600"}`}>Alle</button>
        {Object.entries(STATUS_LABELS).map(([s, label]) => (
          <button key={s} onClick={() => { setStatusFilter(s); setPage(1); }} className={`px-3 py-1 rounded-full text-sm border ${statusFilter === s ? "bg-blue-600 text-white border-blue-600" : "border-gray-300 text-gray-600"}`}>{label}</button>
        ))}
      </div>

      {isLoading ? (
        <div className="text-gray-500 text-sm">Laden...</div>
      ) : (
        <div className="overflow-hidden shadow ring-1 ring-black ring-opacity-5 rounded-lg">
          <table className="min-w-full divide-y divide-gray-300">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Typ</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Titel</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Priorität</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Fällig</th>
                <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Status</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-200 bg-white">
              {data?.items.map((t) => (
                <tr key={t.id} className="hover:bg-gray-50">
                  <td className="px-4 py-3 text-sm text-gray-500">{TYPE_LABELS[t.ticket_type] ?? t.ticket_type}</td>
                  <td className="px-4 py-3 text-sm font-medium text-gray-800">
                    {t.title}
                    {t.sla_breached && <span className="ml-2 text-xs text-red-600">SLA überschritten</span>}
                  </td>
                  <td className="px-4 py-3">
                    <span className={`inline-flex px-2 py-0.5 rounded-full text-xs font-medium ${PRIORITY_COLORS[t.priority]}`}>{t.priority}</span>
                  </td>
                  <td className={`px-4 py-3 text-sm ${isOverdue(t.due_date) && t.status !== "closed" ? "text-red-600 font-semibold" : "text-gray-500"}`}>
                    {t.due_date ? new Date(t.due_date).toLocaleDateString("de-DE") : "—"}
                  </td>
                  <td className="px-4 py-3 text-sm text-gray-500">{STATUS_LABELS[t.status] ?? t.status}</td>
                </tr>
              ))}
              {data?.items.length === 0 && (
                <tr><td colSpan={5} className="px-4 py-8 text-center text-sm text-gray-400">Keine Aufgaben gefunden.</td></tr>
              )}
            </tbody>
          </table>
        </div>
      )}

      {data && data.total > 20 && (
        <div className="mt-4 flex justify-between items-center text-sm text-gray-600">
          <span>{data.total} Einträge gesamt</span>
          <div className="flex gap-2">
            <button disabled={page === 1} onClick={() => setPage(p => p - 1)} className="px-3 py-1 border rounded disabled:opacity-40">Zurück</button>
            <button disabled={page * 20 >= data.total} onClick={() => setPage(p => p + 1)} className="px-3 py-1 border rounded disabled:opacity-40">Weiter</button>
          </div>
        </div>
      )}

      {/* Task modal */}
      {mode === "task" && (
        <Modal title="Neue Aufgabe" onClose={close} error={formError}>
          <Field label="Titel *">
            <input required value={task.title} onChange={(e) => setTask(f => ({ ...f, title: e.target.value }))} className="inp" />
          </Field>
          <div className="grid grid-cols-2 gap-3">
            <Field label="Typ">
              <select value={task.ticket_type} onChange={(e) => setTask(f => ({ ...f, ticket_type: e.target.value }))} className="inp">
                {Object.entries(TYPE_LABELS).map(([v, l]) => <option key={v} value={v}>{l}</option>)}
              </select>
            </Field>
            <Field label="Priorität">
              <select value={task.priority} onChange={(e) => setTask(f => ({ ...f, priority: e.target.value }))} className="inp">
                {["urgent", "high", "normal", "low"].map((p) => <option key={p} value={p}>{p}</option>)}
              </select>
            </Field>
          </div>
          <Field label="Fällig am">
            <input type="date" value={task.due_date ?? ""} onChange={(e) => setTask(f => ({ ...f, due_date: e.target.value }))} className="inp" />
          </Field>
          <Field label="Akten-ID (optional)">
            <input type="number" value={task.matter_id ?? ""} onChange={(e) => setTask(f => ({ ...f, matter_id: e.target.value ? parseInt(e.target.value) : undefined }))} className="inp" />
          </Field>
          <Actions onClose={close} onSubmit={() => createTask.mutate()} pending={createTask.isPending} />
        </Modal>
      )}

      {/* Frist modal */}
      {mode === "frist" && (
        <Modal title="Frist berechnen" onClose={close} error={formError}>
          <p className="text-xs text-gray-500 mb-2">
            Das Fristende wird automatisch nach den gesetzlichen Regeln (§§187–193 BGB,
            Feiertagsanpassung) berechnet und als nachvollziehbarer Vermerk gespeichert.
          </p>
          <Field label="Fristtyp">
            <select value={frist.frist_type} onChange={(e) => setFrist(f => ({ ...f, frist_type: e.target.value }))} className="inp">
              {Object.entries(FRIST_LABELS).map(([v, l]) => <option key={v} value={v}>{l}</option>)}
            </select>
          </Field>
          <Field label="Auslösendes Ereignis (z.B. Zustellungsdatum) *">
            <input required type="date" value={frist.trigger_date} onChange={(e) => setFrist(f => ({ ...f, trigger_date: e.target.value }))} className="inp" />
          </Field>
          <Field label="Akten-ID (optional)">
            <input type="number" value={frist.matter_id ?? ""} onChange={(e) => setFrist(f => ({ ...f, matter_id: e.target.value ? parseInt(e.target.value) : undefined }))} className="inp" />
          </Field>
          <Actions onClose={close} onSubmit={() => createFrist.mutate()} pending={createFrist.isPending} label="Frist erstellen" />
        </Modal>
      )}

      <style>{`.inp{width:100%;padding:0.5rem 0.75rem;border:1px solid #d1d5db;border-radius:0.375rem;font-size:0.875rem}`}</style>
    </div>
  );
}

function Modal({ title, error, children }: { title: string; error: string | null; onClose: () => void; children: React.ReactNode }) {
  return (
    <div className="fixed inset-0 bg-black bg-opacity-40 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg shadow-xl w-full max-w-lg p-6 max-h-screen overflow-y-auto">
        <h2 className="text-lg font-semibold mb-4">{title}</h2>
        {error && <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded text-sm text-red-700">{error}</div>}
        <div className="space-y-4">{children}</div>
      </div>
    </div>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div>
      <label className="block text-sm font-medium text-gray-700 mb-1">{label}</label>
      {children}
    </div>
  );
}

function Actions({ onClose, onSubmit, pending, label = "Erstellen" }: { onClose: () => void; onSubmit: () => void; pending: boolean; label?: string }) {
  return (
    <div className="flex justify-end gap-3 pt-2">
      <button type="button" onClick={onClose} className="px-4 py-2 text-sm text-gray-700 border rounded-md hover:bg-gray-50">Abbrechen</button>
      <button type="button" onClick={onSubmit} disabled={pending} className="px-4 py-2 text-sm bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50">
        {pending ? "Speichern..." : label}
      </button>
    </div>
  );
}
