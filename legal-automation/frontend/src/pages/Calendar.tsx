import { useState, useRef } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { calendarApi, EventCreate, ConflictItem } from "../lib/api/calendar";

const TYPE_LABELS: Record<string, string> = {
  court_hearing: "Gerichtstermin",
  client_meeting: "Mandantengespräch",
  internal_meeting: "interne Besprechung",
  frist_reminder: "Fristerinnerung",
  vacation: "Urlaub",
  other: "Sonstiges",
};

const TYPE_COLORS: Record<string, string> = {
  court_hearing: "bg-red-100 text-red-800 border-red-200",
  client_meeting: "bg-blue-100 text-blue-800 border-blue-200",
  internal_meeting: "bg-gray-100 text-gray-700 border-gray-200",
  frist_reminder: "bg-amber-100 text-amber-800 border-amber-200",
  vacation: "bg-green-100 text-green-800 border-green-200",
  other: "bg-purple-100 text-purple-800 border-purple-200",
};

function startOfMonth(d: Date) { return new Date(d.getFullYear(), d.getMonth(), 1); }
function endOfMonth(d: Date) { return new Date(d.getFullYear(), d.getMonth() + 1, 0, 23, 59, 59); }

export default function CalendarPage() {
  const qc = useQueryClient();
  const [anchor, setAnchor] = useState(new Date());
  const [showForm, setShowForm] = useState(false);
  const [form, setForm] = useState<EventCreate>({ title: "", event_type: "internal_meeting", start_at: "", end_at: "" });
  const [conflicts, setConflicts] = useState<ConflictItem[]>([]);
  const [formError, setFormError] = useState<string | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  const start = startOfMonth(anchor).toISOString();
  const end = endOfMonth(anchor).toISOString();

  const { data: events, isLoading } = useQuery({
    queryKey: ["calendar", start, end],
    queryFn: () => calendarApi.list({ start, end }),
  });

  const createMut = useMutation({
    mutationFn: (force: boolean) => calendarApi.create({ ...form, force }),
    onSuccess: (res) => {
      qc.invalidateQueries({ queryKey: ["calendar"] });
      if (res.created_preparation_ticket_ids.length > 0) {
        qc.invalidateQueries({ queryKey: ["tickets"] });
      }
      close();
    },
    onError: (e: any) => {
      if (e?.response?.status === 409) {
        const c = e.response.data?.detail?.conflicts ?? [];
        setConflicts(c.map((x: any) => ({ kind: x.kind, detail: x.detail, event_id: x.event_id })));
        setFormError("Terminkonflikt — bitte prüfen und ggf. trotzdem speichern.");
      } else {
        setFormError(e?.response?.data?.detail ?? "Fehler beim Speichern");
      }
    },
  });

  const importMut = useMutation({
    mutationFn: (file: File) => calendarApi.importIcs(file),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["calendar"] }),
  });

  function close() {
    setShowForm(false);
    setConflicts([]);
    setFormError(null);
    setForm({ title: "", event_type: "internal_meeting", start_at: "", end_at: "" });
  }

  const isCourtHearing = form.event_type === "court_hearing";

  return (
    <div className="p-6 max-w-6xl mx-auto">
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-2xl font-semibold text-gray-900">Kalender</h1>
        <div className="flex gap-2 items-center">
          <input
            ref={fileRef}
            type="file"
            accept=".ics"
            className="hidden"
            onChange={(e) => { const f = e.target.files?.[0]; if (f) importMut.mutate(f); e.target.value = ""; }}
          />
          <button onClick={() => fileRef.current?.click()} className="px-3 py-2 border border-gray-300 text-gray-700 rounded-md text-sm hover:bg-gray-50">
            .ics importieren
          </button>
          <button onClick={() => setShowForm(true)} className="px-4 py-2 bg-blue-600 text-white rounded-md text-sm font-medium hover:bg-blue-700">
            + Termin
          </button>
        </div>
      </div>

      <div className="flex items-center justify-between mb-4">
        <button onClick={() => setAnchor(new Date(anchor.getFullYear(), anchor.getMonth() - 1, 1))} className="px-3 py-1 border rounded">‹</button>
        <span className="font-medium text-gray-800">{anchor.toLocaleDateString("de-DE", { month: "long", year: "numeric" })}</span>
        <button onClick={() => setAnchor(new Date(anchor.getFullYear(), anchor.getMonth() + 1, 1))} className="px-3 py-1 border rounded">›</button>
      </div>

      {importMut.isSuccess && (
        <div className="mb-3 p-2 bg-green-50 border border-green-200 text-green-700 text-sm rounded">
          {importMut.data.imported} Termin(e) importiert.
        </div>
      )}

      {/* Agenda list (month) */}
      {isLoading ? (
        <div className="text-sm text-gray-500">Laden...</div>
      ) : (
        <div className="space-y-2">
          {events && events.length > 0 ? (
            events.map((ev) => (
              <div key={ev.id} className={`flex items-center gap-3 p-3 rounded-lg border ${TYPE_COLORS[ev.event_type] ?? "bg-white border-gray-200"}`}>
                <div className="text-center w-14 shrink-0">
                  <div className="text-lg font-semibold">{new Date(ev.start_at).getDate()}</div>
                  <div className="text-xs uppercase">{new Date(ev.start_at).toLocaleDateString("de-DE", { weekday: "short" })}</div>
                </div>
                <div className="flex-1 min-w-0">
                  <div className="font-medium truncate">
                    {ev.title}
                    {ev.status === "cancelled" && <span className="ml-2 text-xs line-through">abgesagt</span>}
                  </div>
                  <div className="text-xs opacity-80">
                    {ev.all_day
                      ? "ganztägig"
                      : `${new Date(ev.start_at).toLocaleTimeString("de-DE", { hour: "2-digit", minute: "2-digit" })} – ${new Date(ev.end_at).toLocaleTimeString("de-DE", { hour: "2-digit", minute: "2-digit" })}`}
                    {ev.location ? ` · ${ev.location}` : ""}
                    {ev.travel_buffer_minutes > 0 ? ` · +${ev.travel_buffer_minutes} Min Anfahrt` : ""}
                  </div>
                </div>
                <div className="flex items-center gap-2 shrink-0">
                  <span className="text-xs px-2 py-0.5 rounded-full bg-white/60">{TYPE_LABELS[ev.event_type] ?? ev.event_type}</span>
                  <a href={calendarApi.exportIcsUrl(ev.id)} className="text-xs underline opacity-70 hover:opacity-100">.ics</a>
                </div>
              </div>
            ))
          ) : (
            <div className="text-center text-sm text-gray-400 py-10">Keine Termine in diesem Monat.</div>
          )}
        </div>
      )}

      {showForm && (
        <div className="fixed inset-0 bg-black bg-opacity-40 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg shadow-xl w-full max-w-lg p-6 max-h-screen overflow-y-auto">
            <h2 className="text-lg font-semibold mb-4">Neuer Termin</h2>
            {formError && <div className="mb-4 p-3 bg-amber-50 border border-amber-200 rounded text-sm text-amber-800">{formError}</div>}
            {conflicts.length > 0 && (
              <ul className="mb-4 space-y-1 text-sm">
                {conflicts.map((c, i) => (
                  <li key={i} className={c.kind === "holiday" ? "text-amber-700" : "text-red-700"}>
                    • {c.detail}
                  </li>
                ))}
              </ul>
            )}
            <div className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Titel *</label>
                <input value={form.title} onChange={(e) => setForm(f => ({ ...f, title: e.target.value }))} className="inp" />
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Art</label>
                <select value={form.event_type} onChange={(e) => setForm(f => ({ ...f, event_type: e.target.value }))} className="inp">
                  {Object.entries(TYPE_LABELS).map(([v, l]) => <option key={v} value={v}>{l}</option>)}
                </select>
              </div>
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Beginn *</label>
                  <input type="datetime-local" value={form.start_at} onChange={(e) => setForm(f => ({ ...f, start_at: e.target.value }))} className="inp" />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Ende *</label>
                  <input type="datetime-local" value={form.end_at} onChange={(e) => setForm(f => ({ ...f, end_at: e.target.value }))} className="inp" />
                </div>
              </div>
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Ort</label>
                <input value={form.location ?? ""} onChange={(e) => setForm(f => ({ ...f, location: e.target.value }))} className="inp" />
              </div>
              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Akten-ID</label>
                  <input type="number" value={form.matter_id ?? ""} onChange={(e) => setForm(f => ({ ...f, matter_id: e.target.value ? parseInt(e.target.value) : undefined }))} className="inp" />
                </div>
                {isCourtHearing && (
                  <div>
                    <label className="block text-sm font-medium text-gray-700 mb-1">Anfahrt (Min)</label>
                    <input type="number" value={form.travel_buffer_minutes ?? ""} onChange={(e) => setForm(f => ({ ...f, travel_buffer_minutes: e.target.value ? parseInt(e.target.value) : 0 }))} className="inp" />
                  </div>
                )}
              </div>
              {isCourtHearing && (
                <label className="flex items-center gap-2 text-sm text-gray-700">
                  <input type="checkbox" checked={form.generate_preparation ?? false} onChange={(e) => setForm(f => ({ ...f, generate_preparation: e.target.checked }))} />
                  Vorbereitungs-Aufgaben automatisch erstellen (Aktenstudium, Mandantenbesprechung, Finalisierung)
                </label>
              )}
            </div>
            <div className="flex justify-end gap-3 pt-5">
              <button onClick={close} className="px-4 py-2 text-sm text-gray-700 border rounded-md hover:bg-gray-50">Abbrechen</button>
              <button
                onClick={() => createMut.mutate(conflicts.length > 0)}
                disabled={createMut.isPending || !form.title || !form.start_at || !form.end_at}
                className="px-4 py-2 text-sm bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50"
              >
                {createMut.isPending ? "Speichern..." : conflicts.length > 0 ? "Trotzdem speichern" : "Speichern"}
              </button>
            </div>
          </div>
        </div>
      )}

      <style>{`.inp{width:100%;padding:0.5rem 0.75rem;border:1px solid #d1d5db;border-radius:0.375rem;font-size:0.875rem}`}</style>
    </div>
  );
}
