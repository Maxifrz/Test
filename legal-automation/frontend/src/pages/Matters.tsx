import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { mattersApi, MatterCreate } from "../lib/api/matters";
import { clientsApi } from "../lib/api/clients";

const MATTER_TYPES = ["civil", "criminal", "family", "labor", "admin", "tax", "ip", "other"];
const STATUS_LABELS: Record<string, string> = {
  open: "Offen",
  active: "Aktiv",
  pending_closing: "Abschluss ausstehend",
  closed: "Geschlossen",
  archived: "Archiviert",
};
const STATUS_COLORS: Record<string, string> = {
  open: "bg-yellow-100 text-yellow-800",
  active: "bg-green-100 text-green-800",
  pending_closing: "bg-orange-100 text-orange-800",
  closed: "bg-gray-100 text-gray-600",
  archived: "bg-gray-100 text-gray-400",
};

export default function MattersPage() {
  const qc = useQueryClient();
  const [statusFilter, setStatusFilter] = useState<string>("");
  const [page, setPage] = useState(1);
  const [showForm, setShowForm] = useState(false);
  const [form, setForm] = useState<Partial<MatterCreate>>({
    matter_type: "civil",
    retention_years: 6,
  });
  const [formError, setFormError] = useState<string | null>(null);

  const { data, isLoading } = useQuery({
    queryKey: ["matters", page, statusFilter],
    queryFn: () =>
      mattersApi.list({ page, page_size: 20, status: statusFilter || undefined }),
  });

  const { data: clientsData } = useQuery({
    queryKey: ["clients-all"],
    queryFn: () => clientsApi.list({ page_size: 100 }),
    enabled: showForm,
  });

  const createMutation = useMutation({
    mutationFn: (data: MatterCreate) => mattersApi.create(data),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["matters"] });
      setShowForm(false);
      setForm({ matter_type: "civil", retention_years: 6 });
      setFormError(null);
    },
    onError: (err: any) => {
      setFormError(err?.response?.data?.detail ?? "Fehler beim Erstellen");
    },
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (!form.title || !form.client_id || !form.lead_anwalt_id || !form.matter_type) return;
    createMutation.mutate(form as MatterCreate);
  };

  return (
    <div className="p-6 max-w-6xl mx-auto">
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-2xl font-semibold text-gray-900">Akten</h1>
        <button
          onClick={() => setShowForm(true)}
          className="px-4 py-2 bg-blue-600 text-white rounded-md text-sm font-medium hover:bg-blue-700"
        >
          + Neue Akte
        </button>
      </div>

      {/* Status filter */}
      <div className="mb-4 flex gap-2 flex-wrap">
        <button
          onClick={() => { setStatusFilter(""); setPage(1); }}
          className={`px-3 py-1 rounded-full text-sm border ${statusFilter === "" ? "bg-blue-600 text-white border-blue-600" : "border-gray-300 text-gray-600 hover:bg-gray-50"}`}
        >
          Alle
        </button>
        {Object.entries(STATUS_LABELS).map(([s, label]) => (
          <button
            key={s}
            onClick={() => { setStatusFilter(s); setPage(1); }}
            className={`px-3 py-1 rounded-full text-sm border ${statusFilter === s ? "bg-blue-600 text-white border-blue-600" : "border-gray-300 text-gray-600 hover:bg-gray-50"}`}
          >
            {label}
          </button>
        ))}
      </div>

      {/* Table */}
      {isLoading ? (
        <div className="text-gray-500 text-sm">Laden...</div>
      ) : (
        <>
          <div className="overflow-hidden shadow ring-1 ring-black ring-opacity-5 rounded-lg">
            <table className="min-w-full divide-y divide-gray-300">
              <thead className="bg-gray-50">
                <tr>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Aktenzeichen</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Titel</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Typ</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Status</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Eröffnet</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-200 bg-white">
                {data?.items.map((m) => (
                  <tr key={m.id} className="hover:bg-gray-50">
                    <td className="px-4 py-3 text-sm font-mono text-gray-600">{m.matter_number}</td>
                    <td className="px-4 py-3 text-sm font-medium text-blue-600">
                      <Link to={`/matters/${m.id}`}>{m.title}</Link>
                    </td>
                    <td className="px-4 py-3 text-sm text-gray-500 capitalize">{m.matter_type}</td>
                    <td className="px-4 py-3">
                      <span className={`inline-flex px-2 py-0.5 rounded-full text-xs font-medium ${STATUS_COLORS[m.status] ?? "bg-gray-100 text-gray-600"}`}>
                        {STATUS_LABELS[m.status] ?? m.status}
                      </span>
                    </td>
                    <td className="px-4 py-3 text-sm text-gray-400">
                      {new Date(m.opened_at).toLocaleDateString("de-DE")}
                    </td>
                  </tr>
                ))}
                {data?.items.length === 0 && (
                  <tr>
                    <td colSpan={5} className="px-4 py-8 text-center text-sm text-gray-400">
                      Keine Akten gefunden.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>

          {data && data.total > 20 && (
            <div className="mt-4 flex justify-between items-center text-sm text-gray-600">
              <span>{data.total} Akten gesamt</span>
              <div className="flex gap-2">
                <button disabled={page === 1} onClick={() => setPage(p => p - 1)} className="px-3 py-1 border rounded disabled:opacity-40">Zurück</button>
                <button disabled={page * 20 >= data.total} onClick={() => setPage(p => p + 1)} className="px-3 py-1 border rounded disabled:opacity-40">Weiter</button>
              </div>
            </div>
          )}
        </>
      )}

      {/* Create form modal */}
      {showForm && (
        <div className="fixed inset-0 bg-black bg-opacity-40 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg shadow-xl w-full max-w-lg p-6 max-h-screen overflow-y-auto">
            <h2 className="text-lg font-semibold mb-4">Neue Akte</h2>
            {formError && (
              <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded text-sm text-red-700">
                {formError}
              </div>
            )}
            <form onSubmit={handleSubmit} className="space-y-4">
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Titel *</label>
                <input
                  required
                  type="text"
                  value={form.title ?? ""}
                  onChange={(e) => setForm(f => ({ ...f, title: e.target.value }))}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm"
                  placeholder="z.B. Mustermann ./. Musterbank GmbH"
                />
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Aktentyp *</label>
                  <select
                    value={form.matter_type ?? "civil"}
                    onChange={(e) => setForm(f => ({ ...f, matter_type: e.target.value }))}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm"
                  >
                    {MATTER_TYPES.map((t) => (
                      <option key={t} value={t} className="capitalize">{t}</option>
                    ))}
                  </select>
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Aufbewahrung (Jahre)</label>
                  <input
                    type="number"
                    min={6}
                    value={form.retention_years ?? 6}
                    onChange={(e) => setForm(f => ({ ...f, retention_years: parseInt(e.target.value) }))}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm"
                  />
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Mandant *</label>
                <select
                  required
                  value={form.client_id ?? ""}
                  onChange={(e) => setForm(f => ({ ...f, client_id: parseInt(e.target.value) }))}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm"
                >
                  <option value="">— Mandant wählen —</option>
                  {clientsData?.items.map((c) => (
                    <option key={c.id} value={c.id}>{c.display_name} ({c.client_number})</option>
                  ))}
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Zuständiger Anwalt (User-ID) *</label>
                <input
                  required
                  type="number"
                  value={form.lead_anwalt_id ?? ""}
                  onChange={(e) => setForm(f => ({ ...f, lead_anwalt_id: parseInt(e.target.value) }))}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm"
                  placeholder="User-ID"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Gerichtszeichen</label>
                <input
                  type="text"
                  value={form.court_file_ref ?? ""}
                  onChange={(e) => setForm(f => ({ ...f, court_file_ref: e.target.value }))}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm"
                  placeholder="z.B. 12 O 345/24"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">Beschreibung</label>
                <textarea
                  rows={3}
                  value={form.description ?? ""}
                  onChange={(e) => setForm(f => ({ ...f, description: e.target.value }))}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm resize-none"
                />
              </div>

              <div className="flex justify-end gap-3 pt-2">
                <button
                  type="button"
                  onClick={() => { setShowForm(false); setFormError(null); }}
                  className="px-4 py-2 text-sm text-gray-700 border rounded-md hover:bg-gray-50"
                >
                  Abbrechen
                </button>
                <button
                  type="submit"
                  disabled={createMutation.isPending}
                  className="px-4 py-2 text-sm bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50"
                >
                  {createMutation.isPending ? "Speichern..." : "Erstellen"}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}
