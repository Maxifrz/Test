import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { clientsApi, ClientCreate } from "../lib/api/clients";

const LEGAL_BASIS_OPTIONS = ["contract", "consent", "legal_obligation", "vital_interests", "public_task", "legitimate_interests"];

export default function ClientsPage() {
  const qc = useQueryClient();
  const [search, setSearch] = useState("");
  const [page, setPage] = useState(1);
  const [showForm, setShowForm] = useState(false);
  const [form, setForm] = useState<ClientCreate>({
    first_name: "",
    last_name: "",
    is_company: false,
    dsgvo_legal_basis: "contract",
  });
  const [formError, setFormError] = useState<string | null>(null);

  const { data, isLoading } = useQuery({
    queryKey: ["clients", page, search],
    queryFn: () => clientsApi.list({ page, page_size: 20, search: search || undefined }),
  });

  const createMutation = useMutation({
    mutationFn: clientsApi.create,
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["clients"] });
      setShowForm(false);
      setForm({ first_name: "", last_name: "", is_company: false, dsgvo_legal_basis: "contract" });
      setFormError(null);
    },
    onError: (err: any) => {
      setFormError(err?.response?.data?.detail ?? "Fehler beim Erstellen");
    },
  });

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    createMutation.mutate(form);
  };

  return (
    <div className="p-6 max-w-6xl mx-auto">
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-2xl font-semibold text-gray-900">Mandanten</h1>
        <button
          onClick={() => setShowForm(true)}
          className="px-4 py-2 bg-blue-600 text-white rounded-md text-sm font-medium hover:bg-blue-700"
        >
          + Neuer Mandant
        </button>
      </div>

      {/* Search */}
      <div className="mb-4">
        <input
          type="text"
          placeholder="Suche nach Name, Nummer oder E-Mail..."
          value={search}
          onChange={(e) => { setSearch(e.target.value); setPage(1); }}
          className="w-full max-w-md px-3 py-2 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
        />
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
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Nr.</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Name</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">E-Mail</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Telefon</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Stadt</th>
                  <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 uppercase">Erstellt</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-200 bg-white">
                {data?.items.map((c) => (
                  <tr key={c.id} className="hover:bg-gray-50">
                    <td className="px-4 py-3 text-sm font-mono text-gray-600">{c.client_number}</td>
                    <td className="px-4 py-3 text-sm font-medium text-blue-600">
                      <Link to={`/clients/${c.id}`}>{c.display_name}</Link>
                    </td>
                    <td className="px-4 py-3 text-sm text-gray-500">{c.email ?? "—"}</td>
                    <td className="px-4 py-3 text-sm text-gray-500">{c.phone ?? "—"}</td>
                    <td className="px-4 py-3 text-sm text-gray-500">{c.city ?? "—"}</td>
                    <td className="px-4 py-3 text-sm text-gray-400">
                      {new Date(c.created_at).toLocaleDateString("de-DE")}
                    </td>
                  </tr>
                ))}
                {data?.items.length === 0 && (
                  <tr>
                    <td colSpan={6} className="px-4 py-8 text-center text-sm text-gray-400">
                      Keine Mandanten gefunden.
                    </td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>

          {/* Pagination */}
          {data && data.total > 20 && (
            <div className="mt-4 flex justify-between items-center text-sm text-gray-600">
              <span>{data.total} Mandanten gesamt</span>
              <div className="flex gap-2">
                <button
                  disabled={page === 1}
                  onClick={() => setPage(p => p - 1)}
                  className="px-3 py-1 border rounded disabled:opacity-40"
                >
                  Zurück
                </button>
                <button
                  disabled={page * 20 >= data.total}
                  onClick={() => setPage(p => p + 1)}
                  className="px-3 py-1 border rounded disabled:opacity-40"
                >
                  Weiter
                </button>
              </div>
            </div>
          )}
        </>
      )}

      {/* Create form modal */}
      {showForm && (
        <div className="fixed inset-0 bg-black bg-opacity-40 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg shadow-xl w-full max-w-lg p-6">
            <h2 className="text-lg font-semibold mb-4">Neuer Mandant</h2>
            {formError && (
              <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded text-sm text-red-700">
                {formError}
              </div>
            )}
            <form onSubmit={handleSubmit} className="space-y-4">
              <div className="flex items-center gap-2 mb-2">
                <input
                  type="checkbox"
                  id="is_company"
                  checked={form.is_company}
                  onChange={(e) => setForm(f => ({ ...f, is_company: e.target.checked }))}
                />
                <label htmlFor="is_company" className="text-sm text-gray-700">Unternehmen</label>
              </div>

              {form.is_company && (
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Firmenname</label>
                  <input
                    type="text"
                    value={form.company_name ?? ""}
                    onChange={(e) => setForm(f => ({ ...f, company_name: e.target.value }))}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm"
                  />
                </div>
              )}

              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Vorname *</label>
                  <input
                    required
                    type="text"
                    value={form.first_name}
                    onChange={(e) => setForm(f => ({ ...f, first_name: e.target.value }))}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Nachname *</label>
                  <input
                    required
                    type="text"
                    value={form.last_name}
                    onChange={(e) => setForm(f => ({ ...f, last_name: e.target.value }))}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm"
                  />
                </div>
              </div>

              <div className="grid grid-cols-2 gap-3">
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">E-Mail</label>
                  <input
                    type="email"
                    value={form.email ?? ""}
                    onChange={(e) => setForm(f => ({ ...f, email: e.target.value }))}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm"
                  />
                </div>
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">Telefon</label>
                  <input
                    type="tel"
                    value={form.phone ?? ""}
                    onChange={(e) => setForm(f => ({ ...f, phone: e.target.value }))}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm"
                  />
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Rechtsgrundlage (DSGVO Art. 6)
                </label>
                <select
                  value={form.dsgvo_legal_basis ?? "contract"}
                  onChange={(e) => setForm(f => ({ ...f, dsgvo_legal_basis: e.target.value }))}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm"
                >
                  {LEGAL_BASIS_OPTIONS.map((b) => (
                    <option key={b} value={b}>{b}</option>
                  ))}
                </select>
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
