import { useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { ROLE_LABELS, usersApi, type UserResponse } from "../lib/api/users";
import { useAuth } from "../lib/auth/AuthContext";

const ROLES = ["admin", "anwalt", "sachbearbeiter", "sekretariat"] as const;

/**
 * Einmal-Passwoerter erscheinen genau einmal — beim Anlegen und beim Reset.
 * Sie werden nur gehasht gespeichert und lassen sich danach nicht mehr
 * anzeigen; deshalb bleibt der Kasten stehen, bis er ausdruecklich geschlossen
 * wird.
 */
function SecretBanner({
  label,
  secret,
  onClose,
}: {
  label: string;
  secret: string;
  onClose: () => void;
}) {
  const [copied, setCopied] = useState(false);
  return (
    <div className="mb-4 rounded-lg border border-amber-300 bg-amber-50 p-4">
      <div className="text-sm font-medium text-amber-900">{label}</div>
      <div className="mt-2 flex items-center gap-3">
        <code className="rounded bg-white px-3 py-1.5 font-mono text-base tracking-wide text-gray-900 border">
          {secret}
        </code>
        <button
          onClick={() => {
            navigator.clipboard?.writeText(secret).then(
              () => setCopied(true),
              () => setCopied(false)
            );
          }}
          className="rounded border bg-white px-2 py-1 text-xs text-gray-700 hover:bg-gray-50"
        >
          {copied ? "Kopiert" : "Kopieren"}
        </button>
        <button onClick={onClose} className="ml-auto text-xs text-amber-800 hover:underline">
          Schließen
        </button>
      </div>
      <p className="mt-2 text-xs text-amber-800">
        Wird nur dieses eine Mal angezeigt. Der Nutzer muss es beim ersten Login ändern.
      </p>
    </div>
  );
}

export default function UsersPage() {
  const qc = useQueryClient();
  const { user: me } = useAuth();
  const [search, setSearch] = useState("");
  const [showInactive, setShowInactive] = useState(false);
  const [secret, setSecret] = useState<{ label: string; value: string } | null>(null);
  const [form, setForm] = useState({ email: "", full_name: "", role: "sachbearbeiter" });
  const [formOpen, setFormOpen] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const { data, isLoading } = useQuery({
    queryKey: ["users", search, showInactive],
    queryFn: () =>
      usersApi
        .list({
          search: search || undefined,
          is_active: showInactive ? undefined : true,
        })
        .then((r) => r.data),
  });

  const invalidate = () => qc.invalidateQueries({ queryKey: ["users"] });
  const onError = (e: unknown) => {
    const detail = (e as { response?: { data?: { detail?: string } } })?.response?.data?.detail;
    setError(detail ?? "Aktion fehlgeschlagen.");
  };

  const createMutation = useMutation({
    mutationFn: () => usersApi.create(form).then((r) => r.data),
    onSuccess: (res) => {
      setSecret({
        label: `Einmal-Passwort für ${res.user.full_name} (${res.user.email})`,
        value: res.initial_password,
      });
      setForm({ email: "", full_name: "", role: "sachbearbeiter" });
      setFormOpen(false);
      setError(null);
      invalidate();
    },
    onError,
  });

  // Nur die Felder, die die Verwaltung aendern darf — Partial<UserResponse>
  // waere zu weit (es enthaelt u. a. totp_enabled, das nur der Server setzt).
  type UserPatch = { full_name?: string; role?: string; phone?: string; is_active?: boolean };

  const updateMutation = useMutation({
    mutationFn: ({ id, patch }: { id: number; patch: UserPatch }) => usersApi.update(id, patch),
    onSuccess: () => {
      setError(null);
      invalidate();
    },
    onError,
  });

  const resetPwMutation = useMutation({
    mutationFn: (u: UserResponse) => usersApi.resetPassword(u.id).then((r) => ({ r: r.data, u })),
    onSuccess: ({ r, u }) => {
      setSecret({
        label: `Neues Einmal-Passwort für ${u.full_name} — ${r.sessions_revoked} Sitzung(en) beendet`,
        value: r.initial_password,
      });
      setError(null);
      invalidate();
    },
    onError,
  });

  const reset2faMutation = useMutation({
    mutationFn: (id: number) => usersApi.reset2fa(id),
    onSuccess: () => {
      setError(null);
      invalidate();
    },
    onError,
  });

  const isAdmin = me?.role === "admin";

  return (
    <div className="min-h-screen bg-gray-50 p-8">
      <div className="mx-auto max-w-5xl">
        <div className="mb-6 flex items-center justify-between">
          <h1 className="text-2xl font-semibold text-gray-900">Benutzerverwaltung</h1>
          <Link to="/" className="text-sm text-gray-500 hover:text-gray-700">
            ← Dashboard
          </Link>
        </div>

        {!isAdmin && (
          <div className="mb-4 rounded-lg border border-gray-200 bg-white p-4 text-sm text-gray-600">
            Diese Seite ist Administratoren vorbehalten.
          </div>
        )}

        {secret && (
          <SecretBanner label={secret.label} secret={secret.value} onClose={() => setSecret(null)} />
        )}
        {error && (
          <div className="mb-4 rounded-lg border border-red-200 bg-red-50 p-3 text-sm text-red-800">
            {error}
          </div>
        )}

        <div className="mb-4 flex flex-wrap items-center gap-3">
          <input
            value={search}
            onChange={(e) => setSearch(e.target.value)}
            placeholder="Name oder E-Mail …"
            className="rounded border px-3 py-1.5 text-sm"
          />
          <label className="flex items-center gap-2 text-sm text-gray-600">
            <input
              type="checkbox"
              checked={showInactive}
              onChange={(e) => setShowInactive(e.target.checked)}
            />
            Deaktivierte anzeigen
          </label>
          {isAdmin && (
            <button
              onClick={() => setFormOpen((v) => !v)}
              className="ml-auto rounded bg-blue-600 px-3 py-1.5 text-sm text-white hover:bg-blue-700"
            >
              {formOpen ? "Abbrechen" : "Neuer Benutzer"}
            </button>
          )}
        </div>

        {formOpen && isAdmin && (
          <form
            onSubmit={(e) => {
              e.preventDefault();
              createMutation.mutate();
            }}
            className="mb-6 grid gap-3 rounded-lg bg-white p-5 shadow sm:grid-cols-3"
          >
            <input
              required
              type="email"
              value={form.email}
              onChange={(e) => setForm({ ...form, email: e.target.value })}
              placeholder="E-Mail"
              className="rounded border px-3 py-2 text-sm"
            />
            <input
              required
              value={form.full_name}
              onChange={(e) => setForm({ ...form, full_name: e.target.value })}
              placeholder="Vollständiger Name"
              className="rounded border px-3 py-2 text-sm"
            />
            <select
              value={form.role}
              onChange={(e) => setForm({ ...form, role: e.target.value })}
              className="rounded border px-3 py-2 text-sm"
            >
              {ROLES.map((r) => (
                <option key={r} value={r}>
                  {ROLE_LABELS[r]}
                </option>
              ))}
            </select>
            <button
              type="submit"
              disabled={createMutation.isPending}
              className="rounded bg-blue-600 px-3 py-2 text-sm text-white hover:bg-blue-700 disabled:opacity-50 sm:col-span-3"
            >
              Anlegen (erzeugt ein Einmal-Passwort)
            </button>
          </form>
        )}

        {isLoading ? (
          <div className="text-sm text-gray-400">Lade …</div>
        ) : !data || data.items.length === 0 ? (
          <div className="rounded-lg bg-white p-6 text-sm text-gray-500 shadow">
            Keine Benutzer gefunden.
          </div>
        ) : (
          <div className="overflow-hidden rounded-lg bg-white shadow">
            <table className="w-full text-sm">
              <thead className="bg-gray-50 text-left text-xs uppercase text-gray-500">
                <tr>
                  <th className="px-4 py-3">Name</th>
                  <th className="px-4 py-3">Rolle</th>
                  <th className="px-4 py-3">Status</th>
                  <th className="px-4 py-3">Letzter Login</th>
                  <th className="px-4 py-3 text-right">Aktionen</th>
                </tr>
              </thead>
              <tbody className="divide-y">
                {data.items.map((u) => (
                  <tr key={u.id} className={u.is_active ? "" : "bg-gray-50 text-gray-400"}>
                    <td className="px-4 py-3">
                      <div className="font-medium text-gray-900">{u.full_name}</div>
                      <div className="text-xs text-gray-500">{u.email}</div>
                    </td>
                    <td className="px-4 py-3">
                      {isAdmin && u.id !== me?.id ? (
                        <select
                          value={u.role}
                          onChange={(e) =>
                            updateMutation.mutate({ id: u.id, patch: { role: e.target.value } })
                          }
                          className="rounded border px-2 py-1 text-xs"
                        >
                          {ROLES.map((r) => (
                            <option key={r} value={r}>
                              {ROLE_LABELS[r]}
                            </option>
                          ))}
                        </select>
                      ) : (
                        ROLE_LABELS[u.role] ?? u.role
                      )}
                    </td>
                    <td className="px-4 py-3">
                      <div className="flex flex-wrap gap-1">
                        {!u.is_active && (
                          <span className="rounded bg-gray-200 px-2 py-0.5 text-xs">deaktiviert</span>
                        )}
                        {u.totp_enabled ? (
                          <span className="rounded bg-green-100 px-2 py-0.5 text-xs text-green-800">
                            2FA aktiv
                          </span>
                        ) : (
                          <span className="rounded bg-amber-100 px-2 py-0.5 text-xs text-amber-800">
                            2FA offen
                          </span>
                        )}
                        {u.must_change_password && (
                          <span className="rounded bg-amber-100 px-2 py-0.5 text-xs text-amber-800">
                            Wechsel offen
                          </span>
                        )}
                      </div>
                    </td>
                    <td className="px-4 py-3 text-xs text-gray-500">
                      {u.last_login ? new Date(u.last_login).toLocaleString("de-DE") : "nie"}
                    </td>
                    <td className="px-4 py-3 text-right">
                      {isAdmin && (
                        <div className="flex justify-end gap-2">
                          <button
                            onClick={() => resetPwMutation.mutate(u)}
                            disabled={resetPwMutation.isPending}
                            className="rounded border px-2 py-1 text-xs hover:bg-gray-50"
                          >
                            Passwort
                          </button>
                          <button
                            onClick={() => reset2faMutation.mutate(u.id)}
                            disabled={!u.totp_enabled || reset2faMutation.isPending}
                            className="rounded border px-2 py-1 text-xs hover:bg-gray-50 disabled:opacity-40"
                          >
                            2FA
                          </button>
                          {u.id !== me?.id && (
                            <button
                              onClick={() =>
                                updateMutation.mutate({
                                  id: u.id,
                                  patch: { is_active: !u.is_active },
                                })
                              }
                              className="rounded border px-2 py-1 text-xs hover:bg-gray-50"
                            >
                              {u.is_active ? "Deaktivieren" : "Aktivieren"}
                            </button>
                          )}
                        </div>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
