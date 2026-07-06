import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { Link } from "react-router-dom";
import { contactApi } from "../lib/api/contact";

export default function ContactRequestsPage() {
  const qc = useQueryClient();
  const [statusFilter, setStatusFilter] = useState<"neu" | "erledigt" | "">("neu");

  const { data, isLoading } = useQuery({
    queryKey: ["contact-requests", statusFilter],
    queryFn: () => contactApi.list(statusFilter ? { status: statusFilter } : undefined),
  });

  const statusMutation = useMutation({
    mutationFn: ({ id, status }: { id: number; status: "neu" | "erledigt" }) =>
      contactApi.setStatus(id, status),
    onSuccess: () => qc.invalidateQueries({ queryKey: ["contact-requests"] }),
  });

  return (
    <div className="min-h-screen bg-gray-50 p-8">
      <div className="max-w-4xl mx-auto">
        <div className="flex justify-between items-center mb-6">
          <h1 className="text-2xl font-semibold text-gray-900">Kontaktanfragen (Website)</h1>
          <Link to="/" className="text-sm text-gray-500 hover:text-gray-700">← Dashboard</Link>
        </div>

        <div className="mb-4 flex gap-2">
          {([["neu", "Neu"], ["erledigt", "Erledigt"], ["", "Alle"]] as const).map(([val, label]) => (
            <button
              key={label}
              onClick={() => setStatusFilter(val)}
              className={`px-3 py-1 rounded text-sm ${
                statusFilter === val ? "bg-blue-600 text-white" : "bg-white border text-gray-600"
              }`}
            >
              {label}
            </button>
          ))}
        </div>

        {isLoading ? (
          <div className="text-sm text-gray-400">Lade …</div>
        ) : !data || data.items.length === 0 ? (
          <div className="bg-white rounded-lg shadow p-6 text-sm text-gray-500">
            Keine Anfragen{statusFilter ? ` mit Status „${statusFilter}"` : ""}.
          </div>
        ) : (
          <div className="space-y-3">
            {data.items.map((req) => (
              <div key={req.id} className="bg-white rounded-lg shadow p-5">
                <div className="flex justify-between items-start gap-4">
                  <div>
                    <div className="font-medium text-gray-900">
                      {req.name}
                      {req.rolle && <span className="ml-2 text-xs bg-gray-100 text-gray-600 px-2 py-0.5 rounded">{req.rolle}</span>}
                      {req.standort && <span className="ml-1 text-xs bg-gray-100 text-gray-600 px-2 py-0.5 rounded">{req.standort}</span>}
                    </div>
                    <div className="text-sm text-gray-500 mt-0.5">
                      <a href={`mailto:${req.email}`} className="text-blue-700 hover:underline">{req.email}</a>
                      {req.phone && <span className="ml-2">· {req.phone}</span>}
                      <span className="ml-2">· {new Date(req.created_at).toLocaleString("de-DE")}</span>
                    </div>
                  </div>
                  <button
                    onClick={() =>
                      statusMutation.mutate({ id: req.id, status: req.status === "neu" ? "erledigt" : "neu" })
                    }
                    disabled={statusMutation.isPending}
                    className={`shrink-0 px-3 py-1 rounded text-xs ${
                      req.status === "neu"
                        ? "bg-green-600 text-white hover:bg-green-700"
                        : "bg-gray-100 text-gray-600 hover:bg-gray-200"
                    } disabled:opacity-50`}
                  >
                    {req.status === "neu" ? "Als erledigt markieren" : "Wieder öffnen"}
                  </button>
                </div>
                <p className="mt-3 text-sm text-gray-800 whitespace-pre-wrap border-t pt-3">{req.message}</p>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
