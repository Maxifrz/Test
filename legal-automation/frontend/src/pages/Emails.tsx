import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { emailsApi } from "../lib/api/emails";

export default function EmailsPage() {
  const qc = useQueryClient();
  const [page, setPage] = useState(1);
  const [filter, setFilter] = useState<"all" | "review" | "inbound" | "outbound">("all");
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [fileMatterId, setFileMatterId] = useState("");

  const params = {
    page,
    page_size: 20,
    needs_review: filter === "review" ? true : undefined,
    direction: filter === "inbound" || filter === "outbound" ? filter : undefined,
  };

  const { data, isLoading } = useQuery({
    queryKey: ["emails", params],
    queryFn: () => emailsApi.list(params),
  });

  const { data: detail } = useQuery({
    queryKey: ["email", selectedId],
    queryFn: () => emailsApi.get(selectedId!),
    enabled: selectedId !== null,
  });

  const fileMutation = useMutation({
    mutationFn: () => emailsApi.fileToMatter(selectedId!, parseInt(fileMatterId)),
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["emails"] });
      qc.invalidateQueries({ queryKey: ["email", selectedId] });
      setFileMatterId("");
    },
  });

  const FILTERS: [typeof filter, string][] = [
    ["all", "Alle"],
    ["review", "Zu prüfen"],
    ["inbound", "Eingang"],
    ["outbound", "Ausgang"],
  ];

  return (
    <div className="p-6 max-w-6xl mx-auto">
      <h1 className="text-2xl font-semibold text-gray-900 mb-6">E-Mails</h1>

      <div className="mb-4 flex gap-2">
        {FILTERS.map(([f, label]) => (
          <button
            key={f}
            onClick={() => { setFilter(f); setPage(1); }}
            className={`px-3 py-1 rounded-full text-sm border ${filter === f ? "bg-blue-600 text-white border-blue-600" : "border-gray-300 text-gray-600"}`}
          >
            {label}
          </button>
        ))}
      </div>

      <div className="grid grid-cols-5 gap-4">
        {/* List */}
        <div className="col-span-2 overflow-hidden shadow ring-1 ring-black ring-opacity-5 rounded-lg bg-white">
          {isLoading ? (
            <div className="p-4 text-sm text-gray-500">Laden...</div>
          ) : (
            <ul className="divide-y divide-gray-200">
              {data?.items.map((e) => (
                <li
                  key={e.id}
                  onClick={() => setSelectedId(e.id)}
                  className={`p-3 cursor-pointer hover:bg-gray-50 ${selectedId === e.id ? "bg-blue-50" : ""} ${!e.is_read ? "font-semibold" : ""}`}
                >
                  <div className="flex justify-between items-start">
                    <span className="text-sm text-gray-800 truncate">{e.from_address}</span>
                    <div className="flex gap-1 shrink-0 ml-2">
                      {e.is_confidential && <span className="text-xs bg-red-100 text-red-700 px-1.5 rounded">vertraulich</span>}
                      {e.needs_review && <span className="text-xs bg-amber-100 text-amber-700 px-1.5 rounded">prüfen</span>}
                    </div>
                  </div>
                  <div className="text-sm text-gray-600 truncate">{e.subject ?? "(kein Betreff)"}</div>
                  <div className="text-xs text-gray-400">
                    {e.email_date ? new Date(e.email_date).toLocaleString("de-DE") : ""}
                  </div>
                </li>
              ))}
              {data?.items.length === 0 && (
                <li className="p-8 text-center text-sm text-gray-400">Keine E-Mails.</li>
              )}
            </ul>
          )}
          {data && data.total > 20 && (
            <div className="p-3 flex justify-between items-center text-xs text-gray-600 border-t">
              <span>{data.total} gesamt</span>
              <div className="flex gap-2">
                <button disabled={page === 1} onClick={() => setPage(p => p - 1)} className="px-2 py-0.5 border rounded disabled:opacity-40">‹</button>
                <button disabled={page * 20 >= data.total} onClick={() => setPage(p => p + 1)} className="px-2 py-0.5 border rounded disabled:opacity-40">›</button>
              </div>
            </div>
          )}
        </div>

        {/* Detail */}
        <div className="col-span-3 shadow ring-1 ring-black ring-opacity-5 rounded-lg bg-white p-5">
          {!detail ? (
            <div className="text-sm text-gray-400">Wähle eine E-Mail aus.</div>
          ) : (
            <div>
              <h2 className="text-lg font-semibold text-gray-900">{detail.subject ?? "(kein Betreff)"}</h2>
              <div className="text-sm text-gray-500 mt-1">
                Von: {detail.from_address}<br />
                An: {detail.to_addresses.join(", ")}
              </div>
              <div className="mt-3 flex items-center gap-2 text-xs">
                {detail.matter_id ? (
                  <span className="bg-green-100 text-green-700 px-2 py-0.5 rounded">Akte #{detail.matter_id}</span>
                ) : (
                  <div className="flex items-center gap-1">
                    <input
                      type="number"
                      placeholder="Akten-ID"
                      value={fileMatterId}
                      onChange={(e) => setFileMatterId(e.target.value)}
                      className="w-24 px-2 py-1 border rounded text-xs"
                    />
                    <button
                      onClick={() => fileMutation.mutate()}
                      disabled={!fileMatterId || fileMutation.isPending}
                      className="px-2 py-1 bg-blue-600 text-white rounded text-xs disabled:opacity-50"
                    >
                      Zur Akte ablegen
                    </button>
                  </div>
                )}
              </div>
              <hr className="my-4" />
              <div className="text-sm text-gray-800 whitespace-pre-wrap">
                {detail.body_text ?? "(kein Textinhalt)"}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
