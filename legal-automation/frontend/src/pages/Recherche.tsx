import { useState } from "react";
import { useQuery, useMutation } from "@tanstack/react-query";
import { kiApi, KiQueryResult } from "../lib/api/ki";

const SOURCE_LABELS: Record<string, string> = {
  gesetz: "Gesetz",
  urteil: "Urteil",
  eurlex: "EUR-Lex",
  intern_akte: "Interne Akte",
  intern_schriftsatz: "Schriftsatz",
  intern_transkript: "Transkript",
};

export default function RecherchePage() {
  const [question, setQuestion] = useState("");
  const [matterId, setMatterId] = useState("");
  const [result, setResult] = useState<KiQueryResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [feedbackSent, setFeedbackSent] = useState(false);

  const { data: status } = useQuery({ queryKey: ["ki-status"], queryFn: kiApi.status });

  const queryMut = useMutation({
    mutationFn: () => kiApi.query(question, matterId ? parseInt(matterId) : undefined),
    onSuccess: (r) => { setResult(r); setFeedbackSent(false); },
    onError: (e: any) => setError(e?.response?.data?.detail ?? "Anfrage fehlgeschlagen"),
  });

  const feedbackMut = useMutation({
    mutationFn: ({ fb }: { fb: "up" | "down" }) => kiApi.feedback(result!.query_id!, fb),
    onSuccess: () => setFeedbackSent(true),
  });

  return (
    <div className="p-6 max-w-4xl mx-auto">
      <div className="flex items-center justify-between mb-1">
        <h1 className="text-2xl font-semibold text-gray-900">KI-Rechtsrecherche</h1>
        {status && (
          <span className={`text-xs px-2 py-1 rounded-full ${status.enabled && status.ollama_available ? "bg-green-100 text-green-700" : "bg-gray-100 text-gray-500"}`}>
            {status.enabled
              ? status.ollama_available
                ? `lokal · ${status.llm_model} · ${status.num_documents} Dokumente`
                : "Ollama nicht erreichbar"
              : "deaktiviert (KI_ENABLED=false)"}
          </span>
        )}
      </div>
      <p className="text-sm text-gray-500 mb-5">
        Quellenbelegte Recherche im lokalen Rechtsbestand (GraphRAG). Antworten sind{" "}
        <strong>Entwürfe</strong> — keine Rechtsberatung, anwaltliche Prüfung erforderlich.
      </p>

      <div className="bg-white border border-gray-200 rounded-lg p-4 mb-5">
        <textarea
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          rows={3}
          placeholder="z. B.: Unter welchen Voraussetzungen ist eine Rechtshandlung nach § 133 InsO anfechtbar?"
          className="w-full border border-gray-300 rounded-md px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
        />
        <div className="flex items-end justify-between mt-2">
          <div>
            <label className="block text-xs text-gray-500 mb-1">Akten-Kontext (optional)</label>
            <input type="number" value={matterId} onChange={(e) => setMatterId(e.target.value)}
              placeholder="Akten-ID" className="w-32 border border-gray-300 rounded-md px-3 py-1.5 text-sm" />
          </div>
          <button
            onClick={() => { setError(null); queryMut.mutate(); }}
            disabled={queryMut.isPending || question.trim().length < 5}
            className="px-4 py-2 bg-blue-600 text-white rounded-md text-sm font-medium hover:bg-blue-700 disabled:opacity-50"
          >
            {queryMut.isPending ? "Recherchiere…" : "Recherchieren"}
          </button>
        </div>
      </div>

      {error && (
        <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded text-sm text-red-700">{error}</div>
      )}

      {result && (
        <div className="space-y-4">
          <div className={`border rounded-lg p-5 ${result.grounded ? "bg-white border-gray-200" : "bg-amber-50 border-amber-200"}`}>
            {!result.grounded && (
              <div className="text-xs font-semibold text-amber-700 uppercase mb-2">
                Keine belegbare Grundlage gefunden
              </div>
            )}
            <div className="prose prose-sm max-w-none whitespace-pre-wrap text-gray-800">{result.answer}</div>
            {result.grounded && result.query_id && !feedbackSent && (
              <div className="mt-4 flex gap-2 text-sm">
                <button onClick={() => feedbackMut.mutate({ fb: "up" })} className="px-3 py-1 border rounded hover:bg-gray-50">
                  Hilfreich
                </button>
                <button onClick={() => feedbackMut.mutate({ fb: "down" })} className="px-3 py-1 border rounded hover:bg-gray-50">
                  Nicht hilfreich
                </button>
              </div>
            )}
            {feedbackSent && <div className="mt-3 text-xs text-green-600">Feedback gespeichert — danke.</div>}
          </div>

          {result.sources.length > 0 && (
            <div className="bg-white border border-gray-200 rounded-lg overflow-hidden">
              <div className="px-4 py-2 text-xs font-medium text-gray-500 uppercase border-b bg-gray-50">
                Quellen ({result.sources.length})
              </div>
              <ul className="divide-y divide-gray-100">
                {result.sources.map((s) => (
                  <li key={s.marker + s.chunk_id} className="px-4 py-3 flex items-start gap-3">
                    <span className="text-xs font-mono bg-blue-100 text-blue-700 px-1.5 py-0.5 rounded">[{s.marker}]</span>
                    <div className="text-sm">
                      <div className="text-gray-800">
                        {s.document_title}
                        {s.heading && <span className="text-gray-500"> — {s.heading}</span>}
                      </div>
                      <div className="text-xs text-gray-400">
                        {SOURCE_LABELS[s.source_type] ?? s.source_type}
                        {s.external_id && <> · {s.external_id}</>}
                        {s.url_or_ref && (
                          <> · <a href={s.url_or_ref} target="_blank" rel="noreferrer" className="underline">Fundstelle</a></>
                        )}
                      </div>
                    </div>
                  </li>
                ))}
              </ul>
            </div>
          )}

          <p className="text-xs text-gray-400">{result.disclaimer} · Modell: {result.model}</p>
        </div>
      )}
    </div>
  );
}
