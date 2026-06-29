import { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { transcriptionsApi, Segment } from "../lib/api/transcriptions";

const MEETING_TYPES = ["Besprechung", "Mandantengespräch", "Zeugenvernehmung", "Verhandlung", "Sonstiges"];

const STATUS_BADGE: Record<string, string> = {
  queued: "bg-gray-100 text-gray-600",
  processing: "bg-blue-100 text-blue-700",
  completed: "bg-green-100 text-green-700",
  failed: "bg-red-100 text-red-700",
};

function fmtTime(s: number) {
  const m = Math.floor(s / 60);
  const sec = Math.floor(s % 60);
  return `${m}:${sec.toString().padStart(2, "0")}`;
}

export default function TranscriptionPage() {
  const qc = useQueryClient();
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [showUpload, setShowUpload] = useState(false);
  const [search, setSearch] = useState("");

  const { data: list, isLoading } = useQuery({
    queryKey: ["transcriptions"],
    queryFn: () => transcriptionsApi.list({ page_size: 50 }),
    refetchInterval: (query) => {
      const items = query.state.data?.items ?? [];
      return items.some((t) => t.status === "queued" || t.status === "processing") ? 4000 : false;
    },
  });

  const { data: detail } = useQuery({
    queryKey: ["transcription", selectedId],
    queryFn: () => transcriptionsApi.get(selectedId!),
    enabled: selectedId !== null,
  });

  const { data: searchHits } = useQuery({
    queryKey: ["transcription-search", search],
    queryFn: () => transcriptionsApi.search(search),
    enabled: search.length >= 2,
  });

  return (
    <div className="p-6 max-w-6xl mx-auto">
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-2xl font-semibold text-gray-900">Transkriptionen</h1>
        <button onClick={() => setShowUpload(true)} className="px-4 py-2 bg-blue-600 text-white rounded-md text-sm font-medium hover:bg-blue-700">
          + Audio hochladen
        </button>
      </div>

      <div className="mb-4">
        <input
          value={search}
          onChange={(e) => setSearch(e.target.value)}
          placeholder="Volltextsuche über alle Transkripte (deutsch)…"
          className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm"
        />
        {search.length >= 2 && searchHits && (
          <div className="mt-2 border rounded-md bg-white divide-y">
            {searchHits.length === 0 ? (
              <div className="p-3 text-sm text-gray-400">Keine Treffer.</div>
            ) : (
              searchHits.map((h) => (
                <button key={h.id} onClick={() => { setSelectedId(h.id); setSearch(""); }} className="block w-full text-left p-3 hover:bg-gray-50">
                  <div className="text-sm font-medium text-gray-800">{h.title}</div>
                  <div className="text-xs text-gray-500" dangerouslySetInnerHTML={{ __html: h.snippet }} />
                </button>
              ))
            )}
          </div>
        )}
      </div>

      <div className="grid grid-cols-5 gap-4">
        <div className="col-span-2 shadow ring-1 ring-black ring-opacity-5 rounded-lg bg-white">
          {isLoading ? (
            <div className="p-4 text-sm text-gray-500">Laden…</div>
          ) : (
            <ul className="divide-y divide-gray-200">
              {list?.items.map((t) => (
                <li key={t.id} onClick={() => setSelectedId(t.id)} className={`p-3 cursor-pointer hover:bg-gray-50 ${selectedId === t.id ? "bg-blue-50" : ""}`}>
                  <div className="flex justify-between items-start">
                    <span className="text-sm font-medium text-gray-800 truncate">{t.title}</span>
                    <span className={`text-xs px-2 py-0.5 rounded-full shrink-0 ml-2 ${STATUS_BADGE[t.status] ?? ""}`}>
                      {t.status === "processing" && t.progress_stage ? t.progress_stage : t.status}
                    </span>
                  </div>
                  <div className="text-xs text-gray-500">
                    {t.meeting_type} · {new Date(t.meeting_date).toLocaleDateString("de-DE")}
                  </div>
                </li>
              ))}
              {list?.items.length === 0 && <li className="p-8 text-center text-sm text-gray-400">Noch keine Transkriptionen.</li>}
            </ul>
          )}
        </div>

        <div className="col-span-3 shadow ring-1 ring-black ring-opacity-5 rounded-lg bg-white p-5">
          {!detail ? (
            <div className="text-sm text-gray-400">Wähle eine Transkription aus.</div>
          ) : detail.status === "failed" ? (
            <div className="text-sm text-red-600">Fehlgeschlagen: {detail.error_message}</div>
          ) : detail.status !== "completed" ? (
            <div className="text-sm text-gray-500">Verarbeitung läuft… ({detail.progress_stage ?? detail.status})</div>
          ) : (
            <TranscriptDetail detail={detail} onChanged={() => {
              qc.invalidateQueries({ queryKey: ["transcription", selectedId] });
            }} />
          )}
        </div>
      </div>

      {showUpload && <UploadModal onClose={() => setShowUpload(false)} onDone={() => { setShowUpload(false); qc.invalidateQueries({ queryKey: ["transcriptions"] }); }} />}
    </div>
  );
}

function TranscriptDetail({ detail, onChanged }: { detail: import("../lib/api/transcriptions").TranscriptionDetail; onChanged: () => void }) {
  const [editing, setEditing] = useState<number | null>(null);
  const [draft, setDraft] = useState("");

  const editMut = useMutation({
    mutationFn: (seg: Segment) => transcriptionsApi.editSegment(detail.id, seg.id, { text: draft }),
    onSuccess: () => { setEditing(null); onChanged(); },
  });
  const renameMut = useMutation({
    mutationFn: ({ speaker, label }: { speaker: string; label: string }) =>
      transcriptionsApi.renameSpeaker(detail.id, speaker, label),
    onSuccess: onChanged,
  });

  const speakers = Array.from(new Set(detail.segments.map((s) => s.speaker)));

  return (
    <div>
      <h2 className="text-lg font-semibold text-gray-900">{detail.title}</h2>
      <div className="text-xs text-gray-500 mb-3">
        {detail.meeting_type} · {new Date(detail.meeting_date).toLocaleDateString("de-DE")}
        {detail.duration_seconds ? ` · ${fmtTime(detail.duration_seconds)} min` : ""} · Modell {detail.model_used}
      </div>

      <div className="mb-3 flex flex-wrap gap-2">
        {speakers.map((sp) => {
          const label = detail.segments.find((s) => s.speaker === sp)?.speaker_label;
          return (
            <button
              key={sp}
              onClick={() => {
                const newLabel = prompt(`Sprecher „${sp}" benennen:`, label ?? "");
                if (newLabel) renameMut.mutate({ speaker: sp, label: newLabel });
              }}
              className="text-xs px-2 py-1 rounded-full border border-gray-300 hover:bg-gray-50"
            >
              {label ? `${label} (${sp})` : sp} ✎
            </button>
          );
        })}
      </div>

      <div className="space-y-2 max-h-[60vh] overflow-y-auto pr-2">
        {detail.segments.map((seg) => (
          <div key={seg.id} className="text-sm">
            <span className="text-xs font-semibold text-blue-700 mr-2">
              [{fmtTime(seg.start_seconds)}] {seg.speaker_label ?? seg.speaker}:
            </span>
            {editing === seg.id ? (
              <span className="inline-flex gap-1 items-start">
                <textarea value={draft} onChange={(e) => setDraft(e.target.value)} className="border rounded p-1 text-sm w-96" rows={2} />
                <button onClick={() => editMut.mutate(seg)} className="text-xs text-green-700">✓</button>
                <button onClick={() => setEditing(null)} className="text-xs text-gray-400">✕</button>
              </span>
            ) : (
              <span onClick={() => { setEditing(seg.id); setDraft(seg.text); }} className="cursor-text hover:bg-yellow-50">
                {seg.text}{seg.edited && <span className="ml-1 text-xs text-gray-400">(bearbeitet)</span>}
              </span>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

function UploadModal({ onClose, onDone }: { onClose: () => void; onDone: () => void }) {
  const [title, setTitle] = useState("");
  const [meetingType, setMeetingType] = useState("Besprechung");
  const [meetingDate, setMeetingDate] = useState("");
  const [matterId, setMatterId] = useState("");
  const [file, setFile] = useState<File | null>(null);
  const [error, setError] = useState<string | null>(null);

  const mut = useMutation({
    mutationFn: () =>
      transcriptionsApi.upload({
        title,
        meeting_type: meetingType,
        meeting_date: meetingDate,
        matter_id: matterId ? parseInt(matterId) : undefined,
        file: file!,
      }),
    onSuccess: onDone,
    onError: (e: any) => setError(e?.response?.data?.detail ?? "Upload fehlgeschlagen"),
  });

  return (
    <div className="fixed inset-0 bg-black bg-opacity-40 flex items-center justify-center z-50">
      <div className="bg-white rounded-lg shadow-xl w-full max-w-lg p-6">
        <h2 className="text-lg font-semibold mb-4">Audio hochladen</h2>
        {error && <div className="mb-4 p-3 bg-red-50 border border-red-200 rounded text-sm text-red-700">{error}</div>}
        <div className="space-y-4">
          <input placeholder="Titel" value={title} onChange={(e) => setTitle(e.target.value)} className="inp" />
          <div className="grid grid-cols-2 gap-3">
            <select value={meetingType} onChange={(e) => setMeetingType(e.target.value)} className="inp">
              {MEETING_TYPES.map((t) => <option key={t} value={t}>{t}</option>)}
            </select>
            <input type="date" value={meetingDate} onChange={(e) => setMeetingDate(e.target.value)} className="inp" />
          </div>
          <input type="number" placeholder="Akten-ID (optional)" value={matterId} onChange={(e) => setMatterId(e.target.value)} className="inp" />
          <input type="file" accept="audio/*,video/mp4" onChange={(e) => setFile(e.target.files?.[0] ?? null)} className="text-sm" />
          <p className="text-xs text-gray-500">
            Die Verarbeitung läuft vollständig lokal (faster-whisper + pyannote). Das Original wird
            verschlüsselt abgelegt; das Zwischen-WAV wird nach der Transkription gelöscht.
          </p>
        </div>
        <div className="flex justify-end gap-3 pt-5">
          <button onClick={onClose} className="px-4 py-2 text-sm text-gray-700 border rounded-md hover:bg-gray-50">Abbrechen</button>
          <button onClick={() => { setError(null); mut.mutate(); }} disabled={mut.isPending || !title || !meetingDate || !file} className="px-4 py-2 text-sm bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50">
            {mut.isPending ? "Hochladen…" : "Hochladen & transkribieren"}
          </button>
        </div>
        <style>{`.inp{width:100%;padding:0.5rem 0.75rem;border:1px solid #d1d5db;border-radius:0.375rem;font-size:0.875rem}`}</style>
      </div>
    </div>
  );
}
