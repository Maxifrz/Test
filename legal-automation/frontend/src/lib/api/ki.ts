import api from "./client";

export interface KiSource {
  marker: string;
  chunk_id: number;
  heading: string | null;
  document_title: string;
  source_type: string;
  external_id: string | null;
  url_or_ref: string | null;
}

export interface KiQueryResult {
  query_id: number | null;
  answer: string;
  grounded: boolean;
  sources: KiSource[];
  model: string;
  disclaimer: string;
}

export interface KiStatus {
  enabled: boolean;
  ollama_available: boolean;
  llm_model: string;
  embed_model: string;
  num_documents: number;
  num_chunks: number;
}

export interface KiDocument {
  id: number;
  source_type: string;
  external_id: string | null;
  title: string;
  jurisdiction: string | null;
  matter_id: number | null;
  is_active: boolean;
  created_at: string;
}

export const kiApi = {
  status: () => api.get<KiStatus>("/ki/status").then((r) => r.data),

  query: (question: string, matter_id?: number) =>
    api.post<KiQueryResult>("/ki/query", { question, matter_id }).then((r) => r.data),

  ingest: (data: {
    source_type: string;
    title: string;
    text: string;
    external_id?: string;
    matter_id?: number;
  }) =>
    api
      .post<{ document_id: number | null; num_chunks: number; duplicate: boolean }>(
        "/ki/ingest",
        data
      )
      .then((r) => r.data),

  documents: () => api.get<KiDocument[]>("/ki/documents").then((r) => r.data),

  feedback: (queryId: number, feedback: "up" | "down", note?: string) =>
    api.post(`/ki/queries/${queryId}/feedback`, { feedback, note }).then((r) => r.data),
};
