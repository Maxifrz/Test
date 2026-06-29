import api from "./client";

export interface Segment {
  id: number;
  segment_index: number;
  speaker: string;
  speaker_label: string | null;
  start_seconds: number;
  end_seconds: number;
  text: string;
  confidence: number | null;
  edited: boolean;
}

export interface TranscriptionListItem {
  id: number;
  title: string;
  meeting_type: string;
  meeting_date: string;
  matter_id: number | null;
  status: string;
  progress_stage: string | null;
  duration_seconds: number | null;
  created_at: string;
}

export interface TranscriptionDetail extends TranscriptionListItem {
  language: string;
  model_used: string | null;
  error_message: string | null;
  original_filename: string | null;
  segments: Segment[];
}

export interface TranscriptionListResponse {
  items: TranscriptionListItem[];
  total: number;
  page: number;
  page_size: number;
}

export interface SearchHit {
  id: number;
  title: string;
  meeting_date: string;
  matter_id: number | null;
  snippet: string;
}

export const transcriptionsApi = {
  list: (params?: { page?: number; page_size?: number; matter_id?: number; status?: string }) =>
    api.get<TranscriptionListResponse>("/transcriptions", { params }).then((r) => r.data),

  get: (id: number) =>
    api.get<TranscriptionDetail>(`/transcriptions/${id}`).then((r) => r.data),

  upload: (data: {
    title: string;
    meeting_type: string;
    meeting_date: string;
    matter_id?: number;
    file: File;
  }) => {
    const form = new FormData();
    form.append("title", data.title);
    form.append("meeting_type", data.meeting_type);
    form.append("meeting_date", data.meeting_date);
    if (data.matter_id != null) form.append("matter_id", String(data.matter_id));
    form.append("file", data.file);
    return api
      .post<TranscriptionDetail>("/transcriptions", form, {
        headers: { "Content-Type": "multipart/form-data" },
      })
      .then((r) => r.data);
  },

  search: (q: string) =>
    api.get<SearchHit[]>("/transcriptions/search", { params: { q } }).then((r) => r.data),

  editSegment: (transcriptionId: number, segmentId: number, data: { text?: string; speaker_label?: string }) =>
    api
      .patch<TranscriptionDetail>(`/transcriptions/${transcriptionId}/segments/${segmentId}`, data)
      .then((r) => r.data),

  renameSpeaker: (transcriptionId: number, speaker: string, label: string) =>
    api
      .post<TranscriptionDetail>(`/transcriptions/${transcriptionId}/rename-speaker`, { speaker, label })
      .then((r) => r.data),

  delete: (id: number) => api.delete(`/transcriptions/${id}`),
};
