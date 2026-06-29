import api from "./client";

export interface Attendee {
  id: number;
  user_id: number | null;
  external_name: string | null;
  external_email: string | null;
  response_status: string;
}

export interface CalendarEvent {
  id: number;
  title: string;
  description: string | null;
  event_type: string;
  start_at: string;
  end_at: string;
  all_day: boolean;
  location: string | null;
  travel_buffer_minutes: number;
  matter_id: number | null;
  organizer_id: number;
  ticket_id: number | null;
  status: string;
  recurrence_rule: string | null;
  source: string;
  external_uid: string | null;
  created_at: string;
  attendees: Attendee[];
}

export interface ConflictItem {
  kind: string; // overlap | vacation | holiday
  detail: string;
  event_id: number | null;
}

export interface EventCreate {
  title: string;
  event_type?: string;
  start_at: string;
  end_at: string;
  description?: string;
  location?: string;
  matter_id?: number;
  organizer_id?: number;
  travel_buffer_minutes?: number;
  generate_preparation?: boolean;
  force?: boolean;
}

export interface EventCreateResponse {
  event: CalendarEvent;
  conflicts: ConflictItem[];
  created_preparation_ticket_ids: number[];
}

export interface LadungParseResponse {
  found: boolean;
  hearing_date: string | null;
  hearing_time: string | null;
  aktenzeichen: string | null;
  room: string | null;
  confidence: number;
  note: string;
  suggested_title: string | null;
}

export const calendarApi = {
  list: (params?: {
    start?: string;
    end?: string;
    organizer_id?: number;
    matter_id?: number;
    event_type?: string;
  }) => api.get<CalendarEvent[]>("/calendar", { params }).then((r) => r.data),

  get: (id: number) => api.get<CalendarEvent>(`/calendar/${id}`).then((r) => r.data),

  create: (data: EventCreate) =>
    api.post<EventCreateResponse>("/calendar", data).then((r) => r.data),

  conflictCheck: (organizer_id: number, start_at: string, end_at: string) =>
    api
      .post<{ conflicts: ConflictItem[]; has_blocking: boolean }>("/calendar/conflict-check", {
        organizer_id,
        start_at,
        end_at,
      })
      .then((r) => r.data),

  update: (id: number, data: Partial<EventCreate> & { status?: string }) =>
    api.patch<CalendarEvent>(`/calendar/${id}`, data).then((r) => r.data),

  delete: (id: number) => api.delete(`/calendar/${id}`),

  parseLadung: (email_id: number) =>
    api.post<LadungParseResponse>("/calendar/parse-ladung", { email_id }).then((r) => r.data),

  exportIcsUrl: (id: number) => `/api/calendar/${id}/export.ics`,

  importIcs: (file: File) => {
    const form = new FormData();
    form.append("file", file);
    return api
      .post<{ imported: number; event_ids: number[] }>("/calendar/import-ics", form, {
        headers: { "Content-Type": "multipart/form-data" },
      })
      .then((r) => r.data);
  },
};
