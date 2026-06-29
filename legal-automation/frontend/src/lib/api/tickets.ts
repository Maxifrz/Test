import api from "./client";

export interface Ticket {
  id: number;
  title: string;
  description: string | null;
  ticket_type: string;
  status: string;
  priority: string;
  due_date: string | null;
  frist_basis: string | null;
  frist_trigger_date: string | null;
  frist_calculation_note: string | null;
  matter_id: number | null;
  assignee_id: number | null;
  created_by_id: number;
  parent_id: number | null;
  sla_due_at: string | null;
  sla_breached: boolean;
  recurrence_rule: string | null;
  closed_at: string | null;
  created_at: string;
}

export interface TicketListItem {
  id: number;
  title: string;
  ticket_type: string;
  status: string;
  priority: string;
  due_date: string | null;
  matter_id: number | null;
  assignee_id: number | null;
  sla_breached: boolean;
}

export interface TicketListResponse {
  items: TicketListItem[];
  total: number;
  page: number;
  page_size: number;
}

export interface TicketCreate {
  title: string;
  description?: string;
  ticket_type?: string;
  priority?: string;
  due_date?: string;
  matter_id?: number;
  assignee_id?: number;
  recurrence_rule?: string;
}

export interface FristTicketCreate {
  frist_type: string;
  trigger_date: string;
  matter_id?: number;
  assignee_id?: number;
}

export const ticketsApi = {
  list: (params?: {
    page?: number;
    page_size?: number;
    status?: string;
    ticket_type?: string;
    matter_id?: number;
  }) => api.get<TicketListResponse>("/tickets", { params }).then((r) => r.data),

  get: (id: number) => api.get<Ticket>(`/tickets/${id}`).then((r) => r.data),

  create: (data: TicketCreate) =>
    api.post<Ticket>("/tickets", data).then((r) => r.data),

  createFrist: (data: FristTicketCreate) =>
    api.post<Ticket>("/tickets/frist", data).then((r) => r.data),

  fristTypes: () =>
    api.get<{ frist_types: string[] }>("/tickets/frist-types").then((r) => r.data),

  update: (id: number, data: Partial<TicketCreate> & { status?: string }) =>
    api.patch<Ticket>(`/tickets/${id}`, data).then((r) => r.data),

  delete: (id: number) => api.delete(`/tickets/${id}`),

  addComment: (id: number, body: string, is_internal = true) =>
    api.post(`/tickets/${id}/comments`, { body, is_internal }).then((r) => r.data),
};
