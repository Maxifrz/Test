import api from "./client";

export interface Matter {
  id: number;
  matter_number: string;
  title: string;
  matter_type: string;
  status: string;
  client_id: number;
  lead_anwalt_id: number;
  created_by_id: number;
  court_file_ref: string | null;
  court_name: string | null;
  opposing_party: string | null;
  opposing_counsel: string | null;
  opened_at: string;
  closed_at: string | null;
  statute_of_limitations: string | null;
  retention_years: number;
  description: string | null;
  created_at: string;
}

export interface MatterListItem {
  id: number;
  matter_number: string;
  title: string;
  matter_type: string;
  status: string;
  client_id: number;
  lead_anwalt_id: number;
  opened_at: string;
  created_at: string;
}

export interface MatterListResponse {
  items: MatterListItem[];
  total: number;
  page: number;
  page_size: number;
}

export interface MatterCreate {
  title: string;
  matter_type: string;
  client_id: number;
  lead_anwalt_id: number;
  court_file_ref?: string;
  court_name?: string;
  opposing_party?: string;
  opposing_counsel?: string;
  statute_of_limitations?: string;
  retention_years?: number;
  description?: string;
}

export interface MatterAccessGrant {
  user_id: number;
  matter_role: "lead" | "support" | "readonly";
}

export interface MatterAccess {
  id: number;
  user_id: number;
  matter_id: number;
  matter_role: string;
  granted_by_id: number;
  granted_at: string;
  revoked_at: string | null;
}

export const mattersApi = {
  list: (params?: {
    page?: number;
    page_size?: number;
    status?: string;
    matter_type?: string;
    client_id?: number;
  }) => api.get<MatterListResponse>("/matters", { params }).then((r) => r.data),

  get: (id: number) => api.get<Matter>(`/matters/${id}`).then((r) => r.data),

  create: (data: MatterCreate) =>
    api.post<Matter>("/matters", data).then((r) => r.data),

  update: (id: number, data: Partial<MatterCreate> & { status?: string }) =>
    api.patch<Matter>(`/matters/${id}`, data).then((r) => r.data),

  delete: (id: number) => api.delete(`/matters/${id}`),

  listAccess: (id: number) =>
    api.get<MatterAccess[]>(`/matters/${id}/access`).then((r) => r.data),

  grantAccess: (id: number, data: MatterAccessGrant) =>
    api.post<MatterAccess>(`/matters/${id}/access`, data).then((r) => r.data),

  revokeAccess: (id: number, user_id: number) =>
    api.delete(`/matters/${id}/access`, { data: { user_id } }),
};
