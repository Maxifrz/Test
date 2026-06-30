import api from "./client";

export interface ProcessingRecord {
  id: number;
  name: string;
  purpose: string;
  legal_basis: string;
  data_categories: string[];
  data_subjects: string | null;
  recipients: string | null;
  retention: string | null;
  tom: string | null;
  is_active: boolean;
}

export interface ErasureEligibility {
  allowed: boolean;
  blocking_reasons: string[];
}

export interface ErasureRequest {
  id: number;
  client_id: number;
  requested_by_id: number;
  requested_at: string;
  status: string;
  reason: string | null;
  blocking_reasons: string[] | null;
  decided_at: string | null;
  executed_at: string | null;
  certificate_path: string | null;
}

export interface AdminOverview {
  active_sessions: number;
  locked_users: number;
  users_total: number;
  users_with_2fa: number;
  open_erasure_requests: number;
  blocked_erasure_requests: number;
  matters_past_retention: number;
}

export interface RetentionPolicy {
  id: number;
  name: string;
  matter_type: string | null;
  retention_years: number;
  legal_basis: string | null;
  is_active: boolean;
}

export const dsgvoApi = {
  vvt: () => api.get<ProcessingRecord[]>("/dsgvo/vvt").then((r) => r.data),
  retentionPolicies: () => api.get<RetentionPolicy[]>("/dsgvo/retention-policies").then((r) => r.data),
  overview: () => api.get<AdminOverview>("/dsgvo/admin/overview").then((r) => r.data),

  eligibility: (clientId: number) =>
    api.get<ErasureEligibility>(`/dsgvo/erasure-eligibility/${clientId}`).then((r) => r.data),
  listErasure: () => api.get<ErasureRequest[]>("/dsgvo/erasure-requests").then((r) => r.data),
  createErasure: (client_id: number, reason?: string) =>
    api.post<ErasureRequest>("/dsgvo/erasure-requests", { client_id, reason }).then((r) => r.data),
  executeErasure: (id: number) =>
    api.post<ErasureRequest>(`/dsgvo/erasure-requests/${id}/execute`, {}).then((r) => r.data),
  rejectErasure: (id: number) =>
    api.post<ErasureRequest>(`/dsgvo/erasure-requests/${id}/reject`, {}).then((r) => r.data),

  createExport: (clientId: number) =>
    api.post<{ id: number; token: string; download_path: string; expires_at: string }>(
      `/dsgvo/export/${clientId}`, {}
    ).then((r) => r.data),
};
