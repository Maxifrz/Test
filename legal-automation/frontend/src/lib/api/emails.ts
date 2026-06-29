import api from "./client";

export interface EmailListItem {
  id: number;
  direction: string;
  from_address: string;
  subject: string | null;
  matter_id: number | null;
  client_id: number | null;
  is_read: boolean;
  needs_review: boolean;
  is_confidential: boolean;
  unknown_sender: boolean;
  email_date: string | null;
}

export interface EmailDetail extends EmailListItem {
  to_addresses: string[];
  cc_addresses: string[] | null;
  body_text: string | null;
  body_html: string | null;
  in_reply_to: string | null;
  thread_key: string | null;
  delivery_status: string | null;
}

export interface EmailListResponse {
  items: EmailListItem[];
  total: number;
  page: number;
  page_size: number;
}

export interface EmailRule {
  id: number;
  name: string;
  priority: number;
  conditions: Record<string, unknown>;
  actions: Record<string, unknown>;
  is_active: boolean;
}

export interface EmailTemplate {
  id: number;
  name: string;
  category: string | null;
  subject_template: string;
  body_template: string;
  variables_doc: Record<string, unknown> | null;
  is_active: boolean;
}

export const emailsApi = {
  list: (params?: {
    page?: number;
    page_size?: number;
    direction?: string;
    matter_id?: number;
    needs_review?: boolean;
  }) => api.get<EmailListResponse>("/emails", { params }).then((r) => r.data),

  get: (id: number) => api.get<EmailDetail>(`/emails/${id}`).then((r) => r.data),

  fileToMatter: (id: number, matter_id: number) =>
    api.post<EmailDetail>(`/emails/${id}/file`, { matter_id }).then((r) => r.data),

  send: (data: {
    to_addresses: string[];
    subject: string;
    body_text: string;
    body_html?: string;
    matter_id?: number;
    client_id?: number;
  }) => api.post<EmailDetail>("/emails/send", data).then((r) => r.data),

  listRules: () => api.get<EmailRule[]>("/emails/rules").then((r) => r.data),
  createRule: (data: Omit<EmailRule, "id">) =>
    api.post<EmailRule>("/emails/rules", data).then((r) => r.data),

  listTemplates: () =>
    api.get<EmailTemplate[]>("/emails/templates").then((r) => r.data),
  previewTemplate: (template_id: number, context: Record<string, unknown>) =>
    api
      .post<{ subject: string; body: string }>("/emails/templates/preview", {
        template_id,
        context,
      })
      .then((r) => r.data),
};
