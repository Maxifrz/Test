import api from "./client";

export interface ContactRequest {
  id: number;
  name: string;
  email: string;
  phone: string | null;
  standort: string | null;
  rolle: string | null;
  message: string;
  consent_at: string;
  status: "neu" | "erledigt";
  created_at: string;
}

export interface ContactRequestList {
  items: ContactRequest[];
  total: number;
  page: number;
  page_size: number;
}

export const contactApi = {
  list: (params?: { status?: "neu" | "erledigt"; page?: number; page_size?: number }) =>
    api.get<ContactRequestList>("/contact-requests", { params }).then((r) => r.data),

  setStatus: (id: number, status: "neu" | "erledigt") =>
    api.patch<ContactRequest>(`/contact-requests/${id}`, { status }).then((r) => r.data),
};
