import api from "./client";

export interface Client {
  id: number;
  client_number: string;
  first_name: string;
  last_name: string;
  company_name: string | null;
  is_company: boolean;
  email: string | null;
  phone: string | null;
  address_line1: string | null;
  address_line2: string | null;
  postal_code: string | null;
  city: string | null;
  country: string;
  date_of_birth: string | null;
  tax_id: string | null;
  notes: string | null;
  dsgvo_consent_given_at: string | null;
  dsgvo_legal_basis: string | null;
  created_at: string;
  display_name: string;
}

export interface ClientListItem {
  id: number;
  client_number: string;
  display_name: string;
  email: string | null;
  phone: string | null;
  city: string | null;
  created_at: string;
}

export interface ClientListResponse {
  items: ClientListItem[];
  total: number;
  page: number;
  page_size: number;
}

export interface ClientCreate {
  first_name: string;
  last_name: string;
  company_name?: string;
  is_company?: boolean;
  email?: string;
  phone?: string;
  address_line1?: string;
  postal_code?: string;
  city?: string;
  country?: string;
  date_of_birth?: string;
  tax_id?: string;
  notes?: string;
  dsgvo_legal_basis?: string;
}

export const clientsApi = {
  list: (params?: { page?: number; page_size?: number; search?: string }) =>
    api.get<ClientListResponse>("/clients", { params }).then((r) => r.data),

  get: (id: number) => api.get<Client>(`/clients/${id}`).then((r) => r.data),

  create: (data: ClientCreate) =>
    api.post<Client>("/clients", data).then((r) => r.data),

  update: (id: number, data: Partial<ClientCreate>) =>
    api.patch<Client>(`/clients/${id}`, data).then((r) => r.data),

  delete: (id: number) => api.delete(`/clients/${id}`),
};
