import api from "./client";

export interface UserResponse {
  id: number;
  email: string;
  full_name: string;
  role: string;
  is_active: boolean;
  totp_enabled: boolean;
  must_change_password: boolean;
  phone: string | null;
  last_login: string | null;
  created_at: string | null;
}

export interface UserListResponse {
  items: UserResponse[];
  total: number;
  page: number;
  page_size: number;
}

export interface UserCreateResponse {
  user: UserResponse;
  /** Einmal-Passwort — wird nur bei der Anlage geliefert und nirgends gespeichert. */
  initial_password: string;
}

export interface PasswordResetResponse {
  user_id: number;
  initial_password: string;
  sessions_revoked: number;
}

export const ROLE_LABELS: Record<string, string> = {
  admin: "Administration",
  anwalt: "Anwalt/Anwältin",
  sachbearbeiter: "Sachbearbeitung",
  sekretariat: "Sekretariat",
};

export const usersApi = {
  me: () => api.get<UserResponse>("/users/me"),
  updateMe: (data: { full_name?: string; phone?: string; signature_html?: string }) =>
    api.patch<UserResponse>("/users/me", data),

  list: (params?: {
    role?: string;
    is_active?: boolean;
    search?: string;
    page?: number;
    page_size?: number;
  }) => api.get<UserListResponse>("/users", { params }),

  get: (id: number) => api.get<UserResponse>(`/users/${id}`),

  create: (data: { email: string; full_name: string; role: string; phone?: string }) =>
    api.post<UserCreateResponse>("/users", data),

  update: (
    id: number,
    data: { full_name?: string; role?: string; phone?: string; is_active?: boolean }
  ) => api.patch<UserResponse>(`/users/${id}`, data),

  resetPassword: (id: number) =>
    api.post<PasswordResetResponse>(`/users/${id}/reset-password`),

  reset2fa: (id: number) =>
    api.post<{ detail: string; sessions_revoked: number }>(`/users/${id}/reset-2fa`),

  remove: (id: number) => api.delete(`/users/${id}`),
};
