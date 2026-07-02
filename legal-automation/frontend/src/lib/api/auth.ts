import api from "./client";

export interface LoginPayload {
  email: string;
  password: string;
  totp_code?: string;
}

export interface TokenResponse {
  access_token: string;
  token_type: string;
  requires_totp: boolean;
  totp_setup_required?: boolean;
  password_change_required?: boolean;
}

export const authApi = {
  login: (payload: LoginPayload) =>
    api.post<TokenResponse>("/auth/login", payload),

  logout: () => api.post("/auth/logout"),

  refresh: () => api.post<TokenResponse>("/auth/refresh"),

  setupTotp: () => api.post<{ secret: string; qr_uri: string }>("/auth/totp/setup"),

  changePassword: (current_password: string, new_password: string) =>
    api.post<TokenResponse>("/auth/change-password", { current_password, new_password }),

  confirmTotp: (code: string) => api.post<TokenResponse>("/auth/totp/confirm", { code }),
};
