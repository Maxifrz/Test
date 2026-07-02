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
}

export const authApi = {
  login: (payload: LoginPayload) =>
    api.post<TokenResponse>("/auth/login", payload),

  logout: () => api.post("/auth/logout"),

  refresh: () => api.post<TokenResponse>("/auth/refresh"),

  setupTotp: () => api.post<{ secret: string; qr_uri: string }>("/auth/totp/setup"),

  confirmTotp: (code: string) => api.post<TokenResponse>("/auth/totp/confirm", { code }),
};
