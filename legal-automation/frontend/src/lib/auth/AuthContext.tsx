import { createContext, useContext, useState, useCallback, ReactNode } from "react";
import { authApi } from "../api/auth";

interface AuthUser {
  id: number;
  email: string;
  full_name: string;
  role: string;
}

interface AuthContextValue {
  user: AuthUser | null;
  isAuthenticated: boolean;
  login: (
    email: string,
    password: string,
    totpCode?: string
  ) => Promise<{
    requires_totp: boolean;
    totp_setup_required?: boolean;
    password_change_required?: boolean;
  }>;
  finishTotpSetup: (code: string, email: string) => Promise<void>;
  adoptToken: (token: string, email: string) => void;
  logout: () => Promise<void>;
}

const AuthContext = createContext<AuthContextValue | null>(null);

function applyToken(token: string, email: string, setUser: (u: AuthUser) => void) {
  localStorage.setItem("access_token", token);
  // Decode JWT to get user info (payload is not sensitive)
  const payload = JSON.parse(atob(token.split(".")[1]));
  setUser({ id: parseInt(payload.sub), email, full_name: "", role: payload.role });
}

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<AuthUser | null>(null);

  const login = useCallback(async (email: string, password: string, totpCode?: string) => {
    const { data } = await authApi.login({ email, password, totp_code: totpCode });
    if (data.requires_totp) return { requires_totp: true };
    if (data.password_change_required) {
      // Eingeschränktes Token (nur /auth/change-password) speichern
      localStorage.setItem("access_token", data.access_token);
      return { requires_totp: false, password_change_required: true };
    }
    if (data.totp_setup_required) {
      // Eingeschränktes Setup-Token speichern (erlaubt nur /auth/totp/*),
      // Nutzer gilt noch NICHT als angemeldet.
      localStorage.setItem("access_token", data.access_token);
      return { requires_totp: false, totp_setup_required: true };
    }
    applyToken(data.access_token, email, setUser);
    return { requires_totp: false };
  }, []);

  const finishTotpSetup = useCallback(async (code: string, email: string) => {
    const { data } = await authApi.confirmTotp(code);
    applyToken(data.access_token, email, setUser);
  }, []);

  const adoptToken = useCallback((token: string, email: string) => {
    applyToken(token, email, setUser);
  }, []);

  const logout = useCallback(async () => {
    try { await authApi.logout(); } catch {}
    localStorage.removeItem("access_token");
    setUser(null);
  }, []);

  return (
    <AuthContext.Provider value={{ user, isAuthenticated: !!user, login, finishTotpSetup, adoptToken, logout }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used within AuthProvider");
  return ctx;
}
