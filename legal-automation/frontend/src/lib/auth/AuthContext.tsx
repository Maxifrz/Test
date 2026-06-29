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
  login: (email: string, password: string, totpCode?: string) => Promise<{ requires_totp: boolean }>;
  logout: () => Promise<void>;
}

const AuthContext = createContext<AuthContextValue | null>(null);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<AuthUser | null>(null);

  const login = useCallback(async (email: string, password: string, totpCode?: string) => {
    const { data } = await authApi.login({ email, password, totp_code: totpCode });
    if (data.requires_totp) return { requires_totp: true };
    localStorage.setItem("access_token", data.access_token);
    // Decode JWT to get user info (payload is not sensitive)
    const payload = JSON.parse(atob(data.access_token.split(".")[1]));
    setUser({ id: parseInt(payload.sub), email, full_name: "", role: payload.role });
    return { requires_totp: false };
  }, []);

  const logout = useCallback(async () => {
    try { await authApi.logout(); } catch {}
    localStorage.removeItem("access_token");
    setUser(null);
  }, []);

  return (
    <AuthContext.Provider value={{ user, isAuthenticated: !!user, login, logout }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used within AuthProvider");
  return ctx;
}
