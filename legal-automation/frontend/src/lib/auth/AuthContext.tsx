import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  ReactNode,
} from "react";
import { authApi } from "../api/auth";
import { refreshAccessToken } from "../api/client";
import { usersApi } from "../api/users";
import {
  clearToken,
  decodeToken,
  isExpired,
  isRestricted,
  storeToken,
  storedToken,
} from "./token";

interface AuthUser {
  id: number;
  email: string;
  full_name: string;
  role: string;
}

interface AuthContextValue {
  user: AuthUser | null;
  isAuthenticated: boolean;
  /** true, solange die gespeicherte Sitzung geprueft wird (verhindert Login-Flackern) */
  isRestoring: boolean;
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

/** Nutzerdaten aus dem Token; full_name wird danach vom Server nachgeladen. */
function userFromToken(token: string, email: string): AuthUser | null {
  const payload = decodeToken(token);
  if (!payload) return null;
  return {
    id: Number.parseInt(payload.sub, 10),
    email,
    full_name: "",
    role: payload.role,
  };
}

export function AuthProvider({ children }: { children: ReactNode }) {
  const [user, setUser] = useState<AuthUser | null>(null);
  const [isRestoring, setIsRestoring] = useState(true);
  const refreshTimer = useRef<number | null>(null);

  /** Vollstaendiges Profil holen — liefert full_name, den das Token nicht traegt. */
  const loadProfile = useCallback(async () => {
    try {
      const { data } = await usersApi.me();
      setUser({
        id: data.id,
        email: data.email,
        full_name: data.full_name,
        role: data.role,
      });
    } catch {
      /* Profil ist Beiwerk — die Sitzung bleibt auch ohne gueltig. */
    }
  }, []);

  const applyToken = useCallback(
    (token: string, email: string) => {
      storeToken(token);
      const next = userFromToken(token, email);
      if (next) {
        setUser(next);
        void loadProfile();
      }
    },
    [loadProfile]
  );

  /**
   * Sitzung beim Start wiederherstellen.
   *
   * Vorher fehlte das komplett: das Token lag zwar im localStorage, `user` war
   * nach jedem Reload aber null — und ProtectedRoute schickte den Nutzer zurueck
   * zum Login. Bei jedem F5, jedem Deep-Link, jedem neuen Tab.
   */
  useEffect(() => {
    let cancelled = false;

    (async () => {
      const token = storedToken();
      const payload = decodeToken(token);

      // Eingeschraenkte Tokens (Pflicht-Passwortwechsel / 2FA-Einrichtung)
      // gelten nicht als angemeldete Sitzung.
      if (!token || !payload || isRestricted(payload)) {
        if (!cancelled) setIsRestoring(false);
        return;
      }

      if (isExpired(payload)) {
        const fresh = await refreshAccessToken();
        if (cancelled) return;
        if (!fresh) {
          clearToken();
          setIsRestoring(false);
          return;
        }
        const freshPayload = decodeToken(fresh);
        if (freshPayload) {
          setUser({
            id: Number.parseInt(freshPayload.sub, 10),
            email: "",
            full_name: "",
            role: freshPayload.role,
          });
        }
      } else {
        setUser({
          id: Number.parseInt(payload.sub, 10),
          email: "",
          full_name: "",
          role: payload.role,
        });
      }

      if (!cancelled) {
        await loadProfile();
        setIsRestoring(false);
      }
    })();

    return () => {
      cancelled = true;
    };
  }, [loadProfile]);

  /**
   * Token erneuern, bevor es ablaeuft. Ohne das laeuft der Nutzer alle 15
   * Minuten in einen 401 und die laufende Aktion (ein halb ausgefuellter
   * Fristeintrag) geht verloren, bis der Retry-Interceptor greift.
   */
  useEffect(() => {
    if (!user) return;

    const schedule = () => {
      if (refreshTimer.current) window.clearTimeout(refreshTimer.current);
      const payload = decodeToken(storedToken());
      if (!payload?.exp) return;
      // 60 s vor Ablauf, mindestens 10 s in der Zukunft
      const delay = Math.max(payload.exp * 1000 - Date.now() - 60_000, 10_000);
      refreshTimer.current = window.setTimeout(async () => {
        await refreshAccessToken();
        schedule();
      }, delay);
    };

    schedule();
    return () => {
      if (refreshTimer.current) window.clearTimeout(refreshTimer.current);
    };
  }, [user]);

  const login = useCallback(
    async (email: string, password: string, totpCode?: string) => {
      const { data } = await authApi.login({ email, password, totp_code: totpCode });
      if (data.requires_totp) return { requires_totp: true };
      if (data.password_change_required) {
        // Eingeschraenktes Token (nur /auth/change-password)
        storeToken(data.access_token);
        return { requires_totp: false, password_change_required: true };
      }
      if (data.totp_setup_required) {
        // Eingeschraenktes Setup-Token (nur /auth/totp/*); noch nicht angemeldet
        storeToken(data.access_token);
        return { requires_totp: false, totp_setup_required: true };
      }
      applyToken(data.access_token, email);
      return { requires_totp: false };
    },
    [applyToken]
  );

  const finishTotpSetup = useCallback(
    async (code: string, email: string) => {
      const { data } = await authApi.confirmTotp(code);
      applyToken(data.access_token, email);
    },
    [applyToken]
  );

  const adoptToken = useCallback(
    (token: string, email: string) => applyToken(token, email),
    [applyToken]
  );

  const logout = useCallback(async () => {
    try {
      await authApi.logout();
    } catch {
      /* Auch wenn der Server nicht erreichbar ist: lokal abmelden. */
    }
    clearToken();
    setUser(null);
  }, []);

  const value = useMemo(
    () => ({
      user,
      isAuthenticated: !!user,
      isRestoring,
      login,
      finishTotpSetup,
      adoptToken,
      logout,
    }),
    [user, isRestoring, login, finishTotpSetup, adoptToken, logout]
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error("useAuth must be used within AuthProvider");
  return ctx;
}
