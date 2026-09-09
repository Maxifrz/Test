import axios, { AxiosError, InternalAxiosRequestConfig } from "axios";
import { clearToken, storeToken, storedToken } from "../auth/token";

const api = axios.create({
  baseURL: "/api",
  withCredentials: true,
  headers: { "Content-Type": "application/json" },
});

/**
 * Erneuert das Access-Token ueber das httpOnly-Refresh-Cookie.
 *
 * Das war der gravierendste Fehler im Frontend: `authApi.refresh` existierte,
 * wurde aber nirgends aufgerufen. Das Access-Token laeuft nach 15 Minuten ab,
 * danach schlugen alle Anfragen still mit 401 fehl — ohne Erneuerung und ohne
 * Weiterleitung, weil der Interceptor nur auf den Header `X-Session-Expired`
 * reagierte, den der Backend-Pfad fuer abgelaufene JWTs gar nicht setzt.
 */
let refreshInFlight: Promise<string | null> | null = null;

export async function refreshAccessToken(): Promise<string | null> {
  // Parallele 401er duerfen nur EINEN Refresh ausloesen, sonst ueberholen
  // sich die Anfragen und verbrauchen die Session mehrfach.
  if (refreshInFlight) return refreshInFlight;

  refreshInFlight = (async () => {
    try {
      // Bewusst nicht ueber `api`: sonst wuerde der Response-Interceptor bei
      // einem fehlgeschlagenen Refresh rekursiv erneut refreshen.
      const { data } = await axios.post<{ access_token: string }>(
        "/api/auth/refresh",
        {},
        { withCredentials: true }
      );
      if (data?.access_token) {
        storeToken(data.access_token);
        return data.access_token;
      }
      return null;
    } catch {
      return null;
    } finally {
      refreshInFlight = null;
    }
  })();

  return refreshInFlight;
}

function toLogin(reason: string): void {
  clearToken();
  if (!window.location.pathname.startsWith("/login")) {
    window.location.href = `/login?reason=${reason}`;
  }
}

// Access-Token an jede Anfrage haengen
api.interceptors.request.use((config) => {
  const token = storedToken();
  if (token) config.headers.Authorization = `Bearer ${token}`;
  return config;
});

interface RetriableConfig extends InternalAxiosRequestConfig {
  _retried?: boolean;
}

api.interceptors.response.use(
  (r) => r,
  async (error: AxiosError) => {
    const config = error.config as RetriableConfig | undefined;
    const status = error.response?.status;
    const headers = error.response?.headers as Record<string, string> | undefined;

    // Server hat die Session widerrufen (Logout anderswo, Admin-Reset,
    // Passwortwechsel) — ein Refresh wuerde daran nichts aendern.
    if (status === 401 && headers?.["x-session-expired"]) {
      toLogin("session_expired");
      return Promise.reject(error);
    }

    // Eingeschraenkte Tokens: das Backend verlangt zuerst den Pflicht-Schritt.
    if (status === 403 && headers?.["x-password-change-required"]) {
      window.location.href = "/login?reason=password_change_required";
      return Promise.reject(error);
    }
    if (status === 403 && headers?.["x-2fa-setup-required"]) {
      window.location.href = "/login?reason=totp_setup_required";
      return Promise.reject(error);
    }

    // Abgelaufenes Access-Token: einmal erneuern und die Anfrage wiederholen.
    const isRefreshCall = config?.url?.includes("/auth/refresh");
    if (status === 401 && config && !config._retried && !isRefreshCall) {
      config._retried = true;
      const token = await refreshAccessToken();
      if (token) {
        config.headers.Authorization = `Bearer ${token}`;
        return api.request(config);
      }
      toLogin("session_expired");
    }

    return Promise.reject(error);
  }
);

export default api;
