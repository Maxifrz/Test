/**
 * JWT-Hilfsfunktionen.
 *
 * Warum eine eigene Datei: `atob(token.split(".")[1])` war fehlerhaft. JWT
 * kodiert base64URL — mit "-" und "_" statt "+" und "/" und ohne Padding.
 * `atob` erwartet Standard-Base64 und wirft `InvalidCharacterError`, sobald
 * eines dieser Zeichen auftaucht. Das passiert nicht bei jedem Token, aber
 * zuverlaessig irgendwann — mit einem weissen Bildschirm direkt nach dem Login.
 */

export interface JwtPayload {
  sub: string;
  sid: string;
  role: string;
  exp: number;
  iat: number;
  type?: string;
  scope?: string;
}

/** base64URL → String, mit korrektem Alphabet und wiederhergestelltem Padding. */
export function decodeBase64Url(value: string): string {
  const base64 = value.replace(/-/g, "+").replace(/_/g, "/");
  const padded = base64.padEnd(Math.ceil(base64.length / 4) * 4, "=");
  const binary = atob(padded);
  // Über UTF-8 dekodieren: Umlaute in Namen kämen aus atob sonst als Mojibake.
  const bytes = Uint8Array.from(binary, (c) => c.charCodeAt(0));
  return new TextDecoder("utf-8").decode(bytes);
}

/** Liest die Payload eines JWT. Gibt null zurueck, statt zu werfen. */
export function decodeToken(token: string | null): JwtPayload | null {
  if (!token) return null;
  const parts = token.split(".");
  if (parts.length !== 3) return null;
  try {
    return JSON.parse(decodeBase64Url(parts[1])) as JwtPayload;
  } catch {
    return null;
  }
}

/**
 * Ist das Token abgelaufen? `skewSeconds` sorgt dafuer, dass wir kurz VOR
 * Ablauf erneuern, statt in einen 401 zu laufen.
 */
export function isExpired(payload: JwtPayload | null, skewSeconds = 30): boolean {
  if (!payload?.exp) return true;
  return payload.exp * 1000 <= Date.now() + skewSeconds * 1000;
}

/** Eingeschraenkte Tokens (Pflicht-Passwortwechsel, 2FA-Einrichtung). */
export function isRestricted(payload: JwtPayload | null): boolean {
  return payload?.scope === "pwd_change" || payload?.scope === "totp_setup";
}

export const TOKEN_STORAGE_KEY = "access_token";

export function storedToken(): string | null {
  try {
    return localStorage.getItem(TOKEN_STORAGE_KEY);
  } catch {
    // Privater Modus / blockierte Site-Daten
    return null;
  }
}

export function storeToken(token: string): void {
  try {
    localStorage.setItem(TOKEN_STORAGE_KEY, token);
  } catch {
    /* Kein Speicher verfuegbar — die Sitzung lebt dann nur im Arbeitsspeicher. */
  }
}

export function clearToken(): void {
  try {
    localStorage.removeItem(TOKEN_STORAGE_KEY);
  } catch {
    /* siehe storeToken */
  }
}
