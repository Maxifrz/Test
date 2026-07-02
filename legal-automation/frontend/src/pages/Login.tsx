import { useState } from "react";
import { useNavigate, useSearchParams } from "react-router-dom";
import { useForm } from "react-hook-form";
import { zodResolver } from "@hookform/resolvers/zod";
import { z } from "zod";
import { useAuth } from "../lib/auth/AuthContext";
import { authApi } from "../lib/api/auth";

const loginSchema = z.object({
  email: z.string().email("Ungültige E-Mail-Adresse"),
  password: z.string().min(1, "Passwort erforderlich"),
});

const totpSchema = z.object({
  code: z.string().length(6, "6-stelliger Code erforderlich"),
});

type LoginForm = z.infer<typeof loginSchema>;
type TotpForm = z.infer<typeof totpSchema>;

export default function LoginPage() {
  const { login, finishTotpSetup, adoptToken } = useAuth();
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const [requiresTotp, setRequiresTotp] = useState(false);
  const [setupInfo, setSetupInfo] = useState<{ secret: string; qr_uri: string } | null>(null);
  const [pwdChangeMode, setPwdChangeMode] = useState(false);
  const [newPw, setNewPw] = useState({ current: "", next: "", repeat: "" });
  const [pendingCreds, setPendingCreds] = useState<{ email: string; password: string } | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  const sessionExpired = searchParams.get("reason") === "session_expired";

  const {
    register: registerLogin,
    handleSubmit: handleLoginSubmit,
    formState: { errors: loginErrors },
  } = useForm<LoginForm>({ resolver: zodResolver(loginSchema) });

  const {
    register: registerTotp,
    handleSubmit: handleTotpSubmit,
    formState: { errors: totpErrors },
  } = useForm<TotpForm>({ resolver: zodResolver(totpSchema) });

  const onLogin = async (data: LoginForm) => {
    setError(null);
    setIsLoading(true);
    try {
      const result = await login(data.email, data.password);
      if (result.requires_totp) {
        setPendingCreds({ email: data.email, password: data.password });
        setRequiresTotp(true);
      } else if (result.password_change_required) {
        // Initial-/Pflicht-Passwortwechsel erzwingen
        setPendingCreds({ email: data.email, password: data.password });
        setPwdChangeMode(true);
      } else if (result.totp_setup_required) {
        // Rolle erfordert 2FA, aber noch nicht eingerichtet → Setup erzwingen
        setPendingCreds({ email: data.email, password: data.password });
        const { data: setup } = await authApi.setupTotp();
        setSetupInfo(setup);
      } else {
        navigate("/");
      }
    } catch (err: any) {
      const status = err.response?.status;
      if (status === 429) setError("Konto gesperrt. Bitte warten Sie und versuchen Sie es erneut.");
      else if (status === 401) setError("Ungültige E-Mail-Adresse oder Passwort.");
      else setError("Anmeldung fehlgeschlagen. Bitte versuchen Sie es erneut.");
    } finally {
      setIsLoading(false);
    }
  };

  const onTotp = async (data: TotpForm) => {
    if (!pendingCreds) return;
    setError(null);
    setIsLoading(true);
    try {
      await login(pendingCreds.email, pendingCreds.password, data.code);
      navigate("/");
    } catch {
      setError("Ungültiger 2FA-Code. Bitte versuchen Sie es erneut.");
    } finally {
      setIsLoading(false);
    }
  };

  const onPasswordChange = async () => {
    if (!pendingCreds) return;
    if (newPw.next !== newPw.repeat) {
      setError("Die neuen Passwörter stimmen nicht überein.");
      return;
    }
    setError(null);
    setIsLoading(true);
    try {
      const { data } = await authApi.changePassword(newPw.current || pendingCreds.password, newPw.next);
      if (data.totp_setup_required) {
        // Kette: nach Passwortwechsel ist noch die 2FA-Einrichtung fällig
        localStorage.setItem("access_token", data.access_token);
        setPwdChangeMode(false);
        const { data: setup } = await authApi.setupTotp();
        setSetupInfo(setup);
      } else {
        adoptToken(data.access_token, pendingCreds.email);
        navigate("/");
      }
    } catch (err: any) {
      setError(err.response?.data?.detail ?? "Passwortwechsel fehlgeschlagen.");
    } finally {
      setIsLoading(false);
    }
  };

  const onSetupConfirm = async (data: TotpForm) => {
    if (!pendingCreds) return;
    setError(null);
    setIsLoading(true);
    try {
      await finishTotpSetup(data.code, pendingCreds.email);
      navigate("/");
    } catch {
      setError("Ungültiger Code. Bitte prüfen Sie die Einrichtung in Ihrer Authenticator-App.");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 flex items-center justify-center">
      <div className="w-full max-w-sm bg-white rounded-lg shadow p-8 space-y-6">
        <div>
          <h1 className="text-2xl font-semibold text-gray-900">Kanzlei Automation</h1>
          <p className="text-sm text-gray-500 mt-1">Bitte melden Sie sich an</p>
        </div>

        {sessionExpired && (
          <div className="bg-yellow-50 border border-yellow-200 rounded p-3 text-sm text-yellow-800">
            Ihre Sitzung ist abgelaufen. Bitte melden Sie sich erneut an.
          </div>
        )}

        {error && (
          <div className="bg-red-50 border border-red-200 rounded p-3 text-sm text-red-800">
            {error}
          </div>
        )}

        {pwdChangeMode ? (
          <form onSubmit={(e) => { e.preventDefault(); onPasswordChange(); }} className="space-y-4">
            <div className="bg-amber-50 border border-amber-200 rounded p-3 text-sm text-amber-800">
              <strong>Passwortwechsel erforderlich.</strong> Bitte vergeben Sie jetzt ein
              neues Passwort (mind. 10 Zeichen, Groß-/Kleinbuchstabe, Ziffer, Sonderzeichen).
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Neues Passwort</label>
              <input type="password" autoComplete="new-password" value={newPw.next}
                onChange={(e) => setNewPw((s) => ({ ...s, next: e.target.value }))}
                className="w-full border border-gray-300 rounded px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500" />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Neues Passwort wiederholen</label>
              <input type="password" autoComplete="new-password" value={newPw.repeat}
                onChange={(e) => setNewPw((s) => ({ ...s, repeat: e.target.value }))}
                className="w-full border border-gray-300 rounded px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500" />
            </div>
            <button type="submit" disabled={isLoading || !newPw.next}
              className="w-full bg-blue-600 text-white py-2 px-4 rounded text-sm font-medium hover:bg-blue-700 disabled:opacity-50">
              {isLoading ? "Speichern..." : "Passwort ändern & fortfahren"}
            </button>
          </form>
        ) : setupInfo ? (
          <form onSubmit={handleTotpSubmit(onSetupConfirm)} className="space-y-4">
            <div className="bg-blue-50 border border-blue-200 rounded p-3 text-sm text-blue-800">
              <strong>2FA-Einrichtung erforderlich.</strong> Ihre Rolle verlangt
              Zwei-Faktor-Authentifizierung. Fügen Sie das Konto in Ihrer
              Authenticator-App hinzu (Secret oder URI) und bestätigen Sie mit einem Code.
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Secret (manuell eintragen)</label>
              <code className="block bg-gray-100 rounded px-3 py-2 text-xs break-all select-all">{setupInfo.secret}</code>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">otpauth-URI</label>
              <code className="block bg-gray-100 rounded px-3 py-2 text-xs break-all select-all">{setupInfo.qr_uri}</code>
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Bestätigungscode</label>
              <input
                {...registerTotp("code")}
                type="text"
                inputMode="numeric"
                maxLength={6}
                autoFocus
                className="w-full border border-gray-300 rounded px-3 py-2 text-sm text-center tracking-widest focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
              {totpErrors.code && (
                <p className="text-red-600 text-xs mt-1">{totpErrors.code.message}</p>
              )}
            </div>
            <button
              type="submit"
              disabled={isLoading}
              className="w-full bg-blue-600 text-white py-2 px-4 rounded text-sm font-medium hover:bg-blue-700 disabled:opacity-50"
            >
              {isLoading ? "Aktivierung läuft..." : "2FA aktivieren & anmelden"}
            </button>
          </form>
        ) : !requiresTotp ? (
          <form onSubmit={handleLoginSubmit(onLogin)} className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">E-Mail</label>
              <input
                {...registerLogin("email")}
                type="email"
                autoComplete="email"
                className="w-full border border-gray-300 rounded px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
              {loginErrors.email && (
                <p className="text-red-600 text-xs mt-1">{loginErrors.email.message}</p>
              )}
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Passwort</label>
              <input
                {...registerLogin("password")}
                type="password"
                autoComplete="current-password"
                className="w-full border border-gray-300 rounded px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
              {loginErrors.password && (
                <p className="text-red-600 text-xs mt-1">{loginErrors.password.message}</p>
              )}
            </div>
            <button
              type="submit"
              disabled={isLoading}
              className="w-full bg-blue-600 text-white py-2 px-4 rounded text-sm font-medium hover:bg-blue-700 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isLoading ? "Anmeldung läuft..." : "Anmelden"}
            </button>
          </form>
        ) : (
          <form onSubmit={handleTotpSubmit(onTotp)} className="space-y-4">
            <p className="text-sm text-gray-600">
              Bitte geben Sie den 6-stelligen Code aus Ihrer Authenticator-App ein.
            </p>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">2FA-Code</label>
              <input
                {...registerTotp("code")}
                type="text"
                inputMode="numeric"
                maxLength={6}
                autoFocus
                className="w-full border border-gray-300 rounded px-3 py-2 text-sm text-center tracking-widest focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
              {totpErrors.code && (
                <p className="text-red-600 text-xs mt-1">{totpErrors.code.message}</p>
              )}
            </div>
            <button
              type="submit"
              disabled={isLoading}
              className="w-full bg-blue-600 text-white py-2 px-4 rounded text-sm font-medium hover:bg-blue-700 disabled:opacity-50"
            >
              {isLoading ? "Prüfung läuft..." : "Bestätigen"}
            </button>
          </form>
        )}

        <p className="text-xs text-gray-400 text-center">
          DSGVO-konform · Daten bleiben auf Ihrem Server
        </p>
      </div>
    </div>
  );
}
