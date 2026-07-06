import { Link } from "react-router-dom";
import { useAuth } from "../lib/auth/AuthContext";

const NAV_MODULES = [
  { label: "Mandanten", href: "/clients", ready: true },
  { label: "Akten", href: "/matters", ready: true },
  { label: "E-Mails", href: "/emails", ready: true },
  { label: "Aufgaben & Fristen", href: "/tickets", ready: true },
  { label: "Kalender", href: "/calendar", ready: true },
  { label: "Transkriptionen", href: "/transcriptions", ready: true },
  { label: "Finanzen", href: "/finance", ready: true },
  { label: "DSGVO", href: "/dsgvo", ready: true },
  { label: "KI-Recherche", href: "/recherche", ready: true },
  { label: "Kontaktanfragen", href: "/kontaktanfragen", ready: true },
];

export default function Dashboard() {
  const { user, logout } = useAuth();

  return (
    <div className="min-h-screen bg-gray-50 p-8">
      <div className="max-w-4xl mx-auto">
        <div className="flex justify-between items-center mb-8">
          <h1 className="text-2xl font-semibold text-gray-900">Dashboard</h1>
          <button onClick={logout} className="text-sm text-gray-500 hover:text-gray-700">
            Abmelden
          </button>
        </div>

        <div className="bg-white rounded-lg shadow p-6 mb-6">
          <p className="text-gray-600">
            Willkommen, <strong>{user?.email}</strong>
          </p>
          <p className="text-sm text-gray-400 mt-1">Rolle: {user?.role}</p>
        </div>

        <div className="grid grid-cols-2 gap-4">
          {NAV_MODULES.map(({ label, href }) => (
            <Link
              key={label}
              to={href}
              className="bg-white rounded-lg shadow p-6 flex flex-col gap-1 hover:shadow-md transition-shadow border border-transparent hover:border-blue-200"
            >
              <span className="font-medium text-gray-800">{label}</span>
              <span className="text-xs text-blue-600">Öffnen →</span>
            </Link>
          ))}
        </div>
      </div>
    </div>
  );
}
