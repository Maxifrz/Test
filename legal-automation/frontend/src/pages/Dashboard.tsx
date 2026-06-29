import { useAuth } from "../lib/auth/AuthContext";

export default function Dashboard() {
  const { user, logout } = useAuth();

  return (
    <div className="min-h-screen bg-gray-50 p-8">
      <div className="max-w-4xl mx-auto">
        <div className="flex justify-between items-center mb-8">
          <h1 className="text-2xl font-semibold text-gray-900">Dashboard</h1>
          <button
            onClick={logout}
            className="text-sm text-gray-500 hover:text-gray-700"
          >
            Abmelden
          </button>
        </div>

        <div className="bg-white rounded-lg shadow p-6">
          <p className="text-gray-600">
            Willkommen, <strong>{user?.email}</strong>
          </p>
          <p className="text-sm text-gray-400 mt-1">Rolle: {user?.role}</p>
        </div>

        {/* Module placeholders — filled in Phases 2–7 */}
        <div className="grid grid-cols-2 gap-4 mt-6">
          {[
            { label: "E-Mails", phase: 2 },
            { label: "Aufgaben", phase: 3 },
            { label: "Kalender", phase: 4 },
            { label: "Transkriptionen", phase: 5 },
            { label: "Finanzen", phase: 6 },
            { label: "DSGVO", phase: 7 },
          ].map(({ label, phase }) => (
            <div
              key={label}
              className="bg-white rounded-lg shadow p-6 flex flex-col gap-2"
            >
              <span className="font-medium text-gray-700">{label}</span>
              <span className="text-xs text-gray-400">Phase {phase} — in Entwicklung</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
