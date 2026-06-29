import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { AuthProvider, useAuth } from "./lib/auth/AuthContext";
import LoginPage from "./pages/Login";
import Dashboard from "./pages/Dashboard";
import ClientsPage from "./pages/Clients";
import MattersPage from "./pages/Matters";
import EmailsPage from "./pages/Emails";
import TicketsPage from "./pages/Tickets";
import CalendarPage from "./pages/Calendar";
import TranscriptionPage from "./pages/Transcription";
import FinancePage from "./pages/Finance";

const queryClient = new QueryClient();

function ProtectedRoute({ children }: { children: React.ReactNode }) {
  const { isAuthenticated } = useAuth();
  return isAuthenticated ? <>{children}</> : <Navigate to="/login" replace />;
}

function Protected({ children }: { children: React.ReactNode }) {
  return <ProtectedRoute>{children}</ProtectedRoute>;
}

export default function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <AuthProvider>
        <BrowserRouter>
          <Routes>
            <Route path="/login" element={<LoginPage />} />
            <Route path="/" element={<Protected><Dashboard /></Protected>} />
            <Route path="/clients" element={<Protected><ClientsPage /></Protected>} />
            <Route path="/matters" element={<Protected><MattersPage /></Protected>} />
            <Route path="/emails" element={<Protected><EmailsPage /></Protected>} />
            <Route path="/tickets" element={<Protected><TicketsPage /></Protected>} />
            <Route path="/calendar" element={<Protected><CalendarPage /></Protected>} />
            <Route path="/transcriptions" element={<Protected><TranscriptionPage /></Protected>} />
            <Route path="/finance" element={<Protected><FinancePage /></Protected>} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </BrowserRouter>
      </AuthProvider>
    </QueryClientProvider>
  );
}
