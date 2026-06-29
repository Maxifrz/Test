import axios from "axios";

const api = axios.create({
  baseURL: "/api",
  withCredentials: true,
  headers: { "Content-Type": "application/json" },
});

// Attach access token from localStorage to every request
api.interceptors.request.use((config) => {
  const token = localStorage.getItem("access_token");
  if (token) config.headers.Authorization = `Bearer ${token}`;
  return config;
});

// On 401 with X-Session-Expired header, force re-login
api.interceptors.response.use(
  (r) => r,
  async (error) => {
    if (
      error.response?.status === 401 &&
      error.response?.headers["x-session-expired"]
    ) {
      localStorage.removeItem("access_token");
      window.location.href = "/login?reason=session_expired";
    }
    return Promise.reject(error);
  }
);

export default api;
