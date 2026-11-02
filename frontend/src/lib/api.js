/**
 * API client for the FastAPI backend.
 *
 * Tokens live in `localStorage` (this is a client-rendered SPA shell) and every
 * request attaches the bearer token. `request()` also transparently refreshes an
 * expired access token once before surfacing a 401 to the UI.
 */

export const API_BASE_URL = (
  process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8000"
).replace(/\/$/, "");

export const API_PREFIX = "/api/v1";

const ACCESS_TOKEN_KEY = "rag.access_token";
const REFRESH_TOKEN_KEY = "rag.refresh_token";
const USER_KEY = "rag.user";

/** Error carrying the backend's uniform error envelope. */
export class ApiError extends Error {
  constructor(message, { status, code, details } = {}) {
    super(message);
    this.name = "ApiError";
    this.status = status;
    this.code = code;
    this.details = details;
  }
}

// --------------------------------------------------------------------------- //
// Token storage
// --------------------------------------------------------------------------- //
export const tokenStore = {
  getAccessToken() {
    if (typeof window === "undefined") return null;
    return window.localStorage.getItem(ACCESS_TOKEN_KEY);
  },
  getRefreshToken() {
    if (typeof window === "undefined") return null;
    return window.localStorage.getItem(REFRESH_TOKEN_KEY);
  },
  getUser() {
    if (typeof window === "undefined") return null;
    const raw = window.localStorage.getItem(USER_KEY);
    if (!raw) return null;
    try {
      return JSON.parse(raw);
    } catch {
      return null;
    }
  },
  set({ access_token, refresh_token, user }) {
    if (typeof window === "undefined") return;
    if (access_token) window.localStorage.setItem(ACCESS_TOKEN_KEY, access_token);
    if (refresh_token) window.localStorage.setItem(REFRESH_TOKEN_KEY, refresh_token);
    if (user) window.localStorage.setItem(USER_KEY, JSON.stringify(user));
  },
  setUser(user) {
    if (typeof window === "undefined" || !user) return;
    window.localStorage.setItem(USER_KEY, JSON.stringify(user));
  },
  clear() {
    if (typeof window === "undefined") return;
    window.localStorage.removeItem(ACCESS_TOKEN_KEY);
    window.localStorage.removeItem(REFRESH_TOKEN_KEY);
    window.localStorage.removeItem(USER_KEY);
  },
};

// --------------------------------------------------------------------------- //
// Low-level request helper
// --------------------------------------------------------------------------- //
async function parseError(response) {
  let payload = null;
  try {
    payload = await response.json();
  } catch {
    payload = null;
  }
  const message =
    payload?.message ||
    payload?.detail ||
    `Request failed with status ${response.status}`;
  return new ApiError(message, {
    status: response.status,
    code: payload?.code,
    details: payload?.details,
  });
}

let refreshPromise = null;

async function refreshAccessToken() {
  const refreshToken = tokenStore.getRefreshToken();
  if (!refreshToken) return null;

  // De-duplicate concurrent refreshes triggered by parallel requests.
  if (!refreshPromise) {
    refreshPromise = fetch(`${API_BASE_URL}${API_PREFIX}/auth/refresh`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ refresh_token: refreshToken }),
    })
      .then(async (response) => {
        if (!response.ok) {
          tokenStore.clear();
          return null;
        }
        const data = await response.json();
        tokenStore.set(data);
        return data.access_token;
      })
      .catch(() => {
        tokenStore.clear();
        return null;
      })
      .finally(() => {
        refreshPromise = null;
      });
  }
  return refreshPromise;
}

export async function request(path, options = {}) {
  const { method = "GET", body, headers = {}, auth = true, retry = true } = options;

  const finalHeaders = { ...headers };
  let payload = body;

  if (body && !(body instanceof FormData)) {
    finalHeaders["Content-Type"] = "application/json";
    payload = JSON.stringify(body);
  }

  if (auth) {
    const token = tokenStore.getAccessToken();
    if (token) finalHeaders.Authorization = `Bearer ${token}`;
  }

  const response = await fetch(`${API_BASE_URL}${API_PREFIX}${path}`, {
    method,
    headers: finalHeaders,
    body: payload,
  });

  // One transparent refresh attempt on an expired access token.
  if (response.status === 401 && auth && retry) {
    const refreshed = await refreshAccessToken();
    if (refreshed) {
      return request(path, { ...options, retry: false });
    }
  }

  if (!response.ok) throw await parseError(response);
  if (response.status === 204) return null;

  const contentType = response.headers.get("content-type") || "";
  if (!contentType.includes("application/json")) return response.text();
  return response.json();
}

export const api = {
  request,

  // ---------------------------------------------------------------- auth ---
  register: (payload) =>
    request("/auth/register", { method: "POST", body: payload, auth: false }),
  login: (payload) =>
    request("/auth/login", { method: "POST", body: payload, auth: false }),
  logout: () => request("/auth/logout", { method: "POST" }).catch(() => null),
  me: () => request("/auth/me"),
  updateProfile: (payload) => request("/users/me", { method: "PATCH", body: payload }),
  deactivateAccount: () => request("/users/me", { method: "DELETE" }),
  stats: () => request("/users/me/stats"),

  // ----------------------------------------------------------- documents ---
  listDocuments: (params = {}) => {
    const query = new URLSearchParams(
      Object.entries(params).filter(
        ([, value]) => value !== undefined && value !== null && value !== "",
      ),
    );
    return request(`/documents?${query.toString()}`);
  },
  documentStats: () => request("/documents/stats"),
  getDocument: (id) => request(`/documents/${id}`),
  documentStatus: (id) => request(`/documents/${id}/status`),
  documentChunks: (id, params = {}) =>
    request(`/documents/${id}/chunks?${new URLSearchParams(params).toString()}`),
  uploadDocument: (file, { collection, tags, processNow = true } = {}) => {
    const form = new FormData();
    form.append("file", file);
    if (collection) form.append("collection", collection);
    if (tags) form.append("tags", tags);
    form.append("process_now", String(processNow));
    return request("/documents/upload", { method: "POST", body: form });
  },
  updateDocument: (id, payload) =>
    request(`/documents/${id}`, { method: "PATCH", body: payload }),
  reindexDocument: (id) => request(`/documents/${id}/reindex`, { method: "POST" }),
  deleteDocument: (id) => request(`/documents/${id}`, { method: "DELETE" }),

  // ---------------------------------------------------------------- chat ---
  ask: (payload) => request("/chat/completions", { method: "POST", body: payload }),
  listConversations: (params = {}) =>
    request(`/chat/conversations?${new URLSearchParams(params).toString()}`),
  createConversation: (payload = {}) =>
    request("/chat/conversations", { method: "POST", body: payload }),
  getConversation: (id) => request(`/chat/conversations/${id}`),
  renameConversation: (id, payload) =>
    request(`/chat/conversations/${id}`, { method: "PATCH", body: payload }),
  deleteConversation: (id) =>
    request(`/chat/conversations/${id}`, { method: "DELETE" }),

  // --------------------------------------------------------------- tasks ---
  listTasks: (params = {}) =>
    request(`/tasks?${new URLSearchParams(params).toString()}`),
  getTask: (id) => request(`/tasks/${id}`),

  // -------------------------------------------------------------- health ---
  health: () => request("/health", { auth: false }),
  readiness: () => request("/health/ready", { auth: false }),
};
