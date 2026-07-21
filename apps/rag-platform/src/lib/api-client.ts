import type {
  ChatMessage,
  ChatSession,
  ChatWorkspaceBootstrap,
  CurrentUser,
  LoginResult,
} from "./types";

const API_BASE_URL = process.env.NEXT_PUBLIC_RAG_API_BASE_URL ?? "http://localhost:8003";

export class ApiError extends Error {
  status: number;

  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

async function request<T>(path: string, init: RequestInit = {}): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    credentials: "include",
    ...init,
    headers: {
      ...(init.body ? { "Content-Type": "application/json" } : {}),
      ...init.headers,
    },
  });

  if (!response.ok) {
    const payload = await response.json().catch(() => null);
    const detail = payload && typeof payload.detail === "string" ? payload.detail : "请求失败。";
    throw new ApiError(response.status, detail);
  }
  if (response.status === 204) return undefined as T;
  return response.json() as Promise<T>;
}

export async function login(email: string, password: string): Promise<LoginResult> {
  return request<LoginResult>("/api/auth/login", {
    method: "POST",
    body: JSON.stringify({ email, password }),
  });
}

export async function logout(): Promise<void> {
  await request<void>("/api/auth/logout", { method: "POST" });
}

export async function getCurrentUser(): Promise<CurrentUser> {
  return request<CurrentUser>("/api/auth/me");
}

export async function bootstrapChatWorkspace(
  requestedSessionId?: string,
): Promise<ChatWorkspaceBootstrap> {
  return request<ChatWorkspaceBootstrap>("/api/rag/bootstrap", {
    method: "POST",
    body: JSON.stringify({ requested_session_id: requestedSessionId ?? null }),
  });
}

export async function listSessions(): Promise<ChatSession[]> {
  return request<ChatSession[]>("/api/rag/sessions");
}

export async function createSession(): Promise<ChatSession> {
  return request<ChatSession>("/api/rag/sessions", { method: "POST" });
}

export async function renameSession(sessionId: string, title: string): Promise<ChatSession> {
  return request<ChatSession>(`/api/rag/sessions/${sessionId}`, {
    method: "PATCH",
    body: JSON.stringify({ title }),
  });
}

export async function deleteSession(sessionId: string): Promise<void> {
  await request<void>(`/api/rag/sessions/${sessionId}`, { method: "DELETE" });
}

export async function listMessages(sessionId: string): Promise<ChatMessage[]> {
  return request<ChatMessage[]>(`/api/rag/sessions/${sessionId}/messages`);
}
