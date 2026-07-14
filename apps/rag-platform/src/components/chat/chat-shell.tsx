"use client";

import {
  BookOpen,
  Check,
  FlaskConical,
  LogOut,
  Menu,
  MessageSquarePlus,
  PanelRight,
  Pencil,
  SendHorizontal,
  Square,
  Trash2,
  X,
} from "lucide-react";
import { KeyboardEvent, useEffect, useMemo, useRef, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import {
  ApiError,
  createSession,
  deleteSession,
  getCurrentUser,
  listMessages,
  listSessions,
  logout,
  renameSession,
} from "../../lib/api-client";
import { streamChat } from "../../lib/sse-client";
import type {
  ChatMessage,
  ChatSession,
  Citation,
  CurrentUser,
  FinalResponse,
  PredictionRoute,
} from "../../lib/types";
import { MarkdownAnswer } from "./markdown-answer";
import { EvidencePanel } from "../evidence/evidence-panel";

const QUICK_PROMPTS = [
  "ZnIn2S4 的 CVT 单晶生长温度是多少？",
  "Mn3GaN 怎么做？",
  "比较助熔剂法和 CVT 的适用条件。",
];

type WorkflowActivity = {
  phase: string;
  message: string;
};

const INITIAL_ACTIVITY: WorkflowActivity = {
  phase: "准备处理",
  message: "正在理解你的研究问题",
};
const MAX_VISIBLE_ACTIVITY_STEPS = 4;

// These are intentionally user-facing workflow stages, not raw LangGraph node labels.
const NODE_ACTIVITIES: Partial<Record<string, WorkflowActivity>> = {
  prepare_turn: INITIAL_ACTIVITY,
  load_context: { phase: "研究上下文", message: "正在读取当前研究上下文" },
  load_long_memory: { phase: "研究上下文", message: "正在读取当前研究上下文" },
  analyze_and_route: { phase: "理解问题", message: "正在识别目标材料与研究需求" },
  ask_clarification: { phase: "补充信息", message: "正在整理需要补充的条件" },
  answer_direct: { phase: "整理回答", message: "正在根据当前会话整理回答" },
  plan_retrieval: { phase: "检索真实记录", message: "正在准备检索真实实验记录" },
  retrieve_records: { phase: "检索真实记录", message: "正在检索真实实验记录" },
  assess_retrieval_sufficiency: { phase: "核验证据", message: "正在核验材料匹配与证据完整性" },
  build_evidence_pack: { phase: "整理证据", message: "正在整理文献证据与实验参数" },
  assess_prediction_eligibility: { phase: "判断路径", message: "正在判断可用证据与回答路径" },
  run_prediction: { phase: "候选路线预测", message: "正在生成候选单晶生长路径" },
  answer_with_evidence: { phase: "撰写回答", message: "正在撰写基于真实记录的回答" },
  answer_from_prediction: { phase: "撰写回答", message: "正在整理未验证候选方案" },
  answer_with_limits: { phase: "说明限制", message: "正在整理当前可确认的信息与限制" },
};

function localNow(): string {
  return new Date().toISOString();
}
function routeSessionId(value: string | string[] | undefined): string | null {
  return typeof value === "string" && value.length > 0 ? value : null;
}

function names(items: Array<{ name: string }>): string {
  return items.length > 0 ? items.map((item) => item.name).join("、") : "无";
}

function range(value: unknown, unit: "C" | "h"): string {
  if (!value || typeof value !== "object") return "未提供";
  const field = unit === "C" ? "range_c" : "range_h";
  const bounds = (value as Record<string, unknown>)[field];
  return Array.isArray(bounds) && bounds.length === 2 ? `${bounds[0]}-${bounds[1]} ${unit}` : "未提供";
}

function routeTemperature(route: PredictionRoute): string {
  const growth = route.growth;
  if (route.method === "CVT") {
    return `${range(growth.T_src, "C")} / ${range(growth.T_crys, "C")}`;
  }
  return `${range(growth.T_s, "C")} -> ${range(growth.T_e, "C")}`;
}

function plannedFormula(data: Record<string, unknown>): string | null {
  const filters = data.filters;
  if (!filters || typeof filters !== "object") return null;
  const formula = (filters as Record<string, unknown>).material_formula;
  return typeof formula === "string" && formula.trim() ? formula.trim() : null;
}

type AssistantMessageProps = {
  message: ChatMessage;
  onOpenEvidence: () => void;
  isPending: boolean;
  activity: WorkflowActivity | null;
  activityTrail: WorkflowActivity[];
};

function AssistantMessage({ message, onOpenEvidence, isPending, activity, activityTrail }: AssistantMessageProps) {
  const response = message.response;
  if (response?.evidence_kind === "model_prediction" && response.prediction) {
    const prediction = response.prediction;
    const fallbackReasons = response.retrieval?.outcome?.reason_codes ?? [];
    return (
      <div className="assistant-content prediction-answer">
        <div className="message-kicker model-kicker">可尝试方案 · 模型预测 · 未验证</div>
        <p>当前没有找到足以支持该问题的真实文献或实验记录。</p>
        <p>以下路线可作为实验探索起点，尚未由当前真实记录验证，不是文献事实。</p>
        <div className="model-meta">
          {prediction.model.model_id}@{prediction.model.model_version} · {prediction.formula_std}
        </div>
        <div className="route-table-wrap">
          <table className="route-table">
            <thead>
              <tr>
                <th>候选</th>
                <th>方法</th>
                <th>原料与添加剂</th>
                <th>温度程序</th>
                <th>时长</th>
              </tr>
            </thead>
            <tbody>
              {prediction.routes.map((route) => (
                <tr key={route.rank}>
                  <td>{route.rank}</td>
                  <td>{route.method}</td>
                  <td>
                    <div>{names(route.raw_reactants)}</div>
                    {route.additives.length > 0 ? <small>添加剂：{names(route.additives)}</small> : null}
                  </td>
                  <td>{routeTemperature(route)}</td>
                  <td>{range(route.growth.dur, "h")}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        {prediction.warnings.length > 0 ? (
          <ul className="prediction-warnings">
            {prediction.warnings.map((warning) => <li key={warning}>{warning}</li>)}
          </ul>
        ) : null}
        {fallbackReasons.length > 0 ? <div className="model-reason">检索回退原因：{fallbackReasons.join("、")}</div> : null}
      </div>
    );
  }

  const citations: Citation[] = response?.citations ?? [];
  return (
    <div className="assistant-content">
      {response?.evidence_kind === "literature_record" ? <div className="message-kicker">真实记录回答</div> : null}
      {message.content ? <MarkdownAnswer content={message.content} /> : null}
      {isPending && !message.content ? (
        <div className="response-progress" role="status">
          {activityTrail.slice(0, -1).map((step, index) => (
            <div className="response-step" key={`${step.phase}:${step.message}:${index}`}>
              <Check className="response-step-check" size={14} aria-hidden="true" />
              <span className="activity-phase">{step.phase}</span>
              <span>{step.message}</span>
            </div>
          ))}
          <div className="response-progress-current">
            <span className="activity-pulse" aria-hidden="true" />
            <span className="activity-phase">{activity?.phase ?? INITIAL_ACTIVITY.phase}</span>
            <span>{activity?.message ?? INITIAL_ACTIVITY.message}</span>
          </div>
        </div>
      ) : null}
      {citations.length > 0 ? (
        <button className="citation-summary" onClick={onOpenEvidence} type="button">
          <BookOpen size={15} aria-hidden="true" />
          <span>{citations.length} 条文献证据</span>
        </button>
      ) : null}
    </div>
  );
}

export function ChatShell() {
  const params = useParams<{ sessionId?: string | string[] }>();
  const router = useRouter();
  const requestedSessionId = routeSessionId(params.sessionId);
  const abortRef = useRef<AbortController | null>(null);
  const [user, setUser] = useState<CurrentUser | null>(null);
  const [sessions, setSessions] = useState<ChatSession[]>([]);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [initializing, setInitializing] = useState(true);
  const [isRunning, setIsRunning] = useState(false);
  const [pendingAssistantId, setPendingAssistantId] = useState<string | null>(null);
  const [activity, setActivity] = useState<WorkflowActivity | null>(null);
  const [activityTrail, setActivityTrail] = useState<WorkflowActivity[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [evidenceOpen, setEvidenceOpen] = useState(false);
  const [renamingId, setRenamingId] = useState<string | null>(null);
  const [renameValue, setRenameValue] = useState("");

  const evidenceResponse = useMemo<FinalResponse | null>(() => {
    for (let index = messages.length - 1; index >= 0; index -= 1) {
      const response = messages[index].response;
      if (response) return response;
    }
    return null;
  }, [messages]);
  const activeSession = sessions.find((session) => session.id === requestedSessionId) ?? null;

  async function refreshSessions() {
    const next = await listSessions();
    setSessions(next);
    return next;
  }

  useEffect(() => {
    let cancelled = false;
    async function load() {
      setInitializing(true);
      setError(null);
      try {
        const currentUser = await getCurrentUser();
        if (cancelled) return;
        setUser(currentUser);
        const availableSessions = await refreshSessions();
        if (cancelled) return;
        if (!requestedSessionId) {
          const target = availableSessions[0] ?? (await createSession());
          if (!availableSessions.length) setSessions([target]);
          router.replace(`/chat/${target.id}`);
          return;
        }
        if (!availableSessions.some((session) => session.id === requestedSessionId)) {
          router.replace("/chat");
          return;
        }
        const history = await listMessages(requestedSessionId);
        if (!cancelled) setMessages(history);
      } catch (cause) {
        if (cancelled) return;
        if (cause instanceof ApiError && cause.status === 401) {
          router.replace("/login");
          return;
        }
        setError(cause instanceof Error ? cause.message : "无法加载研究会话。");
      } finally {
        if (!cancelled) setInitializing(false);
      }
    }
    void load();
    return () => {
      cancelled = true;
    };
  }, [requestedSessionId, router]);

  async function startNewSession() {
    try {
      const session = await createSession();
      setSessions((current) => [session, ...current]);
      setSidebarOpen(false);
      router.push(`/chat/${session.id}`);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "无法创建会话。");
    }
  }

  async function removeSession(sessionId: string) {
    try {
      await deleteSession(sessionId);
      const next = sessions.filter((session) => session.id !== sessionId);
      setSessions(next);
      if (sessionId === requestedSessionId) {
        if (next[0]) router.push(`/chat/${next[0].id}`);
        else await startNewSession();
      }
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "无法删除会话。");
    }
  }

  async function commitRename(sessionId: string) {
    const title = renameValue.trim();
    setRenamingId(null);
    if (!title) return;
    try {
      const updated = await renameSession(sessionId, title);
      setSessions((current) => current.map((session) => (session.id === updated.id ? updated : session)));
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "无法修改会话标题。");
    }
  }

  async function signOut() {
    await logout().catch(() => undefined);
    router.replace("/login");
  }

  function recordActivity(next: WorkflowActivity) {
    setActivity(next);
    setActivityTrail((current) => {
      const previous = current[current.length - 1];
      if (previous?.phase === next.phase && previous.message === next.message) return current;
      return [...current, next].slice(-MAX_VISIBLE_ACTIVITY_STEPS);
    });
  }

  async function submit() {
    const content = input.trim();
    if (!content || !requestedSessionId || isRunning) return;

    const assistantId = `assistant-${Date.now()}`;
    const controller = new AbortController();
    abortRef.current = controller;
    setInput("");
    setIsRunning(true);
    setPendingAssistantId(assistantId);
    setActivity(INITIAL_ACTIVITY);
    setActivityTrail([INITIAL_ACTIVITY]);
    setError(null);
    setMessages((current) => [
      ...current,
      { id: `user-${Date.now()}`, role: "user", content, created_at: localNow() },
      { id: assistantId, role: "assistant", content: "", created_at: localNow() },
    ]);

    try {
      await streamChat(
        {
          session_id: requestedSessionId,
          message: content,
          options: { force_retrieve: false, top_k: 12, retrieval_mode: "hybrid", stream_trace: false },
        },
        {
          onEvent(event) {
            if (event.event === "node_started") {
              const nextActivity = NODE_ACTIVITIES[event.data.node];
              if (nextActivity) recordActivity(nextActivity);
            }
            if (event.event === "retrieval_plan") {
              const formula = plannedFormula(event.data);
              recordActivity({
                phase: "检索真实记录",
                message: formula ? `正在检索 ${formula} 的真实实验记录` : "正在检索匹配的真实实验记录",
              });
            }
            if (event.event === "retrieval_outcome") {
              const status = String(event.data.status ?? "");
              if (status === "sufficient") {
                recordActivity({
                  phase: "整理证据",
                  message: "已找到匹配的真实记录，正在提取生长条件",
                });
              } else if (status === "empty" || status === "insufficient") {
                recordActivity({
                  phase: "判断路径",
                  message: "未找到足够的直接真实记录，正在判断后续路径",
                });
              } else if (status === "unavailable") {
                recordActivity({
                  phase: "检索受限",
                  message: "真实记录库暂时不可用，正在整理可确认的信息",
                });
              } else if (status === "invalid_request") {
                recordActivity({
                  phase: "补充信息",
                  message: "目标材料信息尚不完整，正在准备澄清问题",
                });
              }
            }
            if (event.event === "prediction_eligible") {
              if (event.data.eligible) {
                const formula = typeof event.data.formula === "string" ? event.data.formula : "目标材料";
                recordActivity({
                  phase: "候选路线预测",
                  message: `未找到直接真实记录，正在启动 ${formula} 候选路线预测`,
                });
              }
            }
            if (event.event === "prediction_started") {
              recordActivity({ phase: "候选路线预测", message: "正在生成候选单晶生长路径" });
            }
            if (event.event === "prediction_result") {
              recordActivity({
                phase: "核对预测参数",
                message: `已生成 ${event.data.routes.length} 条候选路线，正在核对参数与适用边界`,
              });
            }
            if (event.event === "token") {
              recordActivity({ phase: "撰写回答", message: "正在生成回答" });
              setMessages((current) => current.map((message) => (
                message.id === assistantId ? { ...message, content: message.content + event.data.text } : message
              )));
            }
            if (event.event === "final") {
              setMessages((current) => current.map((message) => (
                message.id === assistantId
                  ? { ...message, content: event.data.answer, response: event.data }
                  : message
              )));
            }
          },
          onError(cause) {
            const message = cause.name === "AbortError" ? "已停止生成。" : `请求失败：${cause.message}`;
            if (cause.name !== "AbortError") setError(cause.message);
            setMessages((current) => current.map((item) => (
              item.id === assistantId && !item.content ? { ...item, content: message } : item
            )));
          },
        },
        controller.signal,
      );
    } catch (cause) {
      const streamError = cause instanceof Error ? cause : new Error(String(cause));
      const message = streamError.name === "AbortError" ? "已停止生成。" : `请求失败：${streamError.message}`;
      if (streamError.name !== "AbortError") setError(streamError.message);
      setMessages((current) => current.map((item) => (
        item.id === assistantId && !item.content ? { ...item, content: message } : item
      )));
    } finally {
      abortRef.current = null;
      setIsRunning(false);
      setPendingAssistantId((current) => (current === assistantId ? null : current));
      setActivity(null);
      setActivityTrail([]);
      void refreshSessions().catch(() => undefined);
    }
  }

  function onComposerKeyDown(event: KeyboardEvent<HTMLTextAreaElement>) {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      void submit();
    }
  }

  return (
    <div className={`workbench ${evidenceOpen ? "evidence-open" : ""}`}>
      <aside className={`conversation-sidebar ${sidebarOpen ? "is-open" : ""}`}>
        <div className="sidebar-top">
          <div className="product-mark"><FlaskConical size={19} aria-hidden="true" /><span>Crystal Research</span></div>
          <button className="sidebar-close icon-control" onClick={() => setSidebarOpen(false)} title="关闭会话栏" type="button"><X size={18} /></button>
        </div>
        <button className="new-session" onClick={() => void startNewSession()} type="button">
          <MessageSquarePlus size={18} aria-hidden="true" />
          <span>新建对话</span>
        </button>
        <nav aria-label="会话列表" className="session-list">
          <div className="session-label">会话</div>
          {sessions.map((session) => (
            <div className={`session-row ${session.id === requestedSessionId ? "selected" : ""}`} key={session.id}>
              {renamingId === session.id ? (
                <input
                  autoFocus
                  className="session-title-input"
                  onBlur={() => void commitRename(session.id)}
                  onChange={(event) => setRenameValue(event.target.value)}
                  onKeyDown={(event) => {
                    if (event.key === "Enter") void commitRename(session.id);
                    if (event.key === "Escape") setRenamingId(null);
                  }}
                  value={renameValue}
                />
              ) : (
                <button className="session-select" onClick={() => { setSidebarOpen(false); router.push(`/chat/${session.id}`); }} type="button">
                  {session.title}
                </button>
              )}
              <div className="session-actions">
                <button className="icon-control" onClick={() => { setRenamingId(session.id); setRenameValue(session.title); }} title="重命名会话" type="button"><Pencil size={14} /></button>
                <button className="icon-control danger-control" onClick={() => void removeSession(session.id)} title="删除会话" type="button"><Trash2 size={14} /></button>
              </div>
            </div>
          ))}
        </nav>
        <div className="account-menu">
          <div className="account-email">{user?.email ?? ""}</div>
          <button className="logout-button" onClick={() => void signOut()} type="button"><LogOut size={16} aria-hidden="true" /><span>退出登录</span></button>
        </div>
      </aside>
      {sidebarOpen ? <button aria-label="关闭会话栏" className="sidebar-backdrop" onClick={() => setSidebarOpen(false)} type="button" /> : null}

      <main className="chat-main">
        <header className="chat-header">
          <div className="header-leading">
            <button className="icon-control mobile-menu" onClick={() => setSidebarOpen(true)} title="打开会话栏" type="button"><Menu size={19} /></button>
            <div>
              <div className="chat-title">{activeSession?.title ?? "单晶生长研究助手"}</div>
              {activity ? (
                <div className="chat-activity" aria-label={`当前进度：${activity.message}`}>
                  <span className="activity-pulse" aria-hidden="true" />
                  <span className="activity-phase">{activity.phase}</span>
                  <span>{activity.message}</span>
                </div>
              ) : null}
            </div>
          </div>
          <button className="icon-control evidence-toggle" onClick={() => setEvidenceOpen((value) => !value)} title="文献证据" type="button"><PanelRight size={19} /></button>
        </header>

        <section className="messages" aria-live="polite">
          {initializing ? <div className="empty-state">正在加载会话...</div> : null}
          {!initializing && messages.length === 0 ? (
            <div className="empty-state welcome-state">
              <h1>单晶生长研究助手</h1>
              <div className="prompt-row">
                {QUICK_PROMPTS.map((prompt) => <button key={prompt} onClick={() => setInput(prompt)} type="button">{prompt}</button>)}
              </div>
            </div>
          ) : null}
          {messages.map((message) => (
            <article className={`chat-message ${message.role}`} key={message.id}>
              {message.role === "assistant" ? (
                <AssistantMessage
                  activity={activity}
                  activityTrail={activityTrail}
                  isPending={isRunning && message.id === pendingAssistantId}
                  message={message}
                  onOpenEvidence={() => setEvidenceOpen(true)}
                />
              ) : <div className="message-text">{message.content}</div>}
            </article>
          ))}
        </section>

        {error ? <div className="chat-error" role="alert">{error}</div> : null}
        <form className="composer" onSubmit={(event) => { event.preventDefault(); void submit(); }}>
          <textarea
            aria-label="研究问题"
            disabled={initializing || !requestedSessionId}
            onChange={(event) => setInput(event.target.value)}
            onKeyDown={onComposerKeyDown}
            placeholder="输入材料、化学式或单晶生长问题"
            value={input}
          />
          {isRunning ? (
            <button className="stop-button" onClick={() => abortRef.current?.abort()} title="停止生成" type="button"><Square size={15} fill="currentColor" /></button>
          ) : (
            <button className="send-button" disabled={!input.trim() || initializing || !requestedSessionId} title="发送" type="submit"><SendHorizontal size={19} /></button>
          )}
        </form>
      </main>

      <EvidencePanel isOpen={evidenceOpen} onClose={() => setEvidenceOpen(false)} response={evidenceResponse} />
    </div>
  );
}
