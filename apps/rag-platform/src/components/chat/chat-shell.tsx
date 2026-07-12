"use client";

import { Send } from "lucide-react";
import { useMemo, useState } from "react";
import { EvidencePanel } from "../evidence/evidence-panel";
import { streamChat } from "../../lib/sse-client";
import type { ChatMessage, Citation, RetrievalTrace, RouteDecision } from "../../lib/types";

export function ChatShell() {
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      id: "welcome",
      role: "assistant",
      content: "请输入目标材料或单晶生长问题。当前后端已接入真实大模型，检索层仍使用 mock 数据，可先验证 LangGraph 数据流和证据展示。",
    },
  ]);
  const [input, setInput] = useState("Mn3GaN 的单晶生长温度程序怎么设置？");
  const [isRunning, setIsRunning] = useState(false);
  const [citations, setCitations] = useState<Citation[]>([]);
  const [route, setRoute] = useState<RouteDecision | null>(null);
  const [retrieval, setRetrieval] = useState<RetrievalTrace | null>(null);
  const [currentNode, setCurrentNode] = useState<string | null>(null);
  const [activity, setActivity] = useState<string[]>([]);
  const sessionId = useMemo(() => "demo-session", []);

  async function submit() {
    const content = input.trim();
    if (!content || isRunning) return;

    const assistantId = `assistant-${Date.now()}`;
    setInput("");
    setIsRunning(true);
    setCitations([]);
    setRoute(null);
    setRetrieval(null);
    setCurrentNode("准备处理请求");
    setActivity([]);
    setMessages((current) => [
      ...current,
      { id: `user-${Date.now()}`, role: "user", content },
      { id: assistantId, role: "assistant", content: "" },
    ]);

    await streamChat(
      {
        user_id: "demo-user",
        session_id: sessionId,
        message: content,
        options: {
          force_retrieve: false,
          top_k: 12,
          retrieval_mode: "hybrid",
          stream_trace: true,
        },
      },
      {
        onEvent(event) {
          if (event.event === "node_started") {
            setCurrentNode(event.data.label);
            setActivity((current) => [`开始：${event.data.label}`, ...current].slice(0, 8));
          }
          if (event.event === "node_finished") {
            const label =
              typeof event.data.label === "string" ? event.data.label : String(event.data.node ?? "节点");
            setActivity((current) => [`完成：${label}`, ...current].slice(0, 8));
          }
          if (event.event === "token") {
            setMessages((current) =>
              current.map((message) =>
                message.id === assistantId
                  ? { ...message, content: message.content + event.data.text }
                  : message,
              ),
            );
          }
          if (event.event === "citation") {
            setCitations((current) => {
              if (current.some((item) => item.record_id === event.data.record_id)) return current;
              return [...current, event.data];
            });
          }
          if (event.event === "route_decision") {
            setRoute(event.data);
          }
          if (event.event === "final") {
            setRetrieval(event.data.retrieval);
            setRoute(event.data.route);
            setCitations(event.data.citations);
          }
          if (event.event === "run_finished") {
            setIsRunning(false);
            setCurrentNode(null);
          }
        },
        onError(error) {
          setIsRunning(false);
          setMessages((current) =>
            current.map((message) =>
              message.id === assistantId
                ? { ...message, content: `请求失败：${error.message}` }
                : message,
            ),
          );
        },
      },
    );
  }

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <section className="panel-section">
          <div className="brand">AgentWeb RAG</div>
          <p className="muted">单晶生长方法检索增强对话</p>
        </section>
        <section className="panel-section">
          <h2 className="section-title">Sessions</h2>
          <div className="session-item">demo-session</div>
        </section>
      </aside>

      <main className="main">
        <header className="topbar">
          <div>
            <div className="brand">GrowthRAG Chat</div>
            <div className="muted">{currentNode ? `正在：${currentNode}` : "LangGraph v0.1"}</div>
          </div>
          <div className="muted">{isRunning ? "running" : "idle"}</div>
        </header>

        <section className="messages">
          {messages.map((message) => (
            <article className={`message ${message.role}`} key={message.id}>
              {message.content || " "}
            </article>
          ))}
        </section>

        <form
          className="composer"
          onSubmit={(event) => {
            event.preventDefault();
            void submit();
          }}
        >
          <textarea
            aria-label="message"
            value={input}
            onChange={(event) => setInput(event.target.value)}
            placeholder="输入材料、化学式或单晶生长问题"
          />
          <button className="send-button" type="submit" disabled={isRunning || !input.trim()}>
            <Send size={19} />
          </button>
        </form>
      </main>

      <EvidencePanel
        citations={citations}
        route={route}
        retrieval={retrieval}
        currentNode={currentNode}
        activity={activity}
      />
    </div>
  );
}
