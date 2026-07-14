"use client";

import { ArrowRight, FlaskConical } from "lucide-react";
import { FormEvent, useState } from "react";
import { useRouter } from "next/navigation";
import { ApiError, login } from "../../lib/api-client";

export function LoginScreen() {
  const router = useRouter();
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [pending, setPending] = useState(false);
  const [error, setError] = useState<string | null>(null);

  async function submit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    if (pending) return;
    setPending(true);
    setError(null);
    try {
      await login(email, password);
      router.replace("/chat");
    } catch (cause) {
      setError(
        cause instanceof ApiError
          ? cause.message
          : "无法连接登录服务。请确认 RAG 服务已启动后重试。",
      );
    } finally {
      setPending(false);
    }
  }

  return (
    <main className="auth-page">
      <section className="auth-surface" aria-labelledby="auth-title">
        <div className="auth-brand">
          <FlaskConical size={25} strokeWidth={1.8} aria-hidden="true" />
          <span>Crystal Research</span>
        </div>
        <div className="auth-heading">
          <h1 id="auth-title">单晶生长研究助手</h1>
          <p>使用邮箱和密码进入你的研究会话。</p>
        </div>
        <form className="auth-form" onSubmit={submit}>
          <label>
            <span>邮箱</span>
            <input
              autoComplete="email"
              inputMode="email"
              name="email"
              onChange={(event) => setEmail(event.target.value)}
              placeholder="name@example.com"
              required
              type="email"
              value={email}
            />
          </label>
          <label>
            <span>密码</span>
            <input
              autoComplete="current-password"
              minLength={10}
              name="password"
              onChange={(event) => setPassword(event.target.value)}
              placeholder="至少 10 个字符"
              required
              type="password"
              value={password}
            />
          </label>
          {error ? <p className="form-error" role="alert">{error}</p> : null}
          <button className="auth-submit" disabled={pending} type="submit">
            <span>{pending ? "正在处理" : "继续"}</span>
            <ArrowRight size={18} aria-hidden="true" />
          </button>
        </form>
      </section>
    </main>
  );
}
