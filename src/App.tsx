import { useRef, useState } from "react";
import DocumentUpload from "./DocumentUpload";

interface RagStep {
  node: string;
  detail: string;
}

interface Source {
  text: string;
  source: string | null;
}

interface Message {
  role: "user" | "assistant";
  content: string;
  steps?: RagStep[];
  sources?: Source[];
  error?: boolean;
}

export default function App() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const listRef = useRef<HTMLDivElement>(null);

  async function send() {
    const question = input.trim();
    if (!question || loading) return;

    setMessages((m) => [...m, { role: "user", content: question }]);
    setInput("");
    setLoading(true);

    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: question })
      });
      const data = await res.json();

      if (!res.ok) throw new Error(data.error || "Request failed");

      setMessages((m) => [
        ...m,
        { role: "assistant", content: data.answer, steps: data.steps, sources: data.sources }
      ]);
    } catch (err) {
      setMessages((m) => [
        ...m,
        { role: "assistant", content: err instanceof Error ? err.message : "Something went wrong", error: true }
      ]);
    } finally {
      setLoading(false);
      requestAnimationFrame(() => listRef.current?.scrollTo(0, listRef.current.scrollHeight));
    }
  }

  return (
    <div className="app">
      <header className="header">
        <span className="dot" />
        <h1>Agentic self-correcting RAG</h1>
      </header>

      <DocumentUpload />

      <div className="messages" ref={listRef}>
        {messages.length === 0 && (
          <div className="empty">
            Ask a question. The agent will decide whether to retrieve, grade what it finds,
            rewrite the query if retrieval is weak, and check its own answer before replying.
          </div>
        )}
        {messages.map((m, i) => (
          <div key={i} className={`msg ${m.role}`}>
            <div className="bubble">{m.content}</div>
            {m.steps && m.steps.length > 0 && <StepTrace steps={m.steps} />}
            {m.sources && m.sources.length > 0 && (
              <details className="sources">
                <summary>{m.sources.length} source{m.sources.length > 1 ? "s" : ""}</summary>
                {m.sources.map((s, j) => (
                  <div key={j} className="source">
                    <span className="source-name">{s.source ?? "unknown"}</span>
                    <p>{s.text}…</p>
                  </div>
                ))}
              </details>
            )}
          </div>
        ))}
        {loading && <div className="msg assistant"><div className="bubble typing">thinking…</div></div>}
      </div>

      <div className="composer">
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && send()}
          placeholder="Ask something…"
          disabled={loading}
        />
        <button onClick={send} disabled={loading || !input.trim()}>
          Send
        </button>
      </div>
    </div>
  );
}

function StepTrace({ steps }: { steps: RagStep[] }) {
  return (
    <details className="trace">
      <summary>{steps.length} pipeline steps</summary>
      <ol>
        {steps.map((s, i) => (
          <li key={i}>
            <span className="node">{s.node}</span>
            <span className="detail">{s.detail}</span>
          </li>
        ))}
      </ol>
    </details>
  );
}
