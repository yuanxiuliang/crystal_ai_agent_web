import type { RagStreamEvent } from "./types";

type StreamHandlers = {
  onEvent: (event: RagStreamEvent) => void;
  onError: (error: Error) => void;
};

const API_BASE_URL = process.env.NEXT_PUBLIC_RAG_API_BASE_URL ?? "http://localhost:8003";

export async function streamChat(
  body: {
    user_id: string;
    session_id: string;
    message: string;
    options: {
      force_retrieve: boolean;
      top_k: number;
      retrieval_mode: "dense" | "sparse" | "hybrid";
      stream_trace: boolean;
    };
  },
  handlers: StreamHandlers,
) {
  const response = await fetch(`${API_BASE_URL}/api/rag/chat/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });

  if (!response.ok || !response.body) {
    throw new Error(`RAG API request failed: ${response.status}`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  try {
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const chunks = buffer.split("\n\n");
      buffer = chunks.pop() ?? "";
      for (const chunk of chunks) {
        const event = parseSseChunk(chunk);
        if (event) handlers.onEvent(event);
      }
    }
  } catch (error) {
    handlers.onError(error instanceof Error ? error : new Error(String(error)));
  }
}

function parseSseChunk(chunk: string): RagStreamEvent | null {
  const lines = chunk.split("\n");
  const eventLine = lines.find((line) => line.startsWith("event:"));
  const dataLine = lines.find((line) => line.startsWith("data:"));
  if (!eventLine || !dataLine) return null;
  const event = eventLine.replace("event:", "").trim();
  const data = JSON.parse(dataLine.replace("data:", "").trim());
  return { event, data } as RagStreamEvent;
}

