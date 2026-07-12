export type ChatRole = "user" | "assistant";

export type ChatMessage = {
  id: string;
  role: ChatRole;
  content: string;
};

export type Citation = {
  record_id: string;
  doi: string | null;
  source_text: string;
  score: number;
  fields_used: string[];
};

export type RouteDecision = {
  intent: string;
  should_retrieve: boolean;
  reason: string;
};

export type RetrievalTrace = {
  query: string;
  filters: Record<string, unknown>;
  top_k: number;
  result_count: number;
  sufficient: boolean | null;
};

export type FinalResponse = {
  message_id: string;
  session_id: string;
  answer: string;
  citations: Citation[];
  route: RouteDecision | null;
  retrieval: RetrievalTrace | null;
  memory: {
    short_term_updated: boolean;
    long_term_written: boolean;
  };
  errors: Array<{
    node: string;
    code: string;
    message: string;
    recoverable: boolean;
  }>;
};

export type RagStreamEvent =
  | { event: "run_started"; data: Record<string, unknown> }
  | { event: "node_started"; data: { node: string; label: string } }
  | { event: "node_finished"; data: Record<string, unknown> }
  | { event: "route_decision"; data: RouteDecision }
  | { event: "retrieval_plan"; data: Record<string, unknown> }
  | { event: "retrieval_result"; data: Record<string, unknown> }
  | { event: "evidence_grade"; data: Record<string, unknown> }
  | { event: "token"; data: { text: string } }
  | { event: "citation"; data: Citation }
  | { event: "memory_update"; data: Record<string, unknown> }
  | { event: "final"; data: FinalResponse }
  | { event: "error"; data: Record<string, unknown> }
  | { event: "run_finished"; data: Record<string, unknown> };
