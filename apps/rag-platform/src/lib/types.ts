export type ChatRole = "user" | "assistant";

export type CurrentUser = {
  id: string;
  email: string;
};

export type LoginResult = {
  user: CurrentUser;
  created: boolean;
};

export type ChatSession = {
  id: string;
  title: string;
  created_at: string;
  updated_at: string;
};

export type ChatMessage = {
  id: string;
  role: ChatRole;
  content: string;
  created_at: string;
  response?: FinalResponse | null;
};

export type Citation = {
  record_id: string;
  doi: string | null;
  source_text: string;
  score: number;
  fields_used: string[];
};

export type LiteratureEvidenceRecord = {
  record_id: string;
  score: number;
  title: string | null;
  material_formula: string | null;
  growth_method: string | null;
  temperature_program: string | null;
  atmosphere: string | null;
  precursors: string[];
  key_facts: string[];
  source_text: string;
  doi: string | null;
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
  outcome?: {
    status: "sufficient" | "empty" | "insufficient" | "invalid_request" | "unavailable";
    reason_codes: string[];
  } | null;
};

export type PredictionRoute = {
  rank: number;
  relative_rank_weight: number;
  method: "Flux" | "CVT";
  raw_reactants: Array<{ name: string }>;
  additives: Array<{ name: string }>;
  growth: Record<string, { range_c?: [number, number]; range_h?: [number, number] } | null>;
};

export type PredictionResult = {
  prediction_run_id: string;
  formula: string;
  formula_std: string;
  routes: PredictionRoute[];
  model: {
    model_id: string;
    model_version: string;
  };
  warnings: string[];
};

export type FinalResponse = {
  message_id: string;
  session_id: string;
  answer: string;
  citations: Citation[];
  evidence_records: LiteratureEvidenceRecord[];
  route: RouteDecision | null;
  retrieval: RetrievalTrace | null;
  evidence_kind: "literature_record" | "model_prediction" | null;
  prediction: PredictionResult | null;
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
  | { event: "retrieval_outcome"; data: Record<string, unknown> }
  | { event: "prediction_eligible"; data: Record<string, unknown> }
  | { event: "prediction_started"; data: Record<string, unknown> }
  | { event: "prediction_result"; data: PredictionResult }
  | { event: "prediction_warning"; data: { message: string } }
  | { event: "token"; data: { text: string } }
  | { event: "citation"; data: Citation }
  | { event: "memory_update"; data: Record<string, unknown> }
  | { event: "final"; data: FinalResponse }
  | { event: "error"; data: Record<string, unknown> }
  | { event: "run_finished"; data: Record<string, unknown> };
