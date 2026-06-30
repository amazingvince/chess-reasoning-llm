export type Side = "white" | "black";
export type LegalFilter = "all" | "legal" | "illegal" | "unknown";

export interface ChatMessage {
  role: string;
  content: string;
  name?: string;
  metadata?: Record<string, unknown>;
}

export interface PromptArtifact {
  prompt_id: string;
  messages: ChatMessage[];
  fen?: string | null;
  task_type?: string | null;
  metadata?: Record<string, unknown>;
}

export interface ParsedAnswer {
  raw_text?: string;
  move_uci?: string | null;
  format_type?: string;
  parse_error?: string | null;
  metadata?: Record<string, unknown>;
}

export interface RolloutArtifact {
  rollout_id: string;
  prompt_id: string;
  model_id: string;
  raw_output: string;
  parsed_answer: ParsedAnswer;
  metadata?: Record<string, unknown>;
}

export interface JudgmentArtifact {
  judgment_id: string;
  rollout_id: string;
  legal?: boolean | null;
  regret_cp?: number | null;
  failure_bucket?: string | null;
  teacher_move_uci?: string | null;
  feedback?: string | null;
  metadata?: Record<string, unknown>;
}

export interface RolloutReviewItem {
  prompt: PromptArtifact | null;
  rollout: RolloutArtifact;
  judgment: JudgmentArtifact | null;
}

export interface OpeningBook {
  book_id: string;
  name: string;
  path: string;
  curated: boolean;
}

export interface BackendHealth {
  status: string;
  model: string;
  llm_mode: string;
}

export interface ToolStatus {
  stockfish: {
    enabled: boolean;
    available: boolean;
    path: string | null;
    name: string | null;
    depth: number;
    threads: number;
    hash_mb: number;
    error: string | null;
  };
  capabilities: {
    legal_moves: boolean;
    opening_book_moves: boolean;
    stockfish_analysis: boolean;
  };
  batch_limit: number;
}

export interface ToolMove {
  uci: string;
  san: string;
  capture: boolean;
  check: boolean;
  promotion?: string | null;
}

export interface ToolBookMove extends ToolMove {
  weight: number;
}

export interface ToolAnalysis {
  fen: string;
  turn: Side;
  legal_move_count: number;
  legal_moves: ToolMove[];
  book_moves: ToolBookMove[];
  stockfish: {
    available: boolean;
    best_move?: string | null;
    cp?: number | null;
    mate?: number | null;
    pv_line?: string;
    depth: number;
    error?: string | null;
  } | null;
}

export interface ScoreRolloutsResponse {
  items: RolloutReviewItem[];
  scored_count: number;
  skipped_count: number;
  error_count: number;
  errors: Array<{ rollout_id: string; error: string }>;
  tool_status: ToolStatus;
}

export interface GameMove {
  source: "human" | "llm" | "book" | string;
  side?: Side;
  move_uci: string;
  fen_before?: string;
  fen_after?: string;
  raw_output?: string | null;
  weight?: number | null;
  rollout_id?: string;
  judgment_id?: string;
  rollout?: RolloutArtifact | null;
  judgment?: JudgmentArtifact | null;
}

export type RecoveryAction = "retry" | "retry_with_legal_moves" | "teacher" | "legal_stub";

export interface PendingRecovery {
  rollout_id: string;
  judgment_id: string;
  reason: string;
  created_at: string;
}

export interface GameState {
  game_id: string;
  human_side: Side;
  turn: Side;
  fen: string;
  created_at: string;
  moves: GameMove[];
  book_line: GameMove[];
  artifact_dir: string;
  last_rollout: RolloutArtifact | null;
  last_judgment: JudgmentArtifact | null;
  pending_recovery: PendingRecovery | null;
  legal_moves: string[];
}

export interface ReviewFilters {
  model: string;
  legal: LegalFilter;
  failureBucket: string;
  split: string;
  minRegret: number | null;
  maxRegret: number | null;
  search: string;
}

export interface ParsedModelOutput {
  reasoning: string;
  moveUci: string | null;
  rawOutput: string;
}
