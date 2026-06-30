import type {
  BackendHealth,
  GameState,
  OpeningBook,
  RecoveryAction,
  RolloutReviewItem,
  ScoreRolloutsResponse,
  Side,
  ToolAnalysis,
  ToolStatus
} from "./domain/types";

const JSON_HEADERS = { "Content-Type": "application/json" };

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, init);
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const payload = (await response.json()) as { detail?: string };
      detail = payload.detail ?? detail;
    } catch {
      // Keep HTTP status detail.
    }
    throw new Error(detail);
  }
  return (await response.json()) as T;
}

export function fetchBooks(): Promise<{ items: OpeningBook[] }> {
  return requestJson("/api/books");
}

export function fetchHealth(): Promise<BackendHealth> {
  return requestJson("/api/health");
}

export function fetchTools(): Promise<ToolStatus> {
  return requestJson("/api/tools");
}

export function analyzePosition(payload: {
  fen: string;
  book_id: string | null;
  include_stockfish: boolean;
}): Promise<ToolAnalysis> {
  return requestJson("/api/tools/analyze", {
    method: "POST",
    headers: JSON_HEADERS,
    body: JSON.stringify(payload)
  });
}

export function createGame(payload: {
  human_side: Side;
  book_id: string | null;
  book_max_plies: number;
  seed?: number | null;
}): Promise<GameState> {
  return requestJson("/api/games", {
    method: "POST",
    headers: JSON_HEADERS,
    body: JSON.stringify(payload)
  });
}

export function fetchGame(gameId: string): Promise<GameState> {
  return requestJson(`/api/games/${gameId}`);
}

export function sendHumanMove(gameId: string, moveUci: string): Promise<GameState> {
  return requestJson(`/api/games/${gameId}/human-move`, {
    method: "POST",
    headers: JSON_HEADERS,
    body: JSON.stringify({ move_uci: moveUci })
  });
}

export function requestLlmMove(gameId: string): Promise<GameState> {
  return requestJson(`/api/games/${gameId}/llm-move`, { method: "POST" });
}

export function recoverGameMove(gameId: string, action: RecoveryAction): Promise<GameState> {
  return requestJson(`/api/games/${gameId}/recovery-move`, {
    method: "POST",
    headers: JSON_HEADERS,
    body: JSON.stringify({ action })
  });
}

export function loadArtifactRun(artifactDir: string): Promise<{ run_id: string; count: number }> {
  return requestJson("/api/artifacts/load", {
    method: "POST",
    headers: JSON_HEADERS,
    body: JSON.stringify({ artifact_dir: artifactDir })
  });
}

export function fetchRollouts(runId: string): Promise<{ items: RolloutReviewItem[] }> {
  return requestJson(`/api/artifacts/${runId}/rollouts`);
}

export function scoreArtifactRollouts(
  runId: string,
  payload: { rollout_ids: string[] }
): Promise<ScoreRolloutsResponse> {
  return requestJson(`/api/artifacts/${runId}/score`, {
    method: "POST",
    headers: JSON_HEADERS,
    body: JSON.stringify(payload)
  });
}
