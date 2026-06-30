import { act, cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import type { GameState } from "./domain/types";

vi.mock("react-chessboard", () => ({
  Chessboard: () => <div data-testid="chessboard" />
}));

vi.mock("./api", () => ({
  createGame: vi.fn(),
  analyzePosition: vi.fn(),
  fetchBooks: vi.fn(),
  fetchGame: vi.fn(),
  fetchHealth: vi.fn(),
  fetchRollouts: vi.fn(),
  fetchTools: vi.fn(),
  loadArtifactRun: vi.fn(),
  recoverGameMove: vi.fn(),
  requestLlmMove: vi.fn(),
  scoreArtifactRollouts: vi.fn(),
  sendHumanMove: vi.fn()
}));

import App from "./App";
import {
  createGame,
  analyzePosition,
  fetchBooks,
  fetchGame,
  fetchHealth,
  fetchRollouts,
  fetchTools,
  loadArtifactRun,
  recoverGameMove,
  requestLlmMove,
  scoreArtifactRollouts,
  sendHumanMove
} from "./api";

const STARTING_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

describe("App", () => {
  beforeEach(() => {
    window.localStorage.clear();
    vi.mocked(fetchBooks).mockResolvedValue({ items: [] });
    vi.mocked(fetchHealth).mockResolvedValue({
      status: "ok",
      model: "unit-model",
      llm_mode: "legal_stub"
    });
    vi.mocked(fetchTools).mockResolvedValue(disabledTools());
    vi.mocked(fetchRollouts).mockResolvedValue({ items: [] });
    vi.mocked(loadArtifactRun).mockResolvedValue({ run_id: "run", count: 0 });
    vi.mocked(recoverGameMove).mockResolvedValue(gameState("recovered-game"));
    vi.mocked(requestLlmMove).mockResolvedValue(gameState("llm-game"));
    vi.mocked(sendHumanMove).mockResolvedValue(gameState("human-game"));
    vi.mocked(analyzePosition).mockResolvedValue(toolAnalysis());
    vi.mocked(scoreArtifactRollouts).mockResolvedValue({
      items: [],
      scored_count: 0,
      skipped_count: 0,
      error_count: 0,
      errors: [],
      tool_status: disabledTools()
    });
  });

  afterEach(() => {
    cleanup();
    vi.clearAllMocks();
  });

  it("does not let a stale restored game overwrite a newly created game", async () => {
    window.localStorage.setItem("chess-llm-workbench.active-game-id", "old-game");
    const restore = deferred<GameState>();
    vi.mocked(fetchGame).mockReturnValue(restore.promise);
    vi.mocked(createGame).mockResolvedValue(gameState("new-game"));

    render(<App />);

    expect(fetchGame).toHaveBeenCalledWith("old-game");
    fireEvent.click(screen.getByRole("button", { name: "New game" }));
    expect(await screen.findByText("new-game")).toBeInTheDocument();

    await act(async () => {
      restore.resolve(gameState("old-game", "black"));
      await restore.promise;
    });

    await waitFor(() => {
      expect(screen.getByText("new-game")).toBeInTheDocument();
      expect(screen.queryByText("old-game")).not.toBeInTheDocument();
    });
  });

  it("renders disabled Stockfish tool status from the backend", async () => {
    render(<App />);

    expect(await screen.findByText("Stockfish disabled")).toBeInTheDocument();
    expect(screen.getByText("Legal moves")).toBeInTheDocument();
    expect(screen.getByText("Opening books")).toBeInTheDocument();
  });

  it("analyzes the active board and displays legal, book, and Stockfish data", async () => {
    vi.mocked(fetchTools).mockResolvedValue(enabledTools());
    vi.mocked(createGame).mockResolvedValue(gameState("analysis-game"));
    vi.mocked(analyzePosition).mockResolvedValue(toolAnalysis());

    render(<App />);
    fireEvent.click(screen.getByRole("button", { name: "New game" }));
    expect(await screen.findByText("analysis-game")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Analyze position" }));

    await waitFor(() => {
      expect(analyzePosition).toHaveBeenCalledWith({
        fen: STARTING_FEN,
        book_id: null,
        include_stockfish: true
      });
    });
    expect(await screen.findByText("20 legal moves")).toBeInTheDocument();
    expect(screen.getAllByText("e2e4").length).toBeGreaterThan(0);
    expect(screen.getAllByText("e4").length).toBeGreaterThan(0);
    expect(screen.getByText("Best e2e4 / +35 cp")).toBeInTheDocument();
  });

  it("shows reasoning and judgment for a selected live LLM timeline move", async () => {
    vi.mocked(createGame).mockResolvedValue(liveHistoryGame("history-game"));

    render(<App />);
    fireEvent.click(screen.getByRole("button", { name: "New game" }));
    expect(await screen.findByText("Second line")).toBeInTheDocument();
    expect(screen.getByText("33 cp")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "2. e7-e5" }));

    expect(await screen.findByText("First line")).toBeInTheDocument();
    expect(screen.getByText("12 cp")).toBeInTheDocument();
    expect(screen.getByText("Historical move")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Resume latest" }));

    expect(await screen.findByText("Second line")).toBeInTheDocument();
    expect(screen.getByText("33 cp")).toBeInTheDocument();
  });

  it("analyzes a selected live human move from that move's starting position", async () => {
    const history = liveHistoryGame("analysis-history");
    vi.mocked(createGame).mockResolvedValue(history);
    vi.mocked(analyzePosition).mockResolvedValue(toolAnalysis());

    render(<App />);
    fireEvent.click(screen.getByRole("button", { name: "New game" }));
    expect(await screen.findByText("Second line")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "1. e2-e4" }));
    fireEvent.click(screen.getByRole("button", { name: "Analyze position" }));

    await waitFor(() => {
      expect(analyzePosition).toHaveBeenCalledWith({
        fen: history.moves[0].fen_before,
        book_id: null,
        include_stockfish: false
      });
    });
    expect(await screen.findByText("No model reasoning for this move.")).toBeInTheDocument();
  });

  it("uses the latest rollout when a selected legacy LLM move only carries ids", async () => {
    const history = liveHistoryGame("legacy-history");
    const latestRollout = history.moves[1].rollout;
    const latestJudgment = history.moves[1].judgment;
    history.moves = history.moves.slice(0, 2).map((move) => ({
      ...move,
      rollout: undefined,
      judgment: undefined
    }));
    history.last_rollout = latestRollout ?? null;
    history.last_judgment = latestJudgment ?? null;
    vi.mocked(createGame).mockResolvedValue(history);

    render(<App />);
    fireEvent.click(screen.getByRole("button", { name: "New game" }));
    expect(await screen.findByText("First line")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "2. e7-e5" }));

    expect(await screen.findByText("First line")).toBeInTheDocument();
    expect(screen.getByText("12 cp")).toBeInTheDocument();
    expect(screen.getByText("Historical move")).toBeInTheDocument();
  });

  it("renders recovery controls when a live LLM move needs recovery", async () => {
    vi.mocked(createGame).mockResolvedValue(pendingRecoveryGame("paused-game"));

    render(<App />);
    fireEvent.click(screen.getByRole("button", { name: "New game" }));

    expect(await screen.findByText("Recovery needed")).toBeInTheDocument();
    expect(screen.getByText("Parsed move e7e5 is illegal in the prompt position.")).toBeInTheDocument();
    expect(screen.getByText("Bad side to move.")).toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Ask LLM" })).toBeDisabled();
    expect(screen.getByRole("button", { name: "Retry" })).toBeEnabled();
    expect(screen.getByRole("button", { name: "Retry with legal moves" })).toBeEnabled();
    expect(screen.getByRole("button", { name: "Teacher move e2e4" })).toBeEnabled();
    expect(screen.getByRole("button", { name: "Legal stub" })).toBeEnabled();
  });

  it("runs an explicit recovery action and replaces the paused game state", async () => {
    vi.mocked(createGame).mockResolvedValue(pendingRecoveryGame("recover-game"));
    vi.mocked(recoverGameMove).mockResolvedValue(recoveredGame("recover-game"));

    render(<App />);
    fireEvent.click(screen.getByRole("button", { name: "New game" }));
    expect(await screen.findByText("Recovery needed")).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Retry with legal moves" }));

    await waitFor(() => {
      expect(recoverGameMove).toHaveBeenCalledWith("recover-game", "retry_with_legal_moves");
    });
    expect(await screen.findByText("Recovered line")).toBeInTheDocument();
    expect(screen.queryByText("Recovery needed")).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "1. a2-a3" })).toBeInTheDocument();
  });

  it("scores the selected rollout and updates the annotation facts", async () => {
    const original = rolloutItem("rollout-1", "model-a", "d2d4");
    const scored = rolloutItem("rollout-1", "model-a", "d2d4", {
      teacher_move_uci: "e2e4",
      regret_cp: 25,
      feedback: "Stockfish scored parsed move d2d4 with regret 25.0 cp."
    });
    vi.mocked(fetchRollouts).mockResolvedValue({ items: [original] });
    vi.mocked(scoreArtifactRollouts).mockResolvedValue({
      items: [scored],
      scored_count: 1,
      skipped_count: 0,
      error_count: 0,
      errors: [],
      tool_status: enabledTools()
    });

    render(<App />);
    fireEvent.change(screen.getByPlaceholderText("path/to/artifacts"), {
      target: { value: "C:/runs/example" }
    });
    fireEvent.click(screen.getByRole("button", { name: "Load run" }));
    expect(await screen.findByRole("button", { name: /model-a\s+d2d4/i })).toBeInTheDocument();

    fireEvent.click(screen.getByRole("button", { name: "Score selected" }));

    await waitFor(() => {
      expect(scoreArtifactRollouts).toHaveBeenCalledWith("run", {
        rollout_ids: ["rollout-1"]
      });
    });
    expect(await screen.findByText("25 cp")).toBeInTheDocument();
    expect(screen.getAllByText("e2e4").length).toBeGreaterThan(0);
  });

  it("scores only filtered visible review rows", async () => {
    vi.mocked(fetchRollouts).mockResolvedValue({
      items: [
        rolloutItem("rollout-a", "model-a", "e2e4"),
        rolloutItem("rollout-b", "model-b", "d2d4")
      ]
    });
    vi.mocked(scoreArtifactRollouts).mockResolvedValue({
      items: [rolloutItem("rollout-a", "model-a", "e2e4", { regret_cp: 0, teacher_move_uci: "e2e4" })],
      scored_count: 1,
      skipped_count: 0,
      error_count: 0,
      errors: [],
      tool_status: enabledTools()
    });

    render(<App />);
    fireEvent.change(screen.getByPlaceholderText("path/to/artifacts"), {
      target: { value: "C:/runs/example" }
    });
    fireEvent.click(screen.getByRole("button", { name: "Load run" }));
    expect(await screen.findByRole("button", { name: /model-a\s+e2e4/i })).toBeInTheDocument();

    fireEvent.change(screen.getByDisplayValue("All models"), {
      target: { value: "model-a" }
    });
    fireEvent.click(screen.getByRole("button", { name: "Score visible" }));

    await waitFor(() => {
      expect(scoreArtifactRollouts).toHaveBeenCalledWith("run", {
        rollout_ids: ["rollout-a"]
      });
    });
  });
});

function gameState(gameId: string, humanSide: "white" | "black" = "white"): GameState {
  return {
    game_id: gameId,
    human_side: humanSide,
    turn: humanSide,
    fen: STARTING_FEN,
    created_at: "2026-06-28T00:00:00Z",
    moves: [],
    book_line: [],
    artifact_dir: `runs/${gameId}`,
    last_rollout: null,
    last_judgment: null,
    pending_recovery: null,
    legal_moves: ["e2e4"]
  };
}

function pendingRecoveryGame(gameId: string): GameState {
  return {
    game_id: gameId,
    human_side: "black",
    turn: "white",
    fen: STARTING_FEN,
    created_at: "2026-06-28T00:00:00Z",
    moves: [],
    book_line: [],
    artifact_dir: `runs/${gameId}`,
    last_rollout: {
      rollout_id: "rollout-failed",
      prompt_id: "prompt-failed",
      model_id: "unit-model",
      raw_output: "<think>Bad side to move.</think><move>e7e5</move>",
      parsed_answer: {
        raw_text: "<move>e7e5</move>",
        move_uci: "e7e5",
        format_type: "move_tag",
        parse_error: null,
        metadata: {}
      },
      metadata: { requires_recovery: true }
    },
    last_judgment: {
      judgment_id: "judgment-failed",
      rollout_id: "rollout-failed",
      legal: false,
      regret_cp: null,
      failure_bucket: "illegal_move",
      teacher_move_uci: "e2e4",
      feedback: "Parsed move e7e5 is illegal in the prompt position.",
      metadata: { requires_recovery: true }
    },
    pending_recovery: {
      rollout_id: "rollout-failed",
      judgment_id: "judgment-failed",
      reason: "Parsed move e7e5 is illegal in the prompt position.",
      created_at: "2026-06-28T00:00:01Z"
    },
    legal_moves: ["e2e4", "g1f3"]
  };
}

function recoveredGame(gameId: string): GameState {
  return {
    game_id: gameId,
    human_side: "black",
    turn: "black",
    fen: "rnbqkbnr/pppppppp/8/8/8/P7/1PPPPPPP/RNBQKBNR b KQkq - 0 1",
    created_at: "2026-06-28T00:00:00Z",
    moves: [
      {
        source: "llm",
        side: "white",
        move_uci: "a2a3",
        fen_before: STARTING_FEN,
        rollout_id: "rollout-recovered",
        judgment_id: "judgment-recovered",
        rollout: {
          rollout_id: "rollout-recovered",
          prompt_id: "prompt-recovered",
          model_id: "unit-model",
          raw_output: "<think>Recovered line</think><move>a2a3</move>",
          parsed_answer: {
            raw_text: "<move>a2a3</move>",
            move_uci: "a2a3",
            format_type: "move_tag",
            parse_error: null,
            metadata: {}
          },
          metadata: { recovery_action: "retry_with_legal_moves" }
        },
        judgment: {
          judgment_id: "judgment-recovered",
          rollout_id: "rollout-recovered",
          legal: true,
          regret_cp: null,
          failure_bucket: "legal_unscored",
          teacher_move_uci: null,
          feedback: "Parsed move a2a3 is legal; engine regret is not scored yet.",
          metadata: { recovery_action: "retry_with_legal_moves" }
        }
      }
    ],
    book_line: [],
    artifact_dir: `runs/${gameId}`,
    last_rollout: {
      rollout_id: "rollout-recovered",
      prompt_id: "prompt-recovered",
      model_id: "unit-model",
      raw_output: "<think>Recovered line</think><move>a2a3</move>",
      parsed_answer: {
        raw_text: "<move>a2a3</move>",
        move_uci: "a2a3",
        format_type: "move_tag",
        parse_error: null,
        metadata: {}
      },
      metadata: { recovery_action: "retry_with_legal_moves" }
    },
    last_judgment: {
      judgment_id: "judgment-recovered",
      rollout_id: "rollout-recovered",
      legal: true,
      regret_cp: null,
      failure_bucket: "legal_unscored",
      teacher_move_uci: null,
      feedback: "Parsed move a2a3 is legal; engine regret is not scored yet.",
      metadata: { recovery_action: "retry_with_legal_moves" }
    },
    pending_recovery: null,
    legal_moves: ["a7a6"]
  };
}

function liveHistoryGame(gameId: string): GameState {
  const firstRollout = {
    rollout_id: "rollout-first",
    prompt_id: "prompt-first",
    model_id: "unit-model",
    raw_output: "<think>First line</think><move>e7e5</move>",
    parsed_answer: {
      raw_text: "<move>e7e5</move>",
      move_uci: "e7e5",
      format_type: "move_tag",
      parse_error: null,
      metadata: {}
    },
    metadata: {}
  };
  const secondRollout = {
    rollout_id: "rollout-second",
    prompt_id: "prompt-second",
    model_id: "unit-model",
    raw_output: "<think>Second line</think><move>b8c6</move>",
    parsed_answer: {
      raw_text: "<move>b8c6</move>",
      move_uci: "b8c6",
      format_type: "move_tag",
      parse_error: null,
      metadata: {}
    },
    metadata: {}
  };
  const firstJudgment = {
    judgment_id: "judgment-first",
    rollout_id: "rollout-first",
    legal: true,
    regret_cp: 12,
    failure_bucket: null,
    teacher_move_uci: "e7e5",
    feedback: "First move scored.",
    metadata: {}
  };
  const secondJudgment = {
    judgment_id: "judgment-second",
    rollout_id: "rollout-second",
    legal: true,
    regret_cp: 33,
    failure_bucket: null,
    teacher_move_uci: "g8f6",
    feedback: "Second move scored.",
    metadata: {}
  };

  return {
    game_id: gameId,
    human_side: "white",
    turn: "white",
    fen: "r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3",
    created_at: "2026-06-28T00:00:00Z",
    moves: [
      {
        source: "human",
        side: "white",
        move_uci: "e2e4",
        fen_before: STARTING_FEN
      },
      {
        source: "llm",
        side: "black",
        move_uci: "e7e5",
        fen_before: "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq - 0 1",
        rollout_id: "rollout-first",
        judgment_id: "judgment-first",
        rollout: firstRollout,
        judgment: firstJudgment
      },
      {
        source: "human",
        side: "white",
        move_uci: "g1f3",
        fen_before: "rnbqkbnr/pppp1ppp/8/4p3/4P3/8/PPPP1PPP/RNBQKBNR w KQkq - 0 2"
      },
      {
        source: "llm",
        side: "black",
        move_uci: "b8c6",
        fen_before: "rnbqkbnr/pppp1ppp/8/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R b KQkq - 1 2",
        rollout_id: "rollout-second",
        judgment_id: "judgment-second",
        rollout: secondRollout,
        judgment: secondJudgment
      }
    ],
    book_line: [],
    artifact_dir: `runs/${gameId}`,
    last_rollout: secondRollout,
    last_judgment: secondJudgment,
    pending_recovery: null,
    legal_moves: ["d2d4"]
  };
}

function disabledTools() {
  return {
    stockfish: {
      enabled: false,
      available: false,
      path: null,
      name: null,
      depth: 16,
      threads: 1,
      hash_mb: 256,
      error: "CHESS_UI_STOCKFISH_PATH is not configured"
    },
    capabilities: {
      legal_moves: true,
      opening_book_moves: true,
      stockfish_analysis: false
    },
    batch_limit: 100
  };
}

function enabledTools() {
  return {
    stockfish: {
      enabled: true,
      available: true,
      path: "C:/stockfish.exe",
      name: "Fake Stockfish",
      depth: 18,
      threads: 2,
      hash_mb: 32,
      error: null
    },
    capabilities: {
      legal_moves: true,
      opening_book_moves: true,
      stockfish_analysis: true
    },
    batch_limit: 2
  };
}

function toolAnalysis() {
  return {
    fen: STARTING_FEN,
    turn: "white" as const,
    legal_move_count: 20,
    legal_moves: [
      { uci: "e2e4", san: "e4", capture: false, check: false, promotion: null },
      { uci: "d2d4", san: "d4", capture: false, check: false, promotion: null }
    ],
    book_moves: [
      { uci: "e2e4", san: "e4", capture: false, check: false, promotion: null, weight: 12 }
    ],
    stockfish: {
      available: true,
      best_move: "e2e4",
      cp: 35,
      mate: null,
      pv_line: "e2e4 e7e5",
      depth: 18
    }
  };
}

function rolloutItem(
  rolloutId: string,
  modelId: string,
  moveUci: string,
  judgmentOverrides: Record<string, unknown> = {}
) {
  return {
    prompt: {
      prompt_id: `prompt-${rolloutId}`,
      messages: [{ role: "user", content: "Choose a move." }],
      fen: STARTING_FEN,
      task_type: "best_move",
      metadata: { split: "unit" }
    },
    rollout: {
      rollout_id: rolloutId,
      prompt_id: `prompt-${rolloutId}`,
      model_id: modelId,
      raw_output: `<move>${moveUci}</move>`,
      parsed_answer: {
        raw_text: `<move>${moveUci}</move>`,
        move_uci: moveUci,
        format_type: "move_tag",
        parse_error: null,
        metadata: {}
      },
      metadata: {}
    },
    judgment: {
      judgment_id: `judgment-${rolloutId}`,
      rollout_id: rolloutId,
      legal: true,
      regret_cp: null,
      failure_bucket: null,
      teacher_move_uci: null,
      feedback: "Legal only.",
      metadata: {},
      ...judgmentOverrides
    }
  };
}

function deferred<T>() {
  let resolve!: (value: T) => void;
  let reject!: (reason?: unknown) => void;
  const promise = new Promise<T>((res, rej) => {
    resolve = res;
    reject = rej;
  });
  return { promise, resolve, reject };
}
