import { describe, expect, it } from "vitest";

import { formatRegret } from "./ReviewControls";
import { formatStockfishScore, stockfishStatusLabel } from "./ToolsPanel";
import type { ToolAnalysis, ToolStatus } from "../domain/types";

describe("panel formatting helpers", () => {
  it("formats review and Stockfish panel values", () => {
    const disabled: ToolStatus = {
      stockfish: {
        enabled: false,
        available: false,
        path: null,
        name: null,
        depth: 16,
        threads: 1,
        hash_mb: 256,
        error: null
      },
      capabilities: {
        legal_moves: true,
        opening_book_moves: true,
        stockfish_analysis: false
      },
      batch_limit: 100
    };
    const stockfish: NonNullable<ToolAnalysis["stockfish"]> = {
      available: true,
      best_move: "e2e4",
      cp: 34,
      mate: null,
      depth: 16,
      error: null
    };

    expect(formatRegret(null)).toBe("-");
    expect(formatRegret(42)).toBe("42 cp");
    expect(stockfishStatusLabel(null)).toBe("Tools loading");
    expect(stockfishStatusLabel(disabled)).toBe("Stockfish disabled");
    expect(formatStockfishScore(stockfish)).toBe("+34 cp");
  });
});
