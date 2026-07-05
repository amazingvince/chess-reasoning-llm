import { Search } from "lucide-react";

import type { ToolAnalysis, ToolStatus } from "../domain/types";

type Mode = "play" | "review";

export function ToolsPanel({
  mode,
  status,
  analysis,
  canAnalyze,
  canScore,
  busy,
  visibleCount,
  onAnalyze,
  onScoreSelected,
  onScoreVisible
}: {
  mode: Mode;
  status: ToolStatus | null;
  analysis: ToolAnalysis | null;
  canAnalyze: boolean;
  canScore: boolean;
  busy: boolean;
  visibleCount: number;
  onAnalyze: () => void;
  onScoreSelected: () => void;
  onScoreVisible: () => void;
}) {
  return (
    <section className="inspector-section tools-panel">
      <h2><Search size={16} /> Tools</h2>
      <div className="tool-status">
        <strong>{stockfishStatusLabel(status)}</strong>
        {status ? (
          <span>
            Depth {status.stockfish.depth} / Threads {status.stockfish.threads} / Batch {status.batch_limit}
          </span>
        ) : (
          <span>Checking backend tools</span>
        )}
      </div>
      <div className="tool-caps">
        <span className={status?.capabilities.legal_moves ? "enabled" : ""}>Legal moves</span>
        <span className={status?.capabilities.opening_book_moves ? "enabled" : ""}>Opening books</span>
        <span className={status?.capabilities.stockfish_analysis ? "enabled" : ""}>Stockfish eval</span>
      </div>
      <div className="tool-actions">
        <button onClick={onAnalyze} disabled={!canAnalyze || busy}>
          <Search size={16} /> Analyze position
        </button>
        {mode === "review" ? (
          <>
            <button onClick={onScoreSelected} disabled={!canScore || busy}>Score selected</button>
            <button onClick={onScoreVisible} disabled={!canScore || busy || visibleCount === 0}>Score visible</button>
          </>
        ) : null}
      </div>
      {analysis ? (
        <div className="tool-results">
          <div className="tool-summary">
            <strong>{analysis.legal_move_count} legal moves</strong>
            <span>Turn: {analysis.turn}</span>
          </div>
          <ToolMoveList title="Legal" moves={analysis.legal_moves.slice(0, 12)} />
          <ToolMoveList title="Book" moves={analysis.book_moves.slice(0, 8)} />
          {analysis.stockfish ? (
            <div className="tool-stockfish">
              {analysis.stockfish.available ? (
                <strong>
                  Best {analysis.stockfish.best_move ?? "-"} / {formatStockfishScore(analysis.stockfish)}
                </strong>
              ) : (
                <strong>Stockfish unavailable</strong>
              )}
              {analysis.stockfish.pv_line ? <span>{analysis.stockfish.pv_line}</span> : null}
              {analysis.stockfish.error ? <span>{analysis.stockfish.error}</span> : null}
            </div>
          ) : null}
        </div>
      ) : null}
    </section>
  );
}

function ToolMoveList({
  title,
  moves
}: {
  title: string;
  moves: Array<{ uci: string; san: string; weight?: number }>;
}) {
  if (moves.length === 0) {
    return null;
  }
  return (
    <div className="tool-move-list">
      <span>{title}</span>
      <div>
        {moves.map((move) => (
          <span className="tool-chip" key={`${title}-${move.uci}`}>
            <strong>{move.uci}</strong>
            <em>{move.san}</em>
            {typeof move.weight === "number" ? <small>{move.weight}</small> : null}
          </span>
        ))}
      </div>
    </div>
  );
}

export function stockfishStatusLabel(status: ToolStatus | null): string {
  if (!status) {
    return "Tools loading";
  }
  if (!status.stockfish.enabled) {
    return "Stockfish disabled";
  }
  if (status.stockfish.available) {
    return status.stockfish.name ? `Stockfish ready (${status.stockfish.name})` : "Stockfish ready";
  }
  return "Stockfish unavailable";
}

export function formatStockfishScore(stockfish: NonNullable<ToolAnalysis["stockfish"]>): string {
  if (typeof stockfish.mate === "number") {
    return `Mate ${stockfish.mate}`;
  }
  if (typeof stockfish.cp === "number") {
    return `${stockfish.cp >= 0 ? "+" : ""}${stockfish.cp} cp`;
  }
  return "no score";
}
