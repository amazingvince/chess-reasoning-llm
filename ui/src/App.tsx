import { useEffect, useMemo, useRef, useState } from "react";
import { Chess } from "chess.js";
import { Chessboard } from "react-chessboard";
import {
  Activity,
  BookOpen,
  Bot,
  FileSearch,
  Loader2,
  Play,
  RotateCcw,
  Search,
  User
} from "lucide-react";

import {
  analyzePosition,
  createGame,
  fetchGame,
  fetchBooks,
  fetchHealth,
  fetchRollouts,
  fetchTools,
  loadArtifactRun,
  recoverGameMove,
  requestLlmMove,
  scoreArtifactRollouts,
  sendHumanMove
} from "./api";
import { clearActiveGameId, readActiveGameId, writeActiveGameId } from "./domain/activeGameStorage";
import {
  buildMoveUci,
  canDragPiece,
  isPromotionMove,
  legalMoveTargets,
  promotionChoiceFromPiece,
  squareStylesForTargets,
  type PromotionChoice
} from "./domain/boardInteraction";
import { moveLabel, normalizeMoveInput, orientationForSide } from "./domain/game";
import { splitModelOutput } from "./domain/reasoning";
import {
  defaultReviewFilters,
  filterRolloutItems
} from "./domain/reviewFilters";
import { Fact, formatRegret, ReviewControls } from "./components/ReviewControls";
import { ToolsPanel } from "./components/ToolsPanel";
import type {
  GameState,
  OpeningBook,
  RecoveryAction,
  ReviewFilters,
  RolloutReviewItem,
  Side,
  BackendHealth,
  ToolAnalysis,
  ToolStatus
} from "./domain/types";

type Mode = "play" | "review";

const STARTING_FEN = "start";
const PROMOTION_CHOICES: Array<{ value: PromotionChoice; label: string }> = [
  { value: "q", label: "Q" },
  { value: "r", label: "R" },
  { value: "b", label: "B" },
  { value: "n", label: "N" }
];

export default function App() {
  const [mode, setMode] = useState<Mode>("play");
  const [health, setHealth] = useState<BackendHealth | null>(null);
  const [toolStatus, setToolStatus] = useState<ToolStatus | null>(null);
  const [toolAnalysis, setToolAnalysis] = useState<ToolAnalysis | null>(null);
  const [books, setBooks] = useState<OpeningBook[]>([]);
  const [humanSide, setHumanSide] = useState<Side>("white");
  const [bookId, setBookId] = useState<string>("");
  const [bookMaxPlies, setBookMaxPlies] = useState(6);
  const [humanMoveInput, setHumanMoveInput] = useState("");
  const [game, setGame] = useState<GameState | null>(null);
  const [selectedPlayMoveIndex, setSelectedPlayMoveIndex] = useState<number | null>(null);
  const [reviewItems, setReviewItems] = useState<RolloutReviewItem[]>([]);
  const [selectedRolloutId, setSelectedRolloutId] = useState<string>("");
  const [artifactDir, setArtifactDir] = useState("");
  const [loadedRun, setLoadedRun] = useState<string>("");
  const [filters, setFilters] = useState<ReviewFilters>(defaultReviewFilters);
  const [previewFen, setPreviewFen] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedSquare, setSelectedSquare] = useState<string | null>(null);
  const [pendingPromotion, setPendingPromotion] = useState<{ from: string; to: string } | null>(null);
  const restoreRequestId = useRef(0);

  useEffect(() => {
    fetchBooks()
      .then((payload) => setBooks(payload.items))
      .catch((err: Error) => setError(err.message));
    fetchHealth()
      .then(setHealth)
      .catch((err: Error) => setError(err.message));
    fetchTools()
      .then(setToolStatus)
      .catch((err: Error) => setError(err.message));
    const storedGameId = readActiveGameId();
    if (storedGameId) {
      const requestId = restoreRequestId.current + 1;
      restoreRequestId.current = requestId;
      fetchGame(storedGameId)
        .then((restoredGame) => {
          if (restoreRequestId.current !== requestId || readActiveGameId() !== storedGameId) {
            return;
          }
          setGame(restoredGame);
          setHumanSide(restoredGame.human_side);
          setSelectedPlayMoveIndex(null);
          setSelectedSquare(null);
          setPendingPromotion(null);
          setToolAnalysis(null);
          setMode("play");
        })
        .catch(() => {
          if (restoreRequestId.current === requestId && readActiveGameId() === storedGameId) {
            clearActiveGameId();
          }
        });
    }
  }, []);

  const filteredItems = useMemo(
    () => filterRolloutItems(reviewItems, filters),
    [reviewItems, filters]
  );
  const selectedItem = useMemo(
    () =>
      filteredItems.find((item) => item.rollout.rollout_id === selectedRolloutId) ??
      filteredItems[0] ??
      null,
    [filteredItems, selectedRolloutId]
  );
  const selectedPlayMove =
    mode === "play" && selectedPlayMoveIndex !== null
      ? game?.moves[selectedPlayMoveIndex] ?? null
      : null;
  const isInspectingHistoricalPlayMove = Boolean(selectedPlayMove);
  const selectedMoveMatchesLastRollout = Boolean(
    selectedPlayMove?.rollout_id && game?.last_rollout?.rollout_id === selectedPlayMove.rollout_id
  );
  const selectedMoveMatchesLastJudgment = Boolean(
    selectedPlayMove &&
      game?.last_judgment &&
      ((selectedPlayMove.judgment_id && game.last_judgment.judgment_id === selectedPlayMove.judgment_id) ||
        (selectedPlayMove.rollout_id && game.last_judgment.rollout_id === selectedPlayMove.rollout_id))
  );
  const playInspectionRollout = selectedPlayMove
    ? selectedPlayMove.rollout ?? (selectedMoveMatchesLastRollout ? game?.last_rollout ?? null : null)
    : game?.last_rollout ?? null;
  const playInspectionJudgment = selectedPlayMove
    ? selectedPlayMove.judgment ?? (selectedMoveMatchesLastJudgment ? game?.last_judgment ?? null : null)
    : game?.last_judgment ?? null;
  const inspectedRollout = mode === "play" ? playInspectionRollout : selectedItem?.rollout ?? null;
  const inspectedJudgment = mode === "play" ? playInspectionJudgment : selectedItem?.judgment ?? null;
  const boardFen =
    mode === "review"
      ? previewFen ?? selectedItem?.prompt?.fen ?? STARTING_FEN
      : selectedPlayMove
        ? selectedPlayMove.fen_before ?? STARTING_FEN
        : game?.fen ?? STARTING_FEN;
  const analysisFen =
    mode === "review"
      ? selectedItem?.prompt?.fen ?? null
      : selectedPlayMove
        ? selectedPlayMove.fen_before ?? null
        : game?.fen ?? null;
  const latestOutput =
    mode === "play" && selectedPlayMove
      ? inspectedRollout?.raw_output ?? selectedPlayMove.raw_output ?? undefined
      : inspectedRollout?.raw_output;
  const parsedOutput = splitModelOutput(latestOutput);
  const hasPendingRecovery = Boolean(game?.pending_recovery);
  const legalTargets = useMemo(
    () =>
      game &&
      mode === "play" &&
      !isInspectingHistoricalPlayMove &&
      !hasPendingRecovery &&
      game.turn === game.human_side &&
      selectedSquare
        ? legalMoveTargets(game.fen, selectedSquare)
        : [],
    [game, hasPendingRecovery, isInspectingHistoricalPlayMove, mode, selectedSquare]
  );
  const squareStyles = useMemo(
    () => squareStylesForTargets(selectedSquare, legalTargets),
    [selectedSquare, legalTargets]
  );
  const boardStatus = pendingPromotion
    ? `Promote ${pendingPromotion.from}-${pendingPromotion.to}`
    : isInspectingHistoricalPlayMove
      ? `Inspecting move ${(selectedPlayMoveIndex ?? 0) + 1}`
    : hasPendingRecovery && mode === "play"
      ? "Recovery paused"
    : selectedSquare
      ? `${selectedSquare}: ${legalTargets.length} legal ${legalTargets.length === 1 ? "move" : "moves"}`
      : game && mode === "play" && game.turn !== game.human_side
        ? "Waiting for LLM"
        : "Ready";
  const liveControlsDisabled = !game || busy || isInspectingHistoricalPlayMove;

  useEffect(() => {
    setSelectedSquare(null);
    setPendingPromotion(null);
  }, [game?.fen, mode]);

  async function runAction(action: () => Promise<void>) {
    setBusy(true);
    setError(null);
    try {
      await action();
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err));
    } finally {
      setBusy(false);
    }
  }

  function applyGameState(nextGame: GameState) {
    cancelPendingRestore();
    setGame(nextGame);
    setSelectedPlayMoveIndex(null);
    setSelectedSquare(null);
    setPendingPromotion(null);
    setToolAnalysis(null);
    writeActiveGameId(nextGame.game_id);
  }

  function cancelPendingRestore() {
    restoreRequestId.current += 1;
  }

  async function submitHumanMove(moveUci: string) {
    if (!game || isInspectingHistoricalPlayMove) {
      return;
    }
    setSelectedSquare(null);
    setPendingPromotion(null);
    await runAction(async () => {
      applyGameState(await sendHumanMove(game.game_id, moveUci));
    });
  }

  async function handleCreateGame() {
    cancelPendingRestore();
    await runAction(async () => {
      const nextGame = await createGame({
        human_side: humanSide,
        book_id: bookId || null,
        book_max_plies: bookId ? bookMaxPlies : 0
      });
      applyGameState(nextGame);
      setMode("play");
    });
  }

  async function handleLlmMove() {
    if (!game || isInspectingHistoricalPlayMove) {
      return;
    }
    if (game.pending_recovery) {
      setError("Choose a recovery action before asking for another move.");
      return;
    }
    await runAction(async () => {
      applyGameState(await requestLlmMove(game.game_id));
    });
  }

  async function handleRecovery(action: RecoveryAction) {
    if (!game || isInspectingHistoricalPlayMove || !game.pending_recovery) {
      return;
    }
    await runAction(async () => {
      applyGameState(await recoverGameMove(game.game_id, action));
    });
  }

  async function handleHumanMoveInput() {
    if (!game || isInspectingHistoricalPlayMove) {
      return;
    }
    const moveUci = normalizeMoveInput(humanMoveInput);
    if (!moveUci) {
      setError("Enter a UCI move.");
      return;
    }
    await runAction(async () => {
      applyGameState(await sendHumanMove(game.game_id, moveUci));
      setHumanMoveInput("");
    });
  }

  async function handleLoadArtifacts() {
    if (!artifactDir.trim()) {
      setError("Enter an artifact directory.");
      return;
    }
    cancelPendingRestore();
    await runAction(async () => {
      const loaded = await loadArtifactRun(artifactDir.trim());
      const rows = await fetchRollouts(loaded.run_id);
      setLoadedRun(loaded.run_id);
      setReviewItems(rows.items);
      setSelectedRolloutId(rows.items[0]?.rollout.rollout_id ?? "");
      setSelectedPlayMoveIndex(null);
      setPreviewFen(null);
      setToolAnalysis(null);
      setMode("review");
    });
  }

  function handleModeChange(nextMode: Mode) {
    cancelPendingRestore();
    setSelectedPlayMoveIndex(null);
    setSelectedSquare(null);
    setPendingPromotion(null);
    setToolAnalysis(null);
    setMode(nextMode);
  }

  function handleDrop(sourceSquare: string, targetSquare: string, piece: string): boolean {
    if (!game || mode !== "play" || isInspectingHistoricalPlayMove || game.pending_recovery || game.turn !== game.human_side || busy) {
      return false;
    }
    const promotion = isPromotionMove(game.fen, sourceSquare, targetSquare)
      ? promotionChoiceFromPiece(piece)
      : "q";
    const moveUci = promotion ? buildMoveUci(game.fen, sourceSquare, targetSquare, promotion) : null;
    if (!moveUci) {
      setSelectedSquare(null);
      setPendingPromotion(null);
      return false;
    }
    void submitHumanMove(moveUci);
    return true;
  }

  function handlePromotionCheck(sourceSquare: string, targetSquare: string): boolean {
    return Boolean(
      game &&
        mode === "play" &&
        !isInspectingHistoricalPlayMove &&
        !game.pending_recovery &&
        isPromotionMove(game.fen, sourceSquare, targetSquare)
    );
  }

  function handlePromotionPieceSelect(
    piece?: string,
    promoteFromSquare?: string,
    promoteToSquare?: string
  ): boolean {
    if (!game || isInspectingHistoricalPlayMove || game.pending_recovery || !piece || !promoteFromSquare || !promoteToSquare) {
      return false;
    }
    const promotion = promotionChoiceFromPiece(piece);
    const moveUci = promotion ? buildMoveUci(game.fen, promoteFromSquare, promoteToSquare, promotion) : null;
    if (!moveUci) {
      return false;
    }
    void submitHumanMove(moveUci);
    return true;
  }

  function handlePieceDragBegin(_piece: string, sourceSquare: string) {
    if (!game || mode !== "play" || isInspectingHistoricalPlayMove || game.pending_recovery || game.turn !== game.human_side || busy) {
      return;
    }
    setPendingPromotion(null);
    setSelectedSquare(sourceSquare);
  }

  function handlePieceDragEnd() {
    setSelectedSquare(null);
  }

  function handleSquareClick(square: string, piece?: string) {
    if (!game || mode !== "play" || isInspectingHistoricalPlayMove || game.pending_recovery || game.turn !== game.human_side || busy) {
      return;
    }
    if (selectedSquare && square !== selectedSquare) {
      if (isPromotionMove(game.fen, selectedSquare, square)) {
        setPendingPromotion({ from: selectedSquare, to: square });
        return;
      }
      const moveUci = buildMoveUci(game.fen, selectedSquare, square);
      if (moveUci) {
        void submitHumanMove(moveUci);
        return;
      }
    }
    if (piece && canDragPiece(game.fen, square, piece, game.human_side, game.turn, busy)) {
      setPendingPromotion(null);
      setSelectedSquare(square);
      return;
    }
    setPendingPromotion(null);
    setSelectedSquare(null);
  }

  function handlePromotionChoice(choice: PromotionChoice) {
    if (!game || isInspectingHistoricalPlayMove || game.pending_recovery || !pendingPromotion) {
      return;
    }
    const moveUci = buildMoveUci(game.fen, pendingPromotion.from, pendingPromotion.to, choice);
    if (!moveUci) {
      setError("Illegal promotion move.");
      setPendingPromotion(null);
      return;
    }
    void submitHumanMove(moveUci);
  }

  function handleIsDraggablePiece({ piece, sourceSquare }: { piece: string; sourceSquare: string }) {
    return Boolean(
      game &&
        mode === "play" &&
        !isInspectingHistoricalPlayMove &&
        !game.pending_recovery &&
        canDragPiece(game.fen, sourceSquare, piece, game.human_side, game.turn, busy)
    );
  }

  function inspectPlayMove(index: number) {
    setSelectedPlayMoveIndex(index);
    setSelectedSquare(null);
    setPendingPromotion(null);
    setToolAnalysis(null);
  }

  function resumeLatestPosition() {
    setSelectedPlayMoveIndex(null);
    setSelectedSquare(null);
    setPendingPromotion(null);
    setToolAnalysis(null);
  }

  function previewMove(moveUci: string | null | undefined) {
    const fen = selectedItem?.prompt?.fen;
    if (!fen || !moveUci) {
      return;
    }
    try {
      const chess = new Chess(fen);
      chess.move({ from: moveUci.slice(0, 2), to: moveUci.slice(2, 4), promotion: moveUci.slice(4) || undefined });
      setPreviewFen(chess.fen());
    } catch {
      setPreviewFen(fen);
    }
  }

  async function handleAnalyzePosition() {
    if (!analysisFen) {
      setError("No position to analyze.");
      return;
    }
    await runAction(async () => {
      setToolAnalysis(
        await analyzePosition({
          fen: analysisFen,
          book_id: bookId || null,
          include_stockfish: Boolean(toolStatus?.stockfish.available)
        })
      );
    });
  }

  async function handleScoreSelected() {
    if (!loadedRun || !selectedItem) {
      setError("Load a run and select a rollout first.");
      return;
    }
    await scoreReviewRollouts([selectedItem.rollout.rollout_id]);
  }

  async function handleScoreVisible() {
    if (!loadedRun) {
      setError("Load a run first.");
      return;
    }
    const limit = toolStatus?.batch_limit ?? filteredItems.length;
    const rolloutIds = filteredItems
      .slice(0, limit)
      .map((item) => item.rollout.rollout_id);
    if (rolloutIds.length === 0) {
      setError("No visible rollouts to score.");
      return;
    }
    await scoreReviewRollouts(rolloutIds);
  }

  async function scoreReviewRollouts(rolloutIds: string[]) {
    await runAction(async () => {
      const response = await scoreArtifactRollouts(loadedRun, { rollout_ids: rolloutIds });
      setToolStatus(response.tool_status);
      mergeReviewItems(response.items);
    });
  }

  function mergeReviewItems(items: RolloutReviewItem[]) {
    const updates = new Map(items.map((item) => [item.rollout.rollout_id, item]));
    setReviewItems((current) =>
      current.map((item) => updates.get(item.rollout.rollout_id) ?? item)
    );
  }

  return (
    <main className="app-shell">
      <aside className="sidebar">
        <div className="brand">
          <Activity size={22} />
          <div>
            <h1>Chess LLM</h1>
            <span>Workbench</span>
          </div>
        </div>

        <div className="segmented" aria-label="Mode">
          <button className={mode === "play" ? "active" : ""} onClick={() => handleModeChange("play")}>
            <Play size={16} /> Play
          </button>
          <button className={mode === "review" ? "active" : ""} onClick={() => handleModeChange("review")}>
            <FileSearch size={16} /> Review
          </button>
        </div>

        <section className="control-section">
          <h2><User size={16} /> Game</h2>
          <label>
            Side
            <select value={humanSide} onChange={(event) => setHumanSide(event.target.value as Side)}>
              <option value="white">White</option>
              <option value="black">Black</option>
            </select>
          </label>
          <label>
            Opening book
            <select value={bookId} onChange={(event) => setBookId(event.target.value)}>
              <option value="">No book seed</option>
              {books.map((book) => (
                <option key={book.book_id} value={book.book_id}>
                  {book.name}{book.curated ? " *" : ""}
                </option>
              ))}
            </select>
          </label>
          <label>
            Book plies
            <input
              type="number"
              min={0}
              max={20}
              value={bookMaxPlies}
              onChange={(event) => setBookMaxPlies(Number(event.target.value))}
            />
          </label>
          <button className="primary-action" onClick={handleCreateGame} disabled={busy}>
            <RotateCcw size={16} /> New game
          </button>
          <label>
            Human move
            <input
              value={humanMoveInput}
              onChange={(event) => setHumanMoveInput(event.target.value)}
              placeholder="e2e4"
              disabled={liveControlsDisabled || hasPendingRecovery || game?.turn !== game?.human_side}
            />
          </label>
          <button
            onClick={handleHumanMoveInput}
            disabled={liveControlsDisabled || hasPendingRecovery || game?.turn !== game?.human_side}
          >
            <User size={16} /> Play move
          </button>
          <button
            onClick={handleLlmMove}
            disabled={liveControlsDisabled || hasPendingRecovery || game?.turn === game?.human_side}
          >
            <Bot size={16} /> Ask LLM
          </button>
          {mode === "play" && game?.pending_recovery ? (
            <div className="recovery-card" aria-live="polite">
              <div>
                <strong>Recovery needed</strong>
                <span>
                  Latest model attempt paused the board at the current position.
                </span>
              </div>
              <div className="recovery-actions">
                <button onClick={() => handleRecovery("retry")} disabled={busy || isInspectingHistoricalPlayMove}>
                  <RotateCcw size={16} /> Retry
                </button>
                <button
                  onClick={() => handleRecovery("retry_with_legal_moves")}
                  disabled={busy || isInspectingHistoricalPlayMove}
                >
                  <Search size={16} /> Retry with legal moves
                </button>
                <button
                  onClick={() => handleRecovery("teacher")}
                  disabled={busy || isInspectingHistoricalPlayMove || !game.last_judgment?.teacher_move_uci}
                >
                  <Activity size={16} /> Teacher move {game.last_judgment?.teacher_move_uci ?? ""}
                </button>
                <button onClick={() => handleRecovery("legal_stub")} disabled={busy || isInspectingHistoricalPlayMove}>
                  <Bot size={16} /> Legal stub
                </button>
              </div>
            </div>
          ) : null}
        </section>

        <section className="control-section">
          <h2><BookOpen size={16} /> Artifacts</h2>
          <label>
            Run directory
            <input
              value={artifactDir}
              onChange={(event) => setArtifactDir(event.target.value)}
              placeholder="path/to/artifacts"
            />
          </label>
          <button onClick={handleLoadArtifacts} disabled={busy}>
            <FileSearch size={16} /> Load run
          </button>
          {loadedRun ? <p className="muted">Loaded {loadedRun}</p> : null}
        </section>
      </aside>

      <section className="board-panel">
        <header className="topbar">
          <div>
            <span className="label">{mode === "play" ? "Live game" : "Rollout review"}</span>
            <strong>{mode === "play" ? game?.game_id ?? "No active game" : loadedRun || "No artifact run"}</strong>
          </div>
          <div className="status-strip">
            {busy ? <Loader2 className="spin" size={16} /> : null}
            {error ? (
              <span className="error">{error}</span>
            ) : (
              <span>
                {mode === "play" ? `Turn: ${game?.turn ?? "white"}` : `${filteredItems.length} rollouts`}
                {health ? ` / LLM: ${health.llm_mode}` : ""}
                {toolStatus ? ` / Stockfish: ${toolStatus.stockfish.available ? "ready" : "off"}` : ""}
              </span>
            )}
          </div>
        </header>

        <div className="board-wrap">
          <Chessboard
            id="chess-llm-board"
            position={boardFen}
            boardOrientation={orientationForSide(humanSide)}
            arePiecesDraggable={mode === "play" && Boolean(game) && game?.turn === game?.human_side && !busy && !isInspectingHistoricalPlayMove && !hasPendingRecovery}
            autoPromoteToQueen={false}
            customSquareStyles={squareStyles}
            isDraggablePiece={handleIsDraggablePiece}
            onPieceDragBegin={handlePieceDragBegin}
            onPieceDragEnd={handlePieceDragEnd}
            onPieceDrop={handleDrop}
            onPromotionCheck={handlePromotionCheck}
            onPromotionPieceSelect={handlePromotionPieceSelect}
            onSquareClick={handleSquareClick}
            promotionDialogVariant="vertical"
            customLightSquareStyle={{ backgroundColor: "#eed9b7" }}
            customDarkSquareStyle={{ backgroundColor: "#66845f" }}
          />
        </div>

        <div className="board-status">
          <span>{boardStatus}</span>
          {isInspectingHistoricalPlayMove ? (
            <button className="resume-action" onClick={resumeLatestPosition}>
              Resume latest
            </button>
          ) : null}
          {pendingPromotion ? (
            <div className="promotion-picker" aria-label="Promotion piece">
              {PROMOTION_CHOICES.map((choice) => (
                <button key={choice.value} onClick={() => handlePromotionChoice(choice.value)}>
                  {choice.label}
                </button>
              ))}
            </div>
          ) : null}
        </div>

        <div className="timeline">
          {mode === "play"
            ? (game?.moves ?? []).map((move, index) => (
                <button
                  key={`${move.move_uci}-${index}`}
                  className={`move-chip ${move.source} ${selectedPlayMoveIndex === index ? "selected" : ""}`}
                  onClick={() => inspectPlayMove(index)}
                >
                  {index + 1}. {moveLabel(move.move_uci)}
                </button>
              ))
            : selectedItem
              ? [{ source: "model", move_uci: selectedItem.rollout.parsed_answer.move_uci ?? "unparsed" }].map((move, index) => (
                  <span key={`${move.move_uci}-${index}`} className={`move-chip ${move.source}`}>
                    {index + 1}. {moveLabel(move.move_uci)}
                  </span>
                ))
              : null}
        </div>
      </section>

      <aside className="inspector">
        {mode === "review" ? (
          <ReviewControls
            filters={filters}
            setFilters={setFilters}
            items={reviewItems}
          />
        ) : null}

        <section className="inspector-section">
          <h2>Reasoning</h2>
          <pre className="reasoning">
            {parsedOutput.reasoning ||
              (mode === "play" && selectedPlayMove && !inspectedRollout
                ? "No model reasoning for this move."
                : "No model output yet.")}
          </pre>
        </section>

        <section className="inspector-section facts">
          <h2>{mode === "play" ? (isInspectingHistoricalPlayMove ? "Historical move" : "Last judgment") : "Annotation"}</h2>
          <Fact
            label="Parsed move"
            value={inspectedRollout?.parsed_answer.move_uci ?? (mode === "play" ? selectedPlayMove?.move_uci : undefined)}
          />
          <Fact label="Legal" value={String(inspectedJudgment?.legal ?? "unknown")} />
          <Fact label="Failure" value={inspectedJudgment?.failure_bucket ?? "none"} />
          <Fact label="Regret" value={formatRegret(inspectedJudgment?.regret_cp)} />
          <Fact label="Teacher" value={inspectedJudgment?.teacher_move_uci} />
          <p className="feedback">{inspectedJudgment?.feedback}</p>
        </section>

        <ToolsPanel
          mode={mode}
          status={toolStatus}
          analysis={toolAnalysis}
          canAnalyze={Boolean(analysisFen)}
          canScore={mode === "review" && Boolean(loadedRun)}
          busy={busy}
          visibleCount={filteredItems.length}
          onAnalyze={handleAnalyzePosition}
          onScoreSelected={handleScoreSelected}
          onScoreVisible={handleScoreVisible}
        />

        {mode === "review" ? (
          <section className="inspector-section rollout-list">
            <h2>Rollouts</h2>
            <div className="list">
              {filteredItems.map((item) => (
                <button
                  key={item.rollout.rollout_id}
                  className={item.rollout.rollout_id === selectedItem?.rollout.rollout_id ? "selected" : ""}
                  onClick={() => {
                    setSelectedRolloutId(item.rollout.rollout_id);
                    setPreviewFen(null);
                  }}
                >
                  <span>{item.rollout.model_id}</span>
                  <strong>{item.rollout.parsed_answer.move_uci ?? "unparsed"}</strong>
                  <small>{item.judgment?.failure_bucket ?? (item.judgment?.legal ? "legal" : "unknown")}</small>
                </button>
              ))}
            </div>
            <div className="preview-actions">
              <button onClick={() => previewMove(selectedItem?.rollout.parsed_answer.move_uci)}>Preview model</button>
              <button onClick={() => previewMove(selectedItem?.judgment?.teacher_move_uci)}>Preview teacher</button>
              <button onClick={() => setPreviewFen(null)}>Reset</button>
            </div>
          </section>
        ) : null}

        <section className="inspector-section">
          <h2>Raw output</h2>
          <pre className="raw-output">{latestOutput || "No raw output."}</pre>
        </section>
      </aside>
    </main>
  );
}
