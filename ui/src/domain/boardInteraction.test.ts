import { describe, expect, it } from "vitest";

import {
  buildMoveUci,
  canDragPiece,
  isPromotionMove,
  legalMoveTargets,
  promotionChoiceFromPiece,
  squareStylesForTargets
} from "./boardInteraction";

const QUEEN_CAPTURE_FEN = "r2qkb1r/2pb1ppp/2Qp1n2/p3N3/2P1P3/8/PP1P1PPP/RNB1KB1R w KQkq - 1 8";
const WHITE_PROMOTION_FEN = "7k/P7/8/8/8/8/8/7K w - - 0 1";
const BLACK_PROMOTION_FEN = "7k/8/8/8/8/8/7p/K7 b - - 0 1";

describe("buildMoveUci", () => {
  it("uses chess.js legal moves for non-pawn back-rank captures", () => {
    expect(buildMoveUci(QUEEN_CAPTURE_FEN, "c6", "a8")).toBe("c6a8");
  });

  it("supports selecting each promotion piece", () => {
    expect(buildMoveUci(WHITE_PROMOTION_FEN, "a7", "a8", "q")).toBe("a7a8q");
    expect(buildMoveUci(WHITE_PROMOTION_FEN, "a7", "a8", "r")).toBe("a7a8r");
    expect(buildMoveUci(WHITE_PROMOTION_FEN, "a7", "a8", "b")).toBe("a7a8b");
    expect(buildMoveUci(WHITE_PROMOTION_FEN, "a7", "a8", "n")).toBe("a7a8n");
    expect(buildMoveUci(BLACK_PROMOTION_FEN, "h2", "h1", "n")).toBe("h2h1n");
  });

  it("returns null for illegal drops", () => {
    expect(buildMoveUci(QUEEN_CAPTURE_FEN, "c6", "c8")).toBeNull();
  });
});

describe("legalMoveTargets", () => {
  it("marks captures and promotions from verbose chess.js moves", () => {
    expect(legalMoveTargets(QUEEN_CAPTURE_FEN, "c6")).toContainEqual({
      square: "a8",
      capture: true,
      promotion: false
    });
    expect(legalMoveTargets(WHITE_PROMOTION_FEN, "a7")).toEqual([
      { square: "a8", capture: false, promotion: true }
    ]);
  });
});

describe("squareStylesForTargets", () => {
  it("styles source, quiet moves, captures, and promotion targets distinctly", () => {
    const styles = squareStylesForTargets("a7", [
      { square: "a8", capture: false, promotion: true },
      { square: "b8", capture: true, promotion: true },
      { square: "a6", capture: false, promotion: false }
    ]);

    expect(styles.a7?.boxShadow).toContain("inset");
    expect(styles.a6?.backgroundImage).toContain("radial-gradient");
    expect(styles.a8?.boxShadow).toContain("255, 170, 0");
    expect(styles.b8?.boxShadow).toContain("47, 109, 79");
  });
});

describe("isPromotionMove", () => {
  it("detects only legal promotion moves", () => {
    expect(isPromotionMove(WHITE_PROMOTION_FEN, "a7", "a8")).toBe(true);
    expect(isPromotionMove(QUEEN_CAPTURE_FEN, "c6", "a8")).toBe(false);
  });
});

describe("promotionChoiceFromPiece", () => {
  it("converts react-chessboard promotion piece options to chess.js promotion symbols", () => {
    expect(promotionChoiceFromPiece("wQ")).toBe("q");
    expect(promotionChoiceFromPiece("wR")).toBe("r");
    expect(promotionChoiceFromPiece("bB")).toBe("b");
    expect(promotionChoiceFromPiece("bN")).toBe("n");
    expect(promotionChoiceFromPiece(undefined)).toBeNull();
  });
});

describe("canDragPiece", () => {
  it("allows only the human side on the human turn and with legal moves", () => {
    expect(canDragPiece(QUEEN_CAPTURE_FEN, "c6", "wQ", "white", "white", false)).toBe(true);
    expect(canDragPiece(QUEEN_CAPTURE_FEN, "a8", "bR", "white", "white", false)).toBe(false);
    expect(canDragPiece(QUEEN_CAPTURE_FEN, "c6", "wQ", "white", "black", false)).toBe(false);
    expect(canDragPiece(QUEEN_CAPTURE_FEN, "c6", "wQ", "white", "white", true)).toBe(false);
  });
});
