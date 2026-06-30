import { describe, expect, it } from "vitest";

import { moveUciFromDrop, normalizeMoveInput, orientationForSide } from "./game";

describe("orientationForSide", () => {
  it("uses the human side as the board orientation", () => {
    expect(orientationForSide("white")).toBe("white");
    expect(orientationForSide("black")).toBe("black");
  });
});

describe("normalizeMoveInput", () => {
  it("trims and lowercases UCI input", () => {
    expect(normalizeMoveInput(" E2E4 ")).toBe("e2e4");
  });
});

describe("moveUciFromDrop", () => {
  const queenCaptureFen = "r2qkb1r/2pb1ppp/2Qp1n2/p3N3/2P1P3/8/PP1P1PPP/RNB1KB1R w KQkq - 1 8";
  const whitePromotionFen = "7k/P7/8/8/8/8/8/7K w - - 0 1";
  const blackPromotionFen = "7k/8/8/8/8/8/7p/K7 b - - 0 1";

  it("does not append promotion notation for non-pawn moves to the back rank", () => {
    expect(moveUciFromDrop(queenCaptureFen, "c6", "a8")).toBe("c6a8");
  });

  it("appends queen promotion notation for pawn moves to the back rank", () => {
    expect(moveUciFromDrop(whitePromotionFen, "a7", "a8")).toBe("a7a8q");
    expect(moveUciFromDrop(blackPromotionFen, "h2", "h1")).toBe("h2h1q");
  });

  it("returns null for illegal drops", () => {
    expect(moveUciFromDrop(queenCaptureFen, "c6", "c8")).toBeNull();
  });
});
