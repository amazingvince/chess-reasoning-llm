import type { Side } from "./types";
import { buildMoveUci } from "./boardInteraction";

export function orientationForSide(side: Side): Side {
  return side;
}

export function normalizeMoveInput(value: string): string {
  return value.trim().toLowerCase();
}

export function moveUciFromDrop(
  fen: string,
  sourceSquare: string,
  targetSquare: string,
  promotion: string = "q",
): string | null {
  return buildMoveUci(fen, sourceSquare, targetSquare, promotion as "q" | "r" | "b" | "n");
}

export function moveLabel(moveUci: string): string {
  if (moveUci.length < 4) {
    return moveUci;
  }
  const promotion = moveUci.slice(4);
  return `${moveUci.slice(0, 2)}-${moveUci.slice(2, 4)}${promotion ? `=${promotion.toUpperCase()}` : ""}`;
}
