import { Chess, type Square } from "chess.js";
import type { CustomSquareStyles } from "react-chessboard/dist/chessboard/types";

import type { Side } from "./types";

export interface LegalMoveTarget {
  square: Square;
  capture: boolean;
  promotion: boolean;
}

export type PromotionChoice = "q" | "r" | "b" | "n";

export function buildMoveUci(
  fen: string,
  sourceSquare: string,
  targetSquare: string,
  promotion: PromotionChoice = "q",
): string | null {
  try {
    const moves = legalVerboseMoves(fen, sourceSquare);
    const candidates = moves.filter((move) => move.to === targetSquare);
    const move = candidates.find((candidate) => candidate.promotion === promotion) ?? candidates[0];
    return move ? `${move.from}${move.to}${move.promotion ?? ""}` : null;
  } catch {
    return null;
  }
}

export function legalMoveTargets(fen: string, sourceSquare: string): LegalMoveTarget[] {
  try {
    const bySquare = new Map<Square, LegalMoveTarget>();
    for (const move of legalVerboseMoves(fen, sourceSquare)) {
      const existing = bySquare.get(move.to);
      bySquare.set(move.to, {
        square: move.to,
        capture: Boolean(existing?.capture || move.isCapture()),
        promotion: Boolean(existing?.promotion || move.promotion),
      });
    }
    return Array.from(bySquare.values()).sort((left, right) => left.square.localeCompare(right.square));
  } catch {
    return [];
  }
}

export function squareStylesForTargets(
  sourceSquare: string | null,
  targets: LegalMoveTarget[],
): CustomSquareStyles {
  const styles: CustomSquareStyles = {};
  if (sourceSquare) {
    styles[sourceSquare as Square] = {
      boxShadow: "inset 0 0 0 4px rgba(47, 109, 79, 0.48)",
    };
  }
  for (const target of targets) {
    if (target.capture) {
      styles[target.square] = {
        boxShadow: "inset 0 0 0 5px rgba(47, 109, 79, 0.72)",
      };
    } else {
      styles[target.square] = {
        backgroundImage: "radial-gradient(circle, rgba(47, 109, 79, 0.55) 18%, transparent 20%)",
      };
    }
    if (target.promotion) {
      styles[target.square] = {
        ...styles[target.square],
        boxShadow: target.capture
          ? `${styles[target.square]?.boxShadow}, inset 0 0 0 9px rgba(255, 170, 0, 0.55)`
          : "inset 0 0 0 5px rgba(255, 170, 0, 0.7)",
      };
    }
  }
  return styles;
}

export function canDragPiece(
  fen: string,
  sourceSquare: string,
  piece: string,
  humanSide: Side,
  turn: Side,
  busy: boolean,
): boolean {
  if (busy || turn !== humanSide || pieceSide(piece) !== humanSide) {
    return false;
  }
  return legalMoveTargets(fen, sourceSquare).length > 0;
}

export function isPromotionMove(fen: string, sourceSquare: string, targetSquare: string): boolean {
  return legalVerboseMoves(fen, sourceSquare).some(
    (move) => move.to === targetSquare && Boolean(move.promotion),
  );
}

export function promotionChoiceFromPiece(piece: string | undefined): PromotionChoice | null {
  const promotion = piece?.[1]?.toLowerCase();
  return promotion === "q" || promotion === "r" || promotion === "b" || promotion === "n"
    ? promotion
    : null;
}

function legalVerboseMoves(fen: string, sourceSquare: string) {
  const chess = new Chess(fen);
  return chess.moves({ square: sourceSquare as Square, verbose: true });
}

function pieceSide(piece: string): Side | null {
  if (piece.startsWith("w")) {
    return "white";
  }
  if (piece.startsWith("b")) {
    return "black";
  }
  return null;
}
