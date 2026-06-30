import type { ParsedModelOutput } from "./types";

const THINK_RE = /<think>\s*([\s\S]*?)\s*<\/think>/i;
const MOVE_RE = /<move>\s*([a-h][1-8][a-h][1-8][qrbn]?)\s*<\/move>/i;

export function splitModelOutput(rawOutput: string | null | undefined): ParsedModelOutput {
  const raw = rawOutput ?? "";
  const thinkMatch = raw.match(THINK_RE);
  const moveMatch = raw.match(MOVE_RE);
  return {
    reasoning: (thinkMatch?.[1] ?? raw).trim(),
    moveUci: moveMatch?.[1]?.toLowerCase() ?? null,
    rawOutput: raw
  };
}
