import { describe, expect, it } from "vitest";

import { splitModelOutput } from "./reasoning";

describe("splitModelOutput", () => {
  it("separates visible think text and move tags", () => {
    const parsed = splitModelOutput("<think>Take the center.</think>\n<move>e2e4</move>");

    expect(parsed.reasoning).toBe("Take the center.");
    expect(parsed.moveUci).toBe("e2e4");
    expect(parsed.rawOutput).toContain("<think>");
  });

  it("falls back to raw output when no think tag is present", () => {
    const parsed = splitModelOutput("I choose e2e4.");

    expect(parsed.reasoning).toBe("I choose e2e4.");
    expect(parsed.moveUci).toBeNull();
  });
});
