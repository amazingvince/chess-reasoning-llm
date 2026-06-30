import { beforeEach, describe, expect, it } from "vitest";

import {
  clearActiveGameId,
  readActiveGameId,
  writeActiveGameId
} from "./activeGameStorage";

describe("active game storage", () => {
  beforeEach(() => {
    window.localStorage.clear();
  });

  it("stores and reads only a game id", () => {
    writeActiveGameId("game-abc123");

    expect(readActiveGameId()).toBe("game-abc123");
  });

  it("treats blank values as missing", () => {
    window.localStorage.setItem("chess-llm-workbench.active-game-id", "   ");

    expect(readActiveGameId()).toBeNull();
  });

  it("clears the stored game id", () => {
    writeActiveGameId("game-abc123");
    clearActiveGameId();

    expect(readActiveGameId()).toBeNull();
  });
});
