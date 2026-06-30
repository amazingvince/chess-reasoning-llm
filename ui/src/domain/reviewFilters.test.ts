import { describe, expect, it } from "vitest";

import { filterRolloutItems } from "./reviewFilters";
import type { RolloutReviewItem } from "./types";

const items: RolloutReviewItem[] = [
  {
    prompt: {
      prompt_id: "prompt-1",
      fen: "start",
      task_type: "best_move",
      metadata: { split: "planning" },
      messages: [{ role: "user", content: "Find a move" }]
    },
    rollout: {
      rollout_id: "rollout-1",
      prompt_id: "prompt-1",
      model_id: "model-a",
      raw_output: "<move>e2e4</move>",
      parsed_answer: { move_uci: "e2e4", format_type: "move_tag", parse_error: null }
    },
    judgment: {
      judgment_id: "judgment-1",
      rollout_id: "rollout-1",
      legal: true,
      regret_cp: 12,
      failure_bucket: null,
      teacher_move_uci: "e2e4",
      feedback: "Good move."
    }
  },
  {
    prompt: {
      prompt_id: "prompt-2",
      fen: "start",
      task_type: "best_move",
      metadata: { split: "rules" },
      messages: [{ role: "user", content: "Find a legal move" }]
    },
    rollout: {
      rollout_id: "rollout-2",
      prompt_id: "prompt-2",
      model_id: "model-b",
      raw_output: "bad",
      parsed_answer: { move_uci: null, format_type: "none", parse_error: "no UCI move found" }
    },
    judgment: {
      judgment_id: "judgment-2",
      rollout_id: "rollout-2",
      legal: false,
      regret_cp: null,
      failure_bucket: "parse_failure",
      teacher_move_uci: null,
      feedback: "Could not parse."
    }
  }
];

describe("filterRolloutItems", () => {
  it("filters by model, legal status, failure bucket, regret, split, and search text", () => {
    const filtered = filterRolloutItems(items, {
      model: "model-a",
      legal: "legal",
      failureBucket: "none",
      split: "planning",
      minRegret: 10,
      maxRegret: 20,
      search: "good"
    });

    expect(filtered.map((item) => item.rollout.rollout_id)).toEqual(["rollout-1"]);
  });
});
