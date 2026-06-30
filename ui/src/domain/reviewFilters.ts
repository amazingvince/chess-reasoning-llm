import type { ReviewFilters, RolloutReviewItem } from "./types";

const DEFAULT_FILTERS: ReviewFilters = {
  model: "",
  legal: "all",
  failureBucket: "",
  split: "",
  minRegret: null,
  maxRegret: null,
  search: ""
};

export function defaultReviewFilters(): ReviewFilters {
  return { ...DEFAULT_FILTERS };
}

export function filterRolloutItems(
  items: RolloutReviewItem[],
  filters: Partial<ReviewFilters>
): RolloutReviewItem[] {
  const resolved = { ...DEFAULT_FILTERS, ...filters };
  const search = resolved.search.trim().toLowerCase();

  return items.filter((item) => {
    if (resolved.model && item.rollout.model_id !== resolved.model) {
      return false;
    }
    if (resolved.legal !== "all") {
      const legal = item.judgment?.legal;
      if (resolved.legal === "legal" && legal !== true) {
        return false;
      }
      if (resolved.legal === "illegal" && legal !== false) {
        return false;
      }
      if (resolved.legal === "unknown" && legal !== null && legal !== undefined) {
        return false;
      }
    }
    if (resolved.failureBucket) {
      const bucket = item.judgment?.failure_bucket ?? "none";
      if (bucket !== resolved.failureBucket) {
        return false;
      }
    }
    if (resolved.split) {
      const split = item.prompt?.metadata?.split;
      if (split !== resolved.split) {
        return false;
      }
    }
    const regret = item.judgment?.regret_cp;
    if (resolved.minRegret !== null && (regret === null || regret === undefined || regret < resolved.minRegret)) {
      return false;
    }
    if (resolved.maxRegret !== null && (regret === null || regret === undefined || regret > resolved.maxRegret)) {
      return false;
    }
    if (search) {
      const haystack = [
        item.prompt?.prompt_id,
        item.prompt?.fen,
        item.prompt?.task_type,
        item.rollout.rollout_id,
        item.rollout.model_id,
        item.rollout.raw_output,
        item.rollout.parsed_answer.move_uci,
        item.rollout.parsed_answer.parse_error,
        item.judgment?.failure_bucket,
        item.judgment?.teacher_move_uci,
        item.judgment?.feedback
      ]
        .filter(Boolean)
        .join(" ")
        .toLowerCase();
      if (!haystack.includes(search)) {
        return false;
      }
    }
    return true;
  });
}

export function uniqueModels(items: RolloutReviewItem[]): string[] {
  return Array.from(new Set(items.map((item) => item.rollout.model_id))).sort();
}

export function uniqueBuckets(items: RolloutReviewItem[]): string[] {
  return Array.from(
    new Set(items.map((item) => item.judgment?.failure_bucket ?? "none"))
  ).sort();
}

export function uniqueSplits(items: RolloutReviewItem[]): string[] {
  return Array.from(
    new Set(
      items
        .map((item) => item.prompt?.metadata?.split)
        .filter((split): split is string => typeof split === "string" && split.length > 0)
    )
  ).sort();
}
