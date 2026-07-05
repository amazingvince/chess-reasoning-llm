import { Search } from "lucide-react";

import {
  uniqueBuckets,
  uniqueModels,
  uniqueSplits
} from "../domain/reviewFilters";
import type { ReviewFilters, RolloutReviewItem } from "../domain/types";

export function ReviewControls({
  filters,
  setFilters,
  items
}: {
  filters: ReviewFilters;
  setFilters: (filters: ReviewFilters) => void;
  items: RolloutReviewItem[];
}) {
  const models = uniqueModels(items);
  const buckets = uniqueBuckets(items);
  const splits = uniqueSplits(items);
  return (
    <section className="inspector-section filters">
      <h2><Search size={16} /> Filters</h2>
      <input
        value={filters.search}
        onChange={(event) => setFilters({ ...filters, search: event.target.value })}
        placeholder="Search output, feedback, IDs"
      />
      <div className="filter-grid">
        <select value={filters.model} onChange={(event) => setFilters({ ...filters, model: event.target.value })}>
          <option value="">All models</option>
          {models.map((model) => <option key={model} value={model}>{model}</option>)}
        </select>
        <select value={filters.legal} onChange={(event) => setFilters({ ...filters, legal: event.target.value as ReviewFilters["legal"] })}>
          <option value="all">All legal states</option>
          <option value="legal">Legal</option>
          <option value="illegal">Illegal</option>
          <option value="unknown">Unknown</option>
        </select>
        <select value={filters.failureBucket} onChange={(event) => setFilters({ ...filters, failureBucket: event.target.value })}>
          <option value="">All buckets</option>
          {buckets.map((bucket) => <option key={bucket} value={bucket}>{bucket}</option>)}
        </select>
        <select value={filters.split} onChange={(event) => setFilters({ ...filters, split: event.target.value })}>
          <option value="">All splits</option>
          {splits.map((split) => <option key={split} value={split}>{split}</option>)}
        </select>
        <input
          type="number"
          placeholder="Min regret"
          value={filters.minRegret ?? ""}
          onChange={(event) => setFilters({ ...filters, minRegret: event.target.value ? Number(event.target.value) : null })}
        />
        <input
          type="number"
          placeholder="Max regret"
          value={filters.maxRegret ?? ""}
          onChange={(event) => setFilters({ ...filters, maxRegret: event.target.value ? Number(event.target.value) : null })}
        />
      </div>
    </section>
  );
}

export function Fact({ label, value }: { label: string; value: unknown }) {
  return (
    <div className="fact-row">
      <span>{label}</span>
      <strong>{value === null || value === undefined || value === "" ? "-" : String(value)}</strong>
    </div>
  );
}

export function formatRegret(value: number | null | undefined): string {
  return value === null || value === undefined ? "-" : `${value} cp`;
}
