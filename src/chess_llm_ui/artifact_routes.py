"""Artifact review routes for the workbench backend."""

from __future__ import annotations

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from chess_llm_ui.artifact_review import (
    ArtifactLoadError,
    ArtifactRun,
    JoinedRollout,
    load_artifact_run,
)
from chess_llm_ui.config import BackendSettings


class LoadArtifactsRequest(BaseModel):
    artifact_dir: str


class ScoreArtifactsRequest(BaseModel):
    rollout_ids: list[str]
    depth: int | None = Field(default=None, ge=1)


def register_artifact_routes(app: FastAPI, settings: BackendSettings) -> None:
    """Register read/score routes for persisted rollout artifacts."""

    @app.post("/api/artifacts/load")
    def load_artifacts(request_payload: LoadArtifactsRequest) -> dict:
        try:
            run = load_artifact_run(request_payload.artifact_dir)
        except FileNotFoundError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except ArtifactLoadError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        app.state.artifact_runs[run.run_id] = run
        return {
            "run_id": run.run_id,
            "artifact_dir": str(run.artifact_dir),
            "count": len(run.items),
        }

    @app.get("/api/artifacts/{run_id}/rollouts")
    def list_rollouts(run_id: str) -> dict:
        run = _get_artifact_run(app, run_id)
        return {"items": [item.to_dict() for item in run.items]}

    @app.get("/api/artifacts/{run_id}/rollouts/{rollout_id}")
    def get_rollout(run_id: str, rollout_id: str) -> dict:
        run = _get_artifact_run(app, run_id)
        for item in run.items:
            if item.rollout.rollout_id == rollout_id:
                return item.to_dict()
        raise HTTPException(status_code=404, detail=f"unknown rollout_id: {rollout_id}")

    @app.post("/api/artifacts/{run_id}/score")
    def score_rollouts(run_id: str, request_payload: ScoreArtifactsRequest) -> dict:
        if not request_payload.rollout_ids:
            raise HTTPException(status_code=400, detail="rollout_ids must not be empty")
        if len(request_payload.rollout_ids) > settings.tool_batch_limit:
            raise HTTPException(
                status_code=400,
                detail=(
                    "rollout_ids exceed configured batch limit "
                    f"({settings.tool_batch_limit})"
                ),
            )

        run = _get_artifact_run(app, run_id)
        requested_ids = set(request_payload.rollout_ids)
        scored_ids: set[str] = set()
        errors: list[dict[str, str]] = []
        updated_items: list[JoinedRollout] = []

        for item in run.items:
            if item.rollout.rollout_id not in requested_ids:
                updated_items.append(item)
                continue

            if item.prompt is None:
                updated_items.append(item)
                continue

            try:
                judgment = app.state.tool_service.judge_rollout(
                    item.prompt,
                    item.rollout,
                    metadata={
                        "source": "ui_review",
                        "run_id": run_id,
                    },
                    depth=request_payload.depth,
                )
            except Exception as exc:
                updated_items.append(item)
                errors.append({"rollout_id": item.rollout.rollout_id, "error": str(exc)})
                continue

            updated_items.append(
                JoinedRollout(
                    prompt=item.prompt,
                    rollout=item.rollout,
                    judgment=judgment,
                )
            )
            scored_ids.add(item.rollout.rollout_id)

        run.items = updated_items
        skipped_count = len(requested_ids - scored_ids)
        return {
            "items": [
                item.to_dict()
                for item in run.items
                if item.rollout.rollout_id in requested_ids
            ],
            "scored_count": len(scored_ids),
            "skipped_count": skipped_count,
            "error_count": len(errors),
            "errors": errors,
            "tool_status": app.state.tool_service.status(),
        }


def _get_artifact_run(app: FastAPI, run_id: str) -> ArtifactRun:
    run = app.state.artifact_runs.get(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"unknown run_id: {run_id}")
    return run

