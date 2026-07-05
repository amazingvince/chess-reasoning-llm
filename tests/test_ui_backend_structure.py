def test_backend_artifact_routes_are_registered_from_route_module():
    from chess_llm_ui.app import create_app
    from chess_llm_ui.artifact_routes import register_artifact_routes

    app = create_app()
    route_paths = {route.path for route in app.routes}

    assert callable(register_artifact_routes)
    assert "/api/artifacts/load" in route_paths
    assert "/api/artifacts/{run_id}/rollouts" in route_paths
    assert "/api/artifacts/{run_id}/rollouts/{rollout_id}" in route_paths
    assert "/api/artifacts/{run_id}/score" in route_paths
