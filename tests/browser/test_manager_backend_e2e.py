"""Opt-in browser-to-Flask integration for road-event publication and rerouting.

Run with ``RUN_MANAGER_BACKEND_E2E=1 python -m pytest -q
tests/browser/test_manager_backend_e2e.py``. The event store is confined to
pytest's temporary directory; no business database or production endpoint is used.
"""

from __future__ import annotations

import os
import threading
from urllib.parse import urlsplit

import pytest
from werkzeug.serving import make_server

try:
    from playwright.sync_api import expect, sync_playwright
except ImportError:
    expect = sync_playwright = None


RUN_E2E = os.getenv("RUN_MANAGER_BACKEND_E2E") == "1" and sync_playwright is not None


@pytest.mark.skipif(not RUN_E2E, reason="set RUN_MANAGER_BACKEND_E2E=1 and install Playwright Chromium")
def test_manager_publication_persists_and_user_route_avoids_closed_edge(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_URL", "")
    monkeypatch.setenv("FLASK_ENV", "development")
    monkeypatch.setenv("ROAD_CONDITIONS_FILE", str(tmp_path / "road_conditions.json"))

    import config
    import api.routes as routes
    from app import create_app
    from spatial import network, road_conditions

    monkeypatch.setattr(config, "ROAD_CONDITION_ADMIN_PASSWORD", "local-browser-e2e-only")
    monkeypatch.setattr(config, "ROAD_CONDITION_ADMIN_TOKEN", "")
    (tmp_path / "road_conditions.json").write_text("[]\n", encoding="utf-8")
    monkeypatch.setattr(road_conditions, "_CONDITIONS_FILE", str(tmp_path / "road_conditions.json"))
    monkeypatch.setattr(road_conditions, "_cache", None)
    monkeypatch.setattr(road_conditions, "_cache_mtime", 0.0)
    monkeypatch.setattr(road_conditions, "_cache_error", None)
    monkeypatch.setattr(routes, "_network_initialized", False)
    monkeypatch.setattr(network, "_G", None)

    app = create_app()
    app.config.update(TESTING=True)
    server = make_server("127.0.0.1", 0, app, threaded=True)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    base_url = f"http://127.0.0.1:{server.server_port}"

    try:
        with sync_playwright() as playwright:
            browser = playwright.chromium.launch()
            page = browser.new_page(viewport={"width": 1440, "height": 1100})
            browser_errors = []
            snapped_roads = []
            page.on("pageerror", lambda error: browser_errors.append(str(error)))

            def local_backend_only(route_request):
                if urlsplit(route_request.request.url).hostname in {"127.0.0.1", "localhost"}:
                    route_request.continue_()
                else:
                    route_request.abort()

            page.route("**/*", local_backend_only)
            page.goto(f"{base_url}/manager", wait_until="domcontentloaded")
            expect(page.locator("#login-panel")).to_be_visible()
            page.locator("#password").fill("local-browser-e2e-only")
            page.locator("#login-form button[type='submit']").click()
            expect(page.locator("#workspace")).to_be_visible(timeout=30000)

            page.locator("#event-type").select_option("event")
            page.locator("#event-action").select_option("closure")
            page.locator("#event-mode-walk").check()
            page.locator("#event-name").fill("本地浏览器联动封路验收")
            page.locator("#pick-road").click()
            map_element = page.locator("#manager-map")
            map_box = map_element.bounding_box()
            assert map_box is not None
            with page.expect_response(
                lambda response: "/api/road-conditions/snap" in response.url
                and response.status == 200,
                timeout=30000,
            ) as snap_response_info:
                map_element.click(position={"x": map_box["width"] / 2, "y": map_box["height"] / 2})
            snap_payload = snap_response_info.value.json()
            if snap_payload.get("data", {}).get("snap"):
                snapped_roads.append(snap_payload["data"]["snap"])
            expect(page.locator("#selected-road")).to_contain_text("吸附距离", timeout=30000)
            assert snapped_roads, "browser map selection must reach the real snap API"

            user = app.test_client()
            graph = network.get_network()
            start_node, end_node = snapped_roads[-1]["u"], snapped_roads[-1]["v"]

            def endpoint(node_id):
                return {
                    "type": "coord",
                    "coordinates": {
                        "lng": float(graph.nodes[node_id]["x"]),
                        "lat": float(graph.nodes[node_id]["y"]),
                    },
                }

            baseline = user.post(
                "/api/route",
                json={
                    "start": endpoint(start_node),
                    "end": endpoint(end_node),
                    "travel_mode": "walk",
                    "strategy": "shortest",
                },
            )
            assert baseline.status_code == 200, baseline.get_json()
            baseline_data = baseline.get_json()["data"]
            assert {tuple(edge) for edge in snapped_roads[-1]["edges"]}.intersection(
                tuple(edge) for edge in baseline_data["shortest_edge_ids"]
            )

            page.locator("#confirm-picked-road").click()
            expect(page.locator("#publish-event")).to_be_enabled()
            page.locator("#publish-event").click()
            expect(page.locator("#form-message")).to_contain_text("事件已发布", timeout=30000)

            stored = road_conditions.list_conditions(include_inactive=True, strict=True)
            assert len(stored) == 1
            event = stored[0]
            assert event["name"] == "本地浏览器联动封路验收"
            assert event["action"] == "closure"
            assert event["affected_modes"] == ["walk"]

            public_snapshot = user.get("/api/road-conditions")
            assert public_snapshot.status_code == 200
            snapshot_data = public_snapshot.get_json()["data"]
            assert snapshot_data["count"] == 1
            assert snapshot_data["conditions"][0]["id"] == event["id"]

            route_response = user.post(
                "/api/route/replan",
                json={
                    "route_state": baseline_data["route_state"],
                    "change": {"refresh_road_conditions": True},
                },
            )
            assert route_response.status_code == 200, route_response.get_json()
            route_data = route_response.get_json()["data"]
            assert route_data["route_state"]["road_condition_version"] == snapshot_data["active_event_version"]
            closed_edges = {tuple(edge) for edge in event["edge"]["edges"]}
            shortest_edges = {tuple(edge) for edge in route_data["shortest_edge_ids"]}
            assert not closed_edges.intersection(shortest_edges)

            event_row = page.locator("#event-list .event-row").filter(has_text=event["name"])
            expect(event_row).to_have_count(1, timeout=30000)
            page.once("dialog", lambda dialog: dialog.accept())
            event_row.get_by_role("button", name="结束事件").click()
            expect(event_row.locator(".event-status")).to_have_text("已结束", timeout=30000)

            active_after_end = user.get("/api/road-conditions")
            assert active_after_end.status_code == 200
            assert active_after_end.get_json()["data"]["count"] == 0
            restored_route = user.post(
                "/api/route/replan",
                json={
                    "route_state": baseline_data["route_state"],
                    "change": {"refresh_road_conditions": True},
                },
            )
            assert restored_route.status_code == 200, restored_route.get_json()
            restored_edges = {
                tuple(edge) for edge in restored_route.get_json()["data"]["shortest_edge_ids"]
            }
            assert closed_edges.intersection(restored_edges)

            # Re-create the Flask app/client and clear only the in-process file cache;
            # admin history must recover the ended event from the durable test file.
            monkeypatch.setattr(road_conditions, "_cache", None)
            monkeypatch.setattr(road_conditions, "_cache_mtime", 0.0)
            restarted_app = create_app()
            restarted_client = restarted_app.test_client()
            restarted_login = restarted_client.post(
                "/api/admin/login",
                json={"password": "local-browser-e2e-only"},
            )
            assert restarted_login.status_code == 200
            restarted = restarted_client.get("/api/road-conditions?all=1")
            assert restarted.status_code == 200
            recovered_event = restarted.get_json()["data"]["conditions"][0]
            assert recovered_event["id"] == event["id"]
            assert recovered_event["status"] == "expired"

            assert browser_errors == []
            browser.close()
    finally:
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=5)

