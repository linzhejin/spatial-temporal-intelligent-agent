"""Run one synthetic D1 active/expired pair through the product Agent.

This is a paid development diagnostic in an isolated database. The event is
patched into this process only and is never published as a real campus event.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from unittest.mock import patch

from dotenv import dotenv_values


ROOT = Path(__file__).resolve().parents[2]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _edges(route: dict) -> list[list[str]]:
    return [[str(part) for part in edge] for edge in route.get("recommended_edge_ids") or []]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--source-link-id", required=True)
    parser.add_argument("--env-file", type=Path, required=True)
    parser.add_argument("--test-database-url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--allow-model-call", action="store_true")
    args = parser.parse_args()
    if not args.allow_model_call:
        raise SystemExit("explicit --allow-model-call is required")
    if args.output.exists():
        raise FileExistsError(args.output)
    bank = json.loads(args.bank.read_text(encoding="utf-8"))
    if bank.get("format") != "synthetic-road-event-bank-v1" or bank.get("split") != "development":
        raise ValueError("only a D1 development bank may be used")
    graph_path = ROOT / "data/whu_road_network.graphml"
    if _sha256(graph_path) != bank.get("graph_sha256"):
        raise ValueError("D1 bank graph differs from product graph")
    cases = [case for case in bank["cases"]
             if case["source_link_id"] == args.source_link_id]
    if len(cases) != 2 or {case["id"].rsplit("-", 1)[-1] for case in cases} != {"active", "expired"}:
        raise ValueError("selected source link needs one active and one expired case")

    env = dotenv_values(args.env_file)
    from tests.db_safety import verify_isolated_test_database
    verify_isolated_test_database(args.test_database_url,
                                  business_urls=(env.get("DATABASE_URL"),))
    key = env.get("DEEPSEEK_API_KEY")
    if not key:
        raise SystemExit("DeepSeek credential missing from selected local config")
    os.environ["DATABASE_URL"] = args.test_database_url
    os.environ["DEEPSEEK_API_KEY"] = key
    os.environ["OPENAI_BASE_URL"] = env.get("OPENAI_BASE_URL") or "https://api.deepseek.com/v1"
    os.environ["LLM_MODEL"] = env.get("LLM_MODEL") or "deepseek-chat"

    from agents import tools
    from agents.planner import run_agent
    from spatial.network import _cache_path
    if Path(_cache_path()).resolve() != graph_path.resolve():
        raise ValueError("product graph cache path differs from D1 input")
    graph = tools._ensure_graph()
    rows = []
    for case in sorted(cases, key=lambda item: item["as_of_epoch"]):
        start, end = int(case["start_node"]), int(case["end_node"])
        event = case["event"]
        active = event["start_time"] <= case["as_of_epoch"] < event["end_time"]
        conditions = [{**event, "id": args.source_link_id, "type": "event",
                       "name": "模拟道路封闭"}] if active else []
        origin = {"name": "地图起点", "lng": graph.nodes[start]["x"],
                  "lat": graph.nodes[start]["y"]}
        destination = {"name": "地图终点", "lng": graph.nodes[end]["x"],
                       "lat": graph.nodes[end]["y"]}
        tool_trace: list[dict] = []
        model_trace: list[dict] = []
        with patch.object(tools, "list_conditions", return_value=conditions):
            response = run_agent(
                "这是模拟道路事件测试。从地图起点步行到地图终点，走可通行的最短路线；说明当前有无模拟管制。",
                coord_start=origin, coord_end=destination, travel_mode="walk",
                tool_trace_callback=tool_trace.append,
                model_trace_callback=model_trace.append,
            )
        route = response.get("route") if isinstance(response.get("route"), dict) else {}
        edges = _edges(route)
        direct = [case["start_node"], case["end_node"], str(event["edge"]["key"])]
        reverse = [direct[1], direct[0], direct[2]]
        expected = case["expected_graph_effect"]
        if active:
            correct = bool(edges and direct not in edges and reverse not in edges)
        else:
            correct = edges == [expected["direct_edge_id"]]
        rows.append({
            "case_id": case["id"], "source_link_id": args.source_link_id,
            "source_spatial_truth": case["source_spatial_truth"],
            "campus_road_assignment": case["campus_road_assignment"],
            "event_active": active, "response_kind": response.get("response_kind"),
            "message": response.get("message"), "route_edge_ids": edges,
            "road_conditions_applied": route.get("road_conditions_applied"),
            "recommended_length_m": route.get("recommended_length_m"),
            "route_verification_status": (route.get("route_verification") or {}).get("status"),
            "direct_edge_rule_pass": correct,
            "tool_trace": tool_trace, "model_trace": model_trace,
        })
    report = {
        "status": "development_agent_pilot_only_not_vision_or_field_score",
        "bank_sha256": _sha256(args.bank), "graph_sha256": _sha256(graph_path),
        "source_link_id": args.source_link_id, "runs": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    passed = sum(row["direct_edge_rule_pass"]
                 and row["route_verification_status"] == "pass" for row in rows)
    print(json.dumps({"runs": len(rows), "route_rule_pass": passed,
                      "status": report["status"]}, ensure_ascii=False))
    return 0 if passed == len(rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
