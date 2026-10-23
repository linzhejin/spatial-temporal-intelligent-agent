"""Replay D1 road fixtures through the product routing tool without a model.

The road binding is hypothetical. This checks tool and event semantics only;
it does not turn a Songdo video into an observed Wuhan road condition.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _edge_ids(route: dict) -> list[list[str]]:
    return [[str(part) for part in edge] for edge in route.get("recommended_edge_ids") or []]


def replay_bank(bank: dict, graph, execute_tool, *, audit_graph=None) -> dict:
    """Check observed product-tool paths against independently built D1 refs."""
    if bank.get("format") != "synthetic-road-event-bank-v1" or bank.get("split") != "development":
        raise ValueError("only a D1 development fixture may be replayed")
    rows = []
    for case in bank["cases"]:
        event = case["event"]
        clock = case["as_of_epoch"]
        active = event["start_time"] <= clock < event["end_time"]
        condition = {**event, "id": case["source_link_id"], "type": "event"}
        conditions = [condition] if active else []
        start, end = int(case["start_node"]), int(case["end_node"])
        args = {
            "start": {"type": "coord", "coordinates": {
                "lng": graph.nodes[start]["x"], "lat": graph.nodes[start]["y"],
            }},
            "end": {"type": "coord", "coordinates": {
                "lng": graph.nodes[end]["x"], "lat": graph.nodes[end]["y"],
            }},
            "mode": case["mode"], "strategy": "shortest",
        }
        result, _ = execute_tool("plan_route", args, {
            "query": "shortest route", "_route_graph_snapshot": graph,
            "_route_conditions_snapshot": conditions,
            "_route_weather_snapshot": None,
        })
        edges = _edge_ids(result)
        expected = case["expected_graph_effect"]
        target = (expected["detour_edge_ids"] if active
                  else [expected["direct_edge_id"]])
        matched = edges == target
        verified = (result.get("route_verification") or {}).get("status") == "pass"
        direct = [str(part) for part in (expected.get("direct_edge_id") or [
            case["start_node"], case["end_node"], event["edge"]["key"]])]
        reverse = [direct[1], direct[0], direct[2]]
        direct_excluded = direct not in edges and reverse not in edges
        independent = None
        if audit_graph is not None:
            from experiments.scenario_v2 import audit_route_edges
            closed = {"|".join(direct), "|".join(reverse)} if active else set()
            independent = audit_route_edges(
                audit_graph, edges, start, end, [], case["mode"], {},
                closed_edges=closed,
            )
        legal = independent is None or independent["route"] == "pass"
        # Multiple legal detours can differ from the independent shortest-path
        # reference because the product applies additional road policy. Do not
        # misclassify a legal avoidance route as a failure for that reason.
        passed = bool(verified and legal and not result.get("error")
                      and (direct_excluded if active else matched))
        rows.append({
            "case_id": case["id"], "source_video_id": case["source_video_id"],
            "source_spatial_truth": case["source_spatial_truth"],
            "campus_road_assignment": case["campus_road_assignment"],
            "as_of_epoch": clock, "event_active": active,
            "expected_edge_ids": target, "observed_edge_ids": edges,
            "route_verification_status": (result.get("route_verification") or {}).get("status"),
            "independent_route_status": independent["route"] if independent else "not_checked",
            "independent_route_issues": independent["issues"] if independent else [],
            "direct_edge_excluded": direct_excluded if active else None,
            "tool_error": result.get("error"), "graph_reference_match": matched,
            "tool_route_pass": passed,
        })
    return {
        "status": "development_tool_replay_only_not_agent_or_vision_score",
        "case_count": len(rows), "tool_route_pass_count": sum(row["tool_route_pass"] for row in rows),
        "cases": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bank", type=Path, required=True)
    parser.add_argument("--graph", type=Path,
                        default=ROOT / "data/whu_road_network.graphml")
    parser.add_argument("--test-database-url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    bank = json.loads(args.bank.read_text(encoding="utf-8"))
    if _sha256(args.graph) != bank.get("graph_sha256"):
        raise ValueError("D1 bank and product graph bytes differ")
    from tests.db_safety import verify_isolated_test_database
    verify_isolated_test_database(args.test_database_url,
                                  business_urls=(os.getenv("DATABASE_URL"),))
    os.environ["DATABASE_URL"] = args.test_database_url
    from agents import tools
    from experiments.scenario_v2 import load_frozen_graph
    from spatial.network import _cache_path
    if Path(_cache_path()).resolve() != args.graph.resolve():
        raise ValueError("product graph cache path differs from D1 bank input")
    graph = tools._ensure_graph()
    audit_graph = load_frozen_graph(
        args.graph, ROOT / "data/edge_overrides.json",
        ROOT / "data/campus_review_decisions.json",
    )
    report = replay_bank(bank, graph, tools.execute_tool, audit_graph=audit_graph)
    report["bank_sha256"] = _sha256(args.bank)
    report["graph_sha256"] = _sha256(args.graph)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                           encoding="utf-8")
    print(json.dumps({"cases": report["case_count"],
                      "tool_route_pass": report["tool_route_pass_count"],
                      "status": report["status"]}, ensure_ascii=False))
    return 0 if report["tool_route_pass_count"] == report["case_count"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
