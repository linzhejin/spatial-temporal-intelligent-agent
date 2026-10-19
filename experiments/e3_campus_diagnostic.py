"""Development-only E3 diagnostics on the checked-in campus graph.

Synthetic edge changes exercise the production conservative reuse predicate
against a fresh NetworkX Dijkstra oracle. This is not a formal experiment or
evidence about real road events, field access, or production latency.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import time

import networkx as nx

from agents.execution_dependencies import (
    can_reuse_exact_shortest_path,
    can_reuse_exact_shortest_path_after_closure_delta,
    effective_graph_delta,
)


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
REFERENCE_SAMPLES = DATA_DIR / "route_reference_samples.json"
_VARIANT_NAMES = (
    "off_path_delete",
    "off_path_increase",
    "on_path_delete",
    "on_path_increase",
    "open_previous_closure",
    "off_path_decrease",
    "mixed_off_path_closure_and_slowdown",
)


def _length(attrs: dict) -> float:
    value = float(attrs["length"])
    if not math.isfinite(value) or value < 0:
        raise ValueError("edge length must be finite and nonnegative")
    return value


def _shortest_route(graph, start: int, end: int) -> dict | None:
    """Return a deterministic concrete edge sequence for a MultiDiGraph path."""
    try:
        nodes = nx.shortest_path(
            graph, start, end,
            weight=lambda _u, _v, keyed: min(_length(attrs) for attrs in keyed.values()),
            method="dijkstra",
        )
    except (nx.NetworkXNoPath, nx.NodeNotFound):
        return None

    edge_ids = []
    total = 0.0
    for u, v in zip(nodes, nodes[1:]):
        choices = [(_length(attrs), repr(key), key)
                   for key, attrs in graph.get_edge_data(u, v).items()]
        cost, _sort_key, selected_key = min(choices)
        edge_ids.append([u, v, selected_key])
        total += cost
    return {"nodes": nodes, "edge_ids": edge_ids, "cost_m": total}


def evaluate_graph_change(
    old_graph, new_graph, start: int, end: int, *,
    strategy: str = "shortest", same_mode_constraints_and_endpoints: bool = True,
) -> dict:
    """Apply the production reuse predicate and compare its choice to Dijkstra."""
    old_route = _shortest_route(old_graph, start, end)
    if old_route is None:
        raise ValueError("old graph must contain a route for this diagnostic")
    decision = can_reuse_exact_shortest_path(
        old_effective_graph=old_graph,
        new_effective_graph=new_graph,
        old_edge_ids=old_route["edge_ids"],
        start_node=start,
        end_node=end,
        strategy=strategy,
        old_exact_optimality_proven=True,
        same_mode_constraints_and_endpoints=same_mode_constraints_and_endpoints,
    )
    oracle = _shortest_route(new_graph, start, end)
    chosen = old_route if decision["reuse_path"] else oracle
    return {
        "reuse_path": decision["reuse_path"],
        "reason": decision["reason"],
        "chosen_edge_ids": chosen["edge_ids"] if chosen else None,
        "chosen_cost_m": chosen["cost_m"] if chosen else None,
        "oracle_edge_ids": oracle["edge_ids"] if oracle else None,
        "oracle_cost_m": oracle["cost_m"] if oracle else None,
        "matches_oracle": (
            chosen is None and oracle is None
            or chosen is not None and oracle is not None
            and math.isclose(chosen["cost_m"], oracle["cost_m"], rel_tol=1e-10, abs_tol=1e-6)
        ),
        "change_set": decision.get("change_set"),
    }


def _haversine_m(lng1, lat1, lng2, lat2) -> float:
    radius = 6_371_000.0
    lat1, lat2 = math.radians(lat1), math.radians(lat2)
    dlat, dlng = lat2 - lat1, math.radians(lng2 - lng1)
    term = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlng / 2) ** 2
    return radius * 2 * math.atan2(math.sqrt(term), math.sqrt(1 - term))


def _poi_coordinates(poi: dict) -> tuple[float, float]:
    coordinates = poi.get("coordinates") or {}
    lng = coordinates.get("lng", poi.get("lng"))
    lat = coordinates.get("lat", poi.get("lat"))
    if lng is None or lat is None:
        raise ValueError(f"POI has no coordinates: {poi.get('name')}")
    return float(lng), float(lat)


def _source_hashes() -> dict[str, str]:
    paths = (
        DATA_DIR / "whu_road_network.graphml",
        DATA_DIR / "road_annotations.json",
        DATA_DIR / "edge_overrides.json",
        DATA_DIR / "campus_review_decisions.json",
        DATA_DIR / "road_conditions.json",
        DATA_DIR / "pois.json",
        REFERENCE_SAMPLES,
        Path(__file__).resolve(),
        ROOT / "agents" / "execution_dependencies.py",
        ROOT / "agents" / "route_reuse.py",
        ROOT / "spatial" / "road_conditions.py",
        ROOT / "spatial" / "routing_index.py",
        ROOT / "spatial" / "network.py",
    )
    return {
        str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in paths if path.is_file()
    }


def _git_metadata() -> dict:
    def run(*args):
        result = subprocess.run(
            ["git", *args], cwd=ROOT, check=False, capture_output=True, text=True,
        )
        return result.stdout.strip() if result.returncode == 0 else None

    return {
        "head": run("rev-parse", "HEAD"),
        "tracked_worktree_dirty": bool(run("status", "--porcelain", "--untracked-files=no")),
    }


def _make_variants(graph, route: dict) -> dict[str, tuple]:
    path_edges = {tuple(edge) for edge in route["edge_ids"]}
    off_path = [
        (u, v, key) for u, v, key, attrs in graph.edges(keys=True, data=True)
        if (u, v, key) not in path_edges and "length" in attrs
    ]
    if len(off_path) < 2 or not route["edge_ids"]:
        raise ValueError("campus graph lacks enough route/off-route edges for variants")
    off_a, off_b = off_path[:2]
    on_path = tuple(route["edge_ids"][0])

    def changed(name):
        old, new = graph.copy(), graph.copy()
        change = {
            "changed_edge_ids": [],
            "previously_closed_edge_ids": [],
            "current_closed_edge_ids": [],
            "requires_full_search": False,
            "full_search_reason": None,
        }
        if name == "off_path_delete":
            new.remove_edge(*off_a)
            change["current_closed_edge_ids"] = [list(off_a)]
            change["changed_edge_ids"] = [list(off_a)]
        elif name == "off_path_increase":
            new[off_a[0]][off_a[1]][off_a[2]]["length"] = _length(new[off_a[0]][off_a[1]][off_a[2]]) * 2
            change["requires_full_search"] = True
            change["full_search_reason"] = "road_slowdown_requires_full_search"
            change["changed_edge_ids"] = [list(off_a)]
        elif name == "on_path_delete":
            new.remove_edge(*on_path)
            change["current_closed_edge_ids"] = [list(on_path)]
            change["changed_edge_ids"] = [list(on_path)]
        elif name == "on_path_increase":
            new[on_path[0]][on_path[1]][on_path[2]]["length"] = _length(new[on_path[0]][on_path[1]][on_path[2]]) * 2
            change["requires_full_search"] = True
            change["full_search_reason"] = "road_slowdown_requires_full_search"
            change["changed_edge_ids"] = [list(on_path)]
        elif name == "open_previous_closure":
            old.remove_edge(*off_a)
            change["previously_closed_edge_ids"] = [list(off_a)]
            change["changed_edge_ids"] = [list(off_a)]
        elif name == "off_path_decrease":
            attrs = new[off_a[0]][off_a[1]][off_a[2]]
            attrs["length"] = max(_length(attrs) * 0.5, 0.01)
            change["requires_full_search"] = True
            change["full_search_reason"] = "edge_cost_decreased"
            change["changed_edge_ids"] = [list(off_a)]
        elif name == "mixed_off_path_closure_and_slowdown":
            new.remove_edge(*off_a)
            attrs = new[off_b[0]][off_b[1]][off_b[2]]
            attrs["length"] = _length(attrs) * 2
            change["current_closed_edge_ids"] = [list(off_a)]
            change["requires_full_search"] = True
            change["full_search_reason"] = "road_slowdown_requires_full_search"
            change["changed_edge_ids"] = [list(off_a), list(off_b)]
        else:
            raise ValueError(f"unknown variant: {name}")
        return old, new, change

    return {name: changed(name) for name in _VARIANT_NAMES}


def _select_production_decision(
    change: dict, *, old_edge_ids, start_node, end_node,
) -> dict:
    """Apply production's conservative policy to one synthetic variant."""
    if change["requires_full_search"]:
        return {
            "reuse_path": False,
            "reason": change["full_search_reason"],
            "change_set": None,
        }
    return can_reuse_exact_shortest_path_after_closure_delta(
        old_edge_ids=old_edge_ids,
        start_node=start_node,
        end_node=end_node,
        previously_closed_edge_ids=change["previously_closed_edge_ids"],
        current_closed_edge_ids=change["current_closed_edge_ids"],
        strategy="shortest",
        old_exact_optimality_proven=True,
        same_mode_constraints_and_endpoints=True,
    )


def _median_ms(fn, repeats: int = 5) -> float:
    durations = []
    for _ in range(repeats):
        started = time.perf_counter()
        fn()
        durations.append((time.perf_counter() - started) * 1000)
    return round(statistics.median(durations), 4)


def run_campus_diagnostic() -> dict:
    from spatial import network
    from spatial.coord_transform import gcj02_to_wgs84
    from spatial.poi import get_poi
    from spatial.routing_index import get_routing_index

    if not (DATA_DIR / "whu_road_network.graphml").is_file():
        raise FileNotFoundError("checked-in campus graph is required; no network download is allowed")
    graph = network.load_or_download_network()
    walk_graph = get_routing_index(graph).for_mode("walk").graph
    samples = json.loads(REFERENCE_SAMPLES.read_text(encoding="utf-8"))["samples"]
    rows = []
    for sample in samples:
        start_poi = get_poi(sample["start"], fuzzy=False)
        end_poi = get_poi(sample["end"], fuzzy=False)
        if not start_poi or not end_poi:
            rows.append({"sample_id": sample["id"], "status": "missing_poi"})
            continue
        start_lng, start_lat = gcj02_to_wgs84(*_poi_coordinates(start_poi))
        end_lng, end_lat = gcj02_to_wgs84(*_poi_coordinates(end_poi))
        start_node = network.get_nearest_node(walk_graph, start_lng, start_lat)
        end_node = network.get_nearest_node(walk_graph, end_lng, end_lat)
        start_coord = network.get_node_coords(walk_graph, start_node)
        end_coord = network.get_node_coords(walk_graph, end_node)
        base_route = _shortest_route(walk_graph, start_node, end_node)
        if base_route is None:
            rows.append({
                "sample_id": sample["id"], "status": "unreachable_after_node_snap",
                "start_node": start_node, "end_node": end_node,
            })
            continue

        variants = []
        for name, (old_graph, new_graph, change) in _make_variants(walk_graph, base_route).items():
            reference = evaluate_graph_change(old_graph, new_graph, start_node, end_node)
            if change["requires_full_search"]:
                production_decision = _select_production_decision(
                    change,
                    old_edge_ids=base_route["edge_ids"],
                    start_node=start_node,
                    end_node=end_node,
                )
                closure_assessment_ms = None
            else:
                def closure_assessment():
                    return _select_production_decision(
                        change,
                        old_edge_ids=base_route["edge_ids"],
                        start_node=start_node,
                        end_node=end_node,
                    )

                production_decision = closure_assessment()
                closure_assessment_ms = _median_ms(closure_assessment)

            oracle = _shortest_route(new_graph, start_node, end_node)
            chosen = base_route if production_decision["reuse_path"] else oracle
            production_matches_oracle = (
                chosen is None and oracle is None
                or chosen is not None and oracle is not None
                and math.isclose(chosen["cost_m"], oracle["cost_m"], rel_tol=1e-10, abs_tol=1e-6)
            )
            variant = {
                "variant": name,
                "synthetic_changed_edge_ids": change["changed_edge_ids"],
                "production_decision": production_decision,
                "production_path_reused": production_decision["reuse_path"],
                "production_reason": production_decision["reason"],
                "chosen_cost_m": chosen["cost_m"] if chosen else None,
                "oracle_cost_m": oracle["cost_m"] if oracle else None,
                "matches_oracle": production_matches_oracle,
                "reference_full_graph_delta_decision": {
                    "reuse_path": reference["reuse_path"],
                    "reason": reference["reason"],
                    "matches_oracle": reference["matches_oracle"],
                },
                "change_set_counts": {
                    key: len(production_decision["change_set"].get(key, []))
                    for key in ("added_edge_ids", "removed_edge_ids", "cost_increases", "cost_decreases")
                } if production_decision["change_set"] else None,
                "legacy_full_graph_delta_median_ms": _median_ms(lambda: effective_graph_delta(
                    old_effective_graph=old_graph,
                    new_effective_graph=new_graph,
                )),
                "closure_set_assessment_median_ms": closure_assessment_ms,
                "full_dijkstra_median_ms": _median_ms(lambda: _shortest_route(new_graph, start_node, end_node)),
            }
            variants.append(variant)

        rows.append({
            "sample_id": sample["id"],
            "scenario": sample["scenario"],
            "status": "evaluated",
            "poi_ids": [start_poi.get("id"), end_poi.get("id")],
            "snapped_nodes": [start_node, end_node],
            "snap_distance_m": [
                round(_haversine_m(start_lng, start_lat, *start_coord), 2),
                round(_haversine_m(end_lng, end_lat, *end_coord), 2),
            ],
            "baseline_route_length_m": round(base_route["cost_m"], 2),
            "baseline_route_edge_count": len(base_route["edge_ids"]),
            "variants": variants,
        })

    evaluated = [row for row in rows if row["status"] == "evaluated"]
    variants = [variant for row in evaluated for variant in row["variants"]]
    expected_reuse = {"off_path_delete"}
    policy_errors = [
        {"sample_id": row["sample_id"], "variant": variant["variant"]}
        for row in evaluated for variant in row["variants"]
        if variant["production_path_reused"] != (variant["variant"] in expected_reuse)
    ]
    return {
        "report_version": "e3-campus-development-diagnostic-v3",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_scope": "development_diagnostic_only",
        "formal_e3_result": False,
        "limitations": [
            "all road changes in this report are synthetic graph perturbations, not observed campus conditions",
            "POI coordinates are snapped to the nearest walk node; this does not validate entrances or field accessibility",
            "timings compare the legacy full graph-delta scan, the closure-set predicate, and one Dijkstra call; they omit authentication, database proof access, event decoding, graph construction, route verification, and production latency",
            "networkx timing is a local diagnostic; no node-expansion count is collected",
            "the reference-route sample file has no independent field truth for these OD pairs",
        ],
        "dataset_hashes": _source_hashes(),
        "git": _git_metadata(),
        "graph": {
            "node_count": graph.number_of_nodes(),
            "directed_edge_count": graph.number_of_edges(),
            "walk_node_count": walk_graph.number_of_nodes(),
            "walk_directed_edge_count": walk_graph.number_of_edges(),
            "condition_file_active_records": len(json.loads((DATA_DIR / "road_conditions.json").read_text(encoding="utf-8"))),
        },
        "summary": {
            "sample_count": len(rows),
            "evaluated_od_count": len(evaluated),
            "synthetic_change_checks": len(variants),
            "reuse_decisions": sum(bool(item["production_path_reused"]) for item in variants),
            "full_search_decisions": sum(not item["production_path_reused"] for item in variants),
            "oracle_cost_mismatches": sum(not item["matches_oracle"] for item in variants),
            "policy_errors": policy_errors,
            "node_expansion_measurement": None,
        },
        "samples": rows,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="new JSON report path; existing files are never overwritten")
    args = parser.parse_args(argv)
    report = run_campus_diagnostic()
    output = args.output or (
        ROOT / "output" / "experiments" /
        f"e3_campus_dev_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    )
    if output.exists():
        parser.error(f"refusing to overwrite existing report: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(output), "summary": report["summary"]}, ensure_ascii=False, indent=2))
    return 0 if not report["summary"]["policy_errors"] and not report["summary"]["oracle_cost_mismatches"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
