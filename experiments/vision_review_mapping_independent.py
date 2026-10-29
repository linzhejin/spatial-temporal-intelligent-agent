"""Audit the eight public-media review pairs without running the product.

The event associations and reviews are simulated.  This only verifies frozen
edge records and SVG geometry; it does not validate observations or model quality.
Outputs are created in a new directory and never replace the original run.
"""
from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import networkx as nx
from shapely import wkt

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = ROOT / "output/experiments/vision_sim_development/public_batch_api_20261028_d"
COLORS = (("baseline_edges", "#216d4b"), ("active_edges", "#6749a3"),
          ("selected_edges", "#d33f36"))


def digest(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def edge_set(values):
    return {tuple(edge) for edge in values}


def geometry(graph, edge):
    u, v, key = edge
    data = graph[u][v][key]
    if data.get("geometry"):
        return list(wkt.loads(data["geometry"]).coords)
    return [(float(graph.nodes[n]["x"]), float(graph.nodes[n]["y"])) for n in (u, v)]


def svg_matches(graph, row, file):
    groups = [(field, color, [geometry(graph, e) for e in row[field]]) for field, color in COLORS]
    all_points = [p for _, _, lines in groups for line in lines for p in line]
    xs, ys = [p[0] * 0.86 for p in all_points], [p[1] for p in all_points]
    minx, maxx, miny, maxy = min(xs), max(xs), min(ys), max(ys)
    scale = min(760 / max(maxx-minx, 1e-8), 320 / max(maxy-miny, 1e-8))
    expected = [(color, " ".join(f"{20+(x*.86-minx)*scale:.2f},{340-(y-miny)*scale:.2f}"
                                  for x, y in line))
                for _, color, lines in groups for line in lines]
    svg = ET.parse(file).getroot()
    actual = [(line.get("stroke"), line.get("points"))
              for line in svg.findall("{http://www.w3.org/2000/svg}polyline")]
    return expected == actual


def audit_row(graph, row, source_dir):
    baseline, active, restored = (row[name] for name in ("baseline_edges", "active_edges", "restored_edges"))
    selected = edge_set(row["selected_edges"])
    bound = edge_set(row["event"]["edge"]["edges"])
    physical = bound | {(v, u, k) for u, v, k in bound}
    on_route = bool(physical & edge_set(baseline))
    sequences = (baseline, active, restored)
    graph_edges_exist = all(graph.has_edge(*edge) for sequence in sequences for edge in sequence)
    graph_edges_exist &= all(graph.has_edge(*edge) for edge in selected)
    contiguous = all(all(a[1] == b[0] for a, b in zip(sequence, sequence[1:]))
                     for sequence in sequences)
    media = Path(row["media_path"])
    inference = Path(row["inference_report"])
    expected_on = row["expected_effect"] == "on_route"
    return {
        "case_id": row["case_id"], "source_id": row["source_id"], "action": row["action"],
        "expected_effect": row["expected_effect"], "simulated": row["simulated"],
        "candidate_count": row["candidate_count"], "registration": row["registration"],
        "selected_intersects_baseline": bool(selected & edge_set(baseline)),
        "published_edge_intersects_baseline_allowing_reverse": on_route,
        "selected_matches_published_edge_allowing_reverse": bool(selected & physical),
        "label_matches_published_edge": on_route == expected_on,
        "recorded_overlap_matches_recalculation": row["baseline_event_overlap"] == on_route,
        "all_edges_in_frozen_raw_graph": bool(graph_edges_exist), "routes_contiguous": contiguous,
        "baseline_restored_exactly": baseline == restored,
        "closed_edge_absent_from_active_route": not bool(physical & edge_set(active))
            if row["action"] == "closure" else None,
        "route_changed": baseline != active, "baseline_length_m": row["baseline_length_m"],
        "active_length_m": row["active_length_m"], "restored_length_m": row["restored_length_m"],
        "svg_matches_recorded_frozen_graph_geometry": svg_matches(graph, row, source_dir / "review_pack" / (row["case_id"] + ".svg")),
        "media_hash_matches": media.is_file() and digest(media) == row["media_sha256"],
        "inference_hash_matches": inference.is_file() and digest(inference) == row["inference_report_sha256"],
        "record_sha256": digest(source_dir / (row["case_id"] + ".json")),
        "svg_sha256": digest(source_dir / "review_pack" / (row["case_id"] + ".svg")),
    }


def counterfactual_closure(graph, pair):
    """Raw-graph reference only, not a product rerun or equivalent policies."""
    result = []
    for row in pair:
        edges = row["baseline_edges"]
        start, end = edges[0][0], edges[-1][1]
        trial = graph.copy()
        removed = []
        for u, v, k in row["event"]["edge"]["edges"]:
            for a, b in ((u, v), (v, u)):
                if trial.has_edge(a, b, k):
                    trial.remove_edge(a, b, k)
                    removed.append([a, b, k])
        base_distance = nx.shortest_path_length(graph, start, end, weight="length")
        try:
            closure_distance = nx.shortest_path_length(trial, start, end, weight="length")
        except nx.NetworkXNoPath:
            closure_distance = None
        result.append({"case_id": row["case_id"], "fixed_action": "closure",
                       "raw_graph_baseline_m": base_distance, "raw_graph_closure_m": closure_distance,
                       "recorded_baseline_m": row["baseline_length_m"], "removed_edges": removed})
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    src = args.input.resolve(strict=True)
    summary = json.loads((src / "summary.json").read_text(encoding="utf8"))
    graph_path = ROOT / "data/whu_road_network.graphml"
    graph_hash = digest(graph_path)
    if graph_hash != summary["current_source_manifest"]["data/whu_road_network.graphml"]:
        raise ValueError("raw graph does not match original recorded hash")
    graph = nx.read_graphml(graph_path, node_type=int, force_multigraph=True)
    pairs, results, pair_rows = [], [], {}
    for kind in ("congestion", "flooding"):
        for number in range(1, 5):
            prefix = f"{kind}-{number:02d}"
            rows = [json.loads((src / f"{prefix}-{i}.json").read_text(encoding="utf8")) for i in (1, 2)]
            pair_rows[prefix] = rows
            results.extend(audit_row(graph, row, src) for row in rows)
            pairs.append({"pair": prefix, "case_ids": [row["case_id"] for row in rows],
                          "same_source": rows[0]["source_id"] == rows[1]["source_id"],
                          "same_baseline": rows[0]["baseline_edges"] == rows[1]["baseline_edges"],
                          "same_action": rows[0]["action"] == rows[1]["action"],
                          "actions": [row["action"] for row in rows]})
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = {"scope": "offline_reaudit_simulated_development_not_field_or_accuracy",
              "input_summary_sha256": digest(src / "summary.json"),
              "raw_graph_sha256": graph_hash, "script_sha256": digest(Path(__file__)),
              "cases": results, "pairs": pairs,
              "same_action_raw_graph_counterfactual": counterfactual_closure(graph, pair_rows["congestion-02"]),
              "limitations": ["This checks recorded simulated associations only, not true media location or human review.",
                              "Raw-graph counterfactual is not a product, tool, event lifecycle, or agent run; effective access rules are not independently reproduced.",
                              "Original run's agents/tools.py and spatial/road_conditions.py differ from the current working tree; this is not an exact replay.",
                              "Flood images have no model candidates; controlled events do not measure detector performance."]}
    (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf8")
    print(json.dumps({"report": str(output / "report.json"), "case_count": len(results),
                      "pair_count": len(pairs), "confounded": [p for p in pairs if not p["same_action"]]},
                     ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
