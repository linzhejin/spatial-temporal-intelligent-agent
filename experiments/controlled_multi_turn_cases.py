"""Source-derived two-turn route revisions for development experiments.

The second utterance omits the origin and refers to prior constraints. Gold
retains those fields explicitly and derives each route from frozen graph data.
"""
from __future__ import annotations

from typing import Any

from experiments.controlled_route_cases import _anchor, _digest, _turn
from experiments.controlled_route_gold import derive_graph_route
from experiments.controlled_route_replay import score_controlled_route_case
from experiments.independent_route_oracle import inspect_route_edges


_MODES = {"walk": "步行", "bike": "骑行", "drive": "驾车"}


def _turn_parameters(parameters: dict[str, Any], *, revised: bool) -> dict[str, Any]:
    return {
        "start_poi_id": parameters["start_poi_id"],
        "end_poi_id": parameters["revised_end_poi_id"] if revised else parameters["initial_end_poi_id"],
        "via_poi_ids": [],
        "mode": parameters["revised_mode"] if revised else parameters["initial_mode"],
        "avoid_steps": parameters["avoid_steps"],
    }


def _derive_turn(graph: Any, anchors: dict[str, dict[str, Any]],
                 evidence: dict[str, dict[str, set[tuple[Any, Any, Any]]]],
                 params: dict[str, Any]) -> dict[str, Any]:
    mode = params["mode"]
    if mode not in evidence or not isinstance(evidence[mode].get("permitted"), set) \
            or not isinstance(evidence[mode].get("unknown"), set):
        raise ValueError(f"mode evidence missing: {mode}")
    start = _anchor(anchors, params["start_poi_id"])
    end = _anchor(anchors, params["end_poi_id"])
    result = derive_graph_route(
        graph, start_node=start["node_id"], end_node=end["node_id"],
        via_nodes=[], mode=mode,
        permitted_edge_ids=evidence[mode]["permitted"],
        unknown_edge_ids=evidence[mode]["unknown"], closed_edge_ids=set(),
        avoid_steps=params["avoid_steps"],
    )
    if result["verdict"] != "feasible":
        raise ValueError("multi-turn development case requires two decidable routes")
    check = inspect_route_edges(
        graph, result["route_edge_ids"], start_node=start["node_id"],
        end_node=end["node_id"], via_nodes=[], mode=mode,
        permitted_edge_ids=evidence[mode]["permitted"],
        uncertain_edge_ids=evidence[mode]["unknown"], closed_edge_ids=set(),
        avoid_steps=params["avoid_steps"],
    )
    if check["verdict"] != "pass" or check["length_m"] != result["length_m"]:
        raise ValueError("independent edge inspection differs from route derivation")
    return result


def make_multi_turn_route_case(
    graph: Any, anchors: dict[str, dict[str, Any]],
    evidence: dict[str, dict[str, set[tuple[Any, Any, Any]]]], *,
    templates: dict[str, str], case_id: str, template_group: str,
    od_group: str, snapshot_id: str, generator_sha256: str,
    template_sha256: str, parameters: dict[str, Any],
) -> dict[str, Any]:
    """Build one first-route-then-revision case without borrowing product state."""
    if (not isinstance(parameters, dict) or not isinstance(parameters.get("avoid_steps"), bool)
            or parameters.get("initial_mode") not in _MODES
            or parameters.get("revised_mode") not in _MODES
            or parameters["initial_mode"] == parameters["revised_mode"]
            or len({parameters.get("start_poi_id"), parameters.get("initial_end_poi_id"),
                    parameters.get("revised_end_poi_id")}) != 3):
        raise ValueError("invalid multi-turn revision parameters")
    first_template = templates.get("first")
    revision_template = templates.get("revision")
    if (not isinstance(first_template, str) or not isinstance(revision_template, str)
            or any(token not in first_template for token in
                   ("{start}", "{old_end}", "{initial_mode_text}", "{constraint_text}"))
            or any(token not in revision_template for token in
                   ("{new_end}", "{revised_mode_text}"))
            or "最短" not in first_template or "最短" not in revision_template
            or "{start}" in revision_template):
        raise ValueError("revision templates must disclose changed fields but omit the origin")
    start = _anchor(anchors, parameters["start_poi_id"])
    old_end = _anchor(anchors, parameters["initial_end_poi_id"])
    new_end = _anchor(anchors, parameters["revised_end_poi_id"])
    rendered = {
        "start": start["name"], "old_end": old_end["name"],
        "new_end": new_end["name"],
        "initial_mode_text": _MODES[parameters["initial_mode"]],
        "revised_mode_text": _MODES[parameters["revised_mode"]],
        "constraint_text": "不要走台阶" if parameters["avoid_steps"] else "按通常通行规则",
    }
    try:
        messages = [{"role": "user", "content": template.format(**rendered)}
                    for template in (first_template, revision_template)]
    except (KeyError, IndexError, ValueError) as exc:
        raise ValueError("multi-turn template cannot be rendered") from exc
    if any(not message["content"].strip() for message in messages):
        raise ValueError("multi-turn prompt cannot be empty")
    params_by_turn = [_turn_parameters(parameters, revised=revised)
                      for revised in (False, True)]
    oracle_turns = [_derive_turn(graph, anchors, evidence, params)
                    for params in params_by_turn]
    oracle_result = {"turns": oracle_turns}
    parameters_sha = _digest(parameters)
    return {
        "id": case_id, "category": "multi_turn_revision",
        "template_group": template_group, "od_group": od_group,
        "snapshot_id": snapshot_id, "messages": messages,
        "expected_turns": [_turn(params, result["verdict"])
                           for params, result in zip(params_by_turn, oracle_turns)],
        "gold_derivation": {
            "kind": "controlled_template", "generator_id": "controlled_multi_turn_cases_v1",
            "template_id": "route_revision", "generator_sha256": generator_sha256,
            "template_sha256": template_sha256,
            "parameters": parameters, "parameters_sha256": parameters_sha,
            "unresolved_ambiguity": False,
            "evidence_basis": ["template_parameter", "independent_graph_oracle"],
            "oracle_result": oracle_result,
        },
        "adjudication": {
            "status": "generated_verified", "method": "deterministic_source_rules",
            "evidence": [
                {"kind": "template_parameter", "sha256": parameters_sha},
                {"kind": "independent_graph_oracle", "sha256": _digest(oracle_result)},
            ],
        },
    }


def replay_multi_turn_route_case(
    case: dict[str, Any], graph: Any,
    anchors: dict[str, dict[str, Any]],
    evidence: dict[str, dict[str, set[tuple[Any, Any, Any]]]], *,
    templates: dict[str, str],
) -> None:
    """Reject edits to either utterance, label, or route derivation."""
    derivation = case.get("gold_derivation")
    if not isinstance(derivation, dict):
        raise ValueError("missing multi-turn derivation")
    rebuilt = make_multi_turn_route_case(
        graph, anchors, evidence, templates=templates, case_id=case["id"],
        template_group=case["template_group"], od_group=case["od_group"],
        snapshot_id=case["snapshot_id"],
        generator_sha256=derivation["generator_sha256"],
        template_sha256=derivation["template_sha256"],
        parameters=derivation["parameters"],
    )
    if case != rebuilt:
        raise ValueError("multi-turn case differs from source-derived template and graph")


def score_controlled_multi_turn_case(
    case: dict[str, Any], observed: dict[str, Any] | None,
    graph: Any, anchors: dict[str, dict[str, Any]],
    evidence: dict[str, dict[str, set[tuple[Any, Any, Any]]]],
) -> dict[str, Any]:
    """Score both turns; a correct first route cannot mask a lost revision."""
    failures: list[str] = []
    unknowns: list[str] = []
    if not isinstance(observed, dict):
        return {"task_success": False, "failures": ["missing_observation"],
                "unknowns": unknowns, "turn_scores": []}
    if observed.get("scenario_id") != case.get("id"):
        failures.append("scenario_mismatch")
    if observed.get("snapshot_id") != case.get("snapshot_id"):
        failures.append("snapshot_mismatch")
    if observed.get("input_messages") != case.get("messages"):
        failures.append("input_mismatch")
    observed_turns = observed.get("turns")
    expected_turns = case.get("expected_turns")
    oracle_turns = case.get("gold_derivation", {}).get("oracle_result", {}).get("turns")
    if (not isinstance(observed_turns, list) or not isinstance(expected_turns, list)
            or not isinstance(oracle_turns, list) or len(observed_turns) != 2
            or len(expected_turns) != 2 or len(oracle_turns) != 2):
        failures.append("turn_count_mismatch")
        return {"task_success": False, "failures": failures,
                "unknowns": unknowns, "turn_scores": []}
    if observed.get("transport_error"):
        failures.append("transport_error")
    turn_scores = []
    parameters = case["gold_derivation"]["parameters"]
    for index in range(2):
        params = _turn_parameters(parameters, revised=bool(index))
        projected_case = {
            "id": case["id"], "category": "mode_access",
            "snapshot_id": case["snapshot_id"],
            "messages": [case["messages"][index]],
            "expected_turns": [expected_turns[index]],
            "gold_derivation": {"parameters": params,
                                "oracle_result": oracle_turns[index]},
        }
        projected_observation = {
            "scenario_id": observed.get("scenario_id"),
            "snapshot_id": observed.get("snapshot_id"),
            "input_messages": [observed["input_messages"][index]]
            if isinstance(observed.get("input_messages"), list)
            and len(observed["input_messages"]) > index else [],
            "turns": [observed_turns[index]],
            "transport_error": observed.get("transport_error"),
        }
        mode = params["mode"]
        score = score_controlled_route_case(
            projected_case, projected_observation, graph, anchors,
            permitted_edge_ids=evidence[mode]["permitted"],
            unknown_edge_ids=evidence[mode]["unknown"], closed_edge_ids=set(),
        )
        turn_scores.append(score)
        failures.extend(code for code in score["failures"] if code not in failures)
        unknowns.extend(code for code in score["unknowns"] if code not in unknowns)
    return {"task_success": not failures and not unknowns,
            "failures": failures, "unknowns": unknowns,
            "turn_scores": turn_scores}
