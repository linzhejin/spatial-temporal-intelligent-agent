"""Derive paired E3 tables from preserved actual API records, without rerunning them."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import statistics
import sys
import zipfile


ROOT = Path(__file__).resolve().parents[2]
POLICIES = ("full_recompute", "conservative_reuse")
LABELS = {
    "unchanged": "无变化", "off_path_closure": "路外新增封闭",
    "on_path_closure": "路内新增封闭", "reopen": "道路恢复",
    "expiry": "自然到期", "revocation": "撤销", "future_event": "尚未生效",
    "expired_event": "已过期事件", "other_mode": "其他方式事件",
    "on_path_slowdown": "路内缓行", "off_path_slowdown": "路外缓行",
    "mixed": "封闭与缓行组合", "notice": "提示事件", "mode_change": "方式修改",
    "start_change": "起点修改", "end_change": "终点修改",
    "constraint_change": "硬约束修改", "joint_change": "方式与事件联合变化",
    "graph_cost_decrease": "基础图路外成本下降", "unreachable": "不可达",
    "recovery": "受阻后恢复", "reopen_shortcut": "恢复更短道路",
    "off_path_cost_decrease": "路外成本下降", "flat_weight_change": "偏好权重变化",
}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def percentile(values, quantile):
    """Linear interpolation over ordered observations; no statistical inference."""
    ordered = sorted(values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * quantile
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def median(values):
    return statistics.median(values) if values else None


def data_of(row):
    body = row.get("response") or {}
    return body.get("data", body)


def flat_record(row):
    metrics = row.get("metrics") or {}
    phases = metrics.get("phases", {})
    audit_metrics = row.get("audit", {}).get("metrics", {})
    data = data_of(row)
    return {
        "case_id": row["case_id"], "family": row.get("family", "controlled"),
        "variant": row.get("variant", row["case_id"]), "attempt": row["attempt"],
        "strategy": row["strategy"], "operation_probe": row["operation_probe"],
        "http_status": row.get("http_status"), "audit_status": row["audit"]["status"],
        "elapsed_ms": row.get("elapsed_ms"),
        "path_reused": bool(data.get("execution_optimization", {}).get("path_reused")),
        "dijkstra_calls": phases.get("dijkstra", {}).get("calls", 0),
        "dijkstra_adjacency_expansions": sum(s["adjacency_expanded_nodes"]
            for s in metrics.get("searches", []) if s.get("adjacency_expanded_nodes") is not None)
            if row["operation_probe"] else None,
        "dijkstra_weight_evaluations": sum(s["weight_evaluations"]
            for s in metrics.get("searches", []) if s.get("weight_evaluations") is not None)
            if row["operation_probe"] else None,
        "distance_m": data.get("recommended_length_m"), "eta_min": data.get("duration_min"),
        "oracle_cost": audit_metrics.get("oracle_cost"),
        "selected_cost": audit_metrics.get("selected_objective_cost"),
        "http_requests_in_update": len(row.get("requests", [])),
        "version_rejections_409": sum(r.get("status") == 409 for r in row.get("requests", [])),
        "rejected_then_rebuilt": row.get("http_status") == 200
            and any(r.get("status") == 409 for r in row.get("requests", [])),
    }


def check_batch(protocol, records):
    probes = protocol.get("operation_probe_repeats", 1)
    expected = {(c["id"], p, n) for c in protocol["cases"] for p in POLICIES
                for n in range(1, protocol["repeats"] + probes + 1)}
    counts = Counter((r["case_id"], r["strategy"], r["attempt"]) for r in records)
    actual = set(counts)
    issues = []
    if actual != expected:
        issues.append({"type": "execution_set_mismatch", "missing": sorted(expected - actual),
                       "unexpected": sorted(actual - expected)})
    if any(n != 1 for n in counts.values()):
        issues.append({"type": "duplicate_execution_keys"})
    for row in records:
        key = (row["case_id"], row["strategy"], row["attempt"])
        if row["operation_probe"] != (row["attempt"] > protocol["repeats"]):
            issues.append({"type": "wrong_probe_label", "key": key})
        if row["audit"]["status"] != "pass":
            issues.append({"type": "independent_audit_not_pass", "key": key})
        if "execution_policy_audit" in row and row["execution_policy_audit"]["status"] != "pass":
            issues.append({"type": "policy_audit_not_pass", "key": key})
        if row.get("elapsed_ms") is None or row["elapsed_ms"] < 0:
            issues.append({"type": "missing_request_measurement", "key": key})
    return issues


def paired_rows(rows):
    groups = defaultdict(dict)
    for row in rows:
        groups[(row["case_id"], row["attempt"])][row["strategy"]] = row
    pairs = []
    for (case_id, attempt), group in sorted(groups.items()):
        if set(group) != set(POLICIES):
            raise ValueError(f"unpaired execution: {case_id}/{attempt}")
        full, reuse = (group[p] for p in POLICIES)
        objective_delta = None
        if full["selected_cost"] is not None and reuse["selected_cost"] is not None:
            objective_delta = reuse["selected_cost"] - full["selected_cost"]
        pairs.append({
            "case_id": case_id, "family": full["family"], "variant": full["variant"],
            "attempt": attempt, "operation_probe": full["operation_probe"],
            "full_ms": full["elapsed_ms"], "reuse_ms": reuse["elapsed_ms"],
            "reuse_minus_full_ms": reuse["elapsed_ms"] - full["elapsed_ms"],
            "reuse_over_full_ratio": reuse["elapsed_ms"] / full["elapsed_ms"],
            "full_status": full["http_status"], "reuse_status": reuse["http_status"],
            "same_outcome": full["http_status"] == reuse["http_status"],
            "objective_delta": objective_delta,
            "cost_comparison_applicable": full["http_status"] == reuse["http_status"] == 200,
            "cost_comparable": objective_delta is not None,
            "cost_equivalent": abs(objective_delta) <= 1e-6 if objective_delta is not None else None,
            "path_reused": reuse["path_reused"],
            "dijkstra_calls_saved": full["dijkstra_calls"] - reuse["dijkstra_calls"],
            "adjacency_expansions_saved": (
                full["dijkstra_adjacency_expansions"] - reuse["dijkstra_adjacency_expansions"]
                if full["operation_probe"] else None),
        })
    return pairs


def aggregates(rows, pairs, dimension):
    output = []
    names = sorted({r[dimension] for r in rows}) if dimension else ["all"]
    for name in names:
        selected = [r for r in rows if not dimension or r[dimension] == name]
        selected_pairs = [r for r in pairs if not dimension or r[dimension] == name]
        timed_pairs = [r for r in selected_pairs if not r["operation_probe"]]
        for policy in POLICIES:
            timed = [r for r in selected if r["strategy"] == policy and not r["operation_probe"]]
            probes = [r for r in selected if r["strategy"] == policy and r["operation_probe"]]
            times = [r["elapsed_ms"] for r in timed]
            output.append({
                "group": name, "label": LABELS.get(name, name), "strategy": policy,
                "timed_executions": len(timed), "operation_probes": len(probes),
                "audit_pass": sum(r["audit_status"] == "pass" for r in timed),
                "route_deliveries_200": sum(r["http_status"] == 200 for r in timed),
                "correct_no_route_404": sum(r["http_status"] == 404 and r["audit_status"] == "pass" for r in timed),
                "update_http_requests": sum(r["http_requests_in_update"] for r in timed),
                "version_rejections_409": sum(r["version_rejections_409"] for r in timed),
                "rejected_then_rebuilt_flows": sum(r["rejected_then_rebuilt"] for r in timed),
                "cost_comparable_pairs": sum(r["cost_comparable"] for r in timed_pairs),
                "cost_equivalent_pairs": sum(r["cost_equivalent"] is True for r in timed_pairs),
                "cost_not_applicable_pairs": sum(not r["cost_comparison_applicable"] for r in timed_pairs),
                "path_reused": sum(r["path_reused"] for r in timed),
                "request_ms_p50": median(times), "request_ms_p95": percentile(times, .95),
                "timed_dijkstra_calls": sum(r["dijkstra_calls"] for r in timed),
                "probe_dijkstra_calls": sum(r["dijkstra_calls"] for r in probes),
                "probe_dijkstra_adjacency_expansions": sum(r["dijkstra_adjacency_expansions"] for r in probes),
                "probe_weight_evaluations": sum(r["dijkstra_weight_evaluations"] for r in probes),
                "paired_reuse_minus_full_ms_p50": median([r["reuse_minus_full_ms"] for r in timed_pairs]),
                "paired_reuse_over_full_ratio_p50": median([r["reuse_over_full_ratio"] for r in timed_pairs]),
            })
    return output


def phase_rows(records):
    groups = defaultdict(list)
    for row in records:
        if row["operation_probe"]:
            continue
        for name, metric in row["metrics"]["phases"].items():
            groups[(row["strategy"], name)].append(metric)
    return [{"strategy": p, "phase": phase, "update_flows": len(values),
             "calls": sum(v["calls"] for v in values),
             "inclusive_ms_p50": median([v["total_ms"] for v in values]),
             "inclusive_ms_p95": percentile([v["total_ms"] for v in values], .95)}
            for (p, phase), values in sorted(groups.items())]


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_batch(path):
    protocol = read_json(path / "protocol.json")
    records = [json.loads(line) for line in (path / "executions.jsonl").read_text(encoding="utf-8").splitlines()]
    summary = read_json(path / "summary.json")
    issues = check_batch(protocol, records)
    accepted = summary.get("acceptance_passed", summary.get("completed_successfully", False))
    if not accepted:
        issues.append({"type": "batch_acceptance_not_passed"})
    if issues:
        raise ValueError(json.dumps({"batch": path.name, "issues": issues}, ensure_ascii=False))
    return protocol, records, summary


def archive_batches(paths, destination):
    manifest = []
    with zipfile.ZipFile(destination, "x", zipfile.ZIP_DEFLATED) as archive:
        for source in paths:
            for path in sorted(source.rglob("*")):
                if not path.is_file():
                    continue
                name = f"{source.name}/{path.relative_to(source).as_posix()}"
                data = path.read_bytes()
                archive.writestr(name, data)
                manifest.append({"path": name, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    with zipfile.ZipFile(destination) as archive:
        if archive.testzip() is not None:
            raise ValueError("archive integrity failed")
        for item in manifest:
            if hashlib.sha256(archive.read(item["path"])).hexdigest() != item["sha256"]:
                raise ValueError("archived source hash mismatch")
    return manifest


def write_readme(output, result, rows, pairs, counter_path):
    """Keep the human-readable report tied to the same validated CSV values."""
    overall = aggregates(rows, pairs, None)
    labels = {"full_recompute": "完整重算", "conservative_reuse": "保守路径维护"}
    timed_pairs = [r for r in pairs if not r["operation_probe"]]
    comparable = sum(r["cost_comparable"] for r in timed_pairs)
    campus = result["campus"]
    counter = result["counterexamples"]
    lines = [
        "# E3 路径维护对照结果", "",
        f"六组已见校园起终点形成{result['campus_fixed_conditions']}个固定变化条件。"
        f"本批完成{campus['actual_timed_executions']}次计时更新流程，"
        f"另有{campus['actual_operation_probes']}次操作数探针。"
        "两种策略都实际调用产品接口，独立评分在计时结束后进行。", "",
        "## 更新结果", "",
        "| 策略 | 计时更新 | 符合预期 | 返回路线 | 正确不可达 | 版本拒绝后重建 | 实际复用 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    history = campus.get("execution_history")
    if history:
        recovery = (
            "本批在系统合盖待机时中断，随后按冻结清单续跑。"
            f"保留母批{history['preserved_parent_records']}条观测，"
            f"仅补入缺失的{history['added_missing_records']}条；"
            f"另有{history['additional_diagnostic_records']}条重复观测单独归档，不替换原结果。"
            f"中断的{history['interrupted_operation_probe_attempts']}次操作探针未取得响应，"
            "保留为中断记录，不记为成功。两策略在同一批段内成对执行，续跑重放原随机顺序；"
            "新进程与主机状态造成的批段差异仍需考虑。"
        )
        lines[4:4] = [recovery, ""]
    for r in overall:
        lines.append(f"| {labels[r['strategy']]} | {r['timed_executions']} | {r['audit_pass']} | "
                     f"{r['route_deliveries_200']} | {r['correct_no_route_404']} | "
                     f"{r['rejected_then_rebuilt_flows']} | {r['path_reused']} |")
    lines += ["", "“版本拒绝后重建”是返回路线的子集，不与路线交付数相加。"
              "图版本改变时，一次更新流程包含409拒绝和重新规划两次HTTP请求。", "",
              f"{len(timed_pairs)}组计时配对的最终结果类型一致；"
              f"其中{comparable}组返回路线，可以比较最优代价，"
              f"代价不一致数为{sum(r['cost_equivalent'] is False for r in timed_pairs)}。"
              "不可达配对不进入最优代价比较分母。", "",
              "## 完整更新时延", "",
              "| 策略 | 更新时延中位数 / s | P95 / s | 计时运行中的Dijkstra调用数 |",
              "|---|---:|---:|---:|"]
    for r in overall:
        lines.append(f"| {labels[r['strategy']]} | {r['request_ms_p50']/1000:.3f} | "
                     f"{r['request_ms_p95']/1000:.3f} | {r['timed_dijkstra_calls']} |")
    delta = result["reused_pairs_delta_ms_p50"]
    ratio = result["reused_pairs_ratio_p50"]
    lines += ["", f"实际复用的{result['reused_timed_pairs']}组计时配对中，"
              f"维护减去完整重算的时延差中位数为{delta/1000:+.3f}秒，"
              f"配对时延比的中位数为{ratio:.3f}。"
              + ("本批可复用条件没有显示配对时延中位数下降。" if delta >= 0
                 else "本批可复用条件的配对时延中位数下降。"), "",
              "时延覆盖请求处理、身份与证明、计算或复用、核验、说明生成及响应。"
              "共同初始建路和事后评分不计入更新时延；操作数探针的耗时也不混入。"
              "不同阶段互相包含，不能将阶段耗时相加。", "",
              "## 搜索操作", "",
              "| 策略 | 独立探针数 | Dijkstra调用数 | 邻接展开次数 | 权重计算次数 |",
              "|---|---:|---:|---:|---:|"]
    for r in overall:
        lines.append(f"| {labels[r['strategy']]} | {r['operation_probes']} | "
                     f"{r['probe_dijkstra_calls']} | {r['probe_dijkstra_adjacency_expansions']} | "
                     f"{r['probe_weight_evaluations']} |")
    lines += ["", "邻接展开按每次Dijkstra搜索中发生权重计算的不同源节点计数，"
              "跨搜索求和；目标提前结束与无出边节点不计。本表不代表自有预算标签搜索的节点数。", "",
              "## 起终点分组", "",
              "| 起终点组 | 完整重算中位数 / s | 保守维护中位数 / s | 配对时延差中位数 / s |",
              "|---|---:|---:|---:|"]
    families = aggregates(rows, pairs, "family")
    for family in sorted({r["group"] for r in families}):
        full, reuse = (next(r for r in families if r["group"] == family and r["strategy"] == p)
                       for p in POLICIES)
        lines.append(f"| {family} | {full['request_ms_p50']/1000:.3f} | "
                     f"{reuse['request_ms_p50']/1000:.3f} | "
                     f"{full['paired_reuse_minus_full_ms_p50']/1000:+.3f} |")
    lines += ["", "三次重复用于描述运行波动；同一起终点派生的条件存在关联，"
              "因此按六组起终点分别报告，不把全部重复当作独立场景作显著性推断。", "",
              "## 仅监听旧路径的反例", "",
              f"两个产品策略完成{counter['actual_timed_executions']}次计时及"
              f"{counter['actual_operation_probes']}次操作探针，独立评分全部通过。"
              "下表是基于实际初始边序列的离线弱监听诊断，没有第三条API策略的耗时。", "",
              "| 变化 | 保留旧路线的代价 | 独立最优代价 | 差额 | 单位 |",
              "|---|---:|---:|---:|---|"]
    weak = [json.loads(x) for x in (counter_path / "weak_diagnostics.jsonl").read_text(encoding="utf-8").splitlines()]
    for case in sorted({r["case_id"] for r in weak}):
        diagnostic = next(r["diagnostic"] for r in weak if r["case_id"] == case)
        unit = "米" if diagnostic["objective"] == "shortest" else "加权代价"
        lines.append(f"| {LABELS[case]} | {diagnostic['retained_objective_cost']:.5f} | "
                     f"{diagnostic['independent_optimum']:.5f} | {diagnostic['objective_gap']:.5f} | {unit} |")
    lines += ["", "三类反例中旧路线仍可行，却已经失去最优性。道路恢复、路外成本下降和目标变化"
              "都需要纳入更新判断。", "",
              "## 记录与适用范围", "",
              "[逐次执行](executions.csv)、[逐次配对](paired_comparison.csv)、"
              "[按变化汇总](variant_summary.csv)、[按起终点汇总](family_summary.csv)、"
              "[阶段观测](phase_summary.csv)、[受控反例](counterexample_executions.csv)、"
              "[完整统计](summary.json)、[原始批次归档](raw_batches.zip)、[归档哈希](archive_manifest.json)。", "",
              "原失败批次和修正依据一并保存在归档中，仅完整验收批次进入本表统计。"
              "本专项检验直接最短路的受限复用；固定道路关联、静态方式过滤和端点吸附作为共同输入。"
              "估时、结构化事件说明和当前路线重新核验，通用跨结果依赖维护继续按论文方案推进。"
              "本批无语言模型调用，与300项未见任务、2700次主实验分别计数。", ""]
    (output / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campus", type=Path, required=True)
    parser.add_argument("--counterexamples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--include-diagnostic", action="append", type=Path, default=[])
    args = parser.parse_args()
    campus_protocol, records, campus_summary = load_batch(args.campus)
    _, counter_records, counter_summary = load_batch(args.counterexamples)
    flat = [flat_record(r) for r in records]
    pairs = paired_rows(flat)
    if any(not r["same_outcome"] or
           (r["cost_comparison_applicable"] and r["cost_equivalent"] is not True) for r in pairs):
        raise ValueError("paired outcome/optimal-cost disagreement requires review")
    args.output.mkdir(parents=True, exist_ok=False)
    write_csv(args.output / "executions.csv", flat)
    write_csv(args.output / "paired_comparison.csv", pairs)
    for filename, dimension in (("overall", None), ("family", "family"), ("variant", "variant")):
        write_csv(args.output / f"{filename}_summary.csv", aggregates(flat, pairs, dimension))
    write_csv(args.output / "phase_summary.csv", phase_rows(records))
    counter_flat = [flat_record(r) for r in counter_records]
    write_csv(args.output / "counterexample_executions.csv", counter_flat)
    write_csv(args.output / "counterexample_summary.csv", aggregates(
        counter_flat, paired_rows(counter_flat), "variant"))
    reused_pairs = [r for r in pairs if r["path_reused"] and not r["operation_probe"]]
    result = {
        "campus": campus_summary, "counterexamples": counter_summary,
        "campus_fixed_conditions": len(campus_protocol["cases"]),
        "campus_families": len({c["family"] for c in campus_protocol["cases"]}),
        "campus_timed_pairs": sum(not r["operation_probe"] for r in pairs),
        "campus_probe_pairs": sum(r["operation_probe"] for r in pairs),
        "reused_timed_pairs": len(reused_pairs),
        "reused_pairs_delta_ms_p50": median([r["reuse_minus_full_ms"] for r in reused_pairs]),
        "reused_pairs_ratio_p50": median([r["reuse_over_full_ratio"] for r in reused_pairs]),
        "outcome_disagreements": sum(not r["same_outcome"] for r in pairs),
        "cost_comparable_pairs": sum(r["cost_comparable"] for r in pairs),
        "cost_comparable_timed_pairs": sum(r["cost_comparable"] and not r["operation_probe"] for r in pairs),
        "cost_comparable_probe_pairs": sum(r["cost_comparable"] and r["operation_probe"] for r in pairs),
        "cost_not_applicable_pairs": sum(not r["cost_comparison_applicable"] for r in pairs),
        "optimal_cost_disagreements": sum(r["cost_equivalent"] is False for r in pairs),
        "statistics": "descriptive paired repeats; six related OD families, no independent-sample significance claim",
        "derivation": {"script": str(Path(__file__).relative_to(ROOT)),
                       "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                       "input_batches": [args.campus.name, args.counterexamples.name],
                       "argv": sys.argv},
    }
    paths = [args.campus, args.counterexamples, *args.include_diagnostic]
    manifest = archive_batches(paths, args.output / "raw_batches.zip")
    for name, value in (("summary.json", result), ("archive_manifest.json", manifest)):
        (args.output / name).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_readme(args.output, result, flat, pairs, args.counterexamples)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
