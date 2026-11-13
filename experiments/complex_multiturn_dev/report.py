"""Produce a conservative per-case development report from saved runs."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from decimal import Decimal
import hashlib
import json
from pathlib import Path

from experiments.complex_multiturn_dev.audit import audit
from experiments.complex_multiturn_dev.budget import BudgetLedger


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "experiments/datasets/complex_multiturn_dev_v1"
OUTPUT = ROOT / "output/experiments/complex_multiturn_dev_v1"


def _read_lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def audit_fingerprint(run_dir, dataset):
    """Never reuse a report audit after its code or saved evidence changes."""
    package = Path(__file__).resolve().parent
    files = [package / name for name in ("audit.py", "graph_checks.py", "contracts.py", "observation.py", "semantic_checks.py")]
    files.extend([Path(dataset), Path(run_dir) / "manifest.json"])
    files.extend(path for name in ("turns.jsonl", "restored_observations.jsonl", "recovered_uncollected.jsonl")
                 if (path := Path(run_dir) / name).exists())
    files.extend(sorted(Path(run_dir).glob("*_candidate_scope_audit.jsonl")))
    files.extend(sorted(Path(run_dir).glob("*_candidate_time_audit.jsonl")))
    rubric = DATA / "reference_route_rubric.json"
    if rubric.exists():
        files.append(rubric)
    semantic_rubric = DATA / "semantic_rubric_v4.json"
    if semantic_rubric.exists():
        files.append(semantic_rubric)
    provider = Path(run_dir) / "provider_evidence.jsonl"
    if provider.exists():
        files.append(provider)
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode("utf-8"))
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    return digest.hexdigest()[:12]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, action="append", default=[])
    parser.add_argument("--run", type=Path, action="append", default=[])
    parser.add_argument("--final", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--budget-limit-cny", type=Decimal, default=Decimal("20"))
    args = parser.parse_args()
    datasets = [DATA / "cases.json", DATA / "cases_event_endpoint_revised.json"]
    by_hash = {hashlib.sha256(path.read_bytes()).hexdigest(): path for path in datasets}
    cases = json.loads(datasets[1].read_text(encoding="utf-8"))["cases"]
    expected = {case["id"]: len(case["turns"]) for case in cases}
    baseline_ids = {path.name for path in args.baseline}
    final_ids = {path.name for path in args.final}
    histories = defaultdict(list)
    browser_cases = defaultdict(list)
    for directory in sorted({*args.baseline, *args.run, *args.final}, key=lambda path: path.name):
        manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
        dataset = by_hash.get(manifest["dataset_sha256"])
        if not dataset:
            raise ValueError(f"unknown dataset for {directory.name}")
        audit_path = directory / ("report_audit_" + audit_fingerprint(directory, dataset) + ".jsonl")
        if not audit_path.exists():
            audit(directory, dataset, audit_path.name)
        grades = _read_lines(audit_path)
        observations = _read_lines(directory / "turns.jsonl") if (directory / "turns.jsonl").exists() else []
        by_case = defaultdict(list)
        for row in grades:
            by_case[row["case"]].append(row)
        for case_id in manifest["case_ids"]:
            rows = by_case[case_id]
            fail = [f"第{row['turn']}轮：{','.join(row.get('failures') or [])}"
                    for row in rows if row.get("verdict") == "fail"]
            missing = len(rows) < expected[case_id] or any(row.get("verdict") == "not_run" for row in rows)
            if not rows:
                verdict = "未运行"
            elif any(any("timeout" in issue for issue in row.get("failures") or []) for row in rows):
                verdict = "超时"
            elif fail:
                verdict = "失败"
            elif missing:
                verdict = "部分未运行"
            elif any(row.get("verdict") == "unknown" for row in rows):
                verdict = "证据不足"
            else:
                verdict = "通过"
            unknown = sorted({reason for row in rows for reason in row.get("unknowns") or []})
            histories[case_id].append({"run": directory.name, "verdict": verdict, "fail": fail,
                                       "unknown": unknown, "baseline": directory.name in baseline_ids,
                                       "final": directory.name in final_ids})
        for row in observations:
            ui = row.get("browser_evidence")
            if not isinstance(ui, dict):
                continue
            failed = bool(ui.get("page_errors") or ui.get("ui_error") or
                          any(ui.get(key) is False for key in
                              ("chat_visible", "cards_match", "route_panel_visible", "route_map_path", "selected_matches")))
            browser_cases[row["case"]].append((directory.name, row["turn"], failed,
                                                bool(row.get("budget_stop_reason"))))

    ledger = BudgetLedger(OUTPUT / "budget.jsonl", limit_cny=args.budget_limit_cny)
    latest_counts, final_counts = Counter(), Counter()
    lines = ["# 珞珈智行复杂多轮开发测试逐题记录", "",
             "本记录为已暴露给开发和修复过程的开发数据，不是正式盲测，也不是三架构对照。"
             "通过仅指该题已记录检查的证据充分；候选语义、无路线结果或图锚点证据不足均不计为通过。", "",
             f"保守累计预算上界（含已预留或按上限计入的请求，非实际账单）：{ledger.spent_or_reserved()} 元；上限 {args.budget_limit_cny} 元。", "",
             "| 题目 | 首测 | 最近一次 | 次数 | 最终批次 | 失败与证据缺口 |", "|---|---|---|---:|---|---|"]
    for case in cases:
        ident, attempts = case["id"], histories[case["id"]]
        initial = next((item for item in attempts if item["baseline"]), None)
        latest = attempts[-1] if attempts else None
        final = next((item for item in reversed(attempts) if item["final"]), None)
        latest_counts[latest["verdict"] if latest else "未运行"] += 1
        final_counts[final["verdict"] if final else "未运行"] += 1
        issues = [f"{item['run']}：{'；'.join(item['fail'])}" for item in attempts if item["fail"]]
        if latest and latest["unknown"]:
            issues.append("最近待核：" + ", ".join(latest["unknown"]))
        lines.append(f"| {ident} {case['title']} | {initial['verdict'] if initial else '未运行'} | "
                     f"{latest['verdict'] if latest else '未运行'} | {len(attempts)} | "
                     f"{final['verdict'] if final else '未运行'} | " + ("；".join(issues) or "—") + " |")
    lines.extend(["", "最近一次记录统计：" + "，".join(f"{key} {value} 题" for key, value in sorted(latest_counts.items())) + "。",
                  "指定最终批次统计：" + "，".join(f"{key} {value} 题" for key, value in sorted(final_counts.items())) + "。",
                  "", "浏览器记录："])
    if not browser_cases:
        lines.append("- 尚无浏览器记录。")
    for ident in sorted(browser_cases):
        all_rows = browser_cases[ident]
        newest = max(row[0] for row in all_rows)
        rows = [row for row in all_rows if row[0] == newest]
        interrupted = sum(row[3] for row in rows)
        if interrupted or len(rows) < expected[ident]:
            status = f"未完整，预算中断 {interrupted} 轮"
        elif any(row[2] for row in rows):
            status = "存在显示失败或异常"
        else:
            status = "记录到的适用聊天、卡片、选点及路线检查未见失败"
        lines.append(f"- {ident}：最近一次 {newest}，{len(rows)} 轮有浏览器证据；{status}。")
    lines.extend(["", "每次运行的 `turns.jsonl`、`model_requests.jsonl`、事件快照和审计文件保留原始证据；"
                  "早期采集器问题另有从隔离数据库恢复的 `restored_observations.jsonl`。"
                  "冻结图只证明建模范围内的连通和限制，不证明真实入口、接驳、营业或现场通行。"
                  "浏览器测试阻止外部地图瓦片请求，路线矢量可见不代表在线底图瓦片已核验。", ""])
    if args.output.exists():
        raise FileExistsError(args.output)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "latest": dict(latest_counts),
                      "final": dict(final_counts), "browser_cases": len(browser_cases)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
