"""Render the frozen task and synthetic-event definitions as Chinese instructions."""
from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "experiments/datasets/complex_multiturn_dev_v1"


def main():
    data = json.loads((DATA / "cases_event_endpoint_revised.json").read_text(encoding="utf-8"))
    fixtures = {row["case_id"]: row for row in json.loads(
        (DATA / "event_fixtures_v2.json").read_text(encoding="utf-8"))["fixtures"]}
    pois = {row["id"]: row["name"] for row in json.loads(
        (ROOT / "data/pois.json").read_text(encoding="utf-8"))["pois"]}
    fields = {"origin": "出发点", "search_anchor": "搜索中心", "mode": "方式", "radius_m": "搜索范围（米）",
              "avoid_steps": "避台阶", "stops": "活动顺序", "clarification_slot": "待澄清字段",
              "selected_matches_first": "选择首个候选", "event_state": "已读取事件阶段",
              "activities": "活动类别"}
    operators = {"eq": "等于", "contains": "包含", "not_contains": "不包含",
                 "ordered_contains": "依次包含", "exists": "存在", "absent": "不存在"}

    def show(value):
        if isinstance(value, list):
            return " → ".join(show(item) for item in value)
        if isinstance(value, bool):
            return "是" if value else "否"
        return str(pois.get(value, value)) if isinstance(value, str) else str(value)

    lines = ["# 珞珈智行复杂多轮开发测试题", "",
             "本题包仅用于开发调试，32 题已向开发过程开放，不能作为正式盲测集。每题使用独立会话，题内保留真实历史；"
             "选点题使用前一轮实际展示的首个候选，不预填答案。以下字段是最低检查项，"
             "还应核对用户当轮实际表达、候选用途、路线、约束和事件，字段相符不等于整题通过。", "",
             "地点来自仓库固定校园 POI 目录，路线检查使用同版本校园图。锚点允许最多 60 米的几何吸附，"
             "接驳与入口尚未现场核实；在线商户、营业和场地开放状态也未核实。找不到候选、不可达或需要澄清时，"
             "应基于证据说明，不得自行放宽条件。", "",
             "道路事件均为隔离环境内的合成记录，不代表真实校园路况；按所列方向、出行方式和生命周期生效。"
             "开始时间为 1792962000，结束时间为 1792969200，采用左闭右开区间；撤销阶段在结束前撤销。", ""]
    for case in data["cases"]:
        lines.extend([f"## {case['id']}｜{case['title']}（{'核心题' if case['phase'] == 'core' else '变体题'}）", ""])
        for number, turn in enumerate(case["turns"], 1):
            checks = [fields.get(row["field"], row["field"]) + operators[row["op"]] +
                      (show(row["value"]) if "value" in row else "") for row in turn["checks"]]
            lines.append(f"{number}. 用户：{turn['user']}<br>")
            lines.append("   检查：" + "；".join(checks) + "。" +
                         (" 选点分支：点击上一轮首张候选卡片，验证提交的地点 ID。"
                          if turn.get("action") == "select_first_candidate" else ""))
            if turn.get("event_stage"):
                lines.append("   事件阶段：" + turn["event_stage"] + "。")
        fixture = fixtures.get(case["id"])
        if fixture:
            edges = ["→".join(map(str, edge)) for edge in fixture["stages"]["active"]["expected_closed_edges"]]
            lines.extend(["", f"事件依据：{show(fixture['start_poi_id'])} → {show(fixture['end_poi_id'])}；"
                          f"类型 {fixture['kind']}；受限方式 {fixture['mode']}；生效边（起点→终点→多重边键）："
                          + "，".join(edges) + "。",
                          f"冻结图基准：无事件 {fixture['baseline']['verdict']}；"
                          f"事件生效时 {fixture['active_reference']['verdict']}。"])
        lines.append("")
    output = DATA / "题目说明.md"
    content = "\n".join(lines).rstrip() + "\n"
    if output.exists() and output.read_text(encoding="utf-8") != content:
        raise FileExistsError("题目说明已存在且与当前冻结题包不同")
    if not output.exists():
        output.write_text(content, encoding="utf-8")
    print(f"Saved {len(data['cases'])} case descriptions: {output}")


if __name__ == "__main__":
    main()
