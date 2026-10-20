"""
批量评估 CLI (E1 意图解析 + E2 路径质量)

用法：
  python experiments/runner.py --method llm|rule|fixed|shortest \
      --queries experiments/datasets/queries_v1.json \
      --gold    experiments/datasets/gold_intents_v1.json \
      --output  experiments/results/<method>_results.csv

E1 方法 (llm/rule)：解析 query → TaskIntent，与 gold 比对，输出意图解析指标。
E2 方法 (fixed/shortest)：解析 query 取起终点 → 调用对应路径算法 → 输出路径质量指标。

LLM 方法在 DEEPSEEK_API_KEY 未配置时会自动走规则兜底（parse_query 内部处理），
此处仅打印告警不阻断。
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Optional

# 注入项目根目录 + experiments 目录，保证 from agents/spatial/experiments 可导入
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
_EXPERIMENTS_DIR = str(Path(__file__).resolve().parent)
if _EXPERIMENTS_DIR not in sys.path:
    sys.path.insert(0, _EXPERIMENTS_DIR)

# 配置项（DEEPSEEK_API_KEY 等可能为空字符串，此处仅用于告警）
try:
    from config import DEEPSEEK_API_KEY, OPENAI_BASE_URL, LLM_MODEL
except Exception as e:  # config 不可用时也允许跑 rule/fixed/shortest
    DEEPSEEK_API_KEY = ""
    OPENAI_BASE_URL = ""
    LLM_MODEL = ""
    logging.warning("config 导入失败 (%s)，LLM 方法将不可用", e)

logger = logging.getLogger("experiments.runner")

# ---------------------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------------------
def load_queries(path: str) -> list[dict]:
    """加载 queries_v1.json，返回 queries 列表。"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data.get("queries", data if isinstance(data, list) else [])


def load_gold(path: str) -> dict:
    """加载 gold_intents_v1.json，返回 {id: intent_dict}。"""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    intents = data.get("intents", data if isinstance(data, dict) else {})
    return intents


def intent_to_dict(intent) -> dict:
    """TaskIntent(pyantic) → dict（兼容 v1/v2）。"""
    if intent is None:
        return {}
    if isinstance(intent, dict):
        return intent
    if hasattr(intent, "model_dump"):
        return intent.model_dump()
    if hasattr(intent, "dict"):
        return intent.dict()
    return {}


def _poi_name_from(ref) -> Optional[str]:
    if not ref:
        return None
    if isinstance(ref, dict):
        n = ref.get("name")
        return str(n).strip() if n else None
    n = getattr(ref, "name", None)
    return str(n).strip() if n else None


# ---------------------------------------------------------------------------
# E1：意图解析评估 (llm / rule)
# ---------------------------------------------------------------------------
def _build_turn_context(prior_turns: list[dict]) -> dict:
    """按前端 handleNlSubmit 的载荷结构，用"前轮 gold 参考轨迹"构造上下文。

    用 gold 而非预测做承接，可以隔离评测 parser+merge 本身（避免前轮预测错误
    级联放大）；前端真实行为的端到端验证由 tests/test_multiturn.py 覆盖。
    prior_turns: [{"query":..., "gold": intent_dict}, ...]
    """
    history = []
    slot = None
    for t in prior_turns:
        g = t["gold"]
        entities = {}
        if g.get("start"):
            entities["start"] = _poi_name_from(g.get("start"))
        if g.get("end"):
            entities["end"] = _poi_name_from(g.get("end"))
        if g.get("mode"):
            entities["mode"] = g.get("mode")
        history.append({"role": "user", "content": t["query"]})
        history.append({"role": "assistant",
                        "content": "已为你处理。",
                        "entities": entities})
        # 前端只在规划相关轮次更新 routeSlot
        if g.get("task_type") == "path_planning" or g.get("ambiguity"):
            slot = g
    ctx = {"history": history[-8:]}
    if slot is not None:
        ctx.update({
            "previous_intent": slot,
            "last_ambiguity": slot.get("ambiguity"),
            "start": slot.get("start"),
            "end": slot.get("end"),
            "constraints": slot.get("constraints"),
            "weights": slot.get("weights"),
        })
    return ctx


def _parse_once(method: str, text: str, context: Optional[dict]):
    """单轮解析；rule 方法在多轮时复用 parser 的承接函数保持与线上一致。"""
    if method == "llm":
        from agents.parser import parse_query
        return parse_query(text, context=context or None)
    if method == "generic":
        from experiments.baselines.generic_llm_baseline import generic_parse
        return generic_parse(text, context=context or None)
    from experiments.baselines.rule_baseline import rule_parse
    intent = rule_parse(text)
    if context:
        from agents.parser import _merge_context_with_intent
        intent = _merge_context_with_intent(intent, context, text)
    return intent


def _row_for(rid: str, text: str, category: str, pred: dict, gold_intent: dict) -> dict:
    return {
        "id": rid,
        "query": text,
        "category": category,
        "pred_task_type": pred.get("task_type", ""),
        "gold_task_type": gold_intent.get("task_type", ""),
        "pred_start": _poi_name_from(pred.get("start")) or "",
        "gold_start": _poi_name_from(gold_intent.get("start")) or "",
        "pred_end": _poi_name_from(pred.get("end")) or "",
        "gold_end": _poi_name_from(gold_intent.get("end")) or "",
        "pred_constraints": json.dumps(pred.get("constraints", {}), ensure_ascii=False),
        "gold_constraints": json.dumps(gold_intent.get("constraints", {}), ensure_ascii=False),
        "pred_weights": json.dumps(pred.get("weights"), ensure_ascii=False) if pred.get("weights") is not None else "null",
        "gold_weights": json.dumps(gold_intent.get("weights"), ensure_ascii=False) if gold_intent.get("weights") is not None else "null",
        "pred_mode": pred.get("mode", ""),
        "gold_mode": gold_intent.get("mode", ""),
        "task_type_match": int(pred.get("task_type") == gold_intent.get("task_type")),
        # 通用 LLM 基线没有工具/图/核验，该字段恒为 0；专用链路为 1（可交付可核验路线）
        "route_executable": int(bool(pred.get("route_executable")))
        if "route_executable" in pred else 1,
    }


def run_e1(method: str, queries: list[dict], gold: dict, output: str) -> dict:
    """E1：对每条 query 调用解析器，与 gold 配对后计算指标。

    普通条目计 1 对；L5 多轮条目（含 conversation 字段）按轮逐对评测，
    row id 形如 L5_003#t2，n_queries 报告对话组数，n_pairs 报告实际配对数。
    """
    from experiments.evaluator import evaluate_all

    pairs = []  # (pred_dict, gold_dict)
    rows = []
    n_conversations = 0
    n_turns = 0

    for q in queries:
        qid = q["id"]
        category = q.get("category", "")
        conversation = q.get("conversation")

        if conversation:
            # 多轮：逐轮解析，上下文来自前轮 gold
            n_conversations += 1
            prior: list[dict] = []
            for ti, turn in enumerate(conversation, start=1):
                text = turn["query"]
                turn_gold = turn["gold"]
                context = _build_turn_context(prior) if ti > 1 else None
                try:
                    intent = _parse_once(method, text, context)
                except Exception as e:
                    logger.warning("[%s#t%d] 解析异常: %s", qid, ti, e)
                    intent = None
                pred = intent_to_dict(intent)
                rid = f"{qid}#t{ti}"
                pairs.append((pred, turn_gold))
                rows.append(_row_for(rid, text, category, pred, turn_gold))
                prior.append({"query": text, "gold": turn_gold})
                n_turns += 1
            continue

        gold_intent = gold.get(qid) or q.get("gold", {})
        try:
            intent = _parse_once(method, q["query"], None)
        except Exception as e:
            logger.warning("解析异常 [%s]: %s", qid, e)
            intent = None
        pred = intent_to_dict(intent)
        pairs.append((pred, gold_intent))
        rows.append(_row_for(qid, q["query"], category, pred, gold_intent))

    metrics = evaluate_all(pairs)

    # 写 per-query CSV
    _write_csv(output, rows)
    # 写汇总 JSON
    summary_path = _summary_path(output)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "method": method,
            "metrics": metrics,
            "n_queries": len(queries),
            "n_conversations": n_conversations,
            "n_conversation_turns": n_turns,
        }, f, ensure_ascii=False, indent=2)

    print(f"\n[E1/{method}] 指标汇总 (条目={len(queries)}, 配对={metrics['n_pairs']}, "
          f"多轮对话={n_conversations} 组/{n_turns} 轮):")
    for k, v in metrics.items():
        print(f"  {k:32s} {v}")
    print(f"  per-query CSV -> {output}")
    print(f"  summary JSON  -> {summary_path}")
    return metrics


# ---------------------------------------------------------------------------
# E2：路径质量评估 (fixed / shortest)
# ---------------------------------------------------------------------------
def _resolve_node(G, poi_name: str):
    """POI 名称 → 路网节点 ID（GCJ-02→WGS-84→nearest_node）。失败返回 None。"""
    from spatial.poi import find_poi_ambiguous
    from spatial.coord_transform import gcj02_to_wgs84
    from spatial.network import get_nearest_node

    poi, _alts = find_poi_ambiguous(poi_name)
    if not poi:
        return None, f"POI 未找到: {poi_name}"
    lng_wgs, lat_wgs = gcj02_to_wgs84(poi["lon"], poi["lat"])
    try:
        node = get_nearest_node(G, lng_wgs, lat_wgs)
        return node, None
    except Exception as e:
        return None, f"最近节点查找失败: {e}"


def run_e2(method: str, queries: list[dict], gold: dict, output: str) -> dict:
    """E2：对每条 path_planning query 计算路径，输出路径质量指标。"""
    from spatial.network import load_or_download_network

    if method == "fixed":
        from experiments.baselines.fixed_weight_baseline import fixed_weight_route as route_fn
    elif method == "shortest":
        from experiments.baselines.shortest_path_baseline import shortest_path_route as route_fn
    else:
        raise ValueError(f"未知 E2 方法: {method}")

    print(f"[E2/{method}] 加载路网 ...")
    G = load_or_download_network()

    rows = []
    ok_count = 0
    for q in queries:
        qid = q["id"]
        query = q["query"]
        category = q.get("category", "")
        gold_intent = gold.get(qid) or q.get("gold", {})

        # 多轮对话不进 E2（承接由 E1 的逐轮评测覆盖，避免按首句误算）
        if q.get("conversation"):
            rows.append({"id": qid, "query": query, "category": category,
                         "status": "skipped", "note": "multiturn conversation"})
            continue

        # 只评估 path_planning 类（其他类型无路径可言）
        if gold_intent.get("task_type") != "path_planning":
            rows.append({"id": qid, "query": query, "category": category,
                         "status": "skipped", "note": f"task_type={gold_intent.get('task_type')}"})
            continue

        start_name = _poi_name_from(gold_intent.get("start"))
        end_name = _poi_name_from(gold_intent.get("end"))
        mode = gold_intent.get("mode", "walk")

        if not start_name or not end_name:
            rows.append({"id": qid, "query": query, "category": category,
                         "status": "skipped", "note": "起终点缺失"})
            continue

        sn, err = _resolve_node(G, start_name)
        if err:
            rows.append({"id": qid, "query": query, "category": category,
                         "status": "error", "note": f"start: {err}"})
            continue
        en, err = _resolve_node(G, end_name)
        if err:
            rows.append({"id": qid, "query": query, "category": category,
                         "status": "error", "note": f"end: {err}"})
            continue

        constraints = gold_intent.get("constraints", {})
        try:
            res = route_fn(G, sn, en, constraints=constraints, mode=mode) if method == "fixed" \
                else route_fn(G, sn, en, mode=mode)
        except ValueError as e:
            # 语义化失败（如驾车不可达）：这是系统的正常输出，应与基础设施错误区分，
            # 论文中可作为"模式可行性"指标单独报告
            rows.append({"id": qid, "query": query, "category": category,
                         "status": "unreachable", "note": str(e)})
            continue
        except Exception as e:
            rows.append({"id": qid, "query": query, "category": category,
                         "status": "error", "note": f"route: {e}"})
            continue

        ok_count += 1
        rows.append({
            "id": qid, "query": query, "category": category, "status": "ok",
            "mode": res.get("mode", mode),
            "recommended_length_m": res.get("recommended_length_m", ""),
            "shortest_length_m": res.get("shortest_length_m", ""),
            "overlap_rate": res.get("overlap_rate", ""),
            "slope_avg_recommended": res.get("slope_avg_recommended", ""),
            "scenery_avg_recommended": res.get("scenery_avg_recommended", ""),
            "slope_avg_shortest": res.get("slope_avg_shortest", ""),
            "scenery_avg_shortest": res.get("scenery_avg_shortest", ""),
            "duration_min": res.get("duration_min", ""),
            "length_capped": res.get("length_capped", ""),
            "filter_status": res.get("filter_status", ""),
        })

    # 汇总：对成功计算的路径取均值
    ok_rows = [r for r in rows if r.get("status") == "ok"]
    def _avg(key):
        vals = [r[key] for r in ok_rows if isinstance(r.get(key), (int, float))]
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    summary = {
        "method": method,
        "n_queries": len(queries),
        "n_path_planning_ok": ok_count,
        "n_skipped": sum(1 for r in rows if r.get("status") == "skipped"),
        "n_unreachable": sum(1 for r in rows if r.get("status") == "unreachable"),
        "n_error": sum(1 for r in rows if r.get("status") == "error"),
        "avg_recommended_length_m": _avg("recommended_length_m"),
        "avg_shortest_length_m": _avg("shortest_length_m"),
        "avg_overlap_rate": _avg("overlap_rate"),
        "avg_slope_recommended": _avg("slope_avg_recommended"),
        "avg_scenery_recommended": _avg("scenery_avg_recommended"),
        "avg_slope_shortest": _avg("slope_avg_shortest"),
        "avg_scenery_shortest": _avg("scenery_avg_shortest"),
        "avg_duration_min": _avg("duration_min"),
    }

    _write_csv(output, rows)
    summary_path = _summary_path(output)
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(f"\n[E2/{method}] 路径质量汇总 (ok={ok_count}/{len(queries)}):")
    for k, v in summary.items():
        print(f"  {k:32s} {v}")
    print(f"  per-query CSV -> {output}")
    print(f"  summary JSON  -> {summary_path}")
    return summary


# ---------------------------------------------------------------------------
# CSV / 路径工具
# ---------------------------------------------------------------------------
def _write_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    # 统一列：取所有 row 的键并集，保持首次出现顺序
    fieldnames = []
    for r in rows:
        for k in r.keys():
            if k not in fieldnames:
                fieldnames.append(k)
    with open(out, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)


def _summary_path(output: str) -> str:
    """汇总 JSON 路径：与输出同目录，<stem>_summary.json。"""
    p = Path(output)
    return str(p.with_name(p.stem + "_summary.json"))


# ---------------------------------------------------------------------------
# CLI 主入口
# ---------------------------------------------------------------------------
def main(argv: Optional[list] = None) -> int:
    parser = argparse.ArgumentParser(
        description="漫步珞珈 实验评估 CLI (E1 意图解析 / E2 路径质量)",
    )
    parser.add_argument("--method", required=True,
                        choices=["llm", "rule", "generic", "fixed", "shortest"],
                        help="评估方法：llm/rule/generic 走 E1，fixed/shortest 走 E2")
    parser.add_argument("--queries", required=True, help="queries JSON 路径")
    parser.add_argument("--gold", required=True, help="gold intents JSON 路径")
    parser.add_argument("--output", required=True, help="输出 CSV 路径")
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    queries = load_queries(args.queries)
    gold = load_gold(args.gold)
    print(f"加载 {len(queries)} 条 query，{len(gold)} 条 gold")

    if args.method in ("llm", "rule", "generic"):
        if args.method in ("llm", "generic") and not DEEPSEEK_API_KEY:
            print("[警告] DEEPSEEK_API_KEY 未配置，"
                  f"{args.method} 方法将返回未解析占位结果（结果仍可产出）")
        run_e1(args.method, queries, gold, args.output)
    else:
        run_e2(args.method, queries, gold, args.output)

    return 0


if __name__ == "__main__":
    sys.exit(main())
