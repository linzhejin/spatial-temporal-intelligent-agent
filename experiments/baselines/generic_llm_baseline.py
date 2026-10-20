"""B2 通用 LLM 直答基线（Generic LLM, 无工具）。

对照对象不是「规则解析」或「固定权重」，而是「把同一个模型直接当聊天接口用」：
同一模型、同一温度（0.0），**不注册任何工具**，只要求它读完 query 后直接给出
结构化意图。它模拟「通用智能体」在没有地理计算能力时的表现。

它用于回答一个具体问题：专用时空智能体（工具编排 + 结果核验）相对通用 LLM
直答多出了什么。差异体现在两条可度量的线上：

1. 意图解析质量（由 experiments/evaluator.py 的 E1 指标计算）；
2. 可执行性：通用 LLM 不产出边序列、坐标与核验结论，因此 route_executable
   恒为 False——路线类请求只能"描述"，无法"交付一条可核验的路线"。

注意：本基线不做任何地名核验、不做路网吸附、不读任务卡、不注入画像。
它拿到的信息与普通聊天用户拿到的完全一致。
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """你是武汉大学校园出行助手。请把用户的话解析成如下 JSON，只输出 JSON：

{
  "task_type": "path_planning | poi_query | help | chat",
  "start": {"name": "起点名"} 或 null,
  "end": {"name": "终点名"} 或 null,
  "mode": "walk | bike | drive",
  "constraints": {"distance": "short|medium|long",
                  "slope": "avoid|normal|prefer",
                  "scenery": "high|normal"},
  "weights": {"distance": 0.0, "slope": 0.0, "scenery": 0.0} 或 null,
  "ambiguity": "若信息不足，说明缺什么；否则 null"
}

规则：
- 用户没表达偏好时 weights 填 null。
- 不要编造用户没说的起点或终点。
- 不要输出任何解释文字。"""


def _client():
    """按项目配置创建 OpenAI 兼容客户端；不可用时返回 None。"""
    try:
        import httpx
        from openai import OpenAI
        import config

        if not getattr(config, "DEEPSEEK_API_KEY", ""):
            return None
        return OpenAI(
            api_key=config.DEEPSEEK_API_KEY,
            base_url=config.OPENAI_BASE_URL,
            default_headers={"Accept-Encoding": "gzip, deflate"},
            timeout=httpx.Timeout(connect=5.0, read=25.0, write=10.0, pool=5.0),
        )
    except Exception as e:
        logger.warning("通用 LLM 基线客户端不可用: %s", e)
        return None


def _fallback_intent(query: str) -> dict:
    """无 API key 时的确定性占位结果：明确标为未解析，绝不伪造地点。"""
    return {
        "task_type": "unknown",
        "start": None,
        "end": None,
        "mode": "walk",
        "constraints": {},
        "weights": None,
        "ambiguity": "LLM 不可用（未配置 API key），本条未解析",
    }


def _extract_json(text: str) -> dict | None:
    """从模型输出里取第一段 JSON 对象（容忍代码块包裹）。"""
    if not text:
        return None
    text = text.strip()
    if text.startswith("```"):
        lines = [line for line in text.splitlines() if not line.strip().startswith("```")]
        text = "\n".join(lines).strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        parsed = json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
    return parsed if isinstance(parsed, dict) else None


_CONSTRAINT_SPECS = {
    "distance": ({"short", "medium", "long"}, "medium"),
    "slope": ({"avoid", "normal", "prefer"}, "normal"),
    "scenery": ({"high", "normal"}, "normal"),
}


def _normalize_constraints(raw) -> dict:
    """补齐三个约束维度并校验取值。

    专用链路里 Constraints 一定有三个键（有默认值）；缺键会让评测器的
    macro-F1 拿到 None 而无法分组。这里按同一套默认值（medium/normal/normal）
    补齐，非法枚举同样回落到默认值，保证对照是同一口径而不是放宽口径。
    """
    raw = raw if isinstance(raw, dict) else {}
    return {
        dim: (raw.get(dim) if raw.get(dim) in allowed else default)
        for dim, (allowed, default) in _CONSTRAINT_SPECS.items()
    }


def _normalize(parsed: dict) -> dict:
    """把模型输出收敛成 evaluator 认识的形状。"""
    weights = parsed.get("weights")
    mode = parsed.get("mode")
    return {
        "task_type": parsed.get("task_type") or "unknown",
        "start": parsed.get("start") if isinstance(parsed.get("start"), dict) else None,
        "end": parsed.get("end") if isinstance(parsed.get("end"), dict) else None,
        "mode": mode if mode in ("walk", "bike", "drive") else "walk",
        "constraints": _normalize_constraints(parsed.get("constraints")),
        "weights": weights if isinstance(weights, dict) else None,
        "ambiguity": parsed.get("ambiguity"),
        # 通用基线没有工具、没有图、没有核验：路线不可执行
        "route_executable": False,
        "geometry_verified": False,
    }


def generic_parse(query: str, context: dict = None) -> dict:
    """通用 LLM 直答解析（无工具、无地名核验、无路网）。

    context 会被忽略——通用智能体没有多轮空间状态，这是需要被对照的差异之一。
    """
    client = _client()
    if client is None:
        return _fallback_intent(query)

    try:
        import config
        response = client.chat.completions.create(
            model=config.LLM_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
            temperature=0.0,
            max_tokens=400,
        )
        text = (response.choices[0].message.content or "").strip()
    except Exception as e:
        logger.warning("通用 LLM 基线调用失败: %s", e)
        return _fallback_intent(query)

    parsed = _extract_json(text)
    if parsed is None:
        return _fallback_intent(query)
    return _normalize(parsed)


if __name__ == "__main__":
    samples = [
        "从珞珈门到樱顶",
        "我想避开陡坡，从行政楼到枫园",
        "樱顶在哪里",
        "今天天气怎么样",
    ]
    for q in samples:
        print(q)
        print("  ->", json.dumps(generic_parse(q), ensure_ascii=False))
