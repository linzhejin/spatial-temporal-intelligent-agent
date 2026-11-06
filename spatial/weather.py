"""
天气感知模块

拉取高德实时天气（武汉），带内存缓存（TTL 30 分钟），
并将天气映射为对步行/骑行路线的动态影响：

  - 雨/雪/冻雨/冰雹：路面湿滑 → 陡坡、台阶边额外惩罚（智能避坡）
  - 高温（≥35°C）：倾向树荫/景观路（scenery 权重上调）
  - 大风：提示（对步行影响小）
  - 雾/霾：能见度低提示

天气是校园步行场景下唯一可自动获取、每天真实变化的动态环境因素
（高德不采集校内步道施工/拥堵，校内事件需走 road_conditions 上报渠道）。
"""

import json
import logging
import os
import threading
import time
import urllib.parse
import urllib.request
from typing import Optional

logger = logging.getLogger(__name__)

# 武汉市 adcode
WUHAN_ADCODE = "420100"
WEATHER_API = "https://restapi.amap.com/v3/weather/weatherInfo"

# 缓存 TTL：30 分钟（天气变化慢，避免每次请求都调 API）
_CACHE_TTL = 30 * 60
_cache_lock = threading.Lock()
_cache = {"data": None, "ts": 0.0}


def _get_web_key() -> str:
    """读取高德 Web 服务 key（优先 AMAP_WEB_KEY，回退 AMAP_KEY）。"""
    key = os.getenv("AMAP_WEB_KEY") or os.getenv("AMAP_KEY", "")
    if key:
        return key.strip()
    # 兜底：从 .env 读取（config 未显式加载 AMAP_WEB_KEY 时）
    try:
        from pathlib import Path
        env_path = Path(__file__).resolve().parent.parent / ".env"
        if env_path.exists():
            for line in env_path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line.startswith("AMAP_WEB_KEY") or line.startswith("AMAP_KEY"):
                    return line.split("=", 1)[1].strip().strip('"').strip("'")
    except Exception:
        pass
    return ""


def fetch_weather_live(force: bool = False) -> Optional[dict]:
    """
    获取武汉实时天气（带缓存）。

    Returns:
        {
            "weather": "晴",
            "temperature": 25.0,
            "humidity": "60",
            "winddirection": "北",
            "windpower": "≤3",
            "reporttime": "2026-09-08 08:00:00",
        }
        失败返回 None。
    """
    with _cache_lock:
        if not force and _cache["data"] is not None:
            if time.time() - _cache["ts"] < _CACHE_TTL:
                return dict(_cache["data"])

    key = _get_web_key()
    if not key:
        logger.warning("未配置高德 Web key，跳过天气获取")
        return None

    params = urllib.parse.urlencode({
        "city": WUHAN_ADCODE,
        "key": key,
        "extensions": "base",  # base=实时天气
    })
    url = f"{WEATHER_API}?{params}"
    try:
        from agents.deadline import limited_timeout
        with urllib.request.urlopen(url, timeout=limited_timeout(8)) as resp:
            data = json.loads(resp.read())
        if data.get("status") != "1":
            logger.warning("天气 API 返回异常: %s", data.get("info"))
            return _cache["data"]
        lives = data.get("lives") or []
        if not lives:
            return _cache["data"]
        live = lives[0]
        result = {
            "weather": live.get("weather", "未知"),
            "temperature": float(live.get("temperature", 0) or 0),
            "humidity": live.get("humidity", ""),
            "winddirection": live.get("winddirection", ""),
            "windpower": live.get("windpower", ""),
            "reporttime": live.get("reporttime", ""),
        }
        with _cache_lock:
            _cache["data"] = result
            _cache["ts"] = time.time()
        return dict(result)
    except Exception as e:
        logger.warning("天气获取失败: %s", e)
        return _cache["data"]


# ====== 天气 → 路况/权重影响 ======

# 湿滑天气关键词（路面湿滑，陡坡台阶危险）
_SLIPPERY_KEYWORDS = ("雨", "雪", "冻", "冰雹", "霰")
# 低能见度关键词
_LOW_VISIBILITY_KEYWORDS = ("雾", "霾")


def classify_weather(weather: Optional[dict]) -> dict:
    """
    将原始天气归类为对路线的影响标志。

    Returns:
        {
            "slippery": bool,      # 路面湿滑（雨雪）→ 陡坡台阶加惩罚
            "low_visibility": bool,# 低能见度（雾霾）
            "hot": bool,           # 高温 → 倾向树荫景观
            "cold": bool,          # 严寒
            "label": str,          # 人类可读影响标签
            "advice": str,         # 给用户的提示语
        }
    """
    if not weather:
        return {"slippery": False, "low_visibility": False, "hot": False,
                "cold": False, "label": "", "advice": ""}

    desc = weather.get("weather", "")
    temp = weather.get("temperature", 0)

    slippery = any(k in desc for k in _SLIPPERY_KEYWORDS)
    low_vis = any(k in desc for k in _LOW_VISIBILITY_KEYWORDS)
    hot = temp >= 35
    cold = temp <= 0

    labels = []
    advice_parts = []
    if slippery:
        labels.append("路面湿滑")
        advice_parts.append("今天有" + desc + "，路面湿滑，已为你尽量避开陡坡和台阶")
    if hot:
        labels.append("高温")
        advice_parts.append("今天高温 " + str(int(temp)) + "°C，已优先安排树荫较多的景观路，注意防暑")
    if low_vis:
        labels.append("能见度低")
        advice_parts.append("今天有" + desc + "，能见度较低，请注意安全")
    if cold:
        advice_parts.append("今天气温较低，注意保暖")

    return {
        "slippery": slippery,
        "low_visibility": low_vis,
        "hot": hot,
        "cold": cold,
        "label": "、".join(labels),
        "advice": "。".join(advice_parts) + ("。" if advice_parts else ""),
    }


def weather_slope_penalty(weather_info: Optional[dict]) -> dict:
    """
    湿滑天气下，对陡坡/台阶边的额外惩罚 {(slope_level): multiplier}。
    返回按 slope_level 的惩罚系数，由 routing 层应用到对应边。
    """
    if not weather_info:
        return {}
    impact = classify_weather(weather_info)
    if impact["slippery"]:
        # 湿滑：level5（陡坡/台阶）×2.5，level4 ×1.5
        return {5: 2.5, 4: 1.5}
    return {}


def adjust_weights_for_weather(base_weights: dict, weather_info: Optional[dict]) -> dict:
    """
    高温天气提升 scenery 权重（倾向树荫景观路），其余按比例缩放归一化。
    """
    if not weather_info:
        return dict(base_weights)
    impact = classify_weather(weather_info)
    w = dict(base_weights)
    if impact["hot"]:
        # scenery 权重 +0.15，distance -0.1，slope -0.05
        w["scenery"] = w.get("scenery", 0.3) + 0.15
        w["distance"] = max(0.05, w.get("distance", 0.5) - 0.10)
        w["slope"] = max(0.05, w.get("slope", 0.2) - 0.05)
    # 归一化
    total = sum(w.values())
    if total > 0:
        w = {k: v / total for k, v in w.items()}
    return w
