"""
珞珈智行 (WHU-Walker) V1 全局配置
"""
import math
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


def _env_float(key: str, default: float) -> float:
    """读取 float 类型 env 变量，非法值 fallback 到默认值（避免 .env typo 导致 app 启动崩溃）"""
    raw = os.getenv(key)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except (ValueError, TypeError):
        import warnings
        warnings.warn(f"config: env {key}={raw!r} 非法，使用默认值 {default}", stacklevel=2)
        return default


def _read_cgroup_cpu_quota(cgroup_root: Path) -> float | None:
    """Return the visible cgroup CPU quota in cores, if one is configured."""
    cpu_max = cgroup_root / "cpu.max"
    try:
        parts = cpu_max.read_text(encoding="ascii").split()
    except OSError:
        parts = []
    if len(parts) >= 2 and parts[0] != "max":
        try:
            quota, period = float(parts[0]), float(parts[1])
        except ValueError:
            quota = period = 0
        if quota > 0 and period > 0 and math.isfinite(quota / period):
            return quota / period

    for base in (cgroup_root / "cpu", cgroup_root):
        try:
            quota = float((base / "cpu.cfs_quota_us").read_text(encoding="ascii").strip())
            period = float((base / "cpu.cfs_period_us").read_text(encoding="ascii").strip())
        except (OSError, ValueError):
            continue
        if quota > 0 and period > 0 and math.isfinite(quota / period):
            return quota / period
    return None


def resolve_vision_cpu_threads(value: str | None = None, *, cpu_count: int | None = None,
                               cgroup_root: Path | None = None) -> int:
    """Resolve an explicit thread override or a conservative local/container default."""
    setting = os.getenv("VISION_CPU_THREADS", "auto") if value is None else value
    setting = str(setting).strip().lower()
    if setting not in {"", "auto"}:
        try:
            return max(1, int(setting))
        except ValueError as error:
            raise ValueError("VISION_CPU_THREADS 必须是正整数或 auto") from error

    if cpu_count is None:
        process_cpu_count = getattr(os, "process_cpu_count", None)
        cpu_count = process_cpu_count() if callable(process_cpu_count) else None
        cpu_count = cpu_count or os.cpu_count() or 1
    available = max(1, int(cpu_count))
    quota = _read_cgroup_cpu_quota(cgroup_root or Path("/sys/fs/cgroup"))
    if quota is not None:
        available = min(available, max(1, math.ceil(quota)))
    return max(1, min(4, available))


# ===== LLM 配置（DEC-006: DeepSeek V4-Flash）=====
# 兼容 OpenAI SDK：仅改 base_url + model id，无需替换 SDK
DEEPSEEK_API_KEY = os.getenv("DEEPSEEK_API_KEY", "")
OPENAI_API_KEY = DEEPSEEK_API_KEY  # 兼容 openai SDK 字段名
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.deepseek.com/v1")
LLM_MODEL = os.getenv("LLM_MODEL", "deepseek-chat")  # DeepSeek V4-Flash 别名

# ===== 高德地图配置 =====
AMAP_KEY = os.getenv("AMAP_KEY", "")
AMAP_SECURITY_CODE = os.getenv("AMAP_SECURITY_CODE", "")  # 高德 JS API 2.0 安全密钥

# ===== 管理员口令（上报/删除路况需登录）=====
# 不设置时禁止所有路况写操作（更安全）；生产环境务必配置
ROAD_CONDITION_ADMIN_PASSWORD = os.getenv("ROAD_CONDITION_ADMIN_PASSWORD", "")
# 管理者系统对接 Token（如保卫部后台）：请求头 X-Admin-Token 或 Authorization: Bearer
# 不设置时仅允许网页 session 登录
ROAD_CONDITION_ADMIN_TOKEN = os.getenv("ROAD_CONDITION_ADMIN_TOKEN", "")

# Aerial inspection is optional and isolated in `python -m vision.worker`.
VISION_UPLOAD_DIR = os.getenv("VISION_UPLOAD_DIR", os.path.join(os.path.dirname(__file__), "data", "vision", "uploads"))
VISION_MODEL_PATH = os.getenv("VISION_MODEL_PATH", "")
VISION_MODEL_ID = os.getenv("VISION_MODEL_ID", "visdrone-rtdetrv4-s")
VISION_MODEL_REVISION = os.getenv(
    "VISION_MODEL_REVISION", "820fca3d962243f7d611dd35cf2a64d267635499",
)
VISION_ACCIDENT_ENABLED = os.getenv("VISION_ACCIDENT_ENABLED", "false").strip().lower() in {
    "1", "true", "yes", "on",
}
VISION_ACCIDENT_MODEL_PATH = os.getenv("VISION_ACCIDENT_MODEL_PATH", "")
VISION_ACCIDENT_REVISION = os.getenv("VISION_ACCIDENT_REVISION", "unverified")
VISION_ACCIDENT_THRESHOLD = _env_float("VISION_ACCIDENT_THRESHOLD", 0.85)
VISION_ACCIDENT_DENSE_TRIGGER_THRESHOLD = _env_float(
    "VISION_ACCIDENT_DENSE_TRIGGER_THRESHOLD", 0.5,
)
VISION_ACCIDENT_DENSE_WINDOW_SECONDS = _env_float(
    "VISION_ACCIDENT_DENSE_WINDOW_SECONDS", 2.0,
)
VISION_FLOOD_MODEL_PATH = os.getenv("VISION_FLOOD_MODEL_PATH", "")
VISION_FLOOD_ENABLED = os.getenv("VISION_FLOOD_ENABLED", "false").strip().lower() in {
    "1", "true", "yes", "on",
}
VISION_FLOOD_REVISION = os.getenv("VISION_FLOOD_REVISION", "unverified")
VISION_FLOOD_MIN_AREA_RATIO = _env_float("VISION_FLOOD_MIN_AREA_RATIO", 0.08)
VISION_CONFIDENCE_THRESHOLD = float(os.getenv("VISION_CONFIDENCE_THRESHOLD", "0.369"))
VISION_TRACKING_MIN_CONFIDENCE = _env_float("VISION_TRACKING_MIN_CONFIDENCE", 0.1)
VISION_CPU_THREADS = resolve_vision_cpu_threads()
VISION_MAX_MEDIA_BYTES = int(os.getenv("VISION_MAX_MEDIA_BYTES", str(24 * 1024 * 1024)))
VISION_MAX_VIDEO_BYTES = int(os.getenv("VISION_MAX_VIDEO_BYTES", str(1024 * 1024 * 1024)))
VISION_CHUNK_BYTES = int(os.getenv("VISION_CHUNK_BYTES", str(8 * 1024 * 1024)))
VISION_MAX_FRAME_PIXELS = int(os.getenv("VISION_MAX_FRAME_PIXELS", "12500000"))
VISION_SEGMENT_FRAMES = int(os.getenv(
    "VISION_SEGMENT_FRAMES", os.getenv("VISION_MAX_VIDEO_FRAMES", "180"),
))
VISION_PROGRESS_UPDATE_SECONDS = float(os.getenv("VISION_PROGRESS_UPDATE_SECONDS", "2"))
VISION_MAX_VIDEO_FRAMES = VISION_SEGMENT_FRAMES  # legacy setting name; now per-segment, not a clip cap
VISION_SAMPLE_FPS = float(os.getenv("VISION_SAMPLE_FPS", "5"))
VISION_MAX_ANALYSIS_SECONDS = float(os.getenv("VISION_MAX_ANALYSIS_SECONDS", "3600"))
VISION_MAX_EVENT_AGE_SECONDS = int(os.getenv("VISION_MAX_EVENT_AGE_SECONDS", "900"))

# ===== 武汉大学校园范围 (经纬度边界) — .env 可覆盖 =====
WHU_BBOX = {
    "north": _env_float("WHU_BBOX_NORTH", 30.5510),
    "south": _env_float("WHU_BBOX_SOUTH", 30.5170),
    "east":  _env_float("WHU_BBOX_EAST",  114.3880),
    "west":  _env_float("WHU_BBOX_WEST",  114.3460),
}

# 地图默认中心点 (武大核心区) — .env 可覆盖
MAP_CENTER = {
    "lat": _env_float("MAP_CENTER_LAT", 30.5365),
    "lng": _env_float("MAP_CENTER_LNG", 114.3630),
}
MAP_ZOOM = int(_env_float("MAP_ZOOM", 16))

# ===== 三学部校园边界多边形（GCJ-02 坐标，与 pois.json 坐标系一致）=====
# 用途：POI 校外点过滤、路网裁剪（保证路线不出校）。
# 路网为 WGS-84，使用时需经 gcj02_to_wgs84 逐点转换；裁剪时外扩缓冲约 120m 保留校门过街连接。
# 边界依据：OSM POI 坐标极值 + 珞瑜路/八一路/东湖南路/东湖 实际地物。
CAMPUS_POLYS_GCJ = {
    # 文理学部（八一路以北、珞珈山、枫园、梅园、桂园、樱园、湖滨/凌波门；
    #   南界八一路并绕开信息学部北侧突出部；东北湖滨、凌波门片区归入本学部）
    "文理学部": [
        (114.3540, 30.5305), (114.3555, 30.5303),
        (114.3560, 30.5302), (114.3560, 30.5320), (114.3625, 30.5320),
        (114.3630, 30.5303), (114.3648, 30.5303), (114.3648, 30.5299),
        (114.3682, 30.5299), (114.3682, 30.5303),
        (114.3765, 30.5310),
        (114.3785, 30.5370), (114.3785, 30.5405), (114.3780, 30.5445),
        (114.3770, 30.5485), (114.3700, 30.5495), (114.3706, 30.5434),
        (114.3703, 30.5424), (114.3700, 30.5421), (114.3703, 30.5431),
        (114.3696, 30.5434), (114.3694, 30.5445), (114.3686, 30.5433),
        (114.3678, 30.5427), (114.3685, 30.5423), (114.3694, 30.5424),
        (114.3680, 30.5410), (114.3665, 30.5408), (114.3620, 30.5419),
        (114.3620, 30.5406), (114.3584, 30.5406),
        (114.3584, 30.54161), (114.3575, 30.54165), (114.3568, 30.5417),
        (114.3555, 30.5393), (114.3518, 30.5350),
    ],
    # 工学部（东湖南路以南原武水校区：松园/工学部教楼群/田园茶港一线；
    #   东以松园操场-湖滨宿舍一线与文理学部相接，锯齿顶点为实测楼群/校门坐标反推）
    "工学部": [
        (114.3535, 30.5386), (114.3548, 30.5408), (114.3560, 30.5438),
        (114.3620, 30.5490), (114.3700, 30.5495), (114.3706, 30.5434),
        (114.3703, 30.5424), (114.3700, 30.5421), (114.3703, 30.5431),
        (114.3696, 30.5434), (114.3694, 30.5445), (114.3686, 30.5433),
        (114.3678, 30.5427), (114.3685, 30.5423), (114.3694, 30.5424),
        (114.3680, 30.5410), (114.3665, 30.5408), (114.3620, 30.5419),
        (114.3620, 30.5406), (114.3584, 30.5406),
        (114.3584, 30.54161), (114.3575, 30.54165), (114.3568, 30.5417),
        (114.3555, 30.5393),
    ],
    # 信息学部（八一路/珞喻路之间，原武测；北界抬至八一路南缘并纳入白锦街/求是大道片区；
    #   珞南门（附中门）处南缘留齿状缺口归入文理学部）
    "信息学部": [
        (114.3540, 30.5306), (114.3560, 30.5306),
        (114.3560, 30.5319), (114.3625, 30.5319), (114.3630, 30.5304),
        (114.3648, 30.5304), (114.3648, 30.5299), (114.3682, 30.5299),
        (114.3682, 30.5304),
        (114.3815, 30.5303), (114.3818, 30.5230), (114.3790, 30.5150),
        (114.3700, 30.5128), (114.3600, 30.5130), (114.3545, 30.5170),
        (114.3540, 30.5240),
    ],
}

# 坐标在校内但名称明显指向其他校区/地点的黑名单（OSM 误 tag，导航会误导）
POI_NAME_BLACKLIST = (
    "国家网络安全学院",   # 网安学院主体在东西湖临空港校区，不在三学部
)

# ===== 多因素成本默认权重（DEC-004 修订 + DEC-010 LLM 直接生成）=====
# 默认场景"从A到B"用户未表达偏好时使用；接近普通导航基线
# 当 LLM 识别到用户空间偏好时，直接输出 weights 覆盖此默认值
DEFAULT_WEIGHTS = {
    "distance": 0.5,   # 距离作为基础（默认场景接近普通导航）
    "slope":    0.2,   # 坡度权重低（武大主要活动区相对平坦）
    "scenery":  0.3,   # 景观权重适中
}

# 权重校验范围（DEC-010 防 LLM 输出极端值）
WEIGHT_BOUNDS = {"min": 0.05, "max": 0.8}

# DEC-011 路径计算参数
PATH_LENGTH_CAP_MULTIPLIER = 3.0
PATH_LENGTH_CAP_MAX = 2000.0

# 注：CONSTRAINT_WEIGHT_MAP 已移除（DEC-010）
# LLM 在解析阶段直接输出 weights 字段，不再通过约束等级规则映射
# 约束等级（constraints）仍保留，用于 DEC-011 硬约束过滤（如 slope=avoid 过滤陡坡路段）

# ===== POI 数据（迁移到 data/pois.json，此处仅喂给 LLM 解析 Agent 拼 Prompt） =====
def _load_whu_pois_for_prompt() -> dict:
    """从 data/pois.json 读取 POI，返回 {name: {type, desc}} 字典供 parser 使用。
    若 JSON 不可用，返回空字典（LLM Prompt 中 POI 列表将为空，已及时发现）。
    """
    import json as _json
    from pathlib import Path as _Path
    pois_path = _Path(__file__).parent / "data" / "pois.json"
    try:
        with open(pois_path, "r", encoding="utf-8") as f:
            raw = _json.load(f)
        result = {}
        for poi in raw.get("pois", []):
            result[poi["name"]] = {
                "type": poi.get("type", "landmark"),
                "desc": poi.get("description", ""),
            }
        return result
    except Exception:
        return {}


WHU_POIS = _load_whu_pois_for_prompt()

# ===== 文件路径 =====
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "output")
ROAD_NETWORK_CACHE = os.path.join(DATA_DIR, "whu_road_network.graphml")
ROAD_ANNOTATIONS_PATH = os.path.join(DATA_DIR, "road_annotations.json")
EDGE_OVERRIDES_PATH = os.path.join(DATA_DIR, "edge_overrides.json")
