"""相机配置与像素内参。

影像自动对应路段需要显式内参，不能用未经验证的 SRT 焦距直接投影。
本模块提供默认机型配置、环境变量覆盖，以及由 35mm 等效焦距换算像素内参。

约定：图像尺寸为原始像素；主点取图像中心；方形像素 fy=fx。
标定状态由配置中的 calibrated/residual_m 决定，未标定只能作为参考。
"""
from __future__ import annotations

import json
import os
from pathlib import Path

DEFAULT_PROFILE = "dji-mavic-3-wide"

PROFILES = {
    "dji-mavic-3-wide": {
        "name": "DJI Mavic 3 (wide)",
        "sensor_width_mm": 17.3,
        "focal_35mm": 24.0,
        "image_width": 3840,
        "image_height": 2160,
        "calibrated": False,
        "residual_m": None,
        "ground_offset_m": 0.0,
        "fx_px": None,
        "fy_px": None,
    },
}


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None or value == "":
        return float(default)
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float(default)
    return result


def _load_overrides() -> dict:
    path = os.getenv("VISION_CAMERA_PROFILE_PATH", "").strip()
    if not path:
        return {}
    candidate = Path(path)
    if not candidate.is_file():
        return {}
    try:
        data = json.loads(candidate.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def resolve_profile(name: str | None = None) -> dict:
    """返回合并后的相机配置；环境变量覆盖优先。"""
    overrides = _load_overrides()
    profile_name = name or overrides.get("name") or os.getenv("VISION_CAMERA_PROFILE") or DEFAULT_PROFILE
    base = dict(PROFILES.get(profile_name) or PROFILES[DEFAULT_PROFILE])
    base.update({key: value for key, value in overrides.items() if value is not None})
    base["name"] = profile_name
    base["ground_offset_m"] = _env_float("VISION_GROUND_OFFSET_M", base.get("ground_offset_m", 0.0))
    return base


def intrinsics_for(width: int | float, height: int | float, profile: dict | None = None) -> dict:
    """由图像尺寸与相机配置生成像素内参。"""
    image_width = float(width)
    image_height = float(height)
    if not (image_width > 0 and image_height > 0):
        raise ValueError("image dimensions must be positive")
    config = profile or resolve_profile()
    fx = config.get("fx_px")
    fy = config.get("fy_px")
    if not (isinstance(fx, (int, float)) and fx > 0):
        focal_35mm = float(config.get("focal_35mm") or 24.0)
        if focal_35mm <= 0:
            raise ValueError("focal_35mm must be positive")
        fx = image_width * focal_35mm / 36.0
    if not (isinstance(fy, (int, float)) and fy > 0):
        fy = fx
    return {
        "fx_px": float(fx),
        "fy_px": float(fy),
        "cx_px": image_width / 2.0,
        "cy_px": image_height / 2.0,
    }


def calibration_status(profile: dict | None = None) -> dict:
    """返回标定状态，供接口与界面区分“已标定”和“仅参考”。"""
    config = profile or resolve_profile()
    calibrated = config.get("calibrated") is True
    residual = config.get("residual_m")
    try:
        residual_value = float(residual) if residual is not None else None
    except (TypeError, ValueError):
        residual_value = None
    return {
        "calibrated": calibrated,
        "status": "calibrated" if calibrated else "nominal",
        "residual_m": residual_value,
        "ground_offset_m": float(config.get("ground_offset_m") or 0.0),
        "profile": config.get("name") or DEFAULT_PROFILE,
    }
