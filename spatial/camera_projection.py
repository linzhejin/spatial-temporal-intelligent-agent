"""Calibrated pinhole rays onto a locally flat ground plane.

The pose and intrinsics are explicit, validated inputs. Raw aircraft SRT
metadata is not a calibration and must not be passed as this structure.
Axes are east/north/up; yaw is clockwise from north, pitch is -90 at nadir,
and positive roll rotates image-right toward image-up around the optical axis.
"""
import math

from spatial.coord_transform import wgs84_to_gcj02


def _number(value, label):
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{label} must be a finite number") from error
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite number")
    return result


def camera_ground_homography(pose, image_w_px, image_h_px):
    """Return a pixel-to-ground homography and WGS/GCJ origin, using no numpy."""
    if not isinstance(pose, dict):
        raise ValueError("pose must be a mapping")
    width = _number(image_w_px, "image width")
    height = _number(image_h_px, "image height")
    if width <= 0 or height <= 0:
        raise ValueError("image dimensions must be positive")
    lng = _number(pose.get("lng"), "longitude")
    lat = _number(pose.get("lat"), "latitude")
    if not -180 <= lng <= 180 or not -90 <= lat <= 90:
        raise ValueError("pose position is out of range")
    altitude = _number(pose.get("height_agl_m"), "ground-relative altitude")
    if altitude <= 0:
        raise ValueError("ground-relative altitude must be positive")
    yaw = _number(pose.get("gb_yaw_deg"), "gimbal yaw")
    pitch = _number(pose.get("gb_pitch_deg"), "gimbal pitch")
    roll = _number(pose.get("gb_roll_deg"), "gimbal roll")
    if not -360 <= yaw <= 360 or not -90 <= pitch < 0 or not -180 <= roll <= 180:
        raise ValueError("camera orientation out of range")
    intrinsic = pose.get("intrinsics")
    if not isinstance(intrinsic, dict):
        raise ValueError("calibrated camera intrinsics are required")
    fx = _number(intrinsic.get("fx_px"), "focal length fx")
    fy = _number(intrinsic.get("fy_px"), "focal length fy")
    cx = _number(intrinsic.get("cx_px"), "principal point cx")
    cy = _number(intrinsic.get("cy_px"), "principal point cy")
    if fx <= 0 or fy <= 0 or not 0 <= cx <= width or not 0 <= cy <= height:
        raise ValueError("invalid camera intrinsics")

    y, t, r = map(math.radians, (yaw, 90.0 + pitch, roll))
    sy, cyaw = math.sin(y), math.cos(y)
    st, ct, sr, cr = math.sin(t), math.cos(t), math.sin(r), math.cos(r)
    right0 = (cyaw, -sy, 0.0)
    up0 = (sy * ct, cyaw * ct, st)
    forward = (sy * st, cyaw * st, -ct)
    right = tuple(cr * right0[i] + sr * up0[i] for i in range(3))
    up = tuple(-sr * right0[i] + cr * up0[i] for i in range(3))

    # ray = right * (x-cx)/fx - up * (y-cy)/fy + forward.
    coeff = [
        (right[i] / fx, -up[i] / fy,
         forward[i] - right[i] * cx / fx + up[i] * cy / fy)
        for i in range(3)
    ]
    matrix = [
        [altitude * value for value in coeff[0]],
        [altitude * value for value in coeff[1]],
        [-value for value in coeff[2]],
    ]
    for x, yy in ((0, 0), (width, 0), (width, height), (0, height)):
        depth = matrix[2][0] * x + matrix[2][1] * yy + matrix[2][2]
        if not math.isfinite(depth) or depth <= 1e-9:
            raise ValueError("camera field of view crosses the horizon")
    gcj_lng, gcj_lat = wgs84_to_gcj02(lng, lat)
    return {"homography": matrix, "origin": {
        "lng_wgs": lng, "lat_wgs": lat, "lng_gcj": gcj_lng, "lat_gcj": gcj_lat,
    }}


def project_pixel(matrix, x, y):
    depth = matrix[2][0] * x + matrix[2][1] * y + matrix[2][2]
    if not math.isfinite(depth) or depth <= 1e-9:
        raise ValueError("pixel ray does not meet the ground")
    return tuple((row[0] * x + row[1] * y + row[2]) / depth for row in matrix[:2])
