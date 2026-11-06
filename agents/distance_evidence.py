"""Validate a model-proposed radius against numeric evidence, not intent keywords."""
from __future__ import annotations

import re

_DIGITS = dict(zip("零〇一二两三四五六七八九", (0, 0, 1, 2, 2, 3, 4, 5, 6, 7, 8, 9)))
_NUMBER = r"(?:\d+(?:\.\d+)?|[零〇一二两三四五六七八九十百千点半]+)"
_DISTANCE = re.compile(rf"(?<![\d.])({_NUMBER})\s*(公里|千米|km|米|m)(半)?(?![a-z])", re.I)
_MULTIPLIER = re.compile(rf"(?<![\d.])({_NUMBER})\s*倍")


def _number(value: str) -> float:
    if value == "半":
        return 0.5
    try:
        return float(value)
    except ValueError:
        pass
    if "点" in value:
        integer, fraction = value.split("点", 1)
        if not fraction or any(char not in _DIGITS for char in fraction):
            raise ValueError("invalid decimal")
        return _number(integer or "零") + float("0." + "".join(str(_DIGITS[char]) for char in fraction))
    if all(char in _DIGITS for char in value):
        return float("".join(str(_DIGITS[char]) for char in value))
    total = 0
    digit = None
    last_unit = 0
    zero_after_unit = False
    for char in value:
        if char in _DIGITS:
            digit = _DIGITS[char]
            if digit == 0 and last_unit:
                zero_after_unit = True
        elif char in "十百千":
            unit = {"十": 10, "百": 100, "千": 1000}[char]
            if last_unit and unit >= last_unit:
                raise ValueError("invalid number order")
            total += (1 if digit is None else digit) * unit
            digit, last_unit, zero_after_unit = None, unit, False
        else:
            raise ValueError("invalid number")
    if digit is not None:
        total += digit * (last_unit // 10 if last_unit >= 100 and not zero_after_unit else 1)
    return float(total)


def radius_supported_by_message(message: str, proposed_m: int, previous_m: int) -> bool:
    """Accept explicit distance, additive change or multiplier from the raw user text.

    The model still decides which condition the number describes; this function
    checks that a changed numeric radius has a possible user-supplied source.
    """
    for match in _DISTANCE.finditer(message or ""):
        try:
            value = _number(match.group(1)) + (0.5 if match.group(3) else 0)
        except ValueError:
            continue
        meters = value * (1000 if match.group(2).lower() in {"公里", "千米", "km"} else 1)
        if any(abs(proposed_m - candidate) < 0.001 for candidate in (meters, previous_m + meters, previous_m - meters)):
            return True
    for match in _MULTIPLIER.finditer(message or ""):
        try:
            if abs(proposed_m - previous_m * _number(match.group(1))) < 0.001:
                return True
        except ValueError:
            continue
    return False
