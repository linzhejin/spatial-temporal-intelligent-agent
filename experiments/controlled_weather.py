"""Load a labelled synthetic weather input for isolated development replay."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from spatial.weather import classify_weather


def load_development_weather(path: Path) -> tuple[dict, dict]:
    """Return a tool snapshot and provenance; never imply live observation."""
    raw = path.read_bytes()
    fixture = json.loads(raw)
    if not isinstance(fixture, dict) or fixture.get("source") != "synthetic_development_fixture":
        raise ValueError("only labelled synthetic development weather is accepted")
    live = fixture.get("live")
    if (not isinstance(live, dict) or not isinstance(live.get("weather"), str)
            or not live["weather"] or type(live.get("temperature")) not in (int, float)
            or not isinstance(live.get("reporttime"), str)):
        raise ValueError("invalid development weather observation")
    provenance = {
        "fixture_id": fixture.get("fixture_id"),
        "source": fixture["source"],
        "observed_at": fixture.get("observed_at"),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }
    return {"live": live, "impact": classify_weather(live)}, provenance
