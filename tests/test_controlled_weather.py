import json
from pathlib import Path

import pytest

from experiments.controlled_weather import load_development_weather


def test_development_weather_has_explicit_synthetic_provenance():
    path = Path(__file__).resolve().parents[1] / "experiments/fixtures/development_weather_clear_v1.json"
    snapshot, provenance = load_development_weather(path)
    assert snapshot["live"]["weather"] == "晴"
    assert snapshot["impact"]["slippery"] is False
    assert provenance["source"] == "synthetic_development_fixture"
    assert len(provenance["sha256"]) == 64


def test_development_weather_rejects_unlabelled_input(tmp_path):
    path = tmp_path / "unlabelled.json"
    path.write_text(json.dumps({"live": {"weather": "晴", "temperature": 22}}),
                    encoding="utf-8")
    with pytest.raises(ValueError, match="labelled synthetic"):
        load_development_weather(path)
