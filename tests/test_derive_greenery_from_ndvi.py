import importlib.util
import io
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "network" / "derive_greenery_from_ndvi.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("derive_greenery_from_ndvi", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    original_stdout = sys.stdout
    sys.stdout = io.TextIOWrapper(io.BytesIO(), encoding="utf-8")
    try:
        spec.loader.exec_module(module)
    finally:
        sys.stdout = original_stdout
    return module


def test_cli_accepts_explicit_input_and_output_paths():
    module = _load_module()

    args = module.build_parser().parse_args([
        "--graph", "custom.graphml",
        "--ndvi", "custom-ndvi.tif",
        "--dem", "custom-dem.tif",
        "--stats-output", "stats.json",
        "--attributes-output", "attributes.json",
        "--dry-run",
    ])

    assert args.graph == Path("custom.graphml")
    assert args.ndvi == Path("custom-ndvi.tif")
    assert args.dem == Path("custom-dem.tif")
    assert args.stats_output == Path("stats.json")
    assert args.attributes_output == Path("attributes.json")
    assert args.dry_run is True


def test_cli_keeps_existing_path_defaults():
    module = _load_module()

    args = module.build_parser().parse_args([])

    assert args.graph == module.GRAPH
    assert args.ndvi == module.NDVI
    assert args.dem == module.DEM
    assert args.stats_output == module.OUT_STATS
    assert args.attributes_output == module.OUT_ATTR
