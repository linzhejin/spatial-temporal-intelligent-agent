import zipfile

from scripts.analysis.select_visdrone_v2 import build_selection


def test_scene_prefix_selection_is_reproducible_and_labeled(tmp_path):
    archive = tmp_path / "sample.zip"
    with zipfile.ZipFile(archive, "w") as sink:
        for scene in ("00001", "00002", "00003"):
            for index in range(3):
                stem = f"{scene}_{index:04d}"
                sink.writestr(f"val/images/{stem}.jpg", b"jpeg-placeholder")
                sink.writestr(f"val/annotations/{stem}.txt", b"annotation-placeholder")
    first = build_selection(archive, count=6, seed=17)
    second = build_selection(archive, count=6, seed=17)
    assert first == second
    assert first["scene_prefix_count"] == 3
    assert len(first["images"]) == len(set(first["images"])) == 6
