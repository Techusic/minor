import sys
from pathlib import Path

from photoholic_ai import paths

REPO = Path(__file__).resolve().parents[2]


def test_from_source_points_into_the_repo():
    assert paths.resource_dir() == REPO
    assert paths.engine_path() == REPO / "image-reader" / "target" / "release" / paths.ENGINE_EXE
    assert paths.model_path() == REPO / "ai" / "models" / "aesthetic_v1.onnx"
    assert (paths.static_dir() / "index.html").is_file()
    assert paths.icon_path().is_file()


def test_packaged_layout(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path), raising=False)
    assert paths.resource_dir() == tmp_path
    assert paths.engine_path() == tmp_path / "engine" / paths.ENGINE_EXE
    assert paths.model_path() == tmp_path / "models" / "aesthetic_v1.onnx"
    assert paths.static_dir() == tmp_path / "app" / "static"


def test_user_data_under_localappdata(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    assert paths.user_data_dir() == tmp_path / "Photoholic"
    assert paths.runs_dir() == tmp_path / "Photoholic" / "runs"
    assert paths.logs_dir() == tmp_path / "Photoholic" / "logs"


def test_run_dir_for_is_stable_distinct_and_safe(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    a = tmp_path / "Shoot 2026" / "Day 1"
    b = tmp_path / "Other" / "Day 1"
    for d in (a, b):
        d.mkdir(parents=True)
    ra, rb = paths.run_dir_for(a), paths.run_dir_for(b)
    assert ra == paths.run_dir_for(a)  # same folder -> same results folder
    assert ra != rb  # same name, different folder -> different results folder
    assert ra.parent == paths.runs_dir()
    assert ra.name.startswith("Day_1-") and len(ra.name) == len("Day_1-") + 8
    weird = tmp_path / "Café & #fotos (2026)"
    weird.mkdir()
    assert set(paths.run_dir_for(weird).name) <= set(
        "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789_-"
    )
