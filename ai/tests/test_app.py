"""Pieces the desktop app relies on: run_folder progress, server in the background, cache safety,
launcher helpers."""

import json
import sys
import urllib.request
from io import BytesIO
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

import run_pipeline
from photoholic_ai import bridge
from photoholic_ai.config import Config
from tests.test_e2e import handmade_filtered, make_shoot

APP_DIR = Path(__file__).resolve().parents[2] / "app"
sys.path.insert(0, str(APP_DIR))
import launcher  # noqa: E402
import server  # noqa: E402


def collect():
    events = []
    return events, lambda stage, done, total, msg: events.append((stage, done, total, msg))


def test_run_folder_reports_progress_without_printing(tmp_path, dummy_model, capsys):
    folder = tmp_path / "Shoot"
    truth = make_shoot(folder)
    handmade_filtered(folder, truth)
    events, progress = collect()

    results, info = run_pipeline.run_folder(
        folder, folder / ".photoholic", Config(), model=dummy_model, skip_engine=True,
        progress=progress,
    )  # fmt: skip

    stages = [e[0] for e in events]
    assert stages[0] == "engine" and stages[-1] == "done"
    assert [s for s in dict.fromkeys(stages)] == [*run_pipeline.STAGES, "done"]
    score = [e for e in events if e[0] == "score"]
    assert score[-1][1] == score[-1][2] == 6  # all six photos scored
    assert results["stats"]["clusters"] == 3
    assert info.csv_path.is_file() and info.total_s > 0
    assert set(info.timings) == set(run_pipeline.STAGES)
    out = capsys.readouterr()
    assert out.out == "" and "scoring" not in out.err  # no console output with a progress callback


@pytest.mark.skipif(not run_pipeline.DEFAULT_ENGINE.is_file(), reason="Rust engine not built")
def test_run_folder_reads_engine_output(tmp_path, dummy_model):
    folder = tmp_path / "Shoot"
    make_shoot(folder)
    events, progress = collect()
    started = []

    run_pipeline.run_folder(folder, tmp_path / "run", Config(), model=dummy_model,
                            progress=progress, on_engine_start=started.append)  # fmt: skip

    messages = [e[3] for e in events if e[0] == "engine"]
    assert "Found 7 photos, checking focus" in messages
    assert any(m.startswith("Kept 6 images, rejected 1") for m in messages)
    assert started and started[0].returncode == 0


def test_engine_failure_is_a_pipeline_error(tmp_path):
    with pytest.raises(run_pipeline.PipelineError, match="engine not found"):
        run_pipeline.run_engine(tmp_path / "nope.exe", tmp_path, tmp_path / "f.json", 10.0)


def test_start_in_background_on_a_free_port(tmp_path, dummy_model):
    folder = tmp_path / "Shoot"
    handmade_filtered(
        folder, make_shoot(folder)
    )  # writes <folder>/.photoholic/filtered_images.json
    run_dir = folder / ".photoholic"
    run_pipeline.run_folder(folder, run_dir, Config(), model=dummy_model, skip_engine=True,
                            progress=lambda *a: None)  # fmt: skip

    srv, url = server.start_in_background(run_dir)
    try:
        assert url.startswith("http://127.0.0.1:") and not url.endswith(":0/")
        with urllib.request.urlopen(url + "api/run", timeout=10) as r:
            data = json.loads(r.read())
        assert len(data["groups"]) == 3 and data["token"]
    finally:
        srv.shutdown()
        srv.server_close()


def test_reused_run_folder_serves_the_new_photos(tmp_path, dummy_model):
    """Regression: thumbnails were cached by id, so a re-run showed the previous photos."""
    run_dir = tmp_path / "run"

    def shoot(name, colour):
        folder = tmp_path / name
        folder.mkdir()
        for i in range(2):
            Image.new("RGB", (64, 48), colour).save(folder / f"IMG_{i}.jpg")
        images = [{"id": f"img_{i + 1:06d}", "path": str(folder / f"IMG_{i}.jpg"),
                   "sharpness_score": 50.0} for i in range(2)]  # fmt: skip
        filtered = {"schema_version": 1, "rejected": [], "images": images}
        bridge.write_json(run_dir / bridge.FILTERED_NAME, filtered)
        run_pipeline.run_folder(folder, run_dir, Config(), model=dummy_model, skip_engine=True,
                                progress=lambda *a: None)  # fmt: skip
        run = server.Run(run_dir)
        thumb, _ = run.image_bytes("img_000001", "thumb")
        return np.asarray(Image.open(BytesIO(thumb)).convert("RGB")).mean(axis=(0, 1)), run.token

    red, token1 = shoot("red", (220, 20, 20))
    blue, token2 = shoot("blue", (20, 20, 220))  # same ids, same run folder
    assert red[0] > 150 and blue[2] > 150  # second run shows the blue photo, not the cached red
    assert token1 != token2  # new token, so the browser does not reuse its cache either


def test_recent_runs_and_summary(tmp_path):
    for name, input_, clusters, ago in (("a", 10, 4, 100), ("b", 20, 7, 0), ("broken", 0, 0, 0)):
        d = tmp_path / name
        d.mkdir()
        if name != "broken":
            stats = {"input": input_, "clusters": clusters}
            (d / bridge.RESULTS_NAME).write_text(json.dumps({"stats": stats}), encoding="utf-8")
            launcher.write_source(d, Path(f"D:/Photos/{name}"))
            import os
            import time

            t = time.time() - ago
            os.utime(d / bridge.RESULTS_NAME, (t, t))
    runs = launcher.recent_runs(tmp_path)
    assert [r.run_dir.name for r in runs] == ["b", "a"]  # newest first, incomplete skipped
    assert runs[0].picks == 7 and runs[0].photos == 20 and runs[0].folder.endswith("b")
    assert launcher.recent_runs(tmp_path / "missing") == []

    results = {"stats": {"input": 1221, "rejected_sharpness": 106, "clusters": 459,
                         "reduction_pct": 58.8}, "model": {"trained": False}}  # fmt: skip
    lines = launcher.summary_lines(results, 82.4)
    assert lines[0] == "1,221 photos found" and "459 best picks kept (59% fewer" in lines[2]
    assert "untrained" in lines[-1]
