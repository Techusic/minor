"""Photoholic end to end: Rust engine -> aesthetic scoring -> grouping -> selection -> report.

    python ai/run_pipeline.py "D:\\Photos\\Trip"
    python ai/run_pipeline.py "D:\\Photos\\Trip" --action copy      # copy best picks
    python ai/run_pipeline.py "D:\\Photos\\Trip" --skip-engine      # reuse filtered_images.json

All outputs go to ``<input_dir>/.photoholic/`` (``filtered_images.json``,
``scored_images.json``, ``embeddings.npy``, ``results.json``, ``report.csv``, ``charts/``).
Photos are only read, unless ``--action copy|move`` is given (never delete, never overwrite).
"""

from __future__ import annotations

import argparse
import logging
import os
import re
import subprocess
import sys
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from tqdm import tqdm

from photoholic_ai import bridge, grouping, paths, report, selection
from photoholic_ai.config import ACTIONS, Config, load_config
from photoholic_ai.scorer import Scorer, score_filtered

log = logging.getLogger("photoholic")

DEFAULT_ENGINE = paths.engine_path()
DEFAULT_MODEL = paths.model_path()
RUN_DIR = ".photoholic"
STAGES = ("engine", "score", "group", "select", "report")
# stage, done, total (None = unknown), message
Progress = Callable[[str, int, "int | None", str], None]
_FOUND = re.compile(r"Found (\d+) image files")


class PipelineError(Exception):
    """A stage failed in a way the user can fix (missing engine/model, bad input)."""


@dataclass
class RunInfo:
    """What a finished run produced, besides ``results.json``."""

    run_dir: Path
    timings: dict[str, float] = field(default_factory=dict)
    total_s: float = 0.0
    csv_path: Path | None = None
    charts: list[Path] = field(default_factory=list)
    actions: list[dict] = field(default_factory=list)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Cull a photo folder: keep the best shot of every burst / near-duplicate.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("input_dir", type=Path, help="folder of JPEG/ARW photos (searched recursively)")
    p.add_argument("--engine", type=Path, default=DEFAULT_ENGINE, help="Rust image-reader binary")
    p.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="aesthetic ONNX model")
    p.add_argument("--run-dir", type=Path, help=f"outputs (default <input_dir>/{RUN_DIR})")
    p.add_argument("--config", type=Path, help="config.toml with overrides")
    p.add_argument("--skip-engine", action="store_true", help="reuse filtered_images.json")
    p.add_argument("--min-sharpness", type=float)
    p.add_argument("--burst-gap", type=float, dest="burst_gap_s", help="seconds")
    p.add_argument("--sim", type=float, dest="sim_threshold_timed", help="cosine, shots in a burst")
    p.add_argument("--sim-untimed", type=float, dest="sim_threshold_untimed")
    p.add_argument(
        "--session-gap",
        type=float,
        dest="session_gap_h",
        help="hours; near-duplicates further apart than this are never grouped (default 3)",
    )
    p.add_argument("--w-sharp", type=float, dest="w_sharpness")
    p.add_argument("--w-aes", type=float, dest="w_aesthetic")
    p.add_argument("--action", choices=ACTIONS, help="none (default), copy best, move rejects")
    p.add_argument("--grouping", choices=grouping.METHODS, default="union_find")
    p.add_argument("--batch-size", type=int)
    p.add_argument("--charts", action="store_true", help="write EDA charts (needs matplotlib)")
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args(argv)

    # one weight implies the other
    if args.w_sharpness is not None and args.w_aesthetic is None:
        args.w_aesthetic = round(1.0 - args.w_sharpness, 6)
    elif args.w_aesthetic is not None and args.w_sharpness is None:
        args.w_sharpness = round(1.0 - args.w_aesthetic, 6)
    return args


@contextmanager
def stage(name: str, timings: dict[str, float]):
    log.info("== %s", name)
    t0 = time.perf_counter()
    yield
    timings[name] = time.perf_counter() - t0


def run_engine(
    engine: Path,
    input_dir: Path,
    out: Path,
    min_sharpness: float,
    on_line: Callable[[str], None] | None = None,
    on_start: Callable[[subprocess.Popen], None] | None = None,
) -> None:
    """Run the Rust engine without a console window, passing each output line to ``on_line``."""
    if not engine.is_file():
        raise PipelineError(
            f"Rust engine not found: {engine}\n"
            "Build it with `cargo build --release` in image-reader/, pass --engine, "
            "or use --skip-engine to reuse an existing filtered_images.json."
        )
    cmd = [str(engine), str(input_dir), "--out", str(out), "--min-sharpness", str(min_sharpness)]
    log.debug("running %s", cmd)
    no_window = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        text=True,
        encoding="utf-8",
        errors="replace",
        creationflags=no_window,
    )
    if on_start is not None:
        on_start(proc)
    assert proc.stdout is not None
    for line in proc.stdout:
        if on_line is not None:
            on_line(line.rstrip("\r\n"))
    code = proc.wait()
    if code != 0:
        raise PipelineError(f"Rust engine failed with exit code {code}")


def run_folder(
    input_dir: Path,
    run_dir: Path,
    cfg: Config,
    *,
    engine: Path = DEFAULT_ENGINE,
    model: Path = DEFAULT_MODEL,
    skip_engine: bool = False,
    grouping_method: str = "union_find",
    charts: bool = False,
    progress: Progress | None = None,
    on_engine_start: Callable[[subprocess.Popen], None] | None = None,
) -> tuple[bridge.Record, RunInfo]:
    """Run every stage on ``input_dir``, writing outputs to ``run_dir``.

    With ``progress`` the caller (e.g. the desktop launcher) gets stage updates and nothing is
    printed; without it, engine output and a scoring bar go to the console as before.
    """
    input_dir = Path(input_dir).resolve()
    if not input_dir.is_dir():
        raise PipelineError(f"input folder not found: {input_dir}")
    run_dir = Path(run_dir).resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    filtered_path = run_dir / bridge.FILTERED_NAME
    info = RunInfo(run_dir)
    t_start = time.perf_counter()

    def emit(name: str, done: int, total: int | None, message: str = "") -> None:
        if progress is not None:
            progress(name, done, total, message)

    def engine_line(line: str) -> None:
        if progress is None:
            print(line)
            return
        found = _FOUND.search(line)
        if found:
            emit("engine", 0, None, f"Found {int(found.group(1)):,} photos, checking focus")
        elif line.startswith("Kept "):
            emit("engine", 0, None, line.split(". ")[0])

    with stage("engine", info.timings):
        emit("engine", 0, None, "Finding photos")
        if skip_engine:
            log.info("skipping engine, reusing %s", filtered_path)
        else:
            run_engine(engine, input_dir, filtered_path, cfg.min_sharpness, engine_line,
                       on_engine_start)  # fmt: skip
        filtered = bridge.load_filtered(filtered_path)

    with stage("score", info.timings):
        n = len(filtered["images"])
        emit("score", 0, n, f"Scoring {n:,} photos")
        scorer = Scorer(model, batch_size=cfg.batch_size)
        # tqdm needs a console; a windowed app has none (sys.stderr is None)
        bar = tqdm(total=n, desc="scoring", unit="img") if progress is None and sys.stderr else None
        done = 0

        def scored_some(k: int) -> None:
            nonlocal done
            done += k
            if bar is not None:
                bar.update(k)
            emit("score", done, n, f"Scoring {done:,} / {n:,}")

        try:
            scored, embeddings = score_filtered(filtered, scorer, run_dir, progress=scored_some)
        finally:
            if bar is not None:
                bar.close()

    with stage("group", info.timings):
        emit("group", 0, None, "Grouping bursts and near-duplicates")
        by_id = {r["id"]: r for r in filtered["images"]}
        ok = [r for r in scored["images"] if r["status"] == "ok"]
        groups = grouping.group_images(
            [r["id"] for r in ok],
            embeddings[[r["embedding_index"] for r in ok]] if ok else embeddings,
            [grouping.parse_capture_time(by_id[r["id"]]) for r in ok],
            cfg,
            method=grouping_method,
        )

    with stage("select", info.timings):
        emit("select", 0, None, "Picking the best photo of each group")
        results = selection.select(filtered, scored, groups, cfg)
        selection.write_results(run_dir, results)
        info.actions = selection.apply_action(results, input_dir, cfg.action, run_dir)

    with stage("report", info.timings):
        emit("report", 0, None, "Writing the report")
        table = report.build_table(filtered, scored, results)
        info.csv_path = report.write_csv(table, run_dir)
        info.charts = report.write_charts(table, run_dir) if charts else []

    info.total_s = time.perf_counter() - t_start
    emit("done", 1, 1, "Done")
    return results, info


def run(argv: list[str] | None = None) -> bridge.Record:
    args = parse_args(argv)
    input_dir = args.input_dir.resolve()
    run_dir = (args.run_dir or input_dir / RUN_DIR).resolve()
    overrides = {
        k: getattr(args, k)
        for k in ("min_sharpness", "burst_gap_s", "sim_threshold_timed", "sim_threshold_untimed",
                  "session_gap_h", "w_sharpness", "w_aesthetic", "action", "batch_size")
    }  # fmt: skip
    cfg = load_config(args.config, **overrides)
    results, info = run_folder(
        input_dir,
        run_dir,
        cfg,
        engine=args.engine,
        model=args.model,
        skip_engine=args.skip_engine,
        grouping_method=args.grouping,
        charts=args.charts,
    )
    print_summary(results, info.timings, info.total_s, info.run_dir, info.csv_path, info.charts,
                  cfg.action, info.actions)  # fmt: skip
    return results


def print_summary(results, timings, total, run_dir, csv_path, charts, action, actions) -> None:
    s = results["stats"]
    model = results.get("model") or {}
    lines = [
        "",
        "Photoholic summary",
        f"  input images        {s['input']}",
        f"  rejected (blur)     {s['rejected_sharpness']}",
        f"  failed to decode    {s['rejected_decode'] + s['decode_errors']}",
        f"  scored              {s['scored']}",
        f"  groups (best picks) {s['clusters']}",
        f"  reduction           {s['reduction_pct']}%",
        "  time                "
        + ", ".join(f"{k} {v:.1f}s" for k, v in timings.items())
        + f" | total {total:.1f}s",
        f"  results             {run_dir / bridge.RESULTS_NAME}",
        f"  report              {csv_path}",
    ]
    if charts:
        lines.append(f"  charts              {charts[0].parent}")
    if action != "none":
        lines.append(f"  action              {action}: {len(actions)} files")
    if model.get("trained") is False:
        lines.append("  NOTE: untrained model - aesthetic scores are placeholders")
    print("\n".join(lines))


def main(argv: list[str] | None = None) -> int:
    verbose = "-v" in (argv or sys.argv) or "--verbose" in (argv or sys.argv)
    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    try:
        run(argv)
    except (PipelineError, bridge.BridgeError, FileNotFoundError, ValueError) as e:
        log.error("%s", e)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
