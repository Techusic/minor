"""Photoholic desktop launcher: pick a folder, run the pipeline, open the review GUI.

This is the entry point of ``Photoholic.exe``; from source:

    ai\\.venv\\Scripts\\python.exe app\\launcher.py

Results go to ``%LOCALAPPDATA%\\Photoholic\\runs\\<folder>-<hash>``; the photo folder is only read.
"""

from __future__ import annotations

import json
import logging
import os
import queue
import sys
import threading
import time
import traceback
import webbrowser
from dataclasses import dataclass
from pathlib import Path

if not getattr(sys, "frozen", False):  # from source: make ai/ and app/ importable
    _here = Path(__file__).resolve().parent
    sys.path.insert(0, str(_here.parent / "ai"))
    sys.path.insert(0, str(_here))

import tkinter as tk  # noqa: E402
from tkinter import filedialog, messagebox, ttk  # noqa: E402

import run_pipeline  # noqa: E402
import server  # noqa: E402
from photoholic_ai import bridge, paths  # noqa: E402
from photoholic_ai.config import load_config  # noqa: E402

log = logging.getLogger("photoholic.app")

SOURCE_NAME = "source.json"  # in each run dir: which photo folder it belongs to
STAGE_LABELS = {
    "engine": "Checking focus",
    "score": "Scoring photos",
    "group": "Grouping",
    "select": "Selecting best photos",
    "report": "Writing report",
    "done": "Done",
}


# ---------------------------------------------------------------- helpers (no Tk, testable)
@dataclass
class RecentRun:
    run_dir: Path
    folder: str
    finished: float
    picks: int
    photos: int


def recent_runs(root: Path | None = None, limit: int = 12) -> list[RecentRun]:
    """Finished runs under ``root`` (default: the app's runs folder), newest first."""
    root = root or paths.runs_dir()
    found = []
    if not root.is_dir():
        return found
    for d in root.iterdir():
        results, source = d / bridge.RESULTS_NAME, d / SOURCE_NAME
        if not (results.is_file() and source.is_file()):
            continue
        try:
            stats = json.loads(results.read_text(encoding="utf-8"))["stats"]
            meta = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, ValueError, KeyError):
            continue
        found.append(RecentRun(d, meta.get("folder", "?"), results.stat().st_mtime,
                               stats.get("clusters", 0), stats.get("input", 0)))  # fmt: skip
    found.sort(key=lambda r: r.finished, reverse=True)
    return found[:limit]


def write_source(run_dir: Path, folder: Path) -> None:
    bridge.write_json(run_dir / SOURCE_NAME, {"folder": str(folder), "finished": time.time()})


def summary_lines(results: bridge.Record, seconds: float) -> list[str]:
    s = results["stats"]
    lines = [
        f"{s['input']:,} photos found",
        f"{s['rejected_sharpness']:,} rejected as blurry",
        f"{s['clusters']:,} best picks kept ({s['reduction_pct']:.0f}% fewer to review)",
        f"took {seconds:.0f} s",
    ]
    if (results.get("model") or {}).get("trained") is False:
        lines.append("Aesthetic model is an untrained placeholder: picks follow sharpness.")
    return lines


def setup_logging() -> Path:
    """Log to a file (the packaged app has no console); also guard against a missing console."""
    paths.logs_dir().mkdir(parents=True, exist_ok=True)
    log_file = paths.logs_dir() / "photoholic.log"
    if sys.stdout is None or sys.stderr is None:  # windowed exe: print() would crash
        sink = open(os.devnull, "w", encoding="utf-8")  # noqa: SIM115
        sys.stdout = sys.stdout or sink
        sys.stderr = sys.stderr or sink
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[logging.FileHandler(log_file, encoding="utf-8")],
    )
    return log_file


# ---------------------------------------------------------------- window
class App:
    def __init__(self, root: tk.Tk, log_file: Path) -> None:
        self.root = root
        self.log_file = log_file
        self.events: queue.Queue = queue.Queue()
        self.worker: threading.Thread | None = None
        self.engine_proc = None
        self.server = None
        self.url: str | None = None

        root.title("Photoholic")
        root.minsize(560, 470)
        try:
            root.iconbitmap(default=str(paths.icon_path()))
        except tk.TclError:
            pass
        pad = {"padx": 16}

        head = ttk.Frame(root)
        head.pack(fill="x", pady=(16, 6), **pad)
        ttk.Label(head, text="Photoholic", font=("Segoe UI", 18, "bold")).pack(anchor="w")
        ttk.Label(head, wraplength=520, foreground="#555555",
                  text="Pick a photo folder. Photoholic removes blurry shots, groups bursts and "
                       "near-duplicates, and keeps the best photo of each group. "
                       "Your photos are only read, never moved or deleted.",
                  ).pack(anchor="w")  # fmt: skip

        self.choose_btn = ttk.Button(root, text="Choose photo folder…", command=self.choose)
        self.choose_btn.pack(fill="x", pady=(8, 6), ipady=6, **pad)

        status = ttk.Frame(root)
        status.pack(fill="x", **pad)
        self.stage_var = tk.StringVar(value="Ready")
        self.detail_var = tk.StringVar(value="")
        ttk.Label(status, textvariable=self.stage_var, font=("Segoe UI", 10, "bold")).pack(
            anchor="w"
        )
        self.bar = ttk.Progressbar(status, mode="determinate", maximum=100)
        self.bar.pack(fill="x", pady=4)
        ttk.Label(status, textvariable=self.detail_var, foreground="#555555", wraplength=520,
                  justify="left").pack(anchor="w")  # fmt: skip

        actions = ttk.Frame(root)
        actions.pack(fill="x", pady=(6, 4), **pad)
        self.open_btn = ttk.Button(actions, text="Open review", command=self.open_review,
                                   state="disabled")  # fmt: skip
        self.open_btn.pack(side="left")
        self.folder_btn = ttk.Button(actions, text="Open results folder", state="disabled",
                                     command=self.open_results_folder)  # fmt: skip
        self.folder_btn.pack(side="left", padx=8)
        self.current_run: Path | None = None

        ttk.Label(root, text="Recent results", font=("Segoe UI", 10, "bold")).pack(
            anchor="w", pady=(10, 2), **pad
        )
        recent = ttk.Frame(root)
        recent.pack(fill="both", expand=True, pady=(0, 6), **pad)
        self.recent_list = tk.Listbox(recent, height=6, activestyle="none")
        self.recent_list.pack(side="left", fill="both", expand=True)
        self.recent_list.bind("<Double-Button-1>", lambda _e: self.open_recent())
        scroll = ttk.Scrollbar(recent, orient="vertical", command=self.recent_list.yview)
        scroll.pack(side="right", fill="y")
        self.recent_list.config(yscrollcommand=scroll.set)
        foot = ttk.Frame(root)
        foot.pack(fill="x", pady=(0, 14), **pad)
        self.recent_btn = ttk.Button(foot, text="Open selected", command=self.open_recent)
        self.recent_btn.pack(side="left")
        ttk.Label(foot, text=f"Log: {log_file}", foreground="#888888").pack(side="right")
        self.recent: list[RecentRun] = []
        self.refresh_recent()

        root.protocol("WM_DELETE_WINDOW", self.close)
        root.after(100, self.poll)

    # ---------------- running
    def choose(self) -> None:
        folder = filedialog.askdirectory(title="Choose a photo folder", mustexist=True)
        if not folder:
            return
        self.start(Path(folder))

    def start(self, folder: Path) -> None:
        self.set_busy(True)
        self.stop_server()
        self.current_run = paths.run_dir_for(folder)
        self.stage_var.set("Starting…")
        self.detail_var.set(str(folder))
        self.bar.config(mode="indeterminate")
        self.bar.start(12)
        self.worker = threading.Thread(
            target=self.work, args=(folder, self.current_run), name="pipeline", daemon=True
        )
        self.worker.start()

    def work(self, folder: Path, run_dir: Path) -> None:
        """Worker thread: never touch Tk here, only post events."""
        log.info("run %s -> %s", folder, run_dir)
        try:
            results, info = run_pipeline.run_folder(
                folder,
                run_dir,
                load_config(),
                progress=lambda *e: self.events.put(("progress", *e)),
                on_engine_start=self._remember_engine,
            )
            if results["stats"]["input"] > 0:  # empty folders are not listed in Recent results
                write_source(run_dir, folder)
            self.events.put(("done", folder, run_dir, results, info.total_s))
        except Exception as e:  # report everything to the user, details to the log
            log.error("run failed: %s\n%s", e, traceback.format_exc())
            self.events.put(("error", str(e)))

    def _remember_engine(self, proc) -> None:
        self.engine_proc = proc

    def poll(self) -> None:
        try:
            while True:
                self.handle(self.events.get_nowait())
        except queue.Empty:
            pass
        self.root.after(100, self.poll)

    def handle(self, event: tuple) -> None:
        kind = event[0]
        if kind == "progress":
            _, stage, done, total, message = event
            self.stage_var.set(STAGE_LABELS.get(stage, stage))
            self.detail_var.set(message)
            if total:
                if str(self.bar.cget("mode")) != "determinate":
                    self.bar.stop()
                    self.bar.config(mode="determinate")
                self.bar["value"] = 100 * done / total
            elif str(self.bar.cget("mode")) != "indeterminate":
                self.bar.config(mode="indeterminate")
                self.bar.start(12)
        elif kind == "done":
            _, folder, run_dir, results, seconds = event
            self.finish_bar()
            self.set_busy(False)
            self.refresh_recent()
            if results["stats"]["input"] == 0:
                self.open_btn.config(state="disabled")
                self.stage_var.set("No photos found")
                self.detail_var.set(
                    f"{folder} has no JPEG or Sony ARW photos (subfolders included)."
                )
                return
            self.stage_var.set("Done")
            self.detail_var.set("\n".join(summary_lines(results, seconds)))
            self.open_review()
        elif kind == "error":
            self.finish_bar()
            self.set_busy(False)
            self.stage_var.set("Something went wrong")
            self.detail_var.set(event[1])
            messagebox.showerror("Photoholic", f"{event[1]}\n\nDetails: {self.log_file}")

    def finish_bar(self) -> None:
        self.bar.stop()
        self.bar.config(mode="determinate")
        self.bar["value"] = 100

    def set_busy(self, busy: bool) -> None:
        state = "disabled" if busy else "normal"
        for b in (self.choose_btn, self.recent_btn):
            b.config(state=state)
        done = not busy and self.current_run is not None
        has_results = done and (self.current_run / bridge.RESULTS_NAME).is_file()
        self.open_btn.config(state="normal" if has_results else "disabled")
        self.folder_btn.config(state="normal" if done else "disabled")

    # ---------------- review
    def open_review(self) -> None:
        if self.current_run is None:
            return
        try:
            if self.server is None:
                self.server, self.url = server.start_in_background(self.current_run)
                log.info("review GUI at %s for %s", self.url, self.current_run)
            if not os.environ.get("PHOTOHOLIC_NO_BROWSER"):  # set for automated checks
                webbrowser.open(self.url)
        except Exception as e:
            log.error("cannot open review: %s\n%s", e, traceback.format_exc())
            messagebox.showerror("Photoholic", f"Cannot open the review: {e}")

    def stop_server(self) -> None:
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()
            self.server, self.url = None, None

    def open_results_folder(self) -> None:
        if self.current_run is not None and self.current_run.is_dir():
            os.startfile(self.current_run)  # noqa: S606 - Windows only, our own folder

    def refresh_recent(self) -> None:
        self.recent = recent_runs()
        self.recent_list.delete(0, "end")
        for r in self.recent:
            when = time.strftime("%d %b %H:%M", time.localtime(r.finished))
            self.recent_list.insert(
                "end", f"{when}   {r.picks:,} picks of {r.photos:,}   {r.folder}"
            )

    def open_recent(self) -> None:
        sel = self.recent_list.curselection()
        if not sel:
            return
        chosen = self.recent[sel[0]]
        self.stop_server()
        self.current_run = chosen.run_dir
        self.stage_var.set("Opened earlier result")
        self.detail_var.set(chosen.folder)
        self.set_busy(False)
        self.open_review()

    def close(self) -> None:
        if self.worker is not None and self.worker.is_alive():
            if not messagebox.askyesno(
                "Photoholic", "A folder is still being processed. Quit anyway?"
            ):
                return
            if self.engine_proc is not None and self.engine_proc.poll() is None:
                self.engine_proc.kill()
        self.stop_server()
        self.root.destroy()


def main() -> int:
    log_file = setup_logging()
    if os.name == "nt":
        try:  # crisp text on high-DPI screens
            import ctypes

            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except (AttributeError, OSError):
            pass
    root = tk.Tk()
    try:
        ttk.Style(root).theme_use("vista")
    except tk.TclError:
        pass
    app = App(root, log_file)
    # a folder dropped onto Photoholic.exe (or given on the command line) starts right away
    if len(sys.argv) > 1 and Path(sys.argv[1]).is_dir():
        root.after(300, lambda: app.start(Path(sys.argv[1])))
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
