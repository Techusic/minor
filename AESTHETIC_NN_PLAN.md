# Photoholic — Aesthetic Neural Network & Rust↔Python Bridge: Implementation Plan

> **Implementation brief.** Read this file top to bottom before writing code.
> Work phase by phase, run the acceptance checks at the end of each phase, and commit after each phase
> using the repo's commit format (see §8). Ask the user before any step marked **[ASK]**.

---

## 1. Project context

Photoholic is an AI-driven photo culling tool (repo: `Techusic/minor`). Given a folder of photos it must:

1. **Rust engine (`image-reader/`) — DONE, needs small changes (Phase 1).** Walks a folder, pairs JPEG/ARW files by stem,
   converts each image to greyscale, computes a **sharpness score** (variance of a 4-neighbour Laplacian),
   drops images below a sharpness threshold, mines EXIF, and writes `filtered_images.json`.
2. **Aesthetic neural network (`ai/`) — TO BUILD.** Gives every surviving image an **aesthetic score**.
3. **Grouping — TO BUILD.** Groups similar / burst images together.
4. **Selection — TO BUILD.** Inside each group, combines sharpness + aesthetic score and picks **one best photo**.

The **bridge** between Rust and Python is a JSON file on disk (already started: Rust writes `filtered_images.json`,
`ai/main.py` reads it). This plan formalises that contract and builds everything on the Python side.

Repo layout today:

```
minor/
├── image-reader/          # Rust engine (Cargo, edition 2024)
│   ├── src/main.rs        # discovery, sharpness, EXIF, JSON output
│   └── src/arw.rs         # extracts embedded JPEG preview from Sony .ARW
├── ai/main.py             # stub: loads filtered_images.json and prints it
├── README.md, logs.md, MINOR PROJECT PROPOSAL.docx
```

The team develops on **Windows** (paths like `D:\Photos`). All Python code must use `pathlib`, open files with
`encoding="utf-8"`, and never assume `/` separators.

---

## 2. Findings from reviewing the current Rust engine

These shape the bridge design. Fix the ones listed in Phase 1; leave the rest of the engine alone.

| # | Finding | Impact | Action |
|---|---------|--------|--------|
| F1 | Target path is hard-coded in `main()` (`D:\Photos\DSC05262.ARW` — a single file, not a folder). | Can't be driven by the Python orchestrator. | Phase 1: take input dir + output path as CLI args. |
| F2 | `extract_metadata` uses `exifreader.read_from_container(...).ok()?` — **any image without readable EXIF is silently dropped** (screenshots, edited exports, some phone JPEGs). | Good photos vanish before reaching the NN. | Phase 1: EXIF failure → keep the record with EXIF fields `null`. |
| F3 | Uses `Tag::DateTime` (file-modified time), not `DateTimeOriginal` (capture time). No sub-second time. | Burst grouping needs exact capture order. | Phase 1: add `DateTimeOriginal` + `SubSecTimeOriginal`. |
| F4 | Raw Laplacian variance is **unbounded** (values from ~15 to several thousand) and depends on resolution. Proposal says `Final Rank = Sharpness + Aesthetic`. | Adding them directly lets sharpness completely dominate. | Python side normalises sharpness **within each group** before combining (Phase 7). Rust keeps emitting the raw value. |
| F5 | Sharpness threshold `15.0` is hard-coded. | Not tunable. | Phase 1: CLI flag `--min-sharpness` (default 15.0). Also log rejected images into the JSON (`rejected` array) so the report can show them. |
| F6 | JSON is a bare array with no version, no source dir, no IDs. | Fragile contract. | Phase 1: wrap in a versioned envelope (§3.1). Python loader must still accept the legacy bare array. |
| F7 | For ARW-only files, `path` points at the `.ARW`. Python can't open ARW with Pillow. | NN input must be decoded on the Python side too. | Phase 5: Python extracts the embedded JPEG preview (same idea as `arw.rs`) via `rawpy`. |
| F8 | `ort` and `crossbeam-channel` are declared but unused; sharpness runs on full-resolution images (slow, but correct). | None for this task. | Leave as-is. Optional Phase 10 moves inference into Rust with `ort`. |

---

## 3. Architecture & bridge contract

```
 ┌──────────────────────┐   filtered_images.json   ┌──────────────────────┐  scored_images.json  ┌──────────────────┐  results.json
 │ Rust: image-reader    │ ───────────────────────▶ │ Python: ai/score      │ ──────────────────▶ │ ai/group + select │ ───────────▶ report /
 │ discover → sharpness  │                          │ ONNX aesthetic model  │  + embeddings.npy   │ clusters → best   │              copy best picks
 │ → EXIF → JSON         │                          │ (score + embedding)   │                     │ per cluster       │
 └──────────────────────┘                           └──────────────────────┘                      └──────────────────┘
          ▲                                                                                                │
          └──────────────────────────── ai/run_pipeline.py orchestrates all stages ◀───────────────────────┘
```

**Key design choice:** the neural network has **two outputs** from one forward pass:
- a 10-bin **score distribution** (NIMA style) → aesthetic mean & std, and
- a **1280-d embedding** (the backbone's pooled features) → used for similarity grouping.

So grouping costs nothing extra — no second model needed.

All bridge files live in one **run directory** (default: `<input_dir>/.photoholic/`), never inside the repo.

### 3.1 `filtered_images.json` (Rust → Python), schema v1

```json
{
  "schema_version": 1,
  "generated_at": "2026-10-24T08:44:00Z",
  "source_dir": "D:\\Photos\\Trip",
  "min_sharpness": 15.0,
  "images": [
    {
      "id": "img_000001",
      "path": "D:\\Photos\\Trip\\DSC05262.JPG",
      "arw_path": "D:\\Photos\\Trip\\DSC05262.ARW",
      "focal_length": "35 mm",
      "iso": "400",
      "shutter_speed": "1/250 s",
      "date_time": "2026:06:01 10:15:22",
      "date_time_original": "2026:06:01 10:15:22",
      "subsec_time_original": "45",
      "sharpness_score": 312.77
    }
  ],
  "rejected": [
    { "path": "D:\\Photos\\Trip\\DSC05263.JPG", "reason": "low_sharpness", "sharpness_score": 9.1 }
  ]
}
```

- `id` must be stable within a run (sequential after sorting by path is fine).
- Any EXIF field may be `null`.
- **Legacy compatibility:** if the file is a bare JSON array (today's format), the Python loader wraps it as
  `{"schema_version": 0, "images": [...], "rejected": []}` and assigns IDs itself.

### 3.2 `scored_images.json` + `embeddings.npy` (scorer → grouper)

```json
{
  "schema_version": 1,
  "model": { "name": "nima_mobilenetv2", "version": "1.0.0", "onnx_sha256": "..." },
  "images": [
    {
      "id": "img_000001",
      "path": "...",
      "sharpness_score": 312.77,
      "aesthetic_mean": 5.84,
      "aesthetic_std": 1.42,
      "aesthetic_dist": [0.01, 0.02, 0.05, 0.12, 0.25, 0.28, 0.17, 0.07, 0.02, 0.01],
      "embedding_index": 0,
      "status": "ok"
    }
  ]
}
```

- `embeddings.npy`: float32 array `[N, 1280]`, L2-normalised, row `i` ↔ `embedding_index = i`.
- If an image fails to decode: `status: "decode_error"`, scores `null`, no embedding row; it is excluded from grouping but listed in the report.

### 3.3 `results.json` (final output)

```json
{
  "schema_version": 1,
  "weights": { "sharpness": 0.4, "aesthetic": 0.6 },
  "clusters": [
    {
      "cluster_id": 0,
      "best_id": "img_000001",
      "members": [
        { "id": "img_000001", "path": "...", "sharpness_norm": 1.0, "aesthetic_norm": 0.54, "final_score": 0.72, "rank": 1 },
        { "id": "img_000002", "path": "...", "sharpness_norm": 0.31, "aesthetic_norm": 0.58, "final_score": 0.47, "rank": 2 }
      ]
    }
  ],
  "stats": { "input": 120, "rejected_sharpness": 14, "scored": 106, "clusters": 41, "reduction_pct": 61.3 }
}
```

---

## 4. Target structure for `ai/`

```
ai/
├── requirements.txt          # runtime: onnxruntime, numpy, pillow, rawpy, scikit-learn, pandas, tqdm
├── requirements-train.txt    # training: torch, torchvision, onnx, matplotlib, scipy
├── photoholic_ai/
│   ├── __init__.py
│   ├── config.py             # dataclass with all thresholds/weights; loads optional config.toml
│   ├── bridge.py             # load/validate filtered_images.json (v0 + v1), write scored/results JSON
│   ├── imaging.py            # open JPEG/ARW → RGB PIL image (rawpy embedded-preview for ARW), EXIF orientation fix
│   ├── preprocess.py         # resize to 224×224, ImageNet normalise → NCHW float32 (MUST match training)
│   ├── scorer.py             # ONNX Runtime session, batched inference, returns dist/mean/std/embedding
│   ├── grouping.py           # time + embedding similarity clustering
│   ├── selection.py          # per-cluster normalisation, weighted final score, ranking
│   └── report.py             # summary table/CSV + optional matplotlib charts (EDA deliverable)
├── training/
│   ├── dataset_ava.py        # AVA parsing, splits, torch Dataset
│   ├── model.py              # NIMA: MobileNetV2 backbone + dist head; forward returns (dist, embedding)
│   ├── losses.py             # Earth Mover's Distance (squared EMD, r=2)
│   ├── train.py              # training loop, checkpoints, early stopping
│   ├── evaluate.py           # SRCC, PLCC, binary accuracy (threshold 5.0), on test split
│   └── export_onnx.py        # export + onnxruntime parity check
├── models/                   # .onnx output (gitignored; see §8 about distributing weights)
├── run_pipeline.py           # CLI orchestrator: rust → score → group → select → report
├── main.py                   # keep; make it a thin wrapper calling run_pipeline
└── tests/
```

---

## 5. Phases

### Phase 0 — Python environment & skeleton
- Create the structure in §4, `requirements*.txt`, and a `pyproject.toml` (or plain package) so `python -m photoholic_ai...` works from `ai/`.
- Python ≥ 3.10. Pin major versions.
- Update `.gitignore`: `ai/models/*.onnx`, `ai/checkpoints/`, `ai/data/`, `.photoholic/`.
- **Accept:** `pip install -r ai/requirements.txt` succeeds; `pytest ai/tests` runs (even with zero tests).

### Phase 1 — Rust bridge changes (scope: `engine`, keep the diff small)
Implement F1, F2, F3, F5, F6 from §2:
1. CLI: `image-reader <INPUT_DIR> --out <JSON_PATH> [--min-sharpness 15.0]`. Use `std::env::args` or add `clap` (derive).
   Default `--out` = `<INPUT_DIR>/.photoholic/filtered_images.json` (create the dir). Exit code ≠ 0 with a clear message if the input dir doesn't exist.
2. EXIF read failure → still emit the record, EXIF fields `None`.
3. Add `date_time_original` (`Tag::DateTimeOriginal`) and `subsec_time_original` (`Tag::SubSecTimeOriginal`). Keep `date_time`.
4. Collect rejected-by-sharpness and decode-failure images into a `rejected` list instead of only `println!`.
5. Emit the v1 envelope (§3.1). Sort `images` by path and assign `id`s after the parallel step so output is deterministic.
6. Keep `discovered_images.txt` but write it into the same `.photoholic/` dir.
- Do **not** change the sharpness algorithm.
- **Accept:** `cargo build --release` passes with no new warnings; running on a folder with a JPEG that has no EXIF still lists it; JSON validates against §3.1.

### Phase 2 — Dataset preparation (AVA)
- Dataset: **AVA (Aesthetic Visual Analysis)** — ~255k photos, each with a histogram of 1–10 votes from many raters. This is the standard dataset for NIMA.
  **[ASK]** before downloading — it is tens of GB. Offer the user: (a) full AVA, (b) a random 30–50k subset (enough for a good MobileNetV2 head + light fine-tune), or (c) the user provides a local path.
  Also offer training on **Google Colab / Kaggle GPU** if the local machine has no CUDA GPU; make `train.py` runnable there (no hard-coded paths, args for data dir).
- `dataset_ava.py`:
  - Parse `AVA.txt` (columns: index, image_id, 10 vote counts, tags…, challenge id). Normalise counts → probability distribution.
  - Skip missing/corrupt images (log count). Use a fixed-seed split: train 90% / val 5% / test 5% (or the standard AVA test list if present).
  - Train transforms: resize to 256×256 → random crop 224 → random horizontal flip → ToTensor → ImageNet normalise.
  - Val/test transforms: **resize directly to 224×224** (no crop — composition matters for aesthetics) → normalise. **Inference uses exactly these val transforms.**
- **Accept:** a script prints dataset sizes and one batch's tensor shape `[B,3,224,224]` and distribution sums ≈ 1.

### Phase 3 — Model & training
- `model.py` — NIMA:
  - Backbone: `torchvision.models.mobilenet_v2(weights=IMAGENET1K_V2)`, take `features` → global average pool → **embedding (1280)**.
  - Head: `Dropout(0.75) → Linear(1280, 10) → Softmax` → **distribution**.
  - `forward(x)` returns `(dist, embedding)`; embedding is the pooled feature **before** dropout.
  - Keep the backbone swappable (`efficientnet_b0` as an option) via a constructor arg.
- `losses.py`: squared EMD between predicted and target CDFs: `sqrt(mean((cdf_p - cdf_t)^2))` averaged over batch.
- `train.py`:
  - Stage A (≈5 epochs): freeze backbone, train head, lr 1e-3, Adam.
  - Stage B (≈10–15 epochs): unfreeze all, backbone lr 3e-6–1e-5, head lr 1e-4; cosine or step decay.
  - Batch 64 (reduce if OOM), AMP on CUDA, `num_workers` arg (Windows: default 0–2), seed everything.
  - Save best checkpoint by val EMD to `ai/checkpoints/`; log train/val loss per epoch to CSV.
  - Resume from checkpoint supported.
- `evaluate.py` on test split: report **SRCC** and **PLCC** between predicted mean and ground-truth mean, **binary accuracy** (mean ≥ 5 = "good"), and EMD.
  Realistic targets for MobileNetV2-NIMA: SRCC ≈ 0.55–0.65 on a subset, ~0.6+ on full AVA. Below 0.45 → something is wrong (check normalisation/labels).
- **Accept:** evaluate script prints metrics; SRCC ≥ 0.50 on held-out test. Save metrics to `ai/models/metrics.json`.

### Phase 4 — ONNX export
- `export_onnx.py`: load best checkpoint → `torch.onnx.export` with input name `input` `[N,3,224,224]` float32,
  outputs `dist` `[N,10]` and `embedding` `[N,1280]`, **dynamic batch axis**, opset 17.
- Write `ai/models/aesthetic_v1.onnx` + `ai/models/aesthetic_v1.json` sidecar (input size, mean/std, model version, sha256, metrics).
- Parity check: run 8 random test images through PyTorch and onnxruntime; assert max abs diff < 1e-4 on `dist`.
- **Accept:** `onnx.checker` passes; parity check passes; file size ≈ 9 MB.

### Phase 5 — Inference runtime (`imaging.py`, `preprocess.py`, `scorer.py`)
- `imaging.load_rgb(path)`:
  - `.jpg/.jpeg` → Pillow, apply `ImageOps.exif_transpose`, convert RGB.
  - `.arw` → `rawpy.imread(path).extract_thumb()`; if JPEG thumb → open bytes with Pillow; if bitmap → `Image.fromarray`. Fallback: `postprocess(half_size=True)`. (Same idea as Rust `arw.rs` — the embedded preview is fast and good enough at 224 px.)
  - Use `Image.draft("RGB", (448, 448))` on JPEGs for fast downscaled decode.
- `preprocess.to_tensor(img)`: resize to 224×224 (bilinear, antialias) → float32 /255 → ImageNet mean/std → CHW. Read mean/std/size from the model sidecar JSON so training and inference can never drift.
- `scorer.Scorer(onnx_path, providers=auto)`: prefer `CUDAExecutionProvider`/`DmlExecutionProvider` (Windows GPU) if available, else CPU.
  Decode images in a `ThreadPoolExecutor`, run inference in batches of 32. Compute `mean = Σ p_i·i`, `std = sqrt(Σ p_i·(i-mean)²)` for i = 1..10. L2-normalise embeddings.
- Writes `scored_images.json` + `embeddings.npy` (§3.2).
- **Accept:** scoring 100 JPEGs on CPU takes < 30 s; a corrupt file yields `status: "decode_error"` instead of crashing.

### Phase 6 — Grouping similar images (`grouping.py`)
Goal: put burst / near-duplicate shots of the same moment in one group.
1. Parse capture time from `date_time_original` (+ `subsec_time_original`); fall back to `date_time`; else `None`.
2. Build edges with union-find:
   - If both images have times **and** `|Δt| ≤ burst_gap_s` (default **10 s**) **and** cosine similarity ≥ `sim_threshold_timed` (default **0.80**) → same group.
   - Otherwise (no times, or far apart) → join only if cosine similarity ≥ `sim_threshold_untimed` (default **0.92**).
   - For speed: sort by time and only compare within a sliding window; for untimed images compare all pairs (N is small — hundreds) with one matrix multiply.
3. Singletons are valid groups of size 1.
4. All thresholds live in `config.py` and can be overridden from CLI.
- Optional alternative behind a flag: `sklearn.cluster.AgglomerativeClustering(metric="cosine", linkage="average", distance_threshold=...)`.
- **Accept:** unit test with synthetic embeddings (two tight clusters + one outlier) yields 3 groups; a manual check on the team's own photo folder looks sensible. **[ASK]** the user to eyeball the first real result and tune thresholds.

### Phase 7 — Decision matrix & selection (`selection.py`)
Fixes F4 — scores are on different scales, so normalise before combining:
- `sharp_log = log1p(sharpness_score)`; inside each group min-max → `sharpness_norm ∈ [0,1]` (group of 1 → 1.0).
- `aesthetic_norm = (aesthetic_mean - 1) / 9` (absolute scale, so a group of mediocre shots stays mediocre).
- `final_score = w_s · sharpness_norm + w_a · aesthetic_norm`, defaults `w_s = 0.4`, `w_a = 0.6`, configurable. (This is the proposal's `Final Rank = Sharpness + Aesthetic`, made scale-safe.)
- Tie-break: higher aesthetic_mean, then lower aesthetic_std, then earlier capture time.
- Rank members; `best_id` = rank 1. Write `results.json` (§3.3).
- **Action stage (safe by default):** `--action none|copy|move` (default `none`). `copy` copies each group's best (JPEG + paired ARW) into `<input_dir>/photoholic_picks/`. `move` moves non-best files to `<input_dir>/photoholic_rejects/`.
  **Never delete files.** Never overwrite existing files (add suffix).
- **Accept:** unit tests for normalisation edge cases (equal sharpness in a group, group of one, missing aesthetic).

### Phase 8 — Orchestrator CLI (`run_pipeline.py`)
```
python ai/run_pipeline.py "D:\Photos\Trip" \
    --engine image-reader/target/release/image-reader(.exe) \
    --model ai/models/aesthetic_v1.onnx \
    [--min-sharpness 15] [--burst-gap 10] [--sim 0.80] [--w-sharp 0.4 --w-aes 0.6] [--action none]
    [--skip-engine]   # reuse existing filtered_images.json
```
- Runs the Rust binary via `subprocess.run([...], check=True)` (list args, no shell), then score → group → select → report.
- Prints a summary: input count, rejected for blur, groups found, reduction %, total time per stage.
- `report.py` writes `report.csv` (one row per image with all scores, group, rank) and optional charts
  (aesthetic histogram, sharpness vs ISO/shutter scatter) for the EDA deliverable.
- Make `ai/main.py` call this so existing usage keeps working.
- **Accept:** end-to-end on a folder of ~100 images completes in < 60 s on CPU (README deliverable).

### Phase 9 — Tests (`ai/tests/`, pytest)
- `test_bridge.py`: loads v0 bare-array and v1 envelope; rejects malformed input with a clear error.
- `test_preprocess.py`: output shape/dtype; values match a hand-computed pixel.
- `test_scorer.py`: with a tiny dummy ONNX model generated in the test (2 outputs, correct shapes) — no dependency on trained weights.
- `test_grouping.py`, `test_selection.py`: synthetic data, edge cases.
- `test_e2e.py`: generate ~6 synthetic JPEGs with Pillow (some blurred copies), run score→group→select with the dummy model, assert schema of `results.json`.
- Generate test images in `tmp_path`; never commit binary fixtures larger than a few KB.

### Phase 10 — (Optional, later) Inference inside Rust
The README's Stage 3 plans `ort` in Rust. Once the Python path works, the same `aesthetic_v1.onnx` + sidecar can be loaded
by the Rust engine to emit `aesthetic_*` fields directly into the JSON, making Python only responsible for grouping/selection.
Only start this if the user asks. Keep the JSON contract identical so Python doesn't change.

---

## 6. Config defaults (single source: `ai/photoholic_ai/config.py`)

| Key | Default | Meaning |
|-----|---------|---------|
| `min_sharpness` | 15.0 | passed to Rust `--min-sharpness` |
| `input_size` | 224 | from model sidecar |
| `batch_size` | 32 | inference batch |
| `burst_gap_s` | 10 | max seconds between burst frames |
| `sim_threshold_timed` | 0.80 | cosine sim for shots close in time |
| `sim_threshold_untimed` | 0.92 | cosine sim when time missing/far |
| `w_sharpness` / `w_aesthetic` | 0.4 / 0.6 | decision-matrix weights (must sum to 1) |
| `action` | `none` | `none` / `copy` / `move` — never delete |

---

## 7. Development rules

1. **Never delete or modify the user's original photos.** Only read them; outputs go to `.photoholic/` or `photoholic_picks/`.
2. **[ASK]** before: downloading datasets, installing CUDA/GPU packages, starting a training run longer than ~10 minutes, or changing the Rust sharpness algorithm.
3. Training and inference must share one preprocessing definition (sidecar JSON). Any mismatch silently ruins scores.
4. Keep the Rust diff in Phase 1 minimal and idiomatic; run `cargo fmt` and `cargo clippy`.
5. Python: type hints, `pathlib`, `logging` (not bare prints in library code), black/ruff formatting.
6. Everything must work on Windows; test path handling with backslashes.
7. After each phase: run its acceptance check, update `logs.md` with a dated entry (match the existing style), commit.

## 8. Repo conventions

- Commit format from README: `type(scope): imperative description`, scopes `engine`, `ai`, `gui`, `repo`.
  Examples: `feat(engine): accept input dir and output path as CLI args`, `feat(ai): add NIMA MobileNetV2 model and EMD loss`,
  `feat(ai): export aesthetic model to ONNX`, `chore(repo): ignore model weights and run outputs`.
- Model weights (`.onnx`, ~9 MB) are gitignored by default. **[ASK]** the user whether to use Git LFS, a GitHub Release asset,
  or a shared drive link, and document the choice in README.

## 9. Open decisions for the team (ask, do not assume)

1. Dataset size: full AVA vs subset, and where training will run (local GPU / Colab / Kaggle).
2. Decision-matrix weights (start 0.4 / 0.6) — tune after looking at real results.
3. Default action: report only, copy best picks, or move rejects.
4. Whether to later move inference into Rust (Phase 10) or keep it in Python.

## 10. Definition of done

- `python ai/run_pipeline.py <folder>` runs Rust → NN → grouping → selection end to end and writes `results.json` + `report.csv`.
- Aesthetic model: SRCC ≥ 0.50 on held-out AVA test, exported to ONNX with parity check passing.
- ~100 images processed in under 60 s on a normal laptop CPU.
- All tests pass; README updated with setup, training, and usage instructions.
