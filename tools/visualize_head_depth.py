#!/usr/bin/env python3
"""Generate side-by-side head RGB and pseudo-color depth video for one episode."""

from pathlib import Path
import json
import subprocess

import cv2
import numpy as np
from tqdm import tqdm

# ====== Edit here ======
DATA_ROOT = Path("/mnt/raid0/AgiBot2Lerobot/AgiBot_Word_Beta")
TASK = "351"
EPISODE = "776577"
OUTPUT_PATH = Path(f"tools/head_depth_vis_task{TASK}_ep{EPISODE}.mp4")
MAX_FRAMES = None  # set int for quick debug, e.g. 100
# =======================


def norm_task(task: str) -> str:
    task = task.strip()
    return task.split("_")[-1] if task.startswith("task_") else task


def collect_depth_files(depth_dir: Path) -> list[Path]:
    files = sorted(depth_dir.glob("head_depth*"))
    if not files:
        raise FileNotFoundError(f"No depth files found in: {depth_dir}")
    return files


def read_depth(path: Path) -> np.ndarray:
    depth = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if depth is None:
        raise RuntimeError(f"Failed to read depth image: {path}")
    depth = depth.astype(np.float32)
    if depth.max() > 20:
        depth = depth / 1000.0  # convert mm to meters
    return depth


def estimate_range(depth_files: list[Path], sample_step: int = 20) -> tuple[float, float]:
    samples = []
    for i in range(0, len(depth_files), sample_step):
        d = read_depth(depth_files[i])
        valid = np.isfinite(d) & (d > 0)
        if valid.any():
            vals = d[valid]
            if vals.size > 20000:
                vals = vals[:: max(1, vals.size // 20000)]
            samples.append(vals)
    if not samples:
        return 0.2, 2.0
    all_vals = np.concatenate(samples)
    d_min = float(np.percentile(all_vals, 2))
    d_max = float(np.percentile(all_vals, 98))
    if d_max <= d_min:
        return 0.2, 2.0
    return d_min, d_max


def depth_to_colormap(depth: np.ndarray, d_min: float, d_max: float) -> np.ndarray:
    valid = np.isfinite(depth) & (depth > 0)
    if d_max <= d_min:
        d_min, d_max = 0.2, 2.0
    clipped = np.clip(depth, d_min, d_max)
    norm = ((clipped - d_min) / (d_max - d_min) * 255.0).astype(np.uint8)
    color = cv2.applyColorMap(norm, cv2.COLORMAP_TURBO)
    color[~valid] = 0
    return color


def probe_video(video_path: Path) -> tuple[int, int, float, int]:
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height,avg_frame_rate,nb_frames",
            "-of",
            "json",
            str(video_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    info = json.loads(out.stdout)["streams"][0]
    w = int(info["width"])
    h = int(info["height"])
    rate = info.get("avg_frame_rate", "30/1")
    if "/" in rate:
        a, b = rate.split("/")
        fps = float(a) / max(float(b), 1.0)
    else:
        fps = float(rate)
    nb = info.get("nb_frames", "0")
    frame_count = int(nb) if isinstance(nb, str) and nb.isdigit() else 0
    return w, h, fps, frame_count


def main() -> None:
    task = norm_task(TASK)
    base = DATA_ROOT / "observations" / task / EPISODE
    rgb_path = base / "videos" / "head_color.mp4"
    depth_dir = base / "depth"

    if not rgb_path.exists():
        raise FileNotFoundError(f"RGB video not found: {rgb_path}")
    depth_files = collect_depth_files(depth_dir)

    w, h, fps, video_frames = probe_video(rgb_path)
    total = min(video_frames if video_frames > 0 else len(depth_files), len(depth_files))
    if isinstance(MAX_FRAMES, int) and MAX_FRAMES > 0:
        total = min(total, MAX_FRAMES)

    d_min, d_max = estimate_range(depth_files)
    print(f"Depth range (m): min={d_min:.3f}, max={d_max:.3f}")
    print(f"Frames: rgb={video_frames}, depth={len(depth_files)}, used={total}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(OUTPUT_PATH),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (w * 2, h),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Cannot open output video: {OUTPUT_PATH}")

    decode = subprocess.Popen(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-threads",
            "1",
            "-i",
            str(rgb_path),
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-",
        ],
        stdout=subprocess.PIPE,
    )

    frame_bytes = w * h * 3
    written = 0
    for i in tqdm(range(total), desc=f"task {task} ep {EPISODE}", unit="frame"):
        buf = decode.stdout.read(frame_bytes) if decode.stdout else b""
        if len(buf) != frame_bytes:
            raise RuntimeError(f"ffmpeg decode ended early at frame {i}, got {len(buf)} bytes")
        rgb = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 3).copy()

        depth = read_depth(depth_files[i])
        if depth.shape[:2] != (h, w):
            depth = cv2.resize(depth, (w, h), interpolation=cv2.INTER_NEAREST)
        depth_color = depth_to_colormap(depth, d_min, d_max)

        cv2.putText(rgb, "head RGB", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(
            depth_color,
            f"head depth pseudo ({d_min:.2f}-{d_max:.2f}m)",
            (12, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )
        writer.write(np.hstack([rgb, depth_color]))
        written += 1

    if decode.stdout:
        decode.stdout.close()
    decode.wait()
    writer.release()
    print(f"Saved: {OUTPUT_PATH} (frames={written})")


if __name__ == "__main__":
    main()
