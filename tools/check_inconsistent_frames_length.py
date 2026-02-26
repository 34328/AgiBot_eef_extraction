#!/usr/bin/env python3
"""Scan specified tasks and record episodes with inconsistent frame lengths."""

from __future__ import annotations

import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import h5py
from tqdm import tqdm
import tyro

STATE_KEYS = [
    ("state.effector.position", "state/effector/position"),
    ("state.head.position", "state/head/position"),
    ("state.joint.position", "state/joint/position"),
    ("state.waist.position", "state/waist/position"),
    ("eef.position", "state/end/position"),
    ("eef.orientation", "state/end/orientation"),
]


def norm_task(task: str) -> str:
    t = task.strip()
    return t.split("_")[-1] if t.startswith("task_") else t


def get_video_frames(video_path: Path) -> tuple[int | None, str | None]:
    try:
        out = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=nb_frames",
                "-of",
                "json",
                str(video_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return None, "ffprobe_not_found"
    except subprocess.CalledProcessError as e:
        msg = (e.stderr or "").strip().splitlines()
        tail = msg[-1] if msg else "ffprobe_failed"
        return None, f"video_unopenable:{tail[:120]}"

    try:
        payload = json.loads(out.stdout)
    except json.JSONDecodeError:
        return None, "ffprobe_bad_json"

    streams = payload.get("streams")
    if not streams:
        return None, "video_no_stream"
    info = streams[0]

    nb_frames = info.get("nb_frames")
    if isinstance(nb_frames, str) and nb_frames.isdigit():
        return int(nb_frames), None
    return None, f"nb_frames_invalid:{nb_frames}"


def get_episode_ids(data_root: Path, task: str) -> list[str]:
    roots = [
        data_root / "observations" / task,
        data_root / "proprio_stats" / task,
        data_root / "parameters" / task,
    ]
    ids = set()
    for root in roots:
        if root.is_dir():
            ids.update(p.name for p in root.iterdir() if p.is_dir())
    return sorted(ids, key=lambda x: int(x) if x.isdigit() else x)


def check_episode(data_root: Path, task: str, episode: str) -> dict | None:
    lengths: dict[str, int | None] = {}
    reasons: list[str] = []

    h5_path = data_root / "proprio_stats" / task / episode / "proprio_stats.h5"
    if h5_path.exists():
        with h5py.File(h5_path, "r") as f:
            for out_key, h5_key in STATE_KEYS:
                if h5_key in f:
                    lengths[out_key] = int(f[h5_key].shape[0])
                else:
                    lengths[out_key] = None
                    reasons.append(f"missing:{out_key}")
    else:
        reasons.append("missing:proprio_stats.h5")

    obs_dir = data_root / "observations" / task / episode
    video_files: list[Path] = []
    videos_dir = obs_dir / "videos"
    if videos_dir.is_dir():
        video_files.extend(sorted(videos_dir.glob("*.mp4")))

    if not video_files:
        reasons.append("missing:videos")

    with ThreadPoolExecutor(max_workers=min(8, len(video_files) or 1)) as ex:
        futures = {ex.submit(get_video_frames, video): video for video in video_files}
        for future, video in futures.items():
            k = f"video.{video.relative_to(obs_dir).as_posix()}"
            try:
                n, reason = future.result()
            except Exception:
                n = None
                reason = "video_check_crash"
            lengths[k] = n
            if n is None:
                reasons.append(f"{reason}:{video.name}")

    depth_dir = obs_dir / "depth"
    if depth_dir.is_dir():
        depth_files = sorted(depth_dir.glob("head_depth*"))
        lengths["depth.head"] = len(depth_files)
        if not depth_files:
            reasons.append("missing:depth")
    else:
        lengths["depth.head"] = None
        reasons.append("missing:depth")

    valid = [v for v in lengths.values() if isinstance(v, int)]
    inconsistent = len(set(valid)) > 1 if valid else True
    if not inconsistent and not reasons:
        return None

    return {
        "task": task,
        "episode": episode,
        "lengths": lengths,
        "unique_lengths": sorted(set(valid)),
        "reasons": sorted(set(reasons)),
    }


def iter_with_progress(task: str, episode_ids: list[str]):
    for episode in tqdm(episode_ids, desc=f"task {task}", unit="ep"):
        yield episode


@dataclass
class ArgsConfig:
    data_root: Path = Path("/mnt/raid0/AgiBot2Lerobot/AgiBot_Word_Beta")
    tasks: list[str] = field(default_factory=lambda: ["359"])
    output: Path = Path("filter_store/Inconsistent_frams_length.jsonl")


def main(cfg: ArgsConfig) -> None:
    if not cfg.tasks:
        raise ValueError("Please provide --tasks, e.g. --tasks 327 or --tasks task_327")

    tasks = [norm_task(t) for t in cfg.tasks]
    bad_items: list[dict] = []

    for task in tasks:
        episode_ids = get_episode_ids(cfg.data_root, task)
        task_bad = 0
        for episode in iter_with_progress(task, episode_ids):
            item = check_episode(cfg.data_root, task, episode)
            if item is not None:
                bad_items.append(item)
                task_bad += 1
                tqdm.write(
                    f"[BAD] task={item['task']} episode={item['episode']} "
                    f"lengths={item['unique_lengths']} reasons={item['reasons']}"
                )
        tqdm.write(f"[TASK DONE] task={task} episodes={len(episode_ids)} inconsistent={task_bad}")
        print()

    cfg.output.parent.mkdir(parents=True, exist_ok=True)

    merged: dict[tuple[str, str], dict] = {}
    if cfg.output.exists():
        with cfg.output.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                task = obj.get("task")
                episode = obj.get("episode")
                if task is None or episode is None:
                    continue
                merged[(str(task), str(episode))] = obj

    appended = 0
    skipped = 0
    for item in bad_items:
        key = (str(item["task"]), str(item["episode"]))
        if key in merged:
            skipped += 1
            continue
        merged[key] = item
        appended += 1

    def _sort_key(k: tuple[str, str]) -> tuple[int, int | str, int, int | str]:
        task, episode = k
        task_is_num = 0 if task.isdigit() else 1
        ep_is_num = 0 if episode.isdigit() else 1
        task_val = int(task) if task.isdigit() else task
        ep_val = int(episode) if episode.isdigit() else episode
        return (task_is_num, task_val, ep_is_num, ep_val)

    with cfg.output.open("w", encoding="utf-8") as f:
        for key in sorted(merged.keys(), key=_sort_key):
            f.write(json.dumps(merged[key], ensure_ascii=False) + "\n")

    print(
        f"tasks={tasks}, inconsistent={len(bad_items)}, appended={appended}, skipped_duplicate={skipped}, total_sorted={len(merged)}, output={cfg.output}"
    )


if __name__ == "__main__":
    cfg = tyro.cli(ArgsConfig)
    main(cfg)
