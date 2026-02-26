#!/usr/bin/env python3
"""Scan tasks and flag episodes whose head-camera EEF visualization is mostly out of frame.

Rule:
- Reuse the same projection logic as EEF visualization.
- For each frame, if both left/right EEF points are outside image bounds, mark frame as missing.
- If missing ratio > threshold (default 0.4), episode is considered bad extrinsic.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import h5py
import numpy as np
import tyro
from tqdm import tqdm


def norm_task(task: str) -> str:
    t = task.strip()
    return t.split("_")[-1] if t.startswith("task_") else t


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


def build_paths(data_root: Path, task: str, episode: str, camera_name: str = "head") -> dict[str, Path]:
    base = data_root / "observations" / task / episode
    return {
        "video": base / "videos" / f"{camera_name}_color.mp4",
        "h5": data_root / "proprio_stats" / task / episode / "proprio_stats.h5",
        "intrinsic": data_root
        / "parameters"
        / task
        / episode
        / "parameters"
        / "camera"
        / f"{camera_name}_intrinsic_params.json",
        "extrinsic": data_root
        / "parameters"
        / task
        / episode
        / "parameters"
        / "camera"
        / f"{camera_name}_extrinsic_params_aligned.json",
    }


def load_intrinsic(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = json.loads(path.read_text())["intrinsic"]
    K = np.array(
        [
            [data["fx"], 0, data.get("ppx", data.get("cx"))],
            [0, data["fy"], data.get("ppy", data.get("cy"))],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )
    dist = np.array(
        [
            data.get("k1", 0),
            data.get("k2", 0),
            data.get("p1", 0),
            data.get("p2", 0),
            data.get("k3", 0),
        ],
        dtype=np.float64,
    )
    return K, dist


def load_extrinsics(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        data = [data]
    R_all, t_all = [], []
    for entry in data:
        extr = entry.get("extrinsic", entry)
        R_all.append(extr["rotation_matrix"])
        t_all.append(extr["translation_vector"])
    return np.array(R_all, dtype=np.float64), np.array(t_all, dtype=np.float64)


def load_eef(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    with h5py.File(path, "r") as f:
        pos = np.array(f["state/end/position"], dtype=np.float64)
        quat = (
            np.array(f["state/end/orientation"], dtype=np.float64)
            if "state/end/orientation" in f
            else None
        )
    return pos, quat


def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    x, y, z, w = q / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def apply_gripper_offset(pos: np.ndarray, quat: np.ndarray | None, offset: float) -> np.ndarray:
    if quat is None or offset == 0:
        return pos
    gripper_pos = pos.copy()
    for i in range(2):
        R_eef = quat_to_rotmat(quat[i])
        z_axis = R_eef[:, 2]
        gripper_pos[i] = pos[i] + offset * z_axis
    return gripper_pos


def get_video_size(video_path: Path) -> tuple[int, int]:
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "v:0",
            "-show_entries",
            "stream=width,height",
            "-of",
            "json",
            str(video_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    info = json.loads(out.stdout)["streams"][0]
    return int(info["width"]), int(info["height"])


def frame_has_visible_eef(
    pos_lr: np.ndarray,
    quat_lr: np.ndarray | None,
    R: np.ndarray,
    t: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    w: int,
    h: int,
    gripper_offset: float,
) -> bool:
    gripper_pos = apply_gripper_offset(pos_lr, quat_lr, gripper_offset)

    R_inv, t_inv = R.T, -R.T @ t
    rvec, _ = cv2.Rodrigues(R_inv)
    pts_2d, _ = cv2.projectPoints(
        gripper_pos.reshape(-1, 1, 3),
        rvec,
        t_inv.reshape(3, 1),
        K,
        dist,
    )
    pts_2d = pts_2d.reshape(-1, 2)

    for i in range(2):
        x, y = int(pts_2d[i, 0]), int(pts_2d[i, 1])
        if 0 <= x < w and 0 <= y < h:
            return True
    return False


def check_episode(
    data_root: Path,
    task: str,
    episode: str,
    threshold: float,
    gripper_offset: float,
    camera_name: str = "head",
) -> dict | None:
    paths = build_paths(data_root, task, episode, camera_name)

    missing_files = [k for k, v in paths.items() if not v.exists()]
    if missing_files:
        return {
            "task": task,
            "episode": episode,
            "missing_ratio": 1.0,
            "missing_frames": None,
            "total_frames": None,
            "reason": f"missing_files:{','.join(missing_files)}",
        }

    try:
        K, dist = load_intrinsic(paths["intrinsic"])
        R_all, t_all = load_extrinsics(paths["extrinsic"])
        pos_all, quat_all = load_eef(paths["h5"])
        w, h = get_video_size(paths["video"])
    except Exception as e:
        return {
            "task": task,
            "episode": episode,
            "missing_ratio": 1.0,
            "missing_frames": None,
            "total_frames": None,
            "reason": f"load_error:{type(e).__name__}",
        }

    num_frames = min(len(pos_all), len(R_all), len(t_all))
    if num_frames <= 0:
        return {
            "task": task,
            "episode": episode,
            "missing_ratio": 1.0,
            "missing_frames": 0,
            "total_frames": 0,
            "reason": "empty_data",
        }

    missing_count = 0
    for i in range(num_frames):
        pos_lr = pos_all[i]
        quat_lr = quat_all[i] if quat_all is not None else None
        visible = frame_has_visible_eef(
            pos_lr=pos_lr,
            quat_lr=quat_lr,
            R=R_all[i],
            t=t_all[i],
            K=K,
            dist=dist,
            w=w,
            h=h,
            gripper_offset=gripper_offset,
        )
        if not visible:
            missing_count += 1

    missing_ratio = missing_count / num_frames
    if missing_ratio > threshold:
        return {
            "task": task,
            "episode": episode,
            "missing_ratio": round(missing_ratio, 6),
            "missing_frames": missing_count,
            "total_frames": num_frames,
            "reason": "eef_out_of_frame_ratio_too_high",
        }
    return None


@dataclass
class ArgsConfig:
    data_root: Path = Path("/mnt/raid0/AgiBot2Lerobot/AgiBot_Word_Beta")
    tasks: list[str] = field(default_factory=lambda: ["359"])
    output: Path = Path("filter_store/camera_extrinsic.jsonl")
    threshold: float = 0.4
    gripper_offset: float = 0.143
    camera_name: str = "head"
    verbose: bool = False


def main(cfg: ArgsConfig) -> None:
    if not cfg.tasks:
        raise ValueError("Please provide --tasks, e.g. --tasks 327 or --tasks task_327")

    tasks = [norm_task(t) for t in cfg.tasks]
    bad_items: list[dict] = []

    for task in tasks:
        episode_ids = get_episode_ids(cfg.data_root, task)
        if not episode_ids:
            tqdm.write(f"[TASK SKIP] task={task} no episodes found")
            continue

        task_bad = 0
        for episode in tqdm(episode_ids, desc=f"task {task}", unit="ep"):
            item = check_episode(
                data_root=cfg.data_root,
                task=task,
                episode=episode,
                threshold=cfg.threshold,
                gripper_offset=cfg.gripper_offset,
                camera_name=cfg.camera_name,
            )
            if item is not None:
                bad_items.append(item)
                task_bad += 1
                if cfg.verbose:
                    tqdm.write(
                        f"[BAD] task={task} episode={episode} "
                        f"ratio={item['missing_ratio']} reason={item['reason']}"
                    )

        tqdm.write(f"[TASK DONE] task={task} episodes={len(episode_ids)} bad={task_bad}")
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
        merged[key] = {
            "task": key[0],
            "episode": key[1],
            "missing_ratio": item.get("missing_ratio"),
            "missing_frames": item.get("missing_frames"),
            "total_frames": item.get("total_frames"),
            "reason": item.get("reason"),
        }
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
        f"tasks={tasks}, bad={len(bad_items)}, appended={appended}, skipped_duplicate={skipped}, total_sorted={len(merged)}, output={cfg.output}"
    )


if __name__ == "__main__":
    cfg = tyro.cli(ArgsConfig)
    main(cfg)
