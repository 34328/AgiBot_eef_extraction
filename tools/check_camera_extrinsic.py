#!/usr/bin/env python3
"""Scan tasks and flag episodes whose head-camera EEF visualization is mostly out of frame.

Rule:
- Reuse the same projection logic as EEF visualization.
- For each frame, if both left/right EEF points are outside image bounds, mark frame as missing.
- If missing ratio > threshold (default 0.4), episode is considered bad extrinsic.
"""

from __future__ import annotations

import base64
import json
import re
import subprocess
import time
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


def extract_video_frame(video_path: Path, frame_idx: int) -> np.ndarray:
    vf = f"select=eq(n\\,{frame_idx})"
    out = subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(video_path),
            "-vf",
            vf,
            "-frames:v",
            "1",
            "-f",
            "image2pipe",
            "-vcodec",
            "png",
            "-",
        ],
        check=True,
        capture_output=True,
    )
    frame = cv2.imdecode(np.frombuffer(out.stdout, np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError("ffmpeg_decode_failed")
    return frame


def project_points(pts: np.ndarray, R: np.ndarray, t: np.ndarray, K: np.ndarray, dist: np.ndarray) -> np.ndarray:
    R_inv, t_inv = R.T, -R.T @ t
    rvec, _ = cv2.Rodrigues(R_inv)
    pts_2d, _ = cv2.projectPoints(
        pts.reshape(-1, 1, 3),
        rvec,
        t_inv.reshape(3, 1),
        K,
        dist,
    )
    return pts_2d.reshape(-1, 2)


def draw_eef_on_frame(
    frame: np.ndarray,
    pos_lr: np.ndarray,
    quat_lr: np.ndarray | None,
    R: np.ndarray,
    t: np.ndarray,
    K: np.ndarray,
    dist: np.ndarray,
    gripper_offset: float,
    axis_len: float,
) -> np.ndarray:
    h, w = frame.shape[:2]
    gripper_pos = apply_gripper_offset(pos_lr, quat_lr, gripper_offset)
    pts_2d = project_points(gripper_pos, R, t, K, dist)
    colors = [(0, 0, 255), (255, 0, 0)]  # L red, R blue
    labels = ["L", "R"]
    axis_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]  # xyz

    for i in range(2):
        x, y = int(pts_2d[i, 0]), int(pts_2d[i, 1])
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(frame, (x, y), 8, colors[i], -1)
            cv2.putText(frame, labels[i], (x + 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[i], 2)

            if quat_lr is not None:
                R_eef = quat_to_rotmat(quat_lr[i])
                axes_3d = gripper_pos[i] + (R_eef @ (np.eye(3) * axis_len)).T
                axes_2d = project_points(axes_3d, R, t, K, dist)
                for j in range(3):
                    ax, ay = int(axes_2d[j, 0]), int(axes_2d[j, 1])
                    if 0 <= ax < w and 0 <= ay < h:
                        cv2.line(frame, (x, y), (ax, ay), axis_colors[j], 2)

    info = f"L:({pts_2d[0,0]:.0f},{pts_2d[0,1]:.0f}) R:({pts_2d[1,0]:.0f},{pts_2d[1,1]:.0f})"
    cv2.putText(frame, info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return frame


def _parse_bool_from_text(text: str) -> bool | None:
    text = text.strip()
    if not text:
        return None
    try:
        obj = json.loads(text)
        if isinstance(obj, bool):
            return obj
        if isinstance(obj, dict):
            for k in ("result", "ok", "is_eef_aligned", "aligned"):
                v = obj.get(k)
                if isinstance(v, bool):
                    return v
    except Exception:
        pass

    lower = text.lower()
    true_match = re.search(r"\btrue\b", lower)
    false_match = re.search(r"\bfalse\b", lower)
    if true_match and false_match:
        return true_match.start() < false_match.start()
    if true_match:
        return True
    if false_match:
        return False
    return None


def call_vl_judge_bool(
    image_path: Path,
    model_id: str,
    api_base: str,
    api_key: str,
    timeout_sec: float,
    retries: int,
) -> tuple[bool | None, str | None]:
    try:
        from openai import OpenAI
    except Exception as e:
        return None, f"import_error:{type(e).__name__}"

    image_b64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    image_url = f"data:image/png;base64,{image_b64}"
    prompt = (
        "You are checking robot EEF visualization quality. "
        "Given this frame with projected left/right EEF markers and axes, "
        "judge whether markers are roughly located at the physical arm end-effectors (grippers). "
        "Return ONLY one word: true or false."
    )

    last_err = None
    for _ in range(max(1, retries)):
        try:
            client = OpenAI(api_key=api_key, base_url=api_base, timeout=timeout_sec)
            resp = client.chat.completions.create(
                model=model_id,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {"type": "image_url", "image_url": {"url": image_url}},
                        ],
                    }
                ],
                temperature=0,
                max_tokens=32,
            )
            text = ""
            if resp.choices and resp.choices[0].message:
                text = resp.choices[0].message.content or ""
            parsed = _parse_bool_from_text(text)
            if parsed is None:
                last_err = "parse_error"
                time.sleep(0.2)
                continue
            return parsed, None
        except Exception as e:
            last_err = f"api_error:{type(e).__name__}"
            time.sleep(0.2)
            continue
    return None, last_err or "unknown_error"


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
    vl_second_stage: bool,
    vl_model_id: str,
    vl_api_base: str,
    vl_api_key: str,
    vl_timeout_sec: float,
    vl_retries: int,
    vl_image_dir: Path,
    vl_axis_len: float,
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

    if not vl_second_stage:
        return None

    mid = num_frames // 2
    try:
        frame = extract_video_frame(paths["video"], mid)
        pos_lr = pos_all[mid]
        quat_lr = quat_all[mid] if quat_all is not None else None
        vis = draw_eef_on_frame(
            frame=frame,
            pos_lr=pos_lr,
            quat_lr=quat_lr,
            R=R_all[mid],
            t=t_all[mid],
            K=K,
            dist=dist,
            gripper_offset=gripper_offset,
            axis_len=vl_axis_len,
        )
        vl_image_dir.mkdir(parents=True, exist_ok=True)
        img_path = vl_image_dir / f"{task}_{episode}_{mid}_pending.png"
        ok = cv2.imwrite(str(img_path), vis)
        if not ok:
            raise RuntimeError("imwrite_failed")
    except Exception as e:
        return {
            "task": task,
            "episode": episode,
            "missing_ratio": round(missing_ratio, 6),
            "missing_frames": missing_count,
            "total_frames": num_frames,
            "vl_result": None,
            "reason": f"vl_render_error:{type(e).__name__}",
        }

    vl_result, vl_err = call_vl_judge_bool(
        image_path=img_path,
        model_id=vl_model_id,
        api_base=vl_api_base,
        api_key=vl_api_key,
        timeout_sec=vl_timeout_sec,
        retries=vl_retries,
    )
    result_tag = str(vl_result).lower() if isinstance(vl_result, bool) else "none"
    final_img_path = vl_image_dir / f"{task}_{episode}_{mid}_{result_tag}.png"
    try:
        img_path.rename(final_img_path)
        img_path = final_img_path
    except Exception:
        pass

    if vl_result is None:
        return {
            "task": task,
            "episode": episode,
            "missing_ratio": round(missing_ratio, 6),
            "missing_frames": missing_count,
            "total_frames": num_frames,
            "vl_result": None,
            "reason": f"vl_api_error:{vl_err}",
            "vl_image": str(img_path),
        }
    if not vl_result:
        return {
            "task": task,
            "episode": episode,
            "missing_ratio": round(missing_ratio, 6),
            "missing_frames": missing_count,
            "total_frames": num_frames,
            "vl_result": False,
            "reason": "vl_not_on_gripper_tip",
            "vl_image": str(img_path),
        }
    return None



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
        task_vl_image_dir = cfg.vl_image_dir.parent / f"{cfg.vl_image_dir.name}_{task}"
        for episode in tqdm(episode_ids, desc=f"task {task}", unit="ep"):
            item = check_episode(
                data_root=cfg.data_root,
                task=task,
                episode=episode,
                threshold=cfg.threshold,
                gripper_offset=cfg.gripper_offset,
                vl_second_stage=cfg.vl_second_stage,
                vl_model_id=cfg.vl_model_id,
                vl_api_base=cfg.vl_api_base,
                vl_api_key=cfg.vl_api_key,
                vl_timeout_sec=cfg.vl_timeout_sec,
                vl_retries=cfg.vl_retries,
                vl_image_dir=task_vl_image_dir,
                vl_axis_len=cfg.vl_axis_len,
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
            "vl_result": item.get("vl_result"),
            "vl_image": item.get("vl_image"),
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

@dataclass
class ArgsConfig:
    data_root: Path = Path("/mnt/raid0/AgiBot2Lerobot/AgiBot_Word_Beta")
    tasks: list[str] = field(default_factory=lambda: ["359"])
    output: Path = Path("filter_store/camera_extrinsic.jsonl")
    threshold: float = 0.4
    gripper_offset: float = 0.143
    vl_second_stage: bool = True
    vl_model_id: str = ""
    vl_api_base: str = ""
    vl_api_key: str = ""
    vl_timeout_sec: float = 30.0
    vl_retries: int = 3
    vl_axis_len: float = 0.08
    vl_image_dir: Path = Path("filter_store/vl_frames")
    camera_name: str = "head"
    verbose: bool = False

if __name__ == "__main__":
    cfg = tyro.cli(ArgsConfig)
    main(cfg)
