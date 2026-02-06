from __future__ import annotations

import os
from pathlib import Path
from functools import lru_cache
import json
import subprocess
from flask import Flask, jsonify, render_template, send_file, abort

from generate_eef_video import get_eef_video_path

# 数据根目录
DATA_ROOT = Path("/mnt/raid0/AgiBot2Lerobot/AgiBot_Word_Beta")
BASE_DIR = DATA_ROOT / "observations"

VIDEO_NAMES = {
    "head": "head_color.mp4",
    "left": "hand_left_color.mp4",
    "right": "hand_right_color.mp4",
}

app = Flask(__name__, static_folder="static", template_folder="templates")


def _is_safe_dir(path: Path) -> bool:
    try:
        path.resolve().relative_to(BASE_DIR.resolve())
        return True
    except ValueError:
        return False


def _list_dirs(path: Path) -> list[str]:
    if not path.exists() or not path.is_dir():
        return []
    items = [p.name for p in path.iterdir() if p.is_dir()]
    return sorted(items, key=lambda x: int(x) if x.isdigit() else x)


def _parse_ffprobe_output(raw: str) -> dict | None:
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None
    if not payload.get("streams"):
        return None
    stream = payload["streams"][0]
    nb_frames = stream.get("nb_frames")
    r_frame_rate = stream.get("r_frame_rate")
    duration = stream.get("duration")

    frames = int(nb_frames) if nb_frames and nb_frames.isdigit() else None
    fps = None
    if r_frame_rate and isinstance(r_frame_rate, str) and "/" in r_frame_rate:
        num, denom = r_frame_rate.split("/", 1)
        try:
            fps = float(num) / float(denom)
        except (ValueError, ZeroDivisionError):
            fps = None
    try:
        duration_value = float(duration) if duration is not None else None
    except ValueError:
        duration_value = None

    return {"frames": frames, "fps": fps, "duration": duration_value}


def _get_video_meta(video_path: Path) -> tuple[dict | None, str | None]:
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=nb_frames,r_frame_rate,duration",
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
    except subprocess.CalledProcessError:
        return None, "ffprobe_failed"

    meta = _parse_ffprobe_output(result.stdout)
    return meta, None


@lru_cache(maxsize=256)
def _cached_tasks() -> list[str]:
    return _list_dirs(BASE_DIR)


@lru_cache(maxsize=1024)
def _cached_episodes(task: str) -> list[str]:
    task_dir = BASE_DIR / task
    if not _is_safe_dir(task_dir):
        return []
    return _list_dirs(task_dir)


@lru_cache(maxsize=2048)
def _cached_meta(task: str, episode: str) -> dict:
    task_dir = BASE_DIR / task
    episode_dir = task_dir / episode
    if not _is_safe_dir(episode_dir) or not episode_dir.is_dir():
        return {"head": None, "left": None, "right": None, "warning": "not_found"}

    warning = None
    payload: dict[str, dict | None] = {}
    for view, filename in VIDEO_NAMES.items():
        video_path = episode_dir / "videos" / filename
        if not video_path.exists():
            payload[view] = None
            warning = warning or "video_missing"
            continue
        meta, local_warning = _get_video_meta(video_path)
        payload[view] = meta
        warning = warning or local_warning

    if warning:
        payload["warning"] = warning
    return payload


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/tasks")
def api_tasks():
    return jsonify({"tasks": _cached_tasks()})


@app.route("/api/episodes/<task>")
def api_episodes(task: str):
    return jsonify({"episodes": _cached_episodes(task)})


@lru_cache(maxsize=256)
def _load_task_info_file(task: str) -> dict | None:
    """加载 task_info JSON 文件。"""
    task_info_path = DATA_ROOT / "task_info" / f"task_{task}.json"
    if not task_info_path.exists():
        return None
    try:
        with open(task_info_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None


@app.route("/api/task_info/<task>/<episode>")
def api_task_info(task: str, episode: str):
    """获取指定 task 和 episode 的详细信息。"""
    task_data = _load_task_info_file(task)
    if task_data is None:
        return jsonify({"error": "Task info not found"}), 404
    
    # task_data 是一个 list，每个元素对应一个 episode
    episode_id = int(episode)
    episode_info = None
    
    # 查找匹配的 episode
    for item in task_data:
        if item.get("episode_id") == episode_id:
            episode_info = item
            break
    
    if episode_info is None:
        return jsonify({"error": "Episode not found in task info"}), 404
    
    # 构建响应
    result = {
        "task_name": episode_info.get("task_name", ""),
        "init_scene_text": episode_info.get("init_scene_text", ""),
        "actions": []
    }
    
    # 提取 action_config
    label_info = episode_info.get("label_info", {})
    action_config = label_info.get("action_config", [])
    for action in action_config:
        result["actions"].append({
            "start_frame": action.get("start_frame", 0),
            "end_frame": action.get("end_frame", 0),
            "action_text": action.get("action_text", ""),
            "skill": action.get("skill", "")
        })
    
    return jsonify(result)


@app.route("/api/video/<task>/<episode>/<view>")
def api_video(task: str, episode: str, view: str):
    if view not in VIDEO_NAMES:
        abort(404)
    task_dir = BASE_DIR / task
    episode_dir = task_dir / episode
    if not _is_safe_dir(episode_dir) or not episode_dir.is_dir():
        abort(404)

    # 对于 head 视角，优先使用带 EEF 标注的视频
    if view == "head":
        eef_video_path = get_eef_video_path(
            data_root=DATA_ROOT,
            task_id=int(task),
            episode_id=int(episode),
            camera_name="head",
        )
        if eef_video_path and eef_video_path.exists():
            return send_file(eef_video_path, mimetype="video/mp4", conditional=True)
    
    # 回退到原始视频
    video_path = episode_dir / "videos" / VIDEO_NAMES[view]
    if not video_path.exists():
        abort(404)

    return send_file(video_path, mimetype="video/mp4", conditional=True)


@app.route("/api/meta/<task>/<episode>")
def api_meta(task: str, episode: str):
    return jsonify(_cached_meta(task, episode))


if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8765"))
    app.run(host=host, port=port, debug=False, threaded=True)
