from __future__ import annotations

import os
from pathlib import Path
from functools import lru_cache
import json
import subprocess
from flask import Flask, jsonify, render_template, send_file, abort

BASE_DIR = Path("/mnt/raid0/AgiBot2Lerobot/AgiBot_Word_Beta/observations")
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


@app.route("/api/video/<task>/<episode>/<view>")
def api_video(task: str, episode: str, view: str):
    if view not in VIDEO_NAMES:
        abort(404)
    task_dir = BASE_DIR / task
    episode_dir = task_dir / episode
    if not _is_safe_dir(episode_dir) or not episode_dir.is_dir():
        abort(404)

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
