from __future__ import annotations

import os
import sys
from pathlib import Path
from functools import lru_cache
import json
import subprocess
from flask import Flask, jsonify, render_template, send_file, abort, request

from generate_eef_video import get_eef_video_path, cancel_all_video_generation, check_eef_cache, start_eef_generation, is_generating, get_generation_error

# 添加项目根目录到路径，确保 agibot_utils / urdf_solver 都可导入
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from agibot_utils.video_marking import score_video

# 数据根目录
DATA_ROOT = Path("/mnt/raid0/AgiBot2Lerobot/AgiBot_Word_Beta")
BASE_DIR = DATA_ROOT / "observations"
FILTER_FILE = Path(__file__).parent / "dataFilter.jsonl"

VIDEO_NAMES = {
    "head": "head_color.mp4",
    "left": "hand_left_color.mp4",
    "right": "hand_right_color.mp4",
}

app = Flask(__name__, static_folder="static", template_folder="templates")


def _load_filter_records() -> list[dict]:
    if not FILTER_FILE.exists():
        return []
    records: list[dict] = []
    try:
        with FILTER_FILE.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    item = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(item, dict):
                    continue
                if not item.get("task") or not item.get("episode"):
                    continue
                if not isinstance(item.get("start_frame"), int) or not isinstance(item.get("end_frame"), int):
                    continue
                records.append({
                    "task": str(item["task"]),
                    "episode": str(item["episode"]),
                    "start_frame": item["start_frame"],
                    "end_frame": item["end_frame"],
                })
    except OSError:
        return []
    return records


def _write_filter_records(records: list[dict]) -> None:
    FILTER_FILE.parent.mkdir(parents=True, exist_ok=True)
    with FILTER_FILE.open("w", encoding="utf-8") as f:
        for item in records:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


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

    # 对于 head 视角，必须使用带 EEF 标注的视频
    if view == "head":
        # 只返回已缓存的视频，不阻塞生成
        eef_video_path = check_eef_cache(
            data_root=DATA_ROOT,
            task_id=int(task),
            episode_id=int(episode),
            camera_name="head",
        )
        if eef_video_path and eef_video_path.exists():
            return send_file(eef_video_path, mimetype="video/mp4", conditional=True)
        else:
            # 视频未就绪，返回 202 Accepted 表示正在处理
            return jsonify({"status": "generating", "message": "EEF video is being generated"}), 202
    
    # left/right 使用原始视频
    video_path = episode_dir / "videos" / VIDEO_NAMES[view]
    if not video_path.exists():
        abort(404)

    return send_file(video_path, mimetype="video/mp4", conditional=True)


@app.route("/api/prepare_video/<task>/<episode>")
def api_prepare_video(task: str, episode: str):
    """触发 head 视频的 EEF 生成（非阻塞）。
    
    前端应在选择 episode 时调用此 API 来触发视频生成。
    返回当前状态：ready（已就绪）、generating（生成中）、started（刚开始生成）、failed（生成失败）。
    """
    # 检查缓存
    cached = check_eef_cache(
        data_root=DATA_ROOT,
        task_id=int(task),
        episode_id=int(episode),
        camera_name="head",
    )
    if cached and cached.exists():
        return jsonify({"status": "ready", "path": str(cached)})
    
    key = f"{task}/{episode}/head"
    
    # 检查是否有失败记录
    error = get_generation_error(key)
    if error:
        return jsonify({"status": "failed", "message": error})
    
    # 检查是否正在生成
    if is_generating(key):
        return jsonify({"status": "generating"})
    
    # 开始后台生成
    started = start_eef_generation(
        data_root=DATA_ROOT,
        task_id=int(task),
        episode_id=int(episode),
        camera_name="head",
    )
    if started:
        return jsonify({"status": "started"})
    else:
        return jsonify({"status": "error", "message": "Failed to start generation"}), 500


@app.route("/api/meta/<task>/<episode>")
def api_meta(task: str, episode: str):
    return jsonify(_cached_meta(task, episode))


@app.route("/api/cancel_video_generation", methods=["POST"])
def api_cancel_video_generation():
    """取消所有正在进行的视频生成任务。
    
    前端应该在切换 episode 时调用此 API。
    """
    cancel_all_video_generation()
    return jsonify({"status": "ok"})


@app.route("/api/score/<task>/<episode>", methods=["POST"])
def api_score(task: str, episode: str):
    """调用 AI 对原始 head 视频进行评分。"""
    task_dir = BASE_DIR / task
    episode_dir = task_dir / episode
    if not _is_safe_dir(episode_dir) or not episode_dir.is_dir():
        return jsonify({"error": "Episode not found"}), 404
    
    # 使用原始视频（不带 EEF 标注）
    video_path = episode_dir / "videos" / VIDEO_NAMES["head"]
    if not video_path.exists():
        return jsonify({"error": "Video not found"}), 404
    
    try:
        score = score_video(str(video_path), verbose=True)
        if score is not None:
            return jsonify({"score": score})
        else:
            return jsonify({"error": "Failed to get score from AI"}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/filter/<task>/<episode>")
def api_get_filter(task: str, episode: str):
    records = _load_filter_records()
    for item in records:
        if item["task"] == str(task) and item["episode"] == str(episode):
            return jsonify({
                "range": {
                    "start_frame": item["start_frame"],
                    "end_frame": item["end_frame"],
                }
            })
    return jsonify({"range": None})


@app.route("/api/filter", methods=["POST"])
def api_save_filter():
    payload = request.get_json(silent=True) or {}
    task = payload.get("task")
    episode = payload.get("episode")
    start_frame = payload.get("start_frame")
    end_frame = payload.get("end_frame")
    total_frames = payload.get("total_frames")

    if not task or not episode:
        return jsonify({"error": "task and episode are required"}), 400
    if not isinstance(start_frame, int) or not isinstance(end_frame, int):
        return jsonify({"error": "start_frame and end_frame must be integers"}), 400
    if start_frame < 0 or end_frame < start_frame:
        return jsonify({"error": "invalid frame range"}), 400

    is_full_range = False
    if isinstance(total_frames, int) and total_frames > 0:
        is_full_range = start_frame == 0 and end_frame >= total_frames - 1

    records = _load_filter_records()
    filtered = [
        item for item in records
        if not (item["task"] == str(task) and item["episode"] == str(episode))
    ]
    if not is_full_range:
        filtered.append({
            "task": str(task),
            "episode": str(episode),
            "start_frame": start_frame,
            "end_frame": end_frame,
        })

    try:
        _write_filter_records(filtered)
    except OSError as e:
        return jsonify({"error": f"failed to write filter file: {e}"}), 500

    return jsonify({"status": "ok", "saved": not is_full_range})


if __name__ == "__main__":
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8765"))
    app.run(host=host, port=port, debug=False, threaded=True)
