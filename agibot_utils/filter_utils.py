import json
import subprocess
import tempfile
from pathlib import Path
from functools import lru_cache


def normalize_task_episode_key(task_id, episode_id) -> tuple[str, str]:
    """将 task_id 和 episode_id 统一规整成 JSONL 检索使用的键格式。"""
    return str(task_id).replace("task_", ""), str(episode_id)


def load_filter_jsonl(jsonl_path: Path) -> dict[tuple[str, str], dict]:
    """读取单个 JSONL 过滤文件，并构建 {(task_id, episode_id): entry} 查询表。"""
    lookup = {}
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            key = normalize_task_episode_key(entry["task"], entry["episode"])
            lookup[key] = entry
    return lookup


def get_episode_filter_match(
    task_id,
    episode_id,
    inconsistent_lookup: dict[tuple[str, str], dict],
    camera_lookup: dict[tuple[str, str], dict],
    still_lookup: dict[tuple[str, str], dict],
) -> tuple[str | None, dict | None]:
    """按既定顺序返回当前 episode 命中的第一条过滤结果：第 1 道 -> 第 2 道 -> 第 3 道。"""
    key = normalize_task_episode_key(task_id, episode_id)

    if key in inconsistent_lookup:
        return "inconsistent_frames_length", inconsistent_lookup[key]
    if key in camera_lookup:
        return "camera_extrinsic", camera_lookup[key]
    if key in still_lookup:
        return "still_frames", still_lookup[key]
    return None, None


def slice_episode_frames(frames: list[dict], start_frame: int, end_frame: int) -> list[dict]:
    """从单个 episode 的 frames 中截取包含两端的 [start_frame, end_frame] 区间。"""
    if not frames:
        raise ValueError("Cannot slice empty episode frames.")
    if start_frame < 0:
        raise ValueError(f"Invalid start_frame={start_frame}. Must be >= 0.")
    if end_frame < start_frame:
        raise ValueError(
            f"Invalid frame range: start_frame={start_frame}, end_frame={end_frame}. end_frame must be >= start_frame."
        )
    if end_frame >= len(frames):
        raise ValueError(
            f"Invalid end_frame={end_frame} for episode length {len(frames)}. end_frame must be < {len(frames)}."
        )
    return frames[start_frame : end_frame + 1]


def format_episode_status_lines(
    idx: int,
    total_episodes: int,
    task_name: str,
    episode_id,
    outcome: str,
    detail: str | None = None,
) -> list[str]:
    """生成终端中每个 episode 顶层状态输出的几行文本。"""
    lines = [f"[{idx}/{total_episodes}] {task_name} episode {episode_id}", f"  - status: {outcome}"]
    if detail:
        lines.append(f"  - detail: {detail}")
    return lines


def format_crop_progress_line(video_idx: int, total_videos: int, video_name: str) -> str:
    """生成第 3 道 still_frames 裁剪时的单条缩进进度文本。"""
    return f"    - crop video {video_idx}/{total_videos}: {video_name}"


def format_episode_separator() -> str:
    """返回每个 episode 输出块结束后使用的分隔线。"""
    return "=" * 100


def get_video_codec_name(video_path: Path) -> str:
    """使用 ffprobe 读取源视频的 codec，供后续裁剪时选择兼容的编码器。"""
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(video_path),
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    codec_name = result.stdout.strip()
    if not codec_name:
        raise ValueError(f"Failed to detect codec for video: {video_path}")
    return codec_name


@lru_cache(maxsize=1)
def get_available_ffmpeg_encoders() -> set[str]:
    """查询当前 ffmpeg 可用的视频编码器，并将结果缓存起来避免重复调用。"""
    result = subprocess.run(
        ["ffmpeg", "-hide_banner", "-encoders"],
        check=True,
        capture_output=True,
        text=True,
    )
    encoders = set()
    for line in result.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0].startswith("V"):
            encoders.add(parts[1])
    return encoders


def select_ffmpeg_encoder_args(codec_name: str, available_encoders: set[str] | None = None) -> list[str]:
    """根据输入视频的 codec，选择裁剪输出时使用的 ffmpeg 编码参数。"""
    available_encoders = available_encoders or get_available_ffmpeg_encoders()

    if codec_name == "av1":
        if "libsvtav1" in available_encoders:
            return ["-c:v", "libsvtav1", "-pix_fmt", "yuv420p", "-g", "2", "-crf", "30"]
        if "libaom-av1" in available_encoders:
            return ["-c:v", "libaom-av1", "-pix_fmt", "yuv420p", "-cpu-used", "8", "-crf", "30", "-b:v", "0"]
        raise RuntimeError(
            "Input video codec is AV1, but no supported AV1 encoder is available in ffmpeg. "
            "Expected one of: libsvtav1, libaom-av1."
        )
    if codec_name == "h264":
        return ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-g", "2", "-crf", "18"]
    if codec_name == "hevc":
        return ["-c:v", "libx265", "-pix_fmt", "yuv420p", "-g", "2", "-crf", "20"]
    raise ValueError(f"Unsupported input video codec for cropping: {codec_name}")


def get_ffmpeg_encoder_args(video_path: Path) -> list[str]:
    """先读取源视频 codec，再映射成对应的 ffmpeg 编码参数。"""
    codec_name = get_video_codec_name(video_path)
    return select_ffmpeg_encoder_args(codec_name)


def crop_video_to_frame_range(video_path: Path, output_path: Path, start_frame: int, end_frame: int, fps: int) -> Path:
    """按包含两端的帧区间裁剪单个视频，并生成当前 episode 的临时输出视频。"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    video_filter = f"select='between(n\\,{start_frame}\\,{end_frame})',setpts=N/FRAME_RATE/TB"
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(video_path),
        "-vf",
        video_filter,
        "-an",
        "-r",
        str(fps),
        *get_ffmpeg_encoder_args(video_path),
        str(output_path),
    ]
    subprocess.run(cmd, check=True)
    return output_path


def crop_episode_videos(
    videos: dict[str, Path],
    start_frame: int,
    end_frame: int,
    fps: int,
    progress_callback=None,
) -> tuple[dict[str, Path], Path]:
    """裁剪单个 episode 下的全部视频到临时目录，并返回裁剪后路径与临时目录路径。"""
    temp_dir = Path(tempfile.mkdtemp(prefix="agibot_episode_crop_"))
    cropped_videos = {}
    total_videos = len(videos)
    for video_idx, (video_key, video_path) in enumerate(videos.items(), 1):
        if progress_callback is not None:
            progress_callback(video_idx, total_videos, video_path)
        cropped_path = temp_dir / Path(video_path).name
        cropped_videos[video_key] = crop_video_to_frame_range(video_path, cropped_path, start_frame, end_frame, fps)
    return cropped_videos, temp_dir
