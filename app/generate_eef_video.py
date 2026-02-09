#!/usr/bin/env python3
"""生成带 EEF 标注的视频模块。

供 app/server.py 调用，将原始视频转换为带 EEF 可视化的视频。
"""

import json
import subprocess
import threading
from pathlib import Path
from typing import Optional, Tuple

import cv2
import h5py
import numpy as np
import time


# 默认参数
DEFAULT_AXIS_LEN = 0.08  # 坐标轴长度 (米)
DEFAULT_GRIPPER_OFFSET = 0.143  # 夹爪偏移量 (米)


def _build_paths(data_root: Path, task_id: int, episode_id: int, camera_name: str) -> dict:
    """构建所有需要的文件路径。"""
    base = data_root / f"observations/{task_id}/{episode_id}"
    return {
        "video": base / f"videos/{camera_name}_color.mp4",
        "h5": data_root / f"proprio_stats/{task_id}/{episode_id}/proprio_stats.h5",
        "intrinsic": data_root / f"parameters/{task_id}/{episode_id}/parameters/camera/{camera_name}_intrinsic_params.json",
        "extrinsic": data_root / f"parameters/{task_id}/{episode_id}/parameters/camera/{camera_name}_extrinsic_params_aligned.json",
        "cache_dir": data_root / ".eef_cache",
        "output": data_root / f".eef_cache/{task_id}/{episode_id}/{camera_name}_eef.mp4",
    }


def load_intrinsic(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """加载相机内参和畸变系数。"""
    data = json.loads(path.read_text())["intrinsic"]
    K = np.array([
        [data["fx"], 0, data.get("ppx", data.get("cx"))],
        [0, data["fy"], data.get("ppy", data.get("cy"))],
        [0, 0, 1]
    ], dtype=np.float64)
    dist = np.array([
        data.get("k1", 0), data.get("k2", 0),
        data.get("p1", 0), data.get("p2", 0), data.get("k3", 0)
    ], dtype=np.float64)
    return K, dist


def load_extrinsics(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    """加载所有帧的外参 (R, t)。"""
    data = json.loads(path.read_text())
    if not isinstance(data, list):
        data = [data]
    R_all, t_all = [], []
    for entry in data:
        extr = entry.get("extrinsic", entry)
        R_all.append(extr["rotation_matrix"])
        t_all.append(extr["translation_vector"])
    return np.array(R_all), np.array(t_all)


def load_eef(path: Path) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """加载所有帧的 EEF 位置和姿态。"""
    with h5py.File(path, "r") as f:
        pos = np.array(f["action/end/position"], dtype=np.float64)
        quat = np.array(f["action/end/orientation"], dtype=np.float64) if "action/end/orientation" in f else None
    return pos, quat


def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """四元数 (xyzw) 转旋转矩阵。"""
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]
    ])


def apply_gripper_offset(pos: np.ndarray, quat: Optional[np.ndarray], offset: float) -> np.ndarray:
    """将手腕位置转换为夹爪中心位置。"""
    if quat is None or offset == 0:
        return pos
    
    gripper_pos = pos.copy()
    for i in range(2):  # L, R
        R_eef = quat_to_rotmat(quat[i])
        z_axis = R_eef[:, 2]
        gripper_pos[i] = pos[i] + offset * z_axis
    return gripper_pos


def generate_eef_video(
    video_path: Path,
    h5_path: Path,
    intrinsic_path: Path,
    extrinsic_path: Path,
    output_path: Path,
    axis_len: float = DEFAULT_AXIS_LEN,
    gripper_offset: float = DEFAULT_GRIPPER_OFFSET,
    use_nvenc: bool = True,
    verbose: bool = False,
    process_key: Optional[str] = None,
) -> Tuple[bool, Optional[str]]:
    """生成带 EEF 标注的视频。
    
    Args:
        video_path: 输入视频路径
        h5_path: EEF 数据 h5 文件路径
        intrinsic_path: 相机内参文件路径
        extrinsic_path: 相机外参文件路径
        output_path: 输出视频路径
        axis_len: 坐标轴长度 (米)
        gripper_offset: 夹爪偏移量 (米)
        use_nvenc: 是否使用 NVENC 硬件编码
        verbose: 是否打印进度信息
        process_key: 进程标识符，用于注册到全局字典以便取消
    
    Returns:
        (success, error_message): 成功返回 (True, None)，失败返回 (False, 错误信息)
    """
    decode = None
    encode = None
    try:
        # 加载数据
        K, dist = load_intrinsic(intrinsic_path)
        R_all, t_all = load_extrinsics(extrinsic_path)
        pos_all, quat_all = load_eef(h5_path)
        
        # 获取视频信息
        probe = subprocess.run(
            f'ffprobe -v error -select_streams v:0 -show_entries stream=width,height,r_frame_rate,nb_frames -of json "{video_path}"',
            shell=True, capture_output=True, text=True
        )
        info = json.loads(probe.stdout)["streams"][0]
        w, h = int(info["width"]), int(info["height"])
        fps_parts = info["r_frame_rate"].split("/")
        fps = int(fps_parts[0]) / int(fps_parts[1]) if len(fps_parts) == 2 else float(fps_parts[0])
        
        # 获取视频帧数
        video_frames = int(info.get("nb_frames", 0))
        if video_frames == 0:
            # 如果 nb_frames 不可用，用 duration 估算
            duration_probe = subprocess.run(
                f'ffprobe -v error -select_streams v:0 -show_entries stream=duration -of json "{video_path}"',
                shell=True, capture_output=True, text=True
            )
            duration_info = json.loads(duration_probe.stdout)["streams"][0]
            duration = float(duration_info.get("duration", 0))
            video_frames = int(duration * fps)
        
        eef_frames = len(pos_all)
        extrinsic_frames = len(R_all)
        
        # ====== 数据质量检查 ======
        if verbose:
            print(f"数据质量检查: 视频={video_frames}帧, EEF={eef_frames}帧, 外参={extrinsic_frames}帧")
        
        if not (video_frames == eef_frames == extrinsic_frames):
            error_msg = f"数据不一致: 视频{video_frames}帧 / EEF{eef_frames}帧 / 外参{extrinsic_frames}帧"
            print(f"❌ {error_msg}")
            return False, error_msg
        
        num_frames = eef_frames
        
        if verbose:
            print(f"✅ 数据一致，处理 {num_frames} 帧 @ {w}x{h}, {fps:.1f} fps")
        
        # 批量预计算所有帧的 2D 投影点
        pts_2d_all = []
        axes_2d_all = []
        
        for i in range(num_frames):
            R, t = R_all[i], t_all[i]
            R_inv, t_inv = R.T, -R.T @ t
            rvec, _ = cv2.Rodrigues(R_inv)
            
            pos_frame = pos_all[i]
            quat_frame = quat_all[i] if quat_all is not None else None
            gripper_pos = apply_gripper_offset(pos_frame, quat_frame, gripper_offset)
            
            pts_2d, _ = cv2.projectPoints(gripper_pos.reshape(-1, 1, 3), rvec, t_inv.reshape(3, 1), K, dist)
            pts_2d_all.append(pts_2d.reshape(-1, 2).astype(np.int32))
            
            if quat_all is not None:
                axes_frame = []
                for j in range(2):
                    R_eef = quat_to_rotmat(quat_all[i, j])
                    axes_3d = gripper_pos[j] + (R_eef @ (np.eye(3) * axis_len)).T
                    axes_2d, _ = cv2.projectPoints(axes_3d.reshape(-1, 1, 3), rvec, t_inv.reshape(3, 1), K, dist)
                    axes_frame.append(axes_2d.reshape(-1, 2).astype(np.int32))
                axes_2d_all.append(axes_frame)
        
        # 确保输出目录存在
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 清理可能存在的旧临时文件
        temp_output_path = output_path.with_suffix(".tmp.mp4")
        if temp_output_path.exists():
            try:
                temp_output_path.unlink()
            except Exception:
                pass
        
        # 启动 ffmpeg 解码，使用硬件加速
        decode = subprocess.Popen(
            f'ffmpeg -hwaccel auto -hide_banner -loglevel error -i "{video_path}" -f rawvideo -pix_fmt bgr24 -',
            shell=True, stdout=subprocess.PIPE
        )
        
        # 编码器配置
        # -g 30: 关键帧间隔30帧，约1秒，提高seek性能
        if use_nvenc:
            encoder_cmd = f'-c:v h264_nvenc -preset p1 -rc vbr -cq 23 -g 30 -pix_fmt yuv420p'
        else:
            encoder_cmd = f'-c:v libx264 -preset fast -crf 23 -g 30 -pix_fmt yuv420p'
        
        # 使用临时文件名进行生成，防止被误判为已完成
        
        encode = subprocess.Popen(
            f'ffmpeg -y -hide_banner -loglevel error -f rawvideo -pix_fmt bgr24 -s {w}x{h} -r {fps} -i - {encoder_cmd} "{temp_output_path}"',
            shell=True, stdin=subprocess.PIPE, stderr=subprocess.PIPE
        )
        
        # 注册进程以便取消
        if process_key:
            with _active_lock:
                _active_processes[process_key] = (decode, encode)
        
        frame_size = w * h * 3
        colors = [(0, 0, 255), (255, 0, 0)]  # L=红, R=蓝
        labels = ["L", "R"]
        axis_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
        
        # 预分配帧缓冲区
        frame_buffer = np.zeros((h, w, 3), dtype=np.uint8)
        
        if verbose:
            print(f"Starting frame loop for {num_frames} frames...")
        
        loop_start = time.time()
        cancelled = False
        for i in range(num_frames):
            # 每50帧检查一次是否被取消，减少锁竞争
            if process_key and i % 50 == 0:
                with _active_lock:
                    if process_key not in _active_processes:
                        cancelled = True
                        break
            
            raw = decode.stdout.read(frame_size)
            if len(raw) != frame_size:
                break
            
            # 使用 copyto 避免频繁分配内存
            np.copyto(frame_buffer, np.frombuffer(raw, np.uint8).reshape(h, w, 3))
            
            pts = pts_2d_all[i]
            for j in range(2):
                x, y = pts[j]
                if 0 <= x < w and 0 <= y < h:
                    cv2.circle(frame_buffer, (x, y), 8, colors[j], -1)
                    # 只有 L/R 的标注，取消文字以提速
                    # cv2.putText(frame_buffer, labels[j], (x+10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[j], 2)
                    
                    if axes_2d_all:
                        for k in range(3):
                            ax, ay = axes_2d_all[i][j][k]
                            if 0 <= ax < w and 0 <= ay < h:
                                cv2.line(frame_buffer, (x, y), (ax, ay), axis_colors[k], 2)
            
            try:
                encode.stdin.write(frame_buffer.tobytes())
            except BrokenPipeError:
                if verbose:
                    _, err = encode.communicate()
                    print(f"FFmpeg encoding error: {err.decode() if err else 'unknown'}")
                cancelled = True
                break
                
            if verbose and (i + 1) % 500 == 0:
                elapsed = time.time() - loop_start
                fps_curr = (i + 1) / elapsed
                print(f"Processed {i+1}/{num_frames} frames... ({fps_curr:.1f} fps)")
        
        loop_end = time.time()
        if verbose and not cancelled:
            print(f"Loop finished in {loop_end - loop_start:.2f}s (Avg {num_frames/(loop_end - loop_start):.1f} fps)")
        
        if cancelled:
            if verbose:
                print(f"Video generation cancelled: {process_key}")
            return False, "Video generation cancelled"
        
        # 正常完成，关闭编码器
        try:
            encode.stdin.close()
            encode.wait(timeout=5)
        except Exception as e:
            if verbose:
                print(f"Warning: encoder cleanup issue: {e}")
        
        try:
            decode.wait(timeout=2)
        except Exception:
            pass
        
        # 成功完成后，将临时文件重命名为正式文件
        if temp_output_path.exists():
            temp_output_path.rename(output_path)
            
        if verbose:
            print(f"Done! Output: {output_path}")
        
        return output_path.exists(), None
        
    except Exception as e:
        if verbose:
            print(f"Error: {e}")
        return False, str(e)
    finally:
        # 从跟踪字典中移除
        if process_key:
            with _active_lock:
                _active_processes.pop(process_key, None)
        
        # 确保进程被清理
        if decode and decode.poll() is None:
            try:
                decode.terminate()
                decode.wait(timeout=2)
            except Exception:
                try:
                    decode.kill()
                except Exception:
                    pass
        
        if encode and encode.poll() is None:
            try:
                encode.stdin.close()
            except Exception:
                pass
            try:
                encode.terminate()
                encode.wait(timeout=2)
            except Exception:
                try:
                    encode.kill()
                except Exception:
                    pass
        
        # 清理临时文件（如果生成失败或被取消）
        temp_output_path = output_path.with_suffix(".tmp.mp4")
        if temp_output_path.exists() and not output_path.exists():
            try:
                temp_output_path.unlink()
            except Exception:
                pass




# 全局字典，跟踪正在生成中的视频任务的 ffmpeg 进程
_active_processes: dict[str, tuple[subprocess.Popen, subprocess.Popen]] = {}
_active_lock = threading.Lock()
_MAX_CONCURRENT_GENERATIONS = 2  # 最多同时生成2个视频，防止资源耗尽

# 跟踪失败的生成任务，储存错误信息
_failed_generations: dict[str, str] = {}


def cancel_video_generation(task_id: int, episode_id: int, camera_name: str = "head") -> bool:
    """取消正在进行的视频生成任务。
    
    Returns:
        如果成功取消返回 True，否则返回 False
    """
    key = f"{task_id}/{episode_id}/{camera_name}"
    with _active_lock:
        if key in _active_processes:
            decode_proc, encode_proc = _active_processes[key]
            try:
                decode_proc.terminate()
                encode_proc.terminate()
            except Exception:
                pass
            _active_processes.pop(key, None)
            print(f"Cancelled video generation: {key}")
            return True
    return False


def cancel_all_video_generation():
    """取消所有正在进行的视频生成任务。"""
    with _active_lock:
        for key, (decode_proc, encode_proc) in list(_active_processes.items()):
            try:
                decode_proc.terminate()
                encode_proc.terminate()
            except Exception:
                pass
            print(f"Cancelled video generation: {key}")
        _active_processes.clear()


def check_eef_cache(
    data_root: Path,
    task_id: int,
    episode_id: int,
    camera_name: str = "head",
    cache_dir: Optional[Path] = None,
) -> Optional[Path]:
    """检查 EEF 视频缓存是否存在且有效。
    
    此函数不会触发视频生成，只检查缓存。
    
    Returns:
        如果缓存存在且有效，返回缓存路径；否则返回 None
    """
    paths = _build_paths(data_root, task_id, episode_id, camera_name)
    
    # 检查必要文件
    src_files = [paths["video"], paths["h5"], paths["intrinsic"], paths["extrinsic"]]
    for p in src_files:
        if not p.exists():
            return None
    
    output_path = paths["output"] if cache_dir is None else cache_dir / f"{task_id}/{episode_id}/{camera_name}_eef.mp4"
    
    # 如果缓存存在且比源文件新，返回缓存路径
    if output_path.exists():
        src_mtime = max(p.stat().st_mtime for p in src_files)
        if output_path.stat().st_mtime >= src_mtime:
            return output_path
    
    return None


def is_generating(process_key: str) -> bool:
    """检查指定的视频是否正在生成中。"""
    with _active_lock:
        return process_key in _active_processes


def get_generation_error(process_key: str) -> Optional[str]:
    """获取生成失败的错误信息。返回 None 表示没有失败记录。"""
    with _active_lock:
        return _failed_generations.get(process_key)


def clear_generation_error(process_key: str):
    """清除生成失败的错误信息。"""
    with _active_lock:
        _failed_generations.pop(process_key, None)


def _background_generate(
    data_root: Path,
    task_id: int,
    episode_id: int,
    camera_name: str,
):
    """后台线程执行视频生成。"""
    paths = _build_paths(data_root, task_id, episode_id, camera_name)
    process_key = f"{task_id}/{episode_id}/{camera_name}"
    
    # 清除之前的失败记录
    clear_generation_error(process_key)
    
    success, error_msg = generate_eef_video(
        video_path=paths["video"],
        h5_path=paths["h5"],
        intrinsic_path=paths["intrinsic"],
        extrinsic_path=paths["extrinsic"],
        output_path=paths["output"],
        verbose=True,
        process_key=process_key,
    )
    
    # 如果生成失败，记录错误信息
    if not success and not paths["output"].exists():
        with _active_lock:
            _failed_generations[process_key] = error_msg or "未知错误"


def start_eef_generation(
    data_root: Path,
    task_id: int,
    episode_id: int,
    camera_name: str = "head",
) -> bool:
    """启动后台 EEF 视频生成。
    
    此函数立即返回，视频在后台线程中生成。
    
    Returns:
        如果成功启动返回 True，否则返回 False
    """
    paths = _build_paths(data_root, task_id, episode_id, camera_name)
    
    # 检查必要文件
    for p in [paths["video"], paths["h5"], paths["intrinsic"], paths["extrinsic"]]:
        if not p.exists():
            return False
    
    # 检查是否已在生成中
    process_key = f"{task_id}/{episode_id}/{camera_name}"
    with _active_lock:
        if process_key in _active_processes:
            return True  # 已在生成中
        
        # 如果并发数达到上限，取消最旧的任务
        if len(_active_processes) >= _MAX_CONCURRENT_GENERATIONS:
            oldest_key = next(iter(_active_processes))
            decode_proc, encode_proc = _active_processes[oldest_key]
            try:
                decode_proc.terminate()
                encode_proc.terminate()
            except Exception:
                pass
            _active_processes.pop(oldest_key, None)
            print(f"Cancelled oldest generation {oldest_key} to make room for {process_key}")
    
    # 启动后台线程
    thread = threading.Thread(
        target=_background_generate,
        args=(data_root, task_id, episode_id, camera_name),
        daemon=True,
    )
    thread.start()
    return True


def get_eef_video_path(
    data_root: Path,
    task_id: int,
    episode_id: int,
    camera_name: str = "head",
    cache_dir: Optional[Path] = None,
) -> Optional[Path]:
    """获取带 EEF 标注的视频路径，如果不存在则生成。
    
    此函数会阻塞直到视频生成完成。可以通过 cancel_video_generation() 取消。
    
    Args:
        data_root: 数据根目录
        task_id: 任务 ID
        episode_id: Episode ID
        camera_name: 相机名称
        cache_dir: 缓存目录，默认使用 data_root 下的 .eef_cache
    
    Returns:
        EEF 视频路径，失败返回 None
    """
    paths = _build_paths(data_root, task_id, episode_id, camera_name)
    src_files = [paths["video"], paths["h5"], paths["intrinsic"], paths["extrinsic"]]
    
    # 检查必要文件
    for p in src_files:
        if not p.exists():
            return None
    
    output_path = paths["output"] if cache_dir is None else cache_dir / f"{task_id}/{episode_id}/{camera_name}_eef.mp4"
    
    # 如果缓存存在且比源文件新，直接返回
    if output_path.exists():
        src_mtime = max(p.stat().st_mtime for p in src_files)
        if output_path.stat().st_mtime >= src_mtime:
            return output_path
    
    # 生成视频（阻塞），使用 process_key 以便可以取消
    process_key = f"{task_id}/{episode_id}/{camera_name}"
    success, _ = generate_eef_video(
        video_path=paths["video"],
        h5_path=paths["h5"],
        intrinsic_path=paths["intrinsic"],
        extrinsic_path=paths["extrinsic"],
        output_path=output_path,
        verbose=True,
        process_key=process_key,
    )
    
    return output_path if success else None
