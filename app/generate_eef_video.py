#!/usr/bin/env python3
"""生成带 EEF 标注的视频模块。

供 app/server.py 调用，将原始视频转换为带 EEF 可视化的视频。
"""

import json
import subprocess
from pathlib import Path
from typing import Optional, Tuple

import cv2
import h5py
import numpy as np


# 默认参数
DEFAULT_AXIS_LEN = 0.08  # 坐标轴长度 (米)
DEFAULT_GRIPPER_OFFSET = 0.143  # 夹爪偏移量 (米)


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
) -> bool:
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
    
    Returns:
        成功返回 True，失败返回 False
    """
    try:
        # 加载数据
        K, dist = load_intrinsic(intrinsic_path)
        R_all, t_all = load_extrinsics(extrinsic_path)
        pos_all, quat_all = load_eef(h5_path)
        
        # 获取视频信息
        probe = subprocess.run(
            f'ffprobe -v error -select_streams v:0 -show_entries stream=width,height,r_frame_rate -of json "{video_path}"',
            shell=True, capture_output=True, text=True
        )
        info = json.loads(probe.stdout)["streams"][0]
        w, h = int(info["width"]), int(info["height"])
        fps_parts = info["r_frame_rate"].split("/")
        fps = int(fps_parts[0]) / int(fps_parts[1]) if len(fps_parts) == 2 else float(fps_parts[0])
        
        num_frames = min(len(pos_all), len(R_all))
        
        if verbose:
            print(f"Processing {num_frames} frames @ {w}x{h}, {fps:.1f} fps")
        
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
        
        # 启动 ffmpeg 解码
        decode = subprocess.Popen(
            f'ffmpeg -hide_banner -loglevel error -i "{video_path}" -f rawvideo -pix_fmt bgr24 -',
            shell=True, stdout=subprocess.PIPE
        )
        
        # 编码器配置
        if use_nvenc:
            encoder_cmd = f'-c:v h264_nvenc -preset p1 -rc vbr -cq 23 -pix_fmt yuv420p'
        else:
            encoder_cmd = f'-c:v libx264 -preset fast -crf 23 -pix_fmt yuv420p'
        
        encode = subprocess.Popen(
            f'ffmpeg -y -hide_banner -loglevel error -f rawvideo -pix_fmt bgr24 -s {w}x{h} -r {fps} -i - {encoder_cmd} "{output_path}"',
            shell=True, stdin=subprocess.PIPE, stderr=subprocess.PIPE
        )
        
        frame_size = w * h * 3
        colors = [(0, 0, 255), (255, 0, 0)]  # L=红, R=蓝
        labels = ["L", "R"]
        axis_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
        
        for i in range(num_frames):
            raw = decode.stdout.read(frame_size)
            if len(raw) != frame_size:
                break
            
            frame = np.frombuffer(raw, np.uint8).reshape(h, w, 3).copy()
            pts = pts_2d_all[i]
            
            for j in range(2):
                x, y = pts[j]
                if 0 <= x < w and 0 <= y < h:
                    cv2.circle(frame, (x, y), 8, colors[j], -1)
                    cv2.putText(frame, labels[j], (x+10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[j], 2)
                    
                    if axes_2d_all:
                        for k in range(3):
                            ax, ay = axes_2d_all[i][j][k]
                            if 0 <= ax < w and 0 <= ay < h:
                                cv2.line(frame, (x, y), (ax, ay), axis_colors[k], 2)
            
            info_text = f"Frame {i} | L:({pts[0,0]},{pts[0,1]}) R:({pts[1,0]},{pts[1,1]})"
            cv2.putText(frame, info_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            encode.stdin.write(frame.tobytes())
        
        encode.stdin.close()
        encode.wait()
        decode.wait()
        
        if verbose:
            print(f"Done! Output: {output_path}")
        
        return output_path.exists()
        
    except Exception as e:
        if verbose:
            print(f"Error: {e}")
        return False


def get_eef_video_path(
    data_root: Path,
    task_id: int,
    episode_id: int,
    camera_name: str = "head",
    cache_dir: Optional[Path] = None,
) -> Optional[Path]:
    """获取带 EEF 标注的视频路径，如果不存在则生成。
    
    Args:
        data_root: 数据根目录
        task_id: 任务 ID
        episode_id: Episode ID
        camera_name: 相机名称
        cache_dir: 缓存目录，默认使用 data_root 下的 .eef_cache
    
    Returns:
        EEF 视频路径，失败返回 None
    """
    # 构建路径
    base = data_root / f"observations/{task_id}/{episode_id}"
    video_path = base / f"videos/{camera_name}_color.mp4"
    h5_path = data_root / f"proprio_stats/{task_id}/{episode_id}/proprio_stats.h5"
    intrinsic_path = data_root / f"parameters/{task_id}/{episode_id}/parameters/camera/{camera_name}_intrinsic_params.json"
    extrinsic_path = data_root / f"parameters/{task_id}/{episode_id}/parameters/camera/{camera_name}_extrinsic_params_aligned.json"
    
    # 检查必要文件
    for p in [video_path, h5_path, intrinsic_path, extrinsic_path]:
        if not p.exists():
            return None
    
    # 缓存目录
    if cache_dir is None:
        cache_dir = data_root / ".eef_cache"
    
    output_path = cache_dir / f"{task_id}/{episode_id}/{camera_name}_eef.mp4"
    
    # 如果缓存存在且比源文件新，直接返回
    if output_path.exists():
        src_mtime = max(p.stat().st_mtime for p in [video_path, h5_path, intrinsic_path, extrinsic_path])
        if output_path.stat().st_mtime >= src_mtime:
            return output_path
    
    # 生成视频
    success = generate_eef_video(
        video_path=video_path,
        h5_path=h5_path,
        intrinsic_path=intrinsic_path,
        extrinsic_path=extrinsic_path,
        output_path=output_path,
        verbose=True,
    )
    
    return output_path if success else None
