#!/usr/bin/env python3
"""可视化 EEF 末端执行器位置到视频帧上。

使用方法：直接修改下方配置参数，然后运行脚本即可。
"""

import json
import subprocess
from pathlib import Path

import h5py
import numpy as np
import cv2



def get_paths(data_root: Path, task_id: int, episode_id: int, camera_name: str):
    """根据 task_id 和 episode_id 构建所有数据路径。"""
    base = data_root / f"observations/{task_id}/{episode_id}"
    return {
        "video": base / f"videos/{camera_name}_color.mp4",
        "h5": data_root / f"proprio_stats/{task_id}/{episode_id}/proprio_stats.h5",
        "intrinsic": data_root / f"parameters/{task_id}/{episode_id}/parameters/camera/{camera_name}_intrinsic_params.json",
        "extrinsic": data_root / f"parameters/{task_id}/{episode_id}/parameters/camera/{camera_name}_extrinsic_params_aligned.json",
    }


def load_intrinsic(path: Path):
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


def load_extrinsics(path: Path):
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


def load_eef(path: Path):
    """加载所有帧的 EEF 位置和姿态。"""
    with h5py.File(path, "r") as f:
        pos = np.array(f["action/end/position"], dtype=np.float64)
        quat = np.array(f["action/end/orientation"], dtype=np.float64) if "action/end/orientation" in f else None
    return pos, quat


def quat_to_rotmat(q):
    """四元数 (xyzw) 转旋转矩阵。"""
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]
    ])


def apply_gripper_offset(pos, quat, offset):
    """将手腕位置转换为夹爪中心位置。
    
    Args:
        pos: (2, 3) 左右手腕的 XYZ 位置
        quat: (2, 4) 左右手腕的四元数 (xyzw)
        offset: 沿 EEF 局部 Z 轴的偏移量 (米)
    
    Returns:
        gripper_pos: (2, 3) 左右夹爪中心的 XYZ 位置
    """
    if quat is None or offset == 0:
        return pos
    
    gripper_pos = pos.copy()
    for i in range(2):  # L, R
        R_eef = quat_to_rotmat(quat[i])
        # 沿 EEF 局部 Z 轴偏移
        z_axis = R_eef[:, 2]  # EEF 的 Z 轴方向
        gripper_pos[i] = pos[i] + offset * z_axis
    return gripper_pos


def project(pts, K, dist, R, t):
    """将 3D 点投影到图像平面。"""
    R_inv, t_inv = R.T, -R.T @ t
    rvec, _ = cv2.Rodrigues(R_inv)
    pts_2d, _ = cv2.projectPoints(pts.reshape(-1, 1, 3), rvec, t_inv.reshape(3, 1), K, dist)
    return pts_2d.reshape(-1, 2)


def draw_eef(frame, pos_lr, quat_lr, K, dist, R, t, frame_idx):
    """在帧上绘制 EEF 位置和坐标轴。"""
    h, w = frame.shape[:2]
    pts = project(pos_lr, K, dist, R, t)
    colors = [(0, 0, 255), (255, 0, 0)]  # L=红, R=蓝
    labels = ["L", "R"]
    
    for i in range(2):
        x, y = int(pts[i, 0]), int(pts[i, 1])
        if 0 <= x < w and 0 <= y < h:
            cv2.circle(frame, (x, y), 8, colors[i], -1)
            cv2.putText(frame, labels[i], (x+10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[i], 2)
        
        # 绘制坐标轴
        if quat_lr is not None:
            R_eef = quat_to_rotmat(quat_lr[i])
            axes_3d = pos_lr[i] + (R_eef @ (np.eye(3) * AXIS_LEN)).T
            axes_2d = project(axes_3d, K, dist, R, t)
            axis_colors = [(0, 0, 255), (0, 255, 0), (255, 0, 0)]
            for j in range(3):
                ax, ay = int(axes_2d[j, 0]), int(axes_2d[j, 1])
                if 0 <= ax < w and 0 <= ay < h:
                    cv2.line(frame, (x, y), (ax, ay), axis_colors[j], 2)
    
    # 添加帧信息
    info = f"Frame {frame_idx} | L:({pts[0,0]:.0f},{pts[0,1]:.0f}) R:({pts[1,0]:.0f},{pts[1,1]:.0f})"
    cv2.putText(frame, info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    return frame


def process_single_frame(paths, frame_idx, output_path):
    """处理单帧。"""
    K, dist = load_intrinsic(paths["intrinsic"])
    R_all, t_all = load_extrinsics(paths["extrinsic"])
    pos_all, quat_all = load_eef(paths["h5"])
    
    # 使用 ffmpeg 提取帧
    cmd = f'ffmpeg -y -hide_banner -loglevel error -i "{paths["video"]}" -vf "select=eq(n\\,{frame_idx})" -frames:v 1 -f image2pipe -vcodec png -'
    result = subprocess.run(cmd, shell=True, capture_output=True)
    frame = cv2.imdecode(np.frombuffer(result.stdout, np.uint8), cv2.IMREAD_COLOR)
    
    R, t = R_all[min(frame_idx, len(R_all)-1)], t_all[min(frame_idx, len(t_all)-1)]
    pos_lr, quat_lr = pos_all[frame_idx], quat_all[frame_idx] if quat_all is not None else None
    
    frame = draw_eef(frame, pos_lr, quat_lr, K, dist, R, t, frame_idx)
    cv2.imwrite(str(output_path), frame)
    print(f"Saved: {output_path}")


def process_video(paths, output_path):
    """处理整个视频（优化版：硬件编码 + 批量预计算）。"""
    import time
    start_time = time.time()
    
    K, dist = load_intrinsic(paths["intrinsic"])
    R_all, t_all = load_extrinsics(paths["extrinsic"])
    pos_all, quat_all = load_eef(paths["h5"])
    
    # 获取视频信息
    probe = subprocess.run(
        f'ffprobe -v error -select_streams v:0 -show_entries stream=width,height,r_frame_rate -of json "{paths["video"]}"',
        shell=True, capture_output=True, text=True
    )
    info = json.loads(probe.stdout)["streams"][0]
    w, h = int(info["width"]), int(info["height"])
    fps_parts = info["r_frame_rate"].split("/")
    fps = int(fps_parts[0]) / int(fps_parts[1]) if len(fps_parts) == 2 else float(fps_parts[0])
    
    num_frames = min(len(pos_all), len(R_all))
    
    # ====== 批量预计算所有帧的 2D 投影点 ======
    precompute_start = time.time()
    pts_2d_all = []  # 每帧 [L, R] 的 2D 坐标
    axes_2d_all = []  # 每帧 [L_axes(3), R_axes(3)] 的 2D 坐标
    
    for i in range(num_frames):
        R, t = R_all[i], t_all[i]
        R_inv, t_inv = R.T, -R.T @ t
        rvec, _ = cv2.Rodrigues(R_inv)
        
        # 应用夹爪偏移：从手腕位置转换为夹爪中心位置
        pos_frame = pos_all[i]
        quat_frame = quat_all[i] if quat_all is not None else None
        gripper_pos = apply_gripper_offset(pos_frame, quat_frame, GRIPPER_OFFSET)
        
        # 投影夹爪位置
        pts_2d, _ = cv2.projectPoints(gripper_pos.reshape(-1, 1, 3), rvec, t_inv.reshape(3, 1), K, dist)
        pts_2d_all.append(pts_2d.reshape(-1, 2).astype(np.int32))
        
        # 投影坐标轴（从夹爪位置出发）
        if quat_all is not None:
            axes_frame = []
            for j in range(2):  # L, R
                R_eef = quat_to_rotmat(quat_all[i, j])
                axes_3d = gripper_pos[j] + (R_eef @ (np.eye(3) * AXIS_LEN)).T
                axes_2d, _ = cv2.projectPoints(axes_3d.reshape(-1, 1, 3), rvec, t_inv.reshape(3, 1), K, dist)
                axes_frame.append(axes_2d.reshape(-1, 2).astype(np.int32))
            axes_2d_all.append(axes_frame)
    
    precompute_time = time.time() - precompute_start
    print(f"Processing {num_frames} frames @ {w}x{h}, {fps:.1f} fps (precompute: {precompute_time:.2f}s)")
    
    # ====== 使用硬件编码 (NVENC) ======
    decode = subprocess.Popen(
        f'ffmpeg -hide_banner -loglevel error -i "{paths["video"]}" -f rawvideo -pix_fmt bgr24 -',
        shell=True, stdout=subprocess.PIPE
    )
    # 尝试 NVENC，失败则回退到 CPU
    encode = subprocess.Popen(
        f'ffmpeg -y -hide_banner -loglevel error -f rawvideo -pix_fmt bgr24 -s {w}x{h} -r {fps} -i - '
        f'-c:v h264_nvenc -preset p1 -rc vbr -cq 23 -pix_fmt yuv420p "{output_path}"',
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
        
        # 绘制 EEF 点和标签
        for j in range(2):
            x, y = pts[j]
            if 0 <= x < w and 0 <= y < h:
                cv2.circle(frame, (x, y), 8, colors[j], -1)
                cv2.putText(frame, labels[j], (x+10, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, colors[j], 2)
                
                # 绘制坐标轴
                if axes_2d_all:
                    for k in range(3):
                        ax, ay = axes_2d_all[i][j][k]
                        if 0 <= ax < w and 0 <= ay < h:
                            cv2.line(frame, (x, y), (ax, ay), axis_colors[k], 2)
        
        # 帧信息
        info = f"Frame {i} | L:({pts[0,0]},{pts[0,1]}) R:({pts[1,0]},{pts[1,1]})"
        cv2.putText(frame, info, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        encode.stdin.write(frame.tobytes())
        
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{num_frames} ({100*(i+1)/num_frames:.0f}%)")
    
    encode.stdin.close()
    encode.wait()
    decode.wait()
    
    elapsed = time.time() - start_time
    print(f"Done! Output: {output_path} ({elapsed:.1f}s, {num_frames/elapsed:.1f} fps)")


def main():
    paths = get_paths(DATA_ROOT, TASK_ID, EPISODE_ID, CAMERA_NAME)
    
    # 检查文件
    for name, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing {name}: {path}")
    
    if MODE == "frame":
        output = OUTPUT_PATH or Path(f"./eef_frame_{FRAME_INDEX:04d}.png")
        process_single_frame(paths, FRAME_INDEX, output)
    else:
        output = OUTPUT_PATH or Path(f"./eef_{TASK_ID}_{EPISODE_ID}.mp4")
        process_video(paths, output)


if __name__ == "__main__":

    # ==================== 配置参数 ====================
    DATA_ROOT = Path("/mnt/raid0/AgiBot2Lerobot/AgiBot_Word_Beta")
    TASK_ID = 351
    EPISODE_ID = 777217
    CAMERA_NAME = "head"

    # 模式: "frame" 单帧模式, "video" 视频模式
    MODE = "video"
    FRAME_INDEX = 0  # 单帧模式时使用

    # 输出路径 (None 则自动生成)
    OUTPUT_PATH = None

    # 可视化参数
    AXIS_LEN = 0.08  # 坐标轴长度 (米)
    GRIPPER_OFFSET = 0.143  # 夹爪偏移量 (米)，从手腕到夹爪中心的距离
    # ================================================
    main()
