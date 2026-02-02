import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.spatial.transform import Rotation as R
from urdf_solver.ikfk_utils import IKFKSolver

# ============================================================================
# 配置选项 - CONFIGURATION
# ============================================================================
# 选择末端执行器模式：
#   "GRIPPER" = 使用 gripper center (通过 right_arm_to_gripper_transform / left_arm_to_gripper_transform 变换)
#   "ARM"     = 使用 arm end (不做任何变换，直接用 FK 输出)
EEF_MODE = "GRIPPER"  # 👈 改这里！"GRIPPER" 或 "ARM"
# ============================================================================

# 读取 HDF5 文件
path = Path("/home/unitree/桌面/agibot_world_eef/sample_dataset/proprio_stats/384/655302/proprio_stats.h5")

with h5py.File(path, "r") as f:
    # 读取状态数据
    state_joint_positions = f["state/joint/position"][:]  # shape: (1422, 14)
    state_end_positions = f["state/end/position"][:]  # shape: (1422, 2, 3) - [left, right]
    state_end_orientations = f["state/end/orientation"][:]  # shape: (1422, 2, 4) - [left, right] quaternion
    state_head_positions = f["state/head/position"][:]  # shape: (1422, 2)
    state_waist_positions = f["state/waist/position"][:]  # shape: (1422, 2)

# 初始化 IK/FK solver
# 使用第一帧的状态作为初始状态
arm_init = state_joint_positions[0]  # 14 joints
head_init = state_head_positions[0]  # 2 values
waist_init = state_waist_positions[0]  # 2 values

print(f"\nInitializing solver with:")
print(f"  arm_init: {arm_init}")
print(f"  head_init: {head_init}")
print(f"  waist_init: {waist_init}")

solver = IKFKSolver(
    arm_init_joint_position=arm_init,
    head_init_position=head_init,
    waist_init_position=waist_init
)

# 选择一些帧进行验证
indices = np.arange(len(state_joint_positions))

# 存储结果 - 位置
gt_left_pos = []
gt_right_pos = []
fk_left_pos = []
fk_right_pos = []

# 存储结果 - RPY 姿态
gt_left_rpy = []
gt_right_rpy = []
fk_left_rpy = []
fk_right_rpy = []

print(f"Mode: {EEF_MODE} ({'use gripper transform' if EEF_MODE == 'GRIPPER' else 'no transform'})")

for idx in indices:
    arm_joints = state_joint_positions[idx]
    
    # ========== Ground Truth (GT) ==========
    # 位置
    gt_left_pos.append(state_end_positions[idx, 0, :])  # left arm
    gt_right_pos.append(state_end_positions[idx, 1, :])  # right arm
    
    # 姿态：HDF5 四元数格式是 [x, y, z, w]，转换为 RPY
    gt_left_quat_xyzw = state_end_orientations[idx, 0, :]  # [x, y, z, w]
    gt_right_quat_xyzw = state_end_orientations[idx, 1, :]  # [x, y, z, w]
    
    # 使用 scipy 转换 xyzw 四元数为 RPY (scalar_last 即 xyzw 格式)
    gt_left_rpy.append(R.from_quat(gt_left_quat_xyzw, scalar_first=False).as_euler("xyz", degrees=False))
    gt_right_rpy.append(R.from_quat(gt_right_quat_xyzw, scalar_first=False).as_euler("xyz", degrees=False))
    
    # ========== FK 计算 ==========
    use_gripper_transform = (EEF_MODE == "GRIPPER")
    left_xyzrpy, right_xyzrpy = solver.compute_abs_eef_in_base(arm_joints, use_gripper_offset=use_gripper_transform)
    
    # 位置
    fk_left_pos.append(left_xyzrpy[:3])
    fk_right_pos.append(right_xyzrpy[:3])
    
    # 姿态 (RPY)
    fk_left_rpy.append(left_xyzrpy[3:])
    fk_right_rpy.append(right_xyzrpy[3:])

# 转换为 numpy 数组
gt_left_pos = np.array(gt_left_pos)
gt_right_pos = np.array(gt_right_pos)
fk_left_pos = np.array(fk_left_pos)
fk_right_pos = np.array(fk_right_pos)

gt_left_rpy = np.array(gt_left_rpy)
gt_right_rpy = np.array(gt_right_rpy)
fk_left_rpy = np.array(fk_left_rpy)
fk_right_rpy = np.array(fk_right_rpy)

# 单图展示：6 行（x, y, z, roll, pitch, yaw）× 2 列（Left/Right）
fig, axes = plt.subplots(6, 2, figsize=(16, 20), sharex=True)
fig.suptitle('FK Verification: GT vs FK (Position + RPY)', fontsize=16)

pos_labels = ['X', 'Y', 'Z']
rpy_labels = ['Roll', 'Pitch', 'Yaw']

# Left arm columns
for i, axis_name in enumerate(pos_labels):
    ax = axes[i, 0]
    ax.plot(indices, gt_left_pos[:, i], 'r-', label='GT', alpha=0.7)
    ax.plot(indices, fk_left_pos[:, i], 'b--', label='FK', alpha=0.7)
    ax.set_ylabel('Position (m)')
    ax.set_title(f'Left Arm - Pos {axis_name}')
    ax.legend()
    ax.grid(True, alpha=0.3)

for i, label in enumerate(rpy_labels):
    row = i + 3
    ax = axes[row, 0]
    ax.plot(indices, gt_left_rpy[:, i], 'r-', label='GT', alpha=0.7)
    ax.plot(indices, fk_left_rpy[:, i], 'b--', label='FK', alpha=0.7)
    ax.set_ylabel(f'{label} (rad)')
    ax.set_title(f'Left Arm - {label}')
    ax.legend()
    ax.grid(True, alpha=0.3)

# Right arm columns
for i, axis_name in enumerate(pos_labels):
    ax = axes[i, 1]
    ax.plot(indices, gt_right_pos[:, i], 'r-', label='GT', alpha=0.7)
    ax.plot(indices, fk_right_pos[:, i], 'b--', label='FK', alpha=0.7)
    ax.set_ylabel('Position (m)')
    ax.set_title(f'Right Arm - Pos {axis_name}')
    ax.legend()
    ax.grid(True, alpha=0.3)

for i, label in enumerate(rpy_labels):
    row = i + 3
    ax = axes[row, 1]
    ax.plot(indices, gt_right_rpy[:, i], 'r-', label='GT', alpha=0.7)
    ax.plot(indices, fk_right_rpy[:, i], 'b--', label='FK', alpha=0.7)
    ax.set_ylabel(f'{label} (rad)')
    ax.set_title(f'Right Arm - {label}')
    ax.legend()
    ax.grid(True, alpha=0.3)

for ax in axes[-1, :]:
    ax.set_xlabel('Frame Index')

plt.tight_layout()
plt.savefig('verify_fk.png', dpi=300)
print("\nPosition + RPY comparison plot saved to verify_fk.png")
