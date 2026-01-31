import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
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
path = Path("/mnt/raid0/AgiBot_Word_Beta/proprio_stats/358/654803/proprio_stats.h5")

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

# 存储结果
computed_left_pos = []
computed_right_pos = []
ground_truth_left_pos = []
ground_truth_right_pos = []

computed_left_quat = []
computed_right_quat = []
ground_truth_left_quat = []
ground_truth_right_quat = []

print(f"Mode: {EEF_MODE} ({'use gripper transform' if EEF_MODE == 'GRIPPER' else 'no transform'})")

for idx in indices:
    arm_joints = state_joint_positions[idx]
    
    # 使用 FK 计算 eef 位置和姿态
    use_gripper_transform = (EEF_MODE == "GRIPPER")
    left_xyzquat, right_xyzquat = solver.compute_abs_eef_in_base_quat(arm_joints, use_gripper_offset=use_gripper_transform)
    
    # 存储计算结果
    computed_left_pos.append(left_xyzquat[:3])
    computed_right_pos.append(right_xyzquat[:3])
    computed_left_quat.append(left_xyzquat[3:])
    computed_right_quat.append(right_xyzquat[3:])
    
    # 存储真实值
    ground_truth_left_pos.append(state_end_positions[idx, 0, :])  # left arm
    ground_truth_right_pos.append(state_end_positions[idx, 1, :])  # right arm
    
    # HDF5 四元数格式是 [x, y, z, w]，需要转换成 [w, x, y, z] 才能和 FK 输出对比
    gt_left_quat_xyzw = state_end_orientations[idx, 0, :]  # [x, y, z, w]
    gt_right_quat_xyzw = state_end_orientations[idx, 1, :]  # [x, y, z, w]
    gt_left_quat_wxyz = np.array([gt_left_quat_xyzw[3], gt_left_quat_xyzw[0], gt_left_quat_xyzw[1], gt_left_quat_xyzw[2]])
    gt_right_quat_wxyz = np.array([gt_right_quat_xyzw[3], gt_right_quat_xyzw[0], gt_right_quat_xyzw[1], gt_right_quat_xyzw[2]])
    
    # 处理四元数的符号歧义：q 和 -q 表示同一个旋转
    # 如果 dot product < 0，翻转 GT 四元数的符号
    if np.dot(left_xyzquat[3:], gt_left_quat_wxyz) < 0:
        gt_left_quat_wxyz = -gt_left_quat_wxyz
    if np.dot(right_xyzquat[3:], gt_right_quat_wxyz) < 0:
        gt_right_quat_wxyz = -gt_right_quat_wxyz
    
    ground_truth_left_quat.append(gt_left_quat_wxyz)  # [w, x, y, z]
    ground_truth_right_quat.append(gt_right_quat_wxyz)  # [w, x, y, z]

# 转换为 numpy 数组
computed_left_pos, computed_right_pos, ground_truth_left_pos, ground_truth_right_pos = map(
    np.array,
    (computed_left_pos, computed_right_pos, ground_truth_left_pos, ground_truth_right_pos),
)
computed_left_quat, computed_right_quat, ground_truth_left_quat, ground_truth_right_quat = map(
    np.array,
    (computed_left_quat, computed_right_quat, ground_truth_left_quat, ground_truth_right_quat),
)

# 单图展示：7 行（x, y, z, w, x, y, z）× 2 列（Left/Right）
fig, axes = plt.subplots(7, 2, figsize=(16, 22), sharex=True)
fig.suptitle('FK Verification: Computed vs Ground Truth (Position + Quaternion)', fontsize=16)

pos_labels = ['X', 'Y', 'Z']
quat_labels = ['w', 'x', 'y', 'z']
row_labels = [f'Pos {p}' for p in pos_labels] + [f'Quat {q}' for q in quat_labels]

# Left arm columns
for i, axis_name in enumerate(pos_labels):
    ax = axes[i, 0]
    ax.plot(indices, computed_left_pos[:, i], 'b-', label='Computed', alpha=0.7)
    ax.plot(indices, ground_truth_left_pos[:, i], 'r--', label='Ground Truth', alpha=0.7)
    ax.set_ylabel('Position (m)')
    ax.set_title(f'Left Arm - {row_labels[i]}')
    ax.legend()
    ax.grid(True, alpha=0.3)

for i, label in enumerate(quat_labels):
    row = i + 3
    ax = axes[row, 0]
    ax.plot(indices, computed_left_quat[:, i], 'b-', label='Computed', alpha=0.7)
    ax.plot(indices, ground_truth_left_quat[:, i], 'r--', label='Ground Truth', alpha=0.7)
    ax.set_ylabel(f'Quat {label}')
    ax.set_title(f'Left Arm - {row_labels[row]}')
    ax.legend()
    ax.grid(True, alpha=0.3)

# Right arm columns
for i, axis_name in enumerate(pos_labels):
    ax = axes[i, 1]
    ax.plot(indices, computed_right_pos[:, i], 'b-', label='Computed', alpha=0.7)
    ax.plot(indices, ground_truth_right_pos[:, i], 'r--', label='Ground Truth', alpha=0.7)
    ax.set_ylabel('Position (m)')
    ax.set_title(f'Right Arm - {row_labels[i]}')
    ax.legend()
    ax.grid(True, alpha=0.3)

for i, label in enumerate(quat_labels):
    row = i + 3
    ax = axes[row, 1]
    ax.plot(indices, computed_right_quat[:, i], 'b-', label='Computed', alpha=0.7)
    ax.plot(indices, ground_truth_right_quat[:, i], 'r--', label='Ground Truth', alpha=0.7)
    ax.set_ylabel(f'Quat {label}')
    ax.set_title(f'Right Arm - {row_labels[row]}')
    ax.legend()
    ax.grid(True, alpha=0.3)

for ax in axes[-1, :]:
    ax.set_xlabel('Frame Index')

plt.tight_layout()
plt.savefig('/home/unitree/桌面/agibot_world_eef/fk_verification_eef.png', dpi=300)
print("\nPosition + quaternion comparison plot ready.")
