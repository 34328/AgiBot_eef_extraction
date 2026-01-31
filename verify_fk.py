import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from ikfk_utils import IKFKSolver

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
num_samples = min(100, len(state_joint_positions))
indices = np.linspace(0, len(state_joint_positions)-1, num_samples, dtype=int)

# 存储结果
computed_left_pos = []
computed_right_pos = []
ground_truth_left_pos = []
ground_truth_right_pos = []

computed_left_quat = []
computed_right_quat = []
ground_truth_left_quat = []
ground_truth_right_quat = []

print(f"\nComputing FK for {num_samples} frames...")
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
    # # 如果 dot product < 0，翻转 GT 四元数的符号
    # if np.dot(left_xyzquat[3:], gt_left_quat_wxyz) < 0:
    #     gt_left_quat_wxyz = -gt_left_quat_wxyz
    # if np.dot(right_xyzquat[3:], gt_right_quat_wxyz) < 0:
    #     gt_right_quat_wxyz = -gt_right_quat_wxyz
    
    ground_truth_left_quat.append(gt_left_quat_wxyz)  # [w, x, y, z]
    ground_truth_right_quat.append(gt_right_quat_wxyz)  # [w, x, y, z]

# 转换为 numpy 数组
computed_left_pos = np.array(computed_left_pos)
computed_right_pos = np.array(computed_right_pos)
ground_truth_left_pos = np.array(ground_truth_left_pos)
ground_truth_right_pos = np.array(ground_truth_right_pos)

computed_left_quat = np.array(computed_left_quat)
computed_right_quat = np.array(computed_right_quat)
ground_truth_left_quat = np.array(ground_truth_left_quat)
ground_truth_right_quat = np.array(ground_truth_right_quat)

# 计算位置误差
left_pos_error = np.linalg.norm(computed_left_pos - ground_truth_left_pos, axis=1)
right_pos_error = np.linalg.norm(computed_right_pos - ground_truth_right_pos, axis=1)



# 计算四元数误差（角度差）
def quaternion_angular_error(q1, q2):
    """
    Calculate angular error between two quaternions in degrees
    q1, q2: quaternions in format [w, x, y, z] (scalar first)
    """
    # 确保归一化
    q1 = q1 / np.linalg.norm(q1)
    q2 = q2 / np.linalg.norm(q2)
    
    # 计算点积
    dot_product = np.abs(np.dot(q1, q2))
    # 限制在 [-1, 1] 范围内避免数值误差
    dot_product = np.clip(dot_product, 0.0, 1.0)
    
    # 角度差 (弧度)
    angle_rad = 2 * np.arccos(dot_product)
    # 转换为度
    angle_deg = np.degrees(angle_rad)
    
    return angle_deg

# 计算每一帧的角度误差
left_quat_errors = []
right_quat_errors = []

for i in range(len(computed_left_quat)):
    left_error_deg = quaternion_angular_error(computed_left_quat[i], ground_truth_left_quat[i])
    right_error_deg = quaternion_angular_error(computed_right_quat[i], ground_truth_right_quat[i])
    left_quat_errors.append(left_error_deg)
    right_quat_errors.append(right_error_deg)

left_quat_errors = np.array(left_quat_errors)
right_quat_errors = np.array(right_quat_errors)


# 可视化位置
fig, axes = plt.subplots(3, 2, figsize=(15, 12))
fig.suptitle('FK Verification: Computed vs Ground Truth (Position)', fontsize=16)

# Left Arm - X, Y, Z
for i, axis_name in enumerate(['X', 'Y', 'Z']):
    ax = axes[i, 0]
    ax.plot(indices, computed_left_pos[:, i], 'b-', label='Computed', alpha=0.7)
    ax.plot(indices, ground_truth_left_pos[:, i], 'r--', label='Ground Truth', alpha=0.7)
    ax.set_xlabel('Frame Index')
    ax.set_ylabel(f'Position (m)')
    ax.set_title(f'Left Arm - {axis_name} Position')
    ax.legend()
    ax.grid(True, alpha=0.3)

# Right Arm - X, Y, Z
for i, axis_name in enumerate(['X', 'Y', 'Z']):
    ax = axes[i, 1]
    ax.plot(indices, computed_right_pos[:, i], 'b-', label='Computed', alpha=0.7)
    ax.plot(indices, ground_truth_right_pos[:, i], 'r--', label='Ground Truth', alpha=0.7)
    ax.set_xlabel('Frame Index')
    ax.set_ylabel(f'Position (m)')
    ax.set_title(f'Right Arm - {axis_name} Position')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/unitree/Desktop/LZH/AgiBot_eef_extraction/fk_verification_positions.png', dpi=150)
print(f"\nPosition plot saved to fk_verification_positions.png")

# 位置误差图
fig, axes = plt.subplots(2, 1, figsize=(12, 8))
fig.suptitle('Position Error Over Time', fontsize=16)

axes[0].plot(indices, left_pos_error * 1000, 'b-', linewidth=2)
axes[0].set_xlabel('Frame Index')
axes[0].set_ylabel('Error (mm)')
axes[0].set_title('Left Arm Position Error')
axes[0].grid(True, alpha=0.3)
axes[0].axhline(y=left_pos_error.mean() * 1000, color='r', linestyle='--', 
                label=f'Mean: {left_pos_error.mean()*1000:.2f} mm')
axes[0].legend()

axes[1].plot(indices, right_pos_error * 1000, 'b-', linewidth=2)
axes[1].set_xlabel('Frame Index')
axes[1].set_ylabel('Error (mm)')
axes[1].set_title('Right Arm Position Error')
axes[1].grid(True, alpha=0.3)
axes[1].axhline(y=right_pos_error.mean() * 1000, color='r', linestyle='--',
                label=f'Mean: {right_pos_error.mean()*1000:.2f} mm')
axes[1].legend()

plt.tight_layout()
plt.savefig('/home/unitree/Desktop/LZH/AgiBot_eef_extraction/fk_verification_pos_errors.png', dpi=150)
print(f"Position error plot saved to fk_verification_pos_errors.png")

# 四元数分量对比图
fig, axes = plt.subplots(4, 2, figsize=(15, 16))
fig.suptitle('FK Verification: Computed vs Ground Truth (Quaternion)', fontsize=16)

quat_labels = ['w', 'x', 'y', 'z']

# Left Arm - w, x, y, z
for i, label in enumerate(quat_labels):
    ax = axes[i, 0]
    ax.plot(indices, computed_left_quat[:, i], 'b-', label='Computed', alpha=0.7)
    ax.plot(indices, ground_truth_left_quat[:, i], 'r--', label='Ground Truth', alpha=0.7)
    ax.set_xlabel('Frame Index')
    ax.set_ylabel(f'Quaternion {label}')
    ax.set_title(f'Left Arm - Quaternion {label}')
    ax.legend()
    ax.grid(True, alpha=0.3)

# Right Arm - w, x, y, z
for i, label in enumerate(quat_labels):
    ax = axes[i, 1]
    ax.plot(indices, computed_right_quat[:, i], 'b-', label='Computed', alpha=0.7)
    ax.plot(indices, ground_truth_right_quat[:, i], 'r--', label='Ground Truth', alpha=0.7)
    ax.set_xlabel('Frame Index')
    ax.set_ylabel(f'Quaternion {label}')
    ax.set_title(f'Right Arm - Quaternion {label}')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/home/unitree/Desktop/LZH/AgiBot_eef_extraction/fk_verification_quaternions.png', dpi=150)
print(f"Quaternion comparison plot saved to fk_verification_quaternions.png")

# 四元数误差图
fig, axes = plt.subplots(2, 1, figsize=(12, 8))
fig.suptitle('Orientation Error Over Time', fontsize=16)

axes[0].plot(indices, left_quat_errors, 'b-', linewidth=2)
axes[0].set_xlabel('Frame Index')
axes[0].set_ylabel('Angular Error (degrees)')
axes[0].set_title('Left Arm Orientation Error')
axes[0].grid(True, alpha=0.3)
axes[0].axhline(y=left_quat_errors.mean(), color='r', linestyle='--', 
                label=f'Mean: {left_quat_errors.mean():.2f}°')
axes[0].legend()

axes[1].plot(indices, right_quat_errors, 'b-', linewidth=2)
axes[1].set_xlabel('Frame Index')
axes[1].set_ylabel('Angular Error (degrees)')
axes[1].set_title('Right Arm Orientation Error')
axes[1].grid(True, alpha=0.3)
axes[1].axhline(y=right_quat_errors.mean(), color='r', linestyle='--',
                label=f'Mean: {right_quat_errors.mean():.2f}°')
axes[1].legend()

plt.tight_layout()
plt.savefig('/home/unitree/Desktop/LZH/AgiBot_eef_extraction/fk_verification_orientation.png', dpi=150)
print(f"Orientation error plot saved to fk_verification_orientation.png")

print("\n=== Verification Complete ===")
if left_pos_error.mean() < 0.01 and right_pos_error.mean() < 0.01:
    print("✓ Position FK computation looks GOOD! Mean error < 1cm")
elif left_pos_error.mean() < 0.05 and right_pos_error.mean() < 0.05:
    print("⚠ Position FK computation is acceptable but has some error (< 5cm)")
else:
    print("✗ Position FK computation has significant error!")

if left_quat_errors.mean() < 1.0 and right_quat_errors.mean() < 1.0:
    print("✓ Orientation FK computation looks GOOD! Mean error < 1°")
elif left_quat_errors.mean() < 5.0 and right_quat_errors.mean() < 5.0:
    print("⚠ Orientation FK computation is acceptable but has some error (< 5°)")
else:
    print("✗ Orientation FK computation has significant error!")
