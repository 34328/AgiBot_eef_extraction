import h5py
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import random

# ================= Configuration =================
# Mode: "BATCH" for multiple random episodes, "SINGLE" for specific episode
MODE = "BATCH" 

# Path to the task directory
DATA_ROOT = Path("/mnt/raid0/AgiBot2Lerobot/AgiBot_Word_Beta/proprio_stats/351/")

# Settings for BATCH mode
BATCH_SIZE = 100

# Settings for SINGLE mode
TARGET_EPISODE_ID = "850864"  # Replace with specific ID string if needed
# =================================================

print(f"Running in {MODE} mode...")
print(f"Data Root: {DATA_ROOT}")

# Get all episodes
all_episodes = [d for d in DATA_ROOT.iterdir() if d.is_dir()]
print(f"Total episodes found: {len(all_episodes)}")

if not all_episodes:
    print("No episodes found!")
    exit()

selected_episodes = []
plot_title_suffix = ""

if MODE == "BATCH":
    count = min(BATCH_SIZE, len(all_episodes))
    selected_episodes = random.sample(all_episodes, count)
    print(f"Selected {len(selected_episodes)} random episodes.")
    plot_title_suffix = f"(Batch of {len(selected_episodes)})"

elif MODE == "SINGLE":
    # Try to find the specific episode
    target_path = DATA_ROOT / TARGET_EPISODE_ID
    if target_path.exists() and target_path.is_dir():
        selected_episodes = [target_path]
        print(f"Selected specific Episode ID: {TARGET_EPISODE_ID}")
        plot_title_suffix = f"(Episode {TARGET_EPISODE_ID})"
    else:
        # Fallback if ID invalid or empty
        print(f"Episode {TARGET_EPISODE_ID} not found, picking random one...")
        selected_episodes = [random.choice(all_episodes)]
        print(f"Selected random Episode ID: {selected_episodes[0].name}")
        plot_title_suffix = f"(Episode {selected_episodes[0].name})"

waist_pos_1 = []
waist_pos_2 = []

for ep_path in selected_episodes:
    h5_file = ep_path / "proprio_stats.h5"
    if h5_file.exists():
        try:
            with h5py.File(h5_file, "r") as f:
                if "state/waist/position" in f:
                    data = f["state/waist/position"][:]
                    # 假设 shape 是 (N, 2)，分别收集两列数据
                    waist_pos_1.extend(data[:, 0])
                    waist_pos_2.extend(data[:, 1])
        except Exception as e:
            print(f"Error extracting {h5_file}: {e}")

# 转换为 numpy 数组以便绘图
waist_pos_1 = np.round(np.array(waist_pos_1), 2)
waist_pos_2 = np.round(np.array(waist_pos_2), 2)

if len(waist_pos_1) == 0:
    print("No data collected.")
    exit()

print(f"Total data points collected: {len(waist_pos_1)}")
print(f"Value 1 (Pitch) Range: [{waist_pos_1.min():.2f}, {waist_pos_1.max():.2f}]")
print(f"Value 1 (Pitch) Std Dev: {waist_pos_1.std():.4f}")
print(f"Value 2 (Lift) Range: [{waist_pos_2.min():.2f}, {waist_pos_2.max():.2f}]")
print(f"Value 2 (Lift) Std Dev: {waist_pos_2.std():.4f}")

# 绘制直方图
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.hist(waist_pos_1, bins=50, color='blue', alpha=0.7)
plt.title(f"Waist Pitch (Val 1) {plot_title_suffix}")
plt.xlabel("Value")
plt.ylabel("Frequency")

plt.subplot(1, 2, 2)
plt.hist(waist_pos_2, bins=50, color='green', alpha=0.7)
plt.title(f"Waist Lift (Val 2) {plot_title_suffix}")
plt.xlabel("Value")
plt.ylabel("Frequency")

plt.tight_layout()
output_file = "waist_position_histogram.png"
plt.savefig(output_file)
print(f"Histogram saved to {output_file}")