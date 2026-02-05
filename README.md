# AgiBot-World to LeRobot 🤖

🧭 AgiBot World 是首个大规模机器人学习数据集，旨在推动多用途机器人策略的发展。它配套提供基础模型、基准测试与生态系统，面向学术界与产业开放高质量机器人数据，推动具身智能的“ImageNet 时刻”。（摘自 [docs](https://agibot-world.com/)）

📘 关于AgibotWorld的介绍 请看：[agibot_world_itrd.md](agibot_world_itrd.md)

## 🧩 安装

```bash
conda create -n AgiBotEEF python=3.10
conda activate AgiBotEEF
conda install -c conda-forge pinocchio

pip install lerobot h5py
pip install -U "ray[default]"
pip install flask
```

## ⚠️ 有问题的任务

| (Gripper) Task ID | (Some episodes) Reason | Fixed By |
| :---------------: | :--------------------: | -------- |
|     task_352      | action_len > state_len | skipping |
|     task_354      | action_len > state_len | skipping |
|     task_359      | action_len > state_len | skipping |
|     task_361      | action_len > state_len | skipping |
|     task_368      | action_len > state_len | skipping |
|     task_376      | action_len > state_len | skipping |
|     task_377      | action_len > state_len | skipping |
|     task_380      |     corrupted mp4      | skipping |
|     task_384      |     corrupted mp4      | skipping |
|     task_410      | action_len > state_len | skipping |
|     task_414      | action_len > state_len | skipping |
|     task_421      | action_len > state_len | skipping |
|     task_428      |     corrupted mp4      | skipping |
|     task_460      |     corrupted mp4      | skipping |
|     task_505      |     corrupted mp4      | skipping |
|     task_510      |     corrupted mp4      | skipping |
|     task_711      |     corrupted mp4      | skipping |

## ✨ 这个脚本的新变化

🧪 在该数据集中，做了几项关键改进：

- **尽量保留 Agibot 原始信息** 🧠：尽可能保留 Agibot 的原始信息，字段名严格遵循原始数据集的命名规范，以保证兼容性与一致性。
- **State 与 Action 使用字典结构** 🧾：将传统的一维 state 与 action 转换为字典结构，便于灵活设计自定义状态与动作，实现模块化、可扩展的处理方式。

- **位移一位**🦾：Aigbot-World-Beta 原始的的 action 因为一些原因没有记录上，是从state 复制而来，这里我们位移一位，将第 $t+1$ 帧的state 赋给第 $t$ 帧的 action，最后一帧的action保持不变。
- **EEF字段整合**🧷： state/end/position 和 state/end/orientation 是基于arm手腕处的，本项目将其使用FK 从joint重新求解以转到 gripper 或者 Dexhand 的center处，concat 一个新的state/end/eef 字段。
- **相机内外参** 📸：在每个 episode的meta中保存了8个相机的内外参，方便后续使用。
- **深度图**：补充了深度图的保存逻辑。

## 🧾 `meta/info.json` 的数据结构如下：

```json
{
  "codebase_version": "v3.0",
  "robot_type": "a2d",
  "total_episodes": 1,
  "total_frames": 4683,
  "total_tasks": 1,
  "chunks_size": 1000,
  "data_files_size_in_mb": 100,
  "video_files_size_in_mb": 200,
  "fps": 30,
  "splits": {
    "train": "0:1"
  },
  "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
  "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
  "features": {
    "observation.images.head": {
      "dtype": "video",
      "shape": [480, 640, 3],
      "names": ["height", "width", "rgb"],
      "info": {
        "video.height": 480,
        "video.width": 640,
        "video.codec": "av1",
        "video.pix_fmt": "yuv420p",
        "video.is_depth_map": false,
        "video.fps": 30,
        "video.channels": 3,
        "has_audio": false
      }
    },
    "observation.images.head_center_fisheye": {
      "dtype": "video",
      "shape": [768, 960, 3],
      "names": ["height", "width", "rgb"],
      "info": {
        "video.height": 768,
        "video.width": 960,
        "video.codec": "av1",
        "video.pix_fmt": "yuv420p",
        "video.is_depth_map": false,
        "video.fps": 30,
        "video.channels": 3,
        "has_audio": false
      }
    },
    ...
    "observation.states.joint.position": {
      "dtype": "float32",
      "shape": [14],
      "names": {
        "motors": [
          "left_arm_0",
          "left_arm_1",
          "left_arm_2",
          "left_arm_3",
          "left_arm_4",
          "left_arm_5",
          "left_arm_6",
          "right_arm_0",
          "right_arm_1",
          "right_arm_2",
          "right_arm_3",
          "right_arm_4",
          "right_arm_5",
          "right_arm_6"
        ]
      }
    },
    "observation.states.head.position": {
      "dtype": "float32",
      "shape": [2],
      "names": {
        "motors": ["yaw", "patch"]
      }
    },
    ...
    "actions.joint.position": {
      "dtype": "float32",
      "shape": [14],
      "names": {
        "motors": [
          "left_arm_0",
          "left_arm_1",
          "left_arm_2",
          "left_arm_3",
          "left_arm_4",
          "left_arm_5",
          "left_arm_6",
          "right_arm_0",
          "right_arm_1",
          "right_arm_2",
          "right_arm_3",
          "right_arm_4",
          "right_arm_5",
          "right_arm_6"
        ]
      }
    },
    "actions.waist.position": {
      "dtype": "float32",
      "shape": [2],
      "names": {
        "motors": ["pitch", "lift"]
      }
    },
    "timestamp": {
      "dtype": "float32",
      "shape": [1],
      "names": null
    },
    "frame_index": {
      "dtype": "int64",
      "shape": [1],
      "names": null
    },
    ...
    "task_index": {
      "dtype": "int64",
      "shape": [1],
      "names": null
    }
  }
}
```
