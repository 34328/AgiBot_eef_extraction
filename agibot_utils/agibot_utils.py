import json
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from scipy.spatial.transform import Rotation as R
from urdf_solver.ikfk_utils import IKFKSolver


def get_task_info(task_json_path: str) -> dict:
    with open(task_json_path, "r") as f:
        task_info: list = json.load(f)
    task_info.sort(key=lambda episode: episode["episode_id"])
    return task_info


def load_depths(root_dir: str, camera_name: str):
    cam_path = Path(root_dir)
    all_imgs = sorted(list(cam_path.glob(f"{camera_name}*")))
    return [np.array(Image.open(f)).astype(np.float32)[:, :, None] / 1000 for f in all_imgs]


def compute_gripper_center_from_fk(joint_positions, head_positions, waist_positions):
    """
    Compute gripper center pose from joint states using FK.

    Args:
        joint_positions: (T, 14) - joint positions for both arms
        head_positions: (T, 2) - head joint positions
        waist_positions: (T, 2) - waist joint positions

    Returns:
        gripper_position: (T, 2, 3) - xyz positions for [left, right] grippers at gripper center
        gripper_orientation: (T, 2, 3) - rpy (roll, pitch, yaw) for [left, right] grippers at gripper center
    """
    T = joint_positions.shape[0]
    gripper_position = np.zeros((T, 2, 3), dtype=np.float32)
    gripper_orientation = np.zeros((T, 2, 3), dtype=np.float32)

    solver = IKFKSolver(
        arm_init_joint_position=joint_positions[0],
        head_init_position=head_positions[0],
        waist_init_position=waist_positions[0],
    )

    for t in range(T):
        left_xyzrpy, right_xyzrpy = solver.compute_abs_eef_in_base(
            joint_positions[t], use_gripper_offset=True
        )
        gripper_position[t, 0] = left_xyzrpy[:3]
        gripper_position[t, 1] = right_xyzrpy[:3]
        gripper_orientation[t, 0] = left_xyzrpy[3:]
        gripper_orientation[t, 1] = right_xyzrpy[3:]

    return gripper_position, gripper_orientation


def load_local_dataset(
    episode_id: int, src_path: str, task_id: int, save_depth: bool, AgiBotWorld_CONFIG: dict
) -> tuple[list, dict]:
    """Load local dataset and return a dict with observations and actions"""
    ob_dir = Path(src_path) / f"observations/{task_id}/{episode_id}"
    proprio_dir = Path(src_path) / f"proprio_stats/{task_id}/{episode_id}"

    state = {}
    with h5py.File(proprio_dir / "proprio_stats.h5", "r") as f:
        # Load raw state data from HDF5
        raw_state_data = {}
        for key in AgiBotWorld_CONFIG["states"]:
            # Handle end.eef specially - need to merge position and orientation
            if key == "end.eef":
                # Read joint/head/waist states and compute gripper center via FK
                joint_positions = np.array(f["state/joint/position"], dtype=np.float32)  # (T, 14)
                head_positions = np.array(f["state/head/position"], dtype=np.float32)  # (T, 2)
                waist_positions = np.array(f["state/waist/position"], dtype=np.float32)  # (T, 2)

                gripper_position, gripper_orientation = compute_gripper_center_from_fk(
                    joint_positions, head_positions, waist_positions
                )

                # Concatenate to get (T, 2, 6) - xyz + rpy at gripper center
                end_eef = np.concatenate([gripper_position, gripper_orientation], axis=-1)
                raw_state_data[key] = end_eef
            else:
                raw_state_data[key] = np.array(f["state/" + key.replace(".", "/")], dtype=np.float32)
        
        # Store state with proper keys
        for key, value in raw_state_data.items():
            state[f"observation.states.{key}"] = value

        num_frames = len(next(iter(raw_state_data.values())))
        
        # Create action data: use t+1 state as t action, last frame uses its own state
        action = {}
        for key in AgiBotWorld_CONFIG["actions"]:
            state_key = f"observation.states.{key}"
            if state_key in state:
                # Shift by 1: action[t] = state[t+1]
                # For the last frame, action[T-1] = state[T-1]
                action_data = np.zeros_like(state[state_key])
                action_data[:-1] = state[state_key][1:]  # t action = t+1 state
                action_data[-1] = state[state_key][-1]   # last action = last state
                action[f"actions.{key}"] = action_data

    if save_depth:
        depth_imgs = load_depths(ob_dir / "depth", "head_depth")
        assert num_frames == len(depth_imgs), "Number of images and states are not equal"

    # Check if any state or action data is empty
    for key, value in state.items():
        if not value.size:
            raise ValueError(f"State data '{key}' is empty! Cannot proceed with empty data.")
    
    for key, value in action.items():
        if not value.size:
            raise ValueError(f"Action data '{key}' is empty! Cannot proceed with empty data.")
    
    frames = [
        {
            **({"observation.images.head_depth": depth_imgs[i]} if save_depth else {}),
            **{key: value[i] for key, value in state.items()},
            **{key: value[i] for key, value in action.items()},
        }
        for i in range(num_frames)
    ]

    videos = {
        f"observation.images.{key}": ob_dir / "videos" / f"{key}_color.mp4"
        if "sensor" not in key
        else ob_dir / "tactile" / f"{key}.mp4"  # HACK: handle tactile videos
        for key in AgiBotWorld_CONFIG["images"]
        if "depth" not in key
    }
    return episode_id, frames, videos
