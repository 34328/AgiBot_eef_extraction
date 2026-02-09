"""
Script to inspect LeRobot dataset structure and field information.
"""
import argparse
from pathlib import Path
from lerobot.datasets.lerobot_dataset import LeRobotDataset


def inspect_dataset(dataset_path: str):
    """Inspect the LeRobot dataset and print field information."""
    
    print(f"Loading dataset from: {dataset_path}")
    print("=" * 80)
    
    # Load the dataset
    dataset = LeRobotDataset(repo_id=Path(dataset_path).name, root=dataset_path, episodes=[208])
    
    # Print basic dataset info
    print("\n📊 Basic Dataset Information:")
    print("-" * 40)
    print(f"  Repo ID: {dataset.repo_id}")
    print(f"  FPS: {dataset.fps}")
    print(f"  Robot Type: {dataset.meta.robot_type}")
    print(f"  Total Episodes: {dataset.num_episodes}")
    print(f"  Total Frames: {dataset.num_frames}")
    
    # Print features info
    print("\n📋 Features (Keys) Information:")
    print("-" * 40)
    for key, feature in dataset.features.items():
        dtype = feature.get("dtype", "N/A")
        shape = feature.get("shape", "N/A")
        names = feature.get("names", None)
        print(f"\n  🔑 {key}:")
        print(f"      dtype: {dtype}")
        if isinstance(shape, (list, tuple)):
            print(f"      shape: {(dataset.num_frames,) + tuple(shape)}")
        else:
            print(f"      shape: {shape}")
        if names:
            print(f"      names: {names}")
    
    # Print video keys if any
    if dataset.meta.video_keys:
        print("\n🎥 Video Keys:")
        print("-" * 40)
        for vk in dataset.meta.video_keys:
            print(f"  - {vk}")
    
    # Print image keys if any
    if dataset.meta.image_keys:
        print("\n🖼️ Image Keys:")
        print("-" * 40)
        for ik in dataset.meta.image_keys:
            print(f"  - {ik}")
    
    # Sample one data point
    print("\n📦 Sample Data (First Frame):")
    print("-" * 40)
    sample = dataset[0]
    for key, value in sample.items():
        if hasattr(value, 'shape'):
            print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
        else:
            print(f"  {key}: {type(value).__name__} = {value}")
    
    # Print tasks
    print("\n📝 Tasks:")
    print("-" * 40)
    tasks_data = dataset.meta.tasks
    try:
        import pandas as pd
        if isinstance(tasks_data, pd.DataFrame):
            pd.set_option('display.max_colwidth', None)
            for idx, row in tasks_data.iterrows():
                print(f"  [{row.get('task_index', idx)}] {idx}")
        elif isinstance(tasks_data, dict):
            for task_idx, task_text in tasks_data.items():
                print(f"  [{task_idx}] {task_text}")
        elif isinstance(tasks_data, list):
            for idx, task_text in enumerate(tasks_data):
                print(f"  [{idx}] {task_text}")
        else:
            print(f"  Type: {type(tasks_data)}")
            print(f"  Content: {tasks_data}")
    except Exception as e:
        print(f"  Error reading tasks: {e}")
        print(f"  Raw data: {tasks_data}")
    
    # Print episode metadata (action_config and camera_params)
    print("\n📋 Episode Metadata (Episode 0):")
    print("-" * 40)
    try:
        episode_data = dataset.meta.episodes[0]
        
        # Print action_config
        if 'action_config' in episode_data:
            action_config = episode_data['action_config']
            print(f"  action_config: {len(action_config)} segments")
            for i, action in enumerate(action_config[:3]):  # Show first 3
                print(f"    [{i}] {action}")
            if len(action_config) > 3:
                print(f"    ... and {len(action_config) - 3} more")
        
        # Print one camera's params
        if 'camera_params' in episode_data:
            import json
            camera_params = episode_data['camera_params']
            print(f"\n  camera_params: {len(camera_params)} cameras")
            print(f"    Cameras: {list(camera_params.keys())}")
            # Show sample camera (head) - full params
            sample_cam = 'head' if 'head' in camera_params else list(camera_params.keys())[0]
            cam_data = camera_params[sample_cam]
            print(f"\n    Sample ({sample_cam}):")
            print(json.dumps(cam_data, indent=6))
    except Exception as e:
        print(f"  Error reading episode metadata: {e}")
    
    print("\n" + "=" * 80)
    print("Inspection complete!")


if __name__ == "__main__":
    dataset_path = "/mnt/raid0/AgiBot2Lerobot/lerobot_v3.0/agibotworld/task_327"
    inspect_dataset(dataset_path)
