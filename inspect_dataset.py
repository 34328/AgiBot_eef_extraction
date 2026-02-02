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
    dataset = LeRobotDataset(repo_id=Path(dataset_path).name, root=dataset_path)
    
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
    sample = dataset[100]
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
    
    print("\n" + "=" * 80)
    print("Inspection complete!")


if __name__ == "__main__":
    dataset_path = "test_output/agibotworld/task_357"
    inspect_dataset(dataset_path)
