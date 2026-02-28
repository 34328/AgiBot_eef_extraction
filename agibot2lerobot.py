import gc
import logging
import tyro
import shutil
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path

import av
import numpy as np
import ray
import torch
from agibot_utils.agibot_utils import get_task_info, load_local_dataset
from agibot_utils.config import AgiBotWorld_TASK_TYPE
from agibot_utils.filter_utils import (
    crop_episode_videos,
    format_crop_progress_line,
    format_episode_separator,
    format_episode_status_lines,
    get_episode_filter_match,
    load_filter_jsonl,
    slice_episode_frames,
)
from agibot_utils.lerobot_utils import compute_episode_stats, generate_features_from_config
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import validate_episode_buffer, validate_frame
from ray.runtime_env import RuntimeEnv


    # 压低 PyAV / FFmpeg 容器层日志，避免终端被无关 mp4 信息刷屏。
logging.getLogger("libav").setLevel(av.logging.ERROR)
av.logging.set_level(av.logging.ERROR)
av.logging.set_libav_level(av.logging.ERROR)
av.logging.set_skip_repeated(True)


class AgiBotDataset(LeRobotDataset):
    def add_frame(self, frame: dict) -> None:
        """
        仅将单帧数据追加到当前 episode_buffer 中。

        注意：
        - 这里不会真正把一整条 episode 落盘。
        - 视频数据只记录路径，真正写入数据集要等 `save_episode()` 调用。
        """
        # 如果上游给的是 torch.Tensor，这里统一转成 numpy，便于后续校验和堆叠。
        for name in frame:
            if isinstance(frame[name], torch.Tensor):
                frame[name] = frame[name].numpy()

        features = {key: value for key, value in self.features.items() if key in self.hf_features}  # 去掉视频键，只校验逐帧数据。
        validate_frame(frame, features)

        if self.episode_buffer is None:
            self.episode_buffer = self.create_episode_buffer()

        # 自动补齐 frame_index / timestamp，并把 task 单独存到 episode 级缓冲区。
        frame_index = self.episode_buffer["size"]
        timestamp = frame.pop("timestamp") if "timestamp" in frame else frame_index / self.fps
        self.episode_buffer["frame_index"].append(frame_index)
        self.episode_buffer["timestamp"].append(timestamp)
        self.episode_buffer["task"].append(frame.pop("task"))  # task 不再保留在单帧字典里。

        # 将当前帧的各字段追加到 episode_buffer，对应字段后续会统一 stack。
        for key, value in frame.items():
            if key not in self.features:
                raise ValueError(
                    f"An element of the frame is not in the features. '{key}' not in '{self.features.keys()}'."
                )

            self.episode_buffer[key].append(value)

        self.episode_buffer["size"] += 1

    def save_episode(self, videos: dict, action_config: list, camera_params: dict = None, episode_data: dict | None = None) -> None:
        """
        将当前 episode_buffer 中缓存的一整条 episode 写入数据集。

        Args:
            episode_data:
                若不为 None，则保存传入的 episode 数据；
                否则默认保存当前 `self.episode_buffer` 中由 `add_frame()` 累积的数据。
        """
        episode_buffer = episode_data if episode_data is not None else self.episode_buffer

        validate_episode_buffer(episode_buffer, self.meta.total_episodes, self.features)

        # `size` 和 `task` 是中间态字段，不直接写入最终的 hf_dataset。
        episode_length = episode_buffer.pop("size")
        tasks = episode_buffer.pop("task")
        episode_tasks = list(set(tasks))
        episode_index = episode_buffer["episode_index"]

        episode_buffer["index"] = np.arange(self.meta.total_frames, self.meta.total_frames + episode_length)
        episode_buffer["episode_index"] = np.full((episode_length,), episode_index)

        # 如有新任务文本，先更新任务表，再生成 task_index。
        self.meta.save_episode_tasks(episode_tasks)

        # 将自然语言任务描述映射成内部的 task_index。
        episode_buffer["task_index"] = np.array([self.meta.get_task_index(task) for task in tasks])

        for key, ft in self.features.items():
            # `index` / `episode_index` / `task_index` 已在上面处理；
            # 视频字段不在这里 stack，而是后面单独写路径和元数据。
            if key in ["index", "episode_index", "task_index"] or ft["dtype"] in ["video"]:
                continue
            episode_buffer[key] = np.stack(episode_buffer[key]).squeeze()

        for key in self.meta.video_keys:
            episode_buffer[key] = str(videos[key])  # PosixPath -> str

        ep_stats = compute_episode_stats(episode_buffer, self.features)

        ep_metadata = self._save_episode_data(episode_buffer)
        has_video_keys = len(self.meta.video_keys) > 0
        use_batched_encoding = self.batch_encoding_size > 1

        self.current_videos = videos
        if has_video_keys and not use_batched_encoding:
            for video_key in self.meta.video_keys:
                ep_metadata.update(self._save_episode_video(video_key, episode_index))

        # 等视频相关元数据准备好后，再把 action_config / camera_params 一并写入 episode metadata。
        ep_metadata.update({"action_config": action_config})
        if camera_params:
            ep_metadata.update({"camera_params": camera_params})
        self.meta.save_episode(episode_index, episode_length, episode_tasks, ep_stats, ep_metadata)

        if has_video_keys and use_batched_encoding:
            # 批量编码模式下，累计到指定 episode 数后再统一触发视频编码。
            self.episodes_since_last_encoding += 1
            if self.episodes_since_last_encoding == self.batch_encoding_size:
                start_ep = self.num_episodes - self.batch_encoding_size
                end_ep = self.num_episodes
                self._batch_save_episode_video(start_ep, end_ep)
                self.episodes_since_last_encoding = 0

        if not episode_data:
            # 保存完成后清空缓冲区，并清理可能残留的临时图像目录。
            self.clear_episode_buffer(delete_images=len(self.meta.image_keys) > 0)

    def _encode_temporary_episode_video(self, video_key: str, episode_index: int) -> Path:
        """
        为当前 episode 生成一个临时视频文件。

        这里不重新编码，只是把当前要写入的数据视频复制到数据集根目录下的临时位置，
        供后续 LeRobot 的保存流程继续处理。
        """
        temp_path = Path(tempfile.mkdtemp(dir=self.root)) / f"{video_key}_{episode_index:03d}.mp4"
        shutil.copy(self.current_videos[video_key], temp_path)
        return temp_path


def get_all_tasks(src_path: Path, output_path: Path):
    """遍历 task_info 目录，为每个 task 生成 (json_file, 输出目录) 二元组。"""
    json_files = src_path.glob("task_info/*.json")
    for json_file in json_files:
        local_dir = output_path / "agibotworld" / json_file.stem
        yield (json_file, local_dir.resolve())


def save_as_lerobot_dataset(agibot_world_config, task: tuple[Path, Path], save_depth, filter_lookups: dict | None = None):
    json_file, local_dir = task
    print(f"processing {json_file.stem}, saving to {local_dir}")
    src_path = json_file.parent.parent
    task_info = get_task_info(json_file)
    task_name = task_info[0]["task_name"]
    task_init_scene = task_info[0]["init_scene_text"]
    task_instruction = f"{task_name} | {task_init_scene}"
    task_id = json_file.stem.split("_")[-1]
    task_info = {episode["episode_id"]: episode for episode in task_info}

    features = generate_features_from_config(agibot_world_config)

    if local_dir.exists():
        shutil.rmtree(local_dir)

    if not save_depth:
        features.pop("observation.states.head_depth")

    dataset: AgiBotDataset = AgiBotDataset.create(
        repo_id=json_file.stem,
        root=local_dir,
        fps=30,
        robot_type="a2d",
        features=features,
    )

    all_subdir = [f.as_posix() for f in src_path.glob(f"observations/{task_id}/*") if f.is_dir()]

    all_subdir_eids = sorted([int(Path(path).name) for path in all_subdir])
    # 三个过滤表都提前在 main() 中加载好，这里只做 episode 级查询。
    inconsistent_lookup = (filter_lookups or {}).get("inconsistent_frames_length", {})
    camera_lookup = (filter_lookups or {}).get("camera_extrinsic", {})
    still_lookup = (filter_lookups or {}).get("still_frames", {})

    total_episodes = len(all_subdir_eids)
    for idx, eid in enumerate(all_subdir_eids, 1):
        if eid not in task_info:
            raise ValueError(f"{json_file.stem}, episode_{eid} not in task_info.json! Missing episode metadata.")

        # 严格按 1 -> 2 -> 3 的顺序检查当前 episode 是否命中过滤规则。
        matched_filter, matched_payload = get_episode_filter_match(
            task_id,
            eid,
            inconsistent_lookup,
            camera_lookup,
            still_lookup,
        )
        if matched_filter == "inconsistent_frames_length":
            # 第 1 道命中后直接跳过，不再继续检查后面的规则。
            for line in format_episode_status_lines(
                idx,
                total_episodes,
                json_file.stem,
                eid,
                "filtered",
                "inconsistent_frames_length",
            ):
                print(line)
            print(format_episode_separator())
            continue
        if matched_filter == "camera_extrinsic":
            # 第 2 道命中后同样直接跳过。
            reason = matched_payload.get("reason", "camera_extrinsic")
            for line in format_episode_status_lines(
                idx,
                total_episodes,
                json_file.stem,
                eid,
                "filtered",
                f"camera_extrinsic ({reason})",
            ):
                print(line)
            print(format_episode_separator())
            continue

        detail = "good episode"
        if matched_filter == "still_frames":
            # 第 3 道不跳过，而是保留指定帧区间并同步裁剪输出视频。
            detail = (
                f"still_frames keep [{matched_payload['start_frame']}, {matched_payload['end_frame']}]"
            )
        for line in format_episode_status_lines(
            idx,
            total_episodes,
            json_file.stem,
            eid,
            "cropping" if matched_filter == "still_frames" else "good",
            detail,
        ):
            print(line)

        action_config = task_info[eid]["label_info"]["action_config"]
        raw_dataset = load_local_dataset(
            eid,
            src_path=src_path,
            task_id=task_id,
            save_depth=save_depth,
            AgiBotWorld_CONFIG=agibot_world_config,
        )

        _, frames, videos, camera_params = raw_dataset
        frame_nums = len(frames)
        episode_videos = videos  # 默认直接使用原始视频路径；命中第 3 道时会替换成临时裁剪视频。
        crop_temp_dir = None
        # 先确认当前 episode 依赖的所有视频文件都存在，再进入保存流程。
        missing_videos = [str(video_path) for video_path in videos.values() if not video_path.exists()]
        if missing_videos:
            raise FileNotFoundError(
                f"{json_file.stem}, episode_{eid}: missing video files:\n" + "\n".join(missing_videos)
            )

        try:
            if matched_filter == "still_frames":
                start_frame = int(matched_payload["start_frame"])
                end_frame = int(matched_payload["end_frame"])
                # 先裁逐帧数据，再裁对应输出视频，保证两边帧区间完全一致。
                frames = slice_episode_frames(frames, start_frame, end_frame)
                episode_videos, crop_temp_dir = crop_episode_videos(
                    videos,
                    start_frame,
                    end_frame,
                    dataset.fps,
                    progress_callback=lambda video_idx, total_videos, video_path: print(
                        format_crop_progress_line(video_idx, total_videos, Path(video_path).name)
                    ),
                )
                frame_nums = len(frames)
                print(f"  - frames kept: {frame_nums}")

            for frame_data in frames:
                frame_data["task"] = task_instruction
                dataset.add_frame(frame_data)
                if save_depth:
                    # depth 已复制进 episode_buffer，这里尽早释放单帧引用，降低峰值内存。
                    frame_data.pop("observation.states.head_depth", None)
            # 当前 episode 的所有帧都已进 buffer，可以释放原 frames 列表。
            del frames

            dataset.save_episode(videos=episode_videos, action_config=action_config, camera_params=camera_params)
            print(f"  - status: done")
            print(f"  - frames saved: {frame_nums}")
        except Exception as e:
            print(f"  - status: failed")
            print(f"  - error: {str(e)}")
            dataset.episode_buffer = None
            # continue
        finally:
            if crop_temp_dir is not None:
                # 第 3 道过滤生成的临时裁剪视频只服务当前 episode，结束后立即清理。
                shutil.rmtree(crop_temp_dir, ignore_errors=True)
        gc.collect()
        print(format_episode_separator())


def main(
    src_path: str,
    output_path: str,
    eef_type: str,
    task_ids: list,
    cpus_per_task: int,
    save_depth: bool,
    filter_inconsistent_frames_length: bool,
    inconsistent_frames_length_path: Path,
    filter_camera_extrinsic: bool,
    camera_extrinsic_path: Path,
    filter_still_frames: bool,
    still_frames_path: Path,
    debug: bool = False,
):
    tasks = get_all_tasks(src_path, output_path)
    # 各过滤 JSONL 只在启动时加载一次，避免每个 episode 重复读文件。
    filter_lookups = {
        "inconsistent_frames_length": (
            load_filter_jsonl(inconsistent_frames_length_path) if filter_inconsistent_frames_length else {}
        ),
        "camera_extrinsic": load_filter_jsonl(camera_extrinsic_path) if filter_camera_extrinsic else {},
        "still_frames": load_filter_jsonl(still_frames_path) if filter_still_frames else {},
    }

    agibot_world_config, type_task_ids = (
        AgiBotWorld_TASK_TYPE[eef_type]["task_config"],
        AgiBotWorld_TASK_TYPE[eef_type]["task_ids"],
    )

    if eef_type == "gripper":
        remaining_ids = AgiBotWorld_TASK_TYPE["dexhand"]["task_ids"] + AgiBotWorld_TASK_TYPE["tactile"]["task_ids"]
        tasks = filter(lambda task: task[0].stem not in remaining_ids, tasks)
    else:
        tasks = filter(lambda task: task[0].stem in type_task_ids, tasks)

    if task_ids:
        # 如果手动指定了 task_ids，则只处理这些 task。
        tasks = filter(lambda task: task[0].stem in task_ids, tasks)

    if debug:
        save_as_lerobot_dataset(agibot_world_config, next(tasks), save_depth, filter_lookups)
    else:
        runtime_env = RuntimeEnv(
            env_vars={"HDF5_USE_FILE_LOCKING": "FALSE", "HF_DATASETS_DISABLE_PROGRESS_BARS": "TRUE"}
        )
        ray.init(runtime_env=runtime_env)
        resources = ray.available_resources()
        cpus = int(resources["CPU"])

        print(f"Available CPUs: {cpus}, num_cpus_per_task: {cpus_per_task}")
        # 非 debug 模式下按 task 并行，每个 task 占用指定 CPU 数。
        remote_task = ray.remote(save_as_lerobot_dataset).options(num_cpus=cpus_per_task)
        futures = []
        for task in tasks:
            futures.append((task[0].stem, remote_task.remote(agibot_world_config, task, save_depth, filter_lookups)))

        for task, future in futures:
            try:
                ray.get(future)
            except Exception as e:
                print(f"Exception occurred for {task}")
                with open("output.txt", "a") as f:
                    f.write(f"{task}, exception details: {str(e)}\n")

        ray.shutdown()


@dataclass
class ArgsConfig:
    # 原始 AgiBotWorld 数据根目录，内部应包含 task_info/ observations/ proprio_stats/ parameters
    src_path: Path = Path("/mnt/raid0/AgiBot2Lerobot/AgiBot_Word_Beta")

    # LeRobot 数据集输出根目录，最终会写入 output_path/agibotworld/task_xxx
    output_path: Path = Path("/mnt/raid0/AgiBot2Lerobot/lerobot_v3.0")

    # 数据类型：gripper / dexhand / tactile，用于选择 task 配置和过滤规则
    eef_type: str = "gripper"

    # 要处理的任务列表（如 ["task_694"]）；为空时按 eef_type 自动筛选
    task_ids: list[str] = field(default_factory=lambda: ["task_327"])

    # 非 debug 并行模式下，每个 Ray 任务占用的 CPU 核数
    cpus_per_task: int = 3

    # 是否保存头部深度图（observation.states.head_depth）
    save_depth: bool = False

    # 第一层过滤：跳过帧长不一致的 episode
    filter_inconsistent_frames_length: bool = True
    inconsistent_frames_length_path: Path = Path("filter_store/Inconsistent_frams_length.jsonl")

    # 第二层过滤：跳过相机外参异常的 episode
    filter_camera_extrinsic: bool = True
    camera_extrinsic_path: Path = Path("filter_store/camera_extrinsic.jsonl")

    # 第三层过滤：按 start_frame/end_frame 裁剪 episode 与输出视频
    filter_still_frames: bool = True
    still_frames_path: Path = Path("filter_store/still_frames.jsonl")

    # 调试模式：只处理筛选结果中的第一个任务
    debug: bool = True


if __name__ == "__main__":
    cfg = tyro.cli(ArgsConfig)
    main(**asdict(cfg))
