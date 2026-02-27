#!/usr/bin/env python3
"""
用途：
- 清理 `observations/<task_id>/<episode>/depth` 目录。
- 默认仅预览（DRY-RUN），不会真实删除。
- 只有 `apply=True`（或命令行传 `--apply`）才执行删除。

推荐用法（按你的习惯在 VSCode 里直接改）：
1. 修改 `ArgsConfig`：
   - `src_root`：数据根目录（内部应包含 observations）
   - `task_ids`：任务 ID 列表，例如 ["327", "351"]
   - `apply`：先设为 False 预览，确认无误后再改为 True
2. 运行脚本，先看 DRY-RUN 输出，再执行真实删除。

可选命令行用法：
- 预览：
  python3 tools/remove_depth_dirs.py --src-root /mnt/raid0/AgiBot2Lerobot/AgiBot_Word_Beta --task-ids 327 351
- 真删：
  python3 tools/remove_depth_dirs.py --src-root /mnt/raid0/AgiBot2Lerobot/AgiBot_Word_Beta --task-ids 327 351 --apply
"""

import shutil
from dataclasses import dataclass, field
from pathlib import Path

import tyro


def iter_episode_dirs(task_obs_dir: Path) -> list[Path]:
    if not task_obs_dir.exists():
        return []
    return sorted([p for p in task_obs_dir.iterdir() if p.is_dir()])


def is_safe_depth_dir(depth_dir: Path, task_obs_dir: Path) -> tuple[bool, str]:
    if depth_dir.name != "depth":
        return False, "target name is not 'depth'"
    if not depth_dir.exists():
        return False, "depth dir does not exist"
    if not depth_dir.is_dir():
        return False, "depth path is not a directory"
    if depth_dir.is_symlink():
        return False, "depth path is a symlink"

    resolved_task_obs = task_obs_dir.resolve()
    resolved_depth = depth_dir.resolve()
    try:
        resolved_depth.relative_to(resolved_task_obs)
    except ValueError:
        return False, "depth dir resolves outside task observations directory"

    return True, ""


def main(cfg: "ArgsConfig") -> int:
    src_root: Path = cfg.src_root
    observations_root = src_root / "observations"

    if not observations_root.exists():
        print(f"[ERROR] observations root not found: {observations_root}")
        return 1

    total_episodes = 0
    total_found = 0
    total_removed = 0
    total_missing = 0

    mode = "APPLY" if cfg.apply else "DRY-RUN"
    print(f"[{mode}] src_root={src_root}")

    for task_id in cfg.task_ids:
        task_obs_dir = observations_root / str(task_id)
        episodes = iter_episode_dirs(task_obs_dir)
        task_found = 0
        task_removed = 0
        task_missing = 0

        if not task_obs_dir.exists():
            print(f"[WARN] task observations dir not found: {task_obs_dir}")
            continue

        for ep_dir in episodes:
            if ep_dir.is_symlink():
                print(f"[SKIP] episode dir is symlink: {ep_dir}")
                continue
            total_episodes += 1
            depth_dir = ep_dir / "depth"
            safe, reason = is_safe_depth_dir(depth_dir, task_obs_dir)
            if safe:
                task_found += 1
                total_found += 1
                if cfg.apply:
                    shutil.rmtree(depth_dir)
                    task_removed += 1
                    total_removed += 1
                    print(f"[DELETE] {depth_dir}")
                else:
                    print(f"[DRY-RUN] would remove: {depth_dir}")
            else:
                if depth_dir.exists() and reason:
                    print(f"[SKIP] {depth_dir}: {reason}")
                task_missing += 1
                total_missing += 1

        print(
            f"[TASK {task_id}] episodes={len(episodes)} depth_found={task_found} "
            f"removed={task_removed} missing={task_missing}"
        )

    print(
        f"[TOTAL] episodes={total_episodes} depth_found={total_found} "
        f"removed={total_removed} missing={total_missing}"
    )
    return 0


@dataclass
class ArgsConfig:
    src_root: Path = Path("/mnt/raid0/AgiBot2Lerobot/AgiBot_Word_Beta")
    task_ids: list[str] = field(default_factory=lambda: ["357", "358", "359", "360", "361", "362", "363", "365"])
    apply: bool = True


if __name__ == "__main__":
    cfg = tyro.cli(ArgsConfig)
    raise SystemExit(main(cfg))
