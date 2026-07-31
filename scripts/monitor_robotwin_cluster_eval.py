#!/usr/bin/env python3
import argparse
import json
import re
import time
from pathlib import Path

import yaml

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
SUCCESS_RE = re.compile(r"Success rate:\s*(\d+)/(\d+)")
EVAL_LOG_RE = re.compile(r"^eval_(.+)_\d{8}_\d{6}\.log$")
ALIGN_KEYS = ("model", "data", "mixed_precision", "eval_num_inference_steps")


def load_yaml(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def latest_task_logs(run_dir: Path) -> dict[str, Path]:
    logs: dict[str, Path] = {}
    for path in run_dir.glob("eval_*.log"):
        match = EVAL_LOG_RE.match(path.name)
        if match is None:
            continue
        task_name = match.group(1)
        previous = logs.get(task_name)
        if previous is None or path.stat().st_mtime > previous.stat().st_mtime:
            logs[task_name] = path
    return logs


def parse_progress(path: Path) -> tuple[int, int]:
    success = total = 0
    text = ANSI_RE.sub("", path.read_text(encoding="utf-8", errors="replace"))
    for match in SUCCESS_RE.finditer(text):
        success, total = int(match.group(1)), int(match.group(2))
    return success, total


def check_alignment(run_dir: Path, train_config: Path) -> list[str]:
    resolved_paths = sorted(run_dir.glob("resolved_sim_config_*.yaml"))
    if not resolved_paths:
        return ["waiting_for_resolved_config"]

    train_cfg = load_yaml(train_config)
    resolved_cfg = load_yaml(resolved_paths[0])
    mismatches = [
        key
        for key in ALIGN_KEYS
        if train_cfg.get(key) != resolved_cfg.get(key)
    ]
    return [f"config_mismatch:{key}" for key in mismatches]


def summarize(name: str, run_dir: Path, train_config: Path) -> dict:
    task_progress = {}
    poor_tasks = []
    for task_name, log_path in latest_task_logs(run_dir).items():
        success, total = parse_progress(log_path)
        if total:
            rate = success / total
            task_progress[task_name] = {
                "success": success,
                "episodes": total,
                "rate": round(rate, 4),
            }
            if total >= 10 and rate <= 0.05:
                poor_tasks.append(task_name)

    manager_log = run_dir / "manager.log"
    manager_text = (
        manager_log.read_text(encoding="utf-8", errors="replace")
        if manager_log.exists()
        else ""
    )
    return {
        "name": name,
        "completed_phases": len(list(run_dir.glob("*/_result_*.txt"))),
        "active_tasks_with_progress": len(task_progress),
        "observed_episodes": sum(item["episodes"] for item in task_progress.values()),
        "worker_failures": manager_text.count("worker failed:"),
        "alignment_alerts": check_alignment(run_dir, train_config),
        "poor_tasks_after_10_episodes": sorted(poor_tasks),
        "task_progress": task_progress,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        action="append",
        nargs=3,
        metavar=("NAME", "RUN_DIR", "TRAIN_CONFIG"),
        required=True,
    )
    parser.add_argument("--watch-seconds", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    while True:
        payload = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "runs": [
                summarize(name, Path(run_dir), Path(train_config))
                for name, run_dir, train_config in args.run
            ],
        }
        text = json.dumps(payload, ensure_ascii=False, indent=2)
        print(text, flush=True)
        if args.output is not None:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text + "\n", encoding="utf-8")
        if args.watch_seconds <= 0:
            break
        time.sleep(args.watch_seconds)


if __name__ == "__main__":
    main()
