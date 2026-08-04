import csv
import json
import os
import shlex
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import hydra
import yaml
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SINGLE_ENTRY = PROJECT_ROOT / "experiments" / "robotwin" / "eval_robotwin_single.py"
TERMINATE_TIMEOUT_SEC = 10
POLL_INTERVAL_SEC = 2


def _resolve_path(path_str: str, *, base: Path) -> Path:
    path = Path(os.path.expanduser(os.path.expandvars(str(path_str))))
    if not path.is_absolute():
        path = (base / path).resolve()
    return path.resolve()


def _resolve_ckpt_tag(ckpt_path: Path) -> str:
    parts = ckpt_path.resolve().parts
    if "runs" in parts:
        runs_idx = parts.index("runs")
        if runs_idx + 2 >= len(parts):
            raise ValueError(
                f"`ckpt` under runs must follow .../runs/<task>/<date_dir>/..., got: {ckpt_path}"
            )
        task_name = parts[runs_idx + 1]
        date_dir = parts[runs_idx + 2]
        if task_name == "" or date_dir == "":
            raise ValueError(
                f"`ckpt` under runs must follow .../runs/<task>/<date_dir>/..., got: {ckpt_path}"
            )
        return f"{task_name}_{date_dir}"
    return ckpt_path.stem


def _is_blocked_override(raw_override: str) -> bool:
    key = raw_override.split("=", 1)[0].lstrip("+~")
    if key in {
        "ckpt",
        "gpu_id",
        "EVALUATION.task_name",
        "EVALUATION.task_config",
        "EVALUATION.output_dir",
    }:
        return True
    return key.startswith("MULTIRUN.") or key.startswith("hydra.")


def _collect_worker_overrides() -> list[str]:
    return [ov for ov in HydraConfig.get().overrides.task if not _is_blocked_override(ov)]


def _load_all_tasks(robotwin_root: Path) -> list[str]:
    task_file = robotwin_root / "task_config" / "_eval_step_limit.yml"
    if not task_file.exists():
        raise FileNotFoundError(f"Task list file not found: {task_file}")
    with task_file.open("r", encoding="utf-8") as f:
        task_map = yaml.safe_load(f)
    if not isinstance(task_map, dict) or len(task_map) == 0:
        raise ValueError(f"Invalid task map in: {task_file}")
    tasks = list(task_map.keys())
    # Keep original order and remove duplicates.
    seen = set()
    dedup_tasks: list[str] = []
    for task in tasks:
        if task in seen:
            continue
        seen.add(task)
        dedup_tasks.append(task)
    return dedup_tasks


def _parse_success_rate(result_file: Path) -> float:
    if not result_file.exists():
        raise FileNotFoundError(f"Result file not found: {result_file}")
    text = result_file.read_text(encoding="utf-8")
    last_value: float | None = None
    for line in text.splitlines():
        stripped = line.strip()
        if stripped == "":
            continue
        try:
            last_value = float(stripped)
        except ValueError:
            continue
    if last_value is None:
        raise ValueError(f"Failed to parse success rate from: {result_file}")
    return last_value


def _phase_result_filename(phase: str) -> str:
    if phase == "clean":
        return "_result_clean.txt"
    if phase == "random":
        return "_result_random.txt"
    raise ValueError(f"Unsupported phase: {phase}")


def _mean_or_none(values: list[float | None]) -> float | None:
    valid = [v for v in values if v is not None]
    if len(valid) == 0:
        return None
    return float(sum(valid) / len(valid))


def _to_jsonable(value: float | None) -> float | None:
    if value is None:
        return None
    return float(value)


def _atomic_replace_text(path: Path, text: str) -> None:
    tmp_path = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    try:
        tmp_path.write_text(text, encoding="utf-8")
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def _format_duration(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "unknown"
    total_seconds = int(round(seconds))
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


@dataclass
class RunningState:
    task_name: str
    gpu_id: int
    phase: str  # "clean" | "random"
    attempt: int
    process: subprocess.Popen[str]
    log_path: Path


@hydra.main(version_base="1.3", config_path="../../configs", config_name="sim_robotwin.yaml")
def main(cfg: DictConfig):
    if cfg.ckpt is None:
        raise ValueError("`ckpt` must not be None.")
    if not SINGLE_ENTRY.exists():
        raise FileNotFoundError(f"Single evaluation entry not found: {SINGLE_ENTRY}")

    ckpt_path = _resolve_path(str(cfg.ckpt), base=PROJECT_ROOT)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    ckpt_tag = _resolve_ckpt_tag(ckpt_path)

    robotwin_root = _resolve_path(str(cfg.EVALUATION.robotwin_root), base=PROJECT_ROOT)
    if not robotwin_root.exists():
        raise FileNotFoundError(f"RoboTwin root not found: {robotwin_root}")

    num_gpus = int(cfg.MULTIRUN.num_gpus)
    if num_gpus <= 0:
        raise ValueError("`MULTIRUN.num_gpus` must be > 0.")
    max_tasks_per_gpu = int(cfg.MULTIRUN.max_tasks_per_gpu)
    if max_tasks_per_gpu <= 0:
        raise ValueError("`MULTIRUN.max_tasks_per_gpu` must be > 0.")
    max_retries = int(cfg.MULTIRUN.get("max_retries", 2))
    if max_retries < 0:
        raise ValueError("`MULTIRUN.max_retries` must be >= 0.")
    gpu_ids = list(range(num_gpus))

    output_dir = _resolve_path(str(cfg.EVALUATION.output_dir), base=PROJECT_ROOT)
    run_ts = output_dir.name
    if run_ts == "":
        raise ValueError(f"Invalid EVALUATION.output_dir (missing run_ts): {output_dir}")
    run_output_dir = PROJECT_ROOT / "evaluate_results" / "robotwin" / ckpt_tag / run_ts
    run_output_dir.mkdir(parents=True, exist_ok=True)
    worker_logs_dir = run_output_dir / "worker_logs"
    worker_logs_dir.mkdir(parents=True, exist_ok=True)

    task_shard_count = int(cfg.MULTIRUN.get("task_shard_count", 1))
    task_shard_index = int(cfg.MULTIRUN.get("task_shard_index", 0))
    if task_shard_count <= 0:
        raise ValueError("`MULTIRUN.task_shard_count` must be > 0.")
    if not 0 <= task_shard_index < task_shard_count:
        raise ValueError(
            "`MULTIRUN.task_shard_index` must satisfy "
            f"0 <= index < count, got index={task_shard_index}, count={task_shard_count}."
        )

    shard_suffix = (
        ""
        if task_shard_count == 1
        else f".shard{task_shard_index}-of-{task_shard_count}"
    )
    manager_log = run_output_dir / f"manager{shard_suffix}.log"
    failed_tasks_file = run_output_dir / f"failed_tasks{shard_suffix}.txt"
    summary_csv = run_output_dir / f"summary{shard_suffix}.csv"
    summary_json = run_output_dir / f"summary{shard_suffix}.json"

    task_name_cfg = cfg.EVALUATION.task_name
    if task_name_cfg is None or str(task_name_cfg).strip() == "":
        tasks = _load_all_tasks(robotwin_root)
    else:
        tasks = [str(task_name_cfg)]

    if task_shard_count > 1:
        tasks = [
            task
            for task_index, task in enumerate(tasks)
            if task_index % task_shard_count == task_shard_index
        ]

    extra_overrides = _collect_worker_overrides()

    task_rates: dict[str, dict[str, float | None]] = {
        task: {"clean": None, "random": None} for task in tasks
    }
    failed_records: list[dict[str, Any]] = []
    pending_phases: deque[tuple[str, str]] = deque()
    running_states: list[RunningState] = []
    attempt_counts: dict[tuple[str, str], int] = {}

    phase_to_task_config = {
        "clean": "demo_clean",
        "random": "demo_randomized",
    }

    def log(msg: str, *, console: bool = True) -> None:
        line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}"
        if console:
            print(line, flush=True)
        with manager_log.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()

    def build_cmd(*, task_name: str, gpu_id: int, phase: str) -> list[str]:
        task_config = phase_to_task_config[phase]
        cmd = [
            sys.executable,
            str(SINGLE_ENTRY),
            f"ckpt={str(ckpt_path)}",
            f"gpu_id={gpu_id}",
            f"EVALUATION.task_name={task_name}",
            f"EVALUATION.task_config={task_config}",
            f"EVALUATION.output_dir={str(output_dir)}",
        ]
        cmd.extend(extra_overrides)
        return cmd

    for task_name in tasks:
        clean_file = run_output_dir / task_name / _phase_result_filename("clean")
        random_file = run_output_dir / task_name / _phase_result_filename("random")
        if clean_file.exists():
            task_rates[task_name]["clean"] = _parse_success_rate(clean_file)
        if random_file.exists():
            task_rates[task_name]["random"] = _parse_success_rate(random_file)

        if task_rates[task_name]["clean"] is None:
            pending_phases.append((task_name, "clean"))
        elif task_rates[task_name]["random"] is None:
            pending_phases.append((task_name, "random"))

    def launch_phase(task_name: str, gpu_id: int, phase: str) -> RunningState:
        phase_key = (task_name, phase)
        attempt = attempt_counts.get(phase_key, 0) + 1
        attempt_counts[phase_key] = attempt
        cmd = build_cmd(task_name=task_name, gpu_id=gpu_id, phase=phase)
        worker_log = worker_logs_dir / (
            f"{task_name}.{phase}.gpu{gpu_id}.attempt{attempt}.log"
        )
        log(
            f"launch task={task_name} phase={phase} gpu={gpu_id} attempt={attempt} "
            f"worker_log={worker_log}",
        )
        log(
            f"worker command task={task_name} phase={phase}: "
            f"{shlex.join(cmd)}",
            console=False,
        )
        with worker_log.open("a", encoding="utf-8") as worker_log_file:
            process = subprocess.Popen(
                cmd,
                cwd=str(PROJECT_ROOT),
                stdout=worker_log_file,
                stderr=subprocess.STDOUT,
                text=True,
            )
        return RunningState(
            task_name=task_name,
            gpu_id=gpu_id,
            phase=phase,
            attempt=attempt,
            process=process,
            log_path=worker_log,
        )

    def terminate_all_running() -> None:
        for state in list(running_states):
            if state.process.poll() is not None:
                continue
            log(f"terminating task={state.task_name} phase={state.phase} gpu={state.gpu_id}")
            state.process.terminate()
        deadline = time.time() + TERMINATE_TIMEOUT_SEC
        for state in list(running_states):
            if state.process.poll() is not None:
                continue
            remaining = max(0.0, deadline - time.time())
            try:
                state.process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                log(f"killing task={state.task_name} phase={state.phase} gpu={state.gpu_id}")
                state.process.kill()
                state.process.wait()

    def gpu_running_count(gpu_id: int) -> int:
        count = 0
        for state in running_states:
            if state.gpu_id != gpu_id:
                continue
            if state.process.poll() is None:
                count += 1
        return count

    def try_launch_pending(gpu_id: int) -> None:
        while len(pending_phases) > 0 and gpu_running_count(gpu_id) < max_tasks_per_gpu:
            task_name, phase = pending_phases.popleft()
            running_states.append(launch_phase(task_name=task_name, gpu_id=gpu_id, phase=phase))

    def write_outputs() -> None:
        clean_mean = _mean_or_none([task_rates[t]["clean"] for t in tasks])
        random_mean = _mean_or_none([task_rates[t]["random"] for t in tasks])

        summary_csv_tmp = summary_csv.with_name(
            f".{summary_csv.name}.tmp.{os.getpid()}"
        )
        with summary_csv_tmp.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["task_name", "clean_success_rate", "random_success_rate"])
            for task in tasks:
                writer.writerow(
                    [
                        task,
                        task_rates[task]["clean"],
                        task_rates[task]["random"],
                    ]
                )
            writer.writerow(["__overall__", clean_mean, random_mean])
        os.replace(summary_csv_tmp, summary_csv)

        payload = {
            "per_task": [
                {
                    "task_name": task,
                    "clean_success_rate": _to_jsonable(task_rates[task]["clean"]),
                    "random_success_rate": _to_jsonable(task_rates[task]["random"]),
                }
                for task in tasks
            ],
            "overall": {
                "clean_mean_success_rate": _to_jsonable(clean_mean),
                "random_mean_success_rate": _to_jsonable(random_mean),
            },
        }
        _atomic_replace_text(
            summary_json,
            json.dumps(payload, ensure_ascii=False, indent=2),
        )

        failed_lines = []
        for rec in failed_records:
            failed_lines.append(
                f"{rec['task_name']},{rec['phase']},gpu={rec['gpu_id']},"
                f"return_code={rec['return_code']},reason={rec['reason']}\n"
            )
        _atomic_replace_text(
            failed_tasks_file,
            "".join(failed_lines),
        )

    initial_accounted_phases = sum(
        rate is not None
        for task in tasks
        for rate in task_rates[task].values()
    )
    manager_start_time = time.time()
    last_progress_log_time = 0.0

    def log_progress(*, force: bool = False) -> None:
        nonlocal last_progress_log_time
        now = time.time()
        if not force and now - last_progress_log_time < 60.0:
            return
        last_progress_log_time = now

        clean_values = [
            task_rates[task]["clean"]
            for task in tasks
            if task_rates[task]["clean"] is not None
        ]
        random_values = [
            task_rates[task]["random"]
            for task in tasks
            if task_rates[task]["random"] is not None
        ]
        completed_phases = len(clean_values) + len(random_values)
        failed_phases = len(failed_records)
        accounted_phases = completed_phases + failed_phases
        total_phases = 2 * len(tasks)
        complete_tasks = sum(
            task_rates[task]["clean"] is not None
            and task_rates[task]["random"] is not None
            for task in tasks
        )

        elapsed = max(now - manager_start_time, 0.0)
        newly_accounted = max(accounted_phases - initial_accounted_phases, 0)
        remaining = max(total_phases - accounted_phases, 0)
        eta_seconds = (
            elapsed * remaining / newly_accounted
            if newly_accounted > 0
            else None
        )
        clean_mean = _mean_or_none(clean_values)
        random_mean = _mean_or_none(random_values)
        clean_text = (
            f"{clean_mean * 100:.2f}%({len(clean_values)})"
            if clean_mean is not None
            else "n/a(0)"
        )
        random_text = (
            f"{random_mean * 100:.2f}%({len(random_values)})"
            if random_mean is not None
            else "n/a(0)"
        )
        log(
            f"progress phases={accounted_phases}/{total_phases} "
            f"tasks_complete={complete_tasks}/{len(tasks)} "
            f"running={len(running_states)} pending={len(pending_phases)} "
            f"failed={failed_phases} clean={clean_text} random={random_text} "
            f"elapsed={_format_duration(elapsed)} eta={_format_duration(eta_seconds)}"
        )

    log(
        f"manager start tasks={len(tasks)} gpu_ids={gpu_ids} "
        f"max_tasks_per_gpu={max_tasks_per_gpu} max_retries={max_retries} "
        f"task_shard={task_shard_index}/{task_shard_count} "
        f"pending_phases={len(pending_phases)} output_dir={run_output_dir}"
    )
    write_outputs()
    log_progress(force=True)

    # Launch initial tasks for each GPU up to capacity.
    for gpu_id in gpu_ids:
        try_launch_pending(gpu_id)

    while len(running_states) > 0:
        progressed = False
        for state in list(running_states):
            gpu_id = state.gpu_id
            return_code = state.process.poll()
            if return_code is None:
                continue
            progressed = True
            running_states.remove(state)

            if return_code != 0:
                failure_message = (
                    f"worker failed: task={state.task_name}, phase={state.phase}, "
                    f"gpu={gpu_id}, attempt={state.attempt}, return_code={return_code}"
                )
                log(failure_message)
                if state.attempt <= max_retries:
                    running_states.append(
                        launch_phase(
                            task_name=state.task_name,
                            gpu_id=gpu_id,
                            phase=state.phase,
                        )
                    )
                else:
                    failed_records.append(
                        {
                            "task_name": state.task_name,
                            "phase": state.phase,
                            "gpu_id": gpu_id,
                            "return_code": return_code,
                            "reason": "process_failed",
                        }
                    )
                    if state.phase == "clean":
                        failed_records.append(
                            {
                                "task_name": state.task_name,
                                "phase": "random",
                                "gpu_id": gpu_id,
                                "return_code": return_code,
                                "reason": "skipped_after_clean_failure",
                            }
                        )
                    write_outputs()
                    log_progress(force=True)
                    try_launch_pending(gpu_id)
                continue

            result_file = run_output_dir / state.task_name / _phase_result_filename(state.phase)
            try:
                success_rate = _parse_success_rate(result_file)
            except Exception as exc:
                failure_message = (
                    f"result parse failed: task={state.task_name}, phase={state.phase}, "
                    f"gpu={gpu_id}, attempt={state.attempt}, error={repr(exc)}"
                )
                log(failure_message)
                if state.attempt <= max_retries:
                    running_states.append(
                        launch_phase(
                            task_name=state.task_name,
                            gpu_id=gpu_id,
                            phase=state.phase,
                        )
                    )
                else:
                    failed_records.append(
                        {
                            "task_name": state.task_name,
                            "phase": state.phase,
                            "gpu_id": gpu_id,
                            "return_code": return_code,
                            "reason": "result_parse_failed",
                        }
                    )
                    if state.phase == "clean":
                        failed_records.append(
                            {
                                "task_name": state.task_name,
                                "phase": "random",
                                "gpu_id": gpu_id,
                                "return_code": return_code,
                                "reason": "skipped_after_clean_failure",
                            }
                        )
                    write_outputs()
                    log_progress(force=True)
                    try_launch_pending(gpu_id)
                continue

            task_rates[state.task_name][state.phase] = success_rate
            write_outputs()
            log(
                f"done task={state.task_name} phase={state.phase} gpu={gpu_id} "
                f"success_rate={success_rate:.4f}"
            )
            log_progress(force=True)

            if state.phase == "clean":
                running_states.append(launch_phase(
                    task_name=state.task_name,
                    gpu_id=gpu_id,
                    phase="random",
                ))
                continue

            try_launch_pending(gpu_id)

        if not progressed:
            log_progress()
            time.sleep(POLL_INTERVAL_SEC)

    write_outputs()
    log(f"summary saved: {summary_csv} and {summary_json}")

    if failed_records:
        raise RuntimeError(
            f"{len(failed_records)} phase(s) failed after retries; see {failed_tasks_file}"
        )

    log("manager finished successfully")


if __name__ == "__main__":
    main()
