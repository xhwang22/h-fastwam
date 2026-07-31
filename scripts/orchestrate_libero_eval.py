#!/usr/bin/env python3
import json
import re
import shlex
import subprocess
import time
from pathlib import Path

import yaml

REPO = Path("/apdcephfs_csgl/share_306089109/shaunxhwang/h-fastwam")
STATE_PATH = REPO / "runs/libero_eval_orchestrator/state.json"
LOG_PATH = REPO / "runs/libero_eval_orchestrator/orchestrator.log"
POLL_SECONDS = 60

TRAININGS = [
    {
        "name": "siglip2",
        "run_dir": REPO / "runs/libero_hfastwam/libero_hfastwam_8card_small_siglip2_ds",
        "output_tag": "libero_full_siglip2_step21700",
        "todo_id": "eval-libero-siglip2",
        "max_tasks_per_gpu": 2,
    },
    {
        "name": "vjepa21_predictor",
        "run_dir": REPO
        / "runs/libero_hfastwam/libero_hfastwam_8card_small_vjepa21_predictor_ds",
        "output_tag": "libero_full_vjepa21_predictor_step21700",
        "todo_id": "eval-libero-vjepa21-predictor",
        "max_tasks_per_gpu": 2,
    },
    {
        "name": "vjepa21",
        "run_dir": REPO / "runs/libero_hfastwam/libero_hfastwam_8card_small_vjepa21_ds",
        "output_tag": "libero_full_vjepa21_step21700",
        "todo_id": "eval-libero-vjepa21",
        "max_tasks_per_gpu": 3,
    },
]

NODES = [
    {
        "ip": "28.195.145.217",
        "robotwin_root": "/tmp/hfastwam_eval_20260724_vjepa",
        "robotwin_pid": "/tmp/hfastwam_eval_20260724_vjepa/manager_w3.pid",
        "robotwin_results": REPO
        / "evaluate_results/robotwin/vjepa_step_029355/cluster_full_vjepa_w3_20260724",
        "robotwin_env": {
            "TRAIN_CONFIG": "/tmp/hfastwam_eval_20260724_vjepa/train_config.yaml",
            "CKPT": "/tmp/hfastwam_eval_20260724_vjepa/vjepa_step_029355.pt",
            "OUTPUT_TAG": "cluster_full_vjepa_w3_20260724",
            "ROBOTWIN_ROOT": "/tmp/hfastwam_eval_20260724_vjepa/RoboTwin",
            "MODEL_BASE_PATH": "/tmp/hfastwam_eval_20260724_vjepa/model_cache",
            "HF_HOME": "/tmp/hfastwam_eval_20260724_vjepa/hf_cache",
            "TORCH_HOME": "/tmp/hfastwam_eval_20260724_vjepa/torch_hub",
            "SWIFTSHADER_ICD": "/tmp/hfastwam_eval_20260724_vjepa/swiftshader/vk_swiftshader_icd.json",
        },
    },
    {
        "ip": "28.195.144.253",
        "robotwin_root": "/tmp/hfastwam_eval_20260724_dino",
        "robotwin_pid": "/tmp/hfastwam_eval_20260724_dino/manager_w3.pid",
        "robotwin_results": REPO
        / "evaluate_results/robotwin/dino_step_029355/cluster_full_dino_w3_20260724",
        "robotwin_env": {
            "TRAIN_CONFIG": "/tmp/hfastwam_eval_20260724_dino/train_config.yaml",
            "CKPT": "/tmp/hfastwam_eval_20260724_dino/dino_step_029355.pt",
            "OUTPUT_TAG": "cluster_full_dino_w3_20260724",
            "ROBOTWIN_ROOT": "/tmp/hfastwam_eval_20260724_dino/RoboTwin",
            "MODEL_BASE_PATH": "/tmp/hfastwam_eval_20260724_dino/model_cache",
            "HF_HOME": "/tmp/hfastwam_eval_20260724_dino/hf_cache",
            "TORCH_HOME": str(REPO / "checkpoints/torch_hub"),
            "SWIFTSHADER_ICD": "/tmp/hfastwam_eval_20260724_dino/swiftshader/vk_swiftshader_icd.json",
        },
    },
    {
        "ip": "28.195.147.172",
        "robotwin_root": "/tmp/hfastwam_eval_20260724_fastwam",
        "robotwin_pid": "/tmp/hfastwam_eval_20260724_fastwam/manager_w3.pid",
        "robotwin_results": REPO
        / "evaluate_results/robotwin/fastwam_step_029355/cluster_full_fastwam_w3_20260724",
        "robotwin_env": {
            "TRAIN_CONFIG": "/tmp/hfastwam_eval_20260724_fastwam/train_config.yaml",
            "CKPT": "/tmp/hfastwam_eval_20260724_fastwam/fastwam_step_029355.pt",
            "OUTPUT_TAG": "cluster_full_fastwam_w3_20260724",
            "ROBOTWIN_ROOT": "/tmp/hfastwam_eval_20260724_fastwam/RoboTwin",
            "MODEL_BASE_PATH": "/tmp/hfastwam_eval_20260724_fastwam/model_cache",
            "HF_HOME": "/tmp/hfastwam_eval_20260724_fastwam/hf_cache",
            "TORCH_HOME": str(REPO / "checkpoints/torch_hub"),
            "SWIFTSHADER_ICD": "/tmp/hfastwam_eval_20260724_fastwam/swiftshader/vk_swiftshader_icd.json",
        },
    },
]


def log(message: str):
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
    print(line, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def ssh(ip: str, command: str, check: bool = True):
    return subprocess.run(
        ["ssh", "-o", "BatchMode=yes", ip, command],
        check=check,
        text=True,
        capture_output=True,
    )


def load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    return {"jobs": {}, "nodes": {}, "robotwin_resumed": False}


def save_state(state: dict):
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")


def max_steps(run_dir: Path) -> int:
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text(encoding="utf-8"))
    configured = cfg.get("max_steps")
    if configured is not None:
        return int(configured)
    save_every = int(cfg.get("save_every", 0) or 0)
    checkpoint_steps = []
    for path in (run_dir / "checkpoints/weights").glob("step_*.pt"):
        match = re.match(r"step_(\d+)\.pt$", path.name)
        if match:
            checkpoint_steps.append(int(match.group(1)))
    if save_every > 0:
        final_candidates = [step for step in checkpoint_steps if step % save_every != 0]
        if final_candidates:
            return max(final_candidates)
    text = (run_dir / "train.log.rank0").read_text(
        encoding="utf-8", errors="replace"
    )[-200_000:]
    matches = re.findall(r"step=\d+/(\d+)", text)
    if not matches:
        raise ValueError(f"Could not determine max_steps from {run_dir}")
    return int(matches[-1])


def training_complete(training: dict) -> tuple[bool, Path]:
    run_dir = training["run_dir"]
    target_step = max_steps(run_dir)
    ckpt = run_dir / f"checkpoints/weights/step_{target_step:06d}.pt"
    return ckpt.is_file() and ckpt.stat().st_size > 0, ckpt


def eval_complete(output_tag: str) -> bool:
    output_dir = REPO / "evaluate_results" / output_tag
    result_count = len(list(output_dir.glob("*/gpu*_task*_results.json")))
    return result_count == 40


def remote_alive(ip: str, pid_file: str) -> bool:
    result = ssh(
        ip,
        f"test -f {shlex.quote(pid_file)} && "
        f"kill -0 $(cat {shlex.quote(pid_file)}) 2>/dev/null",
        check=False,
    )
    return result.returncode == 0


def robotwin_complete(node: dict) -> bool:
    return len(list(node["robotwin_results"].glob("*/_result_*.txt"))) >= 100


def stop_robotwin(node: dict):
    if robotwin_complete(node):
        return False
    escaped_root = node["robotwin_root"].replace("/", "\\/")
    command = f"""
set +e
if test -f {shlex.quote(node['robotwin_pid'])}; then
  PID=$(cat {shlex.quote(node['robotwin_pid'])})
  PGID=$(ps -o pgid= -p "$PID" | tr -d ' ')
  test -n "$PGID" && kill -TERM -- -"$PGID"
fi
sleep 8
for p in $(ps -eo pid=,args= | awk '/{escaped_root}/ && !/awk/ {{print $1}}'); do
  kill -KILL "$p" 2>/dev/null
done
"""
    ssh(node["ip"], command)
    return True


def launch_libero(training: dict, node: dict, ckpt: Path):
    local_root = f"/tmp/libero_eval_{training['name']}"
    pid_file = f"{local_root}/eval.pid"
    log_file = REPO / f"runs/libero_eval_orchestrator/{training['name']}.log"
    command = f"""
set -e
rm -rf {shlex.quote(local_root)}
mkdir -p {shlex.quote(local_root)}
cp {shlex.quote(str(ckpt))} {shlex.quote(local_root + '/checkpoint.pt')}
cp {shlex.quote(str(training['run_dir'] / 'config.yaml'))} {shlex.quote(local_root + '/config.yaml')}
cp {shlex.quote(str(training['run_dir'] / 'dataset_stats.json'))} {shlex.quote(local_root + '/dataset_stats.json')}
cd {shlex.quote(str(REPO))}
nohup setsid env \
  RUN_DIR={shlex.quote(local_root)} \
  CKPT={shlex.quote(local_root + '/checkpoint.pt')} \
  OUTPUT_TAG={shlex.quote(training['output_tag'])} \
  NUM_TRIALS=50 MAX_TASKS_PER_GPU={int(training['max_tasks_per_gpu'])} \
  bash scripts/run_libero_full_eval_from_run.sh \
  >{shlex.quote(str(log_file))} 2>&1 </dev/null &
echo $! > {shlex.quote(pid_file)}
"""
    ssh(node["ip"], command)
    return pid_file


def resume_robotwin(node: dict):
    if robotwin_complete(node):
        return
    env_args = " ".join(
        f"{key}={shlex.quote(value)}" for key, value in node["robotwin_env"].items()
    )
    log_file = REPO / (
        "runs/libero_eval_orchestrator/"
        f"resume_robotwin_{node['ip'].replace('.', '_')}.log"
    )
    command = f"""
cd {shlex.quote(str(REPO))}
nohup setsid env {env_args} WORKERS_PER_GPU=3 NUM_GPUS=8 EVAL_NUM_EPISODES=100 \
  bash scripts/run_robotwin_full_eval_node.sh \
  >{shlex.quote(str(log_file))} 2>&1 </dev/null &
echo $! > {shlex.quote(node['robotwin_pid'])}
"""
    ssh(node["ip"], command)


def main():
    state = load_state()
    while True:
        for training in TRAININGS:
            job = state["jobs"].get(training["name"], {})
            if job.get("status") in {"running", "done"}:
                if job["status"] == "running" and eval_complete(training["output_tag"]):
                    job["status"] = "done"
                    state["jobs"][training["name"]] = job
                    log(f"LIBERO eval completed: {training['name']}")
                    save_state(state)
                elif job["status"] == "running" and not remote_alive(
                    job["node"], job["pid_file"]
                ):
                    node = next(item for item in NODES if item["ip"] == job["node"])
                    ckpt = Path(job["checkpoint"])
                    job["pid_file"] = launch_libero(training, node, ckpt)
                    state["jobs"][training["name"]] = job
                    log(f"Restarted interrupted LIBERO eval: {training['name']}")
                    save_state(state)
                continue

            complete, ckpt = training_complete(training)
            if not complete:
                continue

            assigned_ips = {
                value["node"]
                for value in state["jobs"].values()
                if value.get("status") == "running"
            }
            node = next((item for item in NODES if item["ip"] not in assigned_ips), None)
            if node is None:
                continue

            paused = stop_robotwin(node)
            pid_file = launch_libero(training, node, ckpt)
            state["jobs"][training["name"]] = {
                "status": "running",
                "node": node["ip"],
                "pid_file": pid_file,
                "checkpoint": str(ckpt),
                "robotwin_paused": paused,
            }
            log(
                f"Started LIBERO eval {training['name']} on {node['ip']} "
                f"(robotwin_paused={paused})"
            )
            save_state(state)

        all_done = all(
            state["jobs"].get(training["name"], {}).get("status") == "done"
            for training in TRAININGS
        )
        if all_done and not state.get("robotwin_resumed", False):
            for node in NODES:
                job_for_node = next(
                    (
                        value
                        for value in state["jobs"].values()
                        if value.get("node") == node["ip"]
                    ),
                    None,
                )
                if job_for_node and job_for_node.get("robotwin_paused"):
                    resume_robotwin(node)
            state["robotwin_resumed"] = True
            save_state(state)
            log("All LIBERO evals completed; paused RoboTwin evaluations resumed.")
            return

        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
