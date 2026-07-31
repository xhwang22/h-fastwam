#!/usr/bin/env python3
import json
import shlex
import subprocess
import time
from pathlib import Path

REPO = Path("/apdcephfs_csgl/share_306089109/shaunxhwang/h-fastwam")
STATE_PATH = REPO / "runs/robotwin_eval_cluster/rebalance_state.json"
POLL_SECONDS = 300

PAIRS = [
    {
        "name": "vjepa_to_fastwam",
        "source_results": REPO
        / "evaluate_results/robotwin/vjepa_step_029355/cluster_full_vjepa_w3_20260724",
        "source_ip": "28.195.145.217",
        "source_pid": "/tmp/hfastwam_eval_20260724_vjepa/manager_w3.pid",
        "helper_root": "/tmp/hfastwam_eval_20260724_vjepa",
        "target_results": REPO
        / "evaluate_results/robotwin/fastwam_step_029355/cluster_full_fastwam_w3_20260724",
        "target_ip": "28.195.147.172",
        "target_pid": "/tmp/hfastwam_eval_20260724_fastwam/manager_w3.pid",
        "target_root": "/tmp/hfastwam_eval_20260724_fastwam",
        "checkpoint_source": REPO
        / "runs/robotwin_fastwam/robotwin_fastwam_8card_small_ds/checkpoints/weights/step_029355.pt",
        "config_source": REPO
        / "runs/robotwin_fastwam/robotwin_fastwam_8card_small_ds/config.yaml",
        "checkpoint_name": "fastwam_step_029355.pt",
        "output_tag": "cluster_full_fastwam_w3_20260724",
        "target_log_prefix": "fastwam_rebalanced",
    },
    {
        "name": "dino_to_hfastwam",
        "source_results": REPO
        / "evaluate_results/robotwin/dino_step_029355/cluster_full_dino_w3_20260724",
        "source_ip": "28.195.144.253",
        "source_pid": "/tmp/hfastwam_eval_20260724_dino/manager_w3.pid",
        "helper_root": "/tmp/hfastwam_eval_20260724_dino",
        "target_results": REPO
        / "evaluate_results/robotwin/hfastwam_step_029355/cluster_full_hfastwam_w3_20260724",
        "target_ip": "28.195.145.141",
        "target_pid": "/tmp/hfastwam_eval_20260724_hfastwam/manager_w3.pid",
        "target_root": "/tmp/hfastwam_eval_20260724_hfastwam",
        "checkpoint_source": REPO
        / "runs/robotwin_hfastwam/robotwin_hfastwam_8card_small_ds/checkpoints/weights/step_029355.pt",
        "config_source": REPO
        / "runs/robotwin_hfastwam/robotwin_hfastwam_8card_small_ds/config.yaml",
        "checkpoint_name": "hfastwam_step_029355.pt",
        "output_tag": "cluster_full_hfastwam_w3_20260724",
        "target_log_prefix": "hfastwam_rebalanced",
    },
]


def run_ssh(ip: str, command: str, check: bool = True):
    return subprocess.run(
        ["ssh", "-o", "BatchMode=yes", ip, command],
        check=check,
        text=True,
        capture_output=True,
    )


def phase_count(path: Path) -> int:
    return len(list(path.glob("*/_result_*.txt")))


def remote_alive(ip: str, pid_file: str) -> bool:
    result = run_ssh(
        ip,
        f"test -f {shlex.quote(pid_file)} && "
        f"kill -0 $(cat {shlex.quote(pid_file)}) 2>/dev/null",
        check=False,
    )
    return result.returncode == 0


def stop_run(ip: str, pid_file: str, root: str):
    command = f"""
set +e
if test -f {shlex.quote(pid_file)}; then
  PID=$(cat {shlex.quote(pid_file)})
  PGID=$(ps -o pgid= -p "$PID" | tr -d ' ')
  test -n "$PGID" && kill -TERM -- -"$PGID"
fi
sleep 8
for p in $(ps -eo pid=,args= | awk '/{root.replace("/", "\\/")}/ && !/awk/ {{print $1}}'); do
  kill -KILL "$p" 2>/dev/null
done
"""
    run_ssh(ip, command)


def launch_shard(
    ip: str,
    root: str,
    checkpoint_name: str,
    output_tag: str,
    shard_index: int,
    log_prefix: str,
):
    pid_file = f"{root}/{log_prefix}_shard{shard_index}.pid"
    log_file = REPO / f"runs/robotwin_eval_cluster/{log_prefix}_shard{shard_index}.log"
    torch_home = f"{root}/torch_hub" if Path(root).name.endswith("vjepa") else f"{REPO}/checkpoints/torch_hub"
    command = f"""
set -e
cd {shlex.quote(str(REPO))}
nohup setsid env \
  TRAIN_CONFIG={shlex.quote(root + "/helper_train_config.yaml")} \
  CKPT={shlex.quote(root + "/" + checkpoint_name)} \
  OUTPUT_TAG={shlex.quote(output_tag)} \
  ROBOTWIN_ROOT={shlex.quote(root + "/RoboTwin")} \
  MODEL_BASE_PATH={shlex.quote(root + "/model_cache")} \
  HF_HOME={shlex.quote(root + "/hf_cache")} \
  TORCH_HOME={shlex.quote(torch_home)} \
  SWIFTSHADER_ICD={shlex.quote(root + "/swiftshader/vk_swiftshader_icd.json")} \
  WORKERS_PER_GPU=3 NUM_GPUS=8 EVAL_NUM_EPISODES=100 \
  TASK_SHARD_COUNT=2 TASK_SHARD_INDEX={shard_index} \
  bash scripts/run_robotwin_full_eval_node.sh \
  >{shlex.quote(str(log_file))} 2>&1 </dev/null &
echo $! > {shlex.quote(pid_file)}
"""
    run_ssh(ip, command)


def rebalance(pair: dict):
    stop_run(pair["target_ip"], pair["target_pid"], pair["target_root"])
    helper_checkpoint = f'{pair["helper_root"]}/{pair["checkpoint_name"]}'
    copy_command = (
        f"cp {shlex.quote(str(pair['checkpoint_source']))} {shlex.quote(helper_checkpoint)} && "
        f"cp {shlex.quote(str(pair['config_source']))} "
        f"{shlex.quote(pair['helper_root'] + '/helper_train_config.yaml')}"
    )
    run_ssh(pair["source_ip"], copy_command)

    target_config = pair["target_root"] + "/helper_train_config.yaml"
    run_ssh(
        pair["target_ip"],
        f"cp {shlex.quote(str(pair['config_source']))} {shlex.quote(target_config)}",
    )
    launch_shard(
        pair["target_ip"],
        pair["target_root"],
        pair["checkpoint_name"],
        pair["output_tag"],
        0,
        pair["target_log_prefix"],
    )
    launch_shard(
        pair["source_ip"],
        pair["helper_root"],
        pair["checkpoint_name"],
        pair["output_tag"],
        1,
        pair["target_log_prefix"],
    )


def load_state() -> dict:
    if STATE_PATH.exists():
        return json.loads(STATE_PATH.read_text(encoding="utf-8"))
    return {}


def save_state(state: dict):
    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")


def main():
    state = load_state()
    while True:
        changed = False
        for pair in PAIRS:
            if state.get(pair["name"]) == "rebalanced":
                continue
            source_done = phase_count(pair["source_results"]) >= 100
            source_stopped = not remote_alive(pair["source_ip"], pair["source_pid"])
            target_done = phase_count(pair["target_results"]) >= 100
            if target_done:
                state[pair["name"]] = "target_already_done"
                changed = True
            elif source_done and source_stopped:
                rebalance(pair)
                state[pair["name"]] = "rebalanced"
                changed = True
        if changed:
            save_state(state)
        if all(
            state.get(pair["name"]) in {"rebalanced", "target_already_done"}
            for pair in PAIRS
        ):
            return
        time.sleep(POLL_SECONDS)


if __name__ == "__main__":
    main()
