#!/usr/bin/env bash
# Helper to broadcast-launch a per-node script on every host of a Tencent
# Taiji job from the chief node.
#
# What it does
# ------------
# Reads the OpenMPI hostfile that Taiji places at /etc/taiji/hostfile (one
# `host slots=N` line per node), then SSH-runs the same launcher script on
# every host. The remote shell inherits the platform's per-container env
# (INDEX, CHIEF_IP, ...), so each node knows its own NODE_RANK.
#
# This script is OPTIONAL — you can also just SSH into each node yourself
# and run the inner launcher manually. Useful when you have many nodes.
#
# Usage:
#   bash scripts/launch_all_nodes.sh scripts/run_libero_vjepa2ac_predictor_multinode.sh
#
#   # Pass extra env to the remote launcher:
#   REMOTE_ENV="RUN_NAME=mybigrun FOREGROUND=0" \
#       bash scripts/launch_all_nodes.sh scripts/run_libero_vjepa2ac_predictor_multinode.sh
#
#   # Use a custom hostfile (default /etc/taiji/hostfile):
#   HOSTFILE=/path/to/hosts \
#       bash scripts/launch_all_nodes.sh scripts/run_libero_vjepa2ac_predictor_multinode.sh
#
# Caveats
# -------
# 1. Run this from the CHIEF node only. (You can also run it from any node
#    that has SSH access to all of them — the chief is just convention.)
# 2. The hostfile is read literally. Lines like `28.216.19.99 slots=8` are
#    parsed; only the first column (the host) is used here.
# 3. We do not capture per-node logs centrally. Each node's launcher writes
#    to its own ${LOG_DIR}/train.log.rank${NODE_RANK}. Inspect them via
#    `ssh <host> tail -f <path>`.
# 4. SSH is run with `-o StrictHostKeyChecking=no` because Taiji nodes share
#    keys; first-time fingerprints would otherwise prompt and hang.

set -euo pipefail

INNER_SCRIPT="${1:-}"
if [[ -z "${INNER_SCRIPT}" ]]; then
  echo "Usage: $0 <path-to-launcher-script> [-- <extra args forwarded to launcher>]" >&2
  exit 1
fi
shift || true

if [[ ! -f "${INNER_SCRIPT}" ]]; then
  echo "[launch_all] ERROR: launcher script not found: ${INNER_SCRIPT}" >&2
  exit 1
fi

HOSTFILE="${HOSTFILE:-/etc/taiji/hostfile}"
if [[ ! -f "${HOSTFILE}" ]]; then
  echo "[launch_all] ERROR: hostfile not found: ${HOSTFILE}" >&2
  echo "[launch_all] Set HOSTFILE=<path> to override." >&2
  exit 1
fi

# Convert relative inner-script path to absolute so it works on remote nodes
# (assumes the FastWAM repo is on a shared filesystem like apdcephfs).
INNER_SCRIPT_ABS="$(cd "$(dirname "${INNER_SCRIPT}")" && pwd)/$(basename "${INNER_SCRIPT}")"

REMOTE_ENV="${REMOTE_ENV:-}"

mapfile -t HOSTS < <(awk 'NF && $1 !~ /^#/ {print $1}' "${HOSTFILE}")
if [[ "${#HOSTS[@]}" -eq 0 ]]; then
  echo "[launch_all] ERROR: no hosts parsed from ${HOSTFILE}" >&2
  exit 1
fi

echo "[launch_all] hostfile     = ${HOSTFILE}"
echo "[launch_all] hosts        = ${HOSTS[*]}"
echo "[launch_all] inner script = ${INNER_SCRIPT_ABS}"
echo "[launch_all] remote env   = ${REMOTE_ENV}"
echo "[launch_all] forwarded args = $*"

# Loop sequentially so any SSH failure is visible up front. Each remote
# launcher is itself non-blocking (FOREGROUND=0 default), so kicking them off
# one after another adds only seconds.
for host in "${HOSTS[@]}"; do
  echo "[launch_all] -> ${host}"
  # Quote args so they survive remote-shell parsing.
  printf -v fwd '%q ' "$@"
  ssh -o StrictHostKeyChecking=no -o ConnectTimeout=10 "${host}" \
      "${REMOTE_ENV} bash ${INNER_SCRIPT_ABS} ${fwd}" \
    || { echo "[launch_all] ERROR: launch failed on ${host}" >&2; exit 1; }
done

echo "[launch_all] Done. Each node is now running its own copy."
echo "[launch_all] Tail rank-0 log on the chief, e.g.:"
echo "             tail -f \$(ls -t runs/libero_vjepa2ac_predictor/*/train.log.rank0 | head -1)"
