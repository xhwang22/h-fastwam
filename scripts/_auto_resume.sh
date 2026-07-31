#!/usr/bin/env bash
# _auto_resume.sh — auto-resume helper for preemptible (tidal) resources.
#
# Source this file, then call:
#   _compute_resume_override "${LOG_DIR}"
# which populates the global array RESUME_OVERRIDES with a hydra `resume=<dir>`
# argument (or leaves it empty for a fresh run). Append "${RESUME_OVERRIDES[@]}"
# to the torchrun command.
#
# Behaviour (default = auto-resume, so a bare restart `cd <repo> && bash run.sh`
# automatically continues):
#   * FRESH=1            -> always start from scratch (ignore existing ckpts)
#   * RESUME=<path>      -> resume from this explicit state dir / weights file
#   * RESUME unset/auto  -> find the newest valid checkpoints/state/step_* dir
#
# A checkpoint dir is only considered valid if it contains trainer_state.json
# (so a half-written dir from a mid-save preemption is skipped, and we fall back
# to the previous one — this is why we keep >1 checkpoint).
#
# This runs independently on every node; all nodes read the same shared-FS
# LOG_DIR and therefore agree on the same latest checkpoint.

_compute_resume_override() {
  local log_dir="$1"
  RESUME_OVERRIDES=()
  local state_dir="${log_dir}/checkpoints/state"

  if [[ "${FRESH:-0}" == "1" ]]; then
    echo "[auto-resume] FRESH=1 → training from scratch (ignoring existing checkpoints)."
    return
  fi

  # Explicit resume path (not the 'auto'/'1' sentinels) wins.
  if [[ -n "${RESUME:-}" && "${RESUME}" != "auto" && "${RESUME}" != "1" ]]; then
    RESUME_OVERRIDES=("resume=${RESUME}")
    echo "[auto-resume] using explicit resume=${RESUME}"
    return
  fi

  local latest="" latest_step=-1 d s
  if [[ -d "${state_dir}" ]]; then
    for d in "${state_dir}"/step_*; do
      [[ -d "${d}" ]] || continue
      [[ -f "${d}/trainer_state.json" ]] || continue   # skip half-written ckpts
      s="${d##*/step_}"
      s=$((10#${s}))            # strip leading zeros → decimal
      if (( s > latest_step )); then latest_step=${s}; latest="${d}"; fi
    done
  fi

  if [[ -n "${latest}" ]]; then
    RESUME_OVERRIDES=("resume=${latest}")
    echo "[auto-resume] resuming from ${latest} (step ${latest_step})."
  else
    echo "[auto-resume] no valid checkpoint under ${state_dir} → starting fresh."
  fi
}
