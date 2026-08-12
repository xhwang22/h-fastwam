#!/usr/bin/env bash

fastwam_timestep_sampling_preset() {
  local preset="${1:-baseline}"
  local target="${2:-action}"
  local distribution
  local shift
  local logit_mean="0.0"
  local logit_std="1.0"

  case "${preset}" in
    baseline)
      TIMESTEP_PRESET_SUFFIX="tbaseline_shift5"
      distribution="shifted_uniform"
      shift="5.0"
      ;;
    data)
      TIMESTEP_PRESET_SUFFIX="tdata_shift0p2"
      distribution="shifted_uniform"
      shift="0.2"
      ;;
    noise)
      TIMESTEP_PRESET_SUFFIX="tnoise_shift25"
      distribution="shifted_uniform"
      shift="25.0"
      ;;
    middle)
      TIMESTEP_PRESET_SUFFIX="tmiddle_logitnormal"
      distribution="logit_normal"
      shift="1.0"
      ;;
    uniform)
      TIMESTEP_PRESET_SUFFIX="tuniform"
      distribution="shifted_uniform"
      shift="1.0"
      ;;
    *)
      echo "ERROR: unknown TIMESTEP_SAMPLING_PRESET=${preset}" >&2
      echo "Expected one of: baseline, data, noise, middle, uniform." >&2
      return 1
      ;;
  esac

  local branches
  case "${target}" in
    action)
      branches="action"
      ;;
    video)
      branches="video"
      ;;
    both)
      branches="video action"
      ;;
    *)
      echo "ERROR: unknown timestep sampling target=${target}" >&2
      echo "Expected one of: action, video, both." >&2
      return 1
      ;;
  esac

  TIMESTEP_SAMPLING_OVERRIDES=()
  local branch
  for branch in ${branches}; do
    TIMESTEP_SAMPLING_OVERRIDES+=(
      "model.${branch}_scheduler.sampling_distribution=${distribution}"
      "model.${branch}_scheduler.train_shift=${shift}"
      "model.${branch}_scheduler.logit_mean=${logit_mean}"
      "model.${branch}_scheduler.logit_std=${logit_std}"
    )
  done
}

fastwam_action_timestep_sampling_preset() {
  fastwam_timestep_sampling_preset "${1:-baseline}" action
}
