#!/usr/bin/env bash

fastwam_select_robotwin_data_source() {
  ROBOTWIN_WEBDATASET_ROOT="${ROBOTWIN_WEBDATASET_ROOT:-${ROBOTWIN_DATA_ROOT}/robotwin2.0_webdataset}"
  if [[ -f "${ROBOTWIN_WEBDATASET_ROOT}/dataset.done" ]]; then
    ROBOTWIN_DATA_CONFIG="robotwin_interleaved_webdataset"
    ROBOTWIN_DATA_OVERRIDES=(
      "data.train.preprocessed_root=${ROBOTWIN_WEBDATASET_ROOT}"
      "data.val.preprocessed_root=${ROBOTWIN_WEBDATASET_ROOT}"
    )
    DATALOADER_PREFETCH_FACTOR="${DATALOADER_PREFETCH_FACTOR:-2}"
    echo "[robotwin-data] using indexed WebDataset: ${ROBOTWIN_WEBDATASET_ROOT}"
  else
    ROBOTWIN_DATA_CONFIG="robotwin_interleaved"
    ROBOTWIN_DATA_OVERRIDES=(
      "data.train.dataset_dirs=[${ROBOTWIN_DATA_ROOT}/robotwin2.0/robotwin2.0]"
      "data.train.pretrained_norm_stats=${ROBOTWIN_DATA_ROOT}/robotwin2.0/dataset_stats.json"
      "data.val.dataset_dirs=[${ROBOTWIN_DATA_ROOT}/robotwin2.0/robotwin2.0]"
      "data.val.pretrained_norm_stats=${ROBOTWIN_DATA_ROOT}/robotwin2.0/dataset_stats.json"
    )
    echo "[robotwin-data] complete WebDataset marker not found; using MP4 source data."
  fi
  export ROBOTWIN_WEBDATASET_ROOT ROBOTWIN_DATA_CONFIG DATALOADER_PREFETCH_FACTOR
}
