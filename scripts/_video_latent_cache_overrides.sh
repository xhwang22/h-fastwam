#!/usr/bin/env bash

VIDEO_LATENT_CACHE_OVERRIDES=()
if [[ -n "${VIDEO_LATENT_CACHE_DIR:-}" ]]; then
  VIDEO_LATENT_CACHE_OVERRIDES=(
    "video_latent_cache.enabled=true"
    "video_latent_cache.root=${VIDEO_LATENT_CACHE_DIR}"
    "video_latent_cache.shard_size=${LATENT_CACHE_SHARD_SIZE:-32}"
    "video_latent_cache.batch_size=${LATENT_CACHE_BATCH_SIZE:-1}"
    "video_latent_cache.num_workers=${LATENT_CACHE_NUM_WORKERS:-4}"
    "video_latent_cache.dtype=${LATENT_CACHE_DTYPE:-bf16}"
    "video_latent_cache.drop_train_video=${LATENT_CACHE_DROP_TRAIN_VIDEO:-true}"
    "video_latent_cache.include_val=${LATENT_CACHE_INCLUDE_VAL:-false}"
  )
fi
