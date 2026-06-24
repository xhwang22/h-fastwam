"""Pretrain-specific training script for Wan2.2 continue-pretraining.

This script uses the standard ``Wan22Trainer`` but with a simplified setup
that works with text-video datasets like OpenVid-1M.

Usage::

    # With Hydra configs
    python scripts/pretrain.py task=pretrain_vae_openvid

    # Or with accelerate (multi-GPU)
    accelerate launch scripts/pretrain.py task=pretrain_vae_openvid
"""

import hydra
from omegaconf import DictConfig

from fastwam.runtime import run_training
from fastwam.utils.config_resolvers import register_default_resolvers

register_default_resolvers()


@hydra.main(config_path="../configs", config_name="pretrain", version_base="1.3")
def main(cfg: DictConfig):
    run_training(cfg)


if __name__ == "__main__":
    main()
