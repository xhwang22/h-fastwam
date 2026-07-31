"""Try various inference tricks to improve action MAE."""
import os
import sys
from pathlib import Path

os.environ["MUJOCO_GL"] = "osmesa"
os.environ["PYOPENGL_PLATFORM"] = "osmesa"
os.environ.setdefault("DIFFSYNTH_MODEL_BASE_PATH", os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints"))
os.environ["TOKENIZERS_PARALLELISM"] = "false"

project_root = Path(__file__).resolve().parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import numpy as np
import torch
import pandas as pd
import imageio.v3 as iio
from PIL import Image
from omegaconf import OmegaConf
from hydra import initialize_config_dir, compose
from hydra.utils import instantiate

OmegaConf.register_new_resolver("eval", eval)
OmegaConf.register_new_resolver("max", lambda x: max(x))
OmegaConf.register_new_resolver("split", lambda s, idx: s.split("/")[int(idx)])

from fastwam.datasets.lerobot.processors.fastwam_processor import FastWAMProcessor
from fastwam.datasets.lerobot.utils.normalizer import (
    load_dataset_stats_from_json, SingleFieldLinearNormalizer,
)
from fastwam.datasets.lerobot.robot_video_dataset import DEFAULT_PROMPT


DATASET_DIR = "/apdcephfs_gy2/share_302533218/shaunxhwang/embodied/FastWAM/data/libero_mujoco3.3.2/libero_spatial_no_noops_lerobot"


def _resize(img_np, target_h=224, target_w=224):
    pil = Image.fromarray(img_np)
    src_w, src_h = pil.size
    scale = max(target_w / src_w, target_h / src_h)
    new_w, new_h = round(src_w * scale), round(src_h * scale)
    pil = pil.resize((new_w, new_h), resample=Image.BILINEAR)
    left = max((new_w - target_w) // 2, 0)
    top = max((new_h - target_h) // 2, 0)
    pil = pil.crop((left, top, left + target_w, top + target_h))
    arr = np.asarray(pil, dtype=np.uint8)
    t = torch.tensor(arr).permute(2, 0, 1).float()
    t = t * (2.0 / 255.0) - 1.0
    return t


def _load_episode_video_frames(dataset_dir, episode_idx, key, frame_indices):
    video_path = (
        Path(dataset_dir) / "videos" / "chunk-000" / key
        / f"episode_{episode_idx:06d}.mp4"
    )
    frames = []
    for i, frame in enumerate(iio.imiter(str(video_path))):
        if i in frame_indices:
            frames.append(frame)
            if len(frames) == len(frame_indices):
                break
    return frames


def main():
    config_dir = str(project_root / "configs")
    with initialize_config_dir(version_base="1.3", config_dir=config_dir):
        cfg = compose(
            config_name="sim_libero",
            overrides=[
                "model=fastwam_vjepa2ac_predictor",
                "task=libero_uncond_2cam224_1e-4",
                "ckpt=",
                "EVALUATION.output_dir=/tmp/debug_sanity_check",
                "EVALUATION.dataset_stats_path=runs/libero_vjepa2ac_predictor/w1mn_2026-05-11_13-58-10/dataset_stats.json",
            ],
        )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16

    print("=== Loading model ===")
    model = instantiate(cfg.model, model_dtype=dtype, device=device)
    model.load_checkpoint(str(cfg.ckpt))
    model = model.to(device).eval()

    dataset_stats = load_dataset_stats_from_json(str(cfg.EVALUATION.dataset_stats_path))
    processor = instantiate(cfg.data.train.processor).eval()
    processor.set_normalizer_from_stats(dataset_stats)

    samples = []
    for episode_idx in range(8):
        parquet_path = Path(DATASET_DIR) / "data" / "chunk-000" / f"episode_{episode_idx:06d}.parquet"
        df = pd.read_parquet(str(parquet_path))
        if len(df) < 35:
            continue
        start = max(0, len(df) // 4)
        num_frames = int(cfg.data.train.num_frames)
        stride = int(cfg.data.train.action_video_freq_ratio)
        if start + num_frames > len(df):
            start = len(df) - num_frames
        video_idxs = list(range(start, start + num_frames, stride))

        gt_actions_raw = np.stack(df["action"].iloc[start:start + num_frames - 1].values, axis=0).astype(np.float32)
        state_t0_raw = np.array(df["observation.state"].iloc[start], dtype=np.float32)

        tasks_jsonl = pd.read_json(Path(DATASET_DIR) / "meta" / "tasks.jsonl", lines=True)
        task_idx = int(df["task_index"].iloc[0])
        task_desc = tasks_jsonl[tasks_jsonl["task_index"] == task_idx]["task"].iloc[0]
        prompt = DEFAULT_PROMPT.format(task=task_desc)

        img_frames = _load_episode_video_frames(DATASET_DIR, episode_idx, "observation.images.image", video_idxs)
        wrist_frames = _load_episode_video_frames(DATASET_DIR, episode_idx, "observation.images.wrist_image", video_idxs)
        if len(img_frames) < 9:
            continue

        # Build first-frame-only encoded latents (zero-padded for future)
        # Same as in eval pipeline: T_lat=3 zeros, idx 0 from single-frame encode
        primary = _resize(img_frames[0])
        wrist = _resize(wrist_frames[0])
        first_frame = torch.cat([primary, wrist], dim=2).unsqueeze(0).to(device=device, dtype=dtype)

        with torch.no_grad():
            ff_lat = model.visual_encoder.encode(first_frame.unsqueeze(2), device=device)

        # Get full T_lat shape
        video_per_frame = []
        for i in range(9):
            p = _resize(img_frames[i])
            w = _resize(wrist_frames[i])
            rgb = torch.cat([p, w], dim=2)
            video_per_frame.append(rgb)
        video_full = torch.stack(video_per_frame, dim=1).unsqueeze(0).to(device=device, dtype=dtype)
        with torch.no_grad():
            real_latents_shape = model.visual_encoder.encode(video_full, device=device).shape

        zero_padded = torch.zeros(real_latents_shape, dtype=dtype, device=device)
        zero_padded[:, :, 0:1] = ff_lat[:, :, 0:1] if ff_lat.shape[2] >= 1 else ff_lat

        samples.append({
            "zero_padded": zero_padded,
            "gt_actions_raw": gt_actions_raw,
            "state_t0_raw": state_t0_raw,
            "prompt": prompt,
            "task": task_desc,
        })

    action_stats = dataset_stats['action']['default']
    cur_stats = {k.removeprefix('global_'): v for k, v in action_stats.items() if k.startswith('global_')}
    norm_action = SingleFieldLinearNormalizer(stats=cur_stats, mode='min/max')

    fuse_flag = bool(getattr(model.video_expert, "fuse_vae_embedding_in_latents", False))
    state_meta = processor.shape_meta["state"]
    state_key = state_meta[0]["key"]

    @torch.no_grad()
    def infer_one(rec, num_steps, shift, seeds=(42,)):
        gt_norm = norm_action.forward(torch.tensor(rec["gt_actions_raw"]))
        action_horizon = rec["gt_actions_raw"].shape[0]
        sb = {"state": {state_key: torch.as_tensor(rec["state_t0_raw"], dtype=torch.float32).unsqueeze(0)}}
        sb = processor.action_state_transform(sb)
        sb = processor.normalizer.forward(sb)
        proprio_norm = sb["state"][state_key].to(device=device, dtype=dtype)
        context, context_mask = model.encode_prompt(rec["prompt"])
        context, context_mask = model._append_proprio_to_context(
            context=context, context_mask=context_mask, proprio=proprio_norm,
        )

        preds = []
        for seed in seeds:
            latents_action = torch.randn(
                (1, action_horizon, model.action_expert.action_dim),
                generator=torch.Generator(device="cpu").manual_seed(seed),
                device="cpu", dtype=torch.float32,
            ).to(device=device, dtype=dtype)
            infer_t, infer_d = model.infer_action_scheduler.build_inference_schedule(
                num_inference_steps=num_steps, device=device, dtype=latents_action.dtype,
                shift_override=shift,
            )
            for step_t, step_d in zip(infer_t, infer_d):
                tta = step_t.unsqueeze(0).to(dtype=latents_action.dtype, device=device)
                pv = model._predict_action_noise(
                    first_frame_latents=rec["zero_padded"],
                    latents_action=latents_action,
                    timestep_action=tta,
                    context=context,
                    context_mask=context_mask,
                    fuse_vae_embedding_in_latents=fuse_flag,
                )
                latents_action = model.infer_action_scheduler.step(pv, step_d, latents_action)
            preds.append(latents_action[0].detach().to(device="cpu", dtype=torch.float32))
        # Average over seeds
        avg_pred = torch.stack(preds, dim=0).mean(dim=0)
        return (avg_pred - gt_norm).abs().mean().item()

    @torch.no_grad()
    def single_step_predict(rec, t_init=0.5):
        """One-step Euler from t=t_init: predict velocity once, jump to t=0."""
        gt_norm = norm_action.forward(torch.tensor(rec["gt_actions_raw"]))
        action_horizon = rec["gt_actions_raw"].shape[0]
        sb = {"state": {state_key: torch.as_tensor(rec["state_t0_raw"], dtype=torch.float32).unsqueeze(0)}}
        sb = processor.action_state_transform(sb)
        sb = processor.normalizer.forward(sb)
        proprio_norm = sb["state"][state_key].to(device=device, dtype=dtype)
        context, context_mask = model.encode_prompt(rec["prompt"])
        context, context_mask = model._append_proprio_to_context(
            context=context, context_mask=context_mask, proprio=proprio_norm,
        )

        # noisy at t_init
        noise = torch.randn(
            (1, action_horizon, model.action_expert.action_dim),
            generator=torch.Generator(device="cpu").manual_seed(42),
            device="cpu", dtype=torch.float32,
        ).to(device=device, dtype=dtype)
        latents_action = noise.clone()  # start from pure noise (t=1)

        # actually, noise at t=t_init means: noisy = (1-t)*x + t*noise. At t=1, noisy = noise.
        # We jump to t=0 in one step: x = noisy - 1.0 * pred_velocity (but step delta = -t_init for one big step)
        timestep_action = torch.tensor([t_init * 1000.0], dtype=dtype, device=device)
        # Need noisy at t_init: but we don't have x. So just start at full noise (t=1.0) and Euler-step to t=0.
        timestep_action = torch.tensor([1000.0], dtype=dtype, device=device)
        pred_v = model._predict_action_noise(
            first_frame_latents=rec["zero_padded"],
            latents_action=latents_action,
            timestep_action=timestep_action,
            context=context,
            context_mask=context_mask,
            fuse_vae_embedding_in_latents=fuse_flag,
        )
        # Single Euler step: x = noisy + pred_v * (-1)
        x_clean = latents_action + pred_v * (-1.0)
        x_clean = x_clean[0].detach().to(device="cpu", dtype=torch.float32)
        return (x_clean - gt_norm).abs().mean().item()

    # Run various tests
    print("\n=== Comparing inference strategies ===")
    print(f"{'config':<40} {'mean MAE':>10}")

    # Baseline
    for shift, n in [(5.0, 10), (5.0, 5), (5.0, 20), (3.0, 10), (1.0, 10)]:
        maes = [infer_one(r, n, shift, seeds=(42,)) for r in samples]
        print(f"shift={shift:>3.1f} steps={n:>3} 1 seed                  {np.mean(maes):>10.4f}")

    # Multi-seed average
    for shift, n in [(5.0, 10)]:
        maes = [infer_one(r, n, shift, seeds=(42, 1, 2, 3, 4)) for r in samples]
        print(f"shift={shift:>3.1f} steps={n:>3} 5 seeds avg            {np.mean(maes):>10.4f}")

    # Single-step Euler
    maes = [single_step_predict(r) for r in samples]
    print(f"single Euler step from t=1                {np.mean(maes):>10.4f}")


if __name__ == "__main__":
    main()
