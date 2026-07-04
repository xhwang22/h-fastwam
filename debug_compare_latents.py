"""Compare inference with zero-padded vs real-video-encoded latents_video.

This isolates the suspected bug: at inference, we set future-frame latents to zero,
but at training, those positions hold real V-JEPA features. If the model behaves much
better when given real features, the zero-init is the root cause.
"""
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
EPISODE_IDX = 0


def _resize_for_dataset(img_np, target_h=224, target_w=224):
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


@torch.no_grad()
def predict_action_with_custom_latents_video(
    model, latents_video, prompt, proprio, action_horizon, num_inference_steps, seed=42,
):
    """Replicates infer_action's vjepa branch but lets us pass arbitrary latents_video."""
    from fastwam.models.wan22.fastwam import FastWAM
    model.eval()
    device = model.device
    dtype = model.torch_dtype

    # text encoding
    context, context_mask = model.encode_prompt(prompt)
    if proprio is not None:
        context, context_mask = model._append_proprio_to_context(
            context=context, context_mask=context_mask, proprio=proprio,
        )

    generator = torch.Generator(device="cpu").manual_seed(seed)
    latents_action = torch.randn(
        (1, action_horizon, model.action_expert.action_dim),
        generator=generator, device="cpu", dtype=torch.float32,
    ).to(device=device, dtype=dtype)

    fuse_flag = bool(getattr(model.video_expert, "fuse_vae_embedding_in_latents", False))

    infer_t, infer_d = model.infer_action_scheduler.build_inference_schedule(
        num_inference_steps=num_inference_steps, device=device, dtype=latents_action.dtype,
    )
    for step_t, step_d in zip(infer_t, infer_d):
        timestep_action = step_t.unsqueeze(0).to(dtype=latents_action.dtype, device=device)
        pred_action = model._predict_action_noise(
            first_frame_latents=latents_video,
            latents_action=latents_action,
            timestep_action=timestep_action,
            context=context,
            context_mask=context_mask,
            fuse_vae_embedding_in_latents=fuse_flag,
        )
        latents_action = model.infer_action_scheduler.step(pred_action, step_d, latents_action)

    return latents_action[0].detach().to(device="cpu", dtype=torch.float32)


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

    # --- Load episode + extract frames ---
    parquet_path = Path(DATASET_DIR) / "data" / "chunk-000" / f"episode_{EPISODE_IDX:06d}.parquet"
    df = pd.read_parquet(str(parquet_path))
    start = 0
    num_frames = int(cfg.data.train.num_frames)  # 33
    stride = int(cfg.data.train.action_video_freq_ratio)  # 4
    video_idxs = list(range(start, start + num_frames, stride))  # 9 indices

    gt_actions_raw = np.stack(df["action"].iloc[start:start + num_frames - 1].values, axis=0).astype(np.float32)
    state_t0_raw = np.array(df["observation.state"].iloc[start], dtype=np.float32)

    tasks_jsonl = pd.read_json(Path(DATASET_DIR) / "meta" / "tasks.jsonl", lines=True)
    task_idx = int(df["task_index"].iloc[0])
    task_desc = tasks_jsonl[tasks_jsonl["task_index"] == task_idx]["task"].iloc[0]
    prompt = DEFAULT_PROMPT.format(task=task_desc)

    img_frames = _load_episode_video_frames(DATASET_DIR, EPISODE_IDX, "observation.images.image", video_idxs)
    wrist_frames = _load_episode_video_frames(DATASET_DIR, EPISODE_IDX, "observation.images.wrist_image", video_idxs)

    # Build [3, T_video, H, W=448] tensor by horizontal cat per frame
    video_per_frame = []
    for i in range(num_frames // stride + (1 if num_frames % stride else 0)):
        if i >= len(img_frames):
            break
        p = _resize_for_dataset(img_frames[i])  # [3, 224, 224]
        w = _resize_for_dataset(wrist_frames[i])
        rgb = torch.cat([p, w], dim=2)  # [3, 224, 448]
        video_per_frame.append(rgb)
    video_full = torch.stack(video_per_frame, dim=1)  # [3, T_video=9, H=224, W=448]
    video_full = video_full.unsqueeze(0).to(device=device, dtype=dtype)
    print(f"video_full shape: {video_full.shape}")

    # --- Encode full video to get real latents ---
    print("=== Encoding full video through V-JEPA ===")
    real_latents = model.visual_encoder.encode(video_full, device=device)
    print(f"real_latents shape: {real_latents.shape}")  # [1, 1408, T_lat=3, H_lat, W_lat]

    # --- Build first-frame-only latents (zero-padded for future frames) ---
    print("=== Encoding only first frame (zero-padded for future) ===")
    first_frame_only = video_full[:, :, 0:1].clone()  # [1, 3, 1, 224, 448]
    print(f"first_frame_only shape: {first_frame_only.shape}")
    first_frame_lat = model.visual_encoder.encode(first_frame_only, device=device)
    print(f"first_frame_lat shape: {first_frame_lat.shape}")  # [1, 1408, T_lat=1?, H, W]

    B, C, T_real, H_lat, W_lat = real_latents.shape
    zero_padded_latents = torch.zeros_like(real_latents)
    # The eval code does: latents_video[:, :, 0:1] = first_frame_latents
    # The first-frame encoding likely produces T=1 for single-frame input
    # but possibly different shape. Let's match.
    if first_frame_lat.shape[2] == 1:
        zero_padded_latents[:, :, 0:1] = first_frame_lat
    else:
        # if it's longer, take first slice
        zero_padded_latents[:, :, 0:1] = first_frame_lat[:, :, 0:1]
    print(f"zero_padded_latents shape: {zero_padded_latents.shape}")

    # --- Build "first-frame replicated" latents ---
    print("=== Building first-frame-replicated latents ===")
    repl_latents = real_latents[:, :, 0:1].expand(B, C, T_real, H_lat, W_lat).clone()
    print(f"repl_latents shape: {repl_latents.shape}")

    # --- Compare real-frames latents and first-frame-only-encoded latents at idx 0 ---
    diff_first = (real_latents[:, :, 0] - first_frame_lat[:, :, 0]).abs()
    print(f"\nFirst-frame diff between full-video-encoded and single-frame-encoded:")
    print(f"  MAE: {diff_first.mean().item():.4f}")
    print(f"  max: {diff_first.max().item():.4f}")
    print(f"  full-video-encode first-frame magnitude: {real_latents[:,:,0].abs().mean().item():.4f}")
    print(f"  single-frame-encode magnitude:           {first_frame_lat[:,:,0].abs().mean().item():.4f}")

    # --- Build proprio (t=0) ---
    state_meta = processor.shape_meta["state"]
    state_key = state_meta[0]["key"]
    state_batch = {"state": {state_key: torch.as_tensor(state_t0_raw, dtype=torch.float32).unsqueeze(0)}}
    state_batch = processor.action_state_transform(state_batch)
    state_batch = processor.normalizer.forward(state_batch)
    proprio_norm = state_batch["state"][state_key].to(device=device, dtype=dtype)

    action_horizon = num_frames - 1
    num_steps = int(cfg.get("eval_num_inference_steps", 10))

    # --- Normalize GT for comparison ---
    action_stats = dataset_stats['action']['default']
    cur_stats = {k.removeprefix('global_'): v for k, v in action_stats.items() if k.startswith('global_')}
    norm_action = SingleFieldLinearNormalizer(stats=cur_stats, mode='min/max')
    gt_norm = norm_action.forward(torch.tensor(gt_actions_raw))

    # === Mode A: zero-padded (matches current eval code) ===
    print("\n=== Mode A: zero-padded latents_video (matches eval code) ===")
    pred_a = predict_action_with_custom_latents_video(
        model, zero_padded_latents, prompt, proprio_norm, action_horizon, num_steps,
    )
    diff_a = (pred_a - gt_norm).abs()
    print(f"Per-dim MAE: {diff_a.mean(dim=0).numpy()}")
    print(f"Overall MAE: {diff_a.mean().item():.4f}")
    print(f"Overall MSE: {(pred_a - gt_norm).pow(2).mean().item():.4f}")

    # === Mode B: real video latents (matches training distribution) ===
    print("\n=== Mode B: real V-JEPA latents from full 9-frame video (matches training) ===")
    pred_b = predict_action_with_custom_latents_video(
        model, real_latents, prompt, proprio_norm, action_horizon, num_steps,
    )
    diff_b = (pred_b - gt_norm).abs()
    print(f"Per-dim MAE: {diff_b.mean(dim=0).numpy()}")
    print(f"Overall MAE: {diff_b.mean().item():.4f}")
    print(f"Overall MSE: {(pred_b - gt_norm).pow(2).mean().item():.4f}")

    # === Mode C: first-frame replicated ===
    print("\n=== Mode C: first-frame replicated across temporal axis ===")
    pred_c = predict_action_with_custom_latents_video(
        model, repl_latents, prompt, proprio_norm, action_horizon, num_steps,
    )
    diff_c = (pred_c - gt_norm).abs()
    print(f"Per-dim MAE: {diff_c.mean(dim=0).numpy()}")
    print(f"Overall MAE: {diff_c.mean().item():.4f}")
    print(f"Overall MSE: {(pred_c - gt_norm).pow(2).mean().item():.4f}")

    # === Summary ===
    print("\n=== Summary (10-step inference) ===")
    print(f"  Zero-padded     (current eval): MAE={diff_a.mean().item():.4f}  MSE={(pred_a - gt_norm).pow(2).mean().item():.4f}")
    print(f"  Real-video-enc (training-like): MAE={diff_b.mean().item():.4f}  MSE={(pred_b - gt_norm).pow(2).mean().item():.4f}")
    print(f"  First-frame replicate         : MAE={diff_c.mean().item():.4f}  MSE={(pred_c - gt_norm).pow(2).mean().item():.4f}")

    # === Mode D: more inference steps ===
    print("\n=== Mode D: 50 inference steps (zero-padded) ===")
    pred_d = predict_action_with_custom_latents_video(
        model, zero_padded_latents, prompt, proprio_norm, action_horizon, 50,
    )
    diff_d = (pred_d - gt_norm).abs()
    print(f"Overall MAE: {diff_d.mean().item():.4f}  MSE: {(pred_d - gt_norm).pow(2).mean().item():.4f}")

    print("\n=== Mode E: 50 inference steps (real-video latents) ===")
    pred_e = predict_action_with_custom_latents_video(
        model, real_latents, prompt, proprio_norm, action_horizon, 50,
    )
    diff_e = (pred_e - gt_norm).abs()
    print(f"Overall MAE: {diff_e.mean().item():.4f}  MSE: {(pred_e - gt_norm).pow(2).mean().item():.4f}")

    # === Teacher-forcing test ===
    # Add small noise to GT, run 1 model step, see if predicted velocity makes sense.
    print("\n=== Teacher-forcing: predict velocity at small noise level (sigma=0.05) ===")
    gt_norm_dev = gt_norm.unsqueeze(0).to(device=device, dtype=dtype)
    sigma = 0.05
    timestep_action = torch.tensor([sigma * 1000.0], dtype=dtype, device=device)
    noise = torch.randn(gt_norm_dev.shape, generator=torch.Generator(device="cpu").manual_seed(0), device="cpu").to(device=device, dtype=dtype)
    noisy = (1 - sigma) * gt_norm_dev + sigma * noise

    # encode prompt + proprio
    context, context_mask = model.encode_prompt(prompt)
    if proprio_norm is not None:
        context, context_mask = model._append_proprio_to_context(
            context=context, context_mask=context_mask, proprio=proprio_norm,
        )

    fuse_flag = bool(getattr(model.video_expert, "fuse_vae_embedding_in_latents", False))
    with torch.no_grad():
        pred_velocity = model._predict_action_noise(
            first_frame_latents=real_latents,
            latents_action=noisy,
            timestep_action=timestep_action,
            context=context,
            context_mask=context_mask,
            fuse_vae_embedding_in_latents=fuse_flag,
        )
    target_velocity = noise - gt_norm_dev
    velocity_diff = (pred_velocity.float() - target_velocity.float()).abs()
    print(f"  velocity MAE: {velocity_diff.mean().item():.4f}")
    print(f"  velocity MSE: {(pred_velocity.float()-target_velocity.float()).pow(2).mean().item():.4f}")
    print(f"  velocity magnitude: pred={pred_velocity.float().abs().mean().item():.4f} target={target_velocity.float().abs().mean().item():.4f}")


if __name__ == "__main__":
    main()
