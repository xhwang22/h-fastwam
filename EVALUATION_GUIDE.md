# FastWAM Evaluation Guide

## Overview
This document comprehensively covers the evaluation infrastructure for FastWAM, based on the training script `run_libero_hfastwam_8card_small_vjepa_predictor.sh` and the evaluation framework.

---

## Part 1: Training Script Analysis

### Script: `scripts/run_libero_hfastwam_8card_small_vjepa_predictor.sh`

#### Key Parameters
- **GPU Setup**: 8 GPUs (DDP, non-DeepSpeed)
- **Backend**: torchrun with nproc_per_node=8
- **Global Batch Size**: 8 GPUs × 1 batch × 16 gradient_accumulation = 128

#### Model Configuration
- **Model Name**: `hfastwam_small_vjepa_predictor`
- **Key Difference**: Uses JEPAPredictor instead of WAN-style flow-matching DiT for video generation
  - Deterministic next-frame latent predictor
  - Trained with L1 regression loss in V-JEPA 2-AC encoder space (1408-dim)
  - No flow-matching scheduler for video expert

#### Model Architecture (from `configs/model/hfastwam_small_vjepa_predictor.yaml`):
- **Language Expert**: Frozen Qwen3-VL-2B (frozen, 16 heads × 128 = 2048, 28 layers)
- **Video Expert**: JEPAPredictor (random-init, 16×128=2048, 28 layers, L1 loss)
- **Action Expert**: ActionDiT (random-init, flow-matching denoiser, same dims)
- **Visual Encoder**: Frozen V-JEPA 2-AC ViT-g/16
  - 1408-dim raw features
  - skip_projection=true → straight to JEPAPredictor
- **Trainable Parameters**: ~3.8B

#### Data Configuration
- **Task**: `libero_uncond_2cam224_1e-4`
- **Datasets**:
  - libero_spatial_no_noops_lerobot
  - libero_object_no_noops_lerobot
  - libero_goal_no_noops_lerobot
  - libero_10_no_noops_lerobot
- **Num Frames**: 1 segment per train/val
- **Epochs**: 3

#### Checkpoint Saving
- **Location**: `${LOG_DIR}/checkpoints/weights/step_XXXXXX.pt`
- **Default Log Dir**: `runs/libero_hfastwam/${RUN_NAME}/`
- **Trainer Components**: 
  - Model weights checkpoint
  - Trainer state (global_step, epoch, batch_in_epoch)
  - Logs saved to `${LOG_DIR}/train.log.rank0`

#### Environment Settings
- **Offline Mode**: HF_HUB_OFFLINE=1, TRANSFORMERS_OFFLINE=1
- **NCCL Settings**: bond1 socket interface, IB disabled
- **Profiling**: FASTWAM_PROFILE_STEPS=5 (per-step timing)

---

## Part 2: Evaluation Scripts

### Main Evaluation Script: `experiments/libero/eval_libero_single.py`

#### Entry Point
```bash
python experiments/libero/eval_libero_single.py [CONFIG OVERRIDES]
```

#### Key Components

##### 1. **Model Loading**
- Instantiates model from config: `model=hfastwam_small_vjepa_predictor`
- Loads checkpoint via `model.load_checkpoint(ckpt_path)`
- Sets to eval mode: `model.eval()`
- Device: CPU or CUDA (auto-detected)
- Dtype: bf16, fp16, or fp32 (configurable)

##### 2. **Checkpoint Validation**
- Checks visual encoder compatibility between checkpoint and model config
- Warns if checkpoint has visual_encoder weights but model uses VAE
- Warns if model expects visual_encoder but checkpoint doesn't have weights

##### 3. **Data Loading**
- Loads dataset statistics from `dataset_stats.json`
- Instantiates FastWAMProcessor with normalizer
- Supports multi-camera (1-2 cameras) with configurable concatenation

##### 4. **Action Horizon**
- Default: `num_frames - 1`
- Configurable via `EVALUATION.action_horizon`
- Must be positive integer

##### 5. **Inference Modes**

###### Standard Action-Only Inference:
```python
model.infer_action(
    input_image=input_image,          # [1, 3, H, W] in [-1, 1]
    action_horizon=action_horizon,
    proprio=proprio,
    prompt=prompt,
    num_inference_steps=num_inference_steps,
    seed=42,
    tiled=False,
)
```

###### Joint (Video + Action) Inference (for future video visualization):
```python
model.infer_joint(
    input_image=input_image,
    action_horizon=action_horizon,
    num_video_frames=num_frames,
    prompt=prompt,
    num_inference_steps=num_inference_steps,
    seed=42,
    tiled=False,
)
```

##### 6. **Metrics Computation**

###### Per-Episode Metrics:
- **Success Rate**: Episodes where task completed (done=True)
- **Future Video PSNR** (optional): Mean PSNR of predicted future frames vs ground truth
  - Captured at replan steps
  - Configurable via `EVALUATION.visualize_future_video`

###### Training Loss Metrics:
- `val_loss`: Validation training loss from trainer
- `psnr_rg`: PSNR of rollout vs GT video
- `ssim_rg`: SSIM of rollout vs GT video
- `psnr_rd`: PSNR of rollout vs VAE decode
- `ssim_rd`: SSIM of rollout vs VAE decode
- `psnr_dg`: PSNR of VAE reconstruction vs GT
- `ssim_dg`: SSIM of VAE reconstruction vs GT
- `action_l1`: L1 error of denormalized predicted actions vs GT
- `action_l2`: L2 error of denormalized predicted actions vs GT

##### 7. **Output Structure**
```
output_dir/
  {task_suite_name}/
    videos/
      task{task_id}_trial{trial_idx}.mp4
    predicted_videos/  (if visualize_future_video=true)
      task{task_id}_trial{trial_idx}_replan{replan_idx}.mp4
      task{task_id}_trial{trial_idx}_all.mp4
    gpu{gpu_id}_task{task_id}_results.json
```

##### 8. **Results JSON Structure**
```json
{
  "task_suite": "libero_spatial",
  "task_id": 0,
  "task_description": "Move the {object} to the {target}",
  "successes": 5,
  "total_episodes": 10,
  "success_episodes": [0, 1, 2, 3, 4],
  "failure_episodes": [5, 6, 7, 8, 9],
  "start_time": "2026-06-08 12:00:00",
  "duration": 3600.0,
  "future_video_psnr_mean": 25.3  // if visualize_future_video=true
}
```

---

## Part 3: Running Evaluation on Single Machine with 8 GPUs

### Step 1: Ensure Checkpoints Exist

Checkpoints from training are saved at:
```
runs/libero_hfastwam/{RUN_NAME}/checkpoints/weights/step_XXXXXX.pt
```

Example:
```bash
ls runs/libero_hfastwam/libero_hfastwam_8card_small_vjepa_predictor/checkpoints/weights/
# Output: step_000100.pt, step_000200.pt, etc.
```

### Step 2: Create Evaluation Config

Create a config file (e.g., `eval_config.yaml`) or override via CLI:

```yaml
ckpt: /path/to/checkpoint/step_XXXXXX.pt
gpu_id: 0  # which GPU to use (0-7 for single GPU eval)

# Model config (must match training)
model: hfastwam_small_vjepa_predictor

# Data config
data:
  train:
    processor:
      proprio_output_dim: 20  # match training
      action_output_dim: 8    # match training

# Evaluation parameters
EVALUATION:
  task_suite_name: libero_spatial
  task_id: 0
  num_trials: 5
  action_horizon: null  # default: num_frames - 1
  replan_steps: 5
  num_inference_steps: 20
  text_cfg_scale: 1.0
  negative_prompt: ""
  visualize_future_video: false  # set true for future video prediction metrics
  output_dir: ./evaluate_results

# Optional
seed: 42
mixed_precision: bf16
```

### Step 3: Run Single-GPU Evaluation

**Option A: Single GPU Evaluation (Simplest)**
```bash
python experiments/libero/eval_libero_single.py \
  ckpt=runs/libero_hfastwam/libero_hfastwam_8card_small_vjepa_predictor/checkpoints/weights/step_030000.pt \
  gpu_id=0 \
  EVALUATION.task_suite_name=libero_spatial \
  EVALUATION.task_id=0 \
  EVALUATION.num_trials=5 \
  EVALUATION.output_dir=./evaluate_results
```

**Option B: Multiple GPUs via GNU Parallel or Similar**

To evaluate multiple tasks in parallel (different GPU for each):
```bash
# Evaluate all 10 spatial tasks in parallel on 8 GPUs
for task_id in {0..9}; do
  gpu_id=$((task_id % 8))
  python experiments/libero/eval_libero_single.py \
    ckpt=runs/libero_hfastwam/libero_hfastwam_8card_small_vjepa_predictor/checkpoints/weights/step_030000.pt \
    gpu_id=$gpu_id \
    EVALUATION.task_suite_name=libero_spatial \
    EVALUATION.task_id=$task_id \
    EVALUATION.num_trials=5 \
    EVALUATION.output_dir=./evaluate_results &
done
wait
```

**Option C: Multi-GPU Parallelism (via Manager Script)**

For advanced multi-GPU task parallelism:
```bash
# Use existing manager infrastructure (if available)
python experiments/libero/run_libero_manager.py \
  ckpt=runs/libero_hfastwam/libero_hfastwam_8card_small_vjepa_predictor/checkpoints/weights/step_030000.pt \
  MULTIRUN.enabled=true \
  MULTIRUN.num_gpus=8 \
  MULTIRUN.max_tasks_per_gpu=2
```

### Step 4: Monitor Results

Results are saved to: `evaluate_results/{task_suite}/{now:%Y%m%d_%H%M%S}/`

```bash
# View results
ls evaluate_results/libero_spatial/
cat evaluate_results/libero_spatial/20260608_120000/gpu0_task0_results.json

# Aggregate results
for f in evaluate_results/libero_spatial/20260608_120000/gpu*_task*_results.json; do
  jq '.successes' $f
done | awk '{sum+=$1} END {print "Total successes:", sum}'
```

---

## Part 4: Trainer's Internal Evaluation (During Training)

The trainer also performs inline validation during training. This is separate from the above evaluation script.

### Trainer.evaluate() Method (in `src/fastwam/trainer.py`)

Called every `eval_every` steps during training.

#### What It Does:
1. **Random Validation Sample**: Picks a random sample from val_dataset
2. **Training Loss**: Computes forward pass loss
3. **Inference Metrics** (if VAE mode):
   - Runs model.infer() with `num_inference_steps`
   - Computes PSNR/SSIM: rollout vs GT, rollout vs VAE decode, VAE vs GT
   - Saves stitched videos to `eval_dir`
4. **Action Metrics**:
   - If action GT available: L1/L2 error of denormalized actions
5. **Logs to Wandb**: If wandb enabled
6. **Video Saving**: Saves comparison videos to `${LOG_DIR}/eval/`

#### Configuration
```yaml
eval_every: 200  # evaluate every N steps
eval_num_inference_steps: 20  # inference steps for eval
```

#### Output
- Videos: `${LOG_DIR}/eval/step_XXXXXX_rank_YYY.mp4`
- Metrics: Logged to Wandb and stdout

#### Metrics Dictionary Returned:
```python
{
  "val_loss": float,
  "psnr_rg": float,      # rollout vs GT
  "ssim_rg": float,
  "psnr_rd": float,      # rollout vs VAE decode
  "ssim_rd": float,
  "psnr_dg": float,      # VAE decode vs GT
  "ssim_dg": float,
  "action_l1": float,    # optional
  "action_l2": float,    # optional
  "video_path": str,     # optional
}
```

---

## Part 5: Key Differences: Training vs Evaluation

| Aspect | Training | Evaluation |
|--------|----------|-----------|
| **Mode** | Multi-GPU DDP | Single GPU or Multi-GPU task parallelism |
| **Checkpoint** | Saves every `save_every` steps | Loads pretrained checkpoint |
| **Inline Eval** | Every `eval_every` steps on random val sample | Optional, not on real sim tasks |
| **Real World Tasks** | N/A | Runs full LIBERO episodes in simulator |
| **Metrics** | Video reconstruction (PSNR/SSIM) | Task success rate + optional future video metrics |
| **Output** | Logs in training dir, Wandb | Results JSONs + video rollouts |

---

## Part 6: Configuration Hierarchy

### Config Search Path: `configs/`

1. **Base Config**: `train.yaml` (default hydra root)
2. **Task Config**: `task/{task_name}.yaml` (e.g., `libero_uncond_2cam224_1e-4.yaml`)
3. **Model Config**: `model/{model_name}.yaml` (e.g., `hfastwam_small_vjepa_predictor.yaml`)
4. **Data Config**: `data/{data_name}.yaml` (nested in task)
5. **CLI Overrides**: `key=value` or `key.subkey=value`

### Override Example
```bash
python experiments/libero/eval_libero_single.py \
  ckpt=path/to/ckpt.pt \
  model=hfastwam_small_vjepa_predictor \
  EVALUATION.task_suite_name=libero_spatial \
  EVALUATION.task_id=0 \
  EVALUATION.num_trials=10
```

---

## Part 7: Common Issues & Solutions

### Issue 1: Checkpoint Not Found
```
FileNotFoundError: [Errno 2] No such file or directory: 'path/to/ckpt.pt'
```
**Solution**: Verify checkpoint exists and pass absolute path
```bash
ls -lh runs/libero_hfastwam/*/checkpoints/weights/step_*.pt
```

### Issue 2: Dataset Stats Not Found
```
FileNotFoundError: Failed to locate dataset_stats.json
```
**Solution**: Provide explicit path or ensure it's in checkpoint parent directories
```bash
EVALUATION.dataset_stats_path=/path/to/dataset_stats.json
```

### Issue 3: Visual Encoder Mismatch
```
WARNING: Checkpoint contains `visual_encoder` weights but the current model config uses VAE
```
**Solution**: Ensure model config matches checkpoint. If trained with visual encoder, enable it in eval config:
```yaml
model:
  visual_encoder:
    encoder_type: vjepa2_ac
    model_name: vjepa2_ac_vit_giant
```

### Issue 4: CUDA OOM During Inference
**Solution**: Reduce `num_inference_steps` or batch size (not applicable for single-episode eval)
```bash
EVALUATION.num_inference_steps=10
```

---

## Part 8: Quick Reference Commands

### Training on 8 GPUs
```bash
FASTWAM_USE_ZERO_REDUNDANCY_OPTIMIZER=1 RUN_NAME=exp3_test \
  bash scripts/run_libero_hfastwam_8card_small_vjepa_predictor.sh
```

### Evaluate Single Task on 1 GPU
```bash
python experiments/libero/eval_libero_single.py \
  ckpt=runs/libero_hfastwam/libero_hfastwam_8card_small_vjepa_predictor/checkpoints/weights/step_030000.pt \
  gpu_id=0 \
  EVALUATION.task_suite_name=libero_spatial \
  EVALUATION.task_id=0 \
  EVALUATION.num_trials=5
```

### Evaluate All Spatial Tasks Sequentially on GPU 0
```bash
for task_id in {0..9}; do
  python experiments/libero/eval_libero_single.py \
    ckpt=runs/libero_hfastwam/libero_hfastwam_8card_small_vjepa_predictor/checkpoints/weights/step_030000.pt \
    gpu_id=0 \
    EVALUATION.task_suite_name=libero_spatial \
    EVALUATION.task_id=$task_id \
    EVALUATION.num_trials=5
done
```

### Evaluate with Future Video Prediction Metrics
```bash
python experiments/libero/eval_libero_single.py \
  ckpt=runs/libero_hfastwam/libero_hfastwam_8card_small_vjepa_predictor/checkpoints/weights/step_030000.pt \
  gpu_id=0 \
  EVALUATION.task_suite_name=libero_spatial \
  EVALUATION.task_id=0 \
  EVALUATION.num_trials=5 \
  EVALUATION.visualize_future_video=true \
  EVALUATION.replan_steps=5
```

---

## Summary Table: Evaluation Scripts Found

| Script | Location | Purpose | Supported Platforms |
|--------|----------|---------|---------------------|
| eval_libero_single.py | experiments/libero/ | LIBERO task evaluation on simulator | Single GPU |
| eval_robotwin_single.py | experiments/robotwin/ | RoboTwin task evaluation | Single GPU |

Both follow similar structures and support the same configuration system.

