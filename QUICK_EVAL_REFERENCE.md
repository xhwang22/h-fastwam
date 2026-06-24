# Quick Evaluation Reference Card

## What's Trained and Where

**Training Script**: `scripts/run_libero_hfastwam_8card_small_vjepa_predictor.sh`
- **Model**: H-FastWAM SMALL with JEPAPredictor video expert
- **GPUs**: 8 (DDP, no DeepSpeed)
- **Global Batch**: 128 (8 × 1 × 16 grad_accum)
- **Checkpoints Saved To**: `runs/libero_hfastwam/{RUN_NAME}/checkpoints/weights/step_XXXXXX.pt`

## Training Model Components
- **Language**: Qwen3-VL-2B (frozen) 2048-dim
- **Video**: JEPAPredictor (trainable, 2048-dim, L1 loss in 1408-dim JEPA latent space)
- **Action**: ActionDiT (trainable, 2048-dim, flow-matching)
- **Visual Encoder**: V-JEPA 2-AC ViT-g (frozen)
- **Total Trainable Params**: ~3.8B

## Evaluation Scripts Available

### Main Script (LIBERO Simulator)
```
experiments/libero/eval_libero_single.py
```

### Alternative Script (RoboTwin Simulator)
```
experiments/robotwin/eval_robotwin_single.py
```

## Single-GPU Evaluation (Simplest)

**Step 1: Find checkpoint**
```bash
ls runs/libero_hfastwam/*/checkpoints/weights/step_*.pt
# e.g., runs/libero_hfastwam/libero_hfastwam_8card_small_vjepa_predictor/checkpoints/weights/step_030000.pt
```

**Step 2: Run evaluation**
```bash
python experiments/libero/eval_libero_single.py \
  ckpt=runs/libero_hfastwam/libero_hfastwam_8card_small_vjepa_predictor/checkpoints/weights/step_030000.pt \
  gpu_id=0 \
  EVALUATION.task_suite_name=libero_spatial \
  EVALUATION.task_id=0 \
  EVALUATION.num_trials=5 \
  EVALUATION.output_dir=./evaluate_results
```

**Step 3: Check results**
```bash
cat evaluate_results/libero_spatial/*/gpu0_task0_results.json | jq '.successes'
```

## Multi-GPU Parallel Evaluation (8 GPUs)

Evaluate all 10 spatial tasks across 8 GPUs:
```bash
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

Aggregate results:
```bash
for f in evaluate_results/libero_spatial/*/gpu*_task*_results.json; do
  success=$(jq '.successes' $f)
  total=$(jq '.total_episodes' $f)
  task=$(jq '.task_id' $f)
  echo "Task $task: $success/$total"
done
```

## Key Evaluation Parameters

| Parameter | Meaning | Default | Range |
|-----------|---------|---------|-------|
| `gpu_id` | Which GPU (0-7) | 0 | 0-7 |
| `EVALUATION.task_suite_name` | Which benchmark | libero_spatial | libero_spatial/libero_object/libero_goal/libero_10 |
| `EVALUATION.task_id` | Which task (0-N) | N/A | Depends on suite |
| `EVALUATION.num_trials` | Episodes per task | 5 | 1-100+ |
| `EVALUATION.num_inference_steps` | Diffusion steps (higher=better but slower) | 20 | 1-50 |
| `EVALUATION.replan_steps` | Action chunks | 5 | 1-20 |
| `EVALUATION.visualize_future_video` | Compute future video metrics? | false | true/false |

## Output Files

Results location: `evaluate_results/{task_suite}/{YYYYMMdd_HHMMSS}/`

Files created:
```
evaluate_results/libero_spatial/20260608_120000/
├── gpu0_task0_results.json          # Task results (success rate, metrics)
├── videos/
│   ├── task0_trial0.mp4             # Rollout video (success=1)
│   ├── task0_trial1.mp4             # Rollout video (success=0)
│   └── ...
└── predicted_videos/ (if visualize_future_video=true)
    ├── task0_trial0_replan0.mp4     # Future frame predictions
    └── ...
```

## Results JSON Example

```json
{
  "task_suite": "libero_spatial",
  "task_id": 0,
  "task_description": "Move the red cube to the blue bin",
  "successes": 5,
  "total_episodes": 10,
  "success_episodes": [0, 1, 2, 3, 4],
  "failure_episodes": [5, 6, 7, 8, 9],
  "start_time": "2026-06-08 12:00:00",
  "duration": 3600.0
}
```

## Common Issues & Quick Fixes

| Issue | Symptom | Fix |
|-------|---------|-----|
| Checkpoint not found | FileNotFoundError | Use absolute path: `ckpt=$(pwd)/runs/...` |
| Dataset stats missing | FileNotFoundError for dataset_stats.json | `EVALUATION.dataset_stats_path=/path/to/dataset_stats.json` |
| Wrong model config | Visual encoder mismatch warning | Ensure `model=hfastwam_small_vjepa_predictor` |
| GPU OOM | CUDA out of memory | Reduce `EVALUATION.num_inference_steps=10` |
| Slow inference | Takes very long per step | Reduce `EVALUATION.num_trials=1` or `num_inference_steps` |

## Task Suite Definitions

**libero_spatial** (10 tasks):
- Place items in spatial relationships
- E.g., "Put the red cube in front of the blue bin"

**libero_object** (10 tasks):
- Pick and manipulate objects  
- E.g., "Pick up the bottle and put it in the trash"

**libero_goal** (10 tasks):
- Reach goal states
- E.g., "Make the robot grasp the bowl"

**libero_10** (10 tasks):
- Mixed manipulation tasks
- E.g., "Open the laptop" (hardest)

## Full Config Example (save as `eval.yaml`)

```yaml
ckpt: runs/libero_hfastwam/libero_hfastwam_8card_small_vjepa_predictor/checkpoints/weights/step_030000.pt
gpu_id: 0
model: hfastwam_small_vjepa_predictor
mixed_precision: bf16
seed: 42

EVALUATION:
  task_suite_name: libero_spatial
  task_id: 0
  num_trials: 5
  num_inference_steps: 20
  replan_steps: 5
  action_horizon: null
  text_cfg_scale: 1.0
  negative_prompt: ""
  visualize_future_video: false
  output_dir: ./evaluate_results
  device: cuda
```

Use with:
```bash
python experiments/libero/eval_libero_single.py --config-path . --config-name eval
```

## See Also

- Full documentation: `EVALUATION_GUIDE.md`
- Training script: `scripts/run_libero_hfastwam_8card_small_vjepa_predictor.sh`
- Trainer code: `src/fastwam/trainer.py` (method: `Trainer.evaluate()`)
- Eval script: `experiments/libero/eval_libero_single.py`
