# FastWAM Code Reference - Line Numbers and Quick Lookup

## File Locations

```
/apdcephfs_gy2/share_302533218/shaunxhwang/embodied/FastWAM/
├── src/fastwam/models/wan22/
│   ├── fastwam.py (Main class)
│   ├── wan_video_dit.py (Video expert)
│   ├── action_dit.py (Action expert)
│   ├── wan_video_vae.py (VAE)
│   ├── mot.py (Mixture of Transformers)
│   ├── schedulers/scheduler_continuous.py (Noise scheduling)
│   └── helpers/
│       ├── gradient.py (Gradient checkpointing)
│       └── loader.py (Model loading)
```

---

## wan_video_dit.py (667 lines) - WanVideoDiT

### Class Definition
- **WanVideoDiT** class: Lines 310-667

### Key Methods

| Method | Lines | Input Shapes | Output Shapes | Purpose |
|--------|-------|--------------|---------------|---------|
| `__init__` | 312-406 | - | - | Initialize video expert with 24 DiT blocks |
| `patchify` | 402-408 | `[B, C, T, H, W]` | `[B, hidden_dim, T, H//patch_h, W//patch_w]` | Convert to patch embeddings |
| `build_video_to_video_mask` | 473-507 | `(video_seq_len, tokens_per_frame, ...)` | `[video_seq_len, video_seq_len]` | Create causal attention mask |
| `pre_dit` | 509-620 | Raw inputs (latents, text, timestep) | Dict with tokens, freqs, t_mod, context | Prepare for transformer |
| `forward` | 443-468 | `(x, timestep, context, ...)` | `[B, C, T, H, W]` | Full forward pass |
| `post_dit` | 622-626 | Transformed tokens + pre_state | `[B, out_dim, T, H, W]` | Project back to latent space |

### Module Components

| Component | Lines | Details |
|-----------|-------|---------|
| `patch_embedding` | 367-368 | `nn.Conv3d(in_dim, hidden_dim, kernel_size=patch_size, stride=patch_size)` |
| `text_embedding` | 369-373 | 2-layer MLP: `Linear(text_dim, hidden_dim) → GELU → Linear(hidden_dim, hidden_dim)` |
| `time_embedding` | 374-378 | 2-layer MLP: `Linear(freq_dim, hidden_dim) → SiLU → Linear(hidden_dim, hidden_dim)` |
| `time_projection` | 379-381 | `Sequential(SiLU(), Linear(hidden_dim, hidden_dim*6))` |
| `blocks` | 382-392 | 24 DiTBlock instances |
| `head` | 393-395 | `nn.Linear(hidden_dim, out_dim)` for output projection |
| `freqs` | 396-398 | Precomputed 3D RoPE frequencies |

### DiTBlock Class
- **DiTBlock** class: Lines 230-268

```python
# Structure:
# 1. SelfAttention (with RoPE)
# 2. CrossAttention (text)
# 3. FFN
# 4. AdaLN modulation (6 parameters: shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp)
```

### SelfAttention Class
- **SelfAttention** class: Lines 171-196
- Q, K projections with RoPE
- Scaled dot-product attention: `attn = softmax(Q·K^T / sqrt(head_dim))`

### CrossAttention Class
- **CrossAttention** class: Lines 198-221
- Q from video tokens, K/V from text
- No RoPE applied

### RoPE Function
- **precompute_freqs_cis** function: Lines 57-71
- Computes: `freq_dim // 2` frequency pairs
- For 3D: separate frequencies for temporal, height, width

### sinusoidal_embedding_1d Function
- **sinusoidal_embedding_1d** function: Lines 73-94
- Input: `(freq_dim, timestep)` with shape `timestep: [B,]`
- Output: `[B, freq_dim]` sinusoidal embeddings

---

## action_dit.py (338 lines) - ActionDiT

### Class Definition
- **ActionDiT** class: Lines 32-337

### Key Methods

| Method | Lines | Input Shapes | Output Shapes | Purpose |
|--------|-------|--------------|---------------|---------|
| `__init__` | 45-101 | - | - | Initialize action expert with 24 DiT blocks |
| `from_pretrained` | 112-224 | `(config, path, ...)` | ActionDiT instance | Load from checkpoint |
| `backbone_key_set` | 104-109 | Keys | set[str] | Filter backbone keys (skip action_encoder, head) |
| `pre_dit` | 226-299 | Action tokens, timestep, context | Dict with tokens, freqs, t_mod, context | Prepare for transformer |
| `post_dit` | 301-302 | Tokens, pre_state | `[B, T_action, action_dim]` | Project to action space |
| `forward` | 304-337 | Action tokens, timestep, context | `[B, T_action, action_dim]` | Full forward pass |

### Module Components

| Component | Lines | Details |
|-----------|-------|---------|
| `action_encoder` | 74 | `nn.Linear(action_dim, hidden_dim)` |
| `text_embedding` | 75-79 | 2-layer MLP (same as video expert) |
| `time_embedding` | 80-83 | 2-layer MLP (same as video expert) |
| `time_projection` | 85 | `Sequential(SiLU(), Linear(hidden_dim, hidden_dim*6))` |
| `blocks` | 86-97 | 24 DiTBlock instances (shared with video expert in MoT) |
| `head` | 98 | `nn.Linear(hidden_dim, action_dim)` |
| `freqs` | 99 | Precomputed 1D RoPE frequencies |

### ActionHead Class
- **ActionHead** class: Lines 18-29
- Used for final modulation and projection
- Modulation: `shift, scale = (modulation + t).chunk(2, dim=1)`

### Backbone Key Filtering
- **ACTION_BACKBONE_SKIP_PREFIXES**: Lines 33
  - Skips: `("action_encoder.", "head.")`
  - Keeps: All block parameters for loading from pretrained video expert

- **ACTION_BACKBONE_META_KEYS**: Lines 34-43
  - Metadata to validate: hidden_dim, ffn_dim, num_layers, num_heads, attn_head_dim, text_dim, freq_dim, eps

---

## wan_video_vae.py (1385 lines) - Video VAE

### Class Hierarchy

```
├── WanVideoVAE (wrapper, lines 1057-1078)
│   ├── VideoVAE_ (encoder class, lines 516-616)
│   ├── Decoder3d (decoder class, lines 735-837)
│   └── config: z_dim=16, temporal_downsample=4, upsampling_factor=8
```

### WanVideoVAE Wrapper Class
- **WanVideoVAE** class: Lines 1057-1078
- Initialization parameters:
  - `z_dim = 16`
  - `upsampling_factor = 8`
  - `temporal_downsample_factor = 4`
  - Normalization stats (lines 1062-1072): mean/std for 16 channels

### VideoVAE_ Encoder (lines 516-616)

#### Layer Structure

| Stage | Lines | Input Shape | Output Shape | Components |
|-------|-------|------------|------------|-----------|
| Initial | 539 | `[B, 3, T, H, W]` | `[B, 128, T, H, W]` | CausalConv3d(3, 128) |
| Stage 1 | 542-549 | `[B, 128, T, H, W]` | `[B, 128, T/2, H/2, W/2]` | 2 ResBlocks + spatial/temporal downsample |
| Stage 2 | 550-554 | `[B, 128, T/2, H/2, W/2]` | `[B, 256, T/2, H/4, W/4]` | 2 ResBlocks + spatial downsample |
| Stage 3 | 555-560 | `[B, 256, T/2, H/4, W/4]` | `[B, 512, T/4, H/8, W/8]` | 2 ResBlocks + spatial/temporal downsample |
| Stage 4 | 561-567 | `[B, 512, T/4, H/8, W/8]` | `[B, 512, T/4, H/8, W/8]` | 2 ResBlocks (no downsample) |
| Middle | 560-562 | `[B, 512, T/4, H/8, W/8]` | `[B, 512, T/4, H/8, W/8]` | ResBlock + AttentionBlock + ResBlock |
| Head | 565-566 | `[B, 512, T/4, H/8, W/8]` | `[B, 16, T/4, H/8, W/8]` | CausalConv3d(512, z_dim=16) |

### ResidualBlock Class
- **ResidualBlock** class: Lines 268-301
- Pattern: `RMSNorm → SiLU → CausalConv3d(3) → RMSNorm → SiLU → Dropout → CausalConv3d(3) + shortcut`

### CausalConv3d
- **CausalConv3d** class: Not fully shown in excerpt
- Custom Conv3d that maintains causality in temporal dimension
- Used throughout encoder/decoder to preserve temporal structure

### AttentionBlock Class
- **AttentionBlock** class: Lines 342-410
- Self-attention over spatial-temporal dimensions
- Used in middle blocks of encoder/decoder

### Decoder3d Class
- **Decoder3d** class: Lines 735-837
- Mirrors encoder with upsampling
- Transforms `[B, 16, T/4, H/8, W/8]` → `[B, 3, T, H, W]`

### Encoder Total Layer Count
```
Initial Conv:        1
Stage 1 ResBlocks:   2 × (2 CausalConv3d) = 4
Stage 2 ResBlocks:   2 × (2 CausalConv3d) = 4
Stage 3 ResBlocks:   2 × (2 CausalConv3d) = 4
Stage 4 ResBlocks:   2 × (2 CausalConv3d) = 4
Middle blocks:       ResBlock (2) + AttentionBlock + ResBlock (2) = 5
Output Conv:         1
─────────────────────────────────────────────
Total CausalConv3d:  1 + 4 + 4 + 4 + 4 + 5 + 1 = 23 operations
```

---

## fastwam.py - FastWAM Main Class

### Class Definition
- **FastWAM** class: Lines 19-98+

### Key Methods

| Method | Lines | Purpose |
|--------|-------|---------|
| `__init__` | 22-98 | Initialize all components and schedulers |
| `from_wan22_pretrained` | 101-200+ | Load from pretrained Wan2.2 checkpoint |
| `build_inputs` | 351-457 | Encode video, validate shapes, assemble context |
| `training_loss` | 522-643 | Main training forward pass |
| `_compute_video_loss_per_sample` | 498-520 | Compute MSE loss with padding masks |
| `_build_mot_attention_mask` | 460-481 | Create attention mask for mixed modality |
| `_predict_joint_noise` | 646-? | Joint video-action noise prediction |

### Component Assembly (`__init__` lines 22-98)

```python
self.video_expert = video_expert               # WanVideoDiT
self.action_expert = action_expert             # ActionDiT
self.mot = mot                                 # MoT
self.dit = self.mot  # Alias
self.vae = vae                                 # WanVideoVAE
self.use_visual_encoder = isinstance(...)      # Optional DINO/V-JEPA2
self.visual_encoder = visual_encoder or vae    # Fallback to VAE
self.text_encoder = text_encoder               # CLIP/T5
self.tokenizer = tokenizer                     # BPE
self.text_dim = text_dim                       # 768
self.proprio_dim = proprio_dim                 # Optional
self.proprio_encoder = nn.Linear(proprio_dim, text_dim) if proprio_dim else None
```

### Scheduler Assembly (lines 72-90)

```python
self.train_video_scheduler = WanContinuousFlowMatchScheduler(...)
self.infer_video_scheduler = WanContinuousFlowMatchScheduler(...)
self.train_action_scheduler = WanContinuousFlowMatchScheduler(...)
self.infer_action_scheduler = WanContinuousFlowMatchScheduler(...)
# Aliases:
self.train_scheduler = self.train_video_scheduler
self.infer_scheduler = self.infer_video_scheduler
```

### Loss Weights (lines 94-96)
```python
self.loss_lambda_video = 1.0          # Video loss weight
self.loss_lambda_action = 1.0         # Action loss weight
self.action_loss_detach_video_expert = False  # Detach video for action loss
```

### Training Loss Data Flow (lines 522-643)

#### Step-by-Step Breakdown

| Step | Lines | Operation | Input | Output |
|------|-------|-----------|-------|--------|
| 1 | 523 | Build inputs | raw sample | input_latents, context, actions, masks |
| 2 | 524-530 | Extract components | inputs dict | Unpack all data |
| 3 | 532-538 | Noise video | input_latents, noise_video | noisy latents + target |
| 4 | 544-550 | Noise action | action, noise_action | noisy_action + target |
| 5 | 553-560 | Video pre_dit | noisy latents + context | video_pre dict |
| 6 | 562-567 | Action pre_dit | noisy_action + context | action_pre dict |
| 7 | 569-570 | Extract tokens | pre dicts | video_tokens, action_tokens |
| 8 | 572-577 | MoT mask | token shapes | attention_mask |
| 9 | 578-603 | MoT forward | all pre data | tokens_out dict |
| 10 | 605 | Video post_dit | tokens_out["video"] | pred_video |
| 11 | 607 | Action post_dit | tokens_out["action"] | pred_action |
| 12 | 609-612 | Cond handling | first_frame_latents | Remove first frame if conditioned |
| 13 | 614-623 | Video loss | pred_video, target_video | loss_video |
| 14 | 625-636 | Action loss | pred_action, target_action | loss_action |
| 15 | 638-643 | Combine | loss_video, loss_action | loss_total, loss_dict |

### Attention Mask Structure (`_build_mot_attention_mask` lines 460-481)

```python
def _build_mot_attention_mask(
    self,
    video_seq_len: int,           # N_video
    action_seq_len: int,          # N_action
    video_tokens_per_frame: int,  # tokens_per_frame
    device: torch.device,
) -> torch.Tensor:
    # Returns: [video_seq_len + action_seq_len, 
    #           video_seq_len + action_seq_len]
    
    # Block structure:
    # ┌─ Video-Video ─┬─ Video-Action ─┐
    # ├────────────────┼────────────────┤
    # │ Action-Video   │ Action-Action  │
    # └─ (group caus) ─┴─ (all True) ───┘
```

---

## mot.py - Mixture of Transformers

### Class Definition
- **MoT** class (exact line numbers not provided but typically near similar files)

### Key Components
- Takes video and action experts as input
- Performs mixed attention between modalities
- Returns enhanced token representations for each modality

### Forward Signature
```python
def forward(
    self,
    embeds_all: dict,         # {"video": ..., "action": ...}
    attention_mask: Tensor,   # [N_v+N_a, N_v+N_a]
    freqs_all: dict,          # {"video": ..., "action": ...}
    context_all: dict,        # {"video": {...}, "action": {...}}
    t_mod_all: dict,          # {"video": ..., "action": ...}
    detach_video_for_action: bool,
) -> dict:
```

---

## scheduler_continuous.py - WanContinuousFlowMatchScheduler

### Key Methods (typical for flow matching scheduler)

| Method | Purpose |
|--------|---------|
| `sample_training_t(batch_size, device, dtype)` | Sample random timesteps [0, 1] |
| `add_noise(x, noise, t)` | Add noise: α(t)·x + σ(t)·noise |
| `training_target(x, noise, t)` | Compute target for MSE loss |
| `training_weight(t)` | Weight loss by timestep (e.g., SNR weighting) |

---

## Quick Reference Tables

### Dimension Mapping

```
INPUT PIPELINE:
Video:          [B, 3, T, H, W]              e.g., [4, 3, 8, 256, 256]
  ↓ VAE encode
Latents:        [B, 16, T/4, H/8, W/8]       e.g., [4, 16, 2, 32, 32]
  ↓ Patchify
Patches:        [B, T/4×H/8×W/8, 1280]       e.g., [4, 2048, 1280]
  ↓ Concatenate
Video tokens:   [B, N_video, 1280]           e.g., [4, 2048, 1280]

Action:         [B, T_action, 7]             e.g., [4, 8, 7]
  ↓ Encode
Action tokens:  [B, T_action, 1280]          e.g., [4, 8, 1280]

Text:           [B, L_text]                  e.g., [4, 77]
  ↓ Embed
Text context:   [B, L_text, 1280]            e.g., [4, 77, 1280]

Proprio:        [B, T, proprio_dim]          e.g., [4, 8, 10]
  ↓ Encode (if provided)
Proprio emb:    [B, 1, text_dim]             e.g., [4, 1, 768]
  ↓ Concatenate to context
Context:        [B, L_text+1, 1280]          e.g., [4, 78, 1280]

TRANSFORMER PIPELINE:
Combined:       [B, N_v+N_a+L, 1280]         e.g., [4, 2133, 1280]
  ↓ 24 × MoT blocks
Enhanced:       [B, N_v+N_a+L, 1280]         e.g., [4, 2133, 1280]
  ↓ Extract video tokens
Video out:      [B, N_video, 1280]           e.g., [4, 2048, 1280]
  ↓ Unpatchify
Pred latents:   [B, 16, T/4, H/8, W/8]       e.g., [4, 16, 2, 32, 32]

Action out:     [B, T_action, 1280]          e.g., [4, 8, 1280]
  ↓ Project
Pred action:    [B, T_action, 7]             e.g., [4, 8, 7]
```

### Layer Counts Summary

```
Video Expert (WanVideoDiT):
  - DiT blocks: 24
  - Self-attn per block: 1
  - Cross-attn per block: 1
  - FFN per block: 1
  - Total attention layers: 48

Action Expert (ActionDiT):
  - DiT blocks: 24 (same as video expert for MoT)
  - Total attention layers: 48

VAE Encoder:
  - CausalConv3d layers: 23
  - ResBlocks: 10 (2 per stage: 5 stages + 1 middle)
  - AttentionBlocks: 1 (in middle)

VAE Decoder:
  - CausalConv3d layers: ~23 (symmetric)
  - ResBlocks: ~10 (symmetric)
  - AttentionBlocks: ~1 (symmetric)

MoT (Mixture of Transformers):
  - DiT blocks: 24 (shared from video expert)
  - Each block modified for mixed attention
```

### Hidden Dimensions

```
All hidden dimensions across architecture:

Video Expert:
  - hidden_dim: 1280
  - ffn_dim: 3456 (2.7× hidden_dim)
  - attn_head_dim: 64 (1280 / 20 heads)
  - num_heads: 20

Action Expert:
  - hidden_dim: 1280 (must match video)
  - ffn_dim: 3456 (must match video)
  - attn_head_dim: 64 (must match video)
  - num_heads: 20 (must match video)

Text/Context:
  - text_dim: 768 (CLIP dimension)
  - After embedding: 1280 (matched to experts)

VAE:
  - z_dim: 16
  - Encoder channels: [128, 256, 512, 512]
  - Decoder channels: symmetric

Patch Size:
  - (1, 16, 16) → tokens per patch = 1×16×16 = 256 spatial positions per frame
```

---

## Key File Cross-References

### Data Flow Through Files

```
INPUT (sample)
  ↓
fastwam.py::build_inputs() [line 351-457]
  ├─ wan_video_vae.py::VideoVAE_.encode() [lines 516-616]
  └─ Returns: input_latents, context, action, masks
  
  ↓
fastwam.py::training_loss() [line 522-643]
  ├─ scheduler_continuous.py::add_noise() [sample noise & timestep]
  ├─ scheduler_continuous.py::training_target() [compute targets]
  │
  ├─ wan_video_dit.py::pre_dit() [line 509-620]
  ├─ action_dit.py::pre_dit() [line 226-299]
  │
  ├─ fastwam.py::_build_mot_attention_mask() [line 460-481]
  │
  ├─ mot.py::forward() [mixed attention blocks]
  │   └─ Uses wan_video_dit.py::DiTBlock [line 230-268]
  │   └─ Uses action_dit.py::DiTBlock [same architecture]
  │
  ├─ wan_video_dit.py::post_dit() [line 622-626]
  ├─ action_dit.py::post_dit() [line 301-302]
  │
  ├─ fastwam.py::_compute_video_loss_per_sample() [line 498-520]
  ├─ scheduler_continuous.py::training_weight() [timestep weighting]
  │
  └─ Return: loss_total, loss_dict

INFERENCE (similar but deterministic timesteps):
  ├─ fastwam.py::_predict_joint_noise() [line 646-?]
  └─ Uses infer_video_scheduler / infer_action_scheduler
```

