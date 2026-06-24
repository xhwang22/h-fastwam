# FastWAM Architecture - Complete Technical Specification

## Table of Contents
1. [Overview](#overview)
2. [Video Expert (WanVideoDiT)](#video-expert-wandvidedit)
3. [Action Expert (ActionDiT)](#action-expert-actiondit)
4. [VAE Architecture](#vae-architecture)
5. [Mixture of Transformers (MoT)](#mixture-of-transformers-mot)
6. [FastWAM Main Class](#fastwam-main-class)
7. [Training Loss Data Flow](#training-loss-data-flow)
8. [Attention Mechanisms](#attention-mechanisms)

---

## Overview

FastWAM is a Diffusion Transformer-based world model that unifies video generation and action prediction through:
- **Video Expert**: WanVideoDiT transformer processing patch-tokenized video latents
- **Action Expert**: ActionDiT transformer processing action sequences
- **Mixture of Transformers (MoT)**: Mixed-attention architecture combining video and action experts
- **VAE**: Temporal video compression with CausalConv3d operations
- **Flow Matching**: Noise scheduling for diffusion-based training

---

## Video Expert (WanVideoDiT)

### Architecture Summary
- **Hidden Dimension**: 1280
- **Number of Heads**: 20
- **Attention Head Dimension**: 64 (total: 1280 = 20 × 64)
- **FFN Dimension**: 3456
- **Number of Layers**: 24 DiT blocks
- **Patch Size**: (1, 16, 16) → kernel_size=stride=patch_size for Conv3d

### Input Processing (`pre_dit` method - lines 509-620)

#### Step 1: Patchify Video Latents
```python
# Line 367-368 patch_embedding definition:
nn.Conv3d(in_dim, hidden_dim, kernel_size=patch_size, stride=patch_size)
# For input [B, C, T, H, W] with C=16, patch_size=(1,16,16)
# Output: [B, hidden_dim=1280, T, H/16, W/16]
```
- Input shape: `[B, 16, T, H, W]` (VAE latents)
- Conv3d kernel: (1, 16, 16), stride: (1, 16, 16)
- Output: `[B, 1280, T, H//16, W//16]`

#### Step 2: Time Embedding with Per-Token Timesteps
- For each frame in video, create sinusoidal positional embeddings
- Generate per-frame time embeddings using `sinusoidal_embedding_1d(freq_dim, timestep)`
- Expand to match token sequence length (one timestep per token)
- Modulation projection: `time_projection(t)` → outputs 6×hidden_dim for AdaLN

#### Step 3: Text Embedding via Text Encoder
- Input: Text tokens from CLIP/language model
- Processing: 2-layer linear projection with GELU activation
- Output: `[B, L_text, hidden_dim=1280]`

#### Step 4: Action Embedding with Group Causal Masking
- If actions provided, project to hidden_dim
- Build group causal mask allowing actions to attend to **only first video frame**
- Temporal grouping: actions grouped by frame index

#### Step 5: Flatten and Concatenate Tokens
- Flatten spatial dimensions: `[B, 1280, T, H//16, W//16]` → `[B, T×H//16×W//16, 1280]`
- Concatenate: `[tokens, text_emb, action_emb]` → `[B, video_seq_len + text_len + action_len, 1280]`

#### Step 6: Rotary Position Embeddings (RoPE)
- **3D RoPE**: Independent frequencies for temporal, height, width dimensions
- Precomputed frequencies: `precompute_freqs_cis(attn_head_dim=64, end=1024)`
- Applied in self-attention to Q, K projections

#### Step 7: Context Mask Expansion
- Expand text mask to match sequence length: `[B, 1, 1, L_text]` → `[B, video_seq_len, L_text]`
- Allows each token to attend to corresponding text tokens

### DiT Block Structure (24 layers)

Each block contains:

#### 1. **Self-Attention** (`SelfAttention` class, lines 171-196)
```python
# Input: [B, T, 1280]
# Q, K, V projections: Linear(hidden_dim, hidden_dim)
# Split into num_heads: [B, T, 20, 64]
# Apply RoPE: rotary embeddings on Q, K only
# Attention: softmax(Q·K^T / sqrt(64)) · V
# Output projection: Linear(hidden_dim, hidden_dim)
# Output: [B, T, 1280]
```

#### 2. **Cross-Attention** (`CrossAttention` class, lines 198-221)
```python
# Query source: video/action tokens [B, T_video, 1280]
# Key/Value source: text embeddings [B, L_text, 1280]
# Q projection: Linear(hidden_dim, hidden_dim)
# K, V projections: Linear(hidden_dim, hidden_dim)
# NO RoPE applied to cross-attention
# Attention weight mask applied: [B, video_seq_len, L_text]
# Output: [B, T_video, 1280]
```

#### 3. **Feed-Forward Network (FFN)**
```python
# Layer 1: Linear(1280, 3456) + GELU
# Layer 2: Dropout(0.1)
# Layer 3: Linear(3456, 1280)
# Residual connection
# Output: [B, T, 1280]
```

#### 4. **Adaptive Layer Normalization (AdaLN)**
For each block, compute 6 modulation parameters from time embedding:
```python
modulation = nn.Parameter(torch.randn(1, 2, hidden_dim=1280) / sqrt(1280))
# modulation shape: [1, 2, 1280]
# Added to t_mod: [B, 6, 1280]
# Yields: [shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp]
```

**Modulation application**:
- Self-Attn: `x_norm = norm(x) * (1 + scale_msa) + shift_msa`
- Self-Attn output: `x = x + gate_msa * attn_out`
- FFN: `x_norm = norm(x) * (1 + scale_mlp) + shift_mlp`
- FFN output: `x = x + gate_mlp * ffn_out`

#### 5. **Gradient Checkpointing** (Optional)
- If `use_gradient_checkpointing=True`, recompute forward pass during backward
- Reduces memory footprint for 24-layer model

### Output Processing (`post_dit` method - lines 622-626)

#### Step 1: Head Modulation
```python
shift, scale = (modulation + t).chunk(2, dim=1)  # Final time modulation
x_out = norm(x) * (1 + scale) + shift
```

#### Step 2: Linear Projection to Action Dimension
```python
pred = head(x_out)  # Linear(1280, 16) → predicts latent deltas
```

#### Step 3: Unpatchify
```python
# Reshape from [B, seq_len, 1280] to [B, 1280, T, H//16, W//16]
# Conv3d transpose / pixel shuffle to restore spatial resolution
# Output: [B, 16, T, H, W]  (matches VAE latent space)
```

### Attention Masking (`build_video_to_video_mask` - lines 473-507)

Three causal modes:

#### Mode 1: Bidirectional (No Causal Mask)
```python
# Tokens can attend to all previous AND future tokens
# Mask: all True (no restriction)
```

#### Mode 2: Per-Frame Causal
```python
# Within each frame: bidirectional attention
# Across frames: causal (can only attend to current + previous frames)
# Frame i tokens attend to frames 0...i only
```

#### Mode 3: First-Frame Causal (Default)
```python
# All tokens attend to first frame
# Within same frame: full bidirectional
# Across frames: causal
# Frame 0 attended by all
# Frame i (i>0): tokens attend to frames 0...i
```

---

## Action Expert (ActionDiT)

### Architecture Summary
- **Hidden Dimension**: 1280 (must match video expert)
- **Number of Heads**: 20 (must match video expert)
- **Attention Head Dimension**: 64 (must match video expert)
- **FFN Dimension**: 3456
- **Number of Layers**: 24 DiT blocks (must match video expert)
- **Action Dimension**: 7 (robotic arm action)
- **Frequency Dimension**: 256 (for sinusoidal embeddings)

### Module Initialization (`__init__` - lines 45-101)

```python
self.action_encoder = nn.Linear(action_dim=7, hidden_dim=1280)
    # Projects raw action [B, T_action, 7] → [B, T_action, 1280]

self.text_embedding = nn.Sequential(
    nn.Linear(text_dim, hidden_dim=1280),
    nn.GELU(approximate="tanh"),
    nn.Linear(hidden_dim=1280, hidden_dim=1280),
)
    # 2-layer MLP for text conditioning
    # Input: [B, L_text, text_dim]
    # Output: [B, L_text, 1280]

self.time_embedding = nn.Sequential(
    nn.Linear(freq_dim=256, hidden_dim=1280),
    nn.SiLU(),
    nn.Linear(hidden_dim=1280, hidden_dim=1280),
)
    # 2-layer MLP for timestep embeddings
    # Input: [B, freq_dim=256] from sinusoidal_embedding_1d
    # Output: [B, 1280]

self.time_projection = nn.Sequential(
    nn.SiLU(),
    nn.Linear(hidden_dim=1280, hidden_dim*6=7680),
)
    # Projects time embedding to 6 modulation parameters per block
    # Output: [B, 7680] → reshaped to [B, 6, 1280]

self.blocks = nn.ModuleList([
    DiTBlock(hidden_dim=1280, attn_head_dim=64, num_heads=20, 
             ffn_dim=3456, eps=1e-6)
    for _ in range(num_layers=24)
])
    # 24 identical DiT blocks (same architecture as video expert)

self.head = nn.Linear(hidden_dim=1280, action_dim=7)
    # Projects hidden features back to action dimension
    # Output shape: [B, T_action, 7]

self.freqs = precompute_freqs_cis(attn_head_dim=64, end=1024)
    # Precomputed RoPE frequencies for 1D sequence attention
```

### Input Processing (`pre_dit` method - lines 226-299)

#### Step 1: Input Validation
- Validate `action_tokens` shape: `[B, T_action, action_dim=7]` (3D)
- Validate `timestep` shape: `[B]` or `[1]` (1D)
- Validate `context` shape: `[B, L_text, text_dim]` (3D)
- If timestep length is 1 and batch_size > 1 (inference mode), expand to batch_size

#### Step 2: Generate Sinusoidal Time Embeddings
```python
t = self.time_embedding(sinusoidal_embedding_1d(freq_dim=256, timestep))
# Output: [B, 1280]
```

#### Step 3: Project Time to Modulation Parameters
```python
t_mod = self.time_projection(t).unflatten(1, (6, hidden_dim=1280))
# Output: [B, 6, 1280] for 6 AdaLN parameters per block
```

#### Step 4: Encode Actions
```python
tokens = self.action_encoder(action_tokens)
# Input: [B, T_action, 7]
# Output: [B, T_action, 1280]
```

#### Step 5: Embed Text Context
```python
context_emb = self.text_embedding(context)
# Input: [B, L_text, text_dim]
# Output: [B, L_text, 1280]
```

#### Step 6: Build Context Attention Mask
- Create mask: `[B, T_action, L_text]` all True
- Expand: `context_mask.unsqueeze(1).expand(-1, T_action, -1)`
- Allows each action token to attend to all text tokens

#### Step 7: Extract RoPE Frequencies
```python
freqs = self.freqs[:seq_len].view(seq_len, 1, -1).to(tokens.device)
# Select first seq_len frequencies from precomputed cache
# Output: [seq_len, 1, attn_head_dim//2=32]
```

#### Step 8: Return Pre-Processing Dictionary
```python
return {
    "tokens": tokens,           # [B, T_action, 1280]
    "freqs": freqs,             # [seq_len, 1, 32]
    "t": t,                     # [B, 1280]
    "t_mod": t_mod,             # [B, 6, 1280]
    "context": context_emb,     # [B, L_text, 1280]
    "context_mask": context_attn_mask,  # [B, T_action, L_text]
    "meta": {
        "batch_size": batch_size,
        "seq_len": seq_len,
    },
}
```

### DiT Blocks
- **24 identical blocks** (same as video expert)
- Each block: Self-Attn → Cross-Attn → FFN with AdaLN modulation
- RoPE applied in self-attention (1D for action sequences)
- No causal masking in action expert (full bidirectional attention within sequence)

### Output Processing (`post_dit` method - lines 301-302)

```python
def post_dit(self, tokens: torch.Tensor, pre_state: Dict) -> torch.Tensor:
    return self.head(tokens)
```

- Simple linear projection: `[B, T_action, 1280]` → `[B, T_action, 7]`
- No modulation applied (unlike video expert)
- Direct action space reconstruction

---

## VAE Architecture

### Overview
- **Input**: Video frames `[B, 3, T, H, W]` (e.g., H=256, W=256)
- **Output Latent Dimension**: z_dim = 16
- **Temporal Downsample Factor**: 4 (T → T/4)
- **Spatial Downsample Factor**: 8 (H/8, W/8)
- **Upsampling Factor**: 8 (reconstruction)

### Encoder Structure (`VideoVAE_` class encoder, lines 516-616)

#### Layer 1: Initial Convolution
```python
CausalConv3d(in_channels=3, out_channels=128, kernel_size=3, padding=1)
# Input: [B, 3, T, H, W]
# Output: [B, 128, T, H, W]  (no downsampling)
```

#### Stage 1: Downsampling (2× spatial)
```python
# ResBlock + ResBlock (2 blocks)
# CausalConv3d(128, 128, kernel_size=3, padding=1) [2×]
# Spatial downsampling 2×: stride=(1, 2, 2)
# Temporal downsampling 2×: stride=(2, 1, 1)
# Output: [B, 128, T/2, H/2, W/2]
```

#### Stage 2: Downsampling (2× spatial)
```python
# ResBlock + ResBlock (2 blocks)
# CausalConv3d(128→256 in first, 256 in residual), kernel_size=3, padding=1
# Spatial downsampling 2×: stride=(1, 2, 2)
# Temporal unchanged (no temporal downsample in stage 1)
# Output: [B, 256, T/2, H/4, W/4]
```

#### Stage 3: Downsampling (2× spatial)
```python
# ResBlock + ResBlock (2 blocks)
# CausalConv3d(256→512, 512), kernel_size=3, padding=1
# Spatial downsampling 2×: stride=(1, 2, 2)
# Temporal downsampling 2×: stride=(2, 1, 1)
# Output: [B, 512, T/4, H/8, W/8]
```

#### Stage 4: Downsampling (No spatial, no temporal)
```python
# ResBlock + ResBlock (2 blocks)
# CausalConv3d(512→512, 512), kernel_size=3, padding=1
# No spatial/temporal downsampling
# Output: [B, 512, T/4, H/8, W/8]
```

#### Middle Blocks
```python
# ResBlock(512)
# AttentionBlock(512)  # Self-attention on latent space
# ResBlock(512)
# Output: [B, 512, T/4, H/8, W/8]
```

#### Output Head
```python
CausalConv3d(in_channels=512, out_channels=z_dim=16, kernel_size=3, padding=1)
# Input: [B, 512, T/4, H/8, W/8]
# Output: [B, 16, T/4, H/8, W/8]  (VAE latent space)
```

### Encoder Total Layers
- Initial Conv: 1
- Stage 1: 2 ResBlocks = 2 CausalConv3d pairs = 4 ops
- Stage 2: 2 ResBlocks = 4 ops
- Stage 3: 2 ResBlocks = 4 ops
- Stage 4: 2 ResBlocks = 4 ops
- Middle: 1 ResBlock + AttentionBlock + 1 ResBlock = 3 ops
- Head Conv: 1
- **Total CausalConv3d layers: 1 + 4 + 4 + 4 + 4 + 3 + 1 = 21 layers**

### ResidualBlock Structure (`ResidualBlock` class, lines 268-301)

```python
class ResidualBlock:
    def forward(self, x, t_emb=None, cond=None):
        # 1. LayerNorm
        h = norm1(x)
        # 2. SiLU activation
        h = SiLU(h)
        # 3. CausalConv3d (3×3×3 kernel, padding=1)
        h = conv1(h)
        
        # 4. Timestep/conditioning modulation (optional)
        # h += t_emb projection if provided
        
        # 5. LayerNorm
        h = norm2(h)
        # 6. SiLU activation
        h = SiLU(h)
        # 7. Dropout (0.1 prob)
        h = dropout(h)
        # 8. CausalConv3d (3×3×3 kernel, padding=1)
        h = conv2(h)
        
        # 9. Shortcut connection
        return x + h if x.shape == h.shape else x + conv_shortcut(x)
```

### Decoder Structure (`Decoder3d` class, lines 735-837)

Mirrors encoder with upsampling:
- Initial Conv: 1
- 4 upsampling stages with 2 ResBlocks each: 16 ops
- Middle blocks: 3 ops
- Final Conv to 3 channels: 1
- **Total layers: ~21 (symmetric with encoder)**

### Normalization Statistics (lines 1062-1072)

```python
z_dim = 16 channels
mean = [0.0, 0.0, ..., 0.0]  # 16 values
std = [1.0, 1.0, ..., 1.0]   # 16 values
# Applied during inference for VAE latent normalization
```

---

## Mixture of Transformers (MoT)

### Architecture
Located in `mot.py`, MoT combines video and action experts:

```python
class MoT(nn.Module):
    def __init__(self, mixtures: dict, mot_checkpoint_mixed_attn: bool = True):
        # mixtures: {"video": WanVideoDiT, "action": ActionDiT}
        # mot_checkpoint_mixed_attn: Enable gradient checkpointing
```

### Forward Pass

Input to MoT forward (from training_loss, lines 578-603):
```python
tokens_out = self.mot(
    embeds_all={
        "video": video_tokens,        # [B, N_video, 1280]
        "action": action_tokens,       # [B, N_action, 1280]
    },
    attention_mask=attention_mask,    # [N_video+N_action, N_video+N_action]
    freqs_all={
        "video": video_freqs,          # [N_video, 1, 32]
        "action": action_freqs,        # [N_action, 1, 32]
    },
    context_all={
        "video": {
            "context": video_context,  # [B, L_text, 1280]
            "mask": video_context_mask,# [B, N_video, L_text]
        },
        "action": {
            "context": action_context, # [B, L_text, 1280]
            "mask": action_context_mask# [B, N_action, L_text]
        },
    },
    t_mod_all={
        "video": video_t_mod,          # [B, 6, 1280]
        "action": action_t_mod,        # [B, 6, 1280]
    },
    detach_video_for_action=action_loss_detach_video_expert,
)

# Output:
# tokens_out = {
#     "video": [B, N_video, 1280],
#     "action": [B, N_action, 1280],
# }
```

### Mixed Attention Mechanism
- Video and action tokens **share the same DiT blocks** via mixed attention
- Within each block:
  1. **Video self-attention**: video tokens attend to all video tokens (causal mask)
  2. **Action self-attention**: action tokens attend to all action tokens (no mask)
  3. **Mixed cross-attention**: video ↔ action tokens attend per attention_mask
  4. **Both cross-attend to text context**

### Attention Mask Structure

```python
def _build_mot_attention_mask(video_seq_len, action_seq_len, 
                               video_tokens_per_frame, device):
    # Creates [B, video_seq_len + action_seq_len, 
    #              video_seq_len + action_seq_len] mask
    
    # Block structure:
    # ┌─────────────────┬──────────────────┐
    # │ Video-to-Video  │ Video-to-Action  │
    # ├─────────────────┼──────────────────┤
    # │ Action-to-Video │ Action-to-Action │
    # └─────────────────┴──────────────────┘
    
    # Video-to-Video: Causal mask per frame
    #   - Tokens within frame: bidirectional
    #   - Across frames: tokens can attend to current + previous frames
    
    # Video-to-Action: ALL True (video attends to all actions)
    
    # Action-to-Video: Group causal mask
    #   - Each action group attends only to first video frame (frame 0)
    #   - Based on tokens_per_frame grouping
    
    # Action-to-Action: ALL True (full bidirectional)
```

---

## FastWAM Main Class

### Initialization (`__init__` - lines 22-98)

```python
class FastWAM(nn.Module):
    def __init__(
        self,
        video_expert: WanVideoDiT,        # Loaded pretrained model
        action_expert: ActionDiT,         # Loaded or randomly initialized
        mot: MoT,                         # Mixed attention orchestrator
        vae: WanVideoVAE,                 # Video compression
        text_encoder=None,                # CLIP or similar
        tokenizer=None,                   # Tokenizer for text
        text_dim: int = 768,              # Text embedding dimension
        proprio_dim: Optional[int] = None,# Proprioceptive sensor dim
        device: str = "cpu",
        torch_dtype: torch.dtype = torch.float32,
        video_train_shift: float = 5.0,   # Flow matching shift
        video_infer_shift: float = 5.0,
        video_num_train_timesteps: int = 1000,
        action_train_shift: float = 5.0,
        action_infer_shift: float = 5.0,
        action_num_train_timesteps: int = 1000,
        loss_lambda_video: float = 1.0,   # Weight for video loss
        loss_lambda_action: float = 1.0,  # Weight for action loss
        action_loss_detach_video_expert: bool = False,
        visual_encoder=None,              # DINO/V-JEPA2 alternative
    ):
```

### Component Assembly
```python
self.video_expert = video_expert       # WanVideoDiT (24 blocks, 1280d)
self.action_expert = action_expert     # ActionDiT (24 blocks, 1280d)
self.mot = mot                         # MoT orchestrator
self.dit = self.mot  # Alias for trainer compatibility

self.vae = vae                         # Video autoencoder
self.use_visual_encoder = isinstance(visual_encoder, BaseVisualEncoder)
if use_visual_encoder:
    self.visual_encoder = visual_encoder  # DINO/V-JEPA2
else:
    self.visual_encoder = vae  # Fallback to VAE
    
self.text_encoder = text_encoder       # CLIP encoder
self.tokenizer = tokenizer             # BPE tokenizer
self.text_dim = text_dim               # 768 (CLIP)
self.proprio_dim = proprio_dim         # Optional proprioceptive features
```

### Proprioceptive Encoder (lines 67-70)

```python
if proprio_dim is not None:
    self.proprio_encoder = nn.Linear(proprio_dim, text_dim)
else:
    self.proprio_encoder = None
```

- Simple 1-layer linear projection
- Maps proprioceptive state (e.g., robot joint angles) to text_dim=768
- Concatenated to text context before experts

### Noise Schedulers

```python
self.train_video_scheduler = WanContinuousFlowMatchScheduler(
    num_train_timesteps=1000,
    shift=5.0,  # Flow matching parameter
)
self.infer_video_scheduler = WanContinuousFlowMatchScheduler(
    num_train_timesteps=1000,
    shift=5.0,
)
self.train_action_scheduler = WanContinuousFlowMatchScheduler(
    num_train_timesteps=1000,
    shift=5.0,
)
self.infer_action_scheduler = WanContinuousFlowMatchScheduler(
    num_train_timesteps=1000,
    shift=5.0,
)
# Aliases for backwards compatibility
self.train_scheduler = self.train_video_scheduler
self.infer_scheduler = self.infer_video_scheduler
```

### Loss Weighting Parameters

```python
self.loss_lambda_video = 1.0  # Video loss weight
self.loss_lambda_action = 1.0  # Action loss weight
self.action_loss_detach_video_expert = False
    # If True, detach video expert gradients for action loss
```

---

## Training Loss Data Flow

### Overview
The `training_loss()` method (lines 522-643) orchestrates the complete training forward pass.

### Step 1-2: Build Inputs and Encode

```python
# Line 523
inputs = self.build_inputs(sample, tiled=False)
# Returns:
# {
#     "input_latents": [B, 16, T, H//8, W//8],
#     "first_frame_latents": [B, 16, 1, H//8, W//8] or None,
#     "context": [B, L_text, 768],
#     "context_mask": [B, L_text],
#     "action": [B, T_action, 7],
#     "action_is_pad": [B, T_action],
#     "image_is_pad": [B, T],
#     "fuse_vae_embedding_in_latents": bool,
# }

input_latents = inputs["input_latents"]  # [B, 16, T, H//8, W//8]
batch_size = input_latents.shape[0]
```

### Step 3: Add Video Noise (Lines 532-542)

```python
# Sample random noise
noise_video = torch.randn_like(input_latents)  # [B, 16, T, H//8, W//8]

# Sample random training timesteps
timestep_video = self.train_video_scheduler.sample_training_t(
    batch_size=batch_size,
    device=self.device,
    dtype=input_latents.dtype,
)  # [B,] with values in [0, 1]

# Add noise to latents (flow matching)
latents = self.train_video_scheduler.add_noise(
    input_latents,      # [B, 16, T, H//8, W//8]
    noise_video,        # [B, 16, T, H//8, W//8]
    timestep_video,     # [B,]
)  # [B, 16, T, H//8, W//8] (noisy version)

# Compute training target (depends on loss type: MSE on noise or velocity)
target_video = self.train_video_scheduler.training_target(
    input_latents,      # Original
    noise_video,        # Random noise
    timestep_video,     # Timestep
)  # [B, 16, T, H//8, W//8] (target for diffusion)

# If using first-frame conditioning, keep frame 0 unchanged
if inputs["first_frame_latents"] is not None:
    latents[:, :, 0:1] = inputs["first_frame_latents"]
```

### Step 4: Add Action Noise (Lines 544-551)

```python
# Sample random noise for actions
noise_action = torch.randn_like(action)  # [B, T_action, 7]

# Sample random training timesteps for actions
timestep_action = self.train_action_scheduler.sample_training_t(
    batch_size=batch_size,
    device=self.device,
    dtype=action.dtype,
)  # [B,]

# Add noise to actions
noisy_action = self.train_action_scheduler.add_noise(
    action,             # [B, T_action, 7]
    noise_action,       # [B, T_action, 7]
    timestep_action,    # [B,]
)  # [B, T_action, 7]

# Compute action training target
target_action = self.train_action_scheduler.training_target(
    action,             # Original
    noise_action,       # Random noise
    timestep_action,    # Timestep
)  # [B, T_action, 7]
```

### Step 5: Video Expert Pre-Processing (Lines 553-560)

```python
video_pre = self.video_expert.pre_dit(
    x=latents,                           # [B, 16, T, H//8, W//8]
    timestep=timestep_video,             # [B,]
    context=context,                     # [B, L_text, 768]
    context_mask=context_mask,           # [B, L_text]
    action=action,                       # [B, T_action, 7] (for embedding)
    fuse_vae_embedding_in_latents=inputs["fuse_vae_embedding_in_latents"],
)
# Returns:
# {
#     "tokens": [B, N_video, 1280],
#     "freqs": [N_video, 1, 32],
#     "t": [B, 1280],
#     "t_mod": [B, 6, 1280],
#     "context": [B, L_text, 1280],
#     "context_mask": [B, N_video, L_text],
#     "meta": {"batch_size": B, "tokens_per_frame": int, ...},
# }
```

### Step 6: Action Expert Pre-Processing (Lines 562-567)

```python
action_pre = self.action_expert.pre_dit(
    action_tokens=noisy_action,          # [B, T_action, 7]
    timestep=timestep_action,            # [B,]
    context=context,                     # [B, L_text, 768]
    context_mask=context_mask,           # [B, L_text]
)
# Returns:
# {
#     "tokens": [B, T_action, 1280],
#     "freqs": [T_action, 1, 32],
#     "t": [B, 1280],
#     "t_mod": [B, 6, 1280],
#     "context": [B, L_text, 1280],
#     "context_mask": [B, T_action, L_text],
#     "meta": {"batch_size": B, "seq_len": T_action},
# }
```

### Step 7: Extract Token Sequences

```python
video_tokens = video_pre["tokens"]       # [B, N_video, 1280]
action_tokens = action_pre["tokens"]     # [B, T_action, 1280]
```

### Step 8: Build MoT Attention Mask (Lines 572-577)

```python
attention_mask = self._build_mot_attention_mask(
    video_seq_len=video_tokens.shape[1],         # N_video
    action_seq_len=action_tokens.shape[1],       # T_action
    video_tokens_per_frame=int(video_pre["meta"]["tokens_per_frame"]),
    device=video_tokens.device,
)
# Returns: [N_video + T_action, N_video + T_action] attention mask
```

### Step 9: Forward Through MoT (Lines 578-603)

```python
tokens_out = self.mot(
    embeds_all={
        "video": video_tokens,           # [B, N_video, 1280]
        "action": action_tokens,         # [B, T_action, 1280]
    },
    attention_mask=attention_mask,       # [N_video+T_action, N_video+T_action]
    freqs_all={
        "video": video_pre["freqs"],     # [N_video, 1, 32]
        "action": action_pre["freqs"],   # [T_action, 1, 32]
    },
    context_all={
        "video": {
            "context": video_pre["context"],         # [B, L_text, 1280]
            "mask": video_pre["context_mask"],       # [B, N_video, L_text]
        },
        "action": {
            "context": action_pre["context"],        # [B, L_text, 1280]
            "mask": action_pre["context_mask"],      # [B, T_action, L_text]
        },
    },
    t_mod_all={
        "video": video_pre["t_mod"],     # [B, 6, 1280]
        "action": action_pre["t_mod"],   # [B, 6, 1280]
    },
    detach_video_for_action=self.action_loss_detach_video_expert,
)
# Returns:
# {
#     "video": [B, N_video, 1280],
#     "action": [B, T_action, 1280],
# }
```

### Step 10: Video Expert Post-Processing (Line 605)

```python
pred_video = self.video_expert.post_dit(
    tokens_out["video"],                 # [B, N_video, 1280]
    video_pre,
)
# Returns: [B, 16, T or T-1, H//8, W//8]
```

### Step 11: Action Expert Post-Processing (Line 607)

```python
pred_action = self.action_expert.post_dit(
    tokens_out["action"],                # [B, T_action, 1280]
    action_pre,
)
# Returns: [B, T_action, 7]
```

### Step 12: Handle First-Frame Conditioning (Lines 609-612)

```python
include_initial_video_step = inputs["first_frame_latents"] is None
if inputs["first_frame_latents"] is not None:
    # Remove first frame from predictions (was conditioned)
    pred_video = pred_video[:, :, 1:]    # [B, 16, T-1, H//8, W//8]
    target_video = target_video[:, :, 1:]  # [B, 16, T-1, H//8, W//8]
```

### Step 13: Compute Video Loss (Lines 614-623)

```python
# Compute per-sample MSE loss
loss_video_per_sample = self._compute_video_loss_per_sample(
    pred_video=pred_video,               # [B, 16, T/T-1, H//8, W//8]
    target_video=target_video,           # [B, 16, T/T-1, H//8, W//8]
    image_is_pad=image_is_pad,           # [B, T]
    include_initial_video_step=include_initial_video_step,
)
# Returns: [B,] with per-sample losses

# Apply timestep weighting from scheduler
video_weight = self.train_video_scheduler.training_weight(timestep_video)
# Returns: [B,] with scalar weights per sample

# Weighted mean loss
loss_video = (loss_video_per_sample * video_weight).mean()  # scalar
```

### Step 14: Compute Action Loss (Lines 625-636)

```python
# Compute per-token MSE loss
action_loss_token = F.mse_loss(
    pred_action.float(),                 # [B, T_action, 7]
    target_action.float(),               # [B, T_action, 7]
    reduction="none",
).mean(dim=2)  # Reduce over action dimension → [B, T_action]

# Mask padding tokens
if action_is_pad is not None:
    valid = (~action_is_pad).to(dtype=action_loss_token.dtype)  # [B, T_action]
    valid_sum = valid.sum(dim=1).clamp(min=1.0)  # [B,]
    action_loss_per_sample = (action_loss_token * valid).sum(dim=1) / valid_sum
else:
    action_loss_per_sample = action_loss_token.mean(dim=1)

# Apply timestep weighting
action_weight = self.train_action_scheduler.training_weight(timestep_action)
# Returns: [B,]

# Weighted mean loss
loss_action = (action_loss_per_sample * action_weight).mean()  # scalar
```

### Step 15: Total Loss and Logging (Lines 638-643)

```python
# Combine weighted losses
loss_total = (self.loss_lambda_video * loss_video + 
              self.loss_lambda_action * loss_action)

loss_dict = {
    "loss_video": self.loss_lambda_video * float(loss_video.detach().item()),
    "loss_action": self.loss_lambda_action * float(loss_action.detach().item()),
}

return loss_total, loss_dict
```

---

## Attention Mechanisms Summary

### Self-Attention (Video Frames)
- **Sequence**: Flattened video tokens (spatial dimensions collapsed)
- **Mask Type**: Optional per-frame causal or bidirectional
- **RoPE**: 3D (temporal, height, width) frequencies
- **Computation**: Scaled dot-product with head_dim=64

### Cross-Attention (Text Conditioning)
- **Query**: Video/action tokens
- **Key/Value**: Text embeddings from CLIP
- **Mask Type**: Text attention mask (all True)
- **RoPE**: None (only applied to self-attention)

### Cross-Attention (Action-to-Video in MoT)
- **Query**: Action tokens
- **Key/Value**: Video tokens
- **Mask Type**: Group causal (action group attends to first video frame only)
- **RoPE**: Applied separately (action 1D, video 3D)

### Group Causal Masking (Action ↔ Video)
- Actions are grouped by temporal alignment
- Each action group can attend only to **first frame of video** (frame 0)
- This enforces temporal consistency: actions condition on initial observation

---

## Key Training Features

### Diffusion/Flow Matching
- **Noise Schedule**: WanContinuousFlowMatchScheduler
- **Timesteps**: 1000 (for both video and action)
- **Shift Parameter**: 5.0 (scales the diffusion time axis)
- **Per-Sample Weighting**: `training_weight()` applies SNR-based or similar timestep weighting

### Loss Computation
- **Video Loss**: MSE between predicted and target latents
- **Action Loss**: MSE between predicted and target actions
- **Padding Masking**: Ignored padding tokens in both modalities
- **Combined Loss**: Weighted sum with configurable λ_video and λ_action

### Gradient Management
- **Optional**: Detach video expert for action loss (reduce video→action coupling)
- **Checkpointing**: Optional gradient checkpointing for memory efficiency

### First-Frame Conditioning
- Optional: Keep first video frame fixed (not noised)
- Useful for controllable generation from initial state
- Removes first frame from loss computation

---

## Configuration Summary

| Component | Parameter | Value |
|-----------|-----------|-------|
| Video Expert | hidden_dim | 1280 |
| | num_heads | 20 |
| | attn_head_dim | 64 |
| | ffn_dim | 3456 |
| | num_layers | 24 |
| | patch_size | (1, 16, 16) |
| Action Expert | hidden_dim | 1280 |
| | num_heads | 20 |
| | attn_head_dim | 64 |
| | ffn_dim | 3456 |
| | num_layers | 24 |
| | action_dim | 7 |
| VAE | z_dim | 16 |
| | temporal_downsample | 4× |
| | spatial_downsample | 8× |
| | encoder_layers | 21 |
| FastWAM | text_dim | 768 |
| | video_timesteps | 1000 |
| | action_timesteps | 1000 |
| | flow_shift | 5.0 |

