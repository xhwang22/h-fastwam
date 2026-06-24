# H-FastWAM Three-Expert Architecture: Detailed Construction & Dimension Mapping

## Executive Summary

H-FastWAM comprises three experts in a shared Mixture-of-Transformers (MoT) design:
- **Language Expert** (LM-only, random-init, Qwen3-VL or legacy): `hidden=2048, heads=16, head_dim=128, layers=28`
- **Video Expert** (Wan2.2-TI2V-5B, pretrained): `hidden=3072, heads=24, head_dim=128, layers=30, in/out=1024 (DINO space)`
- **Action Expert** (ActionDiT, pretrained): `hidden=1024(?), heads=24, head_dim=128, layers=30` [**CRITICAL MISMATCH**]

**Key Finding**: Action expert has a **dimensional contradiction** — `hidden_dim=1024` but `num_heads × head_dim = 24 × 128 = 3072`. The MoT uses the latter for shared attention space. Language expert is not pretrained and gets dimension-alignment projections in MoT.

---

## 1. Entry Point: `HFastWAM.from_pretrained_fastwam()`

**File**: `src/fastwam/models/hfastwam/hfastwam.py` **Lines**: 2159–2446

### 1.1 Video Expert Loading

```python
# Line 2266–2277
components = load_wan22_ti2v_5b_components(
    device=device, torch_dtype=torch_dtype,
    model_id=model_id,  # default: "Wan-AI/Wan2.2-TI2V-5B"
    tokenizer_model_id=tokenizer_model_id,
    tokenizer_max_len=tokenizer_max_len,
    redirect_common_files=redirect_common_files,
    dit_config=video_dit_config,  # from config/model/hfastwam.yaml
    skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
    skip_video_dit_load_from_pretrain=skip_video_dit_load_from_pretrain,
    load_text_encoder=load_text_encoder,
    skip_vae_load=(dino_visual_encoder is not None),
)
video_expert = components.dit
```

**Video DiT Config** (from `configs/model/hfastwam.yaml` **Lines**: 74–96):
```yaml
video_dit_config:
  has_image_input: false
  patch_size: [1, 2, 2]
  in_dim: 1024              # DINO ViT-L output (DiT-side projection from 48d)
  hidden_dim: 3072          # ← Internal transformer hidden state
  ffn_dim: 14336            # ← FFN intermediate
  freq_dim: 256
  text_dim: 4096
  out_dim: 1024             # ← Flow matching target (back to DINO space)
  num_heads: 24
  attn_head_dim: 128
  num_layers: 30
  eps: 1.0e-06
  seperated_timestep: true
  require_clip_embedding: false
  require_vae_embedding: false
  fuse_vae_embedding_in_latents: true
  use_gradient_checkpointing: true
  video_attention_mask_mode: "first_frame_causal"
  action_conditioned: false
  action_dim: ${data.train.processor.action_output_dim}
  action_group_causal_mask_mode: "group_diagonal"
```

**Video Expert Constructor**:
- **File**: `src/fastwam/models/wan22/wan_video_dit.py` **Lines**: 310–400
- **Key dimension derivation**:
  ```python
  self.hidden_dim = hidden_dim  # 3072
  self.num_heads = num_heads    # 24
  self.attn_head_dim = attn_head_dim  # 128
  self.patch_embedding = nn.Conv3d(in_dim, hidden_dim, ...)  # 1024 → 3072 patchify
  # ... blocks ...
  self.head = Head(dim=hidden_dim, out_dim=out_dim, ...)  # Projects 3072 → 1024
  ```

**Pretrained Weight Loading** (via `load_wan22_ti2v_5b_components`):
- **File**: `src/fastwam/models/wan22/helpers/loader.py` **Lines**: 155–240+
- **Control flags**:
  - `skip_dit_load_from_pretrain`: if True, skips loading for **both** video and action (legacy)
  - `skip_video_dit_load_from_pretrain`: if True, skips loading for **video only** (action still loads)
- **Default**: Both False, so video expert loads Wan2.2-TI2V-5B weights from HuggingFace via `_load_registered_model()` + hash-based detection + shape mismatch filtering (line 120–132).

---

### 1.2 Action Expert Loading

**File**: `src/fastwam/models/hfastwam/hfastwam.py` **Lines**: 2280–2293

```python
action_expert = ActionDiT.from_pretrained(
    action_dit_config=action_dit_config or {},
    action_dit_pretrained_path=action_dit_pretrained_path,  # from cfg: checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt
    skip_dit_load_from_pretrain=skip_dit_load_from_pretrain,
    device=device, torch_dtype=torch_dtype,
)
```

**Action DiT Config** (from `configs/model/hfastwam.yaml` **Lines**: 99–110):
```yaml
action_dit_config:
  action_dim: ${data.train.processor.action_output_dim}
  hidden_dim: 1024            # ← **CRITICAL MISMATCH**
  ffn_dim: 4096
  num_heads: 24               # num_heads × head_dim = 24 × 128 = 3072
  attn_head_dim: 128
  num_layers: 30
  text_dim: 4096
  freq_dim: 256
  eps: 1.0e-06
  use_gradient_checkpointing: true

action_dit_pretrained_path: checkpoints/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt
```

**Action Expert Constructor**:
- **File**: `src/fastwam/models/wan22/action_dit.py` **Lines**: 32–102
- **Key attributes**:
  ```python
  self.hidden_dim = hidden_dim  # 1024 ← mismatch!
  self.action_dim = action_dim
  self.num_heads = num_heads    # 24
  self.attn_head_dim = attn_head_dim  # 128
  self.action_encoder = nn.Linear(action_dim, hidden_dim)  # action → 1024
  self.blocks = nn.ModuleList([DiTBlock(...) for _ in range(num_layers)])
  self.head = nn.Linear(hidden_dim, action_dim)  # 1024 → action
  ```

**Pretrained Weight Loading**:
- **File**: `src/fastwam/models/wan22/action_dit.py` **Lines**: 112–224
- **Process**:
  1. Initialize ActionDiT with config (line 141–142)
  2. Load checkpoint from `action_dit_pretrained_path` (line 146)
  3. Extract `backbone_state_dict` from checkpoint (line 184)
  4. Verify `meta` (line 156–182) — must match config exactly
  5. Load only "backbone" keys (excluding `action_encoder.*` and `head.*`) to preserve task-specific encoder/head (line 202–217)
  6. Result: Pretrained DiT blocks + randomly initialized action_encoder/head

---

### 1.3 Language Expert Construction

**File**: `src/fastwam/models/hfastwam/hfastwam.py` **Lines**: 2295–2359

```python
lang_hidden = int(video_expert.blocks[0].hidden_dim)  # 3072
lang_ffn = int(language_ffn_dim) if language_ffn_dim is not None else int(
    video_expert.blocks[0].ffn_dim  # 14336
)

if language_backend == "legacy":
    language_expert = LanguageExpert(
        hidden_dim=lang_hidden,          # 3072 ← **matches video_expert**
        num_heads=int(video_expert.num_heads),  # 24
        attn_head_dim=int(video_expert.attn_head_dim),  # 128
        ffn_dim=lang_ffn,                # 14336
        num_layers=int(len(video_expert.blocks)),  # 30
        vocab_size=int(language_vocab_size),  # 32000
        max_task_len=int(language_max_task_len),  # 128
        max_subtask_len=int(language_max_subtask_len),  # 128
        eps=1e-6,
        use_gradient_checkpointing=bool(mot_checkpoint_mixed_attn),
        dtype=torch_dtype,
    ).to(device=device)
    language_tokenizer = components.tokenizer
```

**Language Expert Constructor** (Legacy Path):
- **File**: `src/fastwam/models/hfastwam/language_expert.py` **Lines**: 87–180
- **Key attributes**:
  ```python
  self.hidden_dim = 3072
  self.num_heads = 24
  self.attn_head_dim = 128
  self.token_embedding = nn.Embedding(vocab_size, hidden_dim)  # vocab → 3072
  self.segment_embedding = nn.Embedding(2, hidden_dim)
  self.final_norm = nn.LayerNorm(hidden_dim)
  self.lm_head = nn.Linear(hidden_dim, vocab_size)  # 3072 → vocab (tied weight)
  self.blocks = nn.ModuleList([DiTBlock(...) for _ in range(30)])
  ```
- **All layers randomly initialized** (no pretrain).
- **Visual grounding**: No image encoder; sees video expert's first-frame tokens via MoT attention mask (line 24–30 of hfastwam.py).

**Qwen3-VL Path** (Alternative):
- **File**: `src/fastwam/models/hfastwam/qwen_language_expert.py` **Lines**: 178+
- **From pretrained Qwen3-VL-2B-Instruct** loaded via transformers (line 2317–2325, 2342–2349)
- **Wrapped adapters** to match MoT interface (`_QwenBlockForMoT`, line 118–176)
- **Issue with strict_expert_compat** (line 2326–2341): Qwen3 is 28×2048 (not 30×3072), so strict mode will reject it unless `strict_expert_compat=false`.

---

### 1.4 Shape Validation

**File**: `src/fastwam/models/hfastwam/hfastwam.py` **Lines**: 213–228

```python
@staticmethod
def _validate_expert_shapes(lang, video, action):
    # All three experts MUST share attn-space shape or MoT can't concat.
    if int(lang.num_heads) != int(video.num_heads) or int(action.num_heads) != int(video.num_heads):
        raise ValueError(...)
    if int(lang.attn_head_dim) != int(video.attn_head_dim) or int(action.attn_head_dim) != int(video.attn_head_dim):
        raise ValueError(...)
    if int(len(lang.blocks)) != int(len(video.blocks)) or int(len(action.blocks)) != int(len(video.blocks)):
        raise ValueError(...)
```

**With strict_expert_compat=True** (line 2287–2293):
- **Requires** matching `num_heads`, `attn_head_dim`, `num_layers` across all three
- Language: 24 heads × 128 dim = 3072 attn space ✓
- Video: 24 heads × 128 dim = 3072 attn space ✓
- Action: 24 heads × 128 dim = 3072 attn space ✓
- Layers: 30 = 30 = 30 ✓

---

## 2. Language Expert Details

### 2.1 Actual Dimensions (Legacy Path)

**Language Initialization** (deterministic from video_expert):
```
hidden_dim = 3072
num_heads = 24
attn_head_dim = 128
hidden_dim = num_heads × attn_head_dim = 24 × 128 = 3072 ✓

num_layers = 30
ffn_dim = 14336  (same as video)
vocab_size = 32000
```

**No pretrain** — all weights random. Trained only on CE loss for subtask prediction + language MoT attention.

### 2.2 How It Participates in MoT

**File**: `src/fastwam/models/wan22/mot.py` **Lines**: 48–95

```python
# Each expert's attn-space hidden dim is derived from num_heads × attn_head_dim
self.expert_hidden_dim = {
    name: self.expert_num_heads[name] * self.expert_attn_head_dim[name]
    for name in self.expert_order
}
# language: 24 × 128 = 3072
# video: 24 × 128 = 3072
# action: 24 × 128 = 3072 ← NOT using action.hidden_dim=1024!

self.shared_hidden_dim = self.num_heads * self.attn_head_dim  # 24 × 128 = 3072
```

**Projection Adapters** (lines 98–139):
- If `strict_expert_compat=True`: all projections are **implicit identity** (lines 202–203)
- If `strict_expert_compat=False` and dimensions match: explicit `nn.Identity()` (line 110–114)
- If `strict_expert_compat=False` and dimensions **don't** match: `nn.Linear()` adapters (line 116–139)

**For language** (3072 → 3072): Identity projection in strict mode, or explicit identity in non-strict.

---

## 3. Video Expert Details

### 3.1 Actual Dimensions

**From config**:
```
hidden_dim = 3072       ← Internal transformer state
in_dim = 1024           ← DINO ViT-L features (+ patchify to 3072)
out_dim = 1024          ← Flow-matching target (back to DINO space)
num_heads = 24
attn_head_dim = 128
num_layers = 30
ffn_dim = 14336
```

### 3.2 Input/Output Flow

**Encoding path**:
```
Video [B,3,T,H,W] 
  → VAE encode → [B,C,T,h,w]
  → DINO/ViT-L → [B, num_patches, 1024]
  → patch_embedding (Conv3d) → [B, 3072, T', H', W']
  → flatten → [B, seq_len, 3072]
```

**Code** (`wan_video_dit.py` line 367–368):
```python
self.patch_embedding = nn.Conv3d(in_dim, hidden_dim, kernel_size=patch_size, stride=patch_size)
# 1024 input channels → 3072 output (patch_size [1,2,2])
```

**Decoding path**:
```
MoT output [B, seq_len, 3072]
  → head (Head.forward) → [B, seq_len, out_dim × prod(patch_size)]  [B, seq_len, 1024 × 4]
  → reshape + patchify-inverse
  → VAE decode → [B,3,T,H,W]
```

**Code** (`wan_video_dit.py` line 291–307, 443–448):
```python
class Head(nn.Module):
    def forward(self, x, t_mod):
        # x: [B, seq_len, hidden_dim]  [B, seq_len, 3072]
        return self.head(...)  # [B, seq_len, out_dim × prod(patch_size)]  [B, seq_len, 1024 × 4]

# In WanVideoDiT:
self.head = Head(dim=hidden_dim, out_dim=out_dim, patch_size=patch_size, ...)
# out_dim × prod([1,2,2]) = 1024 × 4 = 4096
```

### 3.3 Pretrained Weight Loading

**Control**: `skip_video_dit_load_from_pretrain`
- Default: False → Loads Wan2.2-TI2V-5B from HuggingFace
- True → Random init (useful for ablation)

**Path**: `src/fastwam/models/wan22/helpers/loader.py` line 155–240+
- **Hash-based detection** (line 96–107): Identifies model type by MD5 of checkpoint file
- **Shape mismatch filtering** (line 120–132): If pretrained shape ≠ model shape, skips that key (random init for that layer)

**Example shape mismatches** (from line 120–132):
```python
for k, v in state_dict.items():
    if k in model_state and v.shape != model_state[k].shape:
        logger.warning("Shape mismatch for '%s': pretrained %s vs model %s — skipping (will be randomly initialised).", k, ...)
        keys_to_remove.append(k)
```

---

## 4. Action Expert Details

### 4.1 **CRITICAL ISSUE**: Hidden Dim Mismatch

**Config declares**:
```yaml
action_dit_config:
  hidden_dim: 1024              ← Used for action_encoder/head
  num_heads: 24                 ← Used for DiTBlock attention
  attn_head_dim: 128            ← Used for attention space
```

**Actual computation**:
```python
# In ActionDiT.__init__ (line 59–85)
self.hidden_dim = 1024
self.num_heads = 24
self.attn_head_dim = 128

# In DiTBlock construction (line 86–97)
for _ in range(num_layers):
    DiTBlock(hidden_dim=1024, attn_head_dim=128, num_heads=24, ...)
    # DiTBlock internally derives: attn_hidden_dim = 24 × 128 = 3072
```

**DiTBlock behavior** (`wan_video_dit.py` line 231–247):
```python
class DiTBlock(nn.Module):
    def __init__(self, hidden_dim: int, attn_head_dim: int, num_heads: int, ffn_dim: int, ...):
        self.hidden_dim = hidden_dim  # 1024
        self.attn_hidden_dim = self.num_heads * self.attn_head_dim  # 24 × 128 = 3072

        self.self_attn = SelfAttention(hidden_dim, attn_head_dim, num_heads, ...)
        # SelfAttention q/k/v projections: 1024 → 3072
        self.ffn = nn.Linear(hidden_dim, ffn_dim)  # 1024 → 4096
```

**Result**: DiTBlock expects:
- **Input**: [B, S, 1024]
- **Attention space**: 24 heads × 128 = 3072
- **Q/K/V projections**: 1024 → 3072
- **Output**: [B, S, 1024]

### 4.2 MoT Sees Attention Space, Not `hidden_dim`

**File**: `src/fastwam/models/wan22/mot.py` **Lines**: 48–54

```python
self.expert_hidden_dim = {
    name: self.expert_num_heads[name] * self.expert_attn_head_dim[name]
    for name in self.expert_order
}
# language: 24 × 128 = 3072
# video: 24 × 128 = 3072
# action: 24 × 128 = 3072

self.shared_hidden_dim = self.num_heads * self.attn_head_dim  # 24 × 128 = 3072
```

**So MoT sees action as 3072-dimensional, NOT 1024**. The action_encoder (1024 → 1024) sits *before* the DiTBlock, not in the MoT token stream.

### 4.3 Input/Output Flow

**Encoding path**:
```
Action tokens [B, T, action_dim]
  → action_encoder: action_dim → 1024
  → DiTBlock input: [B, T, 1024]
  → DiTBlock attention: 1024 → 3072 (Q/K/V) → attention → output back to 1024
  → DiTBlock output: [B, T, 1024]
```

**For MoT**, the pre_dit output is:
```python
# File: src/fastwam/models/wan22/action_dit.py, line 226–299
def pre_dit(self, action_tokens, timestep, context, context_mask=None) -> Dict[str, Any]:
    tokens = self.action_encoder(action_tokens)  # [B, T, action_dim] → [B, T, 1024]
    # ... modulation, context ...
    return {
        "tokens": tokens,  # [B, T, 1024] ← This goes into MoT
        ...
    }
```

**MoT concatenates** language (3072), video (3072), action (1024) tokens — **dimension mismatch!**

### 4.4 MoT Projection Adapters for Action

**File**: `src/fastwam/models/wan22/mot.py` **Lines**: 102–139

```python
if not self.strict_expert_compat:
    for name in self.expert_order:
        in_dim = self.expert_hidden_dim[name]  # For action: 24×128 = 3072 (!)
        # ...
        if in_dim == self.shared_hidden_dim:  # 3072 == 3072 ✓
            # Use Identity
        else:
            # Use Linear adapter
```

**Wait**: MoT computes `expert_hidden_dim = num_heads × attn_head_dim`, so for action it's **3072**, not 1024!

### 4.5 **Reality Check**: Action Expert in MoT

Since action has `num_heads=24, attn_head_dim=128`, MoT sees:
- `expert_hidden_dim["action"] = 24 × 128 = 3072`
- This matches the shared space, so **no adapters needed**

But action_encoder/head still use 1024:
- **Before MoT**: action [B,T,action_dim] → action_encoder → [B,T,1024]
- **DiTBlock inside**: 1024 → 3072 (attn space) → 1024
- **After action_expert.pre_dit()**: tokens are [B,T,1024]

**So the issue is**: action_expert.pre_dit() returns **1024-dim tokens**, but MoT expects **3072-dim tokens** (or applies adapters).

**Question**: How does this work?

**Answer** (from MoT code lines 194–211):
```python
def _project_qkv_to_shared(self, name: str, layer_idx: int, q, k, v):
    # This applies adapters to Q/K/V *inside the DiTBlock*
    # It doesn't touch the token stream; it touches the attention computation
    if self.strict_expert_compat:
        return q, k, v  # No projection
    key = self._proj_key(name, layer_idx)
    if key not in self.q_proj_to_shared:
        return q, k, v
    return (
        self.q_proj_to_shared[key](q),
        self.k_proj_to_shared[key](k),
        self.v_proj_to_shared[key](v),
    )
```

The token stream **stays** in expert-specific dimensions. Only Q/K/V for cross-expert attention are projected to the shared space. So:
- Language tokens: [B, S_lang, 3072]
- Video tokens: [B, S_vid, 3072]
- Action tokens: [B, S_act, 1024]

When combined: [B, S_lang + S_vid + S_act, mixed-dims]

Then in each MoT layer:
1. Extract Q/K/V from each expert's token slice
2. Apply expert-specific Q/K/V norms + RoPE (line 335–350)
3. Project Q/K/V to shared space if dimensions don't match (line 344–350)
4. Concatenate and do shared attention (line 350)
5. Project output back to expert space (line 213–224)

---

## 5. Mixture of Transformers (MoT) Configuration

### 5.1 Construction

**File**: `src/fastwam/models/hfastwam/hfastwam.py` **Lines**: 2361–2372

```python
mot = MoT(
    mixtures={
        "language": language_expert,
        "video": video_expert,
        "action": action_expert,
    },
    mot_checkpoint_mixed_attn=mot_checkpoint_mixed_attn,
    strict_expert_compat=bool(strict_expert_compat),
    layer_alignment_mode=layer_alignment_mode,
    shared_attention_expert=shared_attention_expert,  # default: "video"
)
```

**Config defaults** (from `configs/model/hfastwam.yaml` lines 59–62):
```yaml
strict_expert_compat: false        # Allow heterogeneous dims
layer_alignment_mode: "tail_overlap"  # Non-strict layers run solo
shared_attention_expert: "video"   # Video's heads/head_dim define shared space
```

### 5.2 Dimension Resolution in MoT

**File**: `src/fastwam/models/wan22/mot.py` **Lines**: 48–95

```python
# Step 1: Compute each expert's attention space
self.expert_hidden_dim = {
    name: int(len(expert.blocks)) for name, expert in self.mixtures.items()
}
# With strict=False + tail_overlap:
self.overlap_num_layers = min(self.expert_num_layers.values())  # min(30, 30, 30) = 30
self.layer_start_indices = {
    name: self.expert_num_layers[name] - self.overlap_num_layers
    for name in self.expert_order
}
# language: 30 - 30 = 0
# video: 30 - 30 = 0
# action: 30 - 30 = 0
# → All layers overlap (no solo prefix layers)

# Step 2: Use first expert's attention space as shared (if strict)
# Or use explicit shared_attention_expert (if non-strict)
shared_name = shared_attention_expert  # "video"
self.num_heads = self.expert_num_heads[shared_name]  # 24
self.attn_head_dim = self.expert_attn_head_dim[shared_name]  # 128
self.shared_hidden_dim = 24 * 128  # 3072

# Step 3: Build adapters (non-strict only)
if not self.strict_expert_compat:
    for name in self.expert_order:
        in_dim = self.expert_hidden_dim[name]  # 24*128 for all
        if in_dim == self.shared_hidden_dim:  # 3072 == 3072
            use Identity
        else:
            use Linear(in_dim, shared)
```

### 5.3 Projection Adapters (When Needed)

**Conditions for Identity** (line 110–114):
```python
if in_dim == self.shared_hidden_dim:
    q_proj_to_shared[key] = nn.Identity()
    k_proj_to_shared[key] = nn.Identity()
    v_proj_to_shared[key] = nn.Identity()
    o_proj_from_shared[key] = nn.Identity()
```

**For current setup**:
- Language: 3072 == 3072 → **Identity** ✓
- Video: 3072 == 3072 → **Identity** ✓
- Action: 24×128=3072 == 3072 → **Identity** ✓

**All projections are identity in current config!**

### 5.4 When Projections Are Needed

Example: If you **shrink action to hidden=256**:
- `num_heads=16, attn_head_dim=128` (if you keep head_dim=128, then heads=256/128=2)
- Or `num_heads=4, attn_head_dim=64` (smaller heads)

Then `action_hidden_dim = 4 × 64 = 256 ≠ 3072` → **Linear adapters**:
```python
q_proj_to_shared["action__0"] = nn.Linear(256, 3072, bias=False)
k_proj_to_shared["action__0"] = nn.Linear(256, 3072, bias=False)
v_proj_to_shared["action__0"] = nn.Linear(256, 3072, bias=False)
o_proj_from_shared["action__0"] = nn.Linear(3072, 256, bias=False)
```

---

## 6. Hard-Coded Dimension Couplings & Breakage Points

### 6.1 Video Expert

#### **patch_embedding** (Line: `wan_video_dit.py` 367–368)
```python
self.patch_embedding = nn.Conv3d(in_dim, hidden_dim, ...)
# in_dim=1024 (DINO/ViT-L)
# hidden_dim=3072
```

**If you shrink hidden_dim**: Must match `in_dim`. If `in_dim=1024` and `hidden_dim=2048`:
```python
Conv3d(1024, 2048, kernel_size=[1,2,2], stride=[1,2,2])
```
✓ Works (reinitialized from pretrain).

#### **Head output** (Lines: `wan_video_dit.py` 291–308)
```python
class Head(nn.Module):
    def __init__(self, dim: int, out_dim: int, patch_size: Tuple[int, int, int], eps: float):
        self.head = nn.Linear(dim, out_dim * math.prod(patch_size))
        # dim = hidden_dim = 3072
        # out_dim = 1024 (back to DINO space)
        # prod([1,2,2]) = 4
        # So: 3072 → 1024 * 4 = 4096

    def forward(self, x, t_mod):
        # x: [B, seq_len, 3072]
        return self.head(x)  # [B, seq_len, 4096] → reshape → [B, seq_len, 4, 1024]
```

**If you shrink hidden_dim to 2048**:
- Head becomes: `nn.Linear(2048, 1024 * 4)` = `nn.Linear(2048, 4096)`
- Still outputs 4096 tokens (1024-dim after reshape)
- ✓ Works, but dimensions flow through different scaling.

#### **Loss computation** (Lines: `hfastwam.py` 1770–1786)
```python
def _compute_video_loss(self, pred_video, target_video, fuse_flag, timestep_video):
    if fuse_flag:
        pred_video = pred_video[:, :, 1:]  # Skip first-frame (fused)
        target_video = target_video[:, :, 1:]
    per_sample = F.mse_loss(
        pred_video.float(), target_video.float(), reduction="none",
    ).mean(dim=(1, 2, 3, 4))  # ← Assumes pred_video is [B, C, T, H, W]
    w = self.train_video_scheduler.training_weight(timestep_video)
    return (per_sample * w).mean()
```

**Hard-coded assumption**: `pred_video` has 5 dims `[B, C, T, H, W]`. This is enforced by VAE decode output shape, not expert dimensions. **No breakage** if you shrink hidden_dim (VAE output doesn't change).

### 6.2 Action Expert

#### **action_encoder & head** (Lines: `action_dit.py` 74, 98)
```python
self.action_encoder = nn.Linear(action_dim, hidden_dim)  # action_dim → 1024
# ...
self.head = nn.Linear(hidden_dim, action_dim)  # 1024 → action_dim
```

**If you change action_dim**: Simply change input/output dims. No coupling to video.

**If you change hidden_dim (1024)**: Need to update DiTBlocks accordingly. DiTBlocks still use `num_heads=24, attn_head_dim=128`, so attn space stays 3072, but token stream shrinks to (say) 2048.

#### **Loss computation** (Lines: `hfastwam.py` 1788–1810)
```python
def _compute_action_loss(self, pred_action, target_action, timestep_action, action_is_pad):
    per_sample = F.mse_loss(
        pred_action.float(), target_action.float(), reduction="none",
    ).mean(dim=(1, 2))  # ← Assumes pred_action is [B, T, action_dim]
    w = self.train_action_scheduler.training_weight(timestep_action)
    return (per_sample * w).mean()
```

**No hard-coded dims**. ✓ Works if you change expert hidden_dim.

### 6.3 Language Expert

#### **token_embedding** (Lines: `language_expert.py` 118)
```python
self.token_embedding = nn.Embedding(vocab_size, hidden_dim)  # 32000 → 3072
self.lm_head = nn.Linear(hidden_dim, vocab_size)  # 3072 → 32000
```

**If you change hidden_dim**: Both change automatically. **No breakage**.

#### **Loss computation** (Lines: `hfastwam.py` 1812–1833)
```python
@staticmethod
def _compute_language_token_loss(logits, labels):
    vocab_size = int(logits.shape[-1])
    shift_logits = logits[:, :-1].contiguous().view(-1, vocab_size)
    shift_labels = labels[:, 1:].contiguous().view(-1)
    return F.cross_entropy(shift_logits.float(), shift_labels, ...)
```

**No hard-coded dims**. ✓ Works.

---

## 7. Dimension Coupling Summary: Shrinking Video + Action to Match Language

### Current Config:
```
Language: hidden=3072, heads=16 (?), head_dim=128, layers=28
Video: hidden=3072, heads=24, head_dim=128, layers=30
Action: hidden=1024 (token), heads=24, head_dim=128, layers=30 (attn space: 3072)
```

**Wait**: Language expert's heads should be 24 (from video config), not 16. Let me verify...

From `hfastwam.py` line 2304:
```python
num_heads=int(video_expert.num_heads),  # 24
```

So **Language is 24 heads, not 16**. The log message said `heads=16` if it was from Qwen3.

### To Align All to Language Dims (hidden=2048, heads=16, head_dim=128):

**What breaks**:
1. **Shared attention space**: 16 × 128 = 2048 (smaller than video's 3072)
2. **Adapters needed**: Language (2048→2048 identity), video (3072→2048 linear), action (3072→2048 linear)
3. **Video patch_embedding**: 1024 → 3072 becomes 1024 → 2048
4. **Video head**: 3072 → 1024 becomes 2048 → 1024 (smaller, reshapes same way)
5. **DiTBlocks**: All use smaller hidden_dim, but `num_heads × head_dim` still defines attn space
6. **DINO-space flow matching**: No change (still 1024-dim in DINO space)

**Configuration**:
```yaml
language_expert:
  hidden_dim: 2048
  num_heads: 16
  attn_head_dim: 128
  num_layers: 28
  
video_dit_config:
  hidden_dim: 2048      # shrink from 3072
  num_heads: 16         # shrink from 24
  attn_head_dim: 128    # keep
  num_layers: 28        # shrink from 30
  ffn_dim: 8192         # shrink from 14336 (same ratio ≈3.5x)
  
action_dit_config:
  hidden_dim: 2048      # enlarge from 1024
  num_heads: 16         # shrink from 24
  attn_head_dim: 128    # keep
  num_layers: 28        # shrink from 30
  ffn_dim: 8192         # shrink from 4096
```

**Changes to code**:
- Video `patch_embedding`: 1024 → 2048 (reinitialized)
- Video `head`: 2048 → 1024 (reinitialized)
- Action `action_encoder`: action_dim → 2048 (random init unless pretrain preserved)
- Action `head`: 2048 → action_dim (random init)
- All adapters: Identity (2048 == 2048 everywhere)

---

## 8. Key Files & Line References

| Component | File | Key Lines |
|-----------|------|-----------|
| **Entry point** | `hfastwam.py` | 2159–2446 |
| **Language Expert** | `language_expert.py` | 87–180 |
| **Qwen Language Expert** | `qwen_language_expert.py` | 178–300+ |
| **Video Expert (WanVideoDiT)** | `wan_video_dit.py` | 310–500+ |
| **Action Expert (ActionDiT)** | `action_dit.py` | 32–224 |
| **DiTBlock** | `wan_video_dit.py` | 230–268 |
| **MoT** | `mot.py` | 14–226 |
| **MoT Layer Alignment** | `mot.py` | 48–95 |
| **MoT Projections** | `mot.py` | 98–224 |
| **Attention Mask** | `hfastwam.py` | 637–692 |
| **Shape Validation** | `hfastwam.py` | 213–228 |
| **Config (H-FastWAM)** | `configs/model/hfastwam.yaml` | 1–129 |
| **Loader** | `helpers/loader.py` | 155–240+ |

---

## 9. Critical Insights for Shrinking Video/Action Experts

### 9.1 Projection Adapters Disappear When Dims Match

When all three experts have `hidden_dim=num_heads×attn_head_dim`, MoT uses implicit identity:
- No adaptive projections needed
- Cleaner forward pass, fewer parameters
- **Requirement**: `strict_expert_compat=false` and all dims aligned

### 9.2 Language Expert Is Random-Init

Unlike video (Wan2.2 pretrained) and action (ActionDiT pretrained):
- Language starts random
- Trained **only** on CE loss for task/subtask prediction
- **Benefits of shrinking**: Fewer params, faster convergence (if reducing from 3072 hidden)
- **Risk**: Qwen3 path might require strict_expert_compat=false anyway

### 9.3 Action Expert's Hidden Dim vs Attention Space

Currently:
- Token stream: 1024 dim
- Attention space: 3072 dim (24×128)
- MoT sees: 3072 dim

If shrinking action to match language (2048):
- Token stream: 2048 dim
- Attention space: 2048 dim (16×128)
- MoT sees: 2048 dim
- **Simpler**: No internal dimension mismatch in DiTBlocks

### 9.4 Video Expert is the "Reference"

- `shared_attention_expert="video"` (default, line 2371)
- MoT uses video's `num_heads × attn_head_dim` to define shared space
- If you shrink video, you **must** shrink all others to match (strict_expert_compat=true) or provide adapters (strict_expert_compat=false, but not recommended for 2048-dim video with 3072-dim others)

---

## 10. Recommended Shrinking Strategy

1. **Set all experts to**:
   - `hidden_dim = 2048` (or your target)
   - `num_heads = 16`
   - `attn_head_dim = 128`
   - `num_layers = 28`
   - `ffn_dim = 8192` (approx 4x hidden_dim)

2. **Video expert**:
   - Update `video_dit_config` in hfastwam.yaml
   - Set `skip_video_dit_load_from_pretrain = true` (random init, or fine-tune from smaller Wan checkpoint if available)
   - Patch `patch_embedding` and `head` are reinitialized automatically

3. **Action expert**:
   - Update `action_dit_config` in hfastwam.yaml
   - Set `action_dit_pretrained_path = null` (or find a smaller pretrained checkpoint)
   - Both `action_encoder` and `head` are random init (task-specific)

4. **Language expert**:
   - Automatically derives from video_expert dims in from_pretrained_fastwam (line 2296)
   - Stays random-init (unchanged)

5. **Remove MoT projection adapters**:
   - Ensure `strict_expert_compat = true` (line 2398)
   - All experts have matching `num_heads, attn_head_dim, num_layers` → adapters become identity

6. **Testing**:
   - Verify `_validate_expert_shapes()` passes (line 213–228)
   - Check MoT init logs show identity adapters (or no adapters)
   - Run a forward pass on dummy data to catch shape mismatches

---

## 11. Example Config Change

### Before:
```yaml
video_dit_config:
  hidden_dim: 3072
  num_heads: 24
  attn_head_dim: 128
  num_layers: 30
  ffn_dim: 14336

action_dit_config:
  hidden_dim: 1024
  num_heads: 24
  attn_head_dim: 128
  num_layers: 30
  ffn_dim: 4096
```

### After (Aligned to 2048 / 16 heads):
```yaml
video_dit_config:
  hidden_dim: 2048
  num_heads: 16
  attn_head_dim: 128
  num_layers: 28
  ffn_dim: 8192
  skip_video_dit_load_from_pretrain: true  # Random init (or transfer from smaller pretrain)

action_dit_config:
  hidden_dim: 2048
  num_heads: 16
  attn_head_dim: 128
  num_layers: 28
  ffn_dim: 8192
  action_dit_pretrained_path: null  # Random init
```

### Code changes (minimal):
- No changes to `from_pretrained_fastwam()` — it auto-derives language dims from video
- No changes to MoT — it auto-detects matching dims and uses identity adapters
- Remove any `strict_expert_compat = false` logic if present

