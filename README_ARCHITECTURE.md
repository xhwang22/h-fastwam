# FastWAM Architecture Documentation

Complete technical documentation of the FastWAM model architecture has been generated.

## 📚 Documentation Files

### 1. **FastWAM_Architecture_Complete.md** (32 KB)
**Comprehensive architectural specification**

Contains:
- Complete overview of FastWAM as a Diffusion Transformer world model
- Video Expert (WanVideoDiT) - detailed pre_dit/post_dit methods, DiT block structure, masking
- Action Expert (ActionDiT) - module initialization, input processing, output projection
- VAE Architecture - encoder/decoder layer structure with 23 CausalConv3d layers
- Mixture of Transformers - mixed attention orchestration with causal masking patterns
- FastWAM Main Class - component assembly, scheduler configuration, loss weighting
- Training Loss Data Flow - 15-step training forward pass with exact dimensions
- Attention Mechanisms - self-attention, cross-attention, mixed attention details

**Best for:** Understanding the complete model architecture and implementation details

---

### 2. **FastWAM_Data_Flow_Diagram.txt** (22 KB)
**Visual data flow and dimension tracking**

Contains:
- Complete training loop ASCII diagram (full top-to-bottom flow)
- Module interaction map (component relationships)
- Key dimensions at each pipeline stage
- Attention mask structure and group causal patterns
- Exact tensor shapes through the entire pipeline
- Mask visualization for video/action cross-attention

**Best for:** Tracing tensor shapes, debugging, creating model diagrams

---

### 3. **FastWAM_Code_Reference.md** (17 KB)
**Code-level reference with line numbers**

Contains:
- File locations for all modules
- Class definitions with exact line numbers
- Method signatures and input/output shapes
- Quick lookup tables for dimensions
- Layer counts summary for VAE and experts
- Hidden dimension mapping across architecture
- Cross-file data flow references
- Code cross-references for all key functions

**Best for:** Quick code lookups, finding implementations, understanding file structure

---

### 4. **FastWAM_DOCUMENTATION_SUMMARY.txt** (8.5 KB)
**Quick reference and overview**

Contains:
- Overview of all three documents
- Key architecture summary for each component
- Training data flow (high-level)
- Exact dimensions at each stage
- File reference structure
- Key technical concepts explained
- Use cases for each document
- Quick start checklist

**Best for:** Getting started, understanding what's in each document

---

## 🎯 Quick Architecture Summary

### Components

| Component | Type | Details |
|-----------|------|---------|
| **Video Expert** | WanVideoDiT | 24 DiT blocks, 1280d hidden, 20 heads, 3D RoPE |
| **Action Expert** | ActionDiT | 24 DiT blocks, 1280d hidden, 20 heads, 1D RoPE |
| **VAE** | WanVideoVAE | 23-layer encoder/decoder, z_dim=16, 4×8 compression |
| **MoT** | Mixed Attention | 24 blocks with video↔action cross-attention |
| **Scheduling** | Flow Matching | WanContinuousFlowMatchScheduler, 1000 timesteps |

### Dimensions

```
Input Video:     [B, 3, T, H, W]                    e.g., [4, 3, 8, 256, 256]
↓ VAE encode
Latents:         [B, 16, T/4, H/8, W/8]             e.g., [4, 16, 2, 32, 32]
↓ Patchify
Patches:         [B, T/4·H/8·W/8, 1280]             e.g., [4, 2048, 1280]

Text Context:    [B, L_text, 1280]                  e.g., [4, 77, 1280]
Action Tokens:   [B, T_action, 1280]                e.g., [4, 8, 1280]

MoT Combined:    [B, N_v+N_a+L_text, 1280]          e.g., [4, 2133, 1280]

Predictions:
  Video:         [B, 16, T/4, H/8, W/8]             [B, 4, 16, 2, 32, 32]
  Action:        [B, T_action, 7]                   [B, 4, 8, 7]

Loss:            scalar (λ_v·MSE_v + λ_a·MSE_a)
```

### Training Flow

1. **Build Inputs** → Encode video via VAE
2. **Add Noise** → Sample timesteps, add noise to latents & actions
3. **Video pre_dit** → Patchify, embed time/text/action
4. **Action pre_dit** → Encode actions, embed time/text
5. **Build MoT Mask** → Create causal attention patterns
6. **MoT Forward** → 24 blocks of mixed-modal attention
7. **Post-process** → Project tokens back to prediction space
8. **Compute Loss** → MSE with padding masks & timestep weighting
9. **Backprop** → Optimizer step

---

## 📖 How to Use These Documents

### For Model Architecture Understanding
Start with **FastWAM_DOCUMENTATION_SUMMARY.txt** → then read **FastWAM_Architecture_Complete.md**

### For Creating Diagrams
Use **FastWAM_Data_Flow_Diagram.txt** as reference, adapt ASCII art to PowerPoint/Miro/Visio

### For Implementation
Reference **FastWAM_Code_Reference.md** for exact line numbers and class locations

### For Debugging Training
Use **FastWAM_Data_Flow_Diagram.txt** to trace tensor shapes at each stage

### For Academic Writing
Include sections from **FastWAM_Architecture_Complete.md** with proper attribution

---

## 🔍 Key Technical Highlights

### Mixture of Transformers (MoT)
- Video and action experts share **24 DiT blocks**
- Mixed attention allows cross-modal interaction
- **Attention mask structure**:
  - Video→Video: Per-frame causal (can't attend to future frames)
  - Video→Action: All allowed (video sees all actions)
  - Action→Video: **Group causal** (actions see first frame only!)
  - Action→Action: All allowed (bidirectional)

### Adaptive Layer Normalization (AdaLN)
- Each DiT block has **6 modulation parameters** per block:
  - shift_msa, scale_msa, gate_msa (for self-attention)
  - shift_mlp, scale_mlp, gate_mlp (for FFN)
- Derived from timestep embedding: `t_mod = time_projection(t)`

### Rotary Position Embeddings (RoPE)
- **Video**: 3D RoPE (independent frequencies for temporal, height, width)
- **Action**: 1D RoPE (sequence positions)
- Precomputed and cached for efficiency

### Flow Matching Noise Scheduling
- **Timesteps**: Continuous [0, 1] sampled uniformly
- **Add Noise**: `x_t = α(t)·x₀ + σ(t)·ε` (linear interpolation schedule)
- **Loss Weight**: Per-timestep weighting (typically SNR-based)
- **Targets**: Computed from scheduler.training_target()

---

## 📊 Statistics

| Metric | Value |
|--------|-------|
| Video Expert Layers | 24 |
| Action Expert Layers | 24 |
| MoT Blocks (Shared) | 24 |
| VAE Encoder Layers | 23 |
| VAE Decoder Layers | 23 |
| Hidden Dimension | 1280 |
| Attention Heads | 20 |
| Head Dimension | 64 |
| FFN Dimension | 3456 |
| VAE z_dim | 16 |
| Temporal Compression | 4× |
| Spatial Compression | 8× |
| Noise Timesteps | 1000 |

---

## 🔗 File Locations

All documentation files are located in:
```
/apdcephfs_gy2/share_302533218/shaunxhwang/embodied/FastWAM/
├── FastWAM_Architecture_Complete.md (32 KB)
├── FastWAM_Data_Flow_Diagram.txt (22 KB)
├── FastWAM_Code_Reference.md (17 KB)
├── FastWAM_DOCUMENTATION_SUMMARY.txt (8.5 KB)
└── README_ARCHITECTURE.md (this file)
```

---

## ✅ Verification Checklist

- [x] Video Expert architecture documented (24 blocks, 1280d, 20 heads)
- [x] Action Expert architecture documented (24 blocks, 1280d, 20 heads)
- [x] VAE structure documented (23 layers encoder/decoder, z_dim=16)
- [x] MoT mixed attention documented (causal masking patterns)
- [x] Training loss flow documented (15-step forward pass)
- [x] Attention mechanisms documented (self, cross, mixed)
- [x] Noise scheduling documented (flow matching)
- [x] Data dimensions documented (all stages)
- [x] Line numbers provided (all key methods)
- [x] Code cross-references documented

---

## 📝 Notes

- **Last Updated**: May 9, 2026
- **Total Documentation**: 1,842 lines across 4 files (79 KB)
- **Scope**: Complete FastWAM model as of the current codebase
- **Accuracy**: All information extracted directly from source code with exact line numbers

---

**For questions or clarifications about the architecture, refer to the specific document sections indicated above.**
