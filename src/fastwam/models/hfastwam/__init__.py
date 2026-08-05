"""Hierarchical Fast World-Action Model (H-FastWAM).

Three-expert MoT composition:

  1. **Language Expert** — random-init transformer that shares self-attention
     with the video / action experts. Consumes
     ``[image_tokens ‖ task_tokens ‖ subtask_tokens]`` and predicts the
     next subtask token. Image tokens come from a frozen SigLIP-like
     encoder + a small MLP projection.

  2. **Video Expert** — Wan2.2 TI2V-5B DiT (pretrained), flow-matching
     world model.

  3. **Action Expert** — :class:`fastwam.models.wan22.action_dit.ActionDiT`
     (same init path as vanilla fastwam).

All three experts participate in a *single* shared multi-modal
self-attention per layer (see :class:`fastwam.models.wan22.mot.MoT`).
The language expert's K/V is **detached** when video/action queries
attend to it, so the video/action losses cannot corrupt the language
weights — language is trained only by its own teacher-forced CE loss.

Training phases::

    language_only   →   language_video   →   full
    (lang only)         (lang + video)       (lang + video + action)
"""

from .hfastwam import HFastWAM
from .hfastwam_idm import HFastWAMIDM
from .language_expert import LanguageExpert
from .qwen_language_expert import QwenLanguageExpert

__all__ = ["HFastWAM", "HFastWAMIDM", "LanguageExpert", "QwenLanguageExpert"]
