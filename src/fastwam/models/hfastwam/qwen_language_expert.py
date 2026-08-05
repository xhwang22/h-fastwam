from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
import torch.nn as nn

from .language_expert import LanguageExpert

logger = logging.getLogger(__name__)


class _IdentityNorm(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class _PerHeadNorm(nn.Module):
    """Apply Qwen q/k norm over each attention head before MoT RoPE."""

    def __init__(self, base_norm: nn.Module | None, head_dim: int):
        super().__init__()
        self.base_norm = base_norm if base_norm is not None else _IdentityNorm()
        self.head_dim = int(head_dim)
        if self.head_dim <= 0:
            raise ValueError(f"`head_dim` must be positive, got {head_dim}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] % self.head_dim != 0:
            raise ValueError(
                f"Qwen q/k projection dim {x.shape[-1]} must be divisible by head_dim={self.head_dim}."
            )
        heads = x.shape[-1] // self.head_dim
        y = x.view(*x.shape[:-1], heads, self.head_dim)
        y = self.base_norm(y)
        return y.reshape_as(x)


class _RepeatKVHeads(nn.Module):
    """Expand K/V heads to match query heads when Qwen uses GQA."""

    def __init__(self, base_proj: nn.Module, target_dim: int, head_dim: int):
        super().__init__()
        self.base_proj = base_proj
        self.target_dim = int(target_dim)
        self.head_dim = int(head_dim)
        out_dim = int(getattr(base_proj, "out_features"))
        self.fallback = None
        if out_dim != self.target_dim:
            self.fallback = nn.Linear(out_dim, self.target_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.base_proj(x)
        if y.shape[-1] == self.target_dim:
            return y
        if self.head_dim <= 0:
            if self.fallback is None:
                raise ValueError("KV projection mismatch and no fallback adapter.")
            return self.fallback(y)

        if y.shape[-1] % self.head_dim != 0 or self.target_dim % self.head_dim != 0:
            if self.fallback is None:
                raise ValueError(
                    f"Cannot align KV dims: got {y.shape[-1]} target {self.target_dim} with head_dim={self.head_dim}."
                )
            return self.fallback(y)

        kv_heads = y.shape[-1] // self.head_dim
        target_heads = self.target_dim // self.head_dim
        if kv_heads > 0 and target_heads % kv_heads == 0:
            repeat_factor = target_heads // kv_heads
            y = y.view(*y.shape[:-1], kv_heads, self.head_dim)
            y = y.repeat_interleave(repeat_factor, dim=-2)
            y = y.reshape(*y.shape[:-2], target_heads * self.head_dim)
            return y

        if self.fallback is None:
            raise ValueError(
                f"Cannot align KV heads: kv_heads={kv_heads}, target_heads={target_heads}, no fallback adapter."
            )
        return self.fallback(y)


class _QwenSelfAttentionForMoT(nn.Module):
    def __init__(
        self,
        q_proj: nn.Module,
        k_proj: nn.Module,
        v_proj: nn.Module,
        o_proj: nn.Module,
        target_num_heads: int,
        head_dim: int,
        q_norm: nn.Module | None,
        k_norm: nn.Module | None,
    ):
        super().__init__()
        self.q = q_proj
        q_out_dim = int(getattr(q_proj, "out_features"))
        self.k = _RepeatKVHeads(k_proj, target_dim=q_out_dim, head_dim=head_dim)
        self.v = _RepeatKVHeads(v_proj, target_dim=q_out_dim, head_dim=head_dim)
        self.o = o_proj
        self.norm_q = _PerHeadNorm(q_norm, head_dim=head_dim)
        self.norm_k = _PerHeadNorm(k_norm, head_dim=head_dim)
        self.num_heads = int(target_num_heads)


class _QwenCrossAttentionStub(nn.Module):
    def forward(self, x: torch.Tensor, ctx: torch.Tensor, ctx_mask: torch.Tensor | None = None) -> torch.Tensor:
        return torch.zeros_like(x)


class _QwenGate(nn.Module):
    def forward(self, x: torch.Tensor, gate: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        return x + gate * residual


class _QwenBlockForMoT(nn.Module):
    """Adapter block exposing DiT-like fields consumed by MoT."""

    def __init__(
        self,
        *,
        qwen_layer: nn.Module,
        num_heads: int,
        attn_head_dim: int,
        hidden_dim: int,
    ):
        super().__init__()

        attn = getattr(qwen_layer, "self_attn", None)
        if attn is None:
            raise ValueError("Qwen layer missing `self_attn`.")

        q_proj = getattr(attn, "q_proj", None)
        k_proj = getattr(attn, "k_proj", None)
        v_proj = getattr(attn, "v_proj", None)
        o_proj = getattr(attn, "o_proj", None)
        if q_proj is None or k_proj is None or v_proj is None or o_proj is None:
            raise ValueError("Qwen attention must expose q_proj/k_proj/v_proj/o_proj.")

        q_norm = getattr(attn, "q_norm", None)
        k_norm = getattr(attn, "k_norm", None)

        norm1 = getattr(qwen_layer, "input_layernorm", None)
        norm2 = getattr(qwen_layer, "post_attention_layernorm", None)
        mlp = getattr(qwen_layer, "mlp", None)
        if norm1 is None or norm2 is None or mlp is None:
            raise ValueError("Qwen layer must expose input_layernorm/post_attention_layernorm/mlp.")

        self.hidden_dim = int(hidden_dim)
        self.num_heads = int(num_heads)
        self.attn_head_dim = int(attn_head_dim)
        self.self_attn = _QwenSelfAttentionForMoT(
            q_proj=q_proj,
            k_proj=k_proj,
            v_proj=v_proj,
            o_proj=o_proj,
            target_num_heads=self.num_heads,
            head_dim=self.attn_head_dim,
            q_norm=q_norm,
            k_norm=k_norm,
        )

        self.norm1 = norm1
        self.norm2 = norm2
        self.norm3 = _IdentityNorm()
        self.ffn = mlp
        self.cross_attn = _QwenCrossAttentionStub()
        self.gate = _QwenGate()

        self.modulation = nn.Parameter(torch.zeros(1, 6, self.hidden_dim))
        with torch.no_grad():
            self.modulation[:, 2, :].fill_(1.0)
            self.modulation[:, 5, :].fill_(1.0)


class QwenLanguageExpert(LanguageExpert):
    """MoT-compatible language expert with Qwen block-level adapter."""

    @staticmethod
    def _extract_backbone(model: nn.Module) -> tuple[nn.Module, nn.ModuleList, nn.Module | None]:
        core = getattr(model, "model", None)
        if core is not None and hasattr(core, "language_model"):
            core = core.language_model
        elif hasattr(model, "language_model"):
            core = model.language_model
        elif hasattr(model, "layers"):
            core = model
        if core is None:
            raise ValueError("Unsupported Qwen model: expected `.model` or `.model.language_model` attribute.")

        layers = getattr(core, "layers", None)
        if layers is None:
            raise ValueError("Unsupported Qwen backbone: expected decoder `.layers`.")

        final_norm = getattr(core, "norm", None)
        return core, layers, final_norm

    @staticmethod
    def _get_input_embeddings(model: nn.Module, backbone: nn.Module) -> nn.Embedding | None:
        get_input_embeddings = getattr(model, "get_input_embeddings", None)
        emb = get_input_embeddings() if callable(get_input_embeddings) else None
        if emb is None:
            get_input_embeddings = getattr(backbone, "get_input_embeddings", None)
            emb = get_input_embeddings() if callable(get_input_embeddings) else None
        if emb is None:
            emb = getattr(backbone, "embed_tokens", None)
        return emb

    @staticmethod
    def _get_output_embeddings(model: nn.Module) -> nn.Module | None:
        get_output_embeddings = getattr(model, "get_output_embeddings", None)
        lm_head = get_output_embeddings() if callable(get_output_embeddings) else None
        if lm_head is None:
            lm_head = getattr(model, "lm_head", None)
        return lm_head

    @staticmethod
    def _download_qwen3_vl_weight_files(model_id: str, local_files_only: bool) -> list[str]:
        local_path = Path(model_id)
        if local_path.is_dir():
            index_path = local_path / "model.safetensors.index.json"
            if index_path.is_file():
                with index_path.open("r", encoding="utf-8") as f:
                    weight_map = json.load(f).get("weight_map", {})
                filenames = {
                    filename
                    for key, filename in weight_map.items()
                    if key.startswith("model.language_model.") or key == "lm_head.weight"
                }
                if not filenames:
                    raise ValueError(f"No Qwen3-VL language weights found in {index_path}.")
                return [str(local_path / filename) for filename in sorted(filenames)]

            weights_path = local_path / "model.safetensors"
            if not weights_path.is_file():
                raise FileNotFoundError(
                    f"Qwen3-VL checkpoint directory has no safetensors weights: {local_path}"
                )
            return [str(weights_path)]

        try:
            from huggingface_hub import hf_hub_download
            from huggingface_hub.utils import EntryNotFoundError, LocalEntryNotFoundError
        except Exception as exc:  # pragma: no cover
            raise ImportError("Loading Qwen3-VL language-only weights requires `huggingface_hub`.") from exc

        try:
            index_path = hf_hub_download(
                repo_id=model_id,
                filename="model.safetensors.index.json",
                local_files_only=local_files_only,
            )
        except (EntryNotFoundError, LocalEntryNotFoundError):
            index_path = None

        if index_path is None:
            return [
                hf_hub_download(
                    repo_id=model_id,
                    filename="model.safetensors",
                    local_files_only=local_files_only,
                )
            ]

        with open(index_path, "r", encoding="utf-8") as f:
            weight_map = json.load(f).get("weight_map", {})

        filenames = {
            filename
            for key, filename in weight_map.items()
            if key.startswith("model.language_model.") or key == "lm_head.weight"
        }
        if not filenames:
            raise ValueError(f"No Qwen3-VL language weights found in {index_path}.")

        return [
            hf_hub_download(repo_id=model_id, filename=filename, local_files_only=local_files_only)
            for filename in sorted(filenames)
        ]

    @classmethod
    def _load_qwen3_vl_text_model(
        cls,
        model_id: str,
        dtype: torch.dtype,
        local_files_only: bool,
    ) -> tuple[nn.Module, nn.Module]:
        try:
            from safetensors import safe_open
            from transformers import AutoConfig
            from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextModel
        except Exception as exc:  # pragma: no cover
            raise ImportError(
                "Qwen3-VL language-only loading requires transformers with Qwen3-VL support "
                "(transformers>=4.57.0) and safetensors."
            ) from exc

        config = AutoConfig.from_pretrained(
            model_id,
            trust_remote_code=True,
            local_files_only=local_files_only,
        )
        text_config = getattr(config, "text_config", None)
        if text_config is None:
            raise ValueError(f"{model_id} is not a Qwen3-VL config with `text_config`.")

        text_model = Qwen3VLTextModel(text_config).to(dtype=dtype)
        lm_head = nn.Linear(text_config.hidden_size, text_config.vocab_size, bias=False).to(dtype=dtype)

        text_state = {}
        lm_head_state = {}
        for path in cls._download_qwen3_vl_weight_files(model_id, local_files_only):
            with safe_open(path, framework="pt", device="cpu") as f:
                for key in f.keys():
                    if key.startswith("model.language_model."):
                        text_state[key.removeprefix("model.language_model.")] = f.get_tensor(key)
                    elif key == "lm_head.weight":
                        lm_head_state["weight"] = f.get_tensor(key)

        missing, unexpected = text_model.load_state_dict(text_state, strict=False)
        if missing or unexpected:
            raise ValueError(
                "Failed to load Qwen3-VL text backbone cleanly: "
                f"missing={missing[:8]} unexpected={unexpected[:8]}"
            )

        if lm_head_state:
            missing, unexpected = lm_head.load_state_dict(lm_head_state, strict=False)
            if missing or unexpected:
                raise ValueError(
                    "Failed to load Qwen3-VL lm_head cleanly: "
                    f"missing={missing[:8]} unexpected={unexpected[:8]}"
                )
        else:
            lm_head.weight = text_model.embed_tokens.weight

        return text_model, lm_head

    @classmethod
    def _from_loaded_qwen(
        cls,
        *,
        model_id: str,
        model: nn.Module,
        lm_head: nn.Module | None,
        max_task_len: int,
        max_subtask_len: int,
        eps: float,
        use_gradient_checkpointing: bool,
        dtype: torch.dtype,
    ) -> "QwenLanguageExpert":
        backbone, layers, final_norm = cls._extract_backbone(model)
        emb = cls._get_input_embeddings(model, backbone)
        if emb is None:
            raise ValueError("Qwen model has no input embeddings.")

        if lm_head is None:
            lm_head = cls._get_output_embeddings(model)

        hidden_dim = int(emb.embedding_dim)
        vocab_size = int(emb.num_embeddings)
        num_layers = int(len(layers))

        first_layer = layers[0]
        first_attn = getattr(first_layer, "self_attn", None)
        if first_attn is None:
            raise ValueError("Qwen layer missing `self_attn`.")

        num_heads = int(getattr(first_attn, "num_heads", 0))
        if num_heads <= 0:
            num_heads = int(getattr(model.config, "num_attention_heads", 0))
        if num_heads <= 0:
            raise ValueError("Cannot infer Qwen num_attention_heads.")

        attn_head_dim = int(getattr(first_attn, "head_dim", 0))
        if attn_head_dim <= 0:
            if hidden_dim % num_heads != 0:
                raise ValueError(
                    f"Cannot infer head dim: hidden={hidden_dim}, heads={num_heads}."
                )
            attn_head_dim = hidden_dim // num_heads

        ffn_dim = int(getattr(model.config, "intermediate_size", hidden_dim * 4))

        expert = cls(
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            attn_head_dim=attn_head_dim,
            ffn_dim=ffn_dim,
            num_layers=num_layers,
            vocab_size=vocab_size,
            max_task_len=max_task_len,
            max_subtask_len=max_subtask_len,
            eps=eps,
            use_gradient_checkpointing=use_gradient_checkpointing,
            dtype=dtype,
        )

        expert.token_embedding = emb
        with torch.no_grad():
            expert.segment_embedding.weight.zero_()
        if lm_head is not None:
            expert.lm_head = lm_head
        else:
            expert.lm_head = nn.Linear(hidden_dim, vocab_size, bias=False)
            expert.lm_head.weight = expert.token_embedding.weight

        if final_norm is not None:
            expert.final_norm = final_norm

        adapted_blocks = []
        for layer in layers:
            adapted_blocks.append(
                _QwenBlockForMoT(
                    qwen_layer=layer,
                    num_heads=num_heads,
                    attn_head_dim=attn_head_dim,
                    hidden_dim=hidden_dim,
                )
            )
        expert.blocks = nn.ModuleList(adapted_blocks)
        expert.use_gradient_checkpointing = bool(use_gradient_checkpointing)

        logger.info(
            "Initialized QwenLanguageExpert (block-level) from %s: hidden=%d heads=%d head_dim=%d layers=%d vocab=%d",
            model_id,
            hidden_dim,
            num_heads,
            attn_head_dim,
            num_layers,
            vocab_size,
        )

        return expert

    @classmethod
    def from_pretrained_qwen(
        cls,
        model_id: str,
        max_task_len: int = 128,
        max_subtask_len: int = 128,
        eps: float = 1e-6,
        use_gradient_checkpointing: bool = False,
        dtype: torch.dtype = torch.bfloat16,
        local_files_only: bool = False,
    ) -> "QwenLanguageExpert":
        try:
            from transformers import AutoConfig, AutoModelForCausalLM
        except Exception as exc:  # pragma: no cover
            raise ImportError("QwenLanguageExpert requires `transformers`.") from exc

        config = AutoConfig.from_pretrained(
            model_id,
            trust_remote_code=True,
            local_files_only=local_files_only,
        )
        if getattr(config, "model_type", None) == "qwen3_vl":
            qwen, lm_head = cls._load_qwen3_vl_text_model(
                model_id=model_id,
                dtype=dtype,
                local_files_only=local_files_only,
            )
            expert = cls._from_loaded_qwen(
                model_id=model_id,
                model=qwen,
                lm_head=lm_head,
                max_task_len=max_task_len,
                max_subtask_len=max_subtask_len,
                eps=eps,
                use_gradient_checkpointing=use_gradient_checkpointing,
                dtype=dtype,
            )
            del qwen
            return expert

        qwen = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            trust_remote_code=True,
            local_files_only=local_files_only,
        )
        expert = cls._from_loaded_qwen(
            model_id=model_id,
            model=qwen,
            lm_head=None,
            max_task_len=max_task_len,
            max_subtask_len=max_subtask_len,
            eps=eps,
            use_gradient_checkpointing=use_gradient_checkpointing,
            dtype=dtype,
        )
        del qwen
        return expert
