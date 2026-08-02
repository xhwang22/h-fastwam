#!/usr/bin/env python3
"""Check behavior-preserving FastWAM hot-path optimizations."""

from __future__ import annotations

import copy
import io
from collections import OrderedDict
from types import SimpleNamespace

import torch

from fastwam.models.hfastwam.hfastwam import HFastWAM
from fastwam.models.wan22.action_dit import ActionDiT
from fastwam.models.wan22.jepa_predictor import JEPAPredictor
from fastwam.models.wan22.mot import MoT
from fastwam.models.wan22.wan_video_dit import (
    DiTBlock,
    RMSNorm,
    WanVideoDiT,
    flash_attention,
)


def assert_tensor_equal(actual: torch.Tensor, expected: torch.Tensor, name: str) -> None:
    if not torch.equal(actual, expected):
        max_error = float((actual - expected).abs().max().item())
        raise AssertionError(f"{name} differs; max_abs_error={max_error}")


def check_fused_rms_norm() -> None:
    for dtype in (torch.float32, torch.bfloat16):
        torch.manual_seed(11)
        expected_norm = RMSNorm(64, eps=1e-6).to(dtype=dtype)
        actual_norm = copy.deepcopy(expected_norm)
        expected_input = torch.randn(4, 17, 64, dtype=dtype, requires_grad=True)
        actual_input = expected_input.detach().clone().requires_grad_(True)

        expected_output = (
            expected_input.float()
            * torch.rsqrt(
                expected_input.float().pow(2).mean(dim=-1, keepdim=True) + 1e-6
            )
        ).to(dtype) * expected_norm.weight
        actual_output = actual_norm(actual_input)

        torch.testing.assert_close(
            actual_output,
            expected_output,
            rtol=0,
            atol=0,
            msg=lambda message: f"RMSNorm output ({dtype}): {message}",
        )

        expected_output.float().square().mean().backward()
        actual_output.float().square().mean().backward()
        torch.testing.assert_close(
            actual_input.grad,
            expected_input.grad,
            rtol=1e-3,
            atol=1e-4,
            msg=lambda message: f"RMSNorm input gradient ({dtype}): {message}",
        )
        torch.testing.assert_close(
            actual_norm.weight.grad,
            expected_norm.weight.grad,
            rtol=0,
            atol=0,
            msg=lambda message: f"RMSNorm weight gradient ({dtype}): {message}",
        )


def check_identity_projection_bypass() -> None:
    torch.manual_seed(12)

    def make_expert():
        return JEPAPredictor(
            hidden_dim=16,
            in_dim=8,
            out_dim=8,
            ffn_dim=32,
            text_dim=16,
            patch_size=(1, 2, 2),
            num_heads=2,
            attn_head_dim=8,
            num_layers=1,
            use_text_context=False,
        )

    mot = MoT(
        mixtures={"video": make_expert(), "action": make_expert()},
        mot_checkpoint_mixed_attn=False,
        strict_expert_compat=False,
        layer_alignment_mode="tail_overlap",
        shared_attention_expert="video",
    )
    key = "video__0"
    if not isinstance(mot.q_proj_to_shared[key], torch.nn.Identity):
        raise AssertionError("Expected an identity projection key for matching experts.")

    q = torch.randn(2, 4, 16, requires_grad=True)
    k = torch.randn(2, 4, 16, requires_grad=True)
    v = torch.randn(2, 4, 16, requires_grad=True)
    q_out, k_out, v_out = mot._project_qkv_to_shared(
        name="video",
        layer_idx=0,
        q=q,
        k=k,
        v=v,
    )
    if q_out is not q or k_out is not k or v_out is not v:
        raise AssertionError("Identity QKV projections should return original tensors.")

    mixed = torch.randn(2, 4, 16, requires_grad=True)
    mixed_out = mot._project_mixed_from_shared(
        name="video",
        layer_idx=0,
        mixed=mixed,
    )
    if mixed_out is not mixed:
        raise AssertionError("Identity output projection should return the original tensor.")

    (q_out.sum() + k_out.sum() + v_out.sum() + mixed_out.sum()).backward()
    for name, tensor in (("q", q), ("k", k), ("v", v), ("mixed", mixed)):
        assert_tensor_equal(tensor.grad, torch.ones_like(tensor), f"identity {name} grad")

    payload = io.BytesIO()
    torch.save(mot, payload)
    payload.seek(0)
    restored = torch.load(payload, weights_only=False)
    restored_q, restored_k, restored_v = restored._project_qkv_to_shared(
        name="video",
        layer_idx=0,
        q=q.detach(),
        k=k.detach(),
        v=v.detach(),
    )
    if (
        restored_q.data_ptr() != q.detach().data_ptr()
        or restored_k.data_ptr() != k.detach().data_ptr()
        or restored_v.data_ptr() != v.detach().data_ptr()
    ):
        raise AssertionError("Serialized identity projections should remain bypassed.")


def check_action_rope_cache() -> ActionDiT:
    torch.manual_seed(1)
    model = ActionDiT(
        hidden_dim=16,
        action_dim=4,
        ffn_dim=32,
        text_dim=16,
        freq_dim=8,
        eps=1e-6,
        num_heads=2,
        attn_head_dim=8,
        num_layers=1,
    ).eval()
    actions = torch.randn(2, 3, 4)
    timestep = torch.rand(2)
    context = torch.randn(2, 1, 16)
    context_mask = torch.ones(2, 1, dtype=torch.bool)

    first = model.pre_dit(actions, timestep, context, context_mask)
    second = model.pre_dit(actions, timestep, context, context_mask)
    for key in ("tokens", "freqs", "t", "t_mod", "context", "context_mask"):
        assert_tensor_equal(first[key], second[key], f"ActionDiT.{key}")
    if first["freqs"].data_ptr() != second["freqs"].data_ptr():
        raise AssertionError("ActionDiT RoPE cache was not reused.")
    for seq_len in range(1, 21):
        model.pre_dit(
            torch.randn(2, seq_len, 4),
            timestep,
            context,
            context_mask,
        )
    if len(model._freqs_device_cache) > model._MAX_FREQS_DEVICE_CACHE_ENTRIES:
        raise AssertionError("ActionDiT RoPE LRU exceeded its capacity.")
    return model


def check_jepa_rope_cache() -> JEPAPredictor:
    torch.manual_seed(2)
    model = JEPAPredictor(
        hidden_dim=16,
        in_dim=8,
        out_dim=8,
        ffn_dim=32,
        text_dim=16,
        patch_size=(1, 2, 2),
        num_heads=2,
        attn_head_dim=8,
        num_layers=1,
        video_attention_mask_mode="per_frame_causal",
        use_text_context=True,
    ).eval()
    video = torch.randn(2, 8, 2, 4, 4)
    context = torch.randn(2, 1, 16)
    context_mask = torch.ones(2, 1, dtype=torch.bool)

    first = model.pre_dit(video, context, context_mask)
    second = model.pre_dit(video, context, context_mask)
    for key in ("tokens", "freqs", "t_mod", "context", "context_mask"):
        assert_tensor_equal(first[key], second[key], f"JEPAPredictor.{key}")
    if first["freqs"].data_ptr() != second["freqs"].data_ptr():
        raise AssertionError("JEPAPredictor RoPE cache was not reused.")
    for frames in range(1, 21):
        model.pre_dit(
            torch.randn(2, 8, frames, 4, 4),
            context,
            context_mask,
        )
    if len(model._freqs_device_cache) > model._MAX_FREQS_DEVICE_CACHE_ENTRIES:
        raise AssertionError("JEPAPredictor RoPE LRU exceeded its capacity.")
    return model


def check_wan_video_rope_cache() -> WanVideoDiT:
    torch.manual_seed(4)
    model = WanVideoDiT(
        hidden_dim=16,
        in_dim=8,
        ffn_dim=32,
        out_dim=8,
        text_dim=16,
        freq_dim=8,
        eps=1e-6,
        patch_size=(1, 2, 2),
        num_heads=2,
        attn_head_dim=8,
        num_layers=1,
        has_image_input=False,
        seperated_timestep=True,
        require_vae_embedding=False,
        require_clip_embedding=False,
        fuse_vae_embedding_in_latents=True,
    ).eval()
    video = torch.randn(2, 8, 2, 4, 4)
    timestep = torch.rand(2)
    context = torch.randn(2, 1, 16)
    context_mask = torch.ones(2, 1, dtype=torch.bool)

    first = model.pre_dit(
        video,
        timestep,
        context,
        context_mask,
        fuse_vae_embedding_in_latents=True,
    )
    second = model.pre_dit(
        video,
        timestep,
        context,
        context_mask,
        fuse_vae_embedding_in_latents=True,
    )
    for key in ("tokens", "freqs", "t", "t_mod", "context", "context_mask"):
        assert_tensor_equal(first[key], second[key], f"WanVideoDiT.{key}")
    if first["freqs"].data_ptr() != second["freqs"].data_ptr():
        raise AssertionError("WanVideoDiT RoPE cache was not reused.")
    for frames in range(1, 21):
        model.pre_dit(
            torch.randn(2, 8, frames, 4, 4),
            timestep,
            context,
            context_mask,
            fuse_vae_embedding_in_latents=True,
        )
    if len(model._freqs_device_cache) > model._MAX_FREQS_DEVICE_CACHE_ENTRIES:
        raise AssertionError("WanVideoDiT RoPE LRU exceeded its capacity.")
    return model


def check_modulation_cast_elision() -> None:
    torch.manual_seed(3)
    block = DiTBlock(
        hidden_dim=16,
        attn_head_dim=8,
        num_heads=2,
        ffn_dim=32,
    ).to(dtype=torch.bfloat16)
    t_mod = torch.randn(2, 6, 16, dtype=torch.bfloat16)
    expected = tuple(
        (block.modulation.to(dtype=t_mod.dtype, device=t_mod.device) + t_mod).chunk(6, dim=1)
    )
    actual = MoT._split_modulation(block, t_mod)
    for index, (actual_tensor, expected_tensor) in enumerate(zip(actual, expected)):
        assert_tensor_equal(actual_tensor, expected_tensor, f"modulation[{index}]")


def check_attention_mask_device_fast_path() -> None:
    torch.manual_seed(5)
    q_ref = torch.randn(2, 4, 16, requires_grad=True)
    k_ref = torch.randn(2, 4, 16, requires_grad=True)
    v_ref = torch.randn(2, 4, 16, requires_grad=True)
    mask = torch.tril(torch.ones(4, 4, dtype=torch.bool))

    expected = flash_attention(q_ref, k_ref, v_ref, num_heads=2, ctx_mask=mask)
    expected.square().mean().backward()
    expected_grads = (q_ref.grad.clone(), k_ref.grad.clone(), v_ref.grad.clone())

    q = q_ref.detach().clone().requires_grad_(True)
    k = k_ref.detach().clone().requires_grad_(True)
    v = v_ref.detach().clone().requires_grad_(True)
    dummy = SimpleNamespace(mot_checkpoint_mixed_attn=False, training=True)
    actual = MoT._attention_with_num_heads(
        dummy,
        q_cat=q,
        k_cat=k,
        v_cat=v,
        attention_mask=mask,
        num_heads=2,
    )
    actual.square().mean().backward()

    assert_tensor_equal(actual, expected, "MoT attention output")
    for index, (actual_grad, expected_grad) in enumerate(
        zip((q.grad, k.grad, v.grad), expected_grads)
    ):
        assert_tensor_equal(actual_grad, expected_grad, f"MoT attention grad[{index}]")


def check_context_mask_normalization() -> None:
    mask = torch.tensor(
        [
            [[True, False], [True, True]],
            [[True, True], [False, True]],
        ]
    )
    payload = HFastWAM._context_payload_from_pre_state(
        {"context": torch.randn(2, 2, 8), "context_mask": mask},
        enabled=True,
    )
    expected = mask.unsqueeze(1)
    assert_tensor_equal(payload["mask"], expected, "context mask")


def check_full_attention_mask_cache(video_expert: JEPAPredictor) -> None:
    dummy = SimpleNamespace(
        video_expert=video_expert,
        _full_attention_mask_cache=OrderedDict(),
        _MAX_FULL_ATTENTION_MASK_CACHE_ENTRIES=32,
    )
    kwargs = dict(
        task_len=3,
        subtask_len=0,
        video_seq_len=8,
        action_seq_len=3,
        video_tokens_per_frame=4,
        device=torch.device("cpu"),
    )
    first = HFastWAM._build_full_attention_mask(dummy, **kwargs)
    second = HFastWAM._build_full_attention_mask(dummy, **kwargs)
    assert_tensor_equal(first, second, "full attention mask")
    if first.data_ptr() != second.data_ptr():
        raise AssertionError("Full attention mask cache was not reused.")

    for task_len in range(2, 40):
        HFastWAM._build_full_attention_mask(
            dummy,
            **{**kwargs, "task_len": task_len},
        )
    if len(dummy._full_attention_mask_cache) > 32:
        raise AssertionError("Full attention mask LRU exceeded its capacity.")


def check_zero_language_loss_skip() -> None:
    class FailingLanguageExpert:
        def post_dit(self, *_args, **_kwargs):
            raise AssertionError("zero-weight language post_dit must be skipped")

    dummy = SimpleNamespace(
        loss_lambda_language=0.0,
        language_expert=FailingLanguageExpert(),
    )
    total_loss = torch.tensor(2.0, requires_grad=True)
    loss_dict: dict[str, float] = {}
    result = HFastWAM._add_language_loss(
        dummy,
        total_loss=total_loss,
        loss_dict=loss_dict,
        tokens_out=torch.randn(2, 3, 8),
        pre_state={},
        task_ids=torch.ones(2, 3, dtype=torch.long),
        subtask_ids=torch.empty(2, 0, dtype=torch.long),
        task_len=3,
    )
    if result is not total_loss:
        raise AssertionError("Zero-weight language loss must preserve the total-loss tensor.")
    if loss_dict != {"loss_language": 0.0}:
        raise AssertionError(f"Unexpected zero language loss log: {loss_dict}")
    result.backward()
    assert_tensor_equal(total_loss.grad, torch.ones_like(total_loss), "total loss gradient")


def check_cache_clear_on_module_apply(
    action_model: ActionDiT,
    jepa_model: JEPAPredictor,
    wan_model: WanVideoDiT,
) -> None:
    for name, model in (
        ("ActionDiT", action_model),
        ("JEPAPredictor", jepa_model),
        ("WanVideoDiT", wan_model),
    ):
        if not model._freqs_device_cache:
            raise AssertionError(f"{name} cache must be populated before device migration.")
        model.to(dtype=torch.float64)
        if model._freqs_device_cache:
            raise AssertionError(f"{name} cache was not cleared by module._apply().")

    hfastwam = HFastWAM.__new__(HFastWAM)
    torch.nn.Module.__init__(hfastwam)
    hfastwam._full_attention_mask_cache = OrderedDict(
        [(("cached",), torch.ones(2, 2, dtype=torch.bool))]
    )
    hfastwam.to(dtype=torch.float64)
    if hfastwam._full_attention_mask_cache:
        raise AssertionError("HFastWAM mask cache was not cleared by module._apply().")


def main() -> None:
    check_fused_rms_norm()
    check_identity_projection_bypass()
    action_model = check_action_rope_cache()
    video_expert = check_jepa_rope_cache()
    wan_model = check_wan_video_rope_cache()
    check_modulation_cast_elision()
    check_attention_mask_device_fast_path()
    check_context_mask_normalization()
    check_full_attention_mask_cache(video_expert)
    check_zero_language_loss_skip()
    check_cache_clear_on_module_apply(action_model, video_expert, wan_model)
    print("All safe optimization parity checks passed.")


if __name__ == "__main__":
    main()
