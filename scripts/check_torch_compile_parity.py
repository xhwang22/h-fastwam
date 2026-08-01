#!/usr/bin/env python3
"""Validate opt-in torch.compile for DiT post-attention blocks."""

from __future__ import annotations

import copy

import torch
from torch import nn

from fastwam.models.hfastwam.qwen_language_expert import _QwenBlockForMoT
from fastwam.models.wan22.mot import MoT
from fastwam.models.wan22.wan_video_dit import (
    DiTBlock,
    modulate,
    precompute_freqs_cis,
)
from fastwam.utils.torch_compile import (
    compile_method_in_place,
    restore_eager_method,
)


def reference_forward_post(
    block: DiTBlock,
    residual_x: torch.Tensor,
    mixed_attn_out: torch.Tensor,
    gate_msa: torch.Tensor,
    shift_mlp: torch.Tensor,
    scale_mlp: torch.Tensor,
    gate_mlp: torch.Tensor,
    context: torch.Tensor | None,
    context_mask: torch.Tensor | None,
) -> torch.Tensor:
    x = block.gate(residual_x, gate_msa, block.self_attn.o(mixed_attn_out))
    if context is not None:
        if context_mask is not None and context_mask.dim() == 3:
            context_mask = context_mask.unsqueeze(1)
        x = x + block.cross_attn(block.norm3(x), context, ctx_mask=context_mask)
    mlp_input = modulate(block.norm2(x), shift_mlp, scale_mlp)
    return block.gate(x, gate_mlp, block.ffn(mlp_input))


def make_inputs(seq_len: int, with_context: bool):
    tensors = [
        torch.randn(2, seq_len, 16, requires_grad=True),
        torch.randn(2, seq_len, 16, requires_grad=True),
        torch.randn(2, 1, 16, requires_grad=True),
        torch.randn(2, 1, 16, requires_grad=True),
        torch.randn(2, 1, 16, requires_grad=True),
        torch.randn(2, 1, 16, requires_grad=True),
    ]
    context = (
        torch.randn(2, 1, 16, requires_grad=True)
        if with_context
        else None
    )
    context_mask = (
        torch.ones(2, 1, seq_len, 1, dtype=torch.bool)
        if with_context
        else None
    )
    return tensors, context, context_mask


def run(
    block: DiTBlock,
    seq_len: int,
    with_context: bool,
    *,
    use_reference: bool,
    use_checkpoint: bool,
):
    tensors, context, context_mask = make_inputs(seq_len, with_context)

    def forward(*values):
        args = (*values, context, context_mask)
        if use_reference:
            return reference_forward_post(block, *args)
        return block.forward_post(*args)

    if use_checkpoint:
        if use_reference:
            output = torch.utils.checkpoint.checkpoint(
                forward,
                *tensors,
                use_reentrant=False,
            )
        else:
            mot_runner = MoT.__new__(MoT)
            torch.nn.Module.__init__(mot_runner)
            mot_runner.train()
            output = mot_runner._apply_post_with_optional_checkpoint(
                block=block,
                residual_x=tensors[0],
                gate_msa=tensors[2],
                shift_mlp=tensors[3],
                scale_mlp=tensors[4],
                gate_mlp=tensors[5],
                use_gradient_checkpointing=True,
                mixed_slice=tensors[1],
                context_payload=(
                    {"context": context, "mask": context_mask}
                    if context is not None
                    else None
                ),
            )
    else:
        output = forward(*tensors)
    output.square().mean().backward()

    input_grads = [tensor.grad.detach().clone() for tensor in tensors]
    context_grad = context.grad.detach().clone() if context is not None else None
    parameter_grads = {
        name: parameter.grad.detach().clone()
        for name, parameter in block.named_parameters()
        if parameter.grad is not None
    }
    return output.detach(), input_grads, context_grad, parameter_grads


def assert_result_close(actual, expected, name: str) -> None:
    torch.testing.assert_close(actual[0], expected[0], rtol=1e-4, atol=1e-5)
    for index, (actual_grad, expected_grad) in enumerate(zip(actual[1], expected[1])):
        torch.testing.assert_close(
            actual_grad,
            expected_grad,
            rtol=1e-4,
            atol=1e-5,
            msg=lambda message: f"{name} input_grad[{index}]: {message}",
        )
    if actual[2] is not None or expected[2] is not None:
        torch.testing.assert_close(actual[2], expected[2], rtol=1e-4, atol=1e-5)
    if actual[3].keys() != expected[3].keys():
        raise AssertionError(f"{name} parameter-gradient keys differ.")
    for parameter_name in actual[3]:
        torch.testing.assert_close(
            actual[3][parameter_name],
            expected[3][parameter_name],
            rtol=1e-4,
            atol=1e-5,
            msg=lambda message: f"{name} parameter {parameter_name}: {message}",
        )


def check_whole_block_checkpoint_fallback() -> None:
    torch.manual_seed(456)
    eager_block = DiTBlock(16, 8, 2, 32).train()
    compiled_block = copy.deepcopy(eager_block).train()
    compile_method_in_place(
        compiled_block,
        "forward_post",
        backend="inductor",
        mode="default",
        dynamic=False,
        fullgraph=False,
    )

    def run_block(block: DiTBlock):
        x = torch.randn(2, 4, 16, requires_grad=True)
        context = torch.randn(2, 1, 16, requires_grad=True)
        t_mod = torch.randn(2, 6, 16, requires_grad=True)
        freqs = precompute_freqs_cis(8, end=4).view(4, 1, -1)
        context_mask = torch.ones(2, 4, 1, dtype=torch.bool)
        self_mask = torch.tril(torch.ones(4, 4, dtype=torch.bool))

        def forward(x_value, context_value, t_mod_value):
            return block(
                x_value,
                context_value,
                t_mod_value,
                freqs,
                context_mask=context_mask,
                self_attn_mask=self_mask,
            )

        output = torch.utils.checkpoint.checkpoint(
            forward,
            x,
            context,
            t_mod,
            use_reentrant=False,
        )
        output.square().mean().backward()
        return (
            output.detach(),
            [x.grad.detach(), context.grad.detach(), t_mod.grad.detach()],
            None,
            {
                name: parameter.grad.detach().clone()
                for name, parameter in block.named_parameters()
                if parameter.grad is not None
            },
        )

    rng_state = torch.random.get_rng_state()
    eager = run_block(eager_block)
    torch.random.set_rng_state(rng_state)
    compiled = run_block(compiled_block)
    assert_result_close(
        compiled,
        eager,
        "whole-block checkpoint eager fallback",
    )


def check_qwen_block_eager_fallback() -> None:
    class DummyAttention(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(16, 16)
            self.k_proj = nn.Linear(16, 16)
            self.v_proj = nn.Linear(16, 16)
            self.o_proj = nn.Linear(16, 16)
            self.q_norm = None
            self.k_norm = None

    class DummyLayer(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = DummyAttention()
            self.input_layernorm = nn.LayerNorm(16)
            self.post_attention_layernorm = nn.LayerNorm(16)
            self.mlp = nn.Sequential(
                nn.Linear(16, 32),
                nn.GELU(),
                nn.Linear(32, 16),
            )

    torch.manual_seed(789)
    reference_block = _QwenBlockForMoT(
        qwen_layer=DummyLayer(),
        num_heads=2,
        attn_head_dim=8,
        hidden_dim=16,
    ).train()
    actual_block = copy.deepcopy(reference_block).train()

    def run(block, use_fallback):
        values = [
            torch.randn(2, 4, 16, requires_grad=True),
            torch.randn(2, 4, 16, requires_grad=True),
            torch.randn(2, 1, 16, requires_grad=True),
            torch.randn(2, 1, 16, requires_grad=True),
            torch.randn(2, 1, 16, requires_grad=True),
            torch.randn(2, 1, 16, requires_grad=True),
        ]
        if use_fallback:
            output = MoT._apply_expert_post_block(
                block=block,
                residual_x=values[0],
                mixed_attn_out=values[1],
                gate_msa=values[2],
                shift_mlp=values[3],
                scale_mlp=values[4],
                gate_mlp=values[5],
                context_payload=None,
            )
        else:
            x = block.gate(values[0], values[2], block.self_attn.o(values[1]))
            mlp_input = modulate(block.norm2(x), values[3], values[4])
            output = block.gate(x, values[5], block.ffn(mlp_input))
        output.square().mean().backward()
        return (
            output.detach(),
            [value.grad.detach() for value in values],
            None,
            {
                name: parameter.grad.detach().clone()
                for name, parameter in block.named_parameters()
                if parameter.grad is not None
            },
        )

    rng_state = torch.random.get_rng_state()
    reference = run(reference_block, use_fallback=False)
    torch.random.set_rng_state(rng_state)
    actual = run(actual_block, use_fallback=True)
    assert_result_close(actual, reference, "Qwen block eager fallback")


def main() -> None:
    torch.manual_seed(123)
    reference_block = DiTBlock(
        hidden_dim=16,
        attn_head_dim=8,
        num_heads=2,
        ffn_dim=32,
    ).train()
    eager_block = copy.deepcopy(reference_block).train()
    compiled_block = copy.deepcopy(reference_block).train()

    state_keys_before = tuple(compiled_block.state_dict().keys())
    compile_method_in_place(
        compiled_block,
        "forward_post",
        backend="inductor",
        mode="default",
        dynamic=False,
        fullgraph=False,
    )
    if tuple(compiled_block.state_dict().keys()) != state_keys_before:
        raise AssertionError("Compiled post block changed state_dict keys.")

    for seq_len, with_context in ((4, True), (6, True), (4, False)):
        for block in (reference_block, eager_block, compiled_block):
            block.zero_grad(set_to_none=True)

        rng_state = torch.random.get_rng_state()
        reference = run(
            reference_block,
            seq_len,
            with_context,
            use_reference=True,
            use_checkpoint=True,
        )
        torch.random.set_rng_state(rng_state)
        eager = run(
            eager_block,
            seq_len,
            with_context,
            use_reference=False,
            use_checkpoint=True,
        )
        torch.random.set_rng_state(rng_state)
        compiled = run(
            compiled_block,
            seq_len,
            with_context,
            use_reference=False,
            use_checkpoint=True,
        )

        assert_result_close(eager, reference, f"eager seq={seq_len}")
        assert_result_close(compiled, reference, f"compiled seq={seq_len}")

    warmed_attr = "_fastwam_compiled_forward_post_warmed_signatures"
    if not getattr(compiled_block, warmed_attr, None):
        raise AssertionError("Compiled checkpoint path did not record warmed signatures.")
    compiled_block.to(dtype=torch.float64)
    if getattr(compiled_block, warmed_attr, None):
        raise AssertionError("Module._apply() did not clear compiled warmup signatures.")

    if not restore_eager_method(compiled_block, "forward_post"):
        raise AssertionError("Failed to restore eager forward_post.")
    if tuple(compiled_block.state_dict().keys()) != state_keys_before:
        raise AssertionError("Restoring eager post block changed state_dict keys.")

    check_whole_block_checkpoint_fallback()
    check_qwen_block_eager_fallback()
    print("torch.compile post-block output/gradient/state_dict parity passed.")


if __name__ == "__main__":
    main()
