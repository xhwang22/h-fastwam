#!/usr/bin/env python3
"""Identify the SDPA backend used by the current structured attention shape."""

from __future__ import annotations

import argparse
import contextlib
import time

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.profiler import ProfilerActivity, profile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--heads", type=int, default=16)
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--video-tokens", type=int, default=360)
    parser.add_argument("--action-tokens", type=int, default=32)
    parser.add_argument("--video-states", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument(
        "--mask",
        choices=("structured", "explicit-causal", "causal", "none"),
        default="structured",
    )
    parser.add_argument("--forward-only", action="store_true")
    return parser.parse_args()


def build_structured_mask(
    video_tokens: int,
    action_tokens: int,
    video_states: int,
    device: torch.device,
) -> torch.Tensor:
    if video_tokens % video_states != 0:
        raise ValueError("--video-tokens must be divisible by --video-states.")

    tokens_per_state = video_tokens // video_states
    total_tokens = video_tokens + action_tokens
    mask = torch.zeros((total_tokens, total_tokens), dtype=torch.bool, device=device)

    state_causal = torch.tril(
        torch.ones((video_states, video_states), dtype=torch.bool, device=device)
    )
    mask[:video_tokens, :video_tokens] = state_causal.repeat_interleave(
        tokens_per_state, dim=0
    ).repeat_interleave(tokens_per_state, dim=1)

    mask[video_tokens:, :video_tokens] = True
    mask[video_tokens:, video_tokens:] = torch.tril(
        torch.ones((action_tokens, action_tokens), dtype=torch.bool, device=device)
    )
    return mask


def make_mask(args: argparse.Namespace, device: torch.device) -> tuple[torch.Tensor | None, bool]:
    total_tokens = args.video_tokens + args.action_tokens
    if args.mask == "none":
        return None, False
    if args.mask == "causal":
        return None, True
    if args.mask == "explicit-causal":
        return torch.tril(
            torch.ones((total_tokens, total_tokens), dtype=torch.bool, device=device)
        ), False
    return build_structured_mask(
        video_tokens=args.video_tokens,
        action_tokens=args.action_tokens,
        video_states=args.video_states,
        device=device,
    ), False


def attention_step(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None,
    is_causal: bool,
    backward: bool,
) -> None:
    q.grad = None
    k.grad = None
    v.grad = None
    output = F.scaled_dot_product_attention(
        q,
        k,
        v,
        attn_mask=mask,
        dropout_p=0.0,
        is_causal=is_causal,
    )
    if backward:
        output.float().square().mean().backward()


def time_backend(
    backend: SDPBackend | None,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None,
    is_causal: bool,
    backward: bool,
    warmup: int,
    iterations: int,
) -> float:
    backend_context = contextlib.nullcontext() if backend is None else sdpa_kernel(backend)
    with backend_context:
        for _ in range(warmup):
            attention_step(q, k, v, mask, is_causal, backward)
        torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(iterations):
            attention_step(q, k, v, mask, is_causal, backward)
        torch.cuda.synchronize()
        return (time.perf_counter() - start) * 1000 / iterations


def profile_auto_backend(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None,
    is_causal: bool,
    backward: bool,
) -> None:
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        attention_step(q, k, v, mask, is_causal, backward)
        torch.cuda.synchronize()

    print("\nAUTO backend top CUDA operations:")
    events = []
    for event in prof.key_averages():
        device_time = getattr(
            event,
            "self_device_time_total",
            getattr(event, "self_cuda_time_total", 0.0),
        )
        if device_time > 0:
            events.append((float(device_time), event.key))
    for device_time, key in sorted(events, reverse=True)[:15]:
        print(f"  {device_time / 1000:9.3f} ms  {key}")

    keys = " ".join(key.lower() for _, key in events)
    if "flash_attention" in keys:
        selected = "Flash Attention"
    elif "cudnn_attention" in keys:
        selected = "cuDNN Attention"
    elif "efficient_attention" in keys:
        selected = "Efficient Attention"
    elif "attention_math" in keys or "aten::bmm" in keys:
        selected = "Math fallback"
    else:
        selected = "Unknown; inspect the operation list above"
    print(f"AUTO backend classification: {selected}")


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")
    if min(
        args.batch_size,
        args.heads,
        args.head_dim,
        args.video_tokens,
        args.action_tokens,
        args.video_states,
        args.iterations,
    ) <= 0:
        raise ValueError("Shape and iteration arguments must be positive.")

    device = torch.device("cuda")
    total_tokens = args.video_tokens + args.action_tokens
    backward = not args.forward_only

    q = torch.randn(
        args.batch_size,
        args.heads,
        total_tokens,
        args.head_dim,
        device=device,
        dtype=torch.bfloat16,
        requires_grad=backward,
    )
    k = torch.randn_like(q, requires_grad=backward)
    v = torch.randn_like(q, requires_grad=backward)
    mask, is_causal = make_mask(args, device)

    print(f"torch={torch.__version__} cuda={torch.version.cuda}")
    print(f"gpu={torch.cuda.get_device_name(device)}")
    print(
        f"shape=[B={args.batch_size}, H={args.heads}, S={total_tokens}, D={args.head_dim}] "
        f"mask={args.mask} backward={backward}"
    )

    profile_auto_backend(q, k, v, mask, is_causal, backward)

    print("\nForced backend support and average latency:")
    backends: list[tuple[str, SDPBackend | None]] = [("AUTO", None)]
    for name in ("FLASH_ATTENTION", "CUDNN_ATTENTION", "EFFICIENT_ATTENTION", "MATH"):
        if hasattr(SDPBackend, name):
            backends.append((name, getattr(SDPBackend, name)))

    for name, backend in backends:
        try:
            latency_ms = time_backend(
                backend=backend,
                q=q,
                k=k,
                v=v,
                mask=mask,
                is_causal=is_causal,
                backward=backward,
                warmup=args.warmup,
                iterations=args.iterations,
            )
            print(f"  {name:21s} supported  {latency_ms:9.3f} ms/iteration")
        except Exception as exc:
            print(f"  {name:21s} unsupported  {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
