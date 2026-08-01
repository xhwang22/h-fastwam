"""Opt-in helpers for compiling hot module forwards without replacing modules."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


VALID_COMPILE_MODES = {
    "default",
    "reduce-overhead",
    "max-autotune",
    "max-autotune-no-cudagraphs",
}


def compile_method_in_place(
    module: nn.Module,
    method_name: str,
    *,
    backend: str = "inductor",
    mode: str = "default",
    dynamic: bool = True,
    fullgraph: bool = False,
) -> dict[str, Any]:
    """Compile a bound module method while preserving module/state_dict identity."""
    if not hasattr(module, method_name):
        raise AttributeError(f"{type(module).__name__} has no method {method_name!r}.")
    compiled_methods = dict(getattr(module, "_fastwam_torch_compiled_methods", {}))
    if method_name in compiled_methods:
        return dict(compiled_methods[method_name])
    if mode not in VALID_COMPILE_MODES:
        raise ValueError(
            f"Unsupported torch.compile mode {mode!r}; expected one of "
            f"{sorted(VALID_COMPILE_MODES)}."
        )
    if not hasattr(torch, "compile"):
        raise RuntimeError("This PyTorch build does not provide torch.compile.")

    state_keys_before = tuple(module.state_dict().keys())
    eager_method = getattr(module, method_name)
    compiled_method = torch.compile(
        eager_method,
        backend=backend,
        mode=mode,
        dynamic=bool(dynamic),
        fullgraph=bool(fullgraph),
    )
    eager_attr = f"_fastwam_eager_{method_name}"
    warmed_attr = f"_fastwam_compiled_{method_name}_warmed_signatures"
    if hasattr(module, warmed_attr):
        delattr(module, warmed_attr)
    setattr(module, eager_attr, eager_method)
    setattr(module, method_name, compiled_method)

    state_keys_after = tuple(module.state_dict().keys())
    if state_keys_after != state_keys_before:
        setattr(module, method_name, eager_method)
        delattr(module, eager_attr)
        warmed_attr = f"_fastwam_compiled_{method_name}_warmed_signatures"
        if hasattr(module, warmed_attr):
            delattr(module, warmed_attr)
        raise RuntimeError(
            f"Compiling {method_name} changed state_dict keys; refusing to continue."
        )

    config = {
        "backend": str(backend),
        "mode": str(mode),
        "dynamic": bool(dynamic),
        "fullgraph": bool(fullgraph),
    }
    compiled_methods[method_name] = config
    module._fastwam_torch_compiled_methods = compiled_methods
    return dict(config)


def compile_forward_in_place(
    module: nn.Module,
    *,
    backend: str = "inductor",
    mode: str = "default",
    dynamic: bool = True,
    fullgraph: bool = False,
) -> dict[str, Any]:
    return compile_method_in_place(
        module,
        "forward",
        backend=backend,
        mode=mode,
        dynamic=dynamic,
        fullgraph=fullgraph,
    )


def restore_eager_method(module: nn.Module, method_name: str) -> bool:
    """Restore an original method after an in-place compile."""
    eager_attr = f"_fastwam_eager_{method_name}"
    eager_method = getattr(module, eager_attr, None)
    if eager_method is None:
        return False
    setattr(module, method_name, eager_method)
    delattr(module, eager_attr)
    compiled_methods = dict(getattr(module, "_fastwam_torch_compiled_methods", {}))
    compiled_methods.pop(method_name, None)
    if compiled_methods:
        module._fastwam_torch_compiled_methods = compiled_methods
    elif hasattr(module, "_fastwam_torch_compiled_methods"):
        delattr(module, "_fastwam_torch_compiled_methods")
    return True


def restore_eager_forward(module: nn.Module) -> bool:
    return restore_eager_method(module, "forward")
