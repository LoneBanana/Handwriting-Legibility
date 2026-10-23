"""Low-rank adapters (LoRA) for per-user handwriting personalization.

Every person's handwriting is idiosyncratic, and dysgraphic handwriting especially so:
a letter one writer forms reliably may be ambiguous for everyone else. A small adapter
learned from a person's own corrected lines captures those habits without touching
the shared model. Adapters wrap selected linear layers (attention, MLPs, and the
pointwise convolutions of the last CNN stages, which see stroke shapes) as
``W x + (B A x) * alpha / r``. ``B`` starts at zero, so an inactive or freshly
initialised adapter leaves the base model's output unchanged, and swapping one user's
adapter for another's is a copy of a few hundred kilobytes.
"""

from __future__ import annotations

import fnmatch
import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn

DEFAULT_TARGETS = (
    "cnn.stages.2.*.pw1",
    "cnn.stages.2.*.pw2",
    "cnn.stages.3.*.pw1",
    "cnn.stages.3.*.pw2",
    "encoder.blocks.*.attn.qkv",
    "encoder.blocks.*.attn.proj",
    "encoder.blocks.*.mlp.fc1",
    "encoder.blocks.*.mlp.fc2",
    "decoder.blocks.*.self_attn.qkv",
    "decoder.blocks.*.cross_attn.q",
    "decoder.blocks.*.cross_attn.v",
    "decoder.blocks.*.mlp.fc1",
    "decoder.blocks.*.mlp.fc2",
)


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float):
        super().__init__()
        self.base = base
        self.rank = rank
        self.scale = alpha / rank
        self.lora_a = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device))
        self.lora_b = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device))
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        self.active = False

    def forward(self, x: Tensor) -> Tensor:
        y = self.base(x)
        if self.active:
            y = y + F.linear(F.linear(x, self.lora_a.to(x.dtype)), self.lora_b.to(x.dtype)) * self.scale
        return y


def _parent(model: nn.Module, name: str) -> tuple[nn.Module, str]:
    parent_name, _, child = name.rpartition(".")
    return (model.get_submodule(parent_name) if parent_name else model), child


def inject_lora(model: nn.Module, rank: int = 8, alpha: float = 16.0,
                targets: tuple[str, ...] = DEFAULT_TARGETS) -> list[str]:
    """Wraps matching ``nn.Linear`` layers in place; returns the wrapped module names.

    Load base weights *before* calling this: wrapping moves ``x.weight`` to ``x.base.weight``.
    """
    names = [
        name for name, module in model.named_modules()
        if isinstance(module, nn.Linear) and any(fnmatch.fnmatchcase(name, t) for t in targets)
    ]
    for name in names:
        parent, child = _parent(model, name)
        setattr(parent, child, LoRALinear(getattr(parent, child), rank, alpha))
    return names


def lora_modules(model: nn.Module) -> dict[str, LoRALinear]:
    return {name: m for name, m in model.named_modules() if isinstance(m, LoRALinear)}


def lora_parameters(model: nn.Module) -> list[nn.Parameter]:
    return [p for m in lora_modules(model).values() for p in (m.lora_a, m.lora_b)]


def lora_state(model: nn.Module) -> dict[str, Tensor]:
    state = {}
    for name, m in lora_modules(model).items():
        state[f"{name}.lora_a"] = m.lora_a.detach().cpu().clone()
        state[f"{name}.lora_b"] = m.lora_b.detach().cpu().clone()
    return state


def set_lora_active(model: nn.Module, active: bool) -> None:
    for m in lora_modules(model).values():
        m.active = active


@torch.no_grad()
def clear_lora(model: nn.Module) -> None:
    """Deactivates and zeroes every adapter, so no trace of a previous user's adapter remains."""
    for m in lora_modules(model).values():
        m.lora_b.zero_()
        m.active = False


@torch.no_grad()
def load_lora_state(model: nn.Module, state: dict[str, Tensor]) -> None:
    """Copies an adapter into the model's LoRA layers and activates it."""
    clear_lora(model)
    modules = lora_modules(model)
    for key, value in state.items():
        name, _, which = key.rpartition(".")
        module = modules.get(name)
        if module is None:
            raise KeyError(f"adapter has weights for {name!r}, which this model does not wrap")
        target = module.lora_a if which == "lora_a" else module.lora_b
        if target.shape != value.shape:
            raise ValueError(f"adapter shape mismatch for {key}: {tuple(value.shape)} vs {tuple(target.shape)}")
        target.copy_(value.to(target.device, target.dtype))
    set_lora_active(model, True)
