"""Building blocks shared by the convolutional front end, encoder, and decoder."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class LayerNorm2d(nn.LayerNorm):
    """LayerNorm over the channels of an NCHW tensor, computed independently per pixel.

    Unlike BatchNorm/GroupNorm it never mixes statistics across positions, so right
    padding of variable-width lines cannot leak into real content, and it behaves the
    same at batch size 1 (inference, per-user fine-tuning) as during training.
    """

    def forward(self, x: Tensor) -> Tensor:
        x = x.permute(0, 2, 3, 1)
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x.permute(0, 3, 1, 2)


class DropPath(nn.Module):
    """Stochastic depth: randomly drops whole residual branches per sample."""

    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = p

    def forward(self, x: Tensor) -> Tensor:
        if self.p == 0.0 or not self.training:
            return x
        keep = 1.0 - self.p
        mask = x.new_empty((x.shape[0],) + (1,) * (x.ndim - 1)).bernoulli_(keep)
        return x * mask / keep


class Mlp(nn.Module):
    def __init__(self, dim: int, hidden: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.drop(self.fc2(self.drop(F.gelu(self.fc1(x)))))


# --------------------------------------------------------------------------- positions


def rope_cos_sin(positions: Tensor, dim: int, base: float) -> tuple[Tensor, Tensor]:
    """Rotary tables for 1-D ``positions`` (N,), returned as (N, dim) cos and sin."""
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=positions.device, dtype=torch.float32) / dim))
    angles = positions.float()[:, None] * inv_freq[None, :]
    angles = torch.cat([angles, angles], dim=-1)
    return angles.cos(), angles.sin()


def _rotate_half(x: Tensor) -> Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Rotate ``x`` (..., N, D) by position tables ``cos``/``sin`` (N, D)."""
    return (x.float() * cos + _rotate_half(x.float()) * sin).type_as(x)


class Rope1D:
    """Rotary embedding over a 1-D token sequence (decoder self-attention)."""

    def __init__(self, head_dim: int, base: float):
        if head_dim % 2:
            raise ValueError("head_dim must be even for rotary embeddings")
        self.head_dim = head_dim
        self.base = base

    def tables(self, positions: Tensor) -> tuple[Tensor, Tensor]:
        return rope_cos_sin(positions, self.head_dim, self.base)


class Rope2D:
    """Axial 2-D rotary embedding for a grid of image tokens (encoder self-attention).

    The first half of each head encodes the column, the second half the row, so
    attention scores depend on relative 2-D offsets. Because nothing is learned per
    position, the model extrapolates to lines wider than any it saw in training.
    """

    def __init__(self, head_dim: int, base: float):
        if head_dim % 4:
            raise ValueError("head_dim must be divisible by 4 for 2-D rotary embeddings")
        self.half = head_dim // 2
        self.base = base

    def tables(self, height: int, width: int, device: torch.device) -> tuple[Tensor, Tensor]:
        rows = torch.arange(height, device=device).repeat_interleave(width)
        cols = torch.arange(width, device=device).repeat(height)
        cos_x, sin_x = rope_cos_sin(cols, self.half, self.base)
        cos_y, sin_y = rope_cos_sin(rows, self.half, self.base)
        return torch.cat([cos_x, cos_y], dim=-1), torch.cat([sin_x, sin_y], dim=-1)


def apply_rope_2d(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """Apply axial rotary tables built by :class:`Rope2D` (halves rotated independently)."""
    half = x.shape[-1] // 2
    cx, cy = cos[..., :half], cos[..., half:]
    sx, sy = sin[..., :half], sin[..., half:]
    return torch.cat([apply_rope(x[..., :half], cx, sx), apply_rope(x[..., half:], cy, sy)], dim=-1)


def sinusoidal_1d(positions: Tensor, dim: int) -> Tensor:
    """Classic transformer sinusoidal embedding, (N,) -> (N, dim)."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000.0) * torch.arange(half, device=positions.device, dtype=torch.float32) / half)
    angles = positions.float()[:, None] * freqs[None, :]
    emb = torch.cat([angles.sin(), angles.cos()], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


def sinusoidal_2d(height: int, width: int, dim: int, device: torch.device) -> Tensor:
    """Absolute 2-D sinusoidal embedding for a row-major token grid, (H*W, dim)."""
    half = dim // 2
    rows = sinusoidal_1d(torch.arange(height, device=device), half)
    cols = sinusoidal_1d(torch.arange(width, device=device), dim - half)
    grid = torch.cat(
        [rows[:, None, :].expand(height, width, half), cols[None, :, :].expand(height, width, dim - half)], dim=-1
    )
    return grid.reshape(height * width, dim)


# --------------------------------------------------------------------------- attention


def _key_mask(mask: Tensor | None) -> Tensor | None:
    """(B, S) bool "is valid" -> SDPA boolean mask broadcastable to (B, heads, L, S)."""
    return None if mask is None else mask[:, None, None, :]


class SelfAttention(nn.Module):
    def __init__(self, dim: int, heads: int, attn_dropout: float = 0.0, dropout: float = 0.0):
        super().__init__()
        if dim % heads:
            raise ValueError("d_model must be divisible by n_heads")
        self.heads = heads
        self.head_dim = dim // heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.attn_dropout = attn_dropout
        self.drop = nn.Dropout(dropout)

    def _split(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        b, n, _ = x.shape
        qkv = self.qkv(x).view(b, n, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        return qkv[0], qkv[1], qkv[2]

    def _merge(self, out: Tensor) -> Tensor:
        b, _, n, _ = out.shape
        return self.drop(self.proj(out.transpose(1, 2).reshape(b, n, self.heads * self.head_dim)))

    def forward_2d(self, x: Tensor, cos: Tensor, sin: Tensor, key_mask: Tensor | None) -> Tensor:
        q, k, v = self._split(x)
        q, k = apply_rope_2d(q, cos, sin), apply_rope_2d(k, cos, sin)
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=_key_mask(key_mask), dropout_p=self.attn_dropout if self.training else 0.0
        )
        return self._merge(out)

    def forward_causal(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        # Padding only ever follows real tokens, so the causal mask alone keeps every
        # real query away from padding; padded queries are ignored by the loss.
        q, k, v = self._split(x)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.attn_dropout if self.training else 0.0
        )
        return self._merge(out)

    def forward_step(self, x: Tensor, cos: Tensor, sin: Tensor, cache: dict[str, Tensor]) -> Tensor:
        """Incremental decoding for a single new token; ``cache`` holds past keys/values."""
        q, k, v = self._split(x)
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        if "k" in cache:
            k = torch.cat([cache["k"], k], dim=2)
            v = torch.cat([cache["v"], v], dim=2)
        cache["k"], cache["v"] = k, v
        out = F.scaled_dot_product_attention(q, k, v)
        return self._merge(out)


class CrossAttention(nn.Module):
    """Text queries attending over image memory; keys carry absolute 2-D positions."""

    def __init__(self, dim: int, heads: int, attn_dropout: float = 0.0, dropout: float = 0.0):
        super().__init__()
        self.heads = heads
        self.head_dim = dim // heads
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        self.attn_dropout = attn_dropout
        self.drop = nn.Dropout(dropout)

    def _heads(self, x: Tensor) -> Tensor:
        b, n, _ = x.shape
        return x.view(b, n, self.heads, self.head_dim).transpose(1, 2)

    def memory_kv(self, memory: Tensor, memory_pos: Tensor) -> tuple[Tensor, Tensor]:
        return self._heads(self.k(memory + memory_pos)), self._heads(self.v(memory))

    def forward(self, x: Tensor, k: Tensor, v: Tensor, memory_mask: Tensor | None) -> Tensor:
        q = self._heads(self.q(x))
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=_key_mask(memory_mask), dropout_p=self.attn_dropout if self.training else 0.0
        )
        b, _, n, _ = out.shape
        return self.drop(self.proj(out.transpose(1, 2).reshape(b, n, self.heads * self.head_dim)))
