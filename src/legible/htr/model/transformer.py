"""ViT-style encoder over CNN feature tokens and an autoregressive character decoder."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from .layers import CrossAttention, DropPath, Mlp, Rope1D, Rope2D, SelfAttention, sinusoidal_1d


class GridConv(nn.Module):
    """Depthwise convolution over the token grid (the Conformer convolution module).

    At initialization global attention averages features from the whole line, which
    blurs exactly the left-to-right locality the CTC head needs to align letters; models
    without it sit on the "all blank" plateau far longer. This branch keeps each token
    tied to its neighbours while attention learns where long-range context helps.
    """

    def __init__(self, dim: int, kernel: int, dropout: float):
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, (3, kernel), padding=(1, kernel // 2), groups=dim)
        self.pw = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: Tensor, height: int, width: int, token_mask: Tensor | None) -> Tensor:
        b, n, d = x.shape
        if token_mask is not None:  # padding must not leak into real columns through the kernel
            x = x * token_mask[..., None].to(x.dtype)
        y = self.dw(x.transpose(1, 2).reshape(b, d, height, width)).flatten(2).transpose(1, 2)
        return self.drop(self.pw(F.gelu(y)))


class EncoderBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float, dropout: float, attn_dropout: float, drop_path: float,
                 conv_kernel: int = 0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SelfAttention(dim, heads, attn_dropout, dropout)
        self.conv_norm = nn.LayerNorm(dim) if conv_kernel else None
        self.conv = GridConv(dim, conv_kernel, dropout) if conv_kernel else None
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), dropout)
        self.drop_path = DropPath(drop_path)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor, key_mask: Tensor | None, height: int = 0,
                width: int = 0) -> Tensor:
        x = x + self.drop_path(self.attn.forward_2d(self.norm1(x), cos, sin, key_mask))
        if self.conv is not None:
            x = x + self.drop_path(self.conv(self.conv_norm(x), height, width, key_mask))
        return x + self.drop_path(self.mlp(self.norm2(x)))


class Encoder(nn.Module):
    """Global self-attention over the (rows x columns) token grid with 2-D RoPE.

    Global attention lets the model use context across the whole line, e.g. reading a
    malformed letter from the word around it, or following a drifting baseline.
    """

    def __init__(self, dim: int, depth: int, heads: int, mlp_ratio: float, dropout: float,
                 attn_dropout: float, drop_path: float, rope_base: float, grad_ckpt: bool = False,
                 conv_kernel: int = 0):
        super().__init__()
        rates = [drop_path * i / max(depth - 1, 1) for i in range(depth)]
        self.blocks = nn.ModuleList(
            EncoderBlock(dim, heads, mlp_ratio, dropout, attn_dropout, r, conv_kernel) for r in rates
        )
        self.norm = nn.LayerNorm(dim)
        self.rope = Rope2D(dim // heads, rope_base)
        self.grad_ckpt = grad_ckpt

    def forward(self, x: Tensor, height: int, width: int, key_mask: Tensor | None) -> Tensor:
        cos, sin = self.rope.tables(height, width, x.device)
        for blk in self.blocks:
            if self.grad_ckpt and self.training:
                x = checkpoint(blk, x, cos, sin, key_mask, height, width, use_reentrant=False)
            else:
                x = blk(x, cos, sin, key_mask, height, width)
        return self.norm(x)


class DecoderBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float, dropout: float, attn_dropout: float, drop_path: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = SelfAttention(dim, heads, attn_dropout, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.cross_attn = CrossAttention(dim, heads, attn_dropout, dropout)
        self.norm3 = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, int(dim * mlp_ratio), dropout)
        self.drop_path = DropPath(drop_path)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor, mem_k: Tensor, mem_v: Tensor,
                memory_mask: Tensor | None) -> Tensor:
        x = x + self.drop_path(self.self_attn.forward_causal(self.norm1(x), cos, sin))
        x = x + self.drop_path(self.cross_attn(self.norm2(x), mem_k, mem_v, memory_mask))
        return x + self.drop_path(self.mlp(self.norm3(x)))

    def step(self, x: Tensor, cos: Tensor, sin: Tensor, cache: dict[str, Tensor], mem_k: Tensor,
             mem_v: Tensor, memory_mask: Tensor | None) -> Tensor:
        x = x + self.self_attn.forward_step(self.norm1(x), cos, sin, cache)
        x = x + self.cross_attn(self.norm2(x), mem_k, mem_v, memory_mask)
        return x + self.mlp(self.norm3(x))


@dataclass
class DecoderState:
    """Everything needed to extend hypotheses one token at a time."""

    mem_kv: list[tuple[Tensor, Tensor]]
    memory_mask: Tensor | None
    caches: list[dict[str, Tensor]]
    step: int = 0

    def reorder(self, index: Tensor) -> "DecoderState":
        """Select/duplicate batch rows (beam search bookkeeping)."""
        return DecoderState(
            mem_kv=[(k.index_select(0, index), v.index_select(0, index)) for k, v in self.mem_kv],
            memory_mask=None if self.memory_mask is None else self.memory_mask.index_select(0, index),
            caches=[{name: t.index_select(0, index) for name, t in c.items()} for c in self.caches],
            step=self.step,
        )


class Decoder(nn.Module):
    """Character-level transformer decoder (implicit language model over the line)."""

    def __init__(self, vocab_size: int, dim: int, depth: int, heads: int, mlp_ratio: float, dropout: float,
                 attn_dropout: float, drop_path: float, rope_base: float):
        super().__init__()
        self.dim = dim
        self.embed = nn.Embedding(vocab_size, dim)
        rates = [drop_path * i / max(depth - 1, 1) for i in range(depth)]
        self.blocks = nn.ModuleList(
            DecoderBlock(dim, heads, mlp_ratio, dropout, attn_dropout, r) for r in rates
        )
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size, bias=False)
        self.head.weight = self.embed.weight  # weight tying
        self.rope = Rope1D(dim // heads, rope_base)
        self.drop = nn.Dropout(dropout)
        nn.init.normal_(self.embed.weight, std=dim ** -0.5)

    def _embed(self, tokens: Tensor, offset: int) -> Tensor:
        positions = torch.arange(offset, offset + tokens.shape[1], device=tokens.device)
        x = self.embed(tokens) * self.dim ** 0.5 + sinusoidal_1d(positions, self.dim).to(self.embed.weight.dtype)
        return self.drop(x)

    def forward(self, tokens: Tensor, memory: Tensor, memory_pos: Tensor, memory_mask: Tensor | None) -> Tensor:
        """Teacher-forced logits for ``tokens`` (B, T) -> (B, T, V)."""
        x = self._embed(tokens, 0)
        cos, sin = self.rope.tables(torch.arange(tokens.shape[1], device=tokens.device))
        for blk in self.blocks:
            k, v = blk.cross_attn.memory_kv(memory, memory_pos)
            x = blk(x, cos, sin, k, v, memory_mask)
        return self.head(self.norm(x))

    def init_state(self, memory: Tensor, memory_pos: Tensor, memory_mask: Tensor | None) -> DecoderState:
        return DecoderState(
            mem_kv=[blk.cross_attn.memory_kv(memory, memory_pos) for blk in self.blocks],
            memory_mask=memory_mask,
            caches=[{} for _ in self.blocks],
        )

    def step(self, last_tokens: Tensor, state: DecoderState) -> Tensor:
        """Log-probabilities of the next token given the previous one, (B,) -> (B, V)."""
        x = self._embed(last_tokens[:, None], state.step)
        cos, sin = self.rope.tables(torch.tensor([state.step], device=last_tokens.device))
        for blk, (k, v), cache in zip(self.blocks, state.mem_kv, state.caches):
            x = blk.step(x, cos, sin, cache, k, v, state.memory_mask)
        state.step += 1
        return self.head(self.norm(x[:, 0])).float().log_softmax(-1)
