"""CNN-ViT hybrid handwriting recognizer with joint CTC / attention decoding.

    image (B,1,H,W)
      └─ ConvFrontEnd ──────────────► features (B,C,H/16,W/8)      local stroke shapes
           └─ 1x1 proj + LayerNorm ─► tokens (B,H'·W',D)            "tokenize" the feature grid
                └─ Encoder (2-D RoPE) ► memory                       global attention over the line
                     ├─ row attention-pool ─► CTC head (B,W',V)      fast, alignment-free decoding
                     └─ Decoder (cross-attn) ─► (B,T,V)              context-aware, LM-like decoding

The CTC branch forces a monotonic left-to-right alignment, which stabilizes training
and lets beam-search hypotheses from the attention decoder be rescored; the attention
decoder adds an implicit character language model that helps with letters that are
ambiguous on their own (common in dysgraphic writing).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..config import ModelConfig
from ..vocab import Vocab
from .cnn import ConvFrontEnd
from .layers import sinusoidal_2d
from .transformer import Decoder, Encoder


@dataclass
class Encoded:
    memory: Tensor          # (B, H'*W', D) encoder output, row-major over the grid
    memory_pos: Tensor      # (H'*W', D) absolute 2-D positions added to cross-attention keys
    memory_mask: Tensor     # (B, H'*W') True where the token covers real (unpadded) image
    grid: tuple[int, int]   # (H', W')
    col_lengths: Tensor     # (B,) number of valid columns per line (CTC input lengths)
    tokens: Tensor | None = None  # (B, H'*W', D) CNN tokens before the encoder (intermediate CTC)


class ColumnHead(nn.Module):
    """Per-column character logits: attention-pool each column of the grid over its rows."""

    def __init__(self, dim: int, vocab_size: int):
        super().__init__()
        self.row_pool = nn.Linear(dim, 1)
        self.norm = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, vocab_size)

    def forward(self, tokens: Tensor, grid: tuple[int, int]) -> Tensor:
        h, w = grid
        x = tokens.view(tokens.shape[0], h, w, -1)
        weights = self.row_pool(x).softmax(dim=1)                  # (B, H', W', 1)
        return self.out(self.norm((weights * x).sum(dim=1)))       # (B, W', V)


class HybridHTR(nn.Module):
    def __init__(self, cfg: ModelConfig, vocab_size: int):
        super().__init__()
        self.cfg = cfg
        self.vocab_size = vocab_size
        self.cnn = ConvFrontEnd(cfg)
        self.proj = nn.Conv2d(self.cnn.out_channels, cfg.d_model, 1)
        self.proj_norm = nn.LayerNorm(cfg.d_model)
        self.encoder = Encoder(
            cfg.d_model, cfg.enc_layers, cfg.n_heads, cfg.mlp_ratio, cfg.dropout, cfg.attn_dropout,
            cfg.drop_path, cfg.rope_base, cfg.gradient_checkpointing, cfg.enc_conv_kernel,
        )
        self.row_pool = nn.Linear(cfg.d_model, 1)
        self.ctc_norm = nn.LayerNorm(cfg.d_model)
        self.ctc_head = nn.Linear(cfg.d_model, vocab_size)
        # Intermediate CTC on the CNN tokens (training only): gives the convolutional
        # front end a direct letter-level signal from the first step, so the encoder is
        # handed localized letter features instead of having to discover them.
        self.inter_ctc = ColumnHead(cfg.d_model, vocab_size) if cfg.inter_ctc_weight > 0 else None
        self.decoder = Decoder(
            vocab_size, cfg.d_model, cfg.dec_layers, cfg.n_heads, cfg.mlp_ratio, cfg.dropout, cfg.attn_dropout,
            cfg.drop_path, cfg.rope_base,
        )
        self.apply(self._init_weights)
        # The output head is tied to the embedding, so re-initialize the shared matrix last.
        nn.init.normal_(self.decoder.embed.weight, std=cfg.d_model ** -0.5)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    # ------------------------------------------------------------------ encoding

    def valid_columns(self, widths: Tensor) -> Tensor:
        r = self.cfg.width_reduction
        return torch.div(widths + r - 1, r, rounding_mode="floor")

    def encode(self, images: Tensor, widths: Tensor) -> Encoded:
        """``images`` (B,1,H,W) with ink=1/background=0, right-padded; ``widths`` (B,) true widths."""
        feats = self.proj(self.cnn(images))                       # (B, D, H', W')
        b, d, h, w = feats.shape
        tokens = self.proj_norm(feats.flatten(2).transpose(1, 2))  # (B, H'*W', D)
        col_lengths = self.valid_columns(widths).clamp(1, w)
        col_mask = torch.arange(w, device=images.device)[None, :] < col_lengths[:, None]  # (B, W')
        memory_mask = col_mask[:, None, :].expand(b, h, w).reshape(b, h * w)
        memory = self.encoder(tokens, h, w, memory_mask)
        memory_pos = sinusoidal_2d(h, w, self.cfg.d_model, images.device).to(memory.dtype)
        return Encoded(memory, memory_pos, memory_mask, (h, w), col_lengths, tokens if self.training else None)

    def ctc_logits(self, enc: Encoded) -> Tensor:
        """Per-column logits (B, W', V): attention-pool each column over its rows."""
        h, w = enc.grid
        x = enc.memory.view(enc.memory.shape[0], h, w, -1)
        weights = self.row_pool(x).softmax(dim=1)                  # (B, H', W', 1)
        cols = (weights * x).sum(dim=1)                            # (B, W', D)
        return self.ctc_head(self.ctc_norm(cols))

    def decoder_logits(self, enc: Encoded, tgt_in: Tensor) -> Tensor:
        return self.decoder(tgt_in, enc.memory, enc.memory_pos, enc.memory_mask)

    def forward(self, images: Tensor, widths: Tensor, tgt_in: Tensor | None = None) -> dict[str, Tensor | Encoded]:
        enc = self.encode(images, widths)
        out: dict[str, Tensor | Encoded] = {"encoded": enc, "ctc_logits": self.ctc_logits(enc)}
        if self.inter_ctc is not None and enc.tokens is not None:
            out["inter_ctc_logits"] = self.inter_ctc(enc.tokens, enc.grid)
        if tgt_in is not None:
            out["dec_logits"] = self.decoder_logits(enc, tgt_in)
        return out

    # ------------------------------------------------------------------ loss

    def loss(self, out: dict, batch: dict, vocab: Vocab | None = None) -> dict[str, Tensor]:
        """Joint objective ``w * CTC + (1 - w) * cross-entropy`` (CTC skipped for multi-line samples).

        With intermediate CTC, the CTC term is ``(1 - a) * CTC_final + a * CTC_cnn``.
        """
        cfg = self.cfg
        enc: Encoded = out["encoded"]
        losses: dict[str, Tensor] = {}

        dec_logits = out["dec_logits"]
        ce = F.cross_entropy(
            dec_logits.float().reshape(-1, dec_logits.shape[-1]),
            batch["tgt_out"].reshape(-1),
            ignore_index=Vocab.pad_id,
            label_smoothing=cfg.label_smoothing,
        )
        losses["ce"] = ce

        if cfg.ctc_weight > 0:
            ctc = self._ctc(out["ctc_logits"], enc, batch, ce.device)
            losses["ctc"] = ctc
            if "inter_ctc_logits" in out:
                inter = self._ctc(out["inter_ctc_logits"], enc, batch, ce.device)
                losses["inter_ctc"] = inter
                ctc = (1 - cfg.inter_ctc_weight) * ctc + cfg.inter_ctc_weight * inter
            losses["loss"] = cfg.ctc_weight * ctc + (1 - cfg.ctc_weight) * ce
        else:
            losses["loss"] = ce
        return losses

    @staticmethod
    def _ctc(logits: Tensor, enc: Encoded, batch: dict, device: torch.device) -> Tensor:
        """Mean per-character CTC loss over the samples where CTC applies."""
        log_probs = logits.float().log_softmax(-1).transpose(0, 1)  # (T, B, V)
        targets, target_lengths = batch["ctc_targets"], batch["ctc_lengths"]
        input_lengths = enc.col_lengths
        if log_probs.device.type == "mps":  # CTC has no MPS kernel
            log_probs, targets = log_probs.cpu(), targets.cpu()
            input_lengths, target_lengths = input_lengths.cpu(), target_lengths.cpu()
        per_sample = F.ctc_loss(
            log_probs, targets, input_lengths, target_lengths,
            blank=Vocab.blank_id, reduction="none", zero_infinity=True,
        ).to(device)
        per_sample = per_sample / batch["ctc_lengths"].clamp(min=1).to(per_sample)
        valid = batch["ctc_valid"].to(per_sample)
        return (per_sample * valid).sum() / valid.sum().clamp(min=1)
