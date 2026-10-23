"""Convolutional front end: turns a line image into a grid of local stroke features.

Convolutions supply the locality bias that a pure ViT has to learn from data (the
main reason ViTs are slow to train on handwriting): pen strokes, loops, and junctions
are local patterns, so early layers stay convolutional and only the resulting
low-resolution grid is handed to global self-attention.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..config import ModelConfig
from .layers import DropPath, LayerNorm2d


class ConvNeXtBlock(nn.Module):
    """Depthwise large-kernel conv + inverted-bottleneck MLP with layer scale."""

    def __init__(self, dim: int, kernel: int = 7, mlp_ratio: float = 4.0, drop_path: float = 0.0,
                 layer_scale_init: float = 1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel, padding=kernel // 2, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pw1 = nn.Linear(dim, int(dim * mlp_ratio))
        self.pw2 = nn.Linear(int(dim * mlp_ratio), dim)
        self.gamma = nn.Parameter(layer_scale_init * torch.ones(dim))
        self.drop_path = DropPath(drop_path)

    def forward(self, x: Tensor) -> Tensor:
        y = self.dwconv(x).permute(0, 2, 3, 1)
        y = self.pw2(F.gelu(self.pw1(self.norm(y))))
        y = (self.gamma * y).permute(0, 3, 1, 2)
        return x + self.drop_path(y)


class ConvFrontEnd(nn.Module):
    """Stride-2 conv stem followed by ConvNeXt stages with strided downsampling.

    Default geometry for a 64 px line: (1, 64, W) -> (C, 4, W/8). Width is reduced
    less than height so each character still spans several tokens, which the CTC
    head needs to emit repeated letters ("ll", "ss").
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        chans, depths, strides = cfg.cnn_channels, cfg.cnn_depths, cfg.cnn_strides
        if not (len(chans) == len(depths) == len(strides) + 1):
            raise ValueError("cnn_channels and cnn_depths need one entry per stage; cnn_strides one fewer")
        c0 = chans[0]
        self.stem = nn.Sequential(
            nn.Conv2d(cfg.in_channels, c0, 3, stride=2, padding=1),
            LayerNorm2d(c0),
            nn.GELU(),
            nn.Conv2d(c0, c0, 3, padding=1),
            LayerNorm2d(c0),
            nn.GELU(),
        )
        total_blocks = sum(depths)
        rates = [cfg.drop_path * i / max(total_blocks - 1, 1) for i in range(total_blocks)]
        stages: list[nn.Module] = []
        block = 0
        for i, (c, d) in enumerate(zip(chans, depths)):
            layers: list[nn.Module] = []
            if i > 0:
                layers += [LayerNorm2d(chans[i - 1]), nn.Conv2d(chans[i - 1], c, 3, stride=strides[i - 1], padding=1)]
            for _ in range(d):
                layers.append(ConvNeXtBlock(c, cfg.cnn_kernel, drop_path=rates[block],
                                            layer_scale_init=cfg.layer_scale_init))
                block += 1
            stages.append(nn.Sequential(*layers))
        self.stages = nn.Sequential(*stages)
        self.out_channels = chans[-1]

    def forward(self, x: Tensor) -> Tensor:
        return self.stages(self.stem(x))
