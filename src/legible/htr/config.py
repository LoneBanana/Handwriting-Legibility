"""Typed configuration for the handwriting model, data pipeline, and trainer.

Configs are plain dataclasses loaded from YAML. Any field can be overridden from the
command line with dotted keys, e.g. ``--set model.d_model=512 train.lr=3e-4``.
"""

from __future__ import annotations

import dataclasses
import types
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import yaml


@dataclass
class ModelConfig:
    img_height: int = 64
    in_channels: int = 1
    # Convolutional front end: a stride-2 stem followed by one stage per entry.
    cnn_channels: tuple[int, ...] = (48, 96, 192, 256)
    cnn_depths: tuple[int, ...] = (1, 1, 2, 2)
    # Strides (height, width) of the downsampling conv entering stages 1..N-1.
    # Together with the stem this gives /16 height and /8 width by default.
    cnn_strides: tuple[tuple[int, int], ...] = ((2, 2), (2, 2), (2, 1))
    cnn_kernel: int = 7
    # Initial value of ConvNeXt's per-channel residual scale. 1e-6 (the ImageNet recipe)
    # keeps blocks silent for many steps; 0.1 lets a shallow front end learn strokes early.
    layer_scale_init: float = 0.1
    # Transformer encoder (ViT over CNN feature tokens) and autoregressive decoder.
    d_model: int = 256
    n_heads: int = 4
    enc_layers: int = 6
    enc_conv_kernel: int = 5         # Conformer-style depthwise conv in each encoder block (0 = off)
    dec_layers: int = 4
    mlp_ratio: float = 4.0
    dropout: float = 0.1
    attn_dropout: float = 0.0
    drop_path: float = 0.1
    rope_base: float = 10000.0
    max_decode_len: int = 256
    # Joint CTC / attention objective.
    ctc_weight: float = 0.3
    inter_ctc_weight: float = 0.3    # share of the CTC term given to an extra CTC head on the CNN tokens
    label_smoothing: float = 0.1
    gradient_checkpointing: bool = False

    @property
    def height_reduction(self) -> int:
        r = 2
        for s, _ in self.cnn_strides:
            r *= s
        return r

    @property
    def width_reduction(self) -> int:
        r = 2
        for _, s in self.cnn_strides:
            r *= s
        return r


@dataclass
class DatasetSpec:
    """One training/validation source. ``type`` selects the loader."""

    type: Literal["manifest", "synthetic"] = "manifest"
    path: str | None = None          # manifest (.jsonl) path for type=manifest
    weight: float = 1.0              # relative sampling weight in the training mixture
    name: str | None = None
    # synthetic-only settings
    fonts_dir: str | None = None
    corpus: str | None = None
    size: int = 100_000              # virtual epoch length
    irregularity: float = 0.5        # 0 = neat handwriting fonts, 1 = strongly dysgraphic
    ramp_epochs: int = 0             # curriculum: ramp irregularity up from 20% over this many epochs


@dataclass
class AugmentConfig:
    enabled: bool = True
    strength: float = 0.5            # global multiplier for geometric distortions (0..1)
    p_affine: float = 0.8
    p_baseline_warp: float = 0.5
    p_elastic: float = 0.5
    p_horizontal_jitter: float = 0.5
    p_stroke_width: float = 0.4
    p_ink_fade: float = 0.3
    p_blur: float = 0.3
    p_noise: float = 0.4
    p_paper: float = 0.3
    p_cutout: float = 0.15
    p_jpeg: float = 0.2


@dataclass
class DataConfig:
    train: list[DatasetSpec] = field(default_factory=list)
    val: list[DatasetSpec] = field(default_factory=list)
    max_width: int = 2048            # pixels after height normalization
    min_width: int = 32
    max_text_len: int = 250
    batch_size: int = 32             # lines per GPU per step (upper bound)
    max_pixels: int = 0              # optional pixel budget per batch (height*width*lines); 0 = off
    samples_per_epoch: int = 0       # 0 = sum of dataset sizes
    num_workers: int = 8
    augment: AugmentConfig = field(default_factory=AugmentConfig)


@dataclass
class VocabConfig:
    charset: str | None = None       # None = vocab.DEFAULT_CHARSET
    path: str | None = None          # or load from a saved vocab.json


@dataclass
class TrainConfig:
    output_dir: str = "runs/htr"
    max_steps: int = 200_000
    lr: float = 5e-4
    min_lr: float = 1e-5
    warmup_steps: int = 4_000
    weight_decay: float = 0.05
    betas: tuple[float, float] = (0.9, 0.98)
    grad_clip: float = 1.0
    grad_accum: int = 1
    precision: Literal["auto", "bf16", "fp16", "fp32"] = "auto"
    ema_decay: float = 0.9998
    compile: bool = False
    seed: int = 1337
    log_every: int = 50
    eval_every: int = 2_000
    eval_max_batches: int = 0        # 0 = full validation set
    save_every: int = 2_000
    keep_last: int = 3
    select_metric: Literal["cer_att", "cer_ctc"] = "cer_att"
    init_from: str | None = None     # warm-start weights (fine-tuning)
    resume: str | None = "auto"      # "auto" = latest checkpoint in output_dir
    max_runtime_minutes: float = 0.0  # >0: checkpoint and exit with code 3 before the wall clock
    tensorboard: bool = True


@dataclass
class Config:
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    vocab: VocabConfig = field(default_factory=VocabConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


def _coerce(tp: Any, value: Any) -> Any:
    """Convert YAML values (lists, dicts) into the annotated dataclass field type."""
    origin = typing.get_origin(tp)
    if value is None:
        return None
    if dataclasses.is_dataclass(tp):
        return from_dict(tp, value)
    if origin in (typing.Union, types.UnionType):
        args = [a for a in typing.get_args(tp) if a is not type(None)]
        for arg in args:
            try:
                return _coerce(arg, value)
            except (TypeError, ValueError):
                continue
        return value
    if origin is tuple:
        args = typing.get_args(tp)
        if len(args) == 2 and args[1] is Ellipsis:
            return tuple(_coerce(args[0], v) for v in value)
        return tuple(_coerce(a, v) for a, v in zip(args, value))
    if origin is list:
        (arg,) = typing.get_args(tp)
        return [_coerce(arg, v) for v in value]
    if origin is Literal:
        if value not in typing.get_args(tp):
            raise ValueError(f"{value!r} is not one of {typing.get_args(tp)}")
        return value
    if tp is float and isinstance(value, (int, float)) and not isinstance(value, bool):
        return float(value)
    if tp is float and isinstance(value, str):
        # YAML 1.1 reads exponent floats without a dot ("5e-4") as strings.
        try:
            return float(value)
        except ValueError:
            raise TypeError(f"expected a number, got {value!r}") from None
    if tp is int and isinstance(value, bool):
        raise TypeError("expected int, got bool")
    if tp in (int, str, bool) and not isinstance(value, tp):
        raise TypeError(f"expected {tp.__name__}, got {type(value).__name__}")
    return value


def from_dict(cls: type, data: dict[str, Any]) -> Any:
    if not isinstance(data, dict):
        raise TypeError(f"expected a mapping for {cls.__name__}, got {type(data).__name__}")
    hints = typing.get_type_hints(cls)
    names = {f.name for f in dataclasses.fields(cls)}
    unknown = set(data) - names
    if unknown:
        raise KeyError(f"unknown {cls.__name__} keys: {sorted(unknown)}")
    return cls(**{k: _coerce(hints[k], v) for k, v in data.items()})


def to_dict(cfg: Any) -> dict[str, Any]:
    return dataclasses.asdict(cfg)


def _set_dotted(data: dict[str, Any], key: str, value: Any) -> None:
    """Sets ``a.b.c=value``; numeric parts index into lists (``data.train.0.weight=2``)."""
    node: Any = data
    parts = key.split(".")
    for part in parts[:-1]:
        node = node[int(part)] if isinstance(node, list) else node.setdefault(part, {})
    last = parts[-1]
    if isinstance(node, list):
        node[int(last)] = value
    else:
        node[last] = value


def load_config(path: str | Path | None = None, overrides: list[str] | None = None) -> Config:
    """Load a YAML config (optionally ``base:``-inheriting another) and apply overrides."""
    data: dict[str, Any] = {}
    if path is not None:
        data = _load_yaml_with_base(Path(path))
    for item in overrides or []:
        key, sep, raw = item.partition("=")
        if not sep:
            raise ValueError(f"override must look like key=value, got {item!r}")
        _set_dotted(data, key.strip(), yaml.safe_load(raw))
    return from_dict(Config, data)


def _merge(base: dict[str, Any], top: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in top.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _load_yaml_with_base(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text()) or {}
    base = data.pop("base", None)
    if base:
        data = _merge(_load_yaml_with_base((path.parent / base).resolve()), data)
    return data
