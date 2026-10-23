"""Training/validation datasets and batch collation.

Every sample is addressed by a key ``(source, index, epoch)`` produced by
:class:`~.sampler.BucketBatchSampler`. Randomness (augmentation, synthetic text and
rendering) is derived from that key alone, so a sample is identical no matter which
worker or rank produces it, and runs are reproducible and resumable mid-epoch.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from ..config import DataConfig, DatasetSpec, ModelConfig
from ..vocab import Vocab, normalize_text
from .augment import Augmenter
from .corpus import TextCorpus
from .image import flatten_background, load_gray, resize_to_height, to_ink, trim
from .manifest import read_manifest
from .synthetic import HandwritingRenderer, discover_fonts

log = logging.getLogger(__name__)

SampleKey = tuple[int, int, int]  # (source, index, epoch)


def sample_rng(seed: int, key: SampleKey) -> np.random.Generator:
    return np.random.default_rng([seed, key[0], key[1], key[2]])


def finish_line(ink: np.ndarray, rng: np.random.Generator | None, augmenter: Augmenter | None,
                height: int, max_width: int, min_width: int) -> np.ndarray:
    """Shared tail of the pipeline: trim, augment at 2x the model height, resize."""
    ink = trim(ink)
    if augmenter is not None and rng is not None:
        work_height = 2 * height
        if ink.shape[0] > work_height:
            ink = resize_to_height(ink, work_height, max_width * 4, min_width)
        ink = augmenter(ink, rng)
    return resize_to_height(ink, height, max_width, min_width)


class LineSource(Dataset):
    """A source of (ink map, text) lines with a cheap length estimate for batching."""

    name: str = "source"

    def __len__(self) -> int:
        raise NotImplementedError

    def width_estimate(self, index: int, epoch: int, height: int) -> int:
        """Approximate width in pixels at the model height (used only for batching)."""
        raise NotImplementedError

    def load(self, index: int, epoch: int, rng: np.random.Generator) -> tuple[np.ndarray, str]:
        raise NotImplementedError


class ManifestSource(LineSource):
    def __init__(self, path: str | Path, vocab: Vocab, max_text_len: int, name: str | None = None,
                 min_coverage: float = 0.98):
        self.path = Path(path)
        self.root = self.path.parent
        self.name = name or self.path.stem
        records = read_manifest(self.path)
        kept = []
        for record in records:
            text = normalize_text(record.text)
            if not text or len(text) > max_text_len or vocab.coverage(text) < min_coverage:
                continue
            record.text = text
            kept.append(record)
        if len(kept) < len(records):
            log.info("%s: kept %d of %d lines (empty, too long, or unsupported characters)",
                     self.name, len(kept), len(records))
        if not kept:
            raise ValueError(f"manifest {self.path} has no usable lines")
        self.records = kept

    def __len__(self) -> int:
        return len(self.records)

    def width_estimate(self, index: int, epoch: int, height: int) -> int:
        record = self.records[index]
        if record.width and record.height:
            return int(record.width * height / max(record.height, 1))
        return int(len(record.text) * height * 0.55) + height

    def load(self, index: int, epoch: int, rng: np.random.Generator) -> tuple[np.ndarray, str]:
        record = self.records[index]
        # Same lighting normalisation as inference (segment.prepare_page), so phone photos
        # with shadows look alike at training and serving time.
        return to_ink(flatten_background(load_gray(record.resolve(self.root)))), record.text


class SyntheticSource(LineSource):
    def __init__(self, spec: DatasetSpec, seed: int, max_text_len: int):
        self.name = spec.name or "synthetic"
        self.size = spec.size
        self.seed = seed
        self.max_chars = min(max_text_len, 72)
        self.corpus = TextCorpus.load(spec.corpus)
        self.irregularity = spec.irregularity
        self.ramp_epochs = spec.ramp_epochs
        self.renderer = HandwritingRenderer(discover_fonts(spec.fonts_dir), irregularity=spec.irregularity)

    def irregularity_at(self, epoch: int) -> float:
        """Curriculum: neat-ish writing first, the full irregularity after ``ramp_epochs``."""
        if self.ramp_epochs <= 0:
            return self.irregularity
        t = min(1.0, (epoch + 1) / self.ramp_epochs)
        return self.irregularity * (0.2 + 0.8 * t)

    def __len__(self) -> int:
        return self.size

    def _text(self, index: int, epoch: int) -> str:
        # Text comes from its own RNG stream so the batcher can predict widths cheaply.
        rng = np.random.default_rng([self.seed, 7919, index, epoch])
        return self.corpus.sample(rng, max_chars=self.max_chars)

    def width_estimate(self, index: int, epoch: int, height: int) -> int:
        return int(len(self._text(index, epoch)) * height * 0.6) + height

    def load(self, index: int, epoch: int, rng: np.random.Generator) -> tuple[np.ndarray, str]:
        return self.renderer.render(self._text(index, epoch), rng, irregularity=self.irregularity_at(epoch))


@dataclass
class Mixture:
    """The weighted set of sources one split draws from."""

    sources: list[LineSource]
    weights: list[float]


class LineDataset(Dataset):
    """Maps sample keys to model-ready tensors for one split (train or validation)."""

    def __init__(self, mixture: Mixture, vocab: Vocab, model_cfg: ModelConfig, data_cfg: DataConfig,
                 augment: bool, seed: int):
        self.mixture = mixture
        self.vocab = vocab
        self.height = model_cfg.img_height
        self.max_width = data_cfg.max_width
        self.min_width = data_cfg.min_width
        self.max_text_len = data_cfg.max_text_len
        self.augmenter = Augmenter(data_cfg.augment) if augment and data_cfg.augment.enabled else None
        self.seed = seed

    def __getitem__(self, key: SampleKey) -> dict | None:
        source_id, index, epoch = key
        source = self.mixture.sources[source_id]
        rng = sample_rng(self.seed, key)
        try:
            ink, text = source.load(index, epoch, rng)
            text = normalize_text(text)
            if not text or len(text) > self.max_text_len:
                return None
            image = finish_line(ink, rng if self.augmenter else None, self.augmenter,
                                self.height, self.max_width, self.min_width)
        except Exception as exc:  # noqa: BLE001 - one bad image must not kill a multi-day run
            log.warning("skipping sample %s from %s: %s", key, source.name, exc)
            return None
        return {"image": image, "text": text, "ids": self.vocab.encode(text, normalize=False)}


def build_mixture(specs: list[DatasetSpec], vocab: Vocab, max_text_len: int, seed: int) -> Mixture:
    sources: list[LineSource] = []
    weights: list[float] = []
    for i, spec in enumerate(specs):
        if spec.type == "manifest":
            if not spec.path:
                raise ValueError("manifest datasets need a `path`")
            sources.append(ManifestSource(spec.path, vocab, max_text_len, name=spec.name))
        elif spec.type == "synthetic":
            sources.append(SyntheticSource(spec, seed + 1000 * i, max_text_len))
        else:
            raise ValueError(f"unknown dataset type {spec.type!r}")
        weights.append(spec.weight)
    return Mixture(sources, weights)


class Collator:
    """Pads a list of samples into a batch for :meth:`HybridHTR.forward` and :meth:`loss`.

    Widths are padded up to ``width_multiple`` (a multiple of the model's width
    reduction) so the number of distinct tensor shapes, and kernel autotuning, stays small.
    """

    def __init__(self, width_reduction: int, width_multiple: int = 32):
        self.width_reduction = width_reduction
        self.width_multiple = max(width_multiple, width_reduction)

    def __call__(self, samples: list[dict | None]) -> dict | None:
        samples = [s for s in samples if s is not None]
        if not samples:
            return None
        widths = [s["image"].shape[1] for s in samples]
        height = samples[0]["image"].shape[0]
        m = self.width_multiple
        padded_w = -(-max(widths) // m) * m
        images = torch.zeros(len(samples), 1, height, padded_w)
        for i, s in enumerate(samples):
            images[i, 0, :, : widths[i]] = torch.from_numpy(s["image"])

        lengths = [len(s["ids"]) for s in samples]
        t = max(lengths) + 1
        tgt_in = torch.full((len(samples), t), Vocab.pad_id, dtype=torch.long)
        tgt_out = torch.full((len(samples), t), Vocab.pad_id, dtype=torch.long)
        ctc_targets = torch.zeros(len(samples), max(max(lengths), 1), dtype=torch.long)
        ctc_valid = torch.zeros(len(samples), dtype=torch.bool)
        for i, s in enumerate(samples):
            ids = s["ids"]
            tgt_in[i, 0] = Vocab.bos_id
            tgt_in[i, 1 : len(ids) + 1] = torch.tensor(ids, dtype=torch.long)
            tgt_out[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            tgt_out[i, len(ids)] = Vocab.eos_id
            ctc_targets[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            # CTC needs one column per character plus one per repeated pair, and reads a
            # single left-to-right line; multi-line samples train the decoder only.
            repeats = sum(a == b for a, b in zip(ids, ids[1:]))
            columns = -(-widths[i] // self.width_reduction)
            ctc_valid[i] = "\n" not in s["text"] and columns >= len(ids) + repeats and len(ids) > 0
        return {
            "images": images,
            "widths": torch.tensor(widths, dtype=torch.long),
            "tgt_in": tgt_in,
            "tgt_out": tgt_out,
            "ctc_targets": ctc_targets,
            "ctc_lengths": torch.tensor(lengths, dtype=torch.long),
            "ctc_valid": ctc_valid,
            "texts": [s["text"] for s in samples],
        }
