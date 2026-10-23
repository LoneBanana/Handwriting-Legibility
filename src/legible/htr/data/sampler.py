"""Distributed, width-bucketed batch sampling over a weighted mixture of sources.

Each epoch every rank builds the *same* global batch list from ``(seed, epoch)``:

1. draw ``samples_per_epoch`` keys from the sources in proportion to their weights;
2. sort keys by estimated width inside large random chunks (so batches hold similarly
   sized lines with little padding, while the order stays random);
3. pack batches up to ``batch_size`` lines and ``max_pixels`` padded pixels;
4. shuffle the batches and deal them round-robin to ranks, trimmed so every rank gets
   the same count (an uneven count would deadlock DDP's gradient all-reduce).

Because the list is a pure function of ``(seed, epoch)``, resuming mid-epoch is just
``skip`` = number of batches this rank already consumed.
"""

from __future__ import annotations

from typing import Iterator

import numpy as np
from torch.utils.data import Sampler

from .dataset import Mixture, SampleKey


class BucketBatchSampler(Sampler[list[SampleKey]]):
    def __init__(self, mixture: Mixture, height: int, batch_size: int, max_pixels: int = 0,
                 samples_per_epoch: int = 0, rank: int = 0, world_size: int = 1, seed: int = 0,
                 shuffle: bool = True, chunk_batches: int = 64, max_width: int = 2048):
        self.mixture = mixture
        self.height = height
        self.batch_size = batch_size
        self.max_pixels = max_pixels
        self.samples_per_epoch = samples_per_epoch or sum(len(s) for s in mixture.sources)
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.shuffle = shuffle
        self.chunk = max(batch_size * chunk_batches, batch_size)
        self.max_width = max_width
        self.epoch = 0
        self.skip = 0
        self._cache: tuple[int, list[list[SampleKey]]] | None = None

    def set_epoch(self, epoch: int, skip: int = 0) -> None:
        self.epoch = epoch
        self.skip = skip

    def _draw(self, rng: np.random.Generator) -> list[tuple[int, int]]:
        weights = np.asarray(self.mixture.weights, dtype=np.float64)
        weights = weights / weights.sum()
        if not self.shuffle:  # evaluation: every line of every source exactly once, in order
            return [(s, i) for s, source in enumerate(self.mixture.sources) for i in range(len(source))]
        counts = np.floor(weights * self.samples_per_epoch).astype(int)
        counts[np.argmax(weights)] += self.samples_per_epoch - counts.sum()
        keys: list[tuple[int, int]] = []
        for source_id, (source, count) in enumerate(zip(self.mixture.sources, counts)):
            n = len(source)
            if n == 0 or count <= 0:
                continue
            reps = -(-count // n)
            order = np.concatenate([rng.permutation(n) for _ in range(reps)])[:count]
            keys.extend((source_id, int(i)) for i in order)
        rng.shuffle(keys)
        return keys

    def _batches(self) -> list[list[SampleKey]]:
        if self._cache is None or self._cache[0] != self.epoch:
            self._cache = (self.epoch, self._build())
        return self._cache[1]

    def _build(self) -> list[list[SampleKey]]:
        rng = np.random.default_rng([self.seed, self.epoch])
        drawn = self._draw(rng)
        epoch = self.epoch
        widths = [
            min(self.mixture.sources[s].width_estimate(i, epoch, self.height), self.max_width)
            for s, i in drawn
        ]
        batches: list[list[SampleKey]] = []
        for start in range(0, len(drawn), self.chunk):
            chunk = sorted(range(start, min(start + self.chunk, len(drawn))), key=widths.__getitem__)
            batch: list[SampleKey] = []
            batch_width = 0
            for j in chunk:
                width = max(widths[j], 1)
                new_width = max(batch_width, width)
                too_many = len(batch) >= self.batch_size
                too_big = self.max_pixels > 0 and batch and new_width * self.height * (len(batch) + 1) > self.max_pixels
                if batch and (too_many or too_big):
                    batches.append(batch)
                    batch, new_width = [], width
                s, i = drawn[j]
                batch.append((s, i, epoch))
                batch_width = new_width
            if batch:
                batches.append(batch)
        if self.shuffle:
            order = rng.permutation(len(batches))
            batches = [batches[k] for k in order]
        per_rank = len(batches) // self.world_size if self.shuffle else -(-len(batches) // self.world_size)
        mine = batches[self.rank :: self.world_size][:per_rank]
        return mine

    def __iter__(self) -> Iterator[list[SampleKey]]:
        batches = self._batches()
        yield from batches[self.skip :]

    def __len__(self) -> int:
        return max(len(self._batches()) - self.skip, 0)
