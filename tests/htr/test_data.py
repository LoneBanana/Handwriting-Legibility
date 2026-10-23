import numpy as np
import pytest

from legible.htr.config import AugmentConfig, DataConfig, DatasetSpec, ModelConfig
from legible.htr.data import BucketBatchSampler, Collator, LineDataset, Record, build_mixture, read_manifest, write_manifest
from legible.htr.data.augment import Augmenter
from legible.htr.data.corpus import TextCorpus
from legible.htr.vocab import Vocab

from .helpers import render_line, renderer


def test_corpus_samples_fit_the_length_limit():
    corpus = TextCorpus.builtin()
    rng = np.random.default_rng(0)
    for _ in range(200):
        text = corpus.sample(rng, max_chars=40)
        assert 1 <= len(text) <= 40 and text == text.strip()


def test_renderer_produces_ink_and_a_drawable_label():
    r = renderer(irregularity=0.9)
    for seed in range(10):
        ink, label = r.render("Quick brown fox, 12:30 pm!", np.random.default_rng(seed))
        assert ink.dtype == np.float32 and 0.0 <= ink.min() and ink.max() <= 1.0
        assert ink.shape[0] > 10 and ink.shape[1] > ink.shape[0]
        assert label and set(label) <= set("Quick brown fox, 12:30 pm!")


def test_rendering_is_deterministic_per_seed():
    a = render_line("same seed same line", seed=5)
    b = render_line("same seed same line", seed=5)
    np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("strength", [0.0, 0.5, 1.0])
def test_augmenter_keeps_range_and_is_reproducible(strength):
    ink = render_line("augment me please")
    aug = Augmenter(AugmentConfig(strength=strength))
    out1 = aug(ink, np.random.default_rng(11))
    out2 = aug(ink, np.random.default_rng(11))
    np.testing.assert_array_equal(out1, out2)
    assert out1.dtype == np.float32 and out1.min() >= 0.0 and out1.max() <= 1.0


def test_manifest_roundtrip_and_dataset(tmp_path):
    import cv2

    images = tmp_path / "lines"
    images.mkdir()
    records = []
    for i, text in enumerate(["first line", "second line here", "third"]):
        ink = render_line(text, seed=i)
        cv2.imwrite(str(images / f"{i}.png"), (255 * (1 - ink)).astype(np.uint8))
        records.append(Record(image=f"lines/{i}.png", text=text, writer=f"w{i % 2}"))
    path = tmp_path / "train.jsonl"
    assert write_manifest(path, records) == 3
    assert [r.text for r in read_manifest(path)] == ["first line", "second line here", "third"]

    vocab = Vocab()
    data_cfg = DataConfig(train=[DatasetSpec(type="manifest", path=str(path))])
    mixture = build_mixture(data_cfg.train, vocab, data_cfg.max_text_len, seed=0)
    dataset = LineDataset(mixture, vocab, ModelConfig(), data_cfg, augment=True, seed=0)
    a, b = dataset[(0, 1, 0)], dataset[(0, 1, 0)]
    np.testing.assert_array_equal(a["image"], b["image"])  # same key, same augmented sample
    assert a["image"].shape[0] == 64 and a["text"] == "second line here"
    assert not np.array_equal(a["image"], dataset[(0, 1, 1)]["image"])  # new epoch, new augmentation

    batch = Collator(8)([dataset[(0, i, 0)] for i in range(3)])
    assert batch["images"].shape[0] == 3 and batch["images"].shape[3] % 32 == 0
    assert batch["tgt_in"][:, 0].eq(Vocab.bos_id).all()
    for i, text in enumerate(batch["texts"]):
        assert int(batch["tgt_out"][i, len(text)]) == Vocab.eos_id
    assert batch["ctc_valid"].all()


def _synthetic_mixture(size=200):
    vocab = Vocab()
    spec = DatasetSpec(type="synthetic", size=size)
    return build_mixture([spec], vocab, 250, seed=0)


def test_sampler_shards_are_disjoint_equal_and_resumable():
    mixture = _synthetic_mixture()
    samplers = [BucketBatchSampler(mixture, 64, batch_size=8, rank=r, world_size=3, seed=1) for r in range(3)]
    batches = [list(s) for s in samplers]
    assert len({len(b) for b in batches}) == 1  # every rank gets the same number of batches
    keys = [k for rank in batches for batch in rank for k in batch]
    assert len(keys) == len(set(keys))
    again = BucketBatchSampler(mixture, 64, batch_size=8, rank=1, world_size=3, seed=1)
    again.set_epoch(0, skip=2)
    assert list(again) == batches[1][2:]
    samplers[0].set_epoch(1)
    assert list(samplers[0]) != batches[0]


def test_sampler_respects_pixel_budget():
    mixture = _synthetic_mixture()
    sampler = BucketBatchSampler(mixture, 64, batch_size=32, max_pixels=64 * 1500 * 4, seed=0)
    source = mixture.sources[0]
    for batch in sampler:
        widths = [min(source.width_estimate(i, e, 64), 2048) for _, i, e in batch]
        assert len(batch) == 1 or max(widths) * 64 * len(batch) <= 64 * 1500 * 4


def test_eval_sampler_covers_every_line_once():
    mixture = _synthetic_mixture(size=50)
    seen = []
    for rank in range(4):
        sampler = BucketBatchSampler(mixture, 64, batch_size=6, rank=rank, world_size=4, shuffle=False)
        seen += [i for batch in sampler for _, i, _ in batch]
    assert sorted(seen) == list(range(50))


def test_irregularity_curriculum_ramps_up():
    vocab = Vocab()
    spec = DatasetSpec(type="synthetic", size=10, irregularity=0.8, ramp_epochs=4)
    source = build_mixture([spec], vocab, 250, seed=0).sources[0]
    values = [source.irregularity_at(e) for e in range(6)]
    assert values[0] < values[1] < values[2] < values[3] == pytest.approx(0.8) == values[5]
