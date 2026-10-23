import math

import pytest
import torch

from legible.htr.config import load_config
from legible.htr.data.dataset import Collator
from legible.htr.decoding import attention_greedy, beam_search, ctc_greedy, ctc_sequence_logprob
from legible.htr.vocab import Vocab, normalize_text

from .helpers import render_line, tiny_model


def _batch(vocab, texts, seed=0):
    from legible.htr.data.image import resize_to_height

    samples = []
    for i, text in enumerate(texts):
        image = resize_to_height(render_line(text, seed + i), 64, 1024)
        samples.append({"image": image, "text": text, "ids": vocab.encode(text)})
    return Collator(8)(samples)


def test_vocab_roundtrip_and_normalization():
    vocab = Vocab()
    text = "Hello “world” — 3×4 = 12"
    ids = vocab.encode(text)
    assert vocab.decode(ids) == normalize_text(text) == 'Hello "world" - 3×4 = 12'
    assert vocab.decode(ids + [Vocab.eos_id] + ids) == normalize_text(text)
    assert normalize_text("  a \t b  \r\n  c ") == "a b\nc"


def test_config_overrides_support_nested_lists(tmp_path):
    base = tmp_path / "base.yaml"
    base.write_text("data:\n  train:\n    - {type: synthetic, size: 10}\nmodel:\n  d_model: 64\n  n_heads: 4\n")
    child = tmp_path / "child.yaml"
    child.write_text("base: base.yaml\nmodel:\n  enc_layers: 3\n")
    cfg = load_config(child, ["data.train.0.size=99", "model.cnn_channels=[8,16,24,32]", "train.lr=1e-3"])
    assert cfg.data.train[0].size == 99 and cfg.model.d_model == 64 and cfg.model.enc_layers == 3
    assert cfg.model.cnn_channels == (8, 16, 24, 32) and cfg.train.lr == pytest.approx(1e-3)
    with pytest.raises(KeyError):
        load_config(child, ["model.not_a_field=1"])


def test_forward_shapes_and_finite_loss():
    model, vocab = tiny_model()
    batch = _batch(vocab, ["hello there", "a much longer line of text to read"])
    model.train()
    out = model(batch["images"], batch["widths"], batch["tgt_in"])
    b, _, h, w = batch["images"].shape
    assert out["ctc_logits"].shape == (b, w // model.cfg.width_reduction, len(vocab))
    assert out["dec_logits"].shape == (*batch["tgt_in"].shape, len(vocab))
    losses = model.loss(out, batch)
    assert all(math.isfinite(float(v.detach())) for v in losses.values())
    losses["loss"].backward()
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)


def test_padding_does_not_change_a_lines_encoding():
    """A short line must encode the same alone as when batched next to a much wider one."""
    model, vocab = tiny_model()
    batch = _batch(vocab, ["short", "this line is a great deal wider than the first one"])
    with torch.no_grad():
        both = model.encode(batch["images"], batch["widths"])
        w0 = int(batch["widths"][0])
        cols = int(both.col_lengths[0])
        padded = -(-w0 // 32) * 32
        alone = model.encode(batch["images"][:1, :, :, :padded], batch["widths"][:1])
        h = both.grid[0]
        a = both.memory[0].view(h, -1, both.memory.shape[-1])[:, : cols - 1]
        b = alone.memory[0].view(h, -1, alone.memory.shape[-1])[:, : cols - 1]
    torch.testing.assert_close(a, b, atol=1e-4, rtol=1e-4)


def test_ctc_greedy_collapses_repeats_and_blanks():
    vocab = Vocab()
    seq = [vocab.stoi["a"], vocab.stoi["a"], Vocab.blank_id, vocab.stoi["a"], vocab.stoi["b"], Vocab.blank_id]
    logits = torch.full((1, len(seq), len(vocab)), -10.0)
    for t, k in enumerate(seq):
        logits[0, t, k] = 10.0
    (hyp,) = ctc_greedy(logits, torch.tensor([len(seq)]))
    assert vocab.decode(hyp.ids) == "aab"
    assert len(hyp.char_probs) == 3 and all(p > 0.99 for p in hyp.char_probs)


def test_ctc_sequence_logprob_prefers_the_aligned_text():
    vocab = Vocab()
    seq = [vocab.stoi["c"], Vocab.blank_id, vocab.stoi["a"], vocab.stoi["t"]]
    logits = torch.full((len(seq), len(vocab)), -8.0)
    for t, k in enumerate(seq):
        logits[t, k] = 8.0
    scores = ctc_sequence_logprob(logits.log_softmax(-1), len(seq), [vocab.encode("cat"), vocab.encode("bat")])
    assert scores[0] > scores[1]


def test_beam_of_one_matches_greedy_decoding():
    model, vocab = tiny_model(seed=3)
    batch = _batch(vocab, ["greedy", "beam"])
    with torch.no_grad():
        enc = model.encode(batch["images"], batch["widths"])
        greedy = attention_greedy(model, enc, max_len=12)
        beam = beam_search(model, enc, beam_size=1, max_len=12, ctc_weight=0.0, length_penalty=0.0)
    assert [h.ids for h in greedy] == [h.ids for h in beam]


def test_beam_search_returns_alternatives_and_confidences():
    model, vocab = tiny_model(seed=1)
    batch = _batch(vocab, ["alternatives"])
    with torch.no_grad():
        enc = model.encode(batch["images"], batch["widths"])
        (hyp,) = beam_search(model, enc, beam_size=4, max_len=10, vocab=vocab)
    assert len(hyp.char_probs) == len(hyp.ids)
    assert 0.0 <= hyp.confidence <= 1.0
    assert hyp.alternatives and isinstance(hyp.alternatives[0][0], str)
