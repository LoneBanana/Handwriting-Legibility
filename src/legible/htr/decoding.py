"""Decoding strategies: CTC greedy, attention greedy, and attention beam search with
CTC rescoring. All return per-character confidences, which drive the UI's
"uncertain letter" highlighting and the legibility score."""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import Tensor

from .model.htr import Encoded, HybridHTR
from .vocab import Vocab


@dataclass
class Hypothesis:
    ids: list[int]
    char_probs: list[float]
    score: float = 0.0                      # total log-probability used for ranking
    alternatives: list[tuple[str, float]] = field(default_factory=list)  # n-best (text, score)

    @property
    def confidence(self) -> float:
        if not self.char_probs:
            return 1.0
        return float(torch.tensor(self.char_probs).log().mean().exp())


# ------------------------------------------------------------------------------ CTC


def ctc_greedy(logits: Tensor, lengths: Tensor) -> list[Hypothesis]:
    """Best path decoding: argmax per column, collapse repeats, drop blanks."""
    probs = logits.float().softmax(-1)
    # BOS/EOS share the output layer with the decoder but are never CTC targets.
    probs[..., Vocab.bos_id] = 0.0
    probs[..., Vocab.eos_id] = 0.0
    best_p, best = probs.max(-1)
    results = []
    for b in range(logits.shape[0]):
        n = int(lengths[b])
        ids: list[int] = []
        conf: list[float] = []
        prev = Vocab.blank_id
        for t in range(n):
            k = int(best[b, t])
            if k != prev and k != Vocab.blank_id:
                ids.append(k)
                conf.append(float(best_p[b, t]))
            elif k == prev and k != Vocab.blank_id and conf:
                conf[-1] = max(conf[-1], float(best_p[b, t]))
            prev = k
        total = float(best_p[b, :n].clamp_min(1e-12).log().sum())
        results.append(Hypothesis(ids, conf, total))
    return results


def ctc_sequence_logprob(log_probs: Tensor, input_length: int, hyps: list[list[int]]) -> Tensor:
    """log P(hyp | image) under the CTC head for several hypotheses of one line.

    ``log_probs`` is (T, V) for a single line. Returns a tensor of shape (len(hyps),).
    """
    k = len(hyps)
    lp = log_probs[:input_length].float().unsqueeze(1).expand(-1, k, -1)  # (T, K, V)
    if lp.device.type == "mps":  # CTC has no MPS kernel
        lp = lp.cpu()
    dev = lp.device
    lengths = torch.tensor([len(h) for h in hyps], dtype=torch.long, device=dev)
    flat = torch.tensor([t for h in hyps for t in h], dtype=torch.long, device=dev)
    nll = F.ctc_loss(
        lp, flat, torch.full((k,), input_length, dtype=torch.long, device=dev), lengths,
        blank=Vocab.blank_id, reduction="none", zero_infinity=False,
    )
    return -nll.cpu()


# ------------------------------------------------------------------------------ attention


@torch.no_grad()
def attention_greedy(model: HybridHTR, enc: Encoded, max_len: int | None = None) -> list[Hypothesis]:
    max_len = max_len or model.cfg.max_decode_len
    b = enc.memory.shape[0]
    device = enc.memory.device
    state = model.decoder.init_state(enc.memory, enc.memory_pos, enc.memory_mask)
    last = torch.full((b,), Vocab.bos_id, dtype=torch.long, device=device)
    done = torch.zeros(b, dtype=torch.bool, device=device)
    out_ids = [[] for _ in range(b)]
    out_p = [[] for _ in range(b)]
    scores = torch.zeros(b, device=device)
    for _ in range(max_len):
        logp = model.decoder.step(last, state)
        lp, tok = logp.max(-1)
        scores = scores + torch.where(done, torch.zeros_like(lp), lp)
        for i in range(b):
            if done[i]:
                continue
            t = int(tok[i])
            if t == Vocab.eos_id:
                done[i] = True
            else:
                out_ids[i].append(t)
                out_p[i].append(float(lp[i].exp()))
        if bool(done.all()):
            break
        last = tok
    return [Hypothesis(out_ids[i], out_p[i], float(scores[i])) for i in range(b)]


@torch.no_grad()
def beam_search(
    model: HybridHTR,
    enc: Encoded,
    beam_size: int = 8,
    max_len: int | None = None,
    length_penalty: float = 1.0,
    ctc_weight: float | None = None,
    ctc_logits: Tensor | None = None,
    vocab: Vocab | None = None,
) -> list[Hypothesis]:
    """Attention beam search, then rescoring of each line's n-best list with the CTC head.

    Final score: ``(1-λ)·logP_att/len^α + λ·logP_ctc/len^α``. CTC rescoring suppresses
    the classic attention failure modes (dropped words, repeated loops) because CTC
    must account for every column of ink.
    """
    max_len = max_len or model.cfg.max_decode_len
    lam = model.cfg.ctc_weight if ctc_weight is None else ctc_weight
    if lam > 0 and ctc_logits is None:
        ctc_logits = model.ctc_logits(enc)
    b = enc.memory.shape[0]
    k = beam_size
    device = enc.memory.device
    v = model.vocab_size

    expand = torch.arange(b, device=device).repeat_interleave(k)
    state = model.decoder.init_state(enc.memory, enc.memory_pos, enc.memory_mask).reorder(expand)
    last = torch.full((b * k,), Vocab.bos_id, dtype=torch.long, device=device)
    beam_scores = torch.full((b, k), float("-inf"), device=device)
    beam_scores[:, 0] = 0.0  # all beams start identical; keep one alive initially
    tokens: list[list[list[int]]] = [[[] for _ in range(k)] for _ in range(b)]
    probs: list[list[list[float]]] = [[[] for _ in range(k)] for _ in range(b)]
    finished: list[list[tuple[float, list[int], list[float]]]] = [[] for _ in range(b)]

    for step in range(max_len):
        logp = model.decoder.step(last, state).view(b, k, v)
        cand = (beam_scores[:, :, None] + logp).view(b, k * v)
        top_scores, top_idx = cand.topk(2 * k, dim=-1)
        next_beam = (torch.arange(b, device=device) * k)[:, None].repeat(1, k)  # dead slots stay in-line
        next_tok = torch.zeros(b, k, dtype=torch.long, device=device)
        next_scores = torch.full((b, k), float("-inf"), device=device)
        new_tokens = [[[] for _ in range(k)] for _ in range(b)]
        new_probs = [[[] for _ in range(k)] for _ in range(b)]
        for i in range(b):
            slot = 0
            for score, idx in zip(top_scores[i].tolist(), top_idx[i].tolist()):
                if score == float("-inf"):
                    break
                src, tok = divmod(idx, v)
                p = float(logp[i, src, tok].exp())
                if tok == Vocab.eos_id:
                    if len(finished[i]) < 2 * k:
                        finished[i].append((score, tokens[i][src], probs[i][src]))
                    continue
                if slot < k:
                    next_beam[i, slot] = i * k + src
                    next_tok[i, slot] = tok
                    next_scores[i, slot] = score
                    new_tokens[i][slot] = tokens[i][src] + [tok]
                    new_probs[i][slot] = probs[i][src] + [p]
                    slot += 1
            # Stop expanding a line once its best finished hypothesis cannot be beaten.
            if finished[i] and len(finished[i]) >= k:
                best_done = max(s / _lp(len(t) + 1, length_penalty) for s, t, _ in finished[i])
                best_open = float(next_scores[i].max()) / _lp(max_len, length_penalty)
                if best_open < best_done:
                    next_scores[i] = float("-inf")
        tokens, probs, beam_scores = new_tokens, new_probs, next_scores
        if bool((beam_scores == float("-inf")).all()):
            break
        state = state.reorder(next_beam.view(-1))
        last = next_tok.view(-1)

    results = []
    lengths = enc.col_lengths
    for i in range(b):
        pool = list(finished[i])
        if not pool:  # nothing ended: fall back to open beams
            pool = [(float(beam_scores[i, j]), tokens[i][j], probs[i][j]) for j in range(k)
                    if beam_scores[i, j] > float("-inf")]
        pool.sort(key=lambda x: x[0] / _lp(len(x[1]) + 1, length_penalty), reverse=True)
        pool = pool[: max(k, 1)]
        att = torch.tensor([s for s, _, _ in pool])
        norms = torch.tensor([_lp(len(t) + 1, length_penalty) for _, t, _ in pool])
        final = att / norms
        if lam > 0 and ctc_logits is not None:
            seqs = [t for _, t, _ in pool]
            ctc_lp = ctc_sequence_logprob(ctc_logits[i].float().log_softmax(-1), int(lengths[i]), seqs)
            if torch.isfinite(ctc_lp).any():  # all -inf means no alignment fits; keep attention order
                ctc_lp = torch.nan_to_num(ctc_lp, neginf=-1e4)
                final = (1 - lam) * att / norms + lam * ctc_lp / norms
        order = final.argsort(descending=True).tolist()
        best = order[0]
        _, ids, ps = pool[best]
        alts = []
        if vocab is not None:
            alts = [(vocab.decode(pool[j][1]), float(final[j])) for j in order[:5]]
        results.append(Hypothesis(list(ids), list(ps), float(final[best]), alts))
    return results


def _lp(length: int, alpha: float) -> float:
    """GNMT length penalty."""
    return ((5.0 + length) / 6.0) ** alpha


def decode_batch(model: HybridHTR, images: Tensor, widths: Tensor, strategy: str = "beam",
                 beam_size: int = 8, vocab: Vocab | None = None) -> list[Hypothesis]:
    """Encode once and decode with ``strategy`` in {"ctc", "greedy", "beam"}."""
    enc = model.encode(images, widths)
    if strategy == "ctc":
        return ctc_greedy(model.ctc_logits(enc), enc.col_lengths)
    if strategy == "greedy":
        return attention_greedy(model, enc)
    if strategy == "beam":
        return beam_search(model, enc, beam_size=beam_size, vocab=vocab)
    raise ValueError(f"unknown decoding strategy {strategy!r}")
