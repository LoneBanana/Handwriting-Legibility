"""Trainer for the CNN-ViT handwriting model: one GPU, or many nodes via torchrun/SLURM.

    # single GPU / laptop
    legible-train --config configs/htr/base.yaml

    # one node, 8 GPUs
    torchrun --nproc-per-node 8 -m legible.htr.train --config configs/htr/large.yaml

    # many nodes: scripts/slurm/train_htr.sbatch (torchrun rendezvous over srun)

Features: DDP with width-bucketed batches, bf16/fp16 autocast, gradient accumulation,
warmup + cosine schedule, EMA weights (used for evaluation and export), periodic
distributed validation (CER/WER for both decoding heads), atomic checkpoints with
exact mid-epoch resume, and graceful preemption: on SIGTERM/SIGUSR1 or when
``train.max_runtime_minutes`` is reached, every rank stops at the same step, rank 0
writes a checkpoint plus a ``NEEDS_RESUME`` marker, and the job exits cleanly so the
batch script can requeue it.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import math
import os
import signal
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader

from .checkpoint import (
    inference_payload,
    latest_training_checkpoint,
    load_matching_weights,
    prune_training_checkpoints,
    read_payload,
    save_atomic,
)
from .config import Config, load_config, to_dict
from .data import BucketBatchSampler, Collator, LineDataset, build_mixture
from .decoding import attention_greedy, ctc_greedy
from .device import pick_device
from .metrics import ErrorCounts
from .model import HybridHTR
from .vocab import DEFAULT_CHARSET, Vocab

log = logging.getLogger("legible.train")

RESUME_MARKER = "NEEDS_RESUME"


# ----------------------------------------------------------------------------- runtime


@dataclass
class Runtime:
    rank: int
    world_size: int
    local_rank: int
    device: torch.device

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    @property
    def distributed(self) -> bool:
        return self.world_size > 1


def setup_runtime(device_preference: str = "auto") -> Runtime:
    env = os.environ
    if int(env.get("WORLD_SIZE", "1")) > 1:  # launched by torchrun
        rank, world, local = int(env["RANK"]), int(env["WORLD_SIZE"]), int(env.get("LOCAL_RANK", "0"))
    elif int(env.get("SLURM_NTASKS", "1")) > 1 and "MASTER_ADDR" in env:  # plain srun, one task per GPU
        rank, world, local = int(env["SLURM_PROCID"]), int(env["SLURM_NTASKS"]), int(env.get("SLURM_LOCALID", "0"))
        env.setdefault("MASTER_PORT", "29500")
        env["RANK"], env["WORLD_SIZE"], env["LOCAL_RANK"] = str(rank), str(world), str(local)
    else:
        return Runtime(0, 1, 0, pick_device(device_preference))

    if torch.cuda.is_available():
        torch.cuda.set_device(local)
        device = torch.device("cuda", local)
        backend = "nccl"
    else:
        device = torch.device("cpu")
        backend = "gloo"
    dist.init_process_group(backend, timeout=timedelta(minutes=30))
    return Runtime(rank, world, local, device)


def autocast_dtype(precision: str, device: torch.device) -> torch.dtype | None:
    if precision == "fp32":
        return None
    if device.type == "cuda":
        if precision in ("auto", "bf16") and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16
    if precision == "bf16":
        return torch.bfloat16
    if precision == "fp16" and device.type == "mps":
        return torch.float16
    return None  # CPU/MPS default to full precision


# ----------------------------------------------------------------------------- pieces


class LossModule(nn.Module):
    """Model + loss in one forward, so DDP (and torch.compile) see a single graph."""

    def __init__(self, model: HybridHTR):
        super().__init__()
        self.model = model

    def forward(self, images, widths, tgt_in, tgt_out, ctc_targets, ctc_lengths, ctc_valid):
        out = self.model(images, widths, tgt_in)
        batch = {"tgt_out": tgt_out, "ctc_targets": ctc_targets, "ctc_lengths": ctc_lengths, "ctc_valid": ctc_valid}
        losses = self.model.loss(out, batch)
        zero = losses["loss"].new_zeros(())
        return losses["loss"], losses.get("ce", zero), losses.get("ctc", zero), losses.get("inter_ctc", zero)


class EMA:
    """Exponential moving average of the weights (fp32), with a warmup on the decay."""

    def __init__(self, model: nn.Module, decay: float):
        self.decay = decay
        self.shadow = {k: v.detach().clone().float() for k, v in model.state_dict().items() if v.is_floating_point()}

    @torch.no_grad()
    def update(self, model: nn.Module, step: int) -> None:
        d = min(self.decay, (1.0 + step) / (10.0 + step))
        state = model.state_dict()
        keys = list(self.shadow)
        shadow = [self.shadow[k] for k in keys]
        current = [state[k].detach().float() for k in keys]
        torch._foreach_mul_(shadow, d)
        torch._foreach_add_(shadow, current, alpha=1.0 - d)

    def state_dict(self) -> dict[str, torch.Tensor]:
        return self.shadow

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        for k, v in state.items():
            if k in self.shadow:
                self.shadow[k].copy_(v)

    def full_state(self, model: nn.Module) -> dict[str, torch.Tensor]:
        """EMA floats merged with the model's non-float buffers, loadable into a fresh model."""
        state = {k: v.detach() for k, v in model.state_dict().items()}
        state.update(self.shadow)
        return state


def param_groups(model: nn.Module, weight_decay: float) -> list[dict[str, Any]]:
    """No weight decay on biases, norms, embeddings, and layer-scale vectors."""
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim < 2 or "embed" in name or name.endswith(".gamma"):
            no_decay.append(p)
        else:
            decay.append(p)
    return [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}]


def lr_factor(step: int, warmup: int, total: int, min_ratio: float) -> float:
    if step < warmup:
        return (step + 1) / max(warmup, 1)
    progress = min((step - warmup) / max(total - warmup, 1), 1.0)
    return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))


def _worker_init(_worker_id: int) -> None:
    import cv2

    cv2.setNumThreads(0)  # workers are the parallelism; nested OpenCV threads oversubscribe
    torch.set_num_threads(1)


# ----------------------------------------------------------------------------- trainer


class Trainer:
    def __init__(self, cfg: Config, rt: Runtime):
        self.cfg = cfg
        self.rt = rt
        self.out_dir = Path(cfg.train.output_dir)
        self.vocab = Vocab.load(cfg.vocab.path) if cfg.vocab.path else Vocab(cfg.vocab.charset or DEFAULT_CHARSET)
        self.dtype = autocast_dtype(cfg.train.precision, rt.device)
        self.stop_requested = False
        self.started = time.monotonic()

        torch.manual_seed(cfg.train.seed + rt.rank)
        if rt.device.type == "cuda":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

        self.model = HybridHTR(cfg.model, len(self.vocab)).to(rt.device)
        if cfg.train.init_from:
            load_matching_weights(self.model, cfg.train.init_from)
        self.ema = EMA(self.model, cfg.train.ema_decay)
        self.eval_model = HybridHTR(cfg.model, len(self.vocab)).to(rt.device).eval()

        module: nn.Module = LossModule(self.model)
        self.ddp: DistributedDataParallel | None = None
        if rt.distributed:
            device_ids = [rt.local_rank] if rt.device.type == "cuda" else None
            module = self.ddp = DistributedDataParallel(module, device_ids=device_ids,
                                                        find_unused_parameters=cfg.model.ctc_weight == 0)
        if cfg.train.compile:
            module = torch.compile(module, dynamic=True)
        self.module = module

        fused = rt.device.type == "cuda"
        self.optimizer = torch.optim.AdamW(param_groups(self.model, cfg.train.weight_decay), lr=cfg.train.lr,
                                           betas=cfg.train.betas, fused=fused)
        min_ratio = cfg.train.min_lr / cfg.train.lr
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lambda s: lr_factor(s, cfg.train.warmup_steps, cfg.train.max_steps, min_ratio)
        )
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.dtype == torch.float16 and rt.device.type == "cuda")

        self.step = 0
        self.epoch = 0
        self.batch_in_epoch = 0
        self.best = math.inf
        self._last_eval: tuple[int, dict[str, float]] | None = None
        self._last_save = -1
        self._build_data()
        self.writer = self._tensorboard()

    # ------------------------------------------------------------------ setup

    def _build_data(self) -> None:
        cfg, rt = self.cfg, self.rt
        if not cfg.data.train:
            raise ValueError("config has no data.train datasets")
        train_mix = build_mixture(cfg.data.train, self.vocab, cfg.data.max_text_len, cfg.train.seed)
        self.train_set = LineDataset(train_mix, self.vocab, cfg.model, cfg.data, augment=True, seed=cfg.train.seed)
        self.train_sampler = BucketBatchSampler(
            train_mix, cfg.model.img_height, cfg.data.batch_size, cfg.data.max_pixels,
            cfg.data.samples_per_epoch, rt.rank, rt.world_size, cfg.train.seed, max_width=cfg.data.max_width,
        )
        collate = Collator(cfg.model.width_reduction)
        workers = cfg.data.num_workers
        loader_kw: dict[str, Any] = dict(
            collate_fn=collate, num_workers=workers, pin_memory=rt.device.type == "cuda",
            worker_init_fn=_worker_init if workers else None,
        )
        if workers:
            loader_kw.update(persistent_workers=True, prefetch_factor=4)
        self.train_loader = DataLoader(self.train_set, batch_sampler=self.train_sampler, **loader_kw)

        self.val_loader = None
        if cfg.data.val:
            val_mix = build_mixture(cfg.data.val, self.vocab, cfg.data.max_text_len, cfg.train.seed + 1)
            val_set = LineDataset(val_mix, self.vocab, cfg.model, cfg.data, augment=False, seed=cfg.train.seed + 1)
            val_sampler = BucketBatchSampler(
                val_mix, cfg.model.img_height, cfg.data.batch_size * 2, cfg.data.max_pixels * 2,
                rank=rt.rank, world_size=rt.world_size, seed=cfg.train.seed, shuffle=False,
                max_width=cfg.data.max_width,
            )
            eval_kw = dict(loader_kw, persistent_workers=False) if workers else loader_kw
            if workers:
                eval_kw["num_workers"] = max(1, workers // 2)
            self.val_loader = DataLoader(val_set, batch_sampler=val_sampler, **eval_kw)

    def _tensorboard(self):
        if not (self.rt.is_main and self.cfg.train.tensorboard):
            return None
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError:
            log.info("tensorboard not installed; logging to metrics.jsonl only")
            return None
        return SummaryWriter(self.out_dir / "tb")

    def _install_signal_handlers(self) -> None:
        def request_stop(signum, _frame):
            log.warning("received signal %s: will checkpoint and stop at the next step", signum)
            self.stop_requested = True

        for name in ("SIGTERM", "SIGUSR1"):
            if hasattr(signal, name):
                signal.signal(getattr(signal, name), request_stop)

    # ------------------------------------------------------------------ checkpoints

    def _resume(self) -> None:
        resume = self.cfg.train.resume
        if not resume:
            return
        path = latest_training_checkpoint(self.out_dir) if resume == "auto" else Path(resume)
        if path is None:
            return
        payload = read_payload(path)
        self.model.load_state_dict(payload["state_dict"])
        self.ema.load_state_dict(payload["ema"])
        self.optimizer.load_state_dict(payload["optimizer"])
        self.scheduler.load_state_dict(payload["scheduler"])
        if payload.get("scaler"):
            self.scaler.load_state_dict(payload["scaler"])
        progress = payload["progress"]
        self.step, self.epoch = progress["step"], progress["epoch"]
        self.batch_in_epoch, self.best = progress["batch_in_epoch"], progress["best"]
        log.info("resumed from %s at step %d (epoch %d, batch %d)", path, self.step, self.epoch, self.batch_in_epoch)

    def _meta(self, metrics: dict[str, float] | None = None) -> dict[str, Any]:
        return {
            "version": f"{self.out_dir.name}-s{self.step}",
            "step": self.step,
            "metrics": metrics or {},
            "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "config": to_dict(self.cfg),
        }

    def save_training(self) -> None:
        if self._last_save == self.step:
            return
        self._last_save = self.step
        if self.rt.is_main:
            payload = inference_payload(self.cfg.model, self.vocab, self.model.state_dict(), self._meta())
            payload.update(
                ema=self.ema.full_state(self.model),
                optimizer=self.optimizer.state_dict(),
                scheduler=self.scheduler.state_dict(),
                scaler=self.scaler.state_dict() if self.scaler.is_enabled() else None,
                progress={"step": self.step, "epoch": self.epoch, "batch_in_epoch": self.batch_in_epoch,
                          "best": self.best},
            )
            save_atomic(payload, self.out_dir / "checkpoints" / f"step_{self.step:08d}.pt")
            prune_training_checkpoints(self.out_dir, self.cfg.train.keep_last)
        self._barrier()

    def save_inference(self, name: str, metrics: dict[str, float] | None = None) -> None:
        if self.rt.is_main:
            payload = inference_payload(self.cfg.model, self.vocab, self.ema.full_state(self.model), self._meta(metrics))
            save_atomic(payload, self.out_dir / name)
            self.vocab.save(self.out_dir / "vocab.json")

    # ------------------------------------------------------------------ loop

    @property
    def _reduce_device(self) -> torch.device:
        """Where metric counters live: the collective's device when distributed (NCCL needs
        CUDA tensors), otherwise the CPU (MPS has no float64)."""
        return self.rt.device if self.rt.distributed else torch.device("cpu")

    def _barrier(self) -> None:
        if self.rt.distributed:
            dist.barrier()

    def _should_stop(self) -> bool:
        limit = self.cfg.train.max_runtime_minutes
        local = self.stop_requested or (limit > 0 and time.monotonic() - self.started > limit * 60)
        if not self.rt.distributed:
            return local
        flag = torch.tensor([1.0 if local else 0.0], device=self.rt.device)
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)  # every rank stops at the same step
        return bool(flag.item())

    def _autocast(self):
        if self.dtype is None:
            return contextlib.nullcontext()
        return torch.autocast(self.rt.device.type, dtype=self.dtype)

    def _to_device(self, batch: dict) -> dict:
        nb = self.rt.device.type == "cuda"
        return {k: v.to(self.rt.device, non_blocking=nb) if torch.is_tensor(v) else v for k, v in batch.items()}

    def _forward(self, batch: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        with self._autocast():
            return self.module(batch["images"], batch["widths"], batch["tgt_in"], batch["tgt_out"],
                               batch["ctc_targets"], batch["ctc_lengths"], batch["ctc_valid"])

    def fit(self) -> None:
        cfg, rt = self.cfg, self.rt
        if rt.is_main:
            self.out_dir.mkdir(parents=True, exist_ok=True)
            (self.out_dir / RESUME_MARKER).unlink(missing_ok=True)
            (self.out_dir / "config.json").write_text(json.dumps(to_dict(cfg), indent=2))
            log.info("model: %.1fM parameters, device %s x%d, autocast %s", self.model.num_parameters() / 1e6,
                     rt.device, rt.world_size, self.dtype)
        self._resume()
        self._install_signal_handlers()

        if len(self.train_sampler) == 0:
            raise ValueError("not enough training data to give every rank a batch; add data or lower batch_size")
        accum = max(cfg.train.grad_accum, 1)
        micro = 0
        last_batch: dict | None = None
        window = {"loss": 0.0, "ce": 0.0, "ctc": 0.0, "inter_ctc": 0.0, "lines": 0, "pixels": 0, "real_pixels": 0,
                  "n": 0}
        window_start = time.monotonic()
        self.module.train()

        while self.step < cfg.train.max_steps:
            self.train_sampler.set_epoch(self.epoch, skip=self.batch_in_epoch)
            for batch in self.train_loader:
                self.batch_in_epoch += 1
                skip_update = batch is None
                if skip_update:  # every sample in the batch failed to load; keep ranks in lockstep
                    if last_batch is None:
                        continue
                    batch = last_batch
                batch = self._to_device(batch)
                last_batch = batch
                sync = (micro + 1) % accum == 0
                # Skip the gradient all-reduce on accumulation micro-steps.
                ctx = self.ddp.no_sync() if (self.ddp is not None and not sync) else contextlib.nullcontext()
                with ctx:
                    loss, ce, ctc, inter = self._forward(batch)
                    scale = 0.0 if skip_update else 1.0 / accum
                    self.scaler.scale(loss * scale).backward()
                micro += 1
                window["loss"] += float(loss.detach()) / accum
                window["ce"] += float(ce.detach()) / accum
                window["ctc"] += float(ctc.detach()) / accum
                window["inter_ctc"] += float(inter.detach()) / accum
                window["lines"] += batch["images"].shape[0]
                window["pixels"] += batch["images"].numel()
                window["real_pixels"] += int(batch["widths"].sum()) * batch["images"].shape[2]
                if not sync:
                    continue

                self.scaler.unscale_(self.optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.train.grad_clip)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                self.scheduler.step()
                self.ema.update(self.model, self.step)
                self.step += 1
                micro = 0
                window["n"] += 1

                if self.step % cfg.train.log_every == 0:
                    self._log_train(window, float(grad_norm), time.monotonic() - window_start)
                    window = dict.fromkeys(window, 0)
                    window_start = time.monotonic()
                if self.val_loader is not None and self.step % cfg.train.eval_every == 0:
                    self._evaluate_and_select()
                    self.module.train()
                if self.step % cfg.train.save_every == 0:
                    self.save_training()
                if self._should_stop():
                    self.save_training()
                    if rt.is_main:
                        (self.out_dir / RESUME_MARKER).write_text(f"step {self.step}\n")
                        log.warning("stopped at step %d; checkpoint saved, requeue to resume", self.step)
                    return
                if self.step >= cfg.train.max_steps:
                    break
            else:
                self.epoch += 1
                self.batch_in_epoch = 0
                continue
            break

        self.save_training()
        metrics = self._evaluate_and_select() if self.val_loader is not None else None
        self.save_inference("final.pt", metrics)
        if rt.is_main:
            log.info("finished %d steps; best %s = %.4f", self.step, cfg.train.select_metric, self.best)

    def _log_train(self, window: dict, grad_norm: float, seconds: float) -> None:
        n = max(window["n"], 1)
        totals = torch.tensor([window["lines"], window["pixels"], window["real_pixels"]], dtype=torch.float64,
                              device=self._reduce_device)
        if self.rt.distributed:
            dist.all_reduce(totals)
        if not self.rt.is_main:
            return
        record = {
            "step": self.step,
            "loss": window["loss"] / n,
            "ce": window["ce"] / n,
            "ctc": window["ctc"] / n,
            "inter_ctc": window["inter_ctc"] / n,
            "lr": self.scheduler.get_last_lr()[0],
            "grad_norm": grad_norm,
            "lines_per_s": float(totals[0]) / max(seconds, 1e-6),
            "padding_efficiency": float(totals[2] / totals[1].clamp(min=1)),
            "epoch": self.epoch,
        }
        if self.rt.device.type == "cuda":
            record["max_mem_gb"] = torch.cuda.max_memory_allocated(self.rt.device) / 1e9
        log.info("step %(step)d loss %(loss).4f ce %(ce).4f ctc %(ctc).4f (cnn %(inter_ctc).4f) lr %(lr).2e "
                 "|g| %(grad_norm).2f %(lines_per_s).0f lines/s pad-eff %(padding_efficiency).2f", record)
        self._write_metrics("train", record)

    def _write_metrics(self, split: str, record: dict[str, Any]) -> None:
        with (self.out_dir / "metrics.jsonl").open("a") as f:
            f.write(json.dumps({"split": split, **record}) + "\n")
        if self.writer is not None:
            for key, value in record.items():
                if key != "step" and isinstance(value, (int, float)):
                    self.writer.add_scalar(f"{split}/{key}", value, self.step)

    # ------------------------------------------------------------------ validation

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        """CER/WER of the EMA weights with CTC-greedy and attention-greedy decoding."""
        self.eval_model.load_state_dict(self.ema.full_state(self.model))
        att, ctc = ErrorCounts(), ErrorCounts()
        examples: list[tuple[str, str]] = []
        for i, batch in enumerate(self.val_loader):
            if batch is None:
                continue
            if self.cfg.train.eval_max_batches and i >= self.cfg.train.eval_max_batches:
                break
            images, widths = batch["images"].to(self.rt.device), batch["widths"].to(self.rt.device)
            max_len = int(batch["ctc_lengths"].max()) + 16
            with self._autocast():
                enc = self.eval_model.encode(images, widths)
                hyps_ctc = ctc_greedy(self.eval_model.ctc_logits(enc), enc.col_lengths)
                hyps_att = attention_greedy(self.eval_model, enc, max_len=max_len)
            for ref, h_att, h_ctc in zip(batch["texts"], hyps_att, hyps_ctc):
                pred_att = self.vocab.decode(h_att.ids)
                att.add(ref, pred_att)
                ctc.add(ref, self.vocab.decode(h_ctc.ids))
                if len(examples) < 8:
                    examples.append((ref, pred_att))
        totals = torch.tensor(att.as_tuple() + ctc.as_tuple(), dtype=torch.float64, device=self._reduce_device)
        if self.rt.distributed:
            dist.all_reduce(totals)
        values = [int(v) for v in totals.tolist()]
        att, ctc = ErrorCounts(*values[:5]), ErrorCounts(*values[5:])
        metrics = {"cer_att": att.cer, "wer_att": att.wer, "cer_ctc": ctc.cer, "wer_ctc": ctc.wer,
                   "lines": att.lines}
        if self.rt.is_main:
            log.info("eval step %d: CER att %.4f ctc %.4f | WER att %.4f ctc %.4f (%d lines)", self.step,
                     metrics["cer_att"], metrics["cer_ctc"], metrics["wer_att"], metrics["wer_ctc"], att.lines)
            for ref, hyp in examples[:3]:
                log.info("  ref: %s\n  hyp: %s", ref, hyp)
            self._write_metrics("val", {"step": self.step, **metrics})
        return metrics

    def _evaluate_and_select(self) -> dict[str, float]:
        if self._last_eval is not None and self._last_eval[0] == self.step:
            return self._last_eval[1]
        metrics = self.evaluate()
        self._last_eval = (self.step, metrics)
        if metrics["lines"] == 0:
            if self.rt.is_main:
                log.warning("validation produced no lines (every sample failed to load); best.pt unchanged")
            self._barrier()
            return metrics
        score = metrics[self.cfg.train.select_metric]
        if score < self.best:
            self.best = score
            self.save_inference("best.pt", metrics)
            if self.rt.is_main:
                log.info("new best %s %.4f at step %d", self.cfg.train.select_metric, score, self.step)
        self._barrier()
        return metrics


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Train the Legible handwriting model.")
    parser.add_argument("--config", required=True, help="YAML config (see configs/htr/)")
    parser.add_argument("--set", nargs="*", action="extend", default=[], metavar="KEY=VALUE",
                        help="override config values, e.g. --set train.lr=3e-4 data.batch_size=48")
    parser.add_argument("--device", default="auto", help="single-process device (auto|cpu|cuda|mps)")
    args = parser.parse_args(argv)

    cfg = load_config(args.config, args.set)
    rt = setup_runtime(args.device)
    logging.basicConfig(
        level=logging.INFO if rt.is_main else logging.WARNING,
        format=f"%(asctime)s [rank {rt.rank}] %(levelname)s %(message)s",
        stream=sys.stdout,
    )
    try:
        Trainer(cfg, rt).fit()
    finally:
        if rt.distributed:
            dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
