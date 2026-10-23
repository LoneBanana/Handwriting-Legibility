# Training the handwriting model (Phase 1)

The model reads a photo of handwriting and returns text, with a confidence for every
letter. It is built for hard-to-read handwriting, including handwriting affected by
dysgraphia: letters of uneven size, words that drift off the line, irregular spacing,
reversed letters, and crossed-out false starts.

## Architecture: why a CNN-ViT hybrid

```
image (B,1,64,W) ─ ConvNeXt front end ─► (B,256,4,W/8) ─ 1×1 proj ─► tokens ─┬─ ViT/Conformer encoder (2-D RoPE)
                                                                             │    ├─ CTC head (per column)
                                                                             │    └─ Transformer decoder (cross-attention)
                                                                             └─ intermediate CTC head (training only)
```

* **Convolutions first.** Pen strokes, loops, and junctions are local patterns. A pure
  ViT has to learn that locality from data, which is the main reason ViTs train slowly
  on handwriting. A ConvNeXt front end supplies it for free and hands the transformer a
  short 4-row grid of tokens (a 64 px line becomes 4 rows × W/8 columns).
* **Global attention second.** The encoder attends across the whole line, so it can read
  a malformed letter from the word around it and follow a drifting baseline. Axial 2-D
  rotary position embeddings make attention depend on relative offsets, so the model
  handles lines wider than any it was trained on.
* **Two heads, one encoder.** A CTC head (fast, alignment-free, monotonic) and an
  autoregressive decoder (an implicit character language model, good with ambiguous
  letters) are trained jointly: `loss = 0.3·CTC + 0.7·CE`. CTC stabilises early training
  and later rescores the decoder's beam: it must account for every column of ink, which
  suppresses the decoder's classic failure modes (skipped words, repetition loops).
* **Padding-safe.** LayerNorm instead of BatchNorm and an attention key mask mean that a
  line reads identically alone or batched with wider lines (there is a test for this).
* **Fast convergence despite the transformer.** At initialization, global attention
  averages features across the line, and during CTC's early "predict blank everywhere"
  phase it learns to keep doing so, which erases the left-to-right locality CTC needs.
  Two remedies are built in: an **intermediate CTC head** on the CNN tokens
  (`inter_ctc_weight`, InterCTC; Lee & Watanabe 2021) gives the convolutional front end a
  letter-level signal from step one, and a **Conformer-style depthwise convolution** in
  each encoder block (`enc_conv_kernel`) keeps neighbouring columns tied together. In a
  controlled test (random-letter lines, CTC only, 2 encoder layers) the plain hybrid was
  still at 100 % CER after 600 steps; with intermediate CTC it reached 55 % CER at step
  500, ahead of the same CNN with no transformer at all (69 % at step 400). Keep both on
  unless you have a reason not to.

| Config | Parameters | Use |
|---|---:|---|
| `configs/htr/tiny.yaml` | 0.7 M | smoke test; minutes on a laptop, no downloads |
| `configs/htr/base.yaml` | 12 M | one or a few GPUs |
| `configs/htr/large.yaml` | 74 M | multi-node; the production model |
| `configs/htr/finetune_dysgraphia.yaml` | 74 M | stage 2 on strongly irregular writing |

## 1. Data

### Synthetic dysgraphic handwriting (built in)

There is no large, transcribed corpus of dysgraphic handwriting, so the pipeline
synthesises it (`legible/htr/data/synthetic.py`): text is rendered letter by letter from
handwriting fonts, with per-letter jitter in size, height above the baseline, slant, and
spacing; irregular word gaps; occasional mirrored letters (b/d, p/q); and crossed-out
false starts. Stroke tremor, baseline waves, uneven pressure, ruled paper, blur, noise,
and JPEG artefacts are added by `augment.py`. The label is always the *intended* text,
so the model learns to read through the irregularities. `irregularity` (0 to 1) scales
everything; the effective value per line is capped so a patient human could still read
every training line (training on unreadable lines would teach the model to guess).

```bash
python scripts/download_fonts.py            # ~60 OFL handwriting fonts -> assets/fonts/
legible-prepare corpus my_texts/*.txt --out data/corpus/sentences.txt
```

Without `fonts_dir`, handwriting-style system fonts are used; without `corpus`, a
small built-in set of everyday sentences. For real training, use a large text corpus
(a few million sentences, e.g. from Wikipedia or school-level writing) so the decoder
learns spelling, not memorised sentences.

### Real handwriting

Datasets are described by line manifests (`.jsonl`, one `{"image", "text", "writer"}`
per line; see `legible/htr/data/manifest.py`):

```bash
legible-prepare hf Teklia/IAM-line --out data/iam                # IAM lines from Hugging Face
legible-prepare folder path/to/lines --out data/mine/all.jsonl    # images + same-name .txt files
legible-prepare split data/mine/all.jsonl --val 0.1 --test 0.1    # writer-disjoint split
```

Useful sources: IAM (English, many writers), GNHK (handwriting photographed with
phones, closest to how the app is used), CVL, and Bentham. **Check each license**: IAM,
for example, is free for non-commercial research only. Keep validation and test sets
*writer-disjoint* from training; that is the honest measure of reading a stranger's
handwriting.

**Dysgraphic samples.** The most valuable data is real handwriting from people with
dysgraphia. The app collects it with a dedicated, opt-in consent: in Settings, learners
who keep their samples can additionally choose "Also share my corrected samples to
improve Legible for everyone". Export those lines as a manifest:

```bash
legible-export-research --out data/consented          # uses LEGIBLE_DATABASE_URL / LEGIBLE_DATA_DIR
legible-prepare split data/consented/all.jsonl --val 0.1 --test 0.1
```

Only learners whose sharing setting is on at export time are included, and only lines
whose correction pairs one-to-one with a detected line. Writers appear as a keyed hash
(stable across exports, so splits stay writer-disjoint), never a name or email; the
words themselves can still be personal, so handle exports as research data (ethics
review where required). Then enable the `dysgraphia` manifest in
`finetune_dysgraphia.yaml` with a high weight. Data collected outside the app needs the
same informed consent and de-identification.

## 2. Train

```bash
legible-train --config configs/htr/tiny.yaml                             # smoke test
legible-train --config configs/htr/base.yaml --set data.batch_size=48    # any field can be overridden
torchrun --nproc-per-node 8 -m legible.htr.train --config configs/htr/large.yaml
```

On a SLURM cluster:

```bash
sbatch --nodes=4 scripts/slurm/train_htr.sbatch                            # large.yaml
CONFIG=configs/htr/finetune_dysgraphia.yaml sbatch scripts/slurm/train_htr.sbatch
```

The script launches one `torchrun` per node over `srun` (c10d rendezvous). Training is
**preemption-safe**: the script passes the job's remaining time as
`train.max_runtime_minutes`; when it runs out (or on SIGTERM/SIGUSR1) every rank stops
at the same step, rank 0 writes a checkpoint and a `NEEDS_RESUME` marker, and the script
requeues the job, which resumes on the exact batch it stopped at (`train.resume=auto`).

What the trainer does:

* DDP with width-bucketed batches: lines of similar width are batched together, capped
  by `batch_size` lines and `max_pixels` padded pixels, so long lines get smaller
  batches. Every rank gets the same number of batches, so DDP never deadlocks.
* bf16 autocast on Ampere/Hopper (fp16 with loss scaling on older GPUs), TF32 matmuls,
  fused AdamW, optional `torch.compile` and gradient checkpointing.
* Warmup + cosine learning rate, gradient clipping, EMA weights (evaluation and export
  always use the EMA).
* Validation every `eval_every` steps, sharded across ranks: CER/WER for both the CTC
  and the attention decoder. `best.pt` is kept by `select_metric`.
* Outputs in `train.output_dir`: `checkpoints/step_*.pt` (full state, last `keep_last`),
  `best.pt` and `final.pt` (small inference checkpoints), `metrics.jsonl`, TensorBoard
  logs in `tb/`, and the exact `config.json`.

**Scaling notes.** The effective batch is `batch_size × grad_accum × GPUs`. When you
change it by a factor *k*, scale `lr` by about √*k* and keep warmup at a few thousand
steps. `large.yaml` (bf16, 48 lines per GPU) fits 40 GB GPUs; set
`model.gradient_checkpointing=true` for longer lines or smaller GPUs. Data loading is
CPU-heavy (rendering and augmentation): give each GPU 8–12 workers.

**Recommended schedule:** (1) `large.yaml` on synthetic + IAM + GNHK until validation
CER plateaus; (2) `finetune_dysgraphia.yaml` from `best.pt` with stronger
irregularity/augmentation and any dysgraphic data you have.

## 3. Evaluate

```bash
legible-eval --checkpoint runs/htr-large/best.pt --manifest data/iam/test.jsonl --decoder accurate \
  --predictions runs/htr-large/test.predictions.jsonl
```

`fast` is greedy CTC (one pass); `accurate` is a beam search over the decoder,
rescored with CTC. Report CER on writer-disjoint test sets, and separately on
dysgraphic samples if you have them; the two can differ a lot.

## 4. Serve

```bash
export LEGIBLE_HTR_CHECKPOINT=runs/htr-large/best.pt   # LEGIBLE_HTR_DEVICE=cuda|mps|cpu (default auto)
legible-api
legible-transcribe photo.jpg --checkpoint runs/htr-large/best.pt
legible-export --checkpoint runs/htr-large/best.pt --out models/htr-ctc.onnx   # optional
```

The ONNX export (encoder + CTC head, dynamic batch and width) is verified against
PyTorch with ONNX Runtime during export and records which checkpoint it came from. Set
`LEGIBLE_HTR_ONNX=models/htr-ctc.onnx` and the API runs "Quick" reading through ONNX
Runtime, about 3× faster per page on a CPU than PyTorch (an export from a different
checkpoint is ignored). "Careful" reading and personal adapters always use PyTorch.

Page photos are split into lines by `legible/htr/segment.py`. Instead of one projection
profile, it tracks lines across the page strip by strip, so lines that climb, fall, or
wave are still separated, and ruled notebook lines are erased first. Each line is cut
out with a mask, so ascenders and descenders from neighbouring lines are not read.

## 5. Personalization

When a user turns on "Keep my handwriting samples" and corrects a transcription, the
corrected lines are stored with the line images. Once they have corrected
`LEGIBLE_PERSONALIZE_MIN_SAMPLES` lines (default 8), a background job trains a small
**LoRA adapter** (`legible/htr/lora.py`, rank 8, about 1 % of the parameters, on the
attention/MLP layers and the last CNN stages) on those lines. The adapter is installed
**only if it lowers the error rate on that user's held-out lines**, so personalization
can never make transcription worse. At inference the user's adapter is swapped into the
shared model for their request only. Adapters are tied to the base model version:
after you deploy a new base model, adapters retrain as users keep correcting.

## Privacy

* Photos are stored only when the user opts in, and only for their own adapter.
* Settings → "Delete my account" and `DELETE /api/handwriting/samples` remove photos,
  corrections, and adapters from disk.
* Samples reach the *shared* model only through the separate research-sharing consent
  and `legible-export-research` (see "Dysgraphic samples" above). Turning sharing off,
  deleting samples, or deleting the account stops future exports; copies already
  exported for training are governed by your research-data policy.
