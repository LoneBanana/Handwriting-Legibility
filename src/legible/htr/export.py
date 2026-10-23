"""Exports the encoder + CTC head to ONNX for lightweight serving (CPU, edge, mobile).

    legible-export --checkpoint runs/htr-large/best.pt --out models/htr-ctc.onnx

The exported graph takes ``images`` (N, 1, H, W) ink maps and ``widths`` (N,), with
dynamic batch size and width, and returns per-column log-probabilities plus the valid
column count of each line, ready for greedy or beam CTC decoding (the app's "fast"
mode). The autoregressive decoder stays in PyTorch. The export is verified against
PyTorch with ONNX Runtime at a width different from the one used for tracing.

The checkpoint's model version is stored in the file's metadata
(``legible_model_version``), so a server never pairs an export with a different model.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .checkpoint import load_model

log = logging.getLogger(__name__)


class EncoderCTC(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor, widths: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        enc = self.model.encode(images, widths)
        return self.model.ctc_logits(enc).float().log_softmax(-1), enc.col_lengths


VERSION_KEY = "legible_model_version"


def export_onnx(checkpoint: str | Path, out: str | Path, opset: int = 18) -> Path:
    model, _, meta = load_model(checkpoint, "cpu")
    wrapper = EncoderCTC(model).eval()
    height = model.cfg.img_height
    images = torch.rand(2, 1, height, 256)
    widths = torch.tensor([256, 192])
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (images, widths),
            str(out),
            input_names=["images", "widths"],
            output_names=["log_probs", "lengths"],
            dynamic_axes={"images": {0: "batch", 3: "width"}, "widths": {0: "batch"},
                          "log_probs": {0: "batch", 1: "columns"}, "lengths": {0: "batch"}},
            opset_version=opset,
            dynamo=False,
        )
    _stamp_version(out, str(meta.get("version") or "unversioned"))
    verify(wrapper, out, height)
    return out


def _stamp_version(path: Path, version: str) -> None:
    import onnx

    graph = onnx.load(str(path))
    entry = graph.metadata_props.add()
    entry.key, entry.value = VERSION_KEY, version
    onnx.save(graph, str(path))


def exported_version(session) -> str | None:
    """The model version an ONNX Runtime session was exported from, if recorded."""
    return session.get_modelmeta().custom_metadata_map.get(VERSION_KEY)


def verify(wrapper: nn.Module, path: Path, height: int, atol: float = 2e-3) -> None:
    import onnxruntime as ort

    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    images = torch.rand(3, 1, height, 480)
    widths = torch.tensor([480, 300, 96])
    with torch.no_grad():
        ref_lp, ref_len = wrapper(images, widths)
    got_lp, got_len = session.run(None, {"images": images.numpy(), "widths": widths.numpy()})
    np.testing.assert_array_equal(got_len, ref_len.numpy())
    mask = np.arange(ref_lp.shape[1])[None, :] < ref_len.numpy()[:, None]
    diff = np.abs(got_lp - ref_lp.numpy())[mask].max()
    if diff > atol:
        raise RuntimeError(f"ONNX output differs from PyTorch by {diff:.2e}")
    log.info("verified %s with ONNX Runtime (max |Δ| = %.1e)", path, diff)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Export the encoder + CTC head to ONNX.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--opset", type=int, default=18)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    path = export_onnx(args.checkpoint, args.out, args.opset)
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
