"""Export a torch P3achyGoModel checkpoint (.pt) to ONNX.

Output names + IO layout match the keras path so the resulting .onnx
drops in for `cc/nn/engine/scripts:build_and_run_trt_engine` and the
production TRT pipeline. See `cc/nn/engine/trt_names.h` for the subset
the runtime queries.

FP16: when --fp16 is set, the torch model is converted to fp16 weights
in-process via `model.half()` before tracing. The exported graph keeps
fp32 IO (we cast inputs fp32→fp16 inside the wrapper, and the model's
final `_f32` casts already produce fp32 outputs), so engines built from
this ONNX present the same input/output dtypes as the keras path.

Output filename: defaults to `<stem>_pt.onnx` (note the `_pt` suffix) to
distinguish from the keras-produced `<stem>.onnx`.
"""

from __future__ import annotations

import os

os.environ.setdefault("P3ACHYGO_BACKEND", "torch")

from pathlib import Path

import torch
import torch.nn as nn
from absl import app, flags, logging

FLAGS = flags.FLAGS
flags.DEFINE_string("model_path", "", "Path to the .pt torch checkpoint.")
flags.DEFINE_string(
    "onnx_path",
    "",
    "Output ONNX path. Defaults to <model_dir>/_onnx/<stem>.onnx.",
)
flags.DEFINE_bool(
    "fp16",
    False,
    "Convert torch model to fp16 weights before exporting (IO stays fp32).",
)
flags.DEFINE_integer(
    "opset",
    22,
    "ONNX opset version. Default 22: opset 23 is the first version with a "
    "native `RMSNormalization` op which the installed TRT can't import; "
    "opset 22 forces torch.onnx to decompose RMSNorm into elementwise math "
    "(matching what tf2onnx already emits on the keras path).",
)
flags.DEFINE_bool(
    "dynamo",
    True,
    "Use the dynamo exporter (default). Set to False to use the legacy "
    "TorchScript exporter for diffing.",
)

# Order matches keras `convert_to_onnx.py` (and the torch model's forward
# return tuple).
_OUTPUT_NAMES = [
    "00:pi_logits",
    "01:pi",
    "02:outcome_logits",
    "03:outcome",
    "04:own",
    "05:score_logits",
    "06:score_probs",
    "07:gamma",
    "08:pi_logits_aux",
    "09:q6",
    "10:q16",
    "11:q50",
    "12:q6_err",
    "13:q16_err",
    "14:q50_err",
    "15:q6_score",
    "16:q16_score",
    "17:q50_score",
    "18:q6_score_err",
    "19:q16_score_err",
    "20:q50_score_err",
    "21:pi_logits_soft",
    "22:pi_logits_optimistic",
    "23:mcts_dist_logits",
    "24:mcts_dist_probs",
]


class _ExportWrapper(nn.Module):
    """Pin training=False, let scores default (so it bakes into the graph
    as a constant), and — when fp16 — cast fp32 IO to the model's compute
    dtype at the boundary so the exported graph still presents fp32 IO."""

    def __init__(self, inner: nn.Module, compute_dtype: torch.dtype):
        super().__init__()
        self.inner = inner
        self.compute_dtype = compute_dtype

    def forward(self, board_state, game_state):
        b = board_state.to(self.compute_dtype)
        g = game_state.to(self.compute_dtype)
        return self.inner(b, g, training=False, scores=None)


def main(_):
    if not FLAGS.model_path:
        logging.fatal("--model_path required")

    from backend_torch.model_utils import load_model

    model_path = Path(FLAGS.model_path)
    onnx_path = (
        Path(FLAGS.onnx_path)
        if FLAGS.onnx_path
        else model_path.parent / "_onnx" / (model_path.stem + ".onnx")
    )
    onnx_path.parent.mkdir(parents=True, exist_ok=True)

    logging.info(f"Loading torch model from {model_path}")
    model = load_model(str(model_path)).eval()
    if FLAGS.fp16:
        logging.info("Converting model weights to fp16 …")
        model = model.half()
    compute_dtype = torch.float16 if FLAGS.fp16 else torch.float32

    planes_shape = model.input_planes_shape()
    features_shape = model.input_features_shape()
    # Trace with batch≥2 — dim=1 specializes to a static 1 in torch.export
    # even with `Dim("N")` listed. IO stays fp32; wrapper casts internally.
    _TRACE_BATCH = 2
    dummy_board = torch.zeros([_TRACE_BATCH, *planes_shape], dtype=torch.float32)
    dummy_features = torch.zeros([_TRACE_BATCH, *features_shape], dtype=torch.float32)

    logging.info(f"Exporting → {onnx_path} (dynamo={FLAGS.dynamo})")
    wrapper = _ExportWrapper(model, compute_dtype).eval()
    if FLAGS.dynamo:
        batch = torch.export.Dim("N", min=1, max=4096)
        torch.onnx.export(
            wrapper,
            (dummy_board, dummy_features),
            str(onnx_path),
            input_names=["board_state", "game_state"],
            output_names=_OUTPUT_NAMES,
            dynamic_shapes={
                "board_state": {0: batch},
                "game_state": {0: batch},
            },
            opset_version=FLAGS.opset,
            dynamo=True,
            external_data=False,
        )
    else:
        torch.onnx.export(
            wrapper,
            (dummy_board, dummy_features),
            str(onnx_path),
            input_names=["board_state", "game_state"],
            output_names=_OUTPUT_NAMES,
            dynamic_axes={
                "board_state": {0: "N"},
                "game_state": {0: "N"},
                **{name: {0: "N"} for name in _OUTPUT_NAMES},
            },
            # Legacy exporter caps at opset 20.
            opset_version=min(FLAGS.opset, 20),
            dynamo=False,
        )
    logging.info("Done.")


if __name__ == "__main__":
    app.run(main)
