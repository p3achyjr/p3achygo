"""
tf2onnx custom rewriters for efficient ONNX op lowering.

Pass rewrite_rms_normalization as a custom_rewriter to tf2onnx.convert.from_function.
It runs after all built-in rewriters, so the graph is already in ONNX form.
"""

from __future__ import annotations

from typing import Optional

from tf2onnx.graph import Graph, Node
from tf2onnx.graph_matcher import GraphMatcher, OpTypePattern


# ---------------------------------------------------------------------------
# RMSNormalization rewriter
# ---------------------------------------------------------------------------
#
# Pattern (produced by tf2onnx from keras.layers.RMSNormalization):
#   Mul(x, x) -> GlobalAveragePool -> Add(eps) -> Sqrt -> Reciprocal
#   -> Mul(rsqrt, scale) -> Mul(x, scaled_rsqrt)
#
# Replaced with a single ONNX opset-23 RMSNormalization node.

_RMS_PATTERN = OpTypePattern(
    "Mul",
    name="mul_out",
    inputs=[
        OpTypePattern(
            "Mul",
            name="mul_scale",
            inputs=[
                OpTypePattern(
                    "Reciprocal",
                    name="recip",
                    inputs=[
                        OpTypePattern(
                            "Sqrt",
                            name="sqrt",
                            inputs=[
                                OpTypePattern(
                                    "Add",
                                    name="add_eps",
                                    inputs=[
                                        OpTypePattern(
                                            "GlobalAveragePool",
                                            name="gap",
                                            inputs=[
                                                OpTypePattern(
                                                    "Mul", name="sq", inputs=["*", "*"]
                                                ),
                                            ],
                                        ),
                                        "*",
                                    ],
                                ),
                            ],
                        ),
                    ],
                ),
                "*",
            ],
        ),
        "*",
    ],
)


def _get_const_scalar(node: Node) -> Optional[float]:
    """Return scalar float from a Const node, or None."""
    if not node.is_const():
        return None
    arr = node.get_tensor_value(as_list=False)
    if arr.size == 1:
        return float(arr.flat[0])
    return None


def rewrite_rms_normalization(g: Graph, ops: list[Node]) -> list[Node]:
    """Replace each RMSNorm 7-op subgraph with a single RMSNormalization node."""
    matcher = GraphMatcher(_RMS_PATTERN, allow_reorder=False)
    matches = list(matcher.match_ops(ops))

    for match in matches:
        sq = match.get_op("sq")
        gap = match.get_op("gap")
        add_eps = match.get_op("add_eps")
        sqrt = match.get_op("sqrt")
        recip = match.get_op("recip")
        mul_scale = match.get_op("mul_scale")
        mul_out = match.get_op("mul_out")

        # sq must be x*x (both inputs the same)
        if sq.input[0] != sq.input[1]:
            continue
        x = sq.input[0]

        # mul_out must apply to the same x
        x_check = next((i for i in mul_out.input if i != mul_scale.output[0]), None)
        if x_check != x:
            continue

        # extract epsilon from the Add node
        eps_input = next((i for i in add_eps.input if i != gap.output[0]), None)
        if eps_input is None:
            continue
        eps_node = g.get_node_by_output(eps_input)
        if eps_node is None:
            continue
        epsilon = _get_const_scalar(eps_node)
        if epsilon is None:
            continue

        # extract learned scale tensor name (the non-rsqrt input of mul_scale)
        scale = next((i for i in mul_scale.input if i != recip.output[0]), None)
        if scale is None:
            continue

        # remove matched nodes; keep the output name so downstream is unaffected
        out_name = mul_out.output[0]
        dtype = g.get_dtype(out_name)
        shape = g.get_shape(out_name)

        for node in [mul_out, mul_scale, recip, sqrt, add_eps, gap, sq]:
            g.remove_node(node.name)

        g.make_node(
            "RMSNormalization",
            inputs=[x, scale],
            attr={"epsilon": epsilon, "axis": -1},
            name=mul_out.name + "_rms_norm",
            outputs=[out_name],
            dtypes=[dtype],
            shapes=[shape],
            skip_conversion=False,
        )

    return ops
