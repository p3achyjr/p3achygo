import tensorflow as tf
import keras
import tf2onnx, onnx
import onnxruntime as ort
import numpy as np
import collections
import transforms
from board import GoBoard
from absl import app, flags, logging
from pathlib import Path

import onnx
from onnx import checker
from onnxconverter_common.float16 import convert_float_to_float16
from onnx import TensorProto, helper

from model import P3achyGoModel
from constants import *
from lr_schedule import ConstantLRSchedule

FLAGS = flags.FLAGS
DUMMY_BATCH_SIZE = 32

flags.DEFINE_string("model_path", "", "Path to SavedModel.")
flags.DEFINE_string("onnx_name", "", "Name of ONNX model.")
flags.DEFINE_string("val_ds", "", "Validation DS, to verify conversion.")
flags.DEFINE_integer("num_samples", -1, "Number of samples to collect stats on.")
flags.DEFINE_bool("fp16", False, "Whether to convert to FP16.")


def random_inputs(planes_shape, features_shape):
    return (
        np.random.random([DUMMY_BATCH_SIZE] + planes_shape).astype(np.float32),
        np.random.random([DUMMY_BATCH_SIZE] + features_shape).astype(np.float32),
    )


def update_val_stats(stats, pi_pred, outcome_pred, score_pred, policy, score):
    outcome = score >= 0
    true_move = policy if len(policy.shape) == 1 else np.argmax(policy, axis=1)
    pi_pred = np.argmax(pi_pred, axis=1)
    outcome_pred = np.argmax(outcome_pred, axis=1)
    correct_move = pi_pred == true_move
    correct_outcome = outcome == outcome_pred.astype(np.int32)
    score_pred = np.argmax(score_pred, axis=1) - SCORE_RANGE_MIDPOINT
    score_diff = np.abs(score - score_pred)

    n = pi_pred.size
    stats["num_batches"] += 1
    stats["num_examples"] += n
    stats["correct_moves"] += np.sum(correct_move)
    stats["correct_outcomes"] += np.sum(correct_outcome)
    stats["score_diff"] += np.mean(score_diff)


def prune_unused_graph_inputs(model: onnx.ModelProto) -> onnx.ModelProto:
    g = model.graph

    # Tensors that are actually consumed somewhere (or are outputs)
    used = set()
    for node in g.node:
        used.update([x for x in node.input if x])

    used.update([o.name for o in g.output])  # keep anything that is an output

    # Keep real inputs that are used; drop unused ones
    new_inputs = [i for i in g.input if i.name in used]
    del g.input[:]
    g.input.extend(new_inputs)
    return model


def fix_cast_output_types(model: onnx.ModelProto) -> onnx.ModelProto:
    """Fix Cast nodes where output is marked FP16 but Cast attribute is FP32.

    convert_float_to_float16() updates value_info but doesn't always update
    the Cast node's 'to' attribute, causing type mismatches in TensorRT.
    """
    g = model.graph

    # Build map: tensor_name -> type from value_info
    tensor_types = {}
    for vi in g.value_info:
        tensor_types[vi.name] = vi.type.tensor_type.elem_type

    fixed_count = 0
    for node in g.node:
        if node.op_type == "Cast":
            output_name = node.output[0]
            if output_name in tensor_types:
                expected_type = tensor_types[output_name]
                # Check Cast node's 'to' attribute
                for attr in node.attribute:
                    if attr.name == "to":
                        if attr.i != expected_type and expected_type == TensorProto.FLOAT16:
                            logging.info(f"Fixing Cast '{node.name}': changing 'to' from {attr.i} to FLOAT16")
                            attr.i = TensorProto.FLOAT16
                            fixed_count += 1
                        break

    logging.info(f"fix_cast_output_types: fixed {fixed_count} cast nodes")
    return model


def fix_mixed_precision_inputs(model: onnx.ModelProto) -> onnx.ModelProto:
    """Fix mixed fp32/fp16 inputs caused by keep_io_types.

    keep_io_types inserts Cast(to=fp32) at graph outputs. The fp32 output
    tensor may also be consumed by other (fp16) nodes, causing a type mismatch
    in TRT. For each such consumer, replace the fp32 input with a Cast(to=fp16).
    """
    g = model.graph
    output_names = {o.name for o in g.output}

    # Find fp32 tensors: outputs of Cast(to=fp32) that are graph outputs.
    fp32_outputs = set()
    for node in g.node:
        if node.op_type == "Cast" and node.output[0] in output_names:
            for attr in node.attribute:
                if attr.name == "to" and attr.i == TensorProto.FLOAT:
                    fp32_outputs.add(node.output[0])

    if not fp32_outputs:
        return model

    # Find fp32 outputs that eventually reach a binary op (multiple inputs)
    # through a chain of unary ops. Only those need a Cast(to=fp16).
    # Build map: tensor -> consumers
    tensor_consumers = {}
    for node in g.node:
        for inp in node.input:
            if inp:
                tensor_consumers.setdefault(inp, []).append(node)

    consumed = set()
    for fp32_out in fp32_outputs:
        # BFS through unary consumers to see if we reach a binary op.
        queue = [fp32_out]
        visited = set()
        while queue:
            tensor = queue.pop()
            if tensor in visited:
                continue
            visited.add(tensor)
            for consumer in tensor_consumers.get(tensor, []):
                if consumer.op_type == "Cast":
                    continue
                non_empty_inputs = [i for i in consumer.input if i]
                if len(non_empty_inputs) > 1:
                    consumed.add(fp32_out)
                    queue.clear()
                    break
                else:
                    # Unary op — follow its output
                    for out in consumer.output:
                        queue.append(out)

    if not consumed:
        return model

    # Insert a Cast(to=fp16) before each consumer and rewrite its input.
    cast_cache = {}
    cast_id = 0
    new_nodes = []
    for node in g.node:
        rewritten_inputs = list(node.input)
        needs_rewrite = False
        for idx, inp in enumerate(rewritten_inputs):
            if inp in consumed:
                if inp not in cast_cache:
                    cast_out = f"{inp}_to_fp16_{cast_id}"
                    cast_id += 1
                    cast_cache[inp] = cast_out
                    new_nodes.append(helper.make_node(
                        "Cast", [inp], [cast_out],
                        name=f"_fix_mixed_cast_{cast_id}",
                        to=TensorProto.FLOAT16,
                    ))
                rewritten_inputs[idx] = cast_cache[inp]
                needs_rewrite = True
        if needs_rewrite:
            del node.input[:]
            node.input.extend(rewritten_inputs)
        new_nodes.append(node)

    logging.info(f"fix_mixed_precision: inserted {len(cast_cache)} cast nodes")
    del g.node[:]
    g.node.extend(new_nodes)
    return model


def main(_):
    if not FLAGS.model_path:
        logging.warning("No Model Path Specified.")
        return

    model_path = Path(FLAGS.model_path)
    onnx_dir = model_path.parent / "_onnx"
    # Use model name if onnx_name is default, otherwise use provided name
    if FLAGS.onnx_name == "":
        onnx_name = model_path.stem + ".onnx"
    else:
        onnx_name = FLAGS.onnx_name
    onnx_path = str(onnx_dir / onnx_name)
    logging.info(f"Model Path: {model_path}")
    logging.info(f"Onnx Path: {onnx_path}")

    with tf.device("/cpu:0"):
        tf.keras.mixed_precision.set_global_policy("float32")
        model = keras.models.load_model(
            model_path, custom_objects=P3achyGoModel.custom_objects()
        )
        planes_shape = model.input_planes_shape()
        features_shape = model.input_features_shape()
        model(
            *[
                tf.convert_to_tensor(x)
                for x in random_inputs(planes_shape, features_shape)
            ]
        )
        model.summary()

        # Scores is a constant - embed it in the graph rather than passing as input
        scores = (
            0.05 * tf.cast(tf.range(-SCORE_RANGE // 2, SCORE_RANGE // 2), tf.float32)
            + 0.025
        )

        @tf.function
        def model_fn(board_state: tf.Tensor, game_state: tf.Tensor):
            # v1 model returns 46 outputs (23 FVI + 23 BN)
            # Include all outputs to avoid grappler optimization issues
            (
                pi_logits,
                pi,
                outcome_logits,
                outcome,
                own,
                score_logits,
                score_probs,
                gamma,
                pi_logits_aux,
                q6,
                q16,
                q50,
                q6_err,
                q16_err,
                q50_err,
                q6_score,
                q16_score,
                q50_score,
                q6_score_err,
                q16_score_err,
                q50_score_err,
                pi_logits_soft,
                pi_logits_optimistic,
            ) = model(board_state, game_state, training=False, scores=scores)

            return {
                # FVI outputs (0-22)
                "00:pi_logits": pi_logits,
                "01:pi": pi,
                "02:outcome_logits": outcome_logits,
                "03:outcome": outcome,
                "04:own": own,
                "05:score_logits": score_logits,
                "06:score_probs": score_probs,
                "07:gamma": gamma,
                "08:pi_logits_aux": pi_logits_aux,
                "09:q6": q6,
                "10:q16": q16,
                "11:q50": q50,
                "12:q6_err": q6_err,
                "13:q16_err": q16_err,
                "14:q50_err": q50_err,
                "15:q6_score": q6_score,
                "16:q16_score": q16_score,
                "17:q50_score": q50_score,
                "18:q6_score_err": q6_score_err,
                "19:q16_score_err": q16_score_err,
                "20:q50_score_err": q50_score_err,
                "21:pi_logits_soft": pi_logits_soft,
                "22:pi_logits_optimistic": pi_logits_optimistic,
            }

        input_signature = [
            tf.TensorSpec(
                shape=[None] + model.input_planes_shape(),
                dtype=tf.float32,
                name="board_state",
            ),
            tf.TensorSpec(
                shape=[None] + model.input_features_shape(),
                dtype=tf.float32,
                name="game_state",
            ),
        ]
        # Note: tf2onnx runs optimization passes by default
        # Set optimize=False to disable all optimizations (for debugging)
        # Or use optimizer_exclude to disable specific optimizers
        onnx_model, _ = tf2onnx.convert.from_function(
            model_fn,
            input_signature=input_signature,
            # optimize=False,  # Uncomment to disable all optimizations
        )
        onnx_model = prune_unused_graph_inputs(onnx_model)
        if FLAGS.fp16:
            logging.info("Converting ONNX model to FP16...")
            # Block Softmax and specific Softplus nodes to avoid type mismatches
            # Softmax outputs are model outputs and must stay FP32
            # The value_head gamma Softplus must also remain FP32
            onnx_model = convert_float_to_float16(
                onnx_model,
                keep_io_types=True,
            )

            onnx_model = fix_cast_output_types(onnx_model)
            onnx_model = fix_mixed_precision_inputs(onnx_model)

        onnx_dir.mkdir(exist_ok=True)
        onnx.save(onnx_model, onnx_path)

    if FLAGS.val_ds:
        val_ds = tf.data.TFRecordDataset(FLAGS.val_ds, compression_type="ZLIB")
        val_ds = val_ds.map(transforms.expand, num_parallel_calls=tf.data.AUTOTUNE)
        val_ds = val_ds.batch(48)
        if FLAGS.num_samples != -1:
            val_ds = val_ds.take(FLAGS.num_samples)
        val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

        sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
        stats_ort, stats_tf = (
            collections.defaultdict(float),
            collections.defaultdict(float),
        )

        outcome_mse = 0
        n = 0
        for (
            in_board_state,
            in_global_state,
            _,
            _,
            score,
            _,
            policy,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
            _,
        ) in val_ds:
            out_ort = sess.run(
                None,
                {
                    "board_state": in_board_state.numpy(),
                    "game_state": in_global_state.numpy(),
                },
            )
            out_tf = model(in_board_state, in_global_state)
            outcome_mse += np.mean(np.square(out_ort[3] - out_tf[3]))
            n += 1
            update_val_stats(
                stats_ort, out_ort[0], out_ort[3], out_ort[6], policy, score
            )
            update_val_stats(stats_tf, out_tf[0], out_tf[3], out_tf[6], policy, score)

        def stats_str(stats):
            n = stats["num_examples"]
            b = stats["num_batches"]
            if n == 0:
                n = 1
            if b == 0:
                b = 1
            return "\n".join(
                [
                    f'Correct Move Percentage: {stats["correct_moves"] / n}',
                    f'Correct Outcome Percentage: {stats["correct_outcomes"] / n}',
                    f'Avg Score Diff: {stats["score_diff"] / b}',
                ]
            )

        logging.info(
            f"\nORT:\n{stats_str(stats_ort)}\n\n" + f"TF:\n{stats_str(stats_tf)}"
        )
        logging.info(f"Mean Outcome MSE: {outcome_mse / n}")


if __name__ == "__main__":
    app.run(main)
