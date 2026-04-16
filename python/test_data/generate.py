"""
Generate test_data/mixed_schema.tfrecord — a small TFRecord chunk containing:
  - 3 old-schema records (no pi_aux_dist / mcts_value_dist)
  - 3 new-schema records (with both optional fields)

Run from python/ directory:
  python3 test_data/generate.py
"""

import os
import sys

import numpy as np
import tensorflow as tf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from constants import BOARD_LEN, BLACK, WHITE, NUM_MOVES, NUM_V_BUCKETS


OUT_PATH = os.path.join(os.path.dirname(__file__), "mixed_schema.tfrecord")


def _bytes_feat(data: bytes):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[data]))


def _float_feat(v: float):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[v]))


def _make_base_features(color: int, score_margin: float, pi_aux_index: int) -> dict:
    bsize = BOARD_LEN
    last_moves = np.array([-1, -1, -1, -1, -1], dtype=np.int16)
    pi = np.zeros(bsize * bsize + 1, dtype=np.float32)
    pi[0] = 1.0  # one-hot at move 0

    return {
        "bsize": _bytes_feat(np.array([bsize], dtype=np.uint8).tobytes()),
        "board": _bytes_feat(np.zeros(bsize * bsize, dtype=np.int8).tobytes()),
        "last_moves": _bytes_feat(last_moves.tobytes()),
        "stones_atari": _bytes_feat(np.zeros(bsize * bsize, dtype=np.int8).tobytes()),
        "stones_two_liberties": _bytes_feat(
            np.zeros(bsize * bsize, dtype=np.int8).tobytes()
        ),
        "stones_three_liberties": _bytes_feat(
            np.zeros(bsize * bsize, dtype=np.int8).tobytes()
        ),
        "stones_in_ladder": _bytes_feat(
            np.zeros(bsize * bsize, dtype=np.int8).tobytes()
        ),
        "color": _bytes_feat(np.array([color], dtype=np.int8).tobytes()),
        "komi": _float_feat(6.5),
        "own": _bytes_feat(np.zeros(bsize * bsize, dtype=np.int8).tobytes()),
        "pi": _bytes_feat(pi.tobytes()),
        "pi_aux": _bytes_feat(np.array([pi_aux_index], dtype=np.int16).tobytes()),
        "score_margin": _float_feat(score_margin),
        "q6": _float_feat(0.5),
        "q16": _float_feat(0.5),
        "q50": _float_feat(0.5),
        "q6_score": _float_feat(score_margin),
        "q16_score": _float_feat(score_margin),
        "q50_score": _float_feat(score_margin),
    }


def make_old_schema(color: int, score_margin: float, pi_aux_index: int) -> bytes:
    """Old-schema record: no pi_aux_dist or mcts_value_dist."""
    feat = _make_base_features(color, score_margin, pi_aux_index)
    ex = tf.train.Example(features=tf.train.Features(feature=feat))
    return ex.SerializeToString()


def make_new_schema(
    color: int,
    score_margin: float,
    pi_aux_index: int,
    pi_aux_dist: np.ndarray,
    mcts_value_dist: np.ndarray,
) -> bytes:
    """New-schema record: includes pi_aux_dist and mcts_value_dist."""
    feat = _make_base_features(color, score_margin, pi_aux_index)
    feat["pi_aux_dist"] = _bytes_feat(pi_aux_dist.astype(np.float32).tobytes())
    feat["mcts_value_dist"] = _bytes_feat(mcts_value_dist.astype(np.uint32).tobytes())
    ex = tf.train.Example(features=tf.train.Features(feature=feat))
    return ex.SerializeToString()


def main():
    records = []

    # --- 3 old-schema records ---
    records.append(make_old_schema(BLACK, score_margin=5.5, pi_aux_index=42))
    records.append(make_old_schema(WHITE, score_margin=-5.5, pi_aux_index=100))
    records.append(make_old_schema(BLACK, score_margin=0.5, pi_aux_index=361))  # pass

    # --- 3 new-schema records ---
    # Record 4: uniform pi_aux_dist, sparse mcts_value_dist
    pi_aux_dist_a = np.ones(NUM_MOVES, dtype=np.float32) / NUM_MOVES
    mcts_dist_a = np.zeros(NUM_V_BUCKETS, dtype=np.uint32)
    mcts_dist_a[25] = 10
    mcts_dist_a[26] = 5
    records.append(
        make_new_schema(
            BLACK,
            3.5,
            pi_aux_index=7,
            pi_aux_dist=pi_aux_dist_a,
            mcts_value_dist=mcts_dist_a,
        )
    )

    # Record 5: one-hot pi_aux_dist at index 99, mcts_value_dist with one bucket
    pi_aux_dist_b = np.zeros(NUM_MOVES, dtype=np.float32)
    pi_aux_dist_b[99] = 1.0
    mcts_dist_b = np.zeros(NUM_V_BUCKETS, dtype=np.uint32)
    mcts_dist_b[0] = 100
    records.append(
        make_new_schema(
            WHITE,
            -3.5,
            pi_aux_index=99,
            pi_aux_dist=pi_aux_dist_b,
            mcts_value_dist=mcts_dist_b,
        )
    )

    # Record 6: pass pi_aux, all-bucket mcts_value_dist
    pi_aux_dist_c = np.zeros(NUM_MOVES, dtype=np.float32)
    pi_aux_dist_c[361] = 1.0  # pass
    mcts_dist_c = np.ones(NUM_V_BUCKETS, dtype=np.uint32)
    records.append(
        make_new_schema(
            BLACK,
            0.0,
            pi_aux_index=361,
            pi_aux_dist=pi_aux_dist_c,
            mcts_value_dist=mcts_dist_c,
        )
    )

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with tf.io.TFRecordWriter(OUT_PATH) as w:
        for rec in records:
            w.write(rec)

    print(f"Wrote {len(records)} records to {OUT_PATH}")
    print(f"  Old schema: {3}")
    print(f"  New schema: {3}")


if __name__ == "__main__":
    main()
