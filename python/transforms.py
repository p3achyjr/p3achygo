"""Framework-agnostic per-example transforms (numpy).

Two entry points:
  expand(serialized_bytes)      — parse with tfrecord.example_pb2 (lazy import).
  expand_from_features(feat_map) — caller has already parsed the proto. Use this
                                   when tfrecord and TF can't coexist in the
                                   same process; pass `tf.train.Example`'s
                                   `.features.feature` map directly.

Both paths share `_decode_features` and `_expand_common` for content logic.
"""

from __future__ import annotations

import numpy as np

import symmetry as sym
from constants import *


def _bytes_buf(feature):
    return feature.bytes_list.value[0]


def _decode(feature, dtype):
    return np.frombuffer(_bytes_buf(feature), dtype=dtype)


def as_loc(mv_index, bsize=BOARD_LEN) -> np.ndarray:
    if mv_index < 0:
        return np.array([-1, -1], dtype=np.int32)
    return np.array([mv_index // bsize, mv_index % bsize], dtype=np.int32)


def as_index(move, bsize=BOARD_LEN) -> int:
    return int(move[0]) * bsize + int(move[1])


def as_one_hot(move, bsize=BOARD_LEN) -> np.ndarray:
    arr = np.zeros((bsize, bsize), dtype=np.float32)
    if np.array_equal(move, NON_MOVE) or np.array_equal(move, PASS_MOVE):
        return arr
    arr[int(move[0]), int(move[1])] = 1.0
    return arr


def is_board_move(mv_index) -> bool:
    if mv_index < 0 or mv_index == PASS_MOVE_ENCODING:
        return False
    return True


def apply_loc_symmetry(symmetry, loc, grid_len) -> np.ndarray:
    if np.array_equal(loc, NON_MOVE) or np.array_equal(loc, PASS_MOVE):
        return loc
    return sym.apply_loc_symmetry(symmetry, loc, grid_len)


def get_color(board: np.ndarray, color: int) -> np.ndarray:
    return (board == color).astype(np.float32)


def _decode_features(features):
    """Decode an already-parsed proto Features map into a dict of numpy arrays.

    `features` is the `.features.feature` map from either
    `tfrecord.example_pb2.Example` or `tf.train.Example` — both expose the
    same `.bytes_list.value[0]` / `.float_list.value[0]` / `key in map` API.
    Keeping this proto-module-agnostic lets the TF and torch backends use
    their own parsers without loading the other's protobuf descriptors.
    """
    feat = features

    bsize = int(_decode(feat["bsize"], np.uint8)[0])

    board = _decode(feat["board"], np.int8).astype(np.int32).reshape(bsize, bsize)
    last_moves_idx = _decode(feat["last_moves"], np.int16).astype(np.int32)
    stones_atari = (
        _decode(feat["stones_atari"], np.int8).astype(np.int32).reshape(bsize, bsize)
    )
    stones_two_lib = (
        _decode(feat["stones_two_liberties"], np.int8)
        .astype(np.int32)
        .reshape(bsize, bsize)
    )
    stones_three_lib = (
        _decode(feat["stones_three_liberties"], np.int8)
        .astype(np.int32)
        .reshape(bsize, bsize)
    )
    stones_in_ladder = (
        _decode(feat["stones_in_ladder"], np.int8)
        .astype(np.int32)
        .reshape(bsize, bsize)
    )
    color = int(_decode(feat["color"], np.int8)[0])
    own = _decode(feat["own"], np.int8).astype(np.int32).reshape(bsize, bsize)
    policy = _decode(feat["pi"], np.float32).reshape(bsize * bsize + 1).copy()
    policy_aux = int(_decode(feat["pi_aux"], np.int16)[0])

    last_moves = np.stack([as_loc(int(idx), bsize) for idx in last_moves_idx], axis=0)

    has_pi_aux_dist = (
        "pi_aux_dist" in feat and len(feat["pi_aux_dist"].bytes_list.value) > 0
    )
    if has_pi_aux_dist:
        policy_aux_dist = np.frombuffer(
            feat["pi_aux_dist"].bytes_list.value[0], dtype=np.float32
        ).copy()
    else:
        policy_aux_dist = np.zeros(NUM_MOVES, dtype=np.float32)

    has_mcts_value_dist = (
        "mcts_value_dist" in feat and len(feat["mcts_value_dist"].bytes_list.value) > 0
    )
    if has_mcts_value_dist:
        # Stored as uint32; decoded as int32 (same bit pattern for values < 2^31).
        mcts_value_dist = np.frombuffer(
            feat["mcts_value_dist"].bytes_list.value[0], dtype=np.int32
        ).copy()
    else:
        mcts_value_dist = np.zeros(NUM_V_BUCKETS, dtype=np.int32)

    return {
        "bsize": bsize,
        "board": board,
        "last_moves": last_moves,
        "stones_atari": stones_atari,
        "stones_two_liberties": stones_two_lib,
        "stones_three_liberties": stones_three_lib,
        "stones_in_ladder": stones_in_ladder,
        "color": color,
        "komi": float(feat["komi"].float_list.value[0]),
        "own": own,
        "policy": policy,
        "policy_aux": policy_aux,
        "policy_aux_dist": policy_aux_dist,
        "has_pi_aux_dist": has_pi_aux_dist,
        "mcts_value_dist": mcts_value_dist,
        "has_mcts_value_dist": has_mcts_value_dist,
        "score": float(feat["score_margin"].float_list.value[0]),
        "q6": float(feat["q6"].float_list.value[0]),
        "q16": float(feat["q16"].float_list.value[0]),
        "q50": float(feat["q50"].float_list.value[0]),
        "q6_score": float(feat["q6_score"].float_list.value[0]),
        "q16_score": float(feat["q16_score"].float_list.value[0]),
        "q50_score": float(feat["q50_score"].float_list.value[0]),
    }


def _apply_symmetry_to_grids(
    symmetry,
    bsize,
    board,
    last_moves,
    stones_atari,
    stones_two_lib,
    stones_three_lib,
    own,
    policy,
    policy_aux,
    policy_aux_dist,
    stones_in_ladder,
):
    board = sym.apply_grid_symmetry(symmetry, board)
    last_moves = np.stack(
        [apply_loc_symmetry(symmetry, mv, bsize) for mv in last_moves], axis=0
    )
    stones_atari = sym.apply_grid_symmetry(symmetry, stones_atari)
    stones_two_lib = sym.apply_grid_symmetry(symmetry, stones_two_lib)
    stones_three_lib = sym.apply_grid_symmetry(symmetry, stones_three_lib)
    stones_in_ladder = sym.apply_grid_symmetry(symmetry, stones_in_ladder)
    own = sym.apply_grid_symmetry(symmetry, own)

    board_policy = policy[: bsize * bsize].reshape(bsize, bsize)
    board_policy = sym.apply_grid_symmetry(symmetry, board_policy)
    policy = np.concatenate(
        [
            board_policy.reshape(-1),
            policy[bsize * bsize : bsize * bsize + 1],
        ]
    )

    if is_board_move(int(policy_aux)):
        loc = as_loc(int(policy_aux), bsize=bsize)
        loc = sym.apply_loc_symmetry(symmetry, loc, bsize)
        policy_aux = as_index(loc, bsize=bsize)

    board_paux = policy_aux_dist[: bsize * bsize].reshape(bsize, bsize)
    board_paux = sym.apply_grid_symmetry(symmetry, board_paux)
    policy_aux_dist = np.concatenate(
        [
            board_paux.reshape(-1),
            policy_aux_dist[bsize * bsize : bsize * bsize + 1],
        ]
    )

    return (
        board,
        last_moves,
        stones_atari,
        stones_two_lib,
        stones_three_lib,
        own,
        policy,
        policy_aux,
        policy_aux_dist,
        stones_in_ladder,
    )


def _build_input_planes(
    color,
    bsize,
    board,
    last_moves,
    stones_atari,
    stones_two_lib,
    stones_three_lib,
    stones_in_ladder,
):
    black_stones = get_color(board, BLACK)
    white_stones = get_color(board, WHITE)
    black_atari = get_color(stones_atari, BLACK)
    white_atari = get_color(stones_atari, WHITE)
    black_two = get_color(stones_two_lib, BLACK)
    white_two = get_color(stones_two_lib, WHITE)
    black_three = get_color(stones_three_lib, BLACK)
    white_three = get_color(stones_three_lib, WHITE)
    black_ladder = get_color(stones_in_ladder, BLACK)
    white_ladder = get_color(stones_in_ladder, WHITE)

    if color == BLACK:
        our_stones, opp_stones = black_stones, white_stones
        our_atari, opp_atari = black_atari, white_atari
        our_two, opp_two = black_two, white_two
        our_three, opp_three = black_three, white_three
        our_ladder, opp_ladder = black_ladder, white_ladder
    else:
        our_stones, opp_stones = white_stones, black_stones
        our_atari, opp_atari = white_atari, black_atari
        our_two, opp_two = white_two, black_two
        our_three, opp_three = white_three, black_three
        our_ladder, opp_ladder = white_ladder, black_ladder

    mask_last_moves = np.random.uniform() < 0.05
    no_move = np.zeros((bsize, bsize), dtype=np.float32)

    return [
        our_stones,
        opp_stones,
        no_move if mask_last_moves else as_one_hot(last_moves[0], bsize=bsize),
        no_move if mask_last_moves else as_one_hot(last_moves[1], bsize=bsize),
        no_move if mask_last_moves else as_one_hot(last_moves[2], bsize=bsize),
        no_move if mask_last_moves else as_one_hot(last_moves[3], bsize=bsize),
        no_move if mask_last_moves else as_one_hot(last_moves[4], bsize=bsize),
        our_atari,
        opp_atari,
        our_two,
        opp_two,
        our_three,
        opp_three,
        our_ladder,
        opp_ladder,
    ]


def _build_score_one_hot(score: float) -> np.ndarray:
    score_index = int(np.floor(score)) + SCORE_RANGE_MIDPOINT
    if score_index < 0:
        score_index = 0
    elif score_index >= SCORE_RANGE:
        score_index = SCORE_RANGE - 1
    out = np.zeros(SCORE_RANGE, dtype=np.float32)
    out[score_index] = 1.0
    return out


def _build_global_state(color, last_moves, komi) -> np.ndarray:
    last_move_was_pass = np.array(
        [1.0 if np.array_equal(mv, PASS_MOVE) else 0.0 for mv in last_moves],
        dtype=np.float32,
    )
    color_indicator = np.array(
        [1.0 if color == BLACK else 0.0, 1.0 if color == WHITE else 0.0],
        dtype=np.float32,
    )
    komi_normalized = komi / 15.0
    if color == BLACK:
        komi_normalized = -komi_normalized
    return np.concatenate(
        [
            color_indicator,
            last_move_was_pass,
            np.array([komi_normalized], dtype=np.float32),
        ]
    )


def _expand_common(parsed):
    bsize = parsed["bsize"]
    board = parsed["board"]
    last_moves = parsed["last_moves"]
    stones_atari = parsed["stones_atari"]
    stones_two_lib = parsed["stones_two_liberties"]
    stones_three_lib = parsed["stones_three_liberties"]
    stones_in_ladder = parsed["stones_in_ladder"]
    color = parsed["color"]
    own = parsed["own"]
    policy = parsed["policy"]
    policy_aux = parsed["policy_aux"]
    policy_aux_dist = parsed["policy_aux_dist"]
    komi = parsed["komi"]
    score = parsed["score"]

    if score > 0:
        game_outcome = np.array([0.0, 1.0], dtype=np.float32)
    elif score < 0:
        game_outcome = np.array([1.0, 0.0], dtype=np.float32)
    else:
        game_outcome = np.array([0.5, 0.5], dtype=np.float32)

    symmetry = sym.get_random_symmetry()
    (
        board,
        last_moves,
        stones_atari,
        stones_two_lib,
        stones_three_lib,
        own,
        policy,
        policy_aux,
        policy_aux_dist,
        stones_in_ladder,
    ) = _apply_symmetry_to_grids(
        symmetry,
        bsize,
        board,
        last_moves,
        stones_atari,
        stones_two_lib,
        stones_three_lib,
        own,
        policy,
        policy_aux,
        policy_aux_dist,
        stones_in_ladder,
    )

    if color != BLACK:
        own = -own

    input_planes = _build_input_planes(
        color,
        bsize,
        board,
        last_moves,
        stones_atari,
        stones_two_lib,
        stones_three_lib,
        stones_in_ladder,
    )

    # CHW → HWC
    input_arr = np.transpose(
        np.stack(input_planes, axis=0).astype(np.float32), (1, 2, 0)
    )

    score_one_hot = _build_score_one_hot(score)
    input_global_state = _build_global_state(color, last_moves, komi)

    return {
        "input": input_arr,
        "input_global_state": input_global_state,
        "color": color,
        "own": own,
        "policy": policy,
        "policy_aux": policy_aux,
        "policy_aux_dist": policy_aux_dist,
        "has_pi_aux_dist": parsed["has_pi_aux_dist"],
        "mcts_value_dist": parsed["mcts_value_dist"],
        "has_mcts_value_dist": parsed["has_mcts_value_dist"],
        "score_one_hot": score_one_hot,
        "game_outcome": game_outcome,
    }


def expand(features):
    """Primary entry. Takes a parsed proto Features map, returns the 20-tuple.

    Use this from backend dataset code: parse the proto with whichever module
    your runtime owns (tfrecord.example_pb2 on the torch side, tf.train.Example
    on the TF side) and pass `example.features.feature` here.
    """
    parsed = _decode_features(features)
    expanded = _expand_common(parsed)

    return (
        expanded["input"],
        expanded["input_global_state"],
        np.int32(expanded["color"]),
        np.float32(parsed["komi"]),
        np.float32(parsed["score"]),
        expanded["score_one_hot"],
        expanded["policy"],
        np.int32(expanded["policy_aux"]),
        expanded["policy_aux_dist"],
        np.bool_(expanded["has_pi_aux_dist"]),
        expanded["own"],
        np.float32(parsed["q6"]),
        np.float32(parsed["q16"]),
        np.float32(parsed["q50"]),
        np.float32(parsed["q6_score"]),
        np.float32(parsed["q16_score"]),
        np.float32(parsed["q50_score"]),
        expanded["game_outcome"],
        expanded["mcts_value_dist"],
        np.bool_(expanded["has_mcts_value_dist"]),
    )


def expand_bytes(serialized_bytes):
    """Convenience: parse with `tfrecord.example_pb2`, then call `expand`.

    Lazy-imports `tfrecord` so this module stays free of TF/tfrecord proto
    dependencies until the bytes path is actually used. Use this from the
    torch dataset (which already pulls in tfrecord) and from CLI / tests.
    """
    from tfrecord import example_pb2 as _pb  # lazy

    ex = _pb.Example()
    ex.ParseFromString(serialized_bytes)
    return expand(ex.features.feature)
