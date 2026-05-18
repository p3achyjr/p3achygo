from __future__ import annotations

import numpy as np

IDENTITY = 0
ROT90 = 1
ROT180 = 2
ROT270 = 3
FLIP = 4
FLIPROT90 = 5
FLIPROT180 = 6
FLIPROT270 = 7

__SYM_MAX = 8


def get_random_symmetry() -> int:
    return int(np.random.randint(0, __SYM_MAX))


def flip(x: np.ndarray) -> np.ndarray:
    # Force a contiguous copy. np.flip returns a view with a negative stride;
    # torch.as_tensor (DataLoader collate) rejects negative-stride arrays.
    return np.ascontiguousarray(np.flip(x, axis=1))


def rotate(x: np.ndarray, k: int) -> np.ndarray:
    return np.ascontiguousarray(np.rot90(x, k=k, axes=(1, 0)))


def flip_loc(loc: np.ndarray, n: int) -> np.ndarray:
    return np.array([loc[0], n - loc[1] - 1], dtype=np.asarray(loc).dtype)


def rotate_loc(loc: np.ndarray, k: int, n: int) -> np.ndarray:
    loc = np.asarray(loc)
    if k == 1:
        return np.array([loc[1], n - loc[0] - 1], dtype=loc.dtype)
    elif k == 2:
        return np.array([n - loc[0] - 1, n - loc[1] - 1], dtype=loc.dtype)
    elif k == 3:
        return np.array([n - loc[1] - 1, loc[0]], dtype=loc.dtype)
    return loc


def apply_grid_symmetry(s, grid: np.ndarray) -> np.ndarray:
    if s == ROT90:
        return rotate(grid, 1)
    elif s == ROT180:
        return rotate(grid, 2)
    elif s == ROT270:
        return rotate(grid, 3)
    elif s == FLIP:
        return flip(grid)
    elif s == FLIPROT90:
        return rotate(flip(grid), 1)
    elif s == FLIPROT180:
        return rotate(flip(grid), 2)
    elif s == FLIPROT270:
        return rotate(flip(grid), 3)
    else:
        return grid


def apply_loc_symmetry(s, loc: np.ndarray, grid_len: int) -> np.ndarray:
    if s == ROT90:
        return rotate_loc(loc, 1, grid_len)
    elif s == ROT180:
        return rotate_loc(loc, 2, grid_len)
    elif s == ROT270:
        return rotate_loc(loc, 3, grid_len)
    elif s == FLIP:
        return flip_loc(loc, grid_len)
    elif s == FLIPROT90:
        return rotate_loc(flip_loc(loc, grid_len), 1, grid_len)
    elif s == FLIPROT180:
        return rotate_loc(flip_loc(loc, grid_len), 2, grid_len)
    elif s == FLIPROT270:
        return rotate_loc(flip_loc(loc, grid_len), 3, grid_len)
    else:
        return loc
