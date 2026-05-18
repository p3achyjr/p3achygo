import unittest

import numpy as np

import symmetry as sym

# yapf: disable
GRID = np.array([[0, 1, 2],
                 [3, 4, 5],
                 [6, 7, 8]])
# yapf: enable
LOC, N = np.array([1, 2]), 7


class SymmetryTest(unittest.TestCase):

    def test_id(self):
        np.testing.assert_array_equal(GRID, sym.apply_grid_symmetry(sym.IDENTITY, GRID))
        np.testing.assert_array_equal(LOC, sym.apply_loc_symmetry(sym.IDENTITY, LOC, N))

    def test_rot90(self):
        # yapf: disable
        grid = np.array([[6, 3, 0],
                         [7, 4, 1],
                         [8, 5, 2]])
        loc = np.array([2, 5])
        # yapf: enable
        np.testing.assert_array_equal(grid, sym.apply_grid_symmetry(sym.ROT90, GRID))
        np.testing.assert_array_equal(loc, sym.apply_loc_symmetry(sym.ROT90, LOC, N))

    def test_rot180(self):
        # yapf: disable
        grid = np.array([[8, 7, 6],
                         [5, 4, 3],
                         [2, 1, 0]])
        loc = np.array([5, 4])
        # yapf: enable
        np.testing.assert_array_equal(grid, sym.apply_grid_symmetry(sym.ROT180, GRID))
        np.testing.assert_array_equal(loc, sym.apply_loc_symmetry(sym.ROT180, LOC, N))

    def test_rot270(self):
        # yapf: disable
        grid = np.array([[2, 5, 8],
                         [1, 4, 7],
                         [0, 3, 6]])
        loc = np.array([4, 1])
        # yapf: enable
        np.testing.assert_array_equal(grid, sym.apply_grid_symmetry(sym.ROT270, GRID))
        np.testing.assert_array_equal(loc, sym.apply_loc_symmetry(sym.ROT270, LOC, N))

    def test_flip(self):
        # yapf: disable
        grid = np.array([[2, 1, 0],
                         [5, 4, 3],
                         [8, 7, 6]])
        loc = np.array([1, 4])
        # yapf: enable
        np.testing.assert_array_equal(grid, sym.apply_grid_symmetry(sym.FLIP, GRID))
        np.testing.assert_array_equal(loc, sym.apply_loc_symmetry(sym.FLIP, LOC, N))

    def test_fliprot90(self):
        # yapf: disable
        grid = np.array([[8, 5, 2],
                         [7, 4, 1],
                         [6, 3, 0]])
        loc = np.array([4, 5])
        # yapf: enable
        np.testing.assert_array_equal(
            grid, sym.apply_grid_symmetry(sym.FLIPROT90, GRID)
        )
        np.testing.assert_array_equal(
            loc, sym.apply_loc_symmetry(sym.FLIPROT90, LOC, N)
        )

    def test_fliprot180(self):
        # yapf: disable
        grid = np.array([[6, 7, 8],
                         [3, 4, 5],
                         [0, 1, 2]])
        loc = np.array([5, 2])
        # yapf: enable
        np.testing.assert_array_equal(
            grid, sym.apply_grid_symmetry(sym.FLIPROT180, GRID)
        )
        np.testing.assert_array_equal(
            loc, sym.apply_loc_symmetry(sym.FLIPROT180, LOC, N)
        )

    def test_fliprot270(self):
        # yapf: disable
        grid = np.array([[0, 3, 6],
                         [1, 4, 7],
                         [2, 5, 8]])
        loc = np.array([2, 1])
        # yapf: enable
        np.testing.assert_array_equal(
            grid, sym.apply_grid_symmetry(sym.FLIPROT270, GRID)
        )
        np.testing.assert_array_equal(
            loc, sym.apply_loc_symmetry(sym.FLIPROT270, LOC, N)
        )


if __name__ == "__main__":
    unittest.main()
