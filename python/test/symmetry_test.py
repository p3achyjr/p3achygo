import symmetry as sym
import unittest
import tensorflow as tf

# yapf: disable
GRID = tf.convert_to_tensor([[0, 1, 2],
                             [3, 4, 5],
                             [6, 7, 8]])
# yapf: enable
LOC, N = tf.convert_to_tensor([1, 2]), 7


class SymmetryTest(unittest.TestCase):

  def test_id(self):
    self.assertTrue(
        tf.reduce_all(GRID == sym.apply_grid_symmetry(sym.IDENTITY, GRID)))
    self.assertTrue(
        tf.reduce_all(LOC == sym.apply_loc_symmetry(sym.IDENTITY, LOC, N)))

  def test_rot90(self):
    # yapf: disable
    grid = tf.convert_to_tensor([[6, 3, 0],
                                 [7, 4, 1],
                                 [8, 5, 2]])
    loc = tf.convert_to_tensor([2, 5])
    # yapf: enable
    self.assertTrue(
        tf.reduce_all(grid == sym.apply_grid_symmetry(sym.ROT90, GRID)))
    self.assertTrue(
        tf.reduce_all(loc == sym.apply_loc_symmetry(sym.ROT90, LOC, N)))

  def test_rot180(self):
    # yapf: disable
    grid = tf.convert_to_tensor([[8, 7, 6],
                                 [5, 4, 3],
                                 [2, 1, 0]])
    loc = tf.convert_to_tensor([5, 4])
    # yapf: enable
    self.assertTrue(
        tf.reduce_all(grid == sym.apply_grid_symmetry(sym.ROT180, GRID)))
    self.assertTrue(
        tf.reduce_all(loc == sym.apply_loc_symmetry(sym.ROT180, LOC, N)))

  def test_rot270(self):
    # yapf: disable
    grid = tf.convert_to_tensor([[2, 5, 8],
                                 [1, 4, 7],
                                 [0, 3, 6]])
    loc = tf.convert_to_tensor([4, 1])
    # yapf: enable
    self.assertTrue(
        tf.reduce_all(grid == sym.apply_grid_symmetry(sym.ROT270, GRID)))
    self.assertTrue(
        tf.reduce_all(loc == sym.apply_loc_symmetry(sym.ROT270, LOC, N)))

  def test_flip(self):
    # yapf: disable
    grid = tf.convert_to_tensor([[2, 1, 0],
                                 [5, 4, 3],
                                 [8, 7, 6]])
    loc = tf.convert_to_tensor([1, 4])
    # yapf: enable
    print(sym.apply_loc_symmetry(sym.FLIP, LOC, N))
    self.assertTrue(
        tf.reduce_all(grid == sym.apply_grid_symmetry(sym.FLIP, GRID)))
    self.assertTrue(
        tf.reduce_all(loc == sym.apply_loc_symmetry(sym.FLIP, LOC, N)))

  def test_fliprot90(self):
    # yapf: disable
    grid = tf.convert_to_tensor([[8, 5, 2],
                                 [7, 4, 1],
                                 [6, 3, 0]])
    loc = tf.convert_to_tensor([4, 5])
    # yapf: enable
    self.assertTrue(
        tf.reduce_all(grid == sym.apply_grid_symmetry(sym.FLIPROT90, GRID)))
    self.assertTrue(
        tf.reduce_all(loc == sym.apply_loc_symmetry(sym.FLIPROT90, LOC, N)))

  def test_fliprot180(self):
    # yapf: disable
    grid = tf.convert_to_tensor([[6, 7, 8],
                                 [3, 4, 5],
                                 [0, 1, 2]])
    loc = tf.convert_to_tensor([5, 2])
    # yapf: enable
    self.assertTrue(
        tf.reduce_all(grid == sym.apply_grid_symmetry(sym.FLIPROT180, GRID)))
    self.assertTrue(
        tf.reduce_all(loc == sym.apply_loc_symmetry(sym.FLIPROT180, LOC, N)))

  def test_fliprot270(self):
    # yapf: disable
    grid = tf.convert_to_tensor([[0, 3, 6],
                                 [1, 4, 7],
                                 [2, 5, 8]])
    loc = tf.convert_to_tensor([2, 1])
    # yapf: enable
    self.assertTrue(
        tf.reduce_all(grid == sym.apply_grid_symmetry(sym.FLIPROT270, GRID)))
    self.assertTrue(
        tf.reduce_all(loc == sym.apply_loc_symmetry(sym.FLIPROT270, LOC, N)))


if __name__ == '__main__':
  unittest.main()
