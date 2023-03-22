"""badukmovies_all dataset."""

import tensorflow_datasets as tfds
from . import badukmovies_all_dataset_builder


class BadukmoviesAllTest(tfds.testing.DatasetBuilderTestCase):
  """Tests for badukmovies_all dataset."""
  import datasets.common.constants as constants
  DATASET_CLASS = badukmovies_all_dataset_builder.Builder
  SPLITS = {
      'train': 7864 * constants.REUSE_FACTOR /
               8,  # Number of fake train example
  }


if __name__ == '__main__':
  tfds.testing.test_main()
