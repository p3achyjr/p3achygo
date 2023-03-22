"""badukmovies dataset."""

from __future__ import annotations

import tensorflow_datasets as tfds
from . import badukmovies_dataset_builder


class BadukmoviesTest(tfds.testing.DatasetBuilderTestCase):
  """Tests for badukmovies dataset."""
  import datasets.common.constants as constants
  DATASET_CLASS = badukmovies_dataset_builder.Builder
  SPLITS = {
      'train': 4360 * constants.REUSE_FACTOR /
               8,  # Number of fake train example
  }


if __name__ == '__main__':
  tfds.testing.test_main()
