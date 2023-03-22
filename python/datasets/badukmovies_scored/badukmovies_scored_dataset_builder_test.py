"""badukmovies_scored_games dataset."""

import tensorflow_datasets as tfds
from . import badukmovies_scored_dataset_builder


class BadukmoviesScoredGamesTest(tfds.testing.DatasetBuilderTestCase):
  """Tests for badukmovies_scored_games dataset."""
  # TODO(badukmovies_scored_games):
  import datasets.common.constants as constants
  DATASET_CLASS = badukmovies_scored_dataset_builder.Builder
  SPLITS = {
      'train': 2176 * constants.REUSE_FACTOR /
               8,  # Number of fake train example
  }


if __name__ == '__main__':
  tfds.testing.test_main()
