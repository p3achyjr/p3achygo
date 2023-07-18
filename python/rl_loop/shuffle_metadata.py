'''
Helper functions to calculate training sample size, sample draw probability, and
number of generations to train past the end of training for.
'''

from __future__ import annotations


def training_window_size(num_samples_generated: int,
                         num_samples_offset: int) -> int:
  '''
  Return number of samples to build next chunk from. Follows Katago formula,
  but with bigger sample window and steeper growth.

  Formula is:

  c * (1 + beta * ((n/c)^a - 1) / a)
  '''

  alpha = 0.75
  beta = 0.65
  c = 250000
  min_window = 100000

  num_samples = num_samples_generated + num_samples_offset
  mult = (num_samples / c)**alpha - 1
  mult = beta * mult / alpha
  mult = mult + 1

  return int(max(min_window, mult * c))


def select_sample_probability(training_window_size: int, games_per_gen: int,
                              k: int) -> float:
  '''
  Return probability with which we draw each individual sample.

  Does so via a rough estimate of how many generations each individual sample
  lives in the training window. We calculate this by calculating (roughly) how
  many generations each sample lives in the training window for, and
  multiplying by some constant `k` (so each sample is drawn `k` times on average).

  Formula:

  samples_per_gen = games_per_gen * samples_per_game
  generation_window = training_window_size / samples_per_gen
  p = k / generation_window

  We clamp this probability to .15 to minimize the chance of value overfitting.
  '''
  samples_per_game_est = 75
  samples_per_gen = games_per_gen * samples_per_game_est
  generation_window = training_window_size / samples_per_gen
  p = k / generation_window

  return min(p, .15)
