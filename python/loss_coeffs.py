from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LossCoeffs:
  w_pi: float  # policy
  w_val: float  # val-related (outcome, score_pdf, own) weighting.
  w_outcome: float  # game outcome
  w_score: float  # score
  w_own: float  # own
  w_gamma: float  # gamma

  @staticmethod
  def SLCoeffs():
    return LossCoeffs(1.0, 0.01, 1.0, 0.02, 0, 0.0005)

  @staticmethod
  def RLCoeffs():
    return LossCoeffs(1.0, 1.0, 1.0, 0.02, 0.004, 0.0005)
