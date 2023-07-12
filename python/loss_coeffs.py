from __future__ import annotations

from dataclasses import dataclass


@dataclass
class LossCoeffs:
  w_pi: float  # policy
  w_pi_aux: float  # policy for next player
  w_val: float  # val-related (outcome, score_pdf, own) weighting.
  w_outcome: float  # game outcome
  w_score: float  # score
  w_own: float  # own
  w_q30: float  # q 30 moves later
  w_q100: float  # q 100 moves later
  w_q200: float  # q 200 moves later
  w_gamma: float  # gamma

  @staticmethod
  def SLCoeffs():
    return LossCoeffs(1.0, .15, 0.02, 1.0, 0.02, 0, 0, 0, 0, 0.0005)

  @staticmethod
  def RLCoeffs():
    return LossCoeffs(1.0, .15, 1.0, 1.0, 0.02, 0.45, .05, .05, .05, 0.0005)
