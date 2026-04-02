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
    w_q6: float  # q 6 moves later
    w_q16: float  # q 16 moves later
    w_q50: float  # q 50 moves later
    w_gamma: float  # gamma
    # v1 weights
    w_q_err: float  # weight for q error predictions (12-14)
    w_q_score: float  # weight for q score predictions (15-17)
    w_q_score_err: float  # weight for q score error predictions (18-20)
    w_pi_soft: float  # weight for soft policy (21)
    w_pi_optimistic: float  # weight for optimistic policy (22)

    @staticmethod
    def SLCoeffs():
        return LossCoeffs(1.0, 0.15, 1.0, 1.5, 0.02, 0, 0, 0, 0, 0.005, 0, 0, 0, 0, 0)

    @staticmethod
    def RLCoeffs():
        return LossCoeffs(
            1.0,
            0.06,  # policy aux
            1.0,
            1.5,
            0.02,  # score
            0.45,  # own
            0.7,  # q6
            0.4,  # q16
            0.3,  # q50
            0.005,  # gamma
            3.0,  # q err
            0.2,  # short-term score
            0.2,  # short-term score err
            4.0,  # soft policy
            1.0,  # optimistic policy
        )
