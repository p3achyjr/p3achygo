"""Numerical parity test: keras `compute_losses` vs torch native port.

Runs both implementations on the same synthetic batch and asserts the 16
loss outputs agree within tolerance.

Requires `KERAS_BACKEND=torch` so that `keras.losses.*` and `keras.ops.*`
both dispatch to torch tensors. The torch port is in
`backend_torch/losses.compute_losses`.

Tolerances:
  fp32: atol=1e-5, rtol=1e-3
  fp16: atol=1e-3, rtol=1e-2

Run:
  KERAS_BACKEND=torch python python/test/torch_loss_parity_test.py
"""

from __future__ import annotations

import os
import sys
import types
import unittest

import torch

# This test only meaningfully runs when keras is configured to use torch as
# its backend (so keras.ops dispatches on torch tensors). Refuse to run
# under any other backend rather than silently passing.
if os.environ.get("KERAS_BACKEND") != "torch":
    print(
        "torch_loss_parity_test: requires KERAS_BACKEND=torch; got "
        f"{os.environ.get('KERAS_BACKEND')!r}",
        file=sys.stderr,
    )
    sys.exit(0)

import keras  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from model import LossWeights  # noqa: E402
from backend_torch.train import GroundTruth, ModelPredictions  # noqa: E402
from backend_torch.losses import compute_losses as torch_compute_losses  # noqa: E402


# ---- keras-side compute_losses, run via a mock object ---------------------

# We don't want to instantiate a full P3achyGoModel just to run the loss
# math. Instead, replicate just the loss-instance attributes and bind the
# unbound `compute_losses` / `v1_loss_terms` methods to a stand-in object.

from model import P3achyGoModel  # noqa: E402


def _make_keras_loss_host():
    host = types.SimpleNamespace()
    host.scce_logits = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    host.scce_logits_per_example = keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction="none"
    )
    host.cce_logits = keras.losses.CategoricalCrossentropy(from_logits=True)
    host.mse = keras.losses.MeanSquaredError()
    host.huber = keras.losses.Huber()
    # Bind the unbound methods so they see our attributes.
    host.v1_loss_terms = types.MethodType(P3achyGoModel.v1_loss_terms, host)
    host.compute_losses = types.MethodType(P3achyGoModel.compute_losses, host)
    return host


# ---- synthetic batch builder ----------------------------------------------

NUM_MOVES = 362  # 19*19 + pass
NUM_OUTCOMES = 2  # B win / W win
SCORE_RANGE = 800
NUM_OWN = 361
NUM_V_BUCKETS = 51


def _make_batch(batch: int = 8, dtype=torch.float32, seed: int = 0):
    # keras-on-torch's loss objects move ops to CUDA when available; build
    # tensors on the same device so the keras path doesn't see a mix.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    g = torch.Generator(device="cpu").manual_seed(seed)
    rand = lambda *s: torch.randn(*s, generator=g, dtype=dtype).to(device)
    randu = lambda *s: torch.rand(*s, generator=g, dtype=dtype).to(device)

    # Build legal probability dists (softmax of random logits) for KLD targets.
    def softmax_dist(*shape):
        return torch.softmax(rand(*shape), dim=-1).to(dtype)

    pi_logits = rand(batch, NUM_MOVES)
    pi_logits_aux = rand(batch, NUM_MOVES)
    game_outcome_logits = rand(batch, NUM_OUTCOMES)
    score_logits = rand(batch, SCORE_RANGE)
    own_pred = rand(batch, NUM_OWN, 1)
    q6 = rand(batch)
    q16 = rand(batch)
    q50 = rand(batch)
    gamma = rand(batch, 1)
    # v1
    q6_err = randu(batch).abs() + 1e-3
    q16_err = randu(batch).abs() + 1e-3
    q50_err = randu(batch).abs() + 1e-3
    q6_score = rand(batch) * 5.0
    q16_score = rand(batch) * 5.0
    q50_score = rand(batch) * 5.0
    q6_score_err = randu(batch).abs() + 1e-3
    q16_score_err = randu(batch).abs() + 1e-3
    q50_score_err = randu(batch).abs() + 1e-3
    pi_logits_soft = rand(batch, NUM_MOVES)
    pi_logits_optimistic = rand(batch, NUM_MOVES)
    mcts_dist_logits = rand(batch, NUM_V_BUCKETS)
    mcts_dist_probs = softmax_dist(batch, NUM_V_BUCKETS)

    predictions = ModelPredictions(
        pi_logits=pi_logits,
        pi_logits_aux=pi_logits_aux,
        game_outcome=game_outcome_logits,
        score_logits=score_logits,
        own_pred=own_pred,
        q6_pred=q6,
        q16_pred=q16,
        q50_pred=q50,
        gamma=gamma,
        q6_err_pred=q6_err,
        q16_err_pred=q16_err,
        q50_err_pred=q50_err,
        q6_score_pred=q6_score,
        q16_score_pred=q16_score,
        q50_score_pred=q50_score,
        q6_score_err_pred=q6_score_err,
        q16_score_err_pred=q16_score_err,
        q50_score_err_pred=q50_score_err,
        pi_logits_soft=pi_logits_soft,
        pi_logits_optimistic=pi_logits_optimistic,
        mcts_dist_logits=mcts_dist_logits,
        mcts_dist_probs=mcts_dist_probs,
    )

    # Targets
    policy = softmax_dist(batch, NUM_MOVES).to(torch.float32)
    policy_aux_idx = torch.randint(
        0, NUM_MOVES, (batch,), generator=g, dtype=torch.int64
    ).to(device)
    score_idx = torch.randint(0, SCORE_RANGE, (batch,), generator=g)
    score_one_hot = torch.zeros(batch, SCORE_RANGE, dtype=torch.float32)
    score_one_hot[torch.arange(batch), score_idx] = 1.0
    score_one_hot = score_one_hot.to(device)
    score = score_idx.float().to(device)
    game_outcome_idx = torch.randint(0, NUM_OUTCOMES, (batch,), generator=g)
    game_outcome_oh = torch.zeros(batch, NUM_OUTCOMES, dtype=torch.float32)
    game_outcome_oh[torch.arange(batch), game_outcome_idx] = 1.0
    game_outcome_oh = game_outcome_oh.to(device)
    own = rand(batch, NUM_OWN).clamp(-1, 1).to(torch.float32)

    # Mix of has-aux-dist examples to exercise both per-example branches
    has_pi_aux_dist = (torch.arange(batch) % 2 == 0).to(torch.float32).to(device)
    policy_aux_dist = softmax_dist(batch, NUM_MOVES).to(torch.float32)

    has_mcts_value_dist = (torch.arange(batch) % 2 == 1).to(torch.float32).to(device)
    mcts_value_dist = torch.randint(
        0, 100, (batch, NUM_V_BUCKETS), generator=g, dtype=torch.int32
    ).to(device)

    targets = GroundTruth(
        policy=policy,
        policy_aux=policy_aux_idx,
        score=score,
        score_one_hot=score_one_hot,
        game_outcome=game_outcome_oh,
        own=own,
        q6=q6.clone().detach(),
        q16=q16.clone().detach(),
        q50=q50.clone().detach(),
        q6_score=q6_score.clone().detach(),
        q16_score=q16_score.clone().detach(),
        q50_score=q50_score.clone().detach(),
        policy_aux_dist=policy_aux_dist,
        has_pi_aux_dist=has_pi_aux_dist,
        mcts_value_dist=mcts_value_dist,
        has_mcts_value_dist=has_mcts_value_dist,
    )

    weights = LossWeights(
        w_pi=1.0,
        w_pi_aux=0.15,
        w_val=1.5,
        w_outcome=1.5,
        w_score=0.02,
        w_own=0.05,
        w_q6=0.5,
        w_q16=0.5,
        w_q50=0.5,
        w_gamma=0.0005,
        w_q_err=0.4,
        w_q_score=0.05,
        w_q_score_err=0.05,
        w_pi_soft=0.05,
        w_pi_optimistic=0.05,
        w_mcts_dist=0.5,
    )

    return predictions, targets, weights


_LABELS = [
    "loss",
    "policy_loss",
    "policy_aux_dist_loss",
    "policy_aux_scalar_loss",
    "outcome_loss",
    "q6_loss",
    "q16_loss",
    "q50_loss",
    "score_pdf_loss",
    "score_cdf_loss",
    "own_loss",
    "q_err_loss",
    "q_score_loss",
    "q_score_err_loss",
    "pi_soft_loss",
    "pi_optimistic_loss",
    "mcts_dist_loss",
]


class TorchLossParity(unittest.TestCase):
    def _run_pair(self, dtype, atol, rtol, seed=0):
        predictions, targets, weights = _make_batch(batch=8, dtype=dtype, seed=seed)

        host = _make_keras_loss_host()
        keras_outs = host.compute_losses(predictions, targets, weights)
        torch_outs = torch_compute_losses(predictions, targets, weights)

        self.assertEqual(len(keras_outs), len(torch_outs))
        self.assertEqual(len(keras_outs), len(_LABELS))

        for label, k, t in zip(_LABELS, keras_outs, torch_outs):
            k_t = k if isinstance(k, torch.Tensor) else torch.tensor(float(k))
            t_t = t if isinstance(t, torch.Tensor) else torch.tensor(float(t))
            with self.subTest(loss=label, dtype=str(dtype)):
                torch.testing.assert_close(
                    t_t.detach().float(),
                    k_t.detach().float(),
                    atol=atol,
                    rtol=rtol,
                    msg=f"{label}: torch={float(t_t):.6f} keras={float(k_t):.6f}",
                )

    def test_fp32_parity(self):
        self._run_pair(torch.float32, atol=1e-5, rtol=1e-3)

    def test_fp16_parity(self):
        self._run_pair(torch.float16, atol=1e-3, rtol=1e-2)


if __name__ == "__main__":
    unittest.main()
