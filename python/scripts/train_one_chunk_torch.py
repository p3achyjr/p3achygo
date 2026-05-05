"""Task 7: e2e torch-native training run on a single chunk.

Loads a torch-native checkpoint, iterates batches from a TFRecord chunk,
runs forward + backward + optimizer.step, logs loss + ms/batch.

Loss subset for the benchmark (full-loss parity is a follow-up under
Task 6 once train_shim exposes a backend-agnostic train_step):
  policy CE + game-outcome CE + score-distribution CE
  + ownership BCE + Q-value MSE (q6/q16/q50)

Usage:
    P3ACHYGO_BACKEND=torch PYTHONPATH=python python python/scripts/train_one_chunk_torch.py \\
        --model_path /tmp/model_from_migration.pt \\
        --chunk ~/p3achygo-data/v4-models/example-chunks/chunk_0390.tfrecord.zz \\
        --batch_size 256 --lr 1e-4 --max_batches 200
"""

from __future__ import annotations

import argparse
import os
import time
from contextlib import nullcontext as _nullcontext

import torch
import torch.nn.functional as F


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--chunk", required=True)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--max_batches", type=int, default=200)
    p.add_argument("--device", default=("cuda" if torch.cuda.is_available() else "cpu"))
    p.add_argument(
        "--compile", action="store_true", help="apply torch.compile to the model"
    )
    p.add_argument(
        "--compile_mode",
        default="default",
        choices=("default", "reduce-overhead", "max-autotune"),
    )
    p.add_argument(
        "--cudnn_benchmark",
        action="store_true",
        help="set torch.backends.cudnn.benchmark=True (autotunes cudnn algo)",
    )
    p.add_argument(
        "--channels_last",
        action="store_true",
        help="convert model to channels_last memory format",
    )
    p.add_argument(
        "--amp",
        choices=("off", "fp16", "bf16", "auto"),
        default="auto",
        help="autocast dtype for forward (mixed precision); 'auto' "
        "selects fp16 if the GPU supports it, else bf16, else off",
    )
    args = p.parse_args()

    os.environ.setdefault("P3ACHYGO_BACKEND", "torch")

    if args.cudnn_benchmark:
        torch.backends.cudnn.benchmark = True

    from train_shim import load_model
    from backend_torch.optimizer import ConvMuon, build_convmuon_param_groups
    from backend_torch.dataset import ChunkDataset

    print(f"loading model from {args.model_path}")
    model = load_model(os.path.expanduser(args.model_path))
    model.to(args.device)
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)
    if args.compile:
        model = torch.compile(model, dynamic=False, mode=args.compile_mode)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params:,}")
    # Resolve "auto": prefer fp16 (broadest support, with grad scaler).
    # Fall back to bf16 if fp16 not viable (e.g. CPU). Fall through to off.
    amp_choice = args.amp
    if amp_choice == "auto":
        if args.device == "cuda":
            amp_choice = "fp16"
        elif (
            torch.cpu._is_avx512_bf16_supported()
            if hasattr(torch.cpu, "_is_avx512_bf16_supported")
            else False
        ):
            amp_choice = "bf16"
        else:
            amp_choice = "off"
    print(
        f"device: {args.device}  channels_last={args.channels_last}  "
        f"compile={args.compile}  amp={amp_choice}"
    )

    amp_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(amp_choice)
    scaler = torch.amp.GradScaler("cuda") if amp_choice == "fp16" else None

    groups, wd_factors = build_convmuon_param_groups(
        model,
        lr=args.lr,
        adam_lr_ratio=1.0,
        momentum=0.95,
        nesterov=True,
        ns_steps=5,
    )
    opt = ConvMuon(
        groups,
        wd_factors=wd_factors,
        weight_decay=0.02,
        adam_weight_decay=0.02,
        adam_lr_ratio=1.0,
        rms_rate=0.2,
    )

    chunk_path = os.path.expanduser(args.chunk)
    print(f"chunk: {chunk_path}")
    ds = ChunkDataset(chunk_path, args.batch_size, device=args.device)

    model.train()
    losses, times = [], []

    for i, batch in enumerate(ds):
        if i >= args.max_batches:
            break
        # 20-tuple from transforms.expand: see python/transforms.py:expand.
        (
            input_planes,
            input_global,
            color,
            komi,
            score,
            score_one_hot,
            policy,
            _policy_aux,
            _policy_aux_dist,
            _has_pi_aux,
            own,
            q6,
            q16,
            q50,
            _q6_score,
            _q16_score,
            _q50_score,
            game_outcome,
            _mcts_dist,
            _has_mcts,
        ) = batch

        input_planes = input_planes.float()
        input_global = input_global.float()

        if args.device == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()

        opt.zero_grad()
        autocast_ctx = (
            torch.amp.autocast(device_type="cuda", dtype=amp_dtype)
            if amp_dtype is not None
            else _nullcontext()
        )
        with autocast_ctx:
            out = model(input_planes, input_global, training=True)
            (
                pi_logits,
                _pi_probs,
                outcome_logits,
                _outcome_probs,
                own_pred,
                score_logits,
                _score_probs,
                _gamma,
                _pi_aux,
                q6_pred,
                q16_pred,
                q50_pred,
                _q6e,
                _q16e,
                _q50e,
                _q6sc,
                _q16sc,
                _q50sc,
                _q6sce,
                _q16sce,
                _q50sce,
                _pi_soft,
                _pi_opt,
                _mcts_logits,
                _mcts_probs,
            ) = out

            # ---- losses (subset) ----
            loss_pi = (
                -(policy.float() * F.log_softmax(pi_logits, dim=-1)).sum(-1).mean()
            )
            loss_outcome = F.cross_entropy(outcome_logits, game_outcome.float())
            loss_score = (
                -(score_one_hot.float() * F.log_softmax(score_logits, dim=-1))
                .sum(-1)
                .mean()
            )
            own_t = (own.float() + 1.0) * 0.5
            own_pred_flat = own_pred.reshape(own_pred.shape[0], -1)
            own_t_flat = own_t.reshape(own_t.shape[0], -1)
            if own_pred_flat.shape == own_t_flat.shape:
                loss_own = F.binary_cross_entropy_with_logits(own_pred_flat, own_t_flat)
            else:
                loss_own = torch.tensor(0.0, device=args.device)
            loss_q = (
                F.mse_loss(q6_pred.squeeze(-1), q6.float())
                + F.mse_loss(q16_pred.squeeze(-1), q16.float())
                + F.mse_loss(q50_pred.squeeze(-1), q50.float())
            )
            loss = (
                loss_pi
                + loss_outcome
                + 0.1 * loss_score
                + 0.1 * loss_own
                + 0.1 * loss_q
            )

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            opt.step()
        if args.device == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()

        losses.append(float(loss.item()))
        times.append(t1 - t0)

        if i % 10 == 0 or i < 5:
            print(
                f"step {i:4d}  loss={loss.item():.4f}  "
                f"pi={loss_pi.item():.4f}  out={loss_outcome.item():.4f}  "
                f"score={loss_score.item():.4f}  own={loss_own.item():.4f}  "
                f"q={loss_q.item():.4f}  ms={1000*(t1-t0):.1f}"
            )

    if not times:
        print("no batches iterated; aborting summary")
        return

    warmup = min(10, max(1, len(times) // 10))
    steady = times[warmup:]
    print()
    print(f"summary: total={len(times)} batches, warmup={warmup}")
    print(f"  loss[0]={losses[0]:.4f}  loss[-1]={losses[-1]:.4f}")
    print(
        f"  ms/batch (all): mean={1000*sum(times)/len(times):.1f}  "
        f"min={1000*min(times):.1f}  max={1000*max(times):.1f}"
    )
    if steady:
        print(
            f"  ms/batch (steady, skip warmup): "
            f"mean={1000*sum(steady)/len(steady):.1f}  "
            f"min={1000*min(steady):.1f}  max={1000*max(steady):.1f}"
        )


if __name__ == "__main__":
    main()
