"""Resurrect dead FFN hidden units in a p3achygo torch checkpoint (port of KataGo resurrect_dead_channels.py, FFN part).

Unit j of a SwiGLU FFN (`down(silu(gate x) * (up x))`) is dead when its `ffn_down` column is near zero, or when both its
`ffn_up` and `ffn_gate` rows are near zero. Near zero = norm <= dead_frac * reference norm (median of channels above 0.1%
of the largest, KataGo `Metrics.reference_channel_norm`). Resurrection: up/gate rows <- noisy copy of a random fully live
unit's rows (Gaussian noise with std noise_frac * rms(src), rescaled to the src norm, times copy_scale); down column <- 0,
so the model function is unchanged.
The same edits (same source units and noise) are applied to every checkpoint given, e.g. the live model and the SWA model.
Optimizer state is kept as is.

Usage: python -m scripts.resurrect_ffn --inputs live_in.pt,swa_in.pt --outputs live_out.pt,swa_out.pt --mask_out revived.pt
"""

import argparse, torch


def ref_norm(n):
    live = n[n > 1e-3 * n.max()]
    return live.median() if live.numel() else torch.zeros(())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--outputs", required=True)
    ap.add_argument("--mask_out", required=True)
    ap.add_argument("--dead_frac", type=float, default=0.01)
    ap.add_argument("--noise_frac", type=float, default=1.0)
    ap.add_argument("--copy_scale", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    ins, outs = a.inputs.split(","), a.outputs.split(",")
    assert len(ins) == len(outs)
    ckpts = [torch.load(p, map_location="cpu", weights_only=False) for p in ins]
    ref = ckpts[0]["model"]
    gen = torch.Generator().manual_seed(a.seed)
    prefixes = sorted(
        k[: -len("ffn_up.weight")] for k in ref if k.endswith("ffn_up.weight")
    )
    masks, tot_dead, tot = {}, 0, 0
    for pre in prefixes:
        up, gate, down = (
            ref[pre + "ffn_up.weight"].float(),
            ref[pre + "ffn_gate.weight"].float(),
            ref[pre + "ffn_down.weight"].float(),
        )
        nu, ng, nd = up.norm(dim=1), gate.norm(dim=1), down.norm(dim=0)
        du, dg, dd = (
            nu <= a.dead_frac * ref_norm(nu),
            ng <= a.dead_frac * ref_norm(ng),
            nd <= a.dead_frac * ref_norm(nd),
        )
        dead = dd | (du & dg)
        live = ~(dd | du | dg)
        dead_idx, live_idx = dead.nonzero().flatten(), live.nonzero().flatten()
        src = live_idx[
            torch.randint(live_idx.numel(), (dead_idx.numel(),), generator=gen)
        ]
        new_rows = {}
        for name, w in (("ffn_up.weight", up), ("ffn_gate.weight", gate)):
            s = w[src]  # (n_dead, in)
            rms = s.norm(dim=1, keepdim=True) / s.shape[1] ** 0.5
            out = s + torch.randn(s.shape, generator=gen) * (a.noise_frac * rms)
            new_rows[name] = (
                out
                * (s.norm(dim=1, keepdim=True) / out.norm(dim=1, keepdim=True))
                * a.copy_scale
            )
        for c in ckpts:
            sd = c["model"]
            for name, rows in new_rows.items():
                sd[pre + name][dead_idx] = rows.to(sd[pre + name].dtype)
            sd[pre + "ffn_down.weight"][:, dead_idx] = 0.0
        masks[pre] = dead
        tot_dead += int(dead.sum())
        tot += dead.numel()
        print(f"{pre}: resurrected {int(dead.sum())}/{dead.numel()}")
    print(f"total resurrected {tot_dead}/{tot} = {tot_dead / tot:.1%}")
    for c, p in zip(ckpts, outs):
        torch.save(c, p)
    torch.save(masks, a.mask_out)


if __name__ == "__main__":
    main()
