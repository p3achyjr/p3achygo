"""Convert a BatchNorm nbt_attn checkpoint to norm_kind "fixscalefolded".

Every trunk BatchNorm becomes a FixScaleNorm with per-channel gamma, folded so
that its output equals the BN's eval-mode output:
  BN:   y = (x - mean) / sqrt(var + eps) * w + b
  new:  y = x * (gamma * scale) + beta,  gamma = w / sqrt(var + eps) / scale,
                                         beta  = b - mean * w / sqrt(var + eps)
`scale` takes KataGo's fixscale init values (block.init_fixscale). Head
BatchNorms and every other weight are copied unchanged.

Optimizer: Muon / Adam state of every non-norm param carries over by name;
the converted trunk norm params start with empty (fresh) Adam state.

Checks the converted model against the original in eval mode on --check_n
records of --check_path and prints the max abs difference of every output.

usage: PYTHONPATH=python python -m python.scripts.fold_bn_to_fixscale \
  --in_path=live_0880.pt --out_path=live_0880_folded.pt --check_path=val.tfrecord.zz
"""

import math
import sys

import numpy as np
import torch
from absl import app, flags

from backend_torch import dataset
from backend_torch.model_layers_common import NORM_FIXSCALE_FOLDED, FixScaleNorm
from backend_torch.model_utils import _model_from_config, load_with_optimizer
from backend_torch.optimizer import (
    _MUON_EXCLUDE_LAYERS_TORCH,
    build_convmuon_param_groups,
)

FLAGS = flags.FLAGS
flags.DEFINE_string("in_path", "", "BatchNorm checkpoint (.pt).")
flags.DEFINE_string("out_path", "", "Output checkpoint (.pt).")
flags.DEFINE_string("check_path", "", "tfrecord.zz used for the output check.")
flags.DEFINE_integer("check_n", 256, "Records used for the output check.")


def _param_order(model, num_saved_groups):
    """Params in the positional order the saved optimizer state uses (the
    trainer's build_convmuon_param_groups partition; a third group exists
    only when head_lr_factor was set)."""
    head_lr_factor = 1.0 if num_saved_groups == 3 else None
    groups, _ = build_convmuon_param_groups(
        model, head_lr_factor=head_lr_factor, exclude_layers=_MUON_EXCLUDE_LAYERS_TORCH
    )
    names = {id(p): n for n, p in model.named_parameters()}
    return [[names[id(p)] for p in g["params"]] for g in groups]


def convert_optimizer(opt_state, old, new, folded_prefixes):
    inner = opt_state["inner"]
    saved_groups = inner["param_groups"]
    old_order = _param_order(old, len(saved_groups))
    assert [len(g) for g in old_order] == [
        len(g["params"]) for g in saved_groups
    ], "reconstructed old param partition does not match the saved optimizer"
    old_flat = [n for g in old_order for n in g]
    saved_by_name = {old_flat[i]: s for i, s in inner["state"].items()}

    new_order = _param_order(new, len(saved_groups))
    assert len(new_order) == len(saved_groups)
    state, groups, j, reset = {}, [], 0, 0
    for names, saved_group in zip(new_order, saved_groups):
        idx = []
        for n in names:
            if n.rsplit(".", 1)[0] in folded_prefixes:
                reset += 1  # fresh Adam state for converted norm params
            else:
                state[j] = saved_by_name[n]
            idx.append(j)
            j += 1
        groups.append({**saved_group, "params": idx})
    carried = len(state)
    assert carried == len(saved_by_name) - sum(
        1 for n in saved_by_name if n.rsplit(".", 1)[0] in folded_prefixes
    )
    print(f"optimizer: {carried} param states carried over, {reset} norm params fresh")
    return {
        "inner": {"state": state, "param_groups": groups},
        "scaler": opt_state["scaler"],
    }


def main(_):
    old, opt_state = load_with_optimizer(FLAGS.in_path)
    old.eval()
    config = old.config_dict()
    assert config["generic_arch"].get("norm_kind", "bnorm") == "bnorm"
    config["generic_arch"] = {
        **config["generic_arch"],
        "norm_kind": NORM_FIXSCALE_FOLDED,
    }
    new = _model_from_config(config)
    new.eval()

    with torch.no_grad():
        # KataGo fixscale scales (also re-inits conv weights; overwritten below).
        for i, block in enumerate(new.blocks):
            block.init_fixscale(1.0 / math.sqrt(i + 1.0))

    old_sd, new_sd = old.state_dict(), new.state_dict()
    old_mods = dict(old.named_modules())
    folded = {n for n, m in new.named_modules() if isinstance(m, FixScaleNorm)}
    assert folded and all(n.startswith("blocks.") for n in folded)
    assert all(isinstance(old_mods[n], torch.nn.BatchNorm2d) for n in folded)

    out = {}
    with torch.no_grad():
        for n in folded:
            bn = old_mods[n]
            a = bn.weight / torch.sqrt(bn.running_var + bn.eps)
            scale = new_sd[f"{n}.scale"]
            out[f"{n}.gamma"] = a / scale
            out[f"{n}.beta"] = bn.bias - bn.running_mean * a
            out[f"{n}.scale"] = scale
    folded_old_keys = {
        f"{n}.{s}"
        for n in folded
        for s in (
            "weight",
            "bias",
            "running_mean",
            "running_var",
            "num_batches_tracked",
        )
    }
    for k, v in old_sd.items():
        if k in folded_old_keys:
            continue
        assert k in new_sd and new_sd[k].shape == v.shape, k
        out[k] = v
    assert set(out) == set(new_sd), set(out) ^ set(new_sd)
    new.load_state_dict(out)
    print(f"folded {len(folded)} trunk norms; {len(out)} tensors in new state dict")

    # Output check (eval mode, float32, CPU).
    recs = []
    for r in dataset.iter_records(FLAGS.check_path):
        recs.append((r[0], r[1]))
        if len(recs) >= FLAGS.check_n:
            break
    b = torch.tensor(np.stack([x[0] for x in recs]))
    g = torch.tensor(np.stack([x[1] for x in recs]))
    with torch.no_grad():
        ya, yb = old.float()(b, g), new.float()(b, g)
    worst = 0.0
    for k, (u, v) in enumerate(zip(ya, yb)):
        if u is None:
            continue
        d = (u - v).abs().max().item()
        ref = u.abs().max().item()
        worst = max(worst, d / max(ref, 1e-6))
        print(f"output {k:2d} {tuple(u.shape)}: max|diff| {d:.3e}  max|orig| {ref:.3e}")
    print(f"worst relative diff {worst:.3e}")

    blob = {"model": new.state_dict(), "config": new.config_dict()}
    if opt_state is not None:
        blob["optimizer"] = convert_optimizer(opt_state, old, new, folded)
    torch.save(blob, FLAGS.out_path)
    print(f"wrote {FLAGS.out_path}")


if __name__ == "__main__":
    sys.stdout.reconfigure(line_buffering=True)
    app.run(main)
