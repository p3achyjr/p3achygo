"""KataGo LR warmup schedule."""

import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
os.environ.setdefault("P3ACHYGO_BACKEND", "torch")
from rl_loop.train import KataGoWarmupLRSchedule, katago_warmup_scale  # noqa: E402


def test_scale_steps():
    assert katago_warmup_scale(0) == 1 / 20
    assert katago_warmup_scale(249_999) == 1 / 20
    assert katago_warmup_scale(250_000) == 1 / 14
    assert katago_warmup_scale(1_999_999) == 1 / 1.4
    assert katago_warmup_scale(2_000_000) == 1.0


def test_schedule_counts_samples_from_batch_num():
    s = KataGoWarmupLRSchedule(1e-4, batch_num=976, batch_size=256)  # 249,856 samples
    assert math.isclose(s(123), 1e-4 / 20)  # step0 unset: batch_num only
    s.step0 = 500  # optimizer's counter at the start of this gen
    assert math.isclose(s(500), 1e-4 / 20)
    assert math.isclose(s(501), 1e-4 / 14)  # 977 * 256 = 250,112
    assert math.isclose(
        KataGoWarmupLRSchedule(1e-4, 7813, 256)(None), 1e-4
    )  # 2,000,128


if __name__ == "__main__":
    test_scale_steps()
    test_schedule_counts_samples_from_batch_num()
    print("ok")
