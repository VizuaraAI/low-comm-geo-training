"""Checks that need no GPU and no data download. Run with: python tests/test_core.py

They use a tiny model on random tokens, so they test the arithmetic and the controls and say nothing about training."""
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import islands as I  # noqa: E402


def tiny(torch, H, rule, opt="adamw", seed=1, rounds=3, bug=False, val_fn=None):
    I.MODEL.update(n_layer=2, n_head=2, n_embd=32, block=16, vocab=200)
    I.BATCH = 4
    I.WARMUP_TOKENS = 1000
    rng = np.random.default_rng(0)
    data = I.Data(torch, "cpu", rng.integers(0, 200, 40_000).astype(np.int32), rng.integers(0, 200, 2_000).astype(np.int32))
    with tempfile.TemporaryDirectory() as out:
        return I.run_islands(torch, data, "t", opt, rule, H, 1e-3, 1e-3, 1.0, 0.0, seed, 10**9, 10**9, out, {}, speeds=[1.0] * 8,
                             device="cpu", max_rounds=rounds, inject_view_bug=bug, val_fn=val_fn)


def main():
    torch = I.setup_torch()
    I.selftest(torch)
    for opt in I.OPTS:
        eq = [10] * 8
        a = tiny(torch, eq, "R1", opt)["theta_sha256"]
        assert a == tiny(torch, eq, "R2", opt)["theta_sha256"], "R2 must equal R1 at equal speeds"
        assert a == tiny(torch, eq, "R3a", opt)["theta_sha256"], "R3a must equal R2 at equal speeds"
        assert a != tiny(torch, eq, "R3b", opt)["theta_sha256"], "R3b must differ: its weights are in use"
        assert a != tiny(torch, eq, "R1", opt, seed=2)["theta_sha256"], "seeds must differ"
        het = [4, 4, 6, 6, 13, 13, 30, 30]
        assert tiny(torch, het, "R2", opt)["theta_sha256"] != tiny(torch, het, "R3a", opt)["theta_sha256"], "R3a must differ on a mixed fleet"
    try:
        tiny(torch, [10] * 8, "R2", bug=True)
    except AssertionError as e:
        assert "zero Delta" in str(e)
    else:
        raise AssertionError("the injected view bug must trip the zero-Delta check")
    print("all checks passed")


if __name__ == "__main__":
    main()
