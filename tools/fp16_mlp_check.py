"""Check a checkpoint's complete dense SwiGLU MLP with native FP16 instructions.

    python tools/fp16_mlp_check.py MODEL --layer 0 --rows 1 5 --output report.json

Uses deterministic synthetic FP32 activations and the checkpoint's full matrices;
this is a layer arithmetic check, not token inference or a board speed benchmark.
The ISA simulator's result must match ordered FP16/FP32 references bit for bit at
the up projection, activated gate and down output. RTL coverage is in
tests/test_fp16_tiled.py. No live board or model installation is changed.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from opentpu import fp16 as H, fp32 as F, language as ol, target
from opentpu.kernels.fp16 import swiglu
from opentpu.llm.qwen3 import load_weights
from opentpu.runtime import HalfWeight, Input, Output, compile_kernel, run


@ol.jit
def mlp(x, wg, wu, wd, gate, up, out):
    swiglu(x, wg, wu, wd, out, (gate, up))


def check(weights, rows, cfg, seed):
    wg, wu, wd = weights
    f, k = wg.shape
    n = wd.shape[0]
    x = np.random.default_rng(seed).standard_normal((rows, k), dtype=np.float32)
    start = time.monotonic()
    comp, images = compile_kernel(mlp, cfg, x=Input(x), wg=HalfWeight(wg),
                                  wu=HalfWeight(wu), wd=HalfWeight(wd),
                                  gate=Output((rows, f)), up=Output((rows, f)),
                                  out=Output((rows, n)))
    compiled = time.monotonic()
    got = run(comp, images).outputs
    simulated = time.monotonic()
    a = H.pack(x)
    g, u = H.matmul(a, H.pack(wg.T)), H.matmul(a, H.pack(wu.T))
    hidden = F.mul(F.silu(g, np.float32(-ol.LOG2E)), u)
    want = dict(gate=hidden, up=u, out=H.matmul(H.pack(hidden), H.pack(wd.T)))
    for name, expected in want.items():
        np.testing.assert_array_equal(F.bits(got[name]), F.bits(expected), err_msg=name)
        if not np.isfinite(got[name]).all():
            raise ValueError(f"{name}: nonfinite outputs (inspect binary16 range)")
    return dict(rows=rows, k=k, f=f, n=n, exact=True, finite=True,
                program_words=len(comp.programs[0]) * 8,
                compile_s=compiled - start, isa_s=simulated - compiled,
                reference_s=time.monotonic() - simulated,
                output_sha256=hashlib.sha256(got["out"].tobytes()).hexdigest())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model", type=Path)
    p.add_argument("--layer", type=int, default=0)
    p.add_argument("--rows", type=int, nargs="+", default=[1, 5])
    p.add_argument("--seed", type=int, default=80)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.layer < 0 or any(r <= 0 for r in a.rows):
        p.error("layer must be nonnegative and rows positive")
    W = load_weights(a.model)
    names = [f"model.layers.{a.layer}.mlp.{part}_proj.weight" for part in ("gate", "up", "down")]
    weights = [W[name] for name in names]
    f, k = weights[0].shape
    if weights[1].shape != (f, k) or weights[2].shape[1] != f:
        p.error("checkpoint must have compatible dense gate/up/down matrices")
    # Packed weights plus FP32 inputs, outputs and two scratch arrays, with space
    # for alignment. The memory size is independent of the physical board's HBM.
    size = 2 * sum(w.size for w in weights) + 4 * max(a.rows) * (k + 2 * f + weights[2].shape[0])
    size = -(-(size + (1 << 20)) // 4096) * 4096
    cfg = target.config("v80:fp16", TMEM_WORDS=4096, IMEM_WORDS=4096, DRAM_BYTES=size,
                        ACT_BLOCKS=4)
    report = dict(model=str(a.model.resolve()), layer=a.layer, seed=a.seed, backend="isa",
                  target="v80:fp16", tmem_words=cfg.TMEM_WORDS, imem_words=cfg.IMEM_WORDS,
                  inputs="synthetic FP32 normal activations", runs=[],
                  weights=[dict(name=name, shape=list(w.shape),
                                fp32_sha256=hashlib.sha256(w.tobytes()).hexdigest())
                           for name, w in zip(names, weights)])
    for rows in a.rows:
        result = check(weights, rows, cfg, a.seed)
        report["runs"].append(result)
        print(json.dumps(result), flush=True)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
