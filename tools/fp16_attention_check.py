"""Check Qwen3.5's complete native FP16 attention sublayer with checkpoint weights.

    PYTHONPATH=. python tools/fp16_attention_check.py MODEL --layer 3 --output report.json

Runs two consecutive chunks of 1 and 3 synthetic activation rows. Tests residual
outputs and every packed KV cache word against the ordered FP16/FP32 reference.
This is ISA simulation, not full-model inference or a board throughput benchmark.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from opentpu import fp32 as F, target
from opentpu.llm.qwen3 import load_weights
from opentpu.llm.qwen35 import Spec
from opentpu.llm.qwen35_fp16 import AttentionImage, attention_reference
from opentpu.runtime import run


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model", type=Path)
    p.add_argument("--layer", type=int, default=3)
    p.add_argument("--rows", type=int, nargs="+", default=[1, 3])
    p.add_argument("--seed", type=int, default=80)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if any(r <= 0 for r in a.rows):
        p.error("rows must be positive")
    W, spec = load_weights(a.model), Spec.from_hf(a.model)
    cfg = target.config("v80:fp16", TMEM_WORDS=16384, IMEM_WORDS=32768,
                        DRAM_BYTES=64 << 20, ACT_BLOCKS=4)
    prefix = f"model.layers.{a.layer}."
    names = [prefix + "input_layernorm.weight"] + [prefix + "self_attn." + n + ".weight"
             for n in ("q_proj", "k_proj", "v_proj", "o_proj", "q_norm", "k_norm")]
    report = dict(model=str(a.model.resolve()), layer=a.layer, seed=a.seed, backend="isa",
                  target="v80:fp16", inputs="synthetic FP32 normal activations",
                  tmem_words=cfg.TMEM_WORDS, imem_words=cfg.IMEM_WORDS, runs=[],
                  weights=[dict(name=n, shape=list(W[n].shape),
                                fp32_sha256=hashlib.sha256(W[n].tobytes()).hexdigest()) for n in names])
    for rows in a.rows:
        cap = max(8, 2 * rows)
        img = AttentionImage(spec, W, cfg, a.layer, cap, rows)
        keys = np.zeros((spec.n_kv, cap, spec.head_dim), np.uint16)
        values, images = keys.copy(), img.images
        rng = np.random.default_rng(a.seed)
        for pos in (0, rows):
            x = rng.standard_normal((rows, spec.hidden), dtype=np.float32)
            start = time.monotonic()
            comp = img.compile(pos)
            img.inputs(images, x, pos)
            built = time.monotonic()
            result = run(comp, images)
            simulated = time.monotonic()
            want, keys, values = attention_reference(spec, W, x, a.layer, pos, keys, values)
            np.testing.assert_array_equal(F.bits(result.outputs["out"]), F.bits(want))
            if not np.isfinite(want).all():
                raise ValueError("nonfinite attention output")
            images = result.drams
            for name, expected in (("kt", keys.transpose(0, 2, 1)), ("vc", values)):
                addr = img.layout.params[name][1]
                raw = images[0][addr:addr + 2 * expected.size].view("<u2").reshape(expected.shape)
                np.testing.assert_array_equal(raw, expected)
            record = dict(rows=rows, pos=pos, exact=True, cache_exact=True, finite=True,
                          program_words=len(comp.programs[0]) * 8, compile_s=built - start,
                          isa_s=simulated - built, reference_s=time.monotonic() - simulated,
                          output_sha256=hashlib.sha256(want.tobytes()).hexdigest())
            report["runs"].append(record)
            print(json.dumps(record), flush=True)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
