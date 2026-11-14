"""Qualify native FP16 or explicit mixed-precision Qwen3.5 against the ordered reference.

    PYTHONPATH=. python tools/fp16_model_check.py MODEL --output report.json
    PYTHONPATH=. python tools/fp16_model_check.py MODEL --backend board --dev qdma:BDF@Q --output report.json

Checks every layer residual, all recurrent/KV state and full logits over consecutive
tokens. The reference is an independent host oracle; the engine runs all model math
on the selected backend. --precision/--precisions select mixed matrix modes; the
default remains FP16. A board run requires an already programmed compatible overlay.
"""
import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from opentpu import fp32 as F, target
from opentpu.host.chat import make_backend
from opentpu.llm.qwen3 import load_weights
from opentpu.llm.qwen35 import Spec, reference_logits
from opentpu.llm.qwen35_fp16 import Engine, Reference
from opentpu.llm.precision import MODES


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("model", type=Path)
    p.add_argument("--backend", choices=("isa", "board"), default="isa")
    p.add_argument("--dev")
    p.add_argument("--tokens", type=int, nargs="+", default=[198, 17])
    p.add_argument("--cap", type=int, default=8)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--precision", choices=MODES)
    p.add_argument("--precisions")
    p.add_argument("--head-precision", choices=MODES)
    p.add_argument("--w4-group", type=int, choices=(8, 16, 32, 64, 128), default=128)
    a = p.parse_args()
    W, spec = load_weights(a.model), Spec.from_hf(a.model)
    backend, cfg = make_backend(a.backend, spec, a.cap, a.dev, a.model.name, wformat="fp16")
    cfg = cfg or target.config("v80:mixed" if a.precision else "v80:fp16", DRAM_BYTES=4 << 30)
    start = time.monotonic()
    eng = Engine(spec, W, cfg=cfg, cap=a.cap, backend=backend, precision=a.precision,
                 precisions=a.precisions, head_precision=a.head_precision, group=a.w4_group)
    report = dict(model=str(a.model.resolve()), backend=a.backend, cap=a.cap,
                  precision=a.precision, rules=eng.image.formats, head=eng.image.head_format, group=a.w4_group,
                  image_bytes=eng.image.nbytes, load_s=time.monotonic() - start, runs=[])
    print(json.dumps(report), flush=True)
    ref = Reference(spec, W, a.cap, policy=eng.image.policy, D=cfg.D, group=a.w4_group)
    run, stages, residuals = eng.backend.run, [0], []

    def observe(programs):
        result = run(programs)
        stage = stages[0]
        if stage < 2 * len(spec.kinds) and stage % 2:
            addr = eng.image.layout.params["x"][1]
            residuals.append(eng.backend.read(0, addr, 4 * spec.hidden).view("<f4").copy())
            print(json.dumps(dict(pos=eng.pos, layer=stage // 2, stage="device", stats=result)), flush=True)
        stages[0] += 1
        return result

    eng.backend.run = observe
    try:
        for token in a.tokens:
            stages[0], residuals[:] = 0, []
            start = time.monotonic()
            got = eng.step(token)
            device_s = time.monotonic() - start

            def check_layer(i, x):
                np.testing.assert_array_equal(F.bits(residuals[i]), F.bits(x[0]), err_msg=f"layer {i}")
                print(json.dumps(dict(pos=eng.pos - 1, layer=i, exact=True)), flush=True)

            want = ref.step(token, check_layer)
            np.testing.assert_array_equal(F.bits(got), F.bits(want))
            for i, kind in enumerate(spec.kinds):
                state = ref.states[i]
                names = ("kt", "vc") if kind == "attn" else ("ring", "state")
                for name, expected in zip(names, state):
                    if name == "kt":
                        expected = expected.transpose(0, 2, 1).copy()
                    addr = eng.image.layout.params[f"layer{i}.{name}"][1]
                    raw = eng.backend.read(0, addr, expected.nbytes)
                    np.testing.assert_array_equal(raw, np.ascontiguousarray(expected).view(np.uint8).ravel(),
                                                  err_msg=f"layer {i} {name}")
            record = dict(pos=eng.pos - 1, token=token, exact=True, states_exact=True,
                          finite=bool(np.isfinite(got).all()), argmax=int(got.argmax()),
                          logits_sha256=hashlib.sha256(got.tobytes()).hexdigest(),
                          device_s=device_s, reference_s=time.monotonic() - start - device_s,
                          stats=eng.stats[-1])
            original = reference_logits(spec, W, a.tokens[:eng.pos])[-1].astype(np.float64)
            actual = got.astype(np.float64)
            record["fp32_comparison"] = dict(
                max_abs=float(np.max(np.abs(actual - original))),
                rms=float(np.sqrt(np.mean((actual - original) ** 2))),
                cosine=float(actual @ original / (np.linalg.norm(actual) * np.linalg.norm(original))),
                argmax=int(original.argmax()), same_argmax=bool(got.argmax() == original.argmax()))
            report["runs"].append(record)
            a.output.parent.mkdir(parents=True, exist_ok=True)
            np.save(a.output.with_name(a.output.stem + f"-pos{eng.pos - 1}.npy"), got)
            report["complete"] = eng.pos == len(a.tokens)
            a.output.write_text(json.dumps(report, indent=2) + "\n")
            print(json.dumps(record), flush=True)
    finally:
        eng.close()


if __name__ == "__main__":
    main()
