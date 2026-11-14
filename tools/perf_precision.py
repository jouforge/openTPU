"""Check and time the same tiled projection in each explicit precision mode.

    PYTHONPATH=. python tools/perf_precision.py --backend rtl --output baseline.json
    PYTHONPATH=. python tools/perf_precision.py --backend board --dev qdma:BDF@Q --output card.json

Shapes are M,N,K. Device cycles include the whole projection program: activation
conversion, weight loads/dequantization, matrix math and stores. Compile, upload
and host run times are separate. RTL uses an uncalibrated native AXI memory model;
its cycles compare RTL revisions, not predicted card bandwidth or model tok/s.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

import numpy as np

from opentpu import fp32 as F, language as ol, rtlsim, target
from opentpu.kernels.mixed import matmul
from opentpu.llm.precision import MODES, reference
from opentpu.runtime import HalfWeight, Int4Weight, Input, Output, Weight, collect, compile_kernel, run


@ol.jit
def projection(x, w, out, block):
    matmul(x, w, out, block)


def triple(text):
    try:
        values = tuple(map(int, text.split(",")))
    except ValueError:
        values = ()
    if len(values) != 3 or min(values) <= 0:
        raise argparse.ArgumentTypeError("expected three positive integers: M,N,K")
    return values


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", choices=("isa", "rtl", "board"), default="rtl")
    p.add_argument("--target", default="v80:mixed")
    p.add_argument("--dev")
    p.add_argument("--shapes", type=triple, nargs="+", default=[(1, 128, 128), (4, 128, 128)])
    p.add_argument("--modes", choices=MODES, nargs="+", default=list(MODES))
    p.add_argument("--block", type=triple, default=(1, 128, 128))
    p.add_argument("--group", type=int, choices=(8, 16, 32, 64, 128), default=128)
    p.add_argument("--repeat", type=int, default=3)
    p.add_argument("--stall", type=int, default=0)
    p.add_argument("--seed", type=int, default=86)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    os.environ["OTPU_TARGET"] = a.target
    if a.repeat < 1 or not 0 <= a.stall <= 100:
        p.error("repeat must be positive and stall must be in 0..100")
    cfg = target.config(a.target, TMEM_WORDS=16384, IMEM_WORDS=32768,
                        ACT_BLOCKS=4, DRAM_BYTES=8 << 20)
    if cfg.S != 1 or not cfg.W4A16:
        p.error("requires a single-slice mixed-precision target")
    if "w8a8" in a.modes and any(k % cfg.D for _, _, k in a.shapes):
        p.error("W8A8 shapes require K a multiple of the target D")
    root = Path(__file__).resolve().parents[1]
    sources = sorted([*root.joinpath("rtl").rglob("*.sv"),
                      *root.joinpath("opentpu").rglob("*.py"), Path(__file__).resolve(),
                      root / "targets" / (target.load(a.target).name + ".toml")])
    report = dict(backend=a.backend, target=a.target, cfg=asdict(cfg), block=a.block,
                  group=a.group, seed=a.seed, stall=a.stall, complete=False, records=[],
                  revision=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
                  dirty=subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True),
                  sources={str(f.relative_to(root)): hashlib.sha256(f.read_bytes()).hexdigest() for f in sources})
    a.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        a.output.write_text(json.dumps(report, indent=2) + "\n")

    save()
    for m, n, k in a.shapes:
        rng = np.random.default_rng(a.seed)
        x = rng.normal(size=(m, k)).astype(np.float32)
        w = (rng.normal(size=(n, k)) / 8).astype(np.float32)
        for mode in a.modes:
            weight = dict(w16a16=HalfWeight, w8a8=Weight, w4a16=Int4Weight)[mode]
            t = time.monotonic()
            comp, images = compile_kernel(projection, cfg, x=Input(x),
                w=weight(w, **(dict(group=a.group) if mode == "w4a16" else {})),
                out=Output((m, n)), block=a.block)
            rec = dict(shape=[m, n, k], mode=mode, compile_place_s=time.monotonic() - t,
                       program_instructions=len(comp.programs[0]), runs=[])
            want = F.bits(reference(x, w, mode, D=cfg.D, group=a.group))
            backend = None
            try:
                if a.backend == "board":
                    from opentpu.host.board import BoardBackend, open_transport
                    # out is the last allocation in this kernel's argument list.
                    end = comp.layout.params["out"][1] + m * n * 4
                    t = time.monotonic()
                    backend = BoardBackend(cfg, [img[:end] for img in images],
                        transport=open_transport(a.dev), model="precision projection")
                    rec.update(upload_init_s=time.monotonic() - t,
                               board={k: v for k, v in backend.info.items() if k != "target"})
                for repeat in range(a.repeat):
                    t = time.monotonic()
                    if backend is not None:
                        stats = backend.run(comp.programs)
                        elapsed = time.monotonic() - t
                        got = backend.read(0, comp.layout.params["out"][1], m * n * 4).view("<f4").reshape(m, n)
                    elif a.backend == "rtl":
                        dram, tmem, stats = rtlsim.run(cfg, comp.programs, images,
                            uarch=target.sim_uarch(a.target) | {"AW": cfg.AW}, axi=True,
                            stall=a.stall, seed=a.seed, max_cycles=100000000)
                        elapsed = time.monotonic() - t
                        got = collect(comp, dram, tmem, stats).outputs["out"]
                    else:
                        result = run(comp, images)
                        got, stats = result.outputs["out"], result.stats
                        elapsed = time.monotonic() - t
                    np.testing.assert_array_equal(F.bits(got), want)
                    cycles = stats.get("cycles")
                    rec["runs"].append(dict(repeat=repeat, exact=True, cycles=cycles,
                        run_wall_s=elapsed, device_s=cycles / (backend.info["core_khz"] * 1000)
                        if backend is not None else None,
                        output_sha256=hashlib.sha256(F.bits(got).tobytes()).hexdigest()))
                    print(json.dumps(dict(shape=rec["shape"], mode=mode, **rec["runs"][-1])), flush=True)
            finally:
                if backend is not None:
                    backend.close()
            report["records"].append(rec)
            save()
    report["complete"] = True
    save()


if __name__ == "__main__":
    main()
