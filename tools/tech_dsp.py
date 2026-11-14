"""Compare FPGA arithmetic cells against generic models in Vivado's simulator.

Use the installed UNISIM library; no vendor model is copied into the repository.
Both families use XSim so PRNG, scheduling and uninitialized-state behavior agree.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]


def run(command, cwd, log):
    r = subprocess.run(list(map(str, command)), cwd=cwd, capture_output=True,
                       text=True, timeout=300)
    text = r.stdout + r.stderr
    (cwd / log).write_text(text)
    if r.returncode or re.search(r"^(?:ERROR:|Error:|Fatal:)", text, re.M):
        raise RuntimeError(f"{command[0]} failed: {cwd / log}\n{text[-3000:]}")
    return text


def trace(out, family, bench, seeds, bins):
    work = out / family / bench
    work.mkdir(parents=True, exist_ok=True)
    run([bins / "xvlog", "--sv", ROOT / f"sim/tech/{bench}.sv",
         ROOT / f"rtl/tech/{family}/otpu_tech_dsp.sv",
         bins.parent / "data/verilog/src/glbl.v"], work, "compile.log")
    run([bins / "xelab", bench, "glbl", "-L", "unisims_ver", "-s", bench, "-mt", "4",
         "--timescale", "1ns/1ps"], work, "elaborate.log")
    result = []
    for seed in seeds:
        text = run([bins / "xsim", bench, "--runall", "--testplusarg", f"seed={seed}"],
                   work, f"seed_{seed}.log")
        rows = [line.strip() for line in text.splitlines() if re.match(r"^\d+ ", line)]
        if len(rows) != 3984 or any(re.search(r"[xz]", line, re.I) for line in rows):
            raise RuntimeError(f"{family}/{bench} seed {seed}: incomplete/unknown trace")
        result.append(rows)
    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--family", choices=("xilinx7", "xilinxus", "xilinxversal"), default="xilinxversal")
    ap.add_argument("--vivado-bin", type=Path)
    ap.add_argument("--out", type=Path, default=ROOT / "build/tech_dsp")
    ap.add_argument("--seeds", nargs="+", type=int, default=[3, 7, 11])
    args = ap.parse_args()
    bins = args.vivado_bin
    if bins is None:
        found = shutil.which("xsim")
        if not found:
            ap.error("put Vivado's bin on PATH or set --vivado-bin")
        bins = Path(found).parent
    bins, out = bins.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    result = dict(family=args.family, seeds=args.seeds, benches={}, inputs={})
    for family in ("generic", args.family):
        p = ROOT / f"rtl/tech/{family}/otpu_tech_dsp.sv"
        result["inputs"][str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
    for bench in ("tb_tech_dsp", "tb_tech_ppe"):
        p = ROOT / f"sim/tech/{bench}.sv"
        result["inputs"][str(p.relative_to(ROOT))] = hashlib.sha256(p.read_bytes()).hexdigest()
        expected = trace(out, "generic", bench, args.seeds, bins)
        actual = trace(out, args.family, bench, args.seeds, bins)
        for seed, a, b in zip(args.seeds, expected, actual):
            if a != b:
                i = next(i for i, (x, y) in enumerate(zip(a, b)) if x != y)
                raise RuntimeError(f"{bench} seed {seed}: first mismatch\n{a[i]}\n{b[i]}")
        result["benches"][bench] = dict(cycles_per_seed=3984, matched=True)
        print(f"PASS {args.family} {bench}: {len(args.seeds)} seeds x 3984 cycles", flush=True)
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
