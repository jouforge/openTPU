"""Record AMR inputs and generate local Tcl configuration for the V80 flow.
No tool download, source patch, board access or generated Vivado output in external trees.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[3]
PLATFORM_DIR = "alveo_v80/platforms/v80_base"
BOARD_PART = "xilinx.com:v80:part0:2.0"
SHARED_CONSTRAINTS = ["boards/bcu1525/constraints/otpu_hbm_ch.tcl",
                      "boards/ypcb-00338/constraints/otpu_clk2x.tcl"]
sys.path.insert(0, str(ROOT))
from opentpu import target as T
from opentpu.host import boardmacro as BM


def git(root: Path, *args: str) -> str:
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, data: str) -> None:
    # Preserve timestamps: repeated steps must not invalidate unchanged RTL/IP inputs.
    if not path.exists() or path.read_text() != data:
        path.write_text(data)


def check_repo(root: Path) -> str:
    if git(root, "diff", "HEAD", "--name-only"):
        raise ValueError(f"{root}: tracked sources differ from HEAD")
    return git(root, "rev-parse", "HEAD")


def vivado_info(executable: str) -> dict:
    path = shutil.which(executable)
    if not path:
        raise ValueError(f"Vivado executable not found: {executable}")
    path = str(Path(path).resolve())
    banner = subprocess.check_output([path, "-version"], text=True).strip()
    match = re.search(r"(?im)^vivado v(\d{4}\.\d+(?:\.\d+)?)\b", banner)
    if not match:
        raise ValueError("cannot read Vivado version")
    return dict(executable=path, version=match[1], banner=banner)


def quote(value) -> str:
    s = str(value)
    for a, b in (("\\", "\\\\"), ('"', '\\"'), ("$", "\\$"), ("[", "\\["),
                 ("\n", "\\n"), ("\r", "\\r")):
        s = s.replace(a, b)
    return '"' + s + '"'


def platform_pins(source: str) -> str:
    # The baseline AMR file reverses the dedicated HBM P/N pins. Keep every other
    # line verbatim; the board-local XDC owns these four corrected assignments.
    expected = dict(hbm_ref_clk_0_clk_p="N18", hbm_ref_clk_0_clk_n="N19",
                    hbm_ref_clk_1_clk_p="N38", hbm_ref_clk_1_clk_n="N37")
    found, lines = {}, []
    for line in source.splitlines(keepends=True):
        if "hbm_ref_clk_" not in line:
            lines.append(line)
            continue
        m = re.fullmatch(r'set_property -dict \{ PACKAGE_PIN (\w+)\s*\} '
                         r'\[get_ports "(hbm_ref_clk_[01]_clk_[pn])"\].*\n?', line)
        if not m or m[2] in found:
            raise ValueError("unexpected AMR HBM pin constraint")
        found[m[2]] = m[1]
    if found != expected:
        raise ValueError("AMR HBM pin contract changed")
    return "".join(lines)


def prepare(amr: Path, designs: Path, out: Path, target: str = "v80",
            vivado: str = "vivado") -> dict:
    amr, designs, out = amr.resolve(), designs.resolve(), out.resolve()
    if out.is_relative_to(amr) or out.is_relative_to(designs):
        raise ValueError("OUT_DIR must be outside the external platform checkouts")
    revisions = dict(amr=check_repo(amr), designs=check_repo(designs))
    t = T.load(target)
    if t.platform["board"] != "v80" or t.tech["part"] != "xcv80-lsva4737-2MHP-e-S":
        raise ValueError("target must use the V80 platform/part")
    p = t.platform
    if t.core_khz != 100000:
        raise ValueError("the initial CIPS PL0 clock is 100 MHz")
    if (t.arch["S"], p["channels_per_slice"], p["channel_bytes"], p["beat_bytes"]) != \
       (1, 8, 1 << 32, 128):
        raise ValueError("the initial V80 platform maps one slice with eight 4 GiB channels")
    if (p["mem_base"], p["user_bar"], p["user_bar_base"], p["csr_offset"], p["csr_bytes"]) != \
       (0x4000000000, 2, 0x20200000000, 0x100000, 0x100000):
        raise ValueError("host map differs from the AMR/openTPU platform contract")
    base = designs / PLATFORM_DIR
    for required in ["scripts/config_bd.tcl", "xdc/impl.xdc", "xdc/impl.pins.xdc",
                     "ip/smbus_v1_1/component.xml"]:
        if not (base / required).is_file():
            raise ValueError(f"missing platform input: {base / required}")
    platform_files = git(designs, "ls-files", "--", PLATFORM_DIR).splitlines()
    pins = platform_pins((base / "xdc/impl.pins.xdc").read_text())
    params = BM.board_params(t, 8, 128)
    defines = BM.board_defines(params)
    rtl_files = [ROOT / "rtl" / f for f in T.sources(t)]
    local_files = [ROOT / f for f in ["opentpu/target.py", "opentpu/host/boardmacro.py",
                                    "targets/v80.toml"] + SHARED_CONSTRAINTS] + rtl_files
    local_files += [f for f in (ROOT / "boards/v80").rglob("*")
                    if f.is_file() and "__pycache__" not in f.parts and f.suffix != ".md"]
    inputs = dict(platform={f: digest(designs / f) for f in platform_files},
                  opentpu={str(f.relative_to(ROOT)): digest(f) for f in sorted(local_files)})
    tool = vivado_info(vivado)
    data = dict(schema=2, target=t.to_dict(), arch_hash=f"{T.arch_hash(T.config(t)):08x}",
                revisions=revisions, vivado=tool, board_part=BOARD_PART,
                platform_dir=PLATFORM_DIR,
                inputs=inputs, opentpu_commit=git(ROOT, "rev-parse", "HEAD"),
                amr_root=str(amr), design_root=str(designs), output_dir=str(out),
                board_params=params, board_defines=defines,
                pin_xdc_sha256=hashlib.sha256(pins.encode()).hexdigest(),
                scope="integrated accelerator RTL and AMR platform; no qualified device image")
    manifest = out / "manifest.json"
    if (out / "project").exists():
        old = json.loads(manifest.read_text()) if manifest.exists() else {}
        if data != old:
            raise ValueError("existing project inputs differ; select a new OUT_DIR")
    out.mkdir(parents=True, exist_ok=True)
    write(manifest, json.dumps(data, indent=2, sort_keys=True) + "\n")
    write(out / "amr.pins.xdc", pins)
    cfg = dict(root=ROOT, out=out, base=base, version=tool["version"], part=t.tech["part"],
               board_part=BOARD_PART, csr_addr=p["user_bar_base"] + p["csr_offset"],
               csr_bytes=p["csr_bytes"], mem_base=p["mem_base"], channel_bytes=p["channel_bytes"],
               nch=p["channels_per_slice"], core_hz=t.core_khz * 1000)
    write(out / "config.tcl", "# Generated by boards/v80/scripts/prepare.py\n" +
        "".join(f"set otpu({k}) {quote(v)}\n" for k, v in cfg.items()))
    write(out / "target_params.tcl", "# opentpu.target.rtl_params, for core integration\n" +
        "set otpu_target [dict create]\n" + "".join(
            f"dict set otpu_target {k} {quote(v)}\n" for k, v in T.rtl_params(t).items()))
    write(out / "otpu_v80_defines.vh", defines)
    write(out / "sources.tcl", "# Generated from opentpu.target.sources\n" +
        "set otpu_rtl [list" + "".join(" "+quote(f) for f in rtl_files) + "]\n")
    return data


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--amr-root", type=Path, default=os.environ.get("AMR_ROOT"), required=not os.environ.get("AMR_ROOT"))
    ap.add_argument("--design-root", type=Path, default=os.environ.get("AMR_DESIGN_ROOT"), required=not os.environ.get("AMR_DESIGN_ROOT"))
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--target", default="v80")
    ap.add_argument("--vivado", default=os.environ.get("VIVADO", "vivado"))
    args = ap.parse_args()
    try:
        d = prepare(args.amr_root, args.design_root, args.out_dir, args.target, args.vivado)
    except (ValueError, OSError, subprocess.CalledProcessError) as e:
        ap.exit(1, f"V80 preparation failed: {e}\n")
    print(f"V80 inputs verified: {d['output_dir']}/manifest.json (CFG_HASH {d['arch_hash']})")


if __name__ == "__main__":
    main()
