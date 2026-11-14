"""V80 platform records and local Tcl generation, without a Vivado installation.
The real block design is additionally validated in Vivado (docs/v80.md).
"""
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from opentpu import target as T

ROOT = Path(__file__).resolve().parents[1]
HBM_PINS = """set_property -dict { PACKAGE_PIN N18 } [get_ports "hbm_ref_clk_0_clk_p"]
set_property -dict { PACKAGE_PIN N19 } [get_ports "hbm_ref_clk_0_clk_n"]
set_property -dict { PACKAGE_PIN N38 } [get_ports "hbm_ref_clk_1_clk_p"]
set_property -dict { PACKAGE_PIN N37 } [get_ports "hbm_ref_clk_1_clk_n"]
"""
spec = importlib.util.spec_from_file_location("v80_prepare", ROOT / "boards/v80/scripts/prepare.py")
P = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P)


def git(root, *args):
    return subprocess.check_output(["git", "-C", str(root), *args], text=True).strip()


def commit(root):
    git(root, "add", ".")
    git(root, "-c", "user.name=V80 test", "-c", "user.email=v80@example.invalid",
        "commit", "-qm", "test inputs")
    return git(root, "rev-parse", "HEAD")


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    root, amr, designs = [tmp_path / n for n in ["repo", "amr [$name]", "designs"]]
    for p in [root, amr, designs]:
        p.mkdir()
        git(p, "init", "-q")
    (amr / "README").write_text("AMR fixture\n")
    commit(amr)
    base = designs / "alveo_v80/platforms/v80_base"
    for name in ["scripts/config_bd.tcl", "xdc/impl.xdc", "xdc/impl.pins.xdc",
                 "ip/smbus_v1_1/component.xml"]:
        p = base / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("fixture: tracked external input\n")
    (base / "xdc/impl.pins.xdc").write_text("# other pins preserved\n" + HBM_PINS)
    commit(designs)
    shutil.copytree(ROOT / "boards/v80", root / "boards/v80", ignore=shutil.ignore_patterns("__pycache__"))
    for name in ["opentpu/target.py", "opentpu/host/boardmacro.py", "targets/v80.toml"] + P.SHARED_CONSTRAINTS + [f"rtl/{f}" for f in T.sources("v80")]:
        p = root / name
        p.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT / name, p)
    commit(root)
    monkeypatch.setattr(P, "ROOT", root)
    monkeypatch.setattr(P, "vivado_info", lambda _: dict(
        executable="/test/Vivado/bin/vivado", version="2026.1.1", banner="Vivado v2026.1.1 (test)"))
    return root, amr, designs, tmp_path / "out [$name]"


def test_inputs_and_manifest(inputs):
    root, amr, designs, out = inputs
    # Existing external build products are not source changes.
    (amr / "untracked-build.log").write_text("local result")
    d = P.prepare(amr, designs, out)
    assert json.loads((out / "manifest.json").read_text()) == d
    assert d["schema"] == 2 and "lock" not in d
    assert d["revisions"] == dict(amr=git(amr, "rev-parse", "HEAD"),
                                  designs=git(designs, "rev-parse", "HEAD"))
    assert d["vivado"]["version"] == "2026.1.1"
    assert d["arch_hash"] == f"{T.arch_hash(T.config('v80')):08x}"
    assert len(d["inputs"]["platform"]) == 4
    assert "rtl/boards/v80/otpu_v80_sys.sv" in d["inputs"]["opentpu"]
    assert d["board_params"]["AW"] == 36
    assert (out / "otpu_v80_defines.vh").read_text() == d["board_defines"]
    assert "boards/v80/vivado/bd.tcl" in d["inputs"]["opentpu"]
    assert all(f in d["inputs"]["opentpu"] for f in P.SHARED_CONSTRAINTS)
    assert 'dict set otpu_target PLATFORM "8"' in (out / "target_params.tcl").read_text()
    assert (out / "amr.pins.xdc").read_text() == "# other pins preserved\n"
    assert d["pin_xdc_sha256"] == P.digest(out / "amr.pins.xdc")
    assert not git(designs, "diff", "HEAD", "--name-only")
    # The same inputs can generate products for an existing project; changed ones cannot.
    (out / "project").mkdir()
    assert P.prepare(amr, designs, out) == d
    (root / "boards/v80/vivado/bd.tcl").write_text("changed input")
    with pytest.raises(ValueError, match="existing project inputs differ"):
        P.prepare(amr, designs, out)
    assert json.loads((out / "manifest.json").read_text()) == d


def test_records_new_revision_and_refuses_project_drift(inputs):
    _, amr, designs, out = inputs
    before = P.prepare(amr, designs, out)
    (out / "project").mkdir()
    (amr / "README").write_text("modified source\n")
    with pytest.raises(ValueError, match="tracked sources"):
        P.prepare(amr, designs, out)
    revision = commit(amr)
    with pytest.raises(ValueError, match="existing project inputs differ"):
        P.prepare(amr, designs, out)
    after = P.prepare(amr, designs, out.parent / "new-out")
    assert before["revisions"]["amr"] != after["revisions"]["amr"] == revision


def test_records_new_tool_and_refuses_project_drift(inputs, monkeypatch):
    _, amr, designs, out = inputs
    P.prepare(amr, designs, out)
    (out / "project").mkdir()
    tool = dict(executable="/test/new/vivado", version="2026.2", banner="Vivado v2026.2 (test)")
    monkeypatch.setattr(P, "vivado_info", lambda _: tool)
    with pytest.raises(ValueError, match="existing project inputs differ"):
        P.prepare(amr, designs, out)
    d = P.prepare(amr, designs, out.parent / "new-out")
    assert d["vivado"] == tool


@pytest.mark.parametrize("name", ["Vivado", "vivado"])
def test_vivado_version_from_executable(tmp_path, name):
    tool = tmp_path / "Vivado [$name]"
    tool.write_text(f"#!/bin/sh\nprintf '{name} v2026.1.1 (64-bit)\\nSW Build 123\\n'\n")
    tool.chmod(0o755)
    assert P.vivado_info(str(tool)) == dict(executable=str(tool), version="2026.1.1",
                                           banner=f"{name} v2026.1.1 (64-bit)\nSW Build 123")
    tool.write_text("#!/bin/sh\necho unknown\n")
    with pytest.raises(ValueError, match="cannot read Vivado version"):
        P.vivado_info(str(tool))
    with pytest.raises(ValueError, match="executable not found"):
        P.vivado_info(str(tmp_path / "missing"))


def test_output_must_not_modify_dependencies(inputs):
    _, amr, designs, _ = inputs
    for out in [amr, amr / "build", designs / "build"]:
        with pytest.raises(ValueError, match="outside the external"):
            P.prepare(amr, designs, out)
    out = designs / "new-build"
    env = dict(os.environ, AMR_ROOT=str(amr), AMR_DESIGN_ROOT=str(designs), OUT_DIR=str(out))
    r = subprocess.run(["bash", str(ROOT / "boards/v80/run_vivado.sh"), "project"],
                       env=env, text=True, capture_output=True)
    assert r.returncode != 0 and "outside the external" in r.stderr
    assert not out.exists()


def test_prepare_entry_generates_tcl_without_batch(inputs, tmp_path):
    _, amr, designs, out = inputs
    tool = tmp_path / "vivado test"
    tool.write_text('#!/bin/sh\n[ "$1" = -version ] || exit 99\necho "vivado v2026.1.1 (test)"\n')
    tool.chmod(0o755)
    env = dict(os.environ, AMR_ROOT=str(amr), AMR_DESIGN_ROOT=str(designs), OUT_DIR=str(out),
               PYTHON=sys.executable, VIVADO=str(tool), TARGET="v80")
    r = subprocess.run(["bash", str(ROOT / "boards/v80/run_vivado.sh"), "prepare"],
                       env=env, text=True, capture_output=True)
    assert r.returncode == 0, r.stdout + r.stderr
    assert (out / "config.tcl").exists() and not (out / "project").exists()
    d = json.loads((out / "manifest.json").read_text())
    assert d["vivado"]["executable"] == str(tool)
    assert d["revisions"]["designs"] == git(designs, "rev-parse", "HEAD")


def test_tcl_paths_are_literal(inputs):
    if not shutil.which("tclsh"):
        pytest.skip("tclsh not installed")
    _, amr, designs, out = inputs
    P.prepare(amr, designs, out)
    script = f"source {P.quote(out / 'config.tcl')}\nputs $otpu(base)\nputs $otpu(csr_addr)\n"
    r = subprocess.run(["tclsh"], input=script, text=True, capture_output=True, check=True)
    assert not r.stderr
    assert r.stdout.splitlines() == [str(designs / "alveo_v80/platforms/v80_base"), str(0x20200100000)]
    # Dollar, bracket, backslash, quote and newline characters must roundtrip unchanged.
    value = 'a$unknown[error bad]\\b"c\nd'
    r = subprocess.run(["tclsh"], input=f"puts {P.quote(value)}\n", text=True,
                       capture_output=True, check=True)
    assert r.stdout == value + "\n" and not r.stderr


@pytest.mark.parametrize("failure", ["", "setup", "hold", "pulse", "route", "drc", "pin", "smbus", "skew", "no_skew", "cdc", "cdc_unknown", "cdc_async", "no_cdc"])
def test_implementation_gate(tmp_path, failure):
    """Reject incomplete routes, failing timing/DRC and incorrect physical HBM pins."""
    if not shutil.which("tclsh"):
        pytest.skip("tclsh not installed")
    reports = tmp_path / "reports"
    reports.mkdir()
    whs, wpws = (-0.1 if failure == "hold" else 0.1), (-0.1 if failure == "pulse" else 0.1)
    wns = -0.1 if failure == "setup" else 0.2
    (reports / "impl_timing.txt").write_text(
        "  WNS(ns) TNS(ns) endpoints total WHS(ns) THS(ns) endpoints total WPWS(ns) TPWS(ns) endpoints total\n"
        "  ------- ------- --------- ----- ------- ------- --------- ----- -------- -------- --------- -----\n"
        f"  {wns} 0.000 0 100 {whs} 0.000 0 100 {wpws} 0.000 0 100\n")
    (reports / "impl_bus_skew.txt").write_text(
        "" if failure == "no_skew" else
        "Slack (VIOLATED) : -0.001ns (requirement - actual skew)\n" if failure == "skew" else
        "Slack (MET) : 9.379ns (requirement - actual skew)\n")
    (reports / "impl_cdc.txt").write_text(
        "" if failure == "no_cdc" else
        "Severity Source Clock Destination Clock CDC Type Exceptions Endpoints Safe Unsafe Unknown No ASYNC_REG\n"
        f"Critical axi core Safely Timed Max Delay Datapath Only 214 206 {int(failure == 'cdc')} {int(failure == 'cdc_unknown')} {int(failure == 'cdc_async')}\n")
    script = f"""
set otpu(out) {P.quote(tmp_path)}
set failure {P.quote(failure)}
proc report_route_status {{args}} {{
  if {{[lindex $args 1] eq "ERRORS_IN_ROUTES"}} {{return 0}}
  return [expr {{$::failure ne "route"}}]
}}
proc get_drc_violations {{args}} {{return [expr {{$::failure eq "drc" ? "NSTD-1" : ""}}]}}
proc get_ports {{args}} {{
  if {{$::failure eq "smbus" && [lindex $args 0] eq "-quiet"}} {{return ""}}
  return [lindex $args end]
}}
proc get_property {{prop port}} {{
  if {{$::failure eq "pin"}} {{return WRONG}}
  return [dict get {{hbm_ref_clk_0_clk_p[0] N19 hbm_ref_clk_0_clk_n[0] N18 hbm_ref_clk_1_clk_p[0] N37 hbm_ref_clk_1_clk_n[0] N38}} $port]
}}
if {{[catch {{source {P.quote(ROOT / 'boards/v80/vivado/check_impl.tcl')}}} message]}} {{puts stderr $message; exit 1}}
"""
    r = subprocess.run(["tclsh"], input=script, text=True, capture_output=True)
    assert (r.returncode == 0) == (failure == ""), r.stdout + r.stderr
    assert (reports / "impl_checks.txt").exists()


@pytest.mark.parametrize("source", [HBM_PINS.replace("N18", "N17"),
                                    HBM_PINS + HBM_PINS.splitlines(keepends=True)[0],
                                    "\n".join(HBM_PINS.splitlines()[1:])])
def test_rejects_changed_hbm_pin_contract(source):
    with pytest.raises(ValueError, match="AMR HBM pin"):
        P.platform_pins(source)


@pytest.mark.parametrize("failure", ["", "shared_error", "top", "bridges", "read_pins", "clocks", "ratio", "pumped_mxu"])
def test_timing_contract_gate(failure):
    if not shutil.which("tclsh"):
        pytest.skip("tclsh not installed")
    script = f"""
set failure {P.quote(failure)}
namespace eval ::otpu {{}}
if {{$failure ne "top"}} {{set ::otpu::ok(otpu_v80_top) 1}}
set ::otpu::ok(otpu_hbm_ch) [expr {{$failure eq "bridges" ? 7 : 8}}]
if {{$failure eq "shared_error"}} {{set ::otpu::fail {{missing FIFO}}}}
set c2x_pb {{bank0 bank1}}
set c2x_rp [expr {{$failure eq "read_pins" ? "" : "bank0/ra bank1/ra"}}]
set c2x_qp {{bank0/qa bank1/qa}}
set c2x_cs slow
set c2x_cf fast
proc get_cells {{args}} {{
  if {{[string match *otpu_hbm_cs* [lindex $args end]]}} {{return {{b0 b1 b2 b3 b4 b5 b6 b7}}}}
  return [expr {{$::failure eq "pumped_mxu" ? "mxpc0" : ""}}]
}}
proc get_property {{prop object}} {{
  if {{![info exists ::timing_updated]}} {{return ""}}
  if {{$object eq "slow"}} {{return 10.0}}
  return [expr {{$::failure eq "ratio" ? 6.0 : 5.0}}]
}}
proc update_timing {{}} {{set ::timing_updated 1}}
proc get_clocks {{args}} {{
  if {{$::failure eq "clocks"}} {{return ""}}
  return [lindex $args end]
}}
if {{[catch {{source {P.quote(ROOT / 'boards/v80/vivado/check_timing.tcl')}}} message]}} {{puts stderr $message; exit 1}}
"""
    r = subprocess.run(["tclsh"], input=script, text=True, capture_output=True)
    assert (r.returncode == 0) == (failure == ""), r.stdout + r.stderr
