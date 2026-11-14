"""V80 native/AXI4 bridge: high addresses, partial writes, posted reads and resets.
Uses the upstream bridge's random master and byte scoreboard on the real V80 wrapper.
"""
from pathlib import Path

import pytest

from opentpu import rtlsim

pytestmark = pytest.mark.rtl

ROOT = Path(__file__).resolve().parents[1]
SRC = [ROOT / p for p in [
    "rtl/shell/otpu_afifo.sv", "rtl/shell/otpu_mem_ch.sv", "rtl/mem/otpu_hbm_ch.sv",
    "rtl/boards/v80/otpu_v80_mem.sv", "sim/verilator/otpu_hbm_model.sv",
    "sim/verilator/tb_hbmch.sv", "sim/verilator/tb_v80_mem.sv"]]


@pytest.fixture(scope="module", params=[0, 7], ids=["ch0", "ch7"])
def bridge(request, have_verilator):
    return rtlsim.build("tb_v80_mem", SRC, {"CH": request.param})


@pytest.mark.parametrize("args", [
    ["+offset=0", "+seed=1"],
    ["+offset=3ff80000", "+seq=2", "+nb=8192", "+ntx=800", "+seed=2"],
    ["+offset=fff00000", "+seq=2", "+nb=8192", "+ntx=800", "+seed=3"],
    ["+nb=16", "+raw=40", "+war=40", "+ppct=100", "+seed=4"],
    ["+hbm_ast=60", "+hbm_wst=60", "+hbm_rgap=50", "+ap=3701", "+seed=5"],
    ["+gapw=0", "+gapc=70", "+wpct=80", "+seed=6"],
    ["+ppost=100", "+nslot=4", "+seed=7"],
    ["+ppost=60", "+nslot=2048", "+seed=8"],
    ["+areset=2000", "+arep=5003", "+arlen=8", "+hbm_lat=300", "+seed=9"],
    ["+hreset=2000", "+hrep=7919", "+hrlen=2", "+ap=3301", "+seed=10"],
], ids=["base", "pc-boundary", "channel-top", "partial-hazards", "backpressure",
        "write-ahead", "posted-wrap", "posted-wide", "core-reset", "axi-reset"])
def test_v80_mem(bridge, args):
    defaults = [] if any(a.startswith("+ppost=") for a in args) else ["+ppost=50"]
    r = rtlsim.run_sim([str(bridge), *defaults, *args], timeout=180)
    assert r.returncode == 0 and "PASS cycles=" in r.stdout, r.stdout + r.stderr
    assert "FAIL" not in r.stdout and "ERROR" not in r.stdout


@pytest.mark.parametrize("fault", [1, 2], ids=["bad-bid", "bad-rid"])
def test_v80_response_id(bridge, fault):
    r = rtlsim.run_sim([str(bridge), f"+fault={fault}", "+ntx=50"], timeout=60)
    assert r.returncode == 0 and "PASS cycles=" in r.stdout, r.stdout + r.stderr


def test_v80_ready(have_verilator):
    exe = rtlsim.build("tb_v80_ready", [ROOT / "rtl/boards/v80/otpu_v80_ready.sv",
                                       ROOT / "sim/verilator/tb_v80_ready.sv"])
    r = rtlsim.run_sim([str(exe)], timeout=60)
    assert r.returncode == 0 and "PASS v80 ready" in r.stdout, r.stdout + r.stderr
