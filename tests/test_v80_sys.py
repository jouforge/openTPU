"""Real V80 target parameters, CSR/ready/reset contract and an AXI-loaded HALT."""
from pathlib import Path

import pytest

from opentpu import rtlsim, target as T
from opentpu.host import boardmacro as BM

pytestmark = pytest.mark.rtl

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def system(have_verilator):
    out = ROOT / "build/verilator/v80_sys_params"
    out.mkdir(parents=True, exist_ok=True)
    header = out / "otpu_v80_defines.vh"
    header.write_text(BM.board_defines(BM.board_params(T.load("v80"), 8, 128)))
    # Verilator uses the generic cell models, as the core/model simulations do;
    # tools/tech_dsp.py checks the selected DSP58 primitives against those models.
    sources = [header] + [ROOT / "rtl" / f for f in T.sources("v80", "generic")
                          if not f.endswith("otpu_fpga_top_v80.sv")]
    return rtlsim.build("tb_v80_sys", sources + [ROOT / "sim/verilator/tb_v80_sys.sv"])


@pytest.mark.parametrize("fault", [0, 1, 2], ids=["ready", "bad-id", "slverr"])
def test_v80_sys(system, fault):
    r = rtlsim.run_sim([str(system), f"+fault={fault}"], timeout=120)
    assert r.returncode == 0 and "PASS v80 system" in r.stdout, r.stdout + r.stderr
