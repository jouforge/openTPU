"""V80's simulation baseline: the target's core and logical native-memory channels.

These checks do not model AMR, physical HBM, QDMA or DSP58 cells. DRAM is reduced to the
test image; arithmetic, features, channel geometry and micro-architecture are the target's.
"""
import numpy as np
import pytest

from opentpu import isa as I
from opentpu import target as T


@pytest.mark.rtl
def test_native_mm(have_verilator):
    """All weight formats, tails, replay and accumulation through eight channels with
    random stalls: the complete RTL TMEM equals the ISA simulator's."""
    from test_wg_rtl import _cases, _check

    cfg = T.config("v80", DRAM_BYTES=1 << 23)
    cases = [c for wf in (I.W8, I.W4I, I.W4F, I.WNV) for c in _cases(wf)]
    # tb_top defaults to AW 32; sim_uarch contains only the timing/channel parameters.
    _check(cases, seed=80, cfg=cfg, uarch=T.sim_uarch("v80") | {"AW": cfg.AW},
           axi=True, stall=30)


@pytest.mark.rtl
@pytest.mark.parametrize("wformat", ["int8", "fp4"])
def test_qwen35_prefill_decode(have_verilator, wformat):
    """A tiny random Qwen3.5 continues from nonempty convolution, DeltaNet and KV state.
    Prefill, then two decode steps, give the ISA's logits and final DRAM bit for bit."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from test_qwen35 import _tiny_model
    from opentpu.llm.qwen3 import Engine
    from opentpu.llm.rtl_backend import RtlBackend

    _, weights, spec = _tiny_model(4, 8)
    cfg = T.config("v80", DRAM_BYTES=1 << 25)
    eng = Engine(spec, weights, cap=256, cfg=cfg, wformat=wformat, rows=4)
    toks = [int(t) for t in np.random.default_rng(80).integers(0, 1000, 11)]
    for tok in toks[:5]:
        eng.step(tok)
    isa, n = eng.backend, eng.image.nbytes
    rtl = RtlBackend(cfg, [s.dram[:n].copy() for s in isa.machine.slices],
                     uarch=T.sim_uarch("v80") | {"AW": cfg.AW}, axi=True, stall=30)
    want = [eng.prefill(toks[5:9], chunk=4), eng.step(toks[9]), eng.step(toks[10])]
    eng.backend, eng.pos = rtl, 5
    got = [eng.prefill(toks[5:9], chunk=4), eng.step(toks[9]), eng.step(toks[10])]
    for a, b in zip(want, got):
        assert np.array_equal(a.view(np.uint32), b.view(np.uint32))
    for s in range(cfg.S):
        assert np.array_equal(isa.machine.slices[s].dram[:n], rtl.drams[s][:n]), s
