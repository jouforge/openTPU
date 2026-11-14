"""Mixed projection kernels: packed storage, mode transitions and RTL/ISA agreement."""
from dataclasses import replace

import numpy as np
import pytest

from opentpu import fp16 as H, fp32 as F, int4 as Q4, isa as I, language as ol, rtlsim
from opentpu.kernels.mixed import matmul
from opentpu.runtime import HalfWeight, Int4Weight, Input, Output, Weight, collect, compile_kernel, place, run
from test_int4_isa import CFG
from test_fp16_isa import TMEM3


@ol.jit
def projection(x, w, out, block=(1, 8, 8)):
    matmul(x, w, out, block)


@pytest.mark.parametrize("shape,group", [((3, 17, 19), 8), ((1, 9, 131), 128), ((1, 1, 1), 8)])
def test_w4_projection(shape, group):
    m, n, k = shape
    rng = np.random.default_rng(7)
    a, w = rng.normal(size=(m, k)).astype(np.float32), rng.normal(size=(n, k)).astype(np.float32)
    cfg = replace(CFG, TMEM_WORDS=16384, IMEM_WORDS=32768)
    comp, img = compile_kernel(projection, cfg, x=Input(a), w=Int4Weight(w, group=group), out=Output((m, n)))
    assert any(i.op == I.HDQ for i in comp.programs[0])
    assert any(i.op == I.HMM for i in comp.programs[0])
    np.testing.assert_array_equal(F.bits(run(comp, img).outputs["out"]), F.bits(Q4.matmul(a, Q4.quantize(w, group))))


def test_layout_only_and_shards(monkeypatch):
    cfg = replace(CFG, S=2)
    w = np.random.default_rng(41).normal(size=(18, 32)).astype(np.float32)
    args = dict(w=Int4Weight(w, shard=0, group=8), q=Weight(w, shard=0), h=HalfWeight(w, shard=0))
    filled, layout, bound = place(cfg, args)
    def no_quant(*a, **kw):
        raise AssertionError("layout-only placement must not quantize weights")
    monkeypatch.setattr(Q4, "quantize", no_quant)
    from opentpu import quant as Q
    monkeypatch.setattr(Q, "quantize_grouped", no_quant)
    empty, draft, views = place(cfg, args, initialize=False)
    assert filled.next == empty.next and not any(a.any() for a in empty.images)
    assert {k: a[1] for k, a in layout.params.items()} == {k: a[1] for k, a in draft.params.items()}
    monkeypatch.undo()
    for s in range(2):
        want = Q4.quantize(w[s * 9:(s + 1) * 9], 8)
        data, scales = layout.params["w"][1]
        np.testing.assert_array_equal(filled.images[s][data:data + want.words.nbytes], want.words.view(np.uint8).ravel())
        np.testing.assert_array_equal(filled.images[s][scales:scales + want.scales.nbytes], want.scales.view(np.uint8).ravel())
        assert bound[s]["w"].shape == views[s]["w"].shape == (32, 9)


@ol.jit
def transitions(x, w8, w4, w16, t1, t2, out):
    matmul(x, w8, t1, (1, 8, 8))
    matmul(t1, w4, t2, (1, 8, 8))
    matmul(t2, w16, out, (1, 8, 8))


def mixed_fixture():
    rng = np.random.default_rng(15)
    x = rng.normal(size=(2, 32)).astype(np.float32)
    w = [rng.normal(size=(32, 32)).astype(np.float32) / 8 for _ in range(3)]
    comp, imgs = compile_kernel(transitions, CFG, x=Input(x), w8=Weight(w[0]),
        w4=Int4Weight(w[1], group=8), w16=HalfWeight(w[2]),
        t1=Input(np.zeros((2, 32), np.float32)), t2=Input(np.zeros((2, 32), np.float32)), out=Output((2, 32)))
    # W8A8 is compared to the existing untiled compiler path, independently of
    # the new dispatcher; subsequent stages use the ordered numerical oracles.
    @ol.jit
    def integer(x, w, out):
        ol.store(out, ol.dot(ol.load(x), w))
    first, image = compile_kernel(integer, CFG, x=Input(x), w=Weight(w[0]), out=Output((2, 32)))
    y = run(first, image).outputs["out"]
    y = Q4.matmul(y, Q4.quantize(w[1], 8))
    y = H.matmul(H.pack(y), H.pack(w[2].T))
    return comp, imgs, F.bits(y)


def test_mode_transitions():
    comp, imgs, want = mixed_fixture()
    np.testing.assert_array_equal(F.bits(run(comp, imgs).outputs["out"]), want)


def test_mode_transitions_rtl(have_verilator, tmp_path):
    comp, imgs, want = mixed_fixture()
    dram, tmem, stats = rtlsim.run(CFG, comp.programs, imgs, keep=tmp_path,
        uarch={"SEQ_WC": 1, "WIN": 16, **TMEM3}, max_cycles=1000000)
    np.testing.assert_array_equal(F.bits(collect(comp, dram, tmem, stats).outputs["out"]), want)


@pytest.mark.parametrize("D", [32, 128])
def test_w8_reference_windows(D):
    from opentpu.llm.precision import reference
    rng = np.random.default_rng(84)
    x = rng.normal(size=(3, 2 * D)).astype(np.float32)
    w = rng.normal(size=(13, 2 * D)).astype(np.float32)
    x[:, D // 2:] *= np.float32(2**-22)
    w[:, D // 4:] *= np.float32(2**-18)
    x[0] = 0
    w[0] = 0
    cfg = replace(CFG, D=D, TMEM_WORDS=16384, IMEM_WORDS=32768)
    comp, imgs = compile_kernel(projection, cfg, x=Input(x), w=Weight(w), out=Output((3, 13)))
    np.testing.assert_array_equal(F.bits(run(comp, imgs).outputs["out"]),
                                  F.bits(reference(x, w, "w8a8", D=D)))
