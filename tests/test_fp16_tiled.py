"""Model-sized native half projections: bounded storage and ordered accumulation."""
from dataclasses import replace

import numpy as np
import pytest

from opentpu import fp16 as H, fp32 as F, isa as I, language as ol, rtlsim, target
from opentpu.compiler import Affine, CompileError, HalfTensor, Tensor
from opentpu.kernels.fp16 import matmul, swiglu
from opentpu.runtime import HalfInput, HalfWeight, Input, Output, collect, compile_kernel, place, run
from test_fp16_isa import CFG


@ol.jit
def projection(x, w, out, block=(4, 32, 32)):
    matmul(x, w, out, block)


def fixture(m=9, n=67, k=65, packed=False, cfg=None, block=(4, 32, 32)):
    rng = np.random.default_rng(m + n + k)
    x = rng.normal(size=(m, k)).astype(np.float32)
    w = rng.normal(size=(n, k)).astype(np.float32)
    comp, imgs = compile_kernel(projection, cfg or CFG, x=(HalfInput if packed else Input)(x),
                                w=HalfWeight(w), out=Output((m, n)), block=block)
    return comp, imgs, F.bits(H.matmul(H.pack(x), H.pack(w.T)))


@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("shape", [(1, 1, 1), (4, 32, 32), (9, 67, 65), (9, 67, 513)])
def test_tiled_reference(shape, packed):
    # The last case's input and weights both exceed TMEM; all three dimensions
    # exercise repeated hardware iterations and short tails.
    cfg = replace(CFG, DRAM_BYTES=131072)
    comp, imgs, want = fixture(*shape, packed=packed, cfg=cfg)
    np.testing.assert_array_equal(F.bits(run(comp, imgs).outputs["out"]), want)


@pytest.mark.parametrize("wc", [False, True])
def test_tiled_rtl(have_verilator, tmp_path, wc):
    cfg = replace(CFG, SEQ_WC=wc, WC_WIN=16)
    comp, imgs, want = fixture(cfg=cfg)
    dram, tmem, stats = rtlsim.run(cfg, comp.programs, imgs, keep=tmp_path,
                                  uarch={"SEQ_WC": int(wc), "WIN": 16}, max_cycles=1000000)
    np.testing.assert_array_equal(F.bits(collect(comp, dram, tmem, stats).outputs["out"]), want)


def test_tiled_native_memory(have_verilator, tmp_path):
    cfg = target.config("v80:fp16", TMEM_WORDS=4096, IMEM_WORDS=4096,
                        DRAM_BYTES=65536, ACT_BLOCKS=4)
    comp, imgs, want = fixture(packed=True, cfg=cfg)
    dram, tmem, stats = rtlsim.run(cfg, comp.programs, imgs, keep=tmp_path, axi=True,
                                  stall=37, seed=19,
                                  uarch=target.sim_uarch("v80:fp16") | {"AW": cfg.AW},
                                  max_cycles=1000000)
    np.testing.assert_array_equal(F.bits(collect(comp, dram, tmem, stats).outputs["out"]), want)


@pytest.mark.parametrize("shard", [None, 0])
def test_half_weight_placement_and_sharding(shard):
    @ol.jit
    def k(x, w, out, sharded):
        n = w.shape[1]
        col = ol.program_id() * n if sharded else 0
        matmul(x, w, out[:, col:col + n], (2, 4, 4))

    cfg = replace(CFG, S=2)
    x = np.arange(21, dtype=np.float32).reshape(3, 7) / 13
    w = np.arange(42, dtype=np.float32).reshape(6, 7) / 11
    args = dict(x=Input(x), w=HalfWeight(w, shard), out=Output((3, 6)), sharded=shard == 0)
    comp, imgs = compile_kernel(k, cfg, **args)
    addr = comp.layout.params["w"][1]
    n = 3 if shard == 0 else 6
    for s, img in enumerate(imgs):
        raw = img[addr:addr + 7 * ((n + 1) // 2 * 2) * 2].view("<u2").reshape(7, -1)
        want = H.pack(w[s * n:(s + 1) * n].T if shard == 0 else w.T)
        np.testing.assert_array_equal(raw[:, :n], want)
        assert not raw[:, n:].any()
    np.testing.assert_array_equal(F.bits(run(comp, imgs).outputs["out"]),
                                  F.bits(H.matmul(H.pack(x), H.pack(w.T))))
    dram, _, bound = place(cfg, {"w": args["w"]}, weights_at=32768)
    assert bound[0]["w"].storage.base.static() == 32768


def test_large_projection_has_bounded_program_and_tmem():
    # Qwen-sized projections, compiled without allocating or running model images.
    # Width and prompt length change loop counts, not the size of full-tile code.
    def trace(m, n, k):
        a = Tensor(Affine(0), (m, k), (k, 1))
        b = HalfTensor(Tensor(Affine(1 << 26), (k, n), (n, 1), esize=2))
        out = Tensor(Affine(1 << 28), (m, n), (n, 1))
        cfg = replace(CFG, DRAM_BYTES=1 << 30)
        builder = projection.trace(cfg, 0, dict(x=a, w=b, out=out))
        program = builder.finish()
        return len(program), builder.tmem_peak, sum(p.op == I.HMM for p in program)

    small = trace(8, 64, 64)
    assert trace(256, 3584, 1024) == small
    assert trace(256, 1024, 3584) == small
    assert small[0] * 8 <= CFG.IMEM_WORDS
    assert small[1] < 1024
    assert small[2] == 1


def test_tiled_strided_views():
    @ol.jit
    def k(x, w, out):
        matmul(x[1:6, 2:19], w[1:18, 2:15], out[1:6, 1:14], (2, 4, 6))

    rng = np.random.default_rng(21)
    x, w = rng.normal(size=(7, 21)), rng.normal(size=(19, 17))
    # Input as destination lets us also check the untouched border sentinel.
    comp, imgs = compile_kernel(k, CFG, x=Input(x), w=HalfInput(w),
                                out=Input(np.full((7, 15), 123.0)))
    result = run(comp, imgs)
    addr = comp.layout.params["out"][1]
    got = result.drams[0][addr:addr + 7 * 15 * 4].view(np.float32).reshape(7, 15)
    want = np.full((7, 15), 123.0, np.float32)
    want[1:6, 1:14] = H.matmul(H.pack(x[1:6, 2:19]), H.pack(w[1:18, 2:15]))
    np.testing.assert_array_equal(F.bits(got), F.bits(want))


def test_tiled_special_values_and_poisoned_padding(have_verilator, tmp_path):
    raw = np.array([0, 0x8000, 1, 0x8001, 0x3C00, 0xBC00, 0x7BFF, 0x7C00,
                    0xFC00, 0x7E00], np.uint16)
    x = H.unpack(np.resize(raw, (5, 19)))
    w = H.unpack(np.resize(np.roll(raw, 3), (11, 19)))
    # Keep finite outputs beside NaN/Inf outputs: consuming the poisoned padding
    # must corrupt those finite results rather than merely reproduce a NaN.
    x[0], w[0] = 1.0, -1.0
    x[1] = H.unpack(np.resize(raw[:6], 19))
    w[1] = H.unpack(np.resize(raw[:4], 19))
    cfg = replace(CFG, SEQ_WC=True, WC_WIN=16)
    comp, imgs = compile_kernel(projection, cfg, x=HalfInput(x), w=HalfWeight(w),
                                out=Output((5, 11)), block=(2, 4, 6))
    for name, rows, stride in [("x", 5, 20), ("w", 19, 12)]:
        addr = comp.layout.params[name][1]
        raw = imgs[0][addr:addr + 2 * rows * stride].view("<u2").reshape(rows, stride)
        raw[:, -1] = 0x7E00
    want = F.bits(H.matmul(H.pack(x), H.pack(w.T)))
    np.testing.assert_array_equal(F.bits(run(comp, [v.copy() for v in imgs]).outputs["out"]), want)
    dram, tmem, stats = rtlsim.run(cfg, comp.programs, imgs, keep=tmp_path,
                                  uarch={"SEQ_WC": 1, "WIN": 16}, max_cycles=1000000)
    np.testing.assert_array_equal(F.bits(collect(comp, dram, tmem, stats).outputs["out"]), want)


def test_invalid_tiled_operands():
    a = Tensor(Affine(0), (4, 8), (8, 1))
    b = HalfTensor(Tensor(Affine(4096), (8, 4), (4, 1), esize=2))
    out = Tensor(Affine(8192), (4, 4), (4, 1))
    args = dict(x=a, w=b, out=out)
    for block in [(0, 4, 4), (2, 3, 4), (2, 4, 3), (2, 4), (2, 4, 65536)]:
        with pytest.raises(CompileError, match="block"):
            projection.trace(CFG, 0, args | {"block": block})
    with pytest.raises(CompileError, match="disjoint"):
        projection.trace(CFG, 0, args | {"out": a[:, :4]})
    with pytest.raises(CompileError, match="word-aligned"):
        projection.trace(CFG, 0, args | {"w": b[:, 1:]})
    with pytest.raises(CompileError, match="Config.FP16"):
        projection.trace(replace(CFG, FP16=False), 0, args)
    with pytest.raises(ValueError, match="Config.FP16"):
        place(replace(CFG, FP16=False), {"w": HalfWeight(np.ones((2, 3)))})
    with pytest.raises(ValueError, match="divisible"):
        place(replace(CFG, S=2), {"w": HalfWeight(np.ones((3, 7)), shard=0)})
    with pytest.raises(ValueError, match="shard"):
        HalfWeight(np.ones((3, 7)), shard=1)


@ol.jit
def mlp(x, wg, wu, wd, gate, up, out):
    swiglu(x, wg, wu, wd, out, (gate, up), (2, 8, 8))


def mlp_fixture(cfg):
    rng = np.random.default_rng(94)
    x = rng.normal(size=(5, 17)).astype(np.float32)
    wg, wu = [rng.normal(size=(19, 17)).astype(np.float32) / 4 for _ in range(2)]
    wd = rng.normal(size=(11, 19)).astype(np.float32) / 4
    g = H.matmul(H.pack(x), H.pack(wg.T))
    u = H.matmul(H.pack(x), H.pack(wu.T))
    hidden = F.mul(F.silu(g, np.float32(-ol.LOG2E)), u)
    want = H.matmul(H.pack(hidden), H.pack(wd.T))
    comp, imgs = compile_kernel(mlp, cfg, x=Input(x), wg=HalfWeight(wg), wu=HalfWeight(wu),
                                wd=HalfWeight(wd), gate=Output((5, 19)), up=Output((5, 19)),
                                out=Output((5, 11)))
    return comp, imgs, dict(gate=hidden, up=u, out=want)


@pytest.mark.parametrize("vsig", [False, True])
def test_native_swiglu_reference(vsig):
    comp, imgs, want = mlp_fixture(replace(CFG, VSIG=vsig))
    got = run(comp, imgs).outputs
    for key in want:
        np.testing.assert_array_equal(F.bits(got[key]), F.bits(want[key]))


def test_native_swiglu_rtl(have_verilator, tmp_path):
    cfg = replace(CFG, SEQ_WC=True, WC_WIN=16)
    comp, imgs, want = mlp_fixture(cfg)
    dram, tmem, stats = rtlsim.run(cfg, comp.programs, imgs, keep=tmp_path,
                                  uarch={"SEQ_WC": 1, "WIN": 16}, max_cycles=1000000)
    got = collect(comp, dram, tmem, stats).outputs
    for key in want:
        np.testing.assert_array_equal(F.bits(got[key]), F.bits(want[key]))


def test_native_swiglu_rejects_scratch_alias():
    @ol.jit
    def k(x, wg, wu, wd, gate, up, out):
        swiglu(x, wg, wu, wd, out, (gate, gate))

    comp, imgs, _ = mlp_fixture(CFG)
    args = {name: arg for name, (arg, _) in comp.layout.params.items()}
    with pytest.raises(CompileError, match="disjoint"):
        compile_kernel(k, CFG, **args)
