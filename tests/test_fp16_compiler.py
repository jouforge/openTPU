"""Native half operands stay packed across the host, compiler and DMA boundaries."""
from dataclasses import replace

import numpy as np
import pytest

from opentpu import fp16 as H, fp32 as F, isa as I, language as ol, rtlsim, target
from opentpu.compiler import CompileError
from opentpu.runtime import HalfInput, HalfWeight, Input, Output, collect, compile_kernel, place, run
from test_fp16_isa import CFG


@ol.jit
def kernel(x, w, seed, out, rounded):
    a = ol.pack_fp16(ol.load(x))
    b = ol.load_fp16(w)
    y = ol.matmul_fp16(a, b, acc=ol.load(seed))
    # A VPU read between two writes: accumulator versioning must keep it ordered.
    first = y + 1.0
    ol.matmul_fp16(a, b, acc=y)
    ol.store(out, y + first)
    ol.store(rounded, ol.unpack_fp16(a))


def fixture(cfg=None):
    rng = np.random.default_rng(73)
    a = rng.normal(size=(3, 7)).astype(np.float32)
    b = rng.normal(size=(7, 11)).astype(np.float32)
    seed = rng.normal(size=(3, 11)).astype(np.float32)
    cfg = cfg or replace(CFG, SEQ_WC=True, WC_WIN=16)
    comp, imgs = compile_kernel(kernel, cfg, x=Input(a), w=HalfInput(b), seed=Input(seed),
                                out=Output((3, 11)), rounded=Output((3, 7)))
    first = H.matmul(H.pack(a), H.pack(b), seed)
    second = H.matmul(H.pack(a), H.pack(b), first)
    want = F.add(second, F.add(first, np.float32(1)))
    return comp, imgs, a, b, want


def check(result, a, want):
    np.testing.assert_array_equal(F.bits(result.outputs["out"]), F.bits(want))
    np.testing.assert_array_equal(F.bits(result.outputs["rounded"]), F.bits(H.unpack(H.pack(a))))


def test_half_compiler_reference_and_storage():
    comp, imgs, a, b, want = fixture()
    arg, addr = comp.layout.params["w"]
    raw = imgs[0][addr:addr + arg.array.nbytes].view("<u2").reshape(7, 12)
    np.testing.assert_array_equal(raw[:, :11], H.pack(b))
    assert not raw[:, 11].any()
    assert [p.op for p in comp.programs[0]].count(I.HMM) == 2
    check(run(comp, imgs), a, want)


def test_layout_only_placement_and_bounded_pack(monkeypatch):
    rng = np.random.default_rng(74)
    a = rng.normal(size=(129, 2049)).astype(np.float32)
    original, sizes = H.pack, []

    def pack(x):
        sizes.append(x.size)
        return original(x)

    monkeypatch.setattr(H, "pack", pack)
    half = HalfInput(a)
    assert max(sizes) <= 1 << 18
    np.testing.assert_array_equal(half.array[:, :2049], original(a))
    assert not half.array[:, -1].any()
    args = dict(x=Input(a[:2, :3]), h=half, w=HalfWeight(a[:3, :5]), out=Output((2, 3)))
    cfg = replace(CFG, DRAM_BYTES=1 << 20)
    filled, layout, bound = place(cfg, args)
    sizes.clear()
    empty, draft, views = place(cfg, args, initialize=False)
    assert not sizes and not empty.images[0].any()
    assert filled.next == empty.next
    assert {n: v[1] for n, v in layout.params.items()} == {n: v[1] for n, v in draft.params.items()}
    for name in args:
        a, b = bound[0][name], views[0][name]
        a, b = getattr(a, "storage", a), getattr(b, "storage", b)
        assert (a.base.const, a.shape, a.strides, a.esize) == (b.base.const, b.shape, b.strides, b.esize)


def test_half_compiler_rtl(have_verilator, tmp_path):
    comp, imgs, a, _, want = fixture()
    dram, tmem, stats = rtlsim.run(comp.cfg, comp.programs, imgs, keep=tmp_path,
                                  uarch={"SEQ_WC": 1, "WIN": 16}, max_cycles=100000)
    check(collect(comp, dram, tmem, stats), a, want)


def test_half_compiler_native_memory(have_verilator, tmp_path):
    # V80's grouped architecture and 36-bit addresses, with a small functional
    # memory image. Random native-memory backpressure; this is no bandwidth model.
    cfg = target.config("v80:fp16", TMEM_WORDS=4096, IMEM_WORDS=4096,
                        DRAM_BYTES=65536, ACT_BLOCKS=4)
    comp, imgs, a, _, want = fixture(cfg)
    dram, tmem, stats = rtlsim.run(cfg, comp.programs, imgs, keep=tmp_path, axi=True,
                                  stall=37, seed=17,
                                  uarch=target.sim_uarch("v80:fp16") | {"AW": cfg.AW},
                                  max_cycles=100000)
    check(collect(comp, dram, tmem, stats), a, want)


def test_half_views_and_loop_addresses():
    @ol.jit
    def k(w, out):
        acc = ol.zeros((1, 3))
        a = ol.pack_fp16(ol.full((1, 2), 1.0))
        for i in ol.range(3):
            b = ol.load_fp16(w[i * 2:i * 2 + 2, 2:5])
            ol.matmul_fp16(a, b, acc=acc)
        ol.store(out, acc)

    b = np.arange(42, dtype=np.float32).reshape(6, 7)
    comp, imgs = compile_kernel(k, CFG, w=HalfInput(b), out=Output((1, 3)))
    np.testing.assert_array_equal(run(comp, imgs).outputs["out"], b[:, 2:5].sum(0)[None, :])
    assert any(p.op == I.LD and p.ra for p in comp.programs[0])


def test_half_accumulator_invalidates_fused_rowsum():
    @ol.jit
    def k(x, w, out):
        bias = ol.full((1,), 1.0)
        y = ol.exp2(ol.load(x) - bias[:, None], rowsum=True)
        assert y.rowsum is not None
        a = ol.pack_fp16(ol.full((1, 2), 1.0))
        ol.matmul_fp16(a, ol.load_fp16(w), acc=y)
        ol.store(out, ol.sum(y, axis=1))

    comp, imgs = compile_kernel(k, replace(CFG, FSUM=True), x=Input(np.ones((1, 64))),
                                w=HalfInput(np.ones((2, 64))), out=Output((1,)))
    np.testing.assert_array_equal(run(comp, imgs).outputs["out"], np.float32([192]))


def test_half_split_k_preserves_accumulation_order(have_verilator, tmp_path):
    @ol.jit
    def k(a, b, out):
        y = ol.zeros((3, 9))
        for start, end in [(0, 6), (6, 28), (28, 65)]:
            ol.matmul_fp16(ol.load_fp16(a[:, start:end]), ol.load_fp16(b[start:end, :]), acc=y)
        ol.store(out, y)

    rng = np.random.default_rng(65)
    a = (rng.normal(size=(3, 65)) * 100).astype(np.float32)
    b = (rng.normal(size=(65, 9)) / 100).astype(np.float32)
    cfg = replace(CFG, SEQ_WC=True, WC_WIN=16)
    comp, imgs = compile_kernel(k, cfg, a=HalfInput(a), b=HalfInput(b), out=Output((3, 9)))
    want = F.bits(H.matmul(H.pack(a), H.pack(b)))
    np.testing.assert_array_equal(F.bits(run(comp, [x.copy() for x in imgs]).outputs["out"]), want)
    dram, tmem, stats = rtlsim.run(cfg, comp.programs, imgs, keep=tmp_path,
                                  uarch={"SEQ_WC": 1, "WIN": 16}, max_cycles=100000)
    np.testing.assert_array_equal(F.bits(collect(comp, dram, tmem, stats).outputs["out"]), want)


def test_half_compiler_rejects_wrong_types_shapes_and_alignment():
    @ol.jit
    def k(x, mode):
        if mode == 0:
            ol.load_fp16(x[:, 1:3])
        elif mode == 1:
            ol.matmul_fp16(ol.load(x), ol.load(x))
        elif mode == 2:
            a = ol.pack_fp16(ol.load(x))
            ol.matmul_fp16(a, a)
        else:
            ol.pack_fp16(ol.load(x))

    a = np.ones((2, 4), np.float32)
    for mode, arg, cfg, msg in [(0, HalfInput(a), CFG, "word-aligned"),
                               (1, Input(a), CFG, "HalfTile"),
                               (2, Input(a), CFG, "\\[M, K\\]"),
                               (3, Input(a), replace(CFG, FP16=False), "Config.FP16")]:
        with pytest.raises(CompileError, match=msg):
            compile_kernel(k, cfg, x=arg, mode=mode)
