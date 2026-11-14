"""Native QK / softmax / P.V and word-preserving packed cache appends."""
from dataclasses import replace

import numpy as np
import pytest

from opentpu import fp16 as H, fp32 as F, language as ol, rtlsim, target
from opentpu.compiler import CompileError
from opentpu.kernels.fp16_attention import append, attend
from opentpu.runtime import HalfInput, Input, Output, collect, compile_kernel, run
from test_fp16_isa import CFG


@ol.jit
def attention(q, kt, v, out, scores, mask=None, softmax_block=32):
    attend(q, kt, v, out, scores, mask, softmax_block=softmax_block)


def fixture(m=3, t=67, d=19, dv=10, cfg=CFG, softmax_block=32):
    rng = np.random.default_rng(80)
    q = rng.normal(size=(m, d)).astype(np.float32)
    k = rng.normal(size=(t, d)).astype(np.float32)
    v = rng.normal(size=(t, dv)).astype(np.float32)
    valid = np.minimum(t, np.arange(m) + max(1, t - m - 5))
    mask = np.where(np.arange(t)[None, :] < valid[:, None], np.inf, -np.inf).astype(np.float32)
    comp, imgs = compile_kernel(attention, cfg, q=Input(q), kt=HalfInput(k.T), v=HalfInput(v),
                                out=Output((m, dv)), scores=Output((m, t)), mask=Input(mask),
                                softmax_block=softmax_block)
    out, p = H.attention(H.pack(q), H.pack(k.T), H.pack(v), mask, softmax_block)
    return comp, imgs, dict(out=out, scores=p)


def check(got, want):
    for name in want:
        np.testing.assert_array_equal(F.bits(got[name]), F.bits(want[name]), err_msg=name)


@pytest.mark.parametrize("shape", [(1, 1, 2, 2), (3, 67, 19, 10), (5, 129, 32, 34)])
@pytest.mark.parametrize("softmax_block", [16, 32])
def test_attention_reference(shape, softmax_block):
    comp, imgs, want = fixture(*shape, softmax_block=softmax_block)
    check(run(comp, imgs).outputs, want)


@pytest.mark.parametrize("wc", [False, True])
def test_attention_rtl(have_verilator, tmp_path, wc):
    cfg = replace(CFG, SEQ_WC=wc, WC_WIN=16)
    comp, imgs, want = fixture(cfg=cfg)
    dram, tmem, stats = rtlsim.run(cfg, comp.programs, imgs, keep=tmp_path,
                                  uarch={"SEQ_WC": int(wc), "WIN": 16}, max_cycles=1000000)
    check(collect(comp, dram, tmem, stats).outputs, want)


def test_attention_native_memory(have_verilator, tmp_path):
    cfg = target.config("v80:fp16", TMEM_WORDS=4096, IMEM_WORDS=4096,
                        DRAM_BYTES=65536, ACT_BLOCKS=4)
    comp, imgs, want = fixture(cfg=cfg)
    dram, tmem, stats = rtlsim.run(cfg, comp.programs, imgs, keep=tmp_path, axi=True,
                                  stall=41, seed=23,
                                  uarch=target.sim_uarch("v80:fp16") | {"AW": cfg.AW},
                                  max_cycles=1000000)
    check(collect(comp, dram, tmem, stats).outputs, want)


@ol.jit
def step(q, key, value, kt, v, out, scores, pos):
    append(key, value, kt, v, pos, block=4)
    attend(q, kt[:, :pos + 1], v[:pos + 1, :], out, scores)


@pytest.mark.parametrize("backend", ["isa", "rtl"])
def test_cache_append_and_decode(have_verilator, tmp_path, backend):
    cfg = replace(CFG, SEQ_WC=True, WC_WIN=16)
    rng = np.random.default_rng(107)
    d, dv, cap = 9, 10, 6
    # A finite sentinel in unwritten slots must survive both halves' updates.
    keys = np.full((d, cap), 0.125, np.float32)
    vals = np.full((cap, dv), -0.25, np.float32)
    for pos in range(5):
        q, k = [rng.normal(size=(1, d)).astype(np.float32) for _ in range(2)]
        v = rng.normal(size=(1, dv)).astype(np.float32)
        comp, imgs = compile_kernel(step, cfg, q=Input(q), key=Input(k), value=Input(v),
                                    kt=HalfInput(keys), v=HalfInput(vals), pos=pos,
                                    out=Output((1, dv)), scores=Output((1, pos + 1)))
        if backend == "isa":
            result = run(comp, imgs)
        else:
            dram, tmem, stats = rtlsim.run(cfg, comp.programs, imgs, keep=tmp_path / str(pos),
                                          uarch={"SEQ_WC": 1, "WIN": 16}, max_cycles=1000000)
            result = collect(comp, dram, tmem, stats)
        keys[:, pos] = H.unpack(H.pack(k[0]))
        vals[pos] = H.unpack(H.pack(v[0]))
        out, p = H.attention(H.pack(q), H.pack(keys[:, :pos + 1]), H.pack(vals[:pos + 1]))
        check(result.outputs, dict(out=out, scores=p))
        for name, expected in (("kt", keys), ("v", vals)):
            addr = comp.layout.params[name][1]
            raw = result.drams[0][addr:addr + 2 * expected.size].view("<u2").reshape(expected.shape)
            np.testing.assert_array_equal(raw, H.pack(expected))


def test_attention_reference_close_to_float64():
    rng = np.random.default_rng(81)
    q, k, v = [rng.normal(size=s).astype(np.float32) for s in ((4, 32), (37, 32), (37, 16))]
    result, probs = H.attention(H.pack(q), H.pack(k.T), H.pack(v))
    scores = q.astype(np.float64) @ k.T / np.sqrt(32)
    p = np.exp(scores - scores.max(axis=1, keepdims=True))
    p /= p.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(probs, p, atol=1e-4, rtol=0.003)
    np.testing.assert_allclose(result, p @ v, atol=5e-4, rtol=0.01)


def test_store_fp16_rejects_partial_words():
    @ol.jit
    def k(src, dst):
        ol.store_fp16(dst, ol.pack_fp16(ol.load(src)))

    with pytest.raises(CompileError, match="even width"):
        compile_kernel(k, CFG, src=Input(np.ones((2, 3))), dst=HalfInput(np.zeros((2, 3))))


def test_cache_append_rejects_invalid_position():
    for pos in (-1, 4):
        with pytest.raises(CompileError, match="static position"):
            compile_kernel(step, CFG, q=Input(np.zeros((1, 4))), key=Input(np.zeros((1, 4))),
                            value=Input(np.zeros((1, 4))), kt=HalfInput(np.zeros((4, 4))),
                            v=HalfInput(np.zeros((4, 4))), out=Output((1, 4)),
                            scores=Output((1, 1)), pos=pos)
