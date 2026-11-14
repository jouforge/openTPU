"""Packed FP16 instructions through DMA, TMEM, dispatch and native arithmetic."""
from dataclasses import replace
import re

import numpy as np
import pytest

from opentpu import Config, isa as I, fp16 as H, fp32 as F, rtlsim, target, waits
from opentpu.isasim import Machine, SimError


CFG = Config(FP16=True, TMEM_WORDS=4096, IMEM_WORDS=4096, DRAM_BYTES=65536, ACT_BLOCKS=4)
TMEM3 = dict(TMEM_IMPL=3, TMEM_NSB=2, TMEM_RPL=1, TMEM_TDP=1,
             TMEM_GST=1, TMEM_PMP=1, RPB=128, WPB=1)


def program():
    rng = np.random.default_rng(36)
    m, n, k = 3, 11, 7
    a = rng.normal(size=(m, k)).astype(np.float32)
    b = rng.normal(size=(k, n)).astype(np.float32)
    seed = rng.normal(size=(m, 14)).astype(np.float32)
    img = np.zeros(CFG.DRAM_BYTES, np.uint8)
    img[:a.nbytes] = a.view(np.uint8).ravel()
    img[512:512 + b.nbytes] = b.view(np.uint8).ravel()
    img[1024:1024 + seed.nbytes] = seed.view(np.uint8).ravel()
    img[8192:8224] = 1
    p = [I.ld(0, 0, a.size), I.ld(512, 128, b.size), I.ld(1024, 768, seed.size)]
    p += [I.hcvt(256 + r * 5, r * k, k) for r in range(m)]
    p += [I.hcvt(320 + r * 8, 128 + r * n, n) for r in range(k)]
    p += [I.vop(I.V_FILL, 2048, 0, 0, 1, 32, 32, 32, 0, I.B_SCALAR, 1.0),
          I.qact(2048, 1, 0, 1, 32),
          I.mm(8192, 0, 2112, 1, 1, 32, 1, 1, 0, 4, unit=True),
          I.li(1, 768), I.li(2, 256), I.li(3, 320),
          I.hmm(0, 0, 0, m, n, k, 5, 8, 14, acc=True, ra=1, rb=2, rc=3),
          I.mm(8192, 0, 2113, 1, 1, 32, 1, 1, 0, 4, unit=True),
          I.hmm(768, 256, 320, m, n, k, 5, 8, 14, acc=True),
          I.hcvt(900, 256, k, unpack=True),
          I.st(4096, 768, seed.size), I.st(4352, 900, k), I.halt()]
    want = H.matmul(H.pack(a), H.pack(b), seed[:, :n])
    want = H.matmul(H.pack(a), H.pack(b), want)
    return p, img, want, seed


def test_half_instructions_reference_and_waits():
    p, img, want, seed = program()
    s = Machine(CFG, [p], [img]).run().slices[0]
    got = s.tget(768 + np.arange(3)[:, None] * 14 + np.arange(11))
    np.testing.assert_array_equal(F.bits(got), F.bits(want))
    np.testing.assert_array_equal(s.tget(768 + np.arange(3)[:, None] * 14 + np.arange(11, 14)), seed[:, 11:])
    assert s.tget([2112, 2113]).tolist() == [32.0, 32.0]
    R = [(0, 0)] * 16
    fp = waits.footprint(I.hmm(768, 256, 320, 3, 11, 7, 5, 8, 14, acc=True), R)
    assert (waits.SP_TMEM, 768, 807) in fp.wr
    assert (waits.SP_TMEM, 256, 270) in fp.rd
    assert (waits.SP_TMEM, 320, 374) in fp.rd


def test_half_capability_and_invalid_operands():
    for ins in [I.hcvt(32, 0, 3), I.hmm(32, 0, 8, 1, 3, 2)]:
        with pytest.raises(SimError, match="no FP16"):
            Machine(replace(CFG, FP16=False), [[ins, I.halt()]], [np.zeros(65536, np.uint8)]).run()
        assert I.Instr.decode(ins.encode()).encode() == ins.encode()
    for ins in [I.hcvt(1, 0, 3), I.hmm(4, 0, 8, 2, 3, 2), I.hcvt(4095, 0, 4)]:
        with pytest.raises(SimError, match="overlap|out of range"):
            Machine(CFG, [[ins, I.halt()]], [np.zeros(65536, np.uint8)]).run()
    for call in [lambda: I.hmm(0, 10, 20, 1, 3, 4, ars=1), lambda: I.hcvt(0, 10, 65536)]:
        with pytest.raises(ValueError):
            call()
    # A disabled extension preserves every old target's architecture identity.
    old = replace(CFG, FP16=False)
    assert target.arch_hash(old) != target.arch_hash(CFG)
    v80 = target.load("v80:fp16")
    assert target.config(v80).FP16 and not target.config("v80").FP16
    assert target.rtl_params(v80)["FP16"] == target.sim_uarch(v80)["FP16"] == 1


@pytest.mark.parametrize("wc,banked", [(False, False), (True, False), (True, True)])
def test_half_instructions_rtl(have_verilator, tmp_path, wc, banked):
    cfg = replace(CFG, SEQ_WC=wc, WC_WIN=16)
    p, img, _, _ = program()
    ref = Machine(cfg, [p], [img.copy()]).run().slices[0]
    uarch = {"SEQ_WC": int(wc), "WIN": 16}
    if banked:
        uarch.update(TMEM3)
    dram, tmem, stats = rtlsim.run(cfg, [p], [img.copy()], keep=tmp_path,
                                  uarch=uarch, max_cycles=100000)
    np.testing.assert_array_equal(tmem[0], ref.tmem)
    np.testing.assert_array_equal(dram[0], ref.dram)


@pytest.mark.parametrize("banked", [False, True])
def test_half_edges_rtl(have_verilator, tmp_path, banked):
    cfg = replace(CFG, SEQ_WC=True, WC_WIN=16)
    img = np.zeros(cfg.DRAM_BYTES, np.uint8)
    h = np.array([0, 0x8000, 1, 0x3ff, 0x400, 0x3c00, 0xbc00, 0x7bff,
                  0xfbff, 0x7c00, 0xfc00, 0x7e01, 0xffff, 0x3555, 2, 0x8001, 0x4000], np.uint16)
    x = H.unpack(h)
    img[:x.nbytes] = x.view(np.uint8)
    seed = np.resize(np.array([0x80000000, 0x7fc01234, 1, 0x3f800000], np.uint32), 24)
    img[512:608] = seed.view(np.uint8)
    p = [I.ld(0, 0, len(h)), I.ld(512, 256, len(seed)), I.li(3, 1000),
         I.hcvt(64, 0, len(h)), I.hcvt(96, 64, len(h), unpack=True),
         I.hmm(200, 4095, 4095, 2, 11, 0, ors=13),
         I.hmm(256, 4095, 4095, 2, 11, 0, ors=13, acc=True),
         I.hcvt(0xffffffff, 0xffffffff, 0),
         I.hmm(0xffffffff, 0xffffffff, 0xffffffff, 0, 11, 3),
         I.hmm(0xffffffff, 0xffffffff, 0xffffffff, 2, 0, 3), I.halt()]
    # HCVT count is immediate even when rc is nonzero.
    p[4].rc = 3
    ref = Machine(cfg, [p], [img.copy()]).run().slices[0]
    assert ref.tmem[72] >> 16 == 0
    np.testing.assert_array_equal(ref.tmem[256:280], seed)
    uarch = {"SEQ_WC": 1, "WIN": 16, **(TMEM3 if banked else {})}
    dram, tmem, _ = rtlsim.run(cfg, [p], [img], keep=tmp_path, uarch=uarch, max_cycles=100000)
    np.testing.assert_array_equal(tmem[0], ref.tmem)
    np.testing.assert_array_equal(dram[0], ref.dram)


def test_half_read_replay_with_vpu(have_verilator, tmp_path):
    cfg = replace(CFG, SEQ_WC=True, WC_WIN=16)
    img = np.zeros(cfg.DRAM_BYTES, np.uint8)
    img[:2048] = np.full(512, 0x3c003c00, np.uint32).view(np.uint8)
    img[4096:6144] = np.arange(512, dtype=np.float32).view(np.uint8)
    p = [I.ld(0, 0, 512), I.ld(4096, 1024, 512), I.ld(4096, 2048, 512),
         I.hmm(768, 0, 128, 6, 19, 31)]
    p += [I.vop(I.V_ADD, 3072, 1024, 2048, 1, 512, 512, 512, 512, I.B_FULL, 0.0)
          for _ in range(8)]
    p.append(I.halt())
    ref = Machine(cfg, [p], [img.copy()]).run().slices[0]
    dram, tmem, stats = rtlsim.run(cfg, [p], [img], keep=tmp_path, trace=True,
                                  uarch={"SEQ_WC": 1, "WIN": 16, **TMEM3}, max_cycles=100000)
    np.testing.assert_array_equal(tmem[0], ref.tmem)
    np.testing.assert_array_equal(dram[0], ref.dram)
    replay = re.search(r"TMEMSTAT .* rpl= ?(\d+),", stats["trace"])
    assert replay and int(replay[1]) > 0, "the FP16 port must actually replay a conflicted read"


def test_half_opcodes_refused_by_disabled_rtl(have_verilator, tmp_path):
    cfg = replace(CFG, FP16=False, SEQ_WC=True, WC_WIN=16)
    for i, ins in enumerate([I.hcvt(32, 0, 4), I.hmm(32, 0, 8, 1, 4, 4)]):
        _, tmem, stats = rtlsim.run(cfg, [[ins, I.halt()]], [np.zeros(cfg.DRAM_BYTES, np.uint8)],
                                    keep=tmp_path / str(i), error_ok=True,
                                    uarch={"SEQ_WC": 1, "WIN": 16}, max_cycles=10000)
        assert stats["error"] and not tmem[0].any()


def test_integer_path_with_half_disabled(have_verilator, tmp_path):
    cfg = replace(CFG, FP16=False, SEQ_WC=True, WC_WIN=16)
    p, img, _, _ = program()
    p = [ins for ins in p if ins.op not in (I.HMM, I.HCVT)]
    ref = Machine(cfg, [p], [img.copy()]).run().slices[0]
    dram, tmem, _ = rtlsim.run(cfg, [p], [img], keep=tmp_path,
                              uarch={"SEQ_WC": 1, "WIN": 16}, max_cycles=100000)
    np.testing.assert_array_equal(tmem[0], ref.tmem)
    np.testing.assert_array_equal(dram[0], ref.dram)


def test_half_board_capability_rtl(have_verilator):
    from opentpu.host import regs as R
    from opentpu.host.board import SimTransport
    t = SimTransport(ch_bytes=1 << 16, params={"FP16": 1, "TMEM_WORDS": 4096,
                                               "IMEM_WORDS": 4096})
    ident, caps = t.reg_read_many([R.R_ID, R.R_CAPS2])
    assert ident == R.ID_OTPU and caps & R.CAP2_FP16


@pytest.mark.parametrize("wc,banked", [(False, False), (True, True)])
def test_half_raw_encodings_and_reused_buffers(have_verilator, tmp_path, wc, banked):
    """NaNs/infinities/subnormals, odd row padding, long K and TMEM reuse hazards."""
    cfg = replace(CFG, SEQ_WC=wc, WC_WIN=16)
    rng = np.random.default_rng(1635)
    img = np.zeros(cfg.DRAM_BYTES, np.uint8)
    p, at = [], 0
    for case, k in enumerate([1, 2, 7, 16, 31, 65, 3, 17]):
        m, n = int(rng.integers(1, 6)), int(rng.integers(1, 24))
        ars, brs, ors = (k + 1) // 2 + 2, (n + 1) // 2 + 1, n + 3
        a = rng.integers(0, 65536, (m, ars * 2), np.uint16)
        b = rng.integers(0, 65536, (k, brs * 2), np.uint16)
        # An ignored high halfword must not poison a finite output.
        if k % 2:
            a[:, k] = 0x7e00
        if n % 2:
            b[:, n] = 0x7e00
        seed = rng.normal(size=(m, ors)).astype(np.float32)
        for data, dst in [(a, 0), (b, 512), (seed, 2048)]:
            raw = data.view(np.uint8).ravel()
            img[at:at + len(raw)] = raw
            p.append(I.ld(at, dst, len(raw) // 4))
            at = (at + len(raw) + 127) // 128 * 128
        p += [I.hmm(2048, 0, 512, m, n, k, ars, brs, ors, acc=bool(case % 2)),
              I.st(50000 + case * 1024, 2048, m * ors)]
    assert at < 50000
    p.append(I.halt())
    ref = Machine(cfg, [p], [img.copy()]).run().slices[0]
    dram, tmem, _ = rtlsim.run(cfg, [p], [img], keep=tmp_path,
                              uarch={"SEQ_WC": int(wc), "WIN": 16, **(TMEM3 if banked else {})},
                              max_cycles=1000000)
    np.testing.assert_array_equal(tmem[0], ref.tmem)
    np.testing.assert_array_equal(dram[0], ref.dram)


@pytest.mark.parametrize("lanes", [2, 8, 16])
def test_half_activation_reuse_reset_rtl(have_verilator, lanes):
    exe = rtlsim.build("tb_hmm", [rtlsim.RTL / "vpu/otpu_fp.sv", rtlsim.RTL / "vpu/otpu_fpipe.sv",
        rtlsim.RTL / "top/otpu_pkg.sv", rtlsim.RTL / "mxu/otpu_hdot.sv",
        rtlsim.RTL / "mxu/otpu_hmxu.sv", rtlsim.TB / "tb_hmm.sv"], {"N": lanes})
    result = rtlsim.run_sim([str(exe)], timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "HMM reset/restart cases=100 errors=0" in result.stdout
