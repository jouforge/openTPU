"""Packed INT4 decode through TMEM, native half arithmetic and the sequencer."""
from dataclasses import replace

import numpy as np
import pytest

from opentpu import isa as I, fp16 as H, int4 as Q, rtlsim, target, waits
from opentpu.isasim import Machine, SimError
from test_fp16_isa import CFG as HALF_CFG, TMEM3

CFG = replace(HALF_CFG, W4A16=True, SEQ_WC=True, WC_WIN=16)


def program():
    rng = np.random.default_rng(48)
    img = np.zeros(CFG.DRAM_BYTES, np.uint8)
    p, expected, at = [], [], 0
    for case, n in enumerate((1, 2, 7, 8, 9, 17, 31, 257)):
        q = np.resize(np.arange(-8, 8, dtype=np.int8), n)
        scale = rng.integers(0, 65536, (n + 1) // 2 * 2, np.uint16)
        edge = np.array([0, 0x8000, 1, 0x8001, 0x3ff, 0x400, 0x3c00, 0x7bff,
                         0x7c00, 0xfc00, 0x7fff, 0xffff], np.uint16)
        scale[:min(n, len(edge))] = edge[:n]
        code = np.zeros((n + 7) // 8, np.uint32)
        for j in range(n):
            code[j // 8] |= np.uint32(int(q[j]) & 15) << np.uint32(4 * (j % 8))
        for values, dst in ((code, 0), (scale, 128)):
            raw = values.view(np.uint8)
            img[at:at + len(raw)] = raw
            p.append(I.ld(at, dst, len(raw) // 4))
            at += 1024
        p += [I.li(1, 512), I.li(2, 128), I.li(3, 0),
              I.hdq(0, 0, 0, n, ra=1, rb=3, rc=2),
              I.hcvt(1024, 512, n, unpack=True),
              I.st(32768 + case * 2048, 512, (n + 1) // 2)]
        h = Q.dequantize(q, scale[:n])
        expected.append(h)
    # Decode weights then consume them through HMM; all three read operands are
    # live concurrently. Counts remain immediate, even when rd is nonzero.
    p += [I.hcvt(768, 1024, 17), I.hmm(1536, 768, 512, 1, 17, 1),
          I.hdq(0xffffffff, 0xffffffff, 0xffffffff, 0), I.halt()]
    return p, img, expected


def test_decode_reference_and_footprints():
    p, img, expected = program()
    s = Machine(CFG, [p], [img]).run().slices[0]
    for i, want in enumerate(expected):
        at = 32768 + i * 2048
        got = s.dram[at:at + ((len(want) + 1) // 2) * 4].view(np.uint16)
        np.testing.assert_array_equal(got[:len(want)], want)
        if len(want) % 2:
            assert got[-1] == 0
    ins = I.hdq(0, 0, 0, 17, ra=1, rb=2, rc=3)
    assert I.Instr.decode(ins.encode()).encode() == ins.encode()
    r = [(0, 0), (512, 512), (0, 0), (128, 128)] + [(0, 0)] * 12
    f = waits.footprint(ins, r)
    assert f.rd == ((waits.SP_TMEM, 0, 3), (waits.SP_TMEM, 128, 137))
    assert f.wr == ((waits.SP_TMEM, 512, 521),)


def test_decode_rejections_and_hash():
    img = np.zeros(CFG.DRAM_BYTES, np.uint8)
    with pytest.raises(SimError, match="no W4A16"):
        Machine(HALF_CFG, [[I.hdq(32, 0, 8, 3), I.halt()]], [img]).run()
    for ins in (I.hdq(0, 0, 8, 3), I.hdq(8, 0, 8, 3), I.hdq(4095, 0, 8, 3)):
        with pytest.raises(SimError, match="overlap|out of range"):
            Machine(CFG, [[ins, I.halt()]], [img]).run()
    for field, value in (("flags", 1), ("w", [32, 0, 8, 3, 1, 0, 0])):
        ins = I.hdq(32, 0, 8, 3)
        setattr(ins, field, value)
        with pytest.raises(SimError, match="invalid"):
            Machine(CFG, [[ins, I.halt()]], [img]).run()
    assert target.arch_hash(CFG) != target.arch_hash(replace(CFG, W4A16=False))
    assert f'{target.arch_hash(target.config("v80:fp16")):08x}' == 'b4ac7354'
    with pytest.raises(ValueError):
        I.hdq(0, 0, 0, 65536)


@pytest.mark.parametrize("lanes,banked,wc", [(4, False, False), (4, False, True), (8, True, True)])
def test_decode_rtl(have_verilator, tmp_path, lanes, banked, wc):
    p, img, _ = program()
    cfg = replace(CFG, SEQ_WC=wc)
    ref = Machine(cfg, [p], [img.copy()]).run().slices[0]
    dram, tmem, _ = rtlsim.run(cfg, [p], [img], keep=tmp_path,
        uarch={"SEQ_WC": int(wc), "WIN": 16, "ULANES": lanes, "MXU_E": min(lanes, 8), **(TMEM3 if banked else {})},
        max_cycles=200000)
    np.testing.assert_array_equal(tmem[0], ref.tmem)
    np.testing.assert_array_equal(dram[0], ref.dram)


def test_decode_disabled_rtl(have_verilator, tmp_path):
    cfg = replace(CFG, W4A16=False)
    _, tmem, stats = rtlsim.run(cfg, [[I.hdq(32, 0, 8, 3), I.halt()]],
        [np.zeros(cfg.DRAM_BYTES, np.uint8)], keep=tmp_path, error_ok=True,
        uarch={"SEQ_WC": 1, "WIN": 16}, max_cycles=10000)
    assert stats["error"] and not tmem[0].any()


@pytest.mark.parametrize("lanes", [2, 8])
def test_decode_reset_rtl(have_verilator, lanes):
    exe = rtlsim.build("tb_hdq", [rtlsim.RTL / "vpu/otpu_fp.sv", rtlsim.RTL / "vpu/otpu_fpipe.sv",
        rtlsim.RTL / "top/otpu_pkg.sv", rtlsim.RTL / "mxu/otpu_hdot.sv",
        rtlsim.RTL / "mxu/otpu_hmxu.sv", rtlsim.TB / "tb_hdq.sv"], {"N": lanes})
    result = rtlsim.run_sim([str(exe)], timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "HDQ reset/restart cases=100 errors=0" in result.stdout


def test_board_diagnostic_programs():
    from opentpu.host.opchecks import diag_image, op_checks, OUT
    cfg = target.config("v80:mixed", DRAM_BYTES=1 << 23)
    image = diag_image(cfg)
    checks = [(n, p) for g, n, p in op_checks(cfg) if g == "w4a16"]
    assert len(checks) == 2
    assert not any(g == "w4a16" for g, _, _ in op_checks(replace(cfg, W4A16=False)))
    for name, program in checks:
        result = Machine(cfg, [program], [image.copy()]).run().slices[0].dram
        assert np.any(result[OUT:OUT + 64]), name
