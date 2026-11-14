"""Binary16 numerical contract and native RTL multiply/add, independently checked with NumPy."""
import numpy as np
import pytest

from opentpu import fp16 as H, fp32 as F, rtlsim


def _canonical(x):
    u = np.asarray(x, np.float32).view(np.uint32)
    return np.where(((u >> 23) & 255 == 255) & (u & 0x7FFFFF != 0), 0x7FC00000, u)


def _pack_cases():
    # Every positive half interval's midpoint, its FP32 neighbours, and both signs.
    h = np.arange(0x7C00, dtype=np.uint16).view(np.float16).astype(np.float32)
    mid = (h[:-1] + h[1:]) * np.float32(0.5)
    mid = np.concatenate([mid, np.float32([65520, 2.0 ** -25, 2.0 ** -14])])
    x = np.concatenate([mid, np.nextafter(mid, np.float32(-np.inf)),
                        np.nextafter(mid, np.float32(np.inf))])
    rng = np.random.default_rng(16)
    raw = rng.integers(0, 1 << 32, 100000, dtype=np.uint32).view(np.float32)
    return np.concatenate([x, -x, raw, F.from_bits(np.uint32([
        0, 0x80000000, 0x7F800000, 0xFF800000, 0x7F800001, 0xFFC00000]))])


def _product_cases():
    # Every half encoding appears in each operand, plus all pairs of boundary classes
    # and random pairs across the full exponent/sign range.
    all_h = np.arange(65536, dtype=np.uint16)
    edge = np.uint16([0, 1, 2, 0x3FF, 0x400, 0x401, 0x3BFF, 0x3C00, 0x3C01,
                      0x7BFE, 0x7BFF, 0x7C00, 0x7C01, 0x7E00, 0x7FFF])
    edge = np.concatenate([edge, edge | 0x8000])
    a, b = np.meshgrid(edge, edge)
    rng = np.random.default_rng(17)
    a = np.concatenate([all_h, a.ravel(), rng.integers(0, 65536, 100000, dtype=np.uint16)])
    b = np.concatenate([all_h ^ 0xABCD, b.ravel(),
                        rng.integers(0, 65536, 100000, dtype=np.uint16)])
    return a, b


def test_unpack_all_encodings_and_roundtrip():
    h = np.arange(65536, dtype=np.uint16)
    with np.errstate(invalid="ignore"):
        expected = _canonical(h.view(np.float16).astype(np.float32))
    np.testing.assert_array_equal(F.bits(H.unpack(h)), expected)
    nan = ((h & 0x7C00) == 0x7C00) & ((h & 0x3FF) != 0)
    np.testing.assert_array_equal(H.pack(H.unpack(h)), np.where(nan, 0x7E00, h))


def test_pack_all_rounding_boundaries():
    x = _pack_cases()
    with np.errstate(over="ignore", invalid="ignore"):
        expected = x.astype(np.float16).view(np.uint16)
    expected = np.where(np.isnan(x), 0x7E00, expected)
    np.testing.assert_array_equal(H.pack(x), expected)
    for i in [0, 100, -1]:
        assert H.pack(x[i]) == expected[i]     # scalar and vector paths have the same bits


def test_products_and_fp32_accumulation():
    a, b = _product_cases()
    with np.errstate(invalid="ignore"):
        expected = a.view(np.float16).astype(np.float64) * b.view(np.float16).astype(np.float64)
    np.testing.assert_array_equal(F.bits(H.mul(a, b)), _canonical(expected))
    # A half product must not round to half before its FP32 sum: 65504^2 stays finite;
    # the smallest half subnormal squared stays nonzero.
    np.testing.assert_array_equal(H.mul(np.uint16([0x7BFF, 1]), np.uint16([0x7BFF, 1])),
                                  np.float32([65504.0 ** 2, 2.0 ** -48]))


@pytest.mark.parametrize("m,k,n", [(0, 3, 2), (3, 0, 2), (1, 1, 1), (3, 17, 5), (2, 129, 7)])
def test_matrix_order_and_accumulate(m, k, n):
    rng = np.random.default_rng(18)
    a = rng.normal(size=(m, k)).astype(np.float16)
    b = rng.normal(size=(k, n)).astype(np.float16)
    seed = rng.normal(size=(m, n)).astype(np.float32)
    for acc in (None, seed):
        expected = np.zeros((m, n), np.float32) if acc is None else acc.copy()
        for i in range(k):
            prod = a[:, i, None].astype(np.float64) * b[None, i, :].astype(np.float64)
            expected = (expected.astype(np.float64) + prod).astype(np.float32)
        np.testing.assert_array_equal(F.bits(H.matmul(a.view(np.uint16), b.view(np.uint16), acc)),
                                      F.bits(expected))


def test_matrix_bad_shapes():
    with pytest.raises(ValueError, match="A\\[M,K\\]"):
        H.matmul(np.zeros((2, 3), np.uint16), np.zeros((4, 2), np.uint16))
    with pytest.raises(ValueError, match="accumulator"):
        H.matmul(np.zeros((2, 3), np.uint16), np.zeros((3, 2), np.uint16), np.zeros(2))


@pytest.mark.parametrize("lm,la", [(2, 4), (3, 5)])
def test_fp16_rtl(have_verilator, tmp_path, lm, la):
    rows = []
    def add(op, a, b, c, expected):
        rows.append(np.stack(np.broadcast_arrays(np.uint32(op), np.asarray(a, np.uint32),
                         np.asarray(b, np.uint32), np.asarray(c, np.uint32), expected), axis=-1))
    h = np.arange(65536, dtype=np.uint16)
    x = _pack_cases()
    a, b = _product_cases()
    add(0, h, 0, 0, F.bits(H.unpack(h)))
    add(1, F.bits(x), 0, 0, H.pack(x).astype(np.uint32))
    add(2, a, b, 0, F.bits(H.mul(a, b)))
    rng = np.random.default_rng(19)
    c = F.from_bits(rng.integers(0, 1 << 32, len(a), dtype=np.uint32))
    # Cancellation, signed zeros and special values, as well as arbitrary FP32 accumulators.
    c[:65536] = -H.mul(a[:65536], b[:65536])
    add(3, a, b, F.bits(c), F.bits(H.madd(a, b, c)))
    vec = tmp_path / "fp16.hex"
    np.savetxt(vec, np.concatenate(rows), fmt="%08x")
    exe = rtlsim.build("tb_fp16", [rtlsim.RTL / "vpu/otpu_fp.sv",
                       rtlsim.RTL / "vpu/otpu_fpipe.sv", rtlsim.TB / "tb_fp16.sv"],
                       {"LM": lm, "LA": la})
    run = rtlsim.run_sim([str(exe), f"+vec={vec}"], timeout=180)
    assert run.returncode == 0, run.stdout + run.stderr
    assert f"FP16 cases={sum(len(r) for r in rows)} errors=0" in run.stdout, run.stdout


@pytest.mark.parametrize("lanes,lm,la", [(1, 2, 4), (8, 2, 4), (3, 3, 5)])
def test_matrix_stream_rtl(have_verilator, tmp_path, lanes, lm, la):
    rng = np.random.default_rng(20)
    lines, rows = [], 0
    for k in (1, 2, 7, 16, 65, 129):
        # Four matrix rows, with seeded accumulation and arbitrary half classes.
        a = rng.integers(0, 65536, (4, k), dtype=np.uint16)
        b = rng.integers(0, 65536, (k, lanes), dtype=np.uint16)
        if k > 2:
            a[0] = H.pack(rng.normal(size=k))
            b[:, 0] = H.pack(rng.normal(size=k))
        seed = rng.normal(size=(4, lanes)).astype(np.float32)
        seed[0] = 0
        expected = F.bits(H.matmul(a, b, seed))
        for i in range(4):
            rows += 1
            for j in range(k):
                line = [int(j == 0), int(j == k - 1), int(a[i, j]), *b[j],
                        *F.bits(seed[i]), *expected[i]]
                lines.append(" ".join(f"{int(v):08x}" for v in line))
    vec = tmp_path / "hdot.hex"
    vec.write_text("\n".join(lines) + "\n")
    exe = rtlsim.build("tb_hdot", [rtlsim.RTL / "vpu/otpu_fp.sv",
                      rtlsim.RTL / "vpu/otpu_fpipe.sv", rtlsim.RTL / "mxu/otpu_hdot.sv",
                      rtlsim.TB / "tb_hdot.sv"], {"N": lanes, "LM": lm, "LA": la})
    run = rtlsim.run_sim([str(exe), f"+vec={vec}"], timeout=120)
    assert run.returncode == 0, run.stdout + run.stderr
    assert f"HDOT beats={len(lines)} rows={rows} errors=0" in run.stdout, run.stdout
