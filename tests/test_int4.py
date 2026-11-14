"""W4A16 checkpoint encoding and the independently rounded half decoder."""
import numpy as np
import pytest

from opentpu import fp16 as H, fp32 as F, int4 as Q


def test_decoder_all_codes_and_scales():
    # Every half encoding, including subnormal, signed zero, infinity and NaN.
    scales = np.arange(65536, dtype=np.uint16)
    for code in range(-8, 8):
        with np.errstate(invalid="ignore", over="ignore"):
            values = scales.view(np.float16).astype(np.float32) * np.float32(code)
            expected = values.astype(np.float16).view(np.uint16)
        expected = np.where(np.isnan(values), 0x7e00, expected).astype(np.uint16)
        np.testing.assert_array_equal(Q.dequantize(np.int8(code), scales), expected)


@pytest.mark.parametrize("shape,group", [((1, 1), 8), ((17, 131), 128), ((1031, 19), 16)])
def test_grouped_layout(shape, group):
    w = np.random.default_rng(27).normal(size=shape).astype(np.float32)
    w[0] = 0
    q = Q.quantize(w, group)
    n, k = shape
    assert q.words.dtype == np.dtype("<u4") and q.scales.dtype == np.dtype("<u2")
    assert q.nbytes == k * ((n + 7) // 8) * 4 + ((k + group - 1) // group) * ((n + 1) // 2) * 4
    assert np.all(q.scales[:, n:] == 0)
    if n % 8:
        assert np.all(q.words[:, -1] >> (4 * (n % 8)) == 0)
    want = np.zeros((k, n), np.uint16)
    for col in range(n):
        for start in range(0, k, group):
            x = w[col, start:start + group]
            peak = np.max(np.abs(x))
            s = np.float16(np.float32(peak / np.float32(7)))
            if peak:
                s = np.clip(s, np.float16(2**-24), np.float16(9352))
                code = np.clip(np.rint(x.astype(np.float64) / np.float64(s)), -7, 7).astype(np.int8)
            else:
                code = np.zeros(len(x), np.int8)
            np.testing.assert_equal(q.scales[start // group, col], s.view(np.uint16))
            want[start:start + len(x), col] = (code.astype(np.float32) * np.float32(s)).astype(np.float16).view(np.uint16)
    np.testing.assert_array_equal(q.unpack(), want)


def test_rounding_tails_and_range():
    w = np.array([[0, .5, 1.5, 2.5, 3.5, -2.5, -3.5, 7, 65504],
                  [2**-24, -2**-24, 2**-25, -2**-25, 0, 0, 0, 0, -65504]], np.float32)
    q = Q.quantize(w, 8)
    np.testing.assert_array_equal(H.unpack(q.unpack()[:8, 0]), [0, 0, 2, 2, 4, -2, -4, 7])
    assert q.scales[0, 1] == 1 and np.isfinite(H.unpack(q.unpack())).all()
    assert q.scales[1, 0] == H.pack(np.float32(9352))
    assert H.unpack(q.unpack())[-1, 0] == 65472


def test_matmul_accumulation_boundary():
    rng = np.random.default_rng(2)
    a, w = rng.normal(size=(3, 19)).astype(np.float32), rng.normal(size=(11, 19)).astype(np.float32)
    q = Q.quantize(w, 8)
    seed = rng.normal(size=(3, 11)).astype(np.float32)
    original = seed.copy()
    want = seed.copy()
    x, b = a.astype(np.float16).astype(np.float32), q.unpack().view(np.float16).astype(np.float32)
    for k in range(19):
        want = F.add(want, F.mul(x[:, k, None], b[k]))
    np.testing.assert_array_equal(F.bits(Q.matmul(a, q, seed)), F.bits(want))
    np.testing.assert_array_equal(seed, original)


@pytest.mark.parametrize("w,group", [(np.zeros((0, 2)), 8), ([1, 2], 8),
    ([[np.nan]], 8), ([[np.inf]], 8), ([[65505]], 8), ([[1]], 0), ([[1]], 7), ([[1]], 256)])
def test_bad_weights(w, group):
    with pytest.raises(ValueError):
        Q.quantize(w, group)


@pytest.mark.parametrize("q", [-9, 8, 1.5])
def test_bad_codes(q):
    with pytest.raises(ValueError):
        Q.dequantize(q, H.pack(1))
