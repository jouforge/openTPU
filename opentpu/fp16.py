"""FP16 operands and FP32 accumulation (docs/fp16.md).

Packed operands are uint16 binary16 words; results of unpack, mul and matmul are
float32. Conversion preserves binary16 subnormals, rounds to nearest even and
canonicalizes NaNs. FP32 sums use the existing openTPU arithmetic, including FTZ.
"""
from __future__ import annotations

import numpy as np

from . import fp32 as F


def pack(x) -> np.ndarray:
    """FP32 -> binary16 bits, RNE with gradual underflow and signed infinity on overflow."""
    u = F.bits(x)
    e, m = (u >> 23) & 255, u & 0x7FFFFF
    sign = (u >> 16) & 0x8000
    # Normal half results: rounding the discarded 13 bits may carry into the exponent.
    q = ((u & 0x7FFFFFFF) + 0xFFF + ((u >> 13) & 1)) >> 13
    normal = ((q.astype(np.int64) - 0x1C000) & 0x7FFF).astype(np.uint32)
    # e = 102 includes the tie at 2^-25; e = 112 can round up to the smallest normal.
    sh = np.clip(126 - e.astype(np.int32), 14, 24).astype(np.uint32)
    sig = m | 0x800000
    lo, half = sig & ((np.uint32(1) << sh) - 1), np.uint32(1) << (sh - 1)
    sub = (sig >> sh) + ((lo > half) | ((lo == half) & ((sig >> sh) & 1 != 0)))
    mag = np.where(e < 102, 0, np.where(e < 113, sub,
                   np.where(e < 143, normal, 0x7C00)))
    return np.where((e == 255) & (m != 0), 0x7E00, sign | mag).astype(np.uint16)


def unpack(h) -> np.ndarray:
    """binary16 bits -> exact FP32, including subnormals; every NaN becomes 0x7FC00000."""
    u = np.asarray(h, np.uint16).astype(np.uint32)
    e, m = (u >> 10) & 31, u & 1023
    sign = (u & 0x8000) << 16
    normal = ((e + 112) << 23) | (m << 13)
    sub = F.bits(m.astype(np.float32) * np.float32(2.0 ** -24))
    mag = np.where(e == 0, sub, np.where(e == 31, 0x7F800000, normal))
    return F.from_bits(np.where((e == 31) & (m != 0), 0x7FC00000, sign | mag))


def mul(a, b) -> np.ndarray:
    """The exact product of two finite binary16 operands in FP32 (no FP16 product rounding)."""
    return F.mul(unpack(a), unpack(b))


def madd(a, b, c) -> np.ndarray:
    """binary16 a * b + FP32 c, with one FP32 addition rounding and the existing FTZ policy."""
    return F.add(mul(a, b), c)


def matmul(a, b, acc=None) -> np.ndarray:
    """Packed binary16 A[M,K] @ B[K,N], FP32 accumulation in increasing K order.

    Each output starts at +0, or its FP32 acc value. Empty K returns that initial
    value unchanged. This is the numerical reference, not a model's host fallback.
    """
    a, b = np.asarray(a, np.uint16), np.asarray(b, np.uint16)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0]:
        raise ValueError("FP16 matmul needs A[M,K] and B[K,N]")
    shape = (a.shape[0], b.shape[1])
    out = np.zeros(shape, np.float32) if acc is None else F.f32(acc).copy()
    if out.shape != shape:
        raise ValueError("FP16 matmul accumulator must have shape [M,N]")
    for k in range(a.shape[1]):
        out = madd(a[:, k, None], b[None, k, :], out)
    return out


def attention(q, kt, v, mask=None, softmax_block=32) -> tuple[np.ndarray, np.ndarray]:
    """Packed Q[M,D], K^T[D,T], V[T,Dv] -> (FP32 output, FP32 probabilities).

    Reference for kernels.fp16_attention.attend: scale QK after accumulation,
    stable FP32 softmax, then round normalized P to half before ordered P.V.
    mask is +inf/-inf, with at least one unmasked token per row. Reductions use
    the VPU's partial sums within a chunk, then ordered addition across chunks.
    """
    q, kt, v = (np.asarray(x, np.uint16) for x in (q, kt, v))
    if any(x.ndim != 2 or 0 in x.shape for x in (q, kt, v)) or \
            q.shape[1] != kt.shape[0] or kt.shape[1] != v.shape[0]:
        raise ValueError("FP16 attention requires nonempty [M,D], [D,T], [T,Dv]")
    if not isinstance(softmax_block, int) or not 0 < softmax_block <= 65535:
        raise ValueError("FP16 softmax block must be in 1..65535")
    if any(np.any((x & 0x7C00) == 0x7C00) for x in (q, kt, v)):
        raise ValueError("FP16 attention requires finite operands")
    scores = F.mul(matmul(q, kt), np.float32(1.0 / np.sqrt(q.shape[1])))
    if mask is not None:
        mask = F.f32(mask)
        if mask.shape != scores.shape or not np.isinf(mask).all() or \
                not np.any(mask > 0, axis=1).all():
            raise ValueError("FP16 attention mask must be +/-inf with valid tokens in every row")
        scores = F.fmin(scores, mask)
    p = F.exp2(F.mul(F.sub(scores, F.chain_max(scores)[:, None]),
                    np.float32(1.0 / np.log(2.0))))
    total = np.zeros(q.shape[0], np.float32)
    for t in range(0, kt.shape[1], softmax_block):
        total = F.add(total, F.interleaved_sum(p[:, t:t + softmax_block], F.RED_PARTIALS))
    p = F.mul(p, F.recip(total)[:, None])
    return matmul(pack(p), v), p
