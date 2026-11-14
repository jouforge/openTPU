"""Grouped signed INT4 weights for W4A16 (docs/mixed_precision.md).

This layout is independent of the integer MXU's int4 format. Weights stay packed
in DRAM; the device expands a tile to binary16 before native HMM. The functions
here define checkpoint quantization and the numerical oracle, not a host fallback.
"""
from dataclasses import dataclass

import numpy as np

from . import fp16 as H


def dequantize(q, scales):
    """Signed integer codes times binary16 scales, rounded once to binary16.

    All sixteen codes have meaning. The checkpoint quantizer emits -7..7;
    the decoder also defines -8. Scales are binary16 bit patterns, not floats.
    The exact product fits FP32 before the final RNE/gradual-underflow conversion.
    """
    q = np.asarray(q)
    if q.dtype.kind not in "iu" or np.any(q < -8) or np.any(q > 7):
        raise ValueError("INT4 codes must be integers in -8..7")
    return H.pack(H.mul(H.pack(q.astype(np.float32)), np.asarray(scales, np.uint16)))


@dataclass(frozen=True)
class Packed:
    """Logical weights [N,K], stored K-major, with group scales along K.

    words[K,ceil(N/8)] hold eight low-nibble-first two's-complement codes.
    scales[ceil(K/group),ceil(N/2)*2] hold binary16 scales; padding is zero.
    Shape and group are part of the layout/cache identity.
    """

    words: np.ndarray
    scales: np.ndarray
    shape: tuple
    group: int

    @property
    def nbytes(self):
        return self.words.nbytes + self.scales.nbytes

    def unpack(self):
        """Reference expansion to HMM's binary16 [K,N], including a short last group."""
        n, k = self.shape
        col = np.arange(n)
        q = ((self.words[:, col // 8] >> (4 * (col % 8))) & 15).astype(np.int8)
        q = np.where(q >= 8, q - 16, q).astype(np.int8)
        return dequantize(q, self.scales[np.arange(k) // self.group, :n])


def quantize(w, group=128):
    """Finite FP32 [N,K] -> symmetric signed INT4, one positive half scale per K group.

    Scale = half_RNE(fp32(max(abs(group))/7)), clamped to [2^-24, 9352]
    for nonzero groups. 9352 keeps the largest emitted code finite after half
    rounding. Codes = clip(RNE(float64(w)/float64(scale)), -7, 7). Zero groups
    have scale +0 and code 0. Input magnitudes must fit finite binary16.
    A short final group uses only its real elements; padding never sets its scale.
    """
    w = np.asarray(w, np.float32)
    if w.ndim != 2 or min(w.shape) <= 0 or not np.isfinite(w).all() or \
            np.any(np.abs(w) > 65504):
        raise ValueError("W4A16 weights must be nonempty finite [N,K] within binary16 range")
    if not isinstance(group, int) or group < 8 or group > 128 or group & (group - 1):
        raise ValueError("W4A16 group must be a power of two in 8..128")
    n, k = w.shape
    words = np.zeros((k, (n + 7) // 8), dtype="<u4")
    scales = np.zeros(((k + group - 1) // group, (n + 1) // 2 * 2), dtype="<u2")
    for start in range(0, k, group):
        # Bound the temporaries for a large vocabulary head. Chunks start at a
        # whole packed word, so their final word never overwrites another chunk.
        for row in range(0, n, 1024):
            x = w[row:row + 1024, start:start + group]
            peak = np.max(np.abs(x), axis=1)
            h = H.pack(peak / np.float32(7))
            h = np.where(peak == 0, 0, np.clip(h, 1, int(H.pack(np.float32(9352)))))
            h = h.astype(np.uint16)
            scales[start // group, row:row + len(x)] = h
            scale = H.unpack(h).astype(np.float64)
            q = np.divide(x.astype(np.float64), scale[:, None],
                          out=np.zeros(x.shape, np.float64), where=scale[:, None] != 0)
            q = np.clip(np.rint(q), -7, 7).astype(np.int8).T
            packed = np.zeros((x.shape[1], (len(x) + 7) // 8), dtype=np.uint32)
            for lane in range(8):
                part = (q[:, lane::8].astype(np.int32) & 15).astype(np.uint32)
                packed[:, :part.shape[1]] |= part << (4 * lane)
            words[start:start + x.shape[1], row // 8:(row + len(x) + 7) // 8] = packed
    return Packed(words, scales, (n, k), group)


def matmul(a, w: Packed, acc=None):
    """Reference W4A16: FP32 activations round to half, then ordered FP32 accumulation."""
    return H.matmul(H.pack(a), w.unpack(), acc)
