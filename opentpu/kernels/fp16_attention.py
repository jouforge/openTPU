"""Native FP16 attention over a packed K-transpose / token-major V cache.

QK and P.V use binary16 operands and ordered FP32 accumulation. Softmax works
in FP32 in DRAM scratch, then P rounds to half at the P.V boundary. This is a
bounded-memory reference schedule, with no online rescaling of rounded P tiles.
"""
import math
from dataclasses import replace

from .. import language as ol
from .fp16 import _apart, _matrix, _tiles, matmul


def append(key, value, kt, v, pos: int, block: int = 16) -> None:
    """Append FP32 key[1,D], value[1,Dv] to packed kt[D,cap], v[cap,Dv].

    pos is a static token index; cap and Dv must be even. K updates a pair of
    token halfwords, preserving the neighbour; V writes one complete row. Cache
    storage must be initialized (zero at creation), disjoint from the inputs and
    each other. Inputs are already normalized/rotated as required by the model.
    """
    k, val, keys, values = _matrix(key), _matrix(value), _matrix(kt, True), _matrix(v, True)
    d, cap = keys.shape
    dv = values.shape[1]
    if k.shape != (1, d) or val.shape != (1, dv) or values.shape[0] != cap or \
            cap % 2 or dv % 2:
        raise ol.CompileError("FP16 cache append requires [1,D], [1,Dv], [D,cap], [cap,Dv]; cap/Dv even")
    if not isinstance(pos, int) or not 0 <= pos < cap:
        raise ol.CompileError("FP16 cache append requires a static position within capacity")
    if not isinstance(block, int) or not 0 < block <= 32767:
        raise ol.CompileError("FP16 cache append block must be in 1..32767")
    for src in (k, val):
        _apart(src, keys)
        _apart(src, values)
    _apart(keys, values)
    pair, lane = pos // 2 * 2, pos % 2

    def part(row, n):
        dst = kt[row:row + n, pair:pair + 2]
        old = ol.unpack_fp16(ol.load_fp16(dst))
        old[:, lane:lane + 1].set(ol.load(key[0, row:row + n]).column())
        ol.store_fp16(dst, ol.pack_fp16(old))

    for row, n in _tiles(d, block):
        part(row, n)
    ol.store_fp16(v[pos:pos + 1, :], ol.pack_fp16(ol.load(value)))


def attend(q, kt, v, out, scores, mask=None, block=(4, 32, 32), softmax_block=32) -> None:
    """One head's q[M,D], kt[D,T], v[T,Dv] -> FP32 out[M,Dv].

    q is FP32 or packed half, caches packed half; scores is FP32 [M,T] scratch,
    left holding normalized probabilities. Optional FP32 mask[M,T] is +inf for
    valid tokens, -inf for masked ones (MIN, applied after QK scaling). Each row
    must have at least one valid token. Inputs must be finite; the caller limits
    cache views to initialized tokens. GQA reuses a KV head for its query heads.

    Q rounds before QK; its FP32 result scales by 1/sqrt(D). Exponentials are
    exp2((score-max)*log2(e)). Each softmax chunk sums with the VPU's interleaved
    reduction; partial sums add in token order. softmax_block therefore forms
    part of the numerical contract. Normalized P rounds once before P.V, whose
    FP32 accumulator carries across token tiles in increasing order.
    """
    qt = _matrix(q, isinstance(q, ol.HalfTensor))
    keys, values = _matrix(kt, True), _matrix(v, True)
    dst, work = _matrix(out), _matrix(scores)
    m, d = qt.shape
    t = keys.shape[1]
    if keys.shape[0] != d or values.shape[0] != t or work.shape != (m, t) or \
            dst.shape != (m, values.shape[1]):
        raise ol.CompileError("FP16 attention requires compatible Q, K, V, output and score shapes")
    if not isinstance(softmax_block, int) or not 0 < softmax_block <= 65535:
        raise ol.CompileError("FP16 softmax block must be in 1..65535")
    sources = [qt, keys, values]
    if mask is not None:
        mt = _matrix(mask)
        if mt.shape != (m, t):
            raise ol.CompileError("FP16 attention mask must be [M,T]")
        sources.append(mt)
    for src in sources:
        _apart(src, work)
        _apart(src, dst)
    _apart(work, dst)
    matmul(q, kt, scores, block)
    # matmul registered the host scratch's complete range; subsequent stores
    # overwrite it in hardware loops, leaving the final P values there.
    scores = replace(scores, host=None)

    def row(r):
        maximum = ol.full((1,), float("-inf"))
        total = ol.zeros((1,))

        def scale(c, n):
            view = scores[r:r + 1, c:c + n]
            x = ol.load(view) * (1.0 / math.sqrt(d))
            if mask is not None:
                x = ol.minimum(x, ol.load(mask[r:r + 1, c:c + n]))
            maximum.set(ol.maximum(maximum, ol.max(x, axis=1)))
            ol.store(view, x)

        def exp(c, n):
            view = scores[r:r + 1, c:c + n]
            x = ol.exp2((ol.load(view) - maximum[:, None]) * ol.LOG2E)
            total.set(total + ol.sum(x, axis=1))
            ol.store(view, x)

        for c, n in _tiles(t, softmax_block):
            scale(c, n)
        for c, n in _tiles(t, softmax_block):
            exp(c, n)
        inverse = ol.recip(total)
        for c, n in _tiles(t, softmax_block):
            view = scores[r:r + 1, c:c + n]
            ol.store(view, ol.load(view) * inverse[:, None])

    for r, _ in _tiles(m, 1):
        row(r)
    matmul(scores, v, out, block)
