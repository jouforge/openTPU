"""Projection dispatch for W16A16, W8A8 and packed W4A16; FP32 results."""
from dataclasses import replace

from .. import language as ol, isa as I, quant as Q
from ..compiler import Affine, HalfTensor, Int4Tensor, QTensor, Tensor, current
from .fp16 import _apart, _matrix, _tiles, matmul as half_matmul, swiglu as half_swiglu
from .lib import silu


def _spans(w):
    if isinstance(w, HalfTensor):
        return (w.storage,)
    if isinstance(w, Int4Tensor):
        return w.words, w.scales.storage
    if isinstance(w, QTensor) and w.wf == I.W8:
        if w.parts or w.tiles or w.kt or w.scale is None:
            raise ol.CompileError("W8A8 mixed matmul requires a standard placed weight")
        groups = (w.shape[0] + Q.grp_r("int8", w.grp) - 1) // Q.grp_r("int8", w.grp)
        return tuple(Tensor(base, (groups, stride // 4), (stride // 4, 1))
                     for base, stride in ((w.data, w.rs), (w.scale, w.srs)))
    raise ol.CompileError("mixed matmul requires HalfWeight, Int4Weight or INT8 Weight")


def matmul(a, b, out, block=(1, 128, 128)):
    """FP32 a[M,K] times a placed weight, selected by its storage descriptor.

    W4A16 decodes one K group into TMEM, consumes it with HMM, then reuses the
    tile. W8A8 keeps the existing full-K QACT/MM numerical contract. No automatic
    precision choice, host conversion or persistent expanded-weight image.
    """
    if isinstance(b, HalfTensor):
        return half_matmul(a, b, out, block)
    at, ot = _matrix(a), _matrix(out)
    m, k = at.shape
    if len(block) != 3 or any(not isinstance(v, int) or v <= 0 for v in block) or block[1] % 8:
        raise ol.CompileError("mixed matmul requires positive tile sizes, N a multiple of 8")
    _apart(at, ot)
    if isinstance(b, Int4Tensor):
        bk, n = b.shape
        if bk != k:
            raise ol.CompileError("W4A16 matmul inner dimensions differ")
        _apart(b.words, ot)
        _apart(b.scales.storage, ot)
    elif isinstance(b, QTensor) and b.wf == I.W8:
        n, bk = b.shape
        for src in _spans(b):
            _apart(src, ot)
        if bk != k:
            raise ol.CompileError("W8A8 matmul inner dimensions differ")
    else:
        raise ol.CompileError("mixed matmul requires HalfWeight, Int4Weight or INT8 Weight")
    if ot.shape != (m, n):
        raise ol.CompileError("mixed matmul output must be [M,N]")
    host = out.host
    if host is not None and (not Affine.of(out.base).is_static or not Affine.of(out.offset).is_static):
        raise ol.CompileError("mixed matmul host output must have static addresses")
    out = replace(out, host=None)
    for row, rows in _tiles(m, block[0]):
        for col, cols in _tiles(n, block[1]):
            if isinstance(b, QTensor):
                y = ol.dot(ol.load(a[row:row + rows, :]), b[col:col + cols, :])
            else:
                y = ol.zeros((rows, cols))
                full, tail = divmod(k, b.group)

                def step(start, depth):
                    x = ol.pack_fp16(ol.load(a[row:row + rows, start:start + depth]))
                    w = current().load_int4(b, start, depth, col, cols)
                    ol.matmul_fp16(x, w, acc=y)

                for i, _ in _tiles(full, 1):
                    step(i * b.group, b.group)
                if tail:
                    step(full * b.group, tail)
            ol.store(out[row:row + rows, col:col + cols], y)
    if host is not None:
        off = Affine.of(out.offset).static()
        current().stores.extend((host, off + r * out.strides[0], n) for r in range(m))


def swiglu(a, w_gate, w_up, w_down, out, scratch, block=(1, 128, 128)):
    """Independent gate/up/down matrix modes with the existing FP32 SiLU/product."""
    if all(isinstance(w, HalfTensor) for w in (w_gate, w_up, w_down)):
        return half_swiglu(a, w_gate, w_up, w_down, out, scratch, block)
    if len(scratch) != 2:
        raise ol.CompileError("mixed swiglu requires two scratch matrices")
    g, u = scratch
    if g.shape != u.shape:
        raise ol.CompileError("mixed swiglu scratch matrices must have the same shape")
    destinations = (_matrix(g), _matrix(u), _matrix(out))
    sources = (_matrix(a), *_spans(w_gate), *_spans(w_up), *_spans(w_down))
    for i, dst in enumerate(destinations):
        for src in (*sources, *destinations[:i]):
            _apart(src, dst)
    matmul(a, w_gate, g, block)
    matmul(a, w_up, u, block)
    for row, rows in _tiles(g.shape[0], block[0]):
        for col, cols in _tiles(g.shape[1], block[1]):
            gs, us = g[row:row + rows, col:col + cols], u[row:row + rows, col:col + cols]
            ol.store(replace(gs, host=None), silu(ol.load(gs)) * ol.load(us))
    matmul(replace(g, host=None), w_down, out, block)
