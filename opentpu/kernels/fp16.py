"""Native W16A16 linear algebra, bounded by tile sizes rather than matrix sizes.

The DRAM operands use HalfInput / HalfWeight's word-padded rows. FP32 inputs
are rounded on the device as each tile is loaded. Outputs and carried sums
remain FP32; no partial sum is rounded to half or reduced in another order.
"""
from dataclasses import replace

from .. import language as ol
from ..compiler import Affine, HalfTensor, Tensor, current
from .lib import silu


def _tiles(size, block):
    """Full tiles in one hardware loop, then the short tail (no padded arithmetic)."""
    full, tail = divmod(size, block)
    if full == 1:
        yield 0, block
    elif full > 1:
        for i in ol.range(full):
            yield i * block, block
    if tail:
        yield full * block, tail


def _matrix(x, half=False):
    if not isinstance(x, HalfTensor if half else Tensor):
        raise ol.CompileError("FP16 matmul requires Tensor / HalfTensor DRAM operands")
    t = x.storage if half else x
    if len(t.shape) != 2 or any(n <= 0 for n in t.shape) or \
            t.strides[-1] != 1 or t.strides[0] < t.shape[1] or t.esize != (2 if half else 4):
        raise ol.CompileError("FP16 matmul requires nonempty row-major matrices")
    base = Affine.of(t.base)
    if base.const % 4 or any(c % 4 for c in base.terms.values()) or \
            (half and t.strides[0] % 2):
        raise ol.CompileError("FP16 matmul requires word-aligned rows and addresses")
    return t


def _apart(src, dst):
    """A DRAM output must not destroy a later input tile, including row gaps."""
    delta = Affine.of(dst.base) - Affine.of(src.base)
    ss = ((src.shape[0] - 1) * src.strides[0] + src.shape[1]) * src.esize
    ds = ((dst.shape[0] - 1) * dst.strides[0] + dst.shape[1]) * dst.esize
    if delta.terms or -ds < delta.const < ss:
        raise ol.CompileError("FP16 matmul requires provably disjoint input and output storage")


def matmul(a, b: HalfTensor, out: Tensor, block=(4, 32, 32)) -> None:
    """DRAM a[M,K] @ b[K,N] -> out[M,N], tiled in M, N, K (block's order).

    a is FP32 Tensor or packed HalfTensor; b is packed (HalfWeight transposes
    conventional linear weights [N,K] at placement). Full tiles use hardware
    loops, each short tail its actual size. At most three loop levels are used.
    Every output starts at +0 and accumulates increasing K, bit for bit fp16.matmul.
    Sources may alias each other; out must be disjoint. No bias or activation is
    implicit. block bounds the working set; the caller must leave room in TMEM.
    N and K blocks must be even so packed views start on word boundaries.
    """
    if not current().cfg.FP16:
        raise ol.CompileError("FP16 matmul requires Config.FP16")
    at = _matrix(a, isinstance(a, HalfTensor))
    bt, ot = _matrix(b, True), _matrix(out)
    m, k = at.shape
    n = bt.shape[1]
    if bt.shape[0] != k or ot.shape != (m, n):
        raise ol.CompileError("FP16 matmul requires [M,K], [K,N] and [M,N]")
    if len(block) != 3 or any(not isinstance(v, int) or not 0 < v <= 65535 for v in block) \
            or block[1] % 2 or block[2] % 2:
        raise ol.CompileError("FP16 block must be three positive 16-bit sizes, N and K even")
    _apart(at, ot)
    _apart(bt, ot)
    # store() records static host-output ranges. This kernel writes every element
    # of out, so register those ranges once, after tracing all the dynamic stores.
    # Keep a host-free view inside the loops; no claim is made about row padding.
    host = out.host
    if host is not None and (not Affine.of(out.base).is_static or
                             not Affine.of(out.offset).is_static):
        raise ol.CompileError("FP16 matmul host outputs must have static base and offset")
    out = replace(out, host=None)

    def step(y, row, rows, col, cols, start, depth):
        av = a[row:row + rows, start:start + depth]
        x = ol.load_fp16(av) if isinstance(a, HalfTensor) else ol.pack_fp16(ol.load(av))
        w = ol.load_fp16(b[start:start + depth, col:col + cols])
        ol.matmul_fp16(x, w, acc=y)

    def tile(row, rows, col, cols):
        y = ol.zeros((rows, cols))
        for start, depth in _tiles(k, block[2]):
            step(y, row, rows, col, cols, start, depth)
        ol.store(out[row:row + rows, col:col + cols], y)

    for row, rows in _tiles(m, block[0]):
        for col, cols in _tiles(n, block[1]):
            tile(row, rows, col, cols)
    if host is not None:
        off = Affine.of(out.offset).static()
        current().stores.extend((host, off + r * out.strides[0], n) for r in range(m))


def swiglu(a, w_gate: HalfTensor, w_up: HalfTensor, w_down: HalfTensor,
           out: Tensor, scratch: tuple, block=(4, 32, 32)) -> None:
    """W16A16 gate/up/down projections with FP32 SiLU and elementwise product.

    a[M,K], gate/up weights [K,F], down weights [F,N], out[M,N]. scratch is
    two caller-owned FP32 DRAM matrices [M,F]; the first ends with SiLU(gate)*up,
    the second with up. Scratch and out must be disjoint from each other and all
    inputs. The down projection rounds its activation to FP16 on the device.
    Norm, residual and inter-slice communication belong to the caller.
    """
    at = _matrix(a, isinstance(a, HalfTensor))
    wg, wu, wd = (_matrix(w, True) for w in (w_gate, w_up, w_down))
    ot = _matrix(out)
    if len(scratch) != 2:
        raise ol.CompileError("FP16 swiglu requires two scratch matrices")
    g, u = (_matrix(t) for t in scratch)
    m, k = at.shape
    f = wg.shape[1]
    if wg.shape != (k, f) or wu.shape != (k, f) or wd.shape[0] != f or \
            ot.shape != (m, wd.shape[1]) or g.shape != (m, f) or u.shape != (m, f):
        raise ol.CompileError("FP16 swiglu requires compatible projections and [M,F] scratch")
    for i, dst in enumerate((g, u, ot)):
        for src in (at, wg, wu, wd, *((g, u, ot)[:i])):
            _apart(src, dst)
    matmul(a, w_gate, g, block)
    matmul(a, w_up, u, block)

    def activate(row, rows, col, cols):
        gs = g[row:row + rows, col:col + cols]
        us = u[row:row + rows, col:col + cols]
        # g's whole output range was registered by its projection above.
        ol.store(replace(gs, host=None), silu(ol.load(gs)) * ol.load(us))

    for row, rows in _tiles(m, block[0]):
        for col, cols in _tiles(f, block[1]):
            activate(row, rows, col, cols)
    matmul(g, w_down, out, block)
