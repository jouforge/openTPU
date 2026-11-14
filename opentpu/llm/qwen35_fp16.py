"""Experimental native FP16 Qwen3.5 attention image and ordered reference.

This is one complete gated-attention sublayer, including input RMSNorm, Q/K
norms, partial RoPE, GQA, persistent half KV caches, output projection and
residual. It is not yet the full model Image/Engine or a CLI weight format.
"""
from dataclasses import replace

import numpy as np

from .. import fp16 as H, fp32 as F, language as ol
from ..compiler import Compiled
from ..kernels.fp16 import matmul
from ..kernels.fp16_attention import append, attend
from ..kernels.lib import rmsnorm, rope, sigmoid
from ..runtime import HalfInput, HalfWeight, Input, Output, place
from .qwen3 import rope_tables


def _norm_rows(x, gain, rows, width, heads=1, stride=None, rd=0, cs=None, eps=1e-6):
    x = replace(x, host=None)
    stride = width if stride is None else stride

    def one(r, h):
        view = x[r:r + 1, h * stride:h * stride + width]
        y = rmsnorm(ol.load(view), ol.load(gain), eps)
        if rd:
            c, s = (ol.load(t[r, :]) for t in cs)
            y[:, :rd].set(rope(y[:, :rd], c, s))
        ol.store(view, y)

    for r in ol.range(rows):
        for h in ol.range(heads):
            one(r, h)


@ol.jit
def _attention(a, spec, pos, rows, cap):
    d, nq, nk = spec.head_dim, spec.n_q, spec.n_kv
    # Scratch has no host-output tracking: its dynamic stores remain in DRAM.
    xn, qg, k, val, heads, scores = (replace(a[n], host=None) for n in
                                   ("xn", "qg", "key", "value", "heads", "scores"))
    for r in ol.range(rows):
        ol.store(xn[r:r + 1, :], ol.load(a["x"][r:r + 1, :]))
    _norm_rows(xn, a["gin"], rows, spec.hidden, eps=spec.eps)
    matmul(xn, a["wq"], qg)
    matmul(xn, a["wk"], k)
    matmul(xn, a["wv"], val)
    cs = (a["cos"], a["sin"])
    _norm_rows(qg, a["gq"], rows, d, nq, 2 * d, spec.rope_dim, cs, spec.eps)
    _norm_rows(k, a["gk"], rows, d, nk, d, spec.rope_dim, cs, spec.eps)
    t = pos + rows
    for h in range(nk):
        kt = a["kt"][h * d:(h + 1) * d, :]
        vc = a["vc"][h * cap:(h + 1) * cap, :]
        for r in range(rows):
            append(k[r:r + 1, h * d:(h + 1) * d], val[r:r + 1, h * d:(h + 1) * d],
                   kt, vc, pos + r)
        # One loop level for a KV head's query group; matmul may use three more.
        for g in ol.range(nq // nk):
            qh = h * (nq // nk) + g
            q = qg[:, qh * 2 * d:qh * 2 * d + d]
            out = heads[:, qh * d:qh * d + d]
            attend(q, kt[:, :t], vc[:t, :], out, scores[:, :t], a["mask"][:, :t])
            for r in ol.range(rows):
                gate = ol.load(qg[r:r + 1, qh * 2 * d + d:qh * 2 * d + 2 * d])
                ol.store(out[r:r + 1, :], ol.load(out[r:r + 1, :]) * sigmoid(gate))
    matmul(heads, a["wo"], a["out"])
    out = replace(a["out"], host=None)
    for r in ol.range(rows):
        ol.store(out[r:r + 1, :], ol.load(out[r:r + 1, :]) + ol.load(a["x"][r:r + 1, :]))


class AttentionImage:
    """One S=1 attention sublayer's DRAM image, with persistent KV and scratch.

    compile(pos) specializes the cache prefix at a static position. inputs()
    writes only x and position tables into an image; launch its program through
    the normal runtime/transport. Carry the returned DRAM image to the next run.
    rows is fixed per image; callers must fill every earlier cache position.
    There is no program/image disk cache or full-model dispatch for this class yet.
    """

    def __init__(self, spec, W, cfg, layer: int, cap: int, rows: int = 1):
        if not cfg.FP16 or cfg.S != 1:
            raise ValueError("FP16 attention image requires Config.FP16 and S=1")
        if not 0 <= layer < len(spec.kinds) or spec.kinds[layer] != "attn":
            raise ValueError("FP16 attention image requires an attention layer")
        d, nq, nk = spec.head_dim, spec.n_q, spec.n_kv
        if min(d, nq, nk, spec.hidden, cap, rows) <= 0 or d % 2 or cap % 2 or \
                rows > cap or nq % nk or not 0 < spec.rope_dim <= d or spec.rope_dim % 2:
            raise ValueError("FP16 attention image requires valid GQA dimensions, even cap/head/RoPE")
        self.spec, self.cfg, self.layer, self.cap, self.rows = spec, cfg, layer, cap, rows
        p = f"model.layers.{layer}."
        names = dict(wq="q_proj", wk="k_proj", wv="v_proj", wo="o_proj")
        args = {name: HalfWeight(W[p + "self_attn." + proj + ".weight"])
                for name, proj in names.items()}
        expected = dict(wq=(2 * nq * d, spec.hidden), wk=(nk * d, spec.hidden),
                        wv=(nk * d, spec.hidden), wo=(spec.hidden, nq * d))
        if any(args[n].array.shape != shape for n, shape in expected.items()):
            raise ValueError("FP16 attention image checkpoint projection shapes do not match Spec")
        for name, key, size in (("gin", "input_layernorm", spec.hidden),
                                 ("gq", "self_attn.q_norm", d), ("gk", "self_attn.k_norm", d)):
            gain = F.add(W[p + key + ".weight"], np.float32(1))
            if gain.shape != (size,):
                raise ValueError(f"{key}: expected a gain of {size} elements")
            args[name] = Input(gain)
        shapes = dict(x=(rows, spec.hidden), xn=(rows, spec.hidden), qg=(rows, 2 * nq * d),
                      key=(rows, nk * d), value=(rows, nk * d), heads=(rows, nq * d),
                      scores=(rows, cap), mask=(rows, cap), cos=(rows, spec.rope_dim // 2),
                      sin=(rows, spec.rope_dim // 2))
        args.update({name: Input(np.zeros(shape, np.float32)) for name, shape in shapes.items()})
        args.update(kt=HalfInput(np.zeros((nk * d, cap), np.float32)),
                    vc=HalfInput(np.zeros((nk * cap, d), np.float32)), out=Output((rows, spec.hidden)))
        dram, self.layout, bound = place(cfg, args)
        self.images, self.bound = dram.images, bound[0]

    def _position(self, pos):
        if not isinstance(pos, int) or not 0 <= pos <= self.cap - self.rows:
            raise ValueError("attention position plus rows must fit cache capacity")

    def compile(self, pos):
        self._position(pos)
        b = _attention.trace(self.cfg, 0, dict(a=self.bound, spec=self.spec, pos=pos,
                                             rows=self.rows, cap=self.cap))
        return Compiled(self.cfg, [b.finish()], [b.stores], self.layout)

    def inputs(self, images, x, pos):
        self._position(pos)
        x = F.f32(x)
        if x.shape != (self.rows, self.spec.hidden) or not np.isfinite(x).all():
            raise ValueError("attention input must be finite [rows, hidden]")
        cs = [rope_tables(self.spec, pos + r) for r in range(self.rows)]
        data = dict(x=x, cos=np.stack([c for c, _ in cs]), sin=np.stack([s for _, s in cs]),
                    mask=np.where(np.arange(self.cap)[None, :] <= pos + np.arange(self.rows)[:, None],
                                  np.inf, -np.inf).astype(np.float32))
        for name, values in data.items():
            addr = self.layout.params[name][1]
            raw = np.ascontiguousarray(values).view(np.uint8).reshape(-1)
            images[0][addr:addr + len(raw)] = raw


def attention_reference(spec, W, x, layer, pos, keys, values):
    """Ordered FP16/FP32 reference; caches [nkv,cap,d] are binary16 uint16.

    Returns (residual output, new keys, new values), leaving the input caches
    unchanged. This arithmetic reference is never used as a device fallback.
    """
    x = F.f32(x)
    rows, d, nq, nk = len(x), spec.head_dim, spec.n_q, spec.n_kv
    p = f"model.layers.{layer}."

    def norm(x, name):
        flat = x.reshape(-1, x.shape[-1])
        sums = F.rdot(flat, flat).reshape(x.shape[:-1])
        scale = F.rsqrt(F.add(F.mul(sums, np.float32(1 / x.shape[-1])), np.float32(spec.eps)))
        return F.mul(F.mul(x, scale[..., None]), F.add(W[p + name + ".weight"], np.float32(1)))

    def project(x, name):
        return H.matmul(H.pack(x), H.pack(W[p + "self_attn." + name + "_proj.weight"].T))

    xn = norm(x, "input_layernorm")
    qg = project(xn, "q").reshape(rows, nq, 2 * d)
    q = norm(qg[..., :d], "self_attn.q_norm")
    k = norm(project(xn, "k").reshape(rows, nk, d), "self_attn.k_norm")
    v = project(xn, "v").reshape(rows, nk, d)
    half, rd = spec.rope_dim // 2, spec.rope_dim
    for r in range(rows):
        c, s = rope_tables(spec, pos + r)
        for tensor in (q, k):
            lo, hi = tensor[r, :, :half].copy(), tensor[r, :, half:rd].copy()
            tensor[r, :, :half] = F.sub(F.mul(lo, c), F.mul(hi, s))
            tensor[r, :, half:rd] = F.add(F.mul(hi, c), F.mul(lo, s))
    keys, values = keys.copy(), values.copy()
    keys[:, pos:pos + rows] = H.pack(k.transpose(1, 0, 2))
    values[:, pos:pos + rows] = H.pack(v.transpose(1, 0, 2))
    t = pos + rows
    mask = np.where(np.arange(t)[None, :] <= pos + np.arange(rows)[:, None], np.inf, -np.inf)
    heads = np.empty((rows, nq, d), np.float32)
    for h in range(nq):
        kv = h // (nq // nk)
        out, _ = H.attention(H.pack(q[:, h]), keys[kv, :t].T, values[kv, :t], mask)
        heads[:, h] = F.mul(out, F.sigmoid(qg[:, h, d:], np.float32(-ol.LOG2E)))
    return F.add(x, project(heads.reshape(rows, nq * d), "o")), keys, values
