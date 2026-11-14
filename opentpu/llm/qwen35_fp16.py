"""Native FP16 dense Qwen3.5: tiled projections, FP32 vector/recurrent math,
persistent half KV caches and an Image/Engine for the existing device backends.
"""
from dataclasses import replace

import numpy as np

from .. import fp16 as H, fp32 as F, language as ol
from ..compiler import Compiled
from ..kernels.fp16 import _tiles
from ..kernels.mixed import matmul, swiglu
from . import precision as MP
from ..kernels.deltanet import gates, l2norm_rows
from ..kernels.fp16_attention import append, attend
from ..kernels.lib import rmsnorm, rope, sigmoid, silu
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
def _attention(a, spec, pos, rows, cap, block=(4, 32, 32)):
    d, nq, nk = spec.head_dim, spec.n_q, spec.n_kv
    # Scratch has no host-output tracking: its dynamic stores remain in DRAM.
    xn, qg, k, val, heads, scores = (replace(a[n], host=None) for n in
                                   ("xn", "qg", "key", "value", "heads", "scores"))
    for r in ol.range(rows):
        ol.store(xn[r:r + 1, :], ol.load(a["x"][r:r + 1, :]))
    _norm_rows(xn, a["gin"], rows, spec.hidden, eps=spec.eps)
    matmul(xn, a["wq"], qg, block)
    matmul(xn, a["wk"], k, block)
    matmul(xn, a["wv"], val, block)
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
            attend(q, kt[:, :t], vc[:t, :], out, scores[:, :t], a["mask"][:, :t], block=block)
            for r in ol.range(rows):
                gate = ol.load(qg[r:r + 1, qh * 2 * d + d:qh * 2 * d + 2 * d])
                ol.store(out[r:r + 1, :], ol.load(out[r:r + 1, :]) * sigmoid(gate))
    matmul(heads, a["wo"], a["out"], block)
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

    def __init__(self, spec, W, cfg, layer: int, cap: int, rows: int = 1, *, _defer=False):
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
        if _defer:
            self.arguments = args
            return
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


def attention_reference(spec, W, x, layer, pos, keys, values, projector=None):
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
        w = W[p + "self_attn." + name + "_proj.weight"]
        return projector(x, w, "w" + name) if projector else H.matmul(H.pack(x), H.pack(w.T))

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


@ol.jit
def _linear(a, spec, rows, block=(4, 32, 32)):
    """FP16 projections with FP32 convolution, gates and persistent DSTEP state."""
    a = dict(a)
    for name in ("ring", "state", "decay", "beta"):
        a[name] = replace(a[name], host=None)
    nh, nk, dk, dv, K = spec.lin_heads, spec.lin_nk, spec.lin_dk, spec.lin_dv, spec.conv_k
    C = 2 * nk * dk + nh * dv
    xn, qkv, z, ga, gb, y, heads = (replace(a[n], host=None) for n in
                                   ("xn", "qkv", "z", "a", "b", "y", "heads"))
    for r in ol.range(rows):
        ol.store(xn[r:r + 1, :], rmsnorm(ol.load(a["x"][r:r + 1, :]),
                                        ol.load(a["gin"]), spec.eps))
    for w, dst in (("wqkv", qkv), ("wz", z), ("wa", ga), ("wb", gb)):
        matmul(xn, a[w], dst, block)
    for r in ol.range(rows):
        # Window rows are oldest to newest; evaluate current, newest ... oldest,
        # matching the existing decode convolution's rounded MUL/ADD order.
        for c, n in _tiles(C, 128):
            cur = ol.load(qkv[r, c:c + n])
            u = cur * ol.load(a["taps"][K - 1, c:c + n])
            for j in range(K - 2, -1, -1):
                u = u + ol.load(a["ring"][j, c:c + n]) * ol.load(a["taps"][j, c:c + n])
            ol.store(y[r, c:c + n], silu(u))
            for j in range(K - 2):
                ol.store(a["ring"][j, c:c + n], ol.load(a["ring"][j + 1, c:c + n]))
            if K > 1:
                ol.store(a["ring"][K - 2, c:c + n], cur)
        decay, beta = gates(ol.load(ga[r, :]), ol.load(gb[r, :]),
                            ol.load(a["alog"]), ol.load(a["dtb"]))
        # Persist the gate vectors before head loops (TMEM slices are static).
        ol.store(a["decay"], decay)
        ol.store(a["beta"], beta)
        for kh in ol.range(nk):
            qk = ol.empty((2 * dk,))
            qk[:dk].set(l2norm_rows(ol.load(y[r:r + 1, kh * dk:(kh + 1) * dk]),
                                    dk ** -0.5)[0, :])
            qk[dk:].set(l2norm_rows(ol.load(y[r:r + 1, (nk + kh) * dk:(nk + kh + 1) * dk]))
                        [0, :])
            for g in ol.range(nh // nk):
                h = kh * (nh // nk) + g
                v = ol.load(y[r, 2 * nk * dk + h * dv:2 * nk * dk + (h + 1) * dv])
                o = ol.empty((dv,))
                ol.deltanet_step(a["state"][h * dv:(h + 1) * dv, :], qk, v,
                                 ol.load(a["decay"][h:h + 1]), ol.load(a["beta"][h:h + 1]), o)
                post = rmsnorm(o.reshape(1, dv), ol.load(a["gn"]), spec.eps)
                post = post * silu(ol.load(z[r:r + 1, h * dv:(h + 1) * dv]))
                ol.store(heads[r:r + 1, h * dv:(h + 1) * dv], post)
    matmul(heads, a["wo"], a["out"], block)
    out = replace(a["out"], host=None)
    for r in ol.range(rows):
        ol.store(out[r:r + 1, :], ol.load(out[r:r + 1, :]) + ol.load(a["x"][r:r + 1, :]))


class LinearImage:
    """One native FP16 DeltaNet sublayer; carry ring and FP32 state across calls."""

    def __init__(self, spec, W, cfg, layer, rows=1, *, _defer=False):
        if not cfg.FP16 or cfg.S != 1 or not cfg.DSTEP:
            raise ValueError("FP16 linear image requires Config.FP16, S=1 and DSTEP")
        if not 0 <= layer < len(spec.kinds) or spec.kinds[layer] != "linear":
            raise ValueError("FP16 linear image requires a linear layer")
        nh, nk, dk, dv, K = spec.lin_heads, spec.lin_nk, spec.lin_dk, spec.lin_dv, spec.conv_k
        if min(nh, nk, dk, dv, K, rows) <= 0 or nh % nk or dk > 256 or dk % 64 or dv > 256:
            raise ValueError("FP16 linear image requires valid grouped DeltaNet dimensions")
        self.spec, self.cfg, self.layer, self.rows = spec, cfg, layer, rows
        p = f"model.layers.{layer}."
        C = 2 * nk * dk + nh * dv
        args = {}
        for name, proj, shape in (("wqkv", "in_proj_qkv", (C, spec.hidden)),
                                  ("wz", "in_proj_z", (nh * dv, spec.hidden)),
                                  ("wa", "in_proj_a", (nh, spec.hidden)),
                                  ("wb", "in_proj_b", (nh, spec.hidden)),
                                  ("wo", "out_proj", (spec.hidden, nh * dv))):
            w = W[p + "linear_attn." + proj + ".weight"]
            if w.shape != shape:
                raise ValueError(f"{proj}: expected {shape}")
            args[name] = HalfWeight(w)
        args["gin"] = Input(F.add(W[p + "input_layernorm.weight"], np.float32(1)))
        for name, key in (("gn", "norm.weight"), ("alog", "A_log"), ("dtb", "dt_bias")):
            args[name] = Input(W[p + "linear_attn." + key])
        taps = W[p + "linear_attn.conv1d.weight"].reshape(C, K).T.copy()
        args["taps"] = Input(taps)
        shapes = dict(x=(rows, spec.hidden), xn=(rows, spec.hidden), qkv=(rows, C),
                      z=(rows, nh * dv), a=(rows, nh), b=(rows, nh), y=(rows, C),
                      heads=(rows, nh * dv), ring=(max(1, K - 1), C), state=(nh * dv, dk),
                      decay=(nh,), beta=(nh,))
        args.update({name: Input(np.zeros(shape, np.float32)) for name, shape in shapes.items()})
        args["out"] = Output((rows, spec.hidden))
        if _defer:
            self.arguments = args
            return
        dram, self.layout, bound = place(cfg, args)
        self.images, self.bound = dram.images, bound[0]

    def compile(self):
        b = _linear.trace(self.cfg, 0, dict(a=self.bound, spec=self.spec, rows=self.rows))
        return Compiled(self.cfg, [b.finish()], [b.stores], self.layout)

    def inputs(self, images, x):
        x = F.f32(x)
        if x.shape != (self.rows, self.spec.hidden) or not np.isfinite(x).all():
            raise ValueError("linear input must be finite [rows, hidden]")
        addr = self.layout.params["x"][1]
        images[0][addr:addr + x.nbytes] = np.ascontiguousarray(x).view(np.uint8).reshape(-1)


def linear_reference(spec, W, x, layer, ring, state, projector=None):
    """Ordered FP16 projections / FP32 DeltaNet; state is [heads,dv,dk]."""
    p = f"model.layers.{layer}."
    nh, nk, dk, dv, K = spec.lin_heads, spec.lin_nk, spec.lin_dk, spec.lin_dv, spec.conv_k
    x, ring, state = F.f32(x), ring.copy(), state.copy()

    def norm(x, gain):
        scale = F.rsqrt(F.add(F.mul(F.rdot(x, x), np.float32(1 / x.shape[-1])),
                             np.float32(spec.eps)))
        return F.mul(F.mul(x, scale[:, None]), gain)

    def project(x, name):
        w = W[p + "linear_attn." + name + ".weight"]
        return projector(x, w, "delta") if projector else H.matmul(H.pack(x), H.pack(w.T))

    xn = norm(x, F.add(W[p + "input_layernorm.weight"], np.float32(1)))
    qkv, z, a, b = (project(xn, n) for n in ("in_proj_qkv", "in_proj_z", "in_proj_a", "in_proj_b"))
    alog, dtb = (W[p + "linear_attn." + n] for n in ("A_log", "dt_bias"))
    taps = W[p + "linear_attn.conv1d.weight"].reshape(-1, K).T
    heads = np.empty((len(x), nh * dv), np.float32)
    for r in range(len(x)):
        u = F.mul(qkv[r], taps[-1])
        for j in range(K - 2, -1, -1):
            u = F.add(u, F.mul(ring[j], taps[j]))
        if K > 1:
            ring[:-1] = ring[1:]
            ring[-1] = qkv[r]
        u = F.silu(u, np.float32(-ol.LOG2E))
        q, k = u[:nk * dk].reshape(nk, dk), u[nk * dk:2 * nk * dk].reshape(nk, dk)
        q = F.mul(q, F.mul(F.rsqrt(F.add(F.rdot(q, q), np.float32(1e-6))),
                          np.float32(dk ** -0.5))[:, None])
        k = F.mul(k, F.rsqrt(F.add(F.rdot(k, k), np.float32(1e-6)))[:, None])
        v = u[2 * nk * dk:].reshape(nh, dv)
        ab = F.add(a[r], dtb)
        sp = F.add(F.fmax(ab, np.float32(0)), F.mul(np.float32(ol.LN2),
                   F.log2(F.add(np.float32(1), F.exp2(F.mul(-np.abs(ab), np.float32(ol.LOG2E)))))))
        decay = F.exp2(F.mul(F.mul(sp, F.exp2(F.mul(alog, np.float32(ol.LOG2E)))),
                             np.float32(-ol.LOG2E)))
        beta = F.sigmoid(b[r], np.float32(-ol.LOG2E))
        for h in range(nh):
            kh = h // (nh // nk)
            kv = F.rdot(state[h], k[kh])
            delta = F.mul(F.sub(v[h], F.mul(kv, decay[h])), beta[h])
            state[h] = F.add(F.mul(state[h], decay[h]), F.mul(delta[:, None], k[kh]))
            o = F.rdot(state[h], q[kh]).reshape(1, dv)
            post = norm(o, W[p + "linear_attn.norm.weight"])
            heads[r, h * dv:(h + 1) * dv] = F.mul(post, F.silu(z[r, h * dv:(h + 1) * dv],
                                                                            np.float32(-ol.LOG2E)))
    return F.add(x, project(heads, "out_proj")), ring, state


@ol.jit
def _mlp(a, spec, block):
    xn = replace(a["xn"], host=None)
    ol.store(xn, rmsnorm(ol.load(a["x"]), ol.load(a["gain"]), spec.eps))
    swiglu(xn, a["wg"], a["wu"], a["wd"], a["out"], scratch=(a["gate"], a["up"]), block=block)
    out = replace(a["out"], host=None)
    ol.store(out, ol.load(out) + ol.load(a["x"]))


@ol.jit
def _head(a, spec, block):
    xn = replace(a["xn"], host=None)
    ol.store(xn, rmsnorm(ol.load(a["x"]), ol.load(a["gain"]), spec.eps))
    matmul(xn, a["weight"], a["logits"], block)


def _shape_weights(spec):
    """Zero-stride placeholders: layout construction never loads a checkpoint."""
    h, f, nh, nk, dk, dv = spec.hidden, spec.ffn, spec.lin_heads, spec.lin_nk, spec.lin_dk, spec.lin_dv
    shapes = {"model.embed_tokens.weight": (spec.vocab, h), "model.norm.weight": (h,),
              "lm_head.weight": (spec.vocab, h)}
    for i, kind in enumerate(spec.kinds):
        p = f"model.layers.{i}."
        local = {"input_layernorm.weight": (h,), "post_attention_layernorm.weight": (h,),
                 "mlp.gate_proj.weight": (f, h), "mlp.up_proj.weight": (f, h),
                 "mlp.down_proj.weight": (h, f)}
        if kind == "attn":
            d, nq, nkv = spec.head_dim, spec.n_q, spec.n_kv
            local.update({"self_attn.q_proj.weight": (2 * nq * d, h),
                          "self_attn.k_proj.weight": (nkv * d, h),
                          "self_attn.v_proj.weight": (nkv * d, h),
                          "self_attn.o_proj.weight": (h, nq * d),
                          "self_attn.q_norm.weight": (d,), "self_attn.k_norm.weight": (d,)})
        elif kind == "linear":
            C = 2 * nk * dk + nh * dv
            local.update({"linear_attn.in_proj_qkv.weight": (C, h),
                          "linear_attn.in_proj_z.weight": (nh * dv, h),
                          "linear_attn.in_proj_a.weight": (nh, h),
                          "linear_attn.in_proj_b.weight": (nh, h),
                          "linear_attn.out_proj.weight": (h, nh * dv),
                          "linear_attn.conv1d.weight": (C, 1, spec.conv_k),
                          "linear_attn.norm.weight": (dv,), "linear_attn.A_log": (nh,),
                          "linear_attn.dt_bias": (nh,)})
        else:
            raise ValueError(f"unsupported FP16 layer kind: {kind}")
        shapes.update({p + n: shape for n, shape in local.items()})
    return {n: np.broadcast_to(np.float32(0), shape) for n, shape in shapes.items()}


class Image:
    """Single-slice dense Qwen3.5 W16A16, one shared persistent DRAM image.

    Each sublayer is a separate device program so the full model does not need
    to fit IMEM. Only embeddings and position tables are supplied by the host;
    every projection, normalization, nonlinearity and recurrent update executes
    on openTPU. Prefill uses consecutive decode steps with identical arithmetic.
    """

    wformat = head_format = "fp16"
    formats = ""
    batch = 1

    def __init__(self, spec, cfg, cap, W=None, *, policy=None, group=128):
        if not cfg.FP16 or cfg.S != 1 or not cfg.DSTEP:
            raise ValueError("FP16 model requires Config.FP16, S=1 and DSTEP")
        if getattr(spec, "moe", None) is not None or getattr(spec, "mtp", False):
            raise ValueError("FP16 model currently supports dense Qwen3.5 without MTP")
        if cap <= 0 or cap % 2:
            raise ValueError("FP16 model requires a positive even context capacity")
        self.spec, self.cfg, self.cap = spec, cfg, cap
        self.policy, self.group = policy, group
        if policy is not None:
            if policy.layers != len(spec.kinds):
                raise ValueError("precision policy layer count differs from checkpoint")
            self.wformat, self.head_format = policy.mode, policy.head
            self.formats = policy.label
        # A 128-column K-major tile uses 8K TMEM words and full memory beats.
        # Smaller test targets retain the 32-column tile; K order is identical.
        self.block = (1, 128, 128) if cfg.TMEM_WORDS >= 16384 else (1, 32, 32)
        initialize = W is not None
        if W is None:
            W = _shape_weights(spec)
        else:
            expected = _shape_weights(spec)
            for name, value in expected.items():
                if name == "lm_head.weight" and spec.tied:
                    continue
                if W[name].shape != value.shape:
                    raise ValueError(f"{name}: expected {value.shape}, got {W[name].shape}")
        self.embedding = F.f32(W["model.embed_tokens.weight"])
        args = dict(x=Input(np.zeros((1, spec.hidden), np.float32)),
                    mixed=Input(np.zeros((1, spec.hidden), np.float32)))
        self.layer_names = []
        for i, kind in enumerate(spec.kinds):
            sub = AttentionImage(spec, W, cfg, i, cap, _defer=True) if kind == "attn" else \
                LinearImage(spec, W, cfg, i, _defer=True)
            layer_args = sub.arguments
            del layer_args["x"], layer_args["out"]
            if policy is not None:
                for name, arg in layer_args.items():
                    if isinstance(arg, HalfWeight):
                        mode = policy(name if kind == "attn" else "delta", i)
                        layer_args[name] = MP.weight(arg.array, mode, group)
            names = {n: f"layer{i}.{n}" for n in layer_args}
            args.update({names[n]: a for n, a in layer_args.items()})
            self.layer_names.append(names)
            p = f"model.layers.{i}."
            for name, proj in (("wg", "gate_proj"), ("wu", "up_proj"), ("wd", "down_proj")):
                w = W[p + "mlp." + proj + ".weight"]
                args[f"mlp{i}.{name}"] = HalfWeight(w) if policy is None else \
                    MP.weight(w, policy("down" if name == "wd" else "gateup", i), group)
            args[f"mlp{i}.gain"] = Input(F.add(W[p + "post_attention_layernorm.weight"], np.float32(1)))
        args.update(mlp_xn=Input(np.zeros((1, spec.hidden), np.float32)),
                    gate=Input(np.zeros((1, spec.ffn), np.float32)),
                    up=Input(np.zeros((1, spec.ffn), np.float32)),
                    final_gain=Input(F.add(W["model.norm.weight"], np.float32(1))),
                    head=HalfWeight(self.embedding if spec.tied else W["lm_head.weight"]),
                    logits=Output((1, spec.vocab)))
        if policy is not None:
            args["head"] = MP.weight(args["head"].array, policy.head, group)
        dram, self.layout, bound = place(cfg, args, initialize=initialize)
        self.nbytes, self.bound = dram.next, bound[0]
        self.images = [m[:self.nbytes] for m in dram.images] if initialize else None
        self.mutable = []
        for names in self.layer_names:
            for n in ("ring", "state", "kt", "vc"):
                if n in names:
                    arg, addr = self.layout.params[names[n]]
                    self.mutable.append((addr, arg.array.nbytes))
        self.kv_bytes = sum(nbytes for _, nbytes in self.mutable)
        self.layers = [{n: bound[0][name] for n, name in names.items()} for names in self.layer_names]
        for layer in self.layers:
            layer.update(x=self.bound["x"], out=self.bound["mixed"])

    def build(self, W):
        filled = Image(self.spec, self.cfg, self.cap, W, policy=self.policy, group=self.group)
        if filled.nbytes != self.nbytes:
            raise ValueError("checkpoint shapes changed the FP16 model layout")
        self.embedding = filled.embedding
        return filled.images

    def stages(self, pos):
        if not 0 <= pos < self.cap:
            raise ValueError("FP16 model position exceeds context capacity")
        for i, kind in enumerate(self.spec.kinds):
            kernel = _attention if kind == "attn" else _linear
            kw = dict(a=self.layers[i], spec=self.spec, rows=1, block=self.block)
            if kind == "attn":
                kw.update(pos=pos, cap=self.cap)
            yield ("mixer", i, pos if kind == "attn" else None), kernel, kw
            a = {n: self.bound[f"mlp{i}.{n}"] for n in ("wg", "wu", "wd", "gain")}
            a.update(x=self.bound["mixed"], out=self.bound["x"], xn=self.bound["mlp_xn"],
                     gate=replace(self.bound["gate"], host=None), up=replace(self.bound["up"], host=None))
            yield ("mlp", i), _mlp, dict(a=a, spec=self.spec, block=self.block)
        a = dict(x=self.bound["x"], xn=self.bound["mlp_xn"], gain=self.bound["final_gain"],
                 weight=self.bound["head"], logits=self.bound["logits"])
        yield ("head",), _head, dict(a=a, spec=self.spec, block=self.block)

    def inputs(self, token, pos):
        if not 0 <= token < self.spec.vocab or not 0 <= pos < self.cap:
            raise ValueError("token or position outside FP16 model bounds")
        yield self.layout.params["x"][1], self.embedding[token:token + 1]
        c, s = rope_tables(self.spec, pos)
        mask = np.where(np.arange(self.cap) <= pos, np.inf, -np.inf).astype(np.float32)
        for names in self.layer_names:
            if "cos" in names:
                for name, data in (("cos", c), ("sin", s), ("mask", mask)):
                    yield self.layout.params[names[name]][1], data


class Engine:
    """Native FP16 text engine using the existing ISA/board backend contract."""

    can_generate = False
    staged = True
    snapshot_bytes = 0
    batch = rows = 1

    def __init__(self, spec, W, cap=2048, cfg=None, backend="isa", prog_cache=True, *,
                 precision=None, precisions=None, head_precision=None, group=128):
        from .. import target, imgcache as IC
        from .qwen3 import IsaBackend
        self.spec, self.cap = spec, cap
        if precision is None and (precisions is not None or head_precision is not None):
            raise ValueError("precision rules require an explicit base precision")
        policy = MP.Policy(len(spec.kinds), precision, precisions, head_precision) if precision else None
        self.cfg = cfg or target.config("v80:mixed" if policy else "v80:fp16")
        self.image = Image(spec, self.cfg, cap, policy=policy, group=group)
        self.prog_cache = prog_cache
        self.layout = ("qwen35-native-fp16-v1", spec, self.cfg, cap) if policy is None else \
            ("qwen35-mixed-v1", spec, self.cfg, cap, policy.identity, group)
        self._programs, self.stats, self.poss = {}, [], [0]
        images = IC.build(self.image, W, self.layout)
        self.backend = IsaBackend(self.cfg, images) if backend == "isa" else backend(self.cfg, images)
        if hasattr(self.backend, "attach"):
            self.backend.attach(self)
        self.image.images = None

    @property
    def pos(self):
        return self.poss[0]

    def reset(self):
        for addr, nbytes in self.image.mutable:
            self.backend.write(0, addr, np.zeros(nbytes, np.uint8))
        self.poss[0] = 0
        self.stats.clear()

    def snapshot(self):
        # No hidden host math or state reconstruction: a rollback replays input.
        return None

    def rollback(self, pos):
        if not isinstance(pos, int) or not 0 <= pos <= self.pos:
            raise ValueError("rollback position must be within the current context")
        if pos == self.pos:
            return pos
        self.reset()
        return 0

    def step(self, token, on_start=None, sink=None):
        from .. import progcache as PC
        for addr, data in self.image.inputs(int(token), self.pos):
            self.backend.write(0, addr, data)
        if on_start:
            on_start()
        total = dict(cycles=0)
        for key, kernel, kw in self.image.stages(self.pos):
            if key not in self._programs:
                def compile():
                    b = kernel.trace(self.cfg, 0, kw)
                    p = b.finish()
                    if len(p) * 8 > self.cfg.IMEM_WORDS:
                        raise ValueError(f"FP16 stage {key} exceeds IMEM")
                    return [p], None
                self._programs[key] = PC.get(self.layout, key, compile, disk=self.prog_cache)[0] \
                    if self.prog_cache else compile()[0]
            stats = self.backend.run(self._programs[key])
            total["cycles"] += (stats or {}).get("cycles", 0)
        self.stats.append(total)
        self.poss[0] += 1
        addr = self.image.layout.params["logits"][1]
        logits = self.backend.read(0, addr, 4 * self.spec.vocab).view("<f4").copy()
        if not np.isfinite(logits).all():
            raise RuntimeError("native FP16 model produced nonfinite logits")
        if hasattr(self.backend, "record_token"):
            self.backend.record_token(total["cycles"])
        if sink is not None:
            sink.begin(self.spec.vocab)
            sink.feed(0, logits)
        return logits

    def prefill_chunks(self, tokens, seq=0, chunk=None, **kw):
        if seq != 0 or kw:
            raise ValueError("FP16 prefill supports one text sequence")
        for token in tokens:
            yield [token], self.step(token)

    def prefill(self, tokens, **kw):
        result = None
        for _, result in self.prefill_chunks(tokens, **kw):
            pass
        return result

    def close(self):
        if hasattr(self.backend, "close"):
            self.backend.close()


class Reference:
    """Ordered numerical oracle for qualification; never used by Engine."""

    def __init__(self, spec, W, cap, *, policy=None, D=128, group=128):
        self.spec, self.W, self.cap, self.pos = spec, W, cap, 0
        self.policy, self.D, self.group = policy, D, group
        self.states = []
        for kind in spec.kinds:
            if kind == "attn":
                shape = (spec.n_kv, cap, spec.head_dim)
                self.states.append((np.zeros(shape, np.uint16), np.zeros(shape, np.uint16)))
            else:
                C = 2 * spec.lin_nk * spec.lin_dk + spec.lin_heads * spec.lin_dv
                self.states.append((np.zeros((max(1, spec.conv_k - 1), C), np.float32),
                                    np.zeros((spec.lin_heads, spec.lin_dv, spec.lin_dk), np.float32)))

    def step(self, token, on_layer=None):
        spec, W = self.spec, self.W
        x = W["model.embed_tokens.weight"][token:token + 1]

        def norm(x, name):
            scale = F.rsqrt(F.add(F.mul(F.rdot(x, x), np.float32(1 / spec.hidden)), np.float32(spec.eps)))
            return F.mul(F.mul(x, scale[:, None]), F.add(W[name], np.float32(1)))

        def mm(x, w, kind, layer=0):
            mode = self.policy(kind, layer) if self.policy else "w16a16"
            return MP.reference(x, w, mode, self.D, self.group)

        for i, kind in enumerate(spec.kinds):
            if kind == "attn":
                x, k, v = attention_reference(spec, W, x, i, self.pos, *self.states[i],
                                               projector=lambda x, w, kind: mm(x, w, kind, i))
            else:
                x, k, v = linear_reference(spec, W, x, i, *self.states[i],
                                            projector=lambda x, w, kind: mm(x, w, kind, i))
            self.states[i] = k, v
            p = f"model.layers.{i}."
            xn = norm(x, p + "post_attention_layernorm.weight")
            gate, up = (mm(xn, W[p + "mlp." + n + "_proj.weight"], "gateup", i) for n in ("gate", "up"))
            x = F.add(x, mm(F.mul(F.silu(gate, np.float32(-ol.LOG2E)), up), W[p + "mlp.down_proj.weight"], "down", i))
            if on_layer:
                on_layer(i, x)
        x = norm(x, "model.norm.weight")
        weight = W["model.embed_tokens.weight"] if spec.tied else W["lm_head.weight"]
        logits = np.empty(spec.vocab, np.float32)
        for lo in range(0, spec.vocab, 4096):
            logits[lo:lo + 4096] = mm(x, weight[lo:lo + 4096], "head")[0]
        self.pos += 1
        return logits
