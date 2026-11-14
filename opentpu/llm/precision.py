"""Explicit matrix precision rules, separate from the legacy MX weight-format choices.

The rule grammar and precedence match llm.formats. No model gets a recommended
mixed policy merely by supporting these modes; quality must be measured first.
"""
import os

from . import formats as FM

MODES = ("w16a16", "w8a8", "w4a16")
KINDS = ("attn", "delta", "mlp", "gateup", "down", "head", "wq", "wk", "wv", "wo")


def weight(array, mode, group=128):
    """The placed weight type makes matrix dispatch explicit in a traced program."""
    from ..runtime import HalfWeight, Int4Weight, Weight
    if mode == "w16a16":
        return HalfWeight(array)
    if mode == "w4a16":
        return Int4Weight(array, group=group)
    if mode == "w8a8":
        return Weight(array, fmt="int8")
    raise ValueError(f"unknown matrix precision {mode!r}")


def reference(x, w, mode, D=128, group=128):
    """Projection oracle, independent of the compiler/runtime and never an Engine fallback."""
    import numpy as np
    from .. import fp16 as H, fp32 as F, int4 as Q4, quant as Q
    if mode == "w16a16":
        return H.matmul(H.pack(x), H.pack(w.T))
    if mode == "w4a16":
        return Q4.matmul(x, Q4.quantize(w, group))
    if mode != "w8a8":
        raise ValueError(f"unknown matrix precision {mode!r}")
    # Match QACT and the integer window, including dropped low-exponent groups.
    # The final FP32 scale application has the MXU's two separate roundings.
    qa, sa, da = F.quantize_mx(F.f32(x), D // Q.NSUB)
    da = da.reshape(len(x), -1, Q.NSUB)
    qw, sw, dw = F.quantize_mx(F.f32(w), D)
    sums = np.zeros((len(x), len(w)), np.int64)
    for k in range(w.shape[1] // D):
        for sub in range(Q.NSUB):
            lo = k * D + sub * (D // Q.NSUB)
            hi = lo + D // Q.NSUB
            raw = qa[:, lo:hi].astype(np.int64) @ qw[:, lo:hi].astype(np.int64).T
            shift = 16 - da[:, k, sub].astype(np.int64)[:, None] - dw[:, k].astype(np.int64)[None, :]
            sums += np.where(shift >= 0, raw << np.maximum(shift, 0), 0)
    return F.mul(F.mul(F.i2f(sums), sw[None, :]), F.mul(sa[:, None], np.float32(2**-16)))


class Policy:
    """Resolved matrix choices for dense Qwen3.5, with a canonical cache identity.

    Explicit rules override OTPU_PRECISIONS; an empty string disables environment
    rules. All layer ranges must fit the checkpoint. Wq/wk/wv/wo refine attn;
    gateup/down refine mlp. Head has no range and an explicit head mode wins.
    """

    def __init__(self, layers, mode="w16a16", rules=None, head=None):
        if mode not in MODES or head is not None and head not in MODES:
            raise ValueError(f"matrix precision must be one of {MODES}")
        if not isinstance(layers, int) or layers <= 0:
            raise ValueError("precision policy requires a positive layer count")
        self.mode, self.layers = mode, layers
        rules = os.environ.get("OTPU_PRECISIONS", "") if rules is None else rules
        self.rules = FM.rules(rules, KINDS, allowed=MODES)
        self.label = rules.replace(" ", "")
        if any(ranged and hi >= layers for _, lo, hi, ranged, fmt in self.rules):
            raise ValueError("precision rule's layer range exceeds the checkpoint")
        self.head = head or FM.plain(self.rules).get("head", mode)
        self.identity = (tuple(tuple(self(kind, i) for kind in KINDS if kind != "head")
                               for i in range(layers)), self.head)

    def __call__(self, kind, layer=0):
        if kind not in KINDS or not isinstance(layer, int) or not 0 <= layer < self.layers:
            raise ValueError("precision lookup requires a known kind and valid layer")
        if kind == "head":
            return self.head
        base = FM.pick(self.rules, FM.SUB[kind], layer, self.mode) if kind in FM.SUB else self.mode
        return FM.pick(self.rules, kind, layer, base)
