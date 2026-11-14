"""Explicit matrix precision rules, separate from the legacy MX weight-format choices.

The rule grammar and precedence match llm.formats. No model gets a recommended
mixed policy merely by supporting these modes; quality must be measured first.
"""
import os

from . import formats as FM

MODES = ("w16a16", "w8a8", "w4a16")
KINDS = ("attn", "delta", "mlp", "gateup", "down", "head", "wq", "wk", "wv", "wo")


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
