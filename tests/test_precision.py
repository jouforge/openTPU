"""Matrix policy precedence and cache identity, independent of MX weight formats."""
import pytest

from opentpu.llm import formats as FM
from opentpu.llm.precision import KINDS, Policy


def test_policy_precedence_and_identity(monkeypatch):
    monkeypatch.setenv("OTPU_PRECISIONS", "mlp=w4a16,gateup@2-3=w8a8,attn=w8a8,wq@1=w16a16,head=w8a8")
    p = Policy(4)
    assert p("down", 2) == "w4a16" and p("gateup", 2) == "w8a8"
    assert p("wq", 1) == "w16a16" and p("wk", 1) == "w8a8"
    assert p("delta", 0) == "w16a16" and p("head") == "w8a8"
    assert Policy(4, rules="").identity == Policy(4, rules="attn=w16a16").identity
    assert p.identity != Policy(4, head="w4a16").identity
    assert Policy(4, rules="mlp=w8a8,mlp=w4a16")("down", 0) == "w8a8"
    assert Policy(4, rules="mlp=w8a8,mlp@1=w4a16")("down", 1) == "w4a16"


@pytest.mark.parametrize("rules", ["head@0=w8a8", "mlp@4=w8a8", "mlp@3-1=w8a8",
    "mlp=int4", "mlp=fp16", "unknown=w8a8", "mlp@-1=w8a8", "mlp@0-=w8a8"])
def test_bad_rules(rules):
    with pytest.raises(ValueError):
        Policy(4, rules=rules)


def test_no_implicit_legacy_format_extension(monkeypatch):
    monkeypatch.setenv("OTPU_FORMATS", "mlp=fp4")
    p = Policy(2, rules="")
    assert all(p(k) == "w16a16" for k in KINDS)
    for mode in ("w16a16", "w8a8", "w4a16"):
        with pytest.raises(ValueError):
            FM.rules("mlp=" + mode, ("mlp",))
    with pytest.raises(ValueError):
        Policy(2, mode="int8")
    with pytest.raises(ValueError):
        p("mlp", 2)
