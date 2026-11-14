"""Complete staged model with explicit projection modes and persistent state."""
from dataclasses import replace

import numpy as np
import pytest

from opentpu import fp32 as F
from opentpu.llm.precision import Policy
from opentpu.llm.qwen35_fp16 import Engine, Image, Reference, _shape_weights
from test_qwen35_fp16 import model_fixture


def fixture():
    spec, _, cfg = model_fixture()
    spec = replace(spec, hidden=32)
    cfg = replace(cfg, W4A16=True)
    rng = np.random.default_rng(71)
    weights = {name: (rng.normal(size=value.shape) * 0.1).astype(np.float32)
               for name, value in _shape_weights(spec).items()}
    return spec, weights, cfg


@pytest.mark.parametrize("mode,rules", [("w16a16", ""), ("w8a8", ""), ("w4a16", ""),
    ("w16a16", "delta=w8a8,mlp=w4a16,down=w8a8,wq=w4a16,head=w8a8")])
def test_model_and_reset(mode, rules):
    spec, weights, cfg = fixture()
    policy = Policy(len(spec.kinds), mode, rules)
    engine = Engine(spec, weights, cap=8, cfg=cfg, prog_cache=False,
                    precision=mode, precisions=rules, group=8)
    ref = Reference(spec, weights, 8, policy=policy, D=cfg.D, group=8)
    expected = []
    for token in (3, 9):
        expected.append(ref.step(token))
        np.testing.assert_array_equal(F.bits(engine.step(token)), F.bits(expected[-1]))
        for i, kind in enumerate(spec.kinds):
            names = ("kt", "vc") if kind == "attn" else ("ring", "state")
            for name, value in zip(names, ref.states[i]):
                if name == "kt":
                    value = value.transpose(0, 2, 1)
                addr = engine.image.layout.params[f"layer{i}.{name}"][1]
                raw = engine.backend.read(0, addr, value.nbytes).view(value.dtype).reshape(value.shape)
                np.testing.assert_array_equal(raw.view(np.uint8), value.copy().view(np.uint8))
    assert engine.rollback(1) == 0
    np.testing.assert_array_equal(F.bits(engine.step(3)), F.bits(expected[0]))


def test_legacy_layout_and_capability(monkeypatch):
    spec, weights, cfg = fixture()
    monkeypatch.setenv("OTPU_PRECISIONS", "mlp=w4a16")
    legacy = Engine(spec, weights, cap=8, cfg=cfg, prog_cache=False)
    assert legacy.image.policy is None
    assert legacy.layout[0] == "qwen35-native-fp16-v1"
    mixed = Engine(spec, weights, cap=8, cfg=cfg, prog_cache=False, precision="w16a16")
    assert mixed.image.policy("mlp") == "w4a16"
    assert mixed.layout != legacy.layout
    with pytest.raises(ValueError, match="W4A16"):
        Image(spec, replace(cfg, W4A16=False), 8, policy=Policy(2, "w4a16"))


@pytest.mark.parametrize("mode", ["w8a8", "w4a16"])
def test_real_shapes_fit_v80(mode):
    from opentpu import target
    spec, _, _ = fixture()
    spec = replace(spec, hidden=1024, n_q=8, n_kv=2, head_dim=256, rope_dim=64,
                   lin_heads=16, lin_kheads=16, lin_dk=128, lin_dv=128, ffn=3584, vocab=248320)
    image = Image(spec, target.config("v80:mixed"), 2048, policy=Policy(2, mode))
    for key, kernel, kw in image.stages(2047):
        b = kernel.trace(image.cfg, 0, kw)
        program = b.finish()
        assert len(program) * 8 <= image.cfg.IMEM_WORDS, (mode, key, len(program))


def test_model_native_rtl(have_verilator):
    from opentpu import target
    from opentpu.llm.rtl_backend import RtlBackend
    spec, weights, _ = fixture()
    spec = replace(spec, hidden=128, ffn=128, head_dim=32, lin_dv=32)
    rng = np.random.default_rng(72)
    weights = {name: (rng.normal(size=value.shape) * 0.1).astype(np.float32)
               for name, value in _shape_weights(spec).items()}
    cfg = target.config("v80:mixed", TMEM_WORDS=16384, IMEM_WORDS=32768,
                        DRAM_BYTES=1 << 20, ACT_BLOCKS=4)
    rules = "delta=w8a8,mlp=w4a16,wq=w4a16,head=w8a8"
    backend = lambda c, imgs: RtlBackend(c, imgs, axi=True, stall=29,
        uarch=target.sim_uarch("v80:mixed") | {"AW": cfg.AW})
    eng = Engine(spec, weights, cap=8, cfg=cfg, backend=backend, prog_cache=False,
                 precision="w16a16", precisions=rules, group=8)
    ref = Reference(spec, weights, 8, policy=eng.image.policy, D=cfg.D, group=8)
    try:
        first = None
        for token in (3, 9):
            want = ref.step(token)
            np.testing.assert_array_equal(F.bits(eng.step(token)), F.bits(want))
            if first is None:
                first = want
        eng.reset()
        np.testing.assert_array_equal(F.bits(eng.step(3)), F.bits(first))
    finally:
        eng.close()


def test_policy_image_cache(tmp_path, monkeypatch):
    from opentpu import imgcache as IC
    spec, weights, cfg = fixture()

    class Weights(dict):
        model_dir = tmp_path / "checkpoint"

    weights = Weights(weights)
    weights.model_dir.mkdir()
    (weights.model_dir / "config.json").write_text("{}")
    (weights.model_dir / "weights.safetensors").write_bytes((2).to_bytes(8, "little") + b"{}")
    monkeypatch.setenv("OTPU_IMG_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("OTPU_IMG_CACHE_FREE_GB", "0")
    layouts = set()
    for mode, group in (("w16a16", 8), ("w8a8", 8), ("w4a16", 8), ("w4a16", 16)):
        kw = dict(cap=8, cfg=cfg, prog_cache=False, precision=mode, precisions="", group=group)
        first = Engine(spec, weights, **kw)
        assert IC.last[0] == "miss"
        assert first.layout not in layouts
        layouts.add(first.layout)
        want = first.step(3)
        with monkeypatch.context() as patch:
            def no_build(*args):
                raise AssertionError("a cache hit must not pack weights")
            patch.setattr(Image, "build", no_build)
            second = Engine(spec, weights, **kw)
            assert IC.last[0] == "hit"
            np.testing.assert_array_equal(F.bits(second.step(3)), F.bits(want))


def test_chat_precision_cli(tmp_path, monkeypatch, capsys):
    from opentpu.host import chat as C
    spec, weights, cfg = fixture()

    class Tokenizer:
        chat_template = "test"

        def apply_chat_template(self, messages, **kw):
            return [3, 9]

        def decode(self, ids, **kw):
            return " ".join(str(i) for i in ids)

    monkeypatch.setattr(C, "model_dir", lambda name: tmp_path)
    monkeypatch.setattr(C, "has_chat_template", lambda path: True)
    monkeypatch.setattr(C, "load_tokenizer", lambda path: lambda: Tokenizer())
    monkeypatch.setattr(C, "load_spec", lambda path: spec)
    monkeypatch.setattr(C, "load_weights", lambda path: weights)
    monkeypatch.setattr(C, "make_backend", lambda *a, **kw: ("isa", cfg))
    base = ["--model", str(tmp_path), "--cap", "8"]
    C.main(base + ["--precision", "w16a16", "--precisions", "mlp=w4a16", "--w4-group", "8",
                  "--head-precision", "w8a8", "--prompt", "hi", "--max-new", "2", "--greedy"])
    output = capsys.readouterr().out
    assert "weights: w16a16 + mlp=w4a16, head w8a8" in output
    for args in (["--precision", "w4a16", "--wformat", "int4"],
                 ["--precisions", "mlp=w4a16"], ["--head-precision", "w8a8"]):
        with pytest.raises(SystemExit):
            C.main(base + args)


def test_shadowed_mode_does_not_require_unused_capability():
    spec, weights, cfg = fixture()
    policy = Policy(2, "w4a16", "delta=w16a16,attn=w16a16,mlp=w16a16,head=w16a16")
    img = Image(spec, replace(cfg, W4A16=False), 8, policy=policy)
    assert img.matrix_modes == {"w16a16"}
