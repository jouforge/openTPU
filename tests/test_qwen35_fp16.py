"""Complete FP16 gated-attention sublayer, GQA and persistent KV across runs."""
from dataclasses import replace

import numpy as np
import pytest

from opentpu import fp32 as F, rtlsim
from opentpu import fp16 as H
from opentpu.llm.qwen35 import Spec
from opentpu.llm.qwen35_fp16 import AttentionImage, LinearImage, attention_reference, linear_reference
from opentpu.llm.qwen35_fp16 import Engine
from opentpu.runtime import collect, run
from test_fp16_isa import CFG


def fixture(rows=1, cap=8):
    spec = Spec(16, ("attn",), 4, 2, 8, 4, 2, 8, 8, 32, 64)
    rng = np.random.default_rng(35)
    p = "model.layers.0."
    shapes = {"input_layernorm": (16,), "self_attn.q_norm": (8,), "self_attn.k_norm": (8,),
              "self_attn.q_proj": (64, 16), "self_attn.k_proj": (16, 16),
              "self_attn.v_proj": (16, 16), "self_attn.o_proj": (16, 32)}
    W = {p + name + ".weight": rng.normal(size=shape).astype(np.float32) * 0.15
         for name, shape in shapes.items()}
    cfg = replace(CFG, SEQ_WC=True, WC_WIN=16)
    return AttentionImage(spec, W, cfg, 0, cap, rows), W, rng


@pytest.mark.parametrize("rows", [1, 3])
@pytest.mark.parametrize("backend", ["isa", "rtl"])
def test_attention_sublayer(have_verilator, tmp_path, rows, backend):
    img, W, rng = fixture(rows)
    keys = np.zeros((img.spec.n_kv, img.cap, img.spec.head_dim), np.uint16)
    vals = keys.copy()
    images = img.images
    for pos in range(0, 2 * rows, rows):
        x = rng.normal(size=(rows, img.spec.hidden)).astype(np.float32)
        comp = img.compile(pos)
        img.inputs(images, x, pos)
        if backend == "isa":
            result = run(comp, images)
        else:
            dram, tmem, stats = rtlsim.run(img.cfg, comp.programs, images, keep=tmp_path / str(pos),
                                          uarch={"SEQ_WC": 1, "WIN": 16}, max_cycles=1000000)
            result = collect(comp, dram, tmem, stats)
        want, keys, vals = attention_reference(img.spec, W, x, 0, pos, keys, vals)
        np.testing.assert_array_equal(F.bits(result.outputs["out"]), F.bits(want))
        images = result.drams
        for name, expected in (("kt", keys.transpose(0, 2, 1)), ("vc", vals)):
            addr = img.layout.params[name][1]
            raw = images[0][addr:addr + 2 * expected.size].view("<u2").reshape(expected.shape)
            np.testing.assert_array_equal(raw, expected)


def test_attention_image_rejects_wrong_target_and_positions():
    img, W, _ = fixture()
    for cfg in (replace(CFG, FP16=False), replace(CFG, S=2)):
        with pytest.raises(ValueError, match="Config.FP16 and S=1"):
            AttentionImage(img.spec, W, cfg, 0, 8)
    for pos in (-1, 8):
        with pytest.raises(ValueError, match="position"):
            img.compile(pos)
    with pytest.raises(ValueError, match="finite"):
        img.inputs(img.images, np.full((1, 16), np.nan), 0)


def test_prefill_chunks_match_consecutive_decode():
    batch, _, rng = fixture(rows=3)
    decode, _, _ = fixture(rows=1)
    x = rng.normal(size=(6, 16)).astype(np.float32)
    outputs = []
    for image in (batch, decode):
        parts, images = [], image.images
        for pos in range(0, len(x), image.rows):
            image.inputs(images, x[pos:pos + image.rows], pos)
            result = run(image.compile(pos), images)
            images = result.drams
            parts.append(result.outputs["out"])
        outputs.append(np.concatenate(parts))
    np.testing.assert_array_equal(F.bits(outputs[0]), F.bits(outputs[1]))


def test_long_context_program_fits_v80_imem():
    small, W, _ = fixture()
    cfg = replace(CFG, DRAM_BYTES=1 << 20, IMEM_WORDS=32768)
    image = AttentionImage(small.spec, W, cfg, 0, cap=2048, rows=8)
    comp = image.compile(2040)
    assert len(comp.programs[0]) * 8 <= cfg.IMEM_WORDS


def linear_fixture(rows=1, nk=2):
    spec = Spec(16, ("linear",), 4, 2, 8, 4, 4, 64, 8, 32, 64, lin_kheads=nk)
    rng = np.random.default_rng(36)
    C = 2 * nk * 64 + 4 * 8
    p = "model.layers.0."
    shapes = {"input_layernorm.weight": (16,), "linear_attn.in_proj_qkv.weight": (C, 16),
              "linear_attn.in_proj_z.weight": (32, 16), "linear_attn.in_proj_a.weight": (4, 16),
              "linear_attn.in_proj_b.weight": (4, 16), "linear_attn.out_proj.weight": (16, 32),
              "linear_attn.conv1d.weight": (C, 1, 4), "linear_attn.norm.weight": (8,),
              "linear_attn.A_log": (4,), "linear_attn.dt_bias": (4,)}
    W = {p + name: rng.normal(size=shape).astype(np.float32) * 0.15
         for name, shape in shapes.items()}
    cfg = replace(CFG, TMEM_WORDS=16384, DSTEP=True, SEQ_WC=True, WC_WIN=16, DRAM_BYTES=1 << 20, IMEM_WORDS=32768)
    return LinearImage(spec, W, cfg, 0, rows), W, rng


@pytest.mark.parametrize("rows,nk", [(1, 4), (3, 2)])
@pytest.mark.parametrize("backend", ["isa", "rtl"])
def test_linear_sublayer(have_verilator, tmp_path, rows, nk, backend):
    img, W, rng = linear_fixture(rows, nk)
    C = 2 * nk * 64 + 32
    ring = rng.normal(size=(3, C)).astype(np.float32) * 0.1
    state = rng.normal(size=(4, 8, 64)).astype(np.float32) * 0.1
    images = img.images
    for name, value in (("ring", ring), ("state", state)):
        addr = img.layout.params[name][1]
        images[0][addr:addr + value.nbytes] = value.view(np.uint8).ravel()
    comp = img.compile()
    for step in range(2):
        x = rng.normal(size=(rows, 16)).astype(np.float32)
        img.inputs(images, x)
        if backend == "isa":
            result = run(comp, images)
        else:
            drams, tmem, stats = rtlsim.run(img.cfg, comp.programs, images,
                                           keep=tmp_path / str(step), max_cycles=2000000,
                                           uarch={"SEQ_WC": 1, "WIN": 16})
            result = collect(comp, drams, tmem, stats)
        want, ring, state = linear_reference(img.spec, W, x, 0, ring, state)
        np.testing.assert_array_equal(F.bits(result.outputs["out"]), F.bits(want))
        images = result.drams
        for name, expected in (("ring", ring), ("state", state)):
            addr = img.layout.params[name][1]
            got = images[0][addr:addr + expected.nbytes].view("<f4").reshape(expected.shape)
            np.testing.assert_array_equal(F.bits(got), F.bits(expected))


def model_fixture():
    lin, W, rng = linear_fixture()
    attn, wa, _ = fixture()
    W.update({k.replace("layers.0.", "layers.1."): v for k, v in wa.items()})
    spec = replace(lin.spec, kinds=("linear", "attn"), tied=True)
    W["model.embed_tokens.weight"] = rng.normal(size=(spec.vocab, spec.hidden)).astype(np.float32)
    W["model.norm.weight"] = rng.normal(size=(spec.hidden,)).astype(np.float32) * 0.1
    for i in range(2):
        for name, shape in (("post_attention_layernorm", (16,)), ("mlp.gate_proj", (32, 16)),
                            ("mlp.up_proj", (32, 16)), ("mlp.down_proj", (16, 32))):
            W[f"model.layers.{i}.{name}.weight"] = rng.normal(size=shape).astype(np.float32) * 0.1
    return spec, W, lin.cfg


def model_reference(spec, W, tokens, cap):
    from opentpu import language as ol
    ring = np.zeros((spec.conv_k - 1, 2 * spec.lin_nk * spec.lin_dk + spec.lin_heads * spec.lin_dv), np.float32)
    state = np.zeros((spec.lin_heads, spec.lin_dv, spec.lin_dk), np.float32)
    keys = np.zeros((spec.n_kv, cap, spec.head_dim), np.uint16)
    vals = keys.copy()
    outputs = []

    def norm(x, gain):
        scale = F.rsqrt(F.add(F.mul(F.rdot(x, x), np.float32(1 / spec.hidden)), np.float32(spec.eps)))
        return F.mul(F.mul(x, scale[:, None]), F.add(gain, np.float32(1)))

    def mm(x, w):
        return H.matmul(H.pack(x), H.pack(w.T))

    for pos, token in enumerate(tokens):
        x = W["model.embed_tokens.weight"][token:token + 1]
        for i, kind in enumerate(spec.kinds):
            if kind == "linear":
                x, ring, state = linear_reference(spec, W, x, i, ring, state)
            else:
                x, keys, vals = attention_reference(spec, W, x, i, pos, keys, vals)
            p = f"model.layers.{i}."
            xn = norm(x, W[p + "post_attention_layernorm.weight"])
            gate, up = (mm(xn, W[p + "mlp." + n + "_proj.weight"]) for n in ("gate", "up"))
            x = F.add(x, mm(F.mul(F.silu(gate, np.float32(-ol.LOG2E)), up), W[p + "mlp.down_proj.weight"]))
        outputs.append(mm(norm(x, W["model.norm.weight"]), W["model.embed_tokens.weight"])[0])
    return outputs


@pytest.mark.parametrize("backend", ["isa", "rtl", "native"])
def test_full_model_engine_and_reset(have_verilator, backend):
    from opentpu import target
    from opentpu.llm.rtl_backend import RtlBackend
    spec, W, cfg = model_fixture()
    if backend == "native":
        cfg = target.config("v80:fp16", TMEM_WORDS=16384, IMEM_WORDS=32768,
                            DRAM_BYTES=1 << 20, ACT_BLOCKS=4)
        be = lambda c, imgs: RtlBackend(c, imgs, axi=True, stall=29,
                                        uarch=target.sim_uarch("v80:fp16") | {"AW": cfg.AW})
    elif backend == "rtl":
        be = lambda c, imgs: RtlBackend(c, imgs, uarch={"SEQ_WC": 1, "WIN": 16})
    else:
        be = "isa"
    eng = Engine(spec, W, cfg=cfg, cap=8, prog_cache=False, backend=be)
    tokens = [3, 9, 12] if backend == "isa" else [3, 9]
    want = model_reference(spec, W, tokens, 8)
    got = [eng.step(t) for t in tokens]
    np.testing.assert_array_equal(F.bits(got), F.bits(want))
    if backend != "isa":
        return
    eng.reset()
    np.testing.assert_array_equal(F.bits(eng.prefill(tokens)), F.bits(want[-1]))
    assert eng.pos == len(tokens)
    assert eng.rollback(1) == 0
    np.testing.assert_array_equal(F.bits(eng.step(tokens[0])), F.bits(want[0]))


def test_full_model_image_cache(tmp_path, monkeypatch):
    from opentpu import imgcache as IC
    from opentpu.llm.qwen35_fp16 import Image
    spec, weights, cfg = model_fixture()

    class Weights(dict):
        model_dir = tmp_path / "checkpoint"

    W = Weights(weights)
    W.model_dir.mkdir()
    (W.model_dir / "config.json").write_text("{}")
    (W.model_dir / "weights.safetensors").write_bytes((2).to_bytes(8, "little") + b"{}")
    monkeypatch.setenv("OTPU_IMG_CACHE", str(tmp_path / "cache"))
    monkeypatch.setenv("OTPU_IMG_CACHE_FREE_GB", "0")
    first = Engine(spec, W, cfg=cfg, cap=8, prog_cache=False)
    want = first.step(3)
    assert IC.last[0] == "miss"

    def no_build(*args):
        raise AssertionError("cache hit must not build or pack weights")

    monkeypatch.setattr(Image, "build", no_build)
    second = Engine(spec, W, cfg=cfg, cap=8, prog_cache=False)
    assert IC.last[0] == "hit"
    np.testing.assert_array_equal(F.bits(second.step(3)), F.bits(want))


def test_full_model_program_disk_cache(tmp_path, monkeypatch):
    from opentpu import progcache as PC
    spec, W, cfg = model_fixture()
    monkeypatch.setenv("OTPU_PROG_CACHE", str(tmp_path))
    monkeypatch.setenv("OTPU_PROG_CACHE_FREE_GB", "0")
    monkeypatch.setattr(PC, "_mem", {})
    first = Engine(spec, W, cfg=cfg, cap=8)
    want = first.step(3)
    PC._mem.clear()
    hits = PC.stats["disk"]
    second = Engine(spec, W, cfg=cfg, cap=8)
    got = second.step(3)
    assert PC.stats["disk"] - hits == 2 * len(spec.kinds) + 1
    np.testing.assert_array_equal(F.bits(got), F.bits(want))


def test_full_model_long_context_and_shape_validation():
    from opentpu import target
    from opentpu.llm.qwen35_fp16 import Image
    spec, W, cfg = model_fixture()
    real = replace(spec, hidden=1024, kinds=("linear", "attn"), n_q=8, n_kv=2,
                   head_dim=256, rope_dim=64, lin_heads=16, lin_kheads=16,
                   lin_dk=128, lin_dv=128, ffn=3584, vocab=248320)
    image = Image(real, target.config("v80:fp16"), 2048)
    for _, kernel, kw in image.stages(2047):
        b = kernel.trace(image.cfg, 0, kw)
        assert len(b.finish()) * 8 <= image.cfg.IMEM_WORDS
    W["model.norm.weight"] = W["model.norm.weight"].reshape(1, -1)
    with pytest.raises(ValueError, match="model.norm.weight: expected"):
        Image(spec, cfg, 8, W)


def test_native_fp16_chat_cli(tmp_path, monkeypatch, capsys):
    from opentpu.host import chat as C
    spec, W, cfg = model_fixture()

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
    monkeypatch.setattr(C, "load_weights", lambda path: W)
    monkeypatch.setattr(C, "make_backend", lambda *a, **kw: ("isa", cfg))
    C.main(["--model", str(tmp_path), "--wformat", "fp16", "--prompt", "hi",
            "--max-new", "2", "--greedy", "--cap", "8", "--no-prog-cache"])
    output = capsys.readouterr().out
    assert "weights: fp16, head fp16" in output and "prefill 2 tokens" in output
    monkeypatch.setattr(C, "make_backend", lambda *a, **kw: ("isa", replace(cfg, FP16=False)))
    with pytest.raises(SystemExit):
        C.main(["--model", str(tmp_path), "--wformat", "fp16", "--cap", "8"])
    assert "requires a native FP16 bitstream" in capsys.readouterr().err
