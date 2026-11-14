"""Complete FP16 gated-attention sublayer, GQA and persistent KV across runs."""
from dataclasses import replace

import numpy as np
import pytest

from opentpu import fp32 as F, rtlsim
from opentpu.llm.qwen35 import Spec
from opentpu.llm.qwen35_fp16 import AttentionImage, attention_reference
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
