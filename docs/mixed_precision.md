# Mixed matrix precision

`feat/mixed-precision` starts from the qualified FP16 revision `c3f961eb`.
Its scope is explicit W16A16, W8A8 and W4A16 selection within dense Qwen3.5,
including actual packed storage, device arithmetic, cache identities and numerical
checks. Resource scaling and clock tuning remain in `perf/v80`.

## Numerical boundaries

| Mode | Weight storage | Activation boundary | Accumulation/result |
| --- | --- | --- | --- |
| `w16a16` | binary16 | device FP32 to binary16, RNE | increasing-K FP32, existing HMM |
| `w8a8` | existing MX INT8 with its scales | existing QACT INT8 quantization | existing integer MM and FP32 epilogue |
| `w4a16` | signed INT4 and a binary16 scale per output/K group | device FP32 to binary16, RNE | expand a weight tile to binary16, then increasing-K FP32 HMM |

The W8A8 contract is the existing [MX arithmetic](mx_mxu.md); its integer block
accumulation/scale application is preserved. W16A16 and the expanded W4A16 operands
use the [FP16 contract](fp16.md). Norms, nonlinearities, residuals and DeltaNet
state retain the staged model's FP32 arithmetic. Its KV cache and attention
matrix operands retain their FP16 boundaries independently of projection choices.
Switching projection precision does not silently change recurrent/cache formats.

W4A16 here is symmetric signed INT4, independent of the existing MX `int4`, FP4
and NVFP4 layouts. For checkpoint weights `[N,K]`, groups run along K separately
for each output. The default group is 128; powers of two from 8 through 128 are
supported by the host contract. A short final group uses only its real elements.

For each group, let `peak=max(abs(w))`. A zero group has scale +0 and code 0.
Otherwise the scale is `half_RNE(fp32(peak/7))`, clamped to the finite half interval
`[2^-24, 9352]`. The upper bound keeps `half_RNE(7*scale)` finite, even at the
largest accepted checkpoint magnitude 65504. Inputs must be finite FP32 values
within the finite binary16 range; unsupported values fail instead of silently
becoming NaNs or infinities. Codes use round-to-nearest-even of the float64
quotient `w/scale`, then clip to `[-7,7]`. This is checkpoint preprocessing only.

The device decoder defines all sixteen two's-complement codes, including -8:
`weight_half = half_RNE(exact_fp32(code * scale_half))`. This rounding is part of
the contract; multiplying a half activation by an unrounded FP32 weight is a
different computation. Decode preserves half subnormals and signed zeros and
uses the FP16 canonical NaN/overflow rules for arbitrary instruction operands.
Every matrix output accumulates its half products in increasing K order. There
is no group-level partial-sum reduction or extra accumulator rounding.

## Packed layout and selection

`opentpu/int4.py` defines checkpoint packing and the numerical oracle. The logical
`[N,K]` weight is stored K-major as `uint32[K,ceil(N/8)]`: output column `j` uses
bits `4*(j%8)+:4` of word `j//8`, low nibble first. Scales are little-endian
binary16 `[ceil(K/group),ceil(N/2)*2]`. Trailing code nibbles and scale halfwords
are zero. Shape, group size, mode and this layout's version must enter image and
program cache identities. Packed device weights must not be expanded permanently
on the host or in HBM and then presented as W4A16 support.

`opentpu/llm/precision.py` provides explicit matrix policies. Rules use the existing
`kind@first-last=value` grammar: ranged rules precede global rules; the first rule
among equals wins. `gateup` and `down` inherit `mlp`; `wq`, `wk`, `wv`, `wo` inherit
`attn`; DeltaNet uses `delta`; `head` has no layer range. An explicit head choice
wins over the head rule. Out-of-range layer rules fail. For example:

```python
from opentpu.llm.precision import Policy

policy = Policy(24, mode="w16a16",
                rules="mlp=w4a16,attn=w8a8,wq@3=w16a16", head="w16a16")
```

Explicit rules override `OTPU_PRECISIONS`; an explicit empty string disables it.
`OTPU_FORMATS` and existing `--wformat int4/fp4/nvfp4/mix` retain their current
meaning. No new mixed policy is recommended automatically before quality checks.
The policy identity records resolved choices, so equivalent rule spellings share
an identity while different operator choices do not alias.

## Integration and acceptance

The numerical contract, HDQ instruction, compiler/runtime placement, kernels and
staged model dispatch are implemented. `v80:mixed` enables FP16 and W4A16;
`v80:fp16` retains its qualified architecture hash. The new variant is **not yet
board-qualified**. Keep the qualified FP16 image until its replacement passes
route and deployment checks.

The explicit chat interface is:

```bash
otpu-chat --model /path/to/Qwen3.5-0.8B --backend isa \
  --precision w16a16 --precisions 'mlp=w4a16,attn=w8a8' \
  --head-precision w16a16 --w4-group 128 --cap 2048
```

This example illustrates selection, not a recommended quality policy. Use
`--backend board --dev qdma:BDF@Q` only after programming a matching image.
Do not combine `--precision` with legacy `--wformat` or `--head-format`.
The staged engine currently supports dense Qwen3.5 text, one sequence, no MTP,
and a positive even context capacity. Full-K W8A8 projections require K divisible
by the target's D and enough ACT/TMEM capacity; W4A16 uses K groups independently
of the requested kernel K tile size. W16A16 retains its existing K tiling.
The cache ABI includes the resolved precision policy and W4 group size. Normal
FP16 invocation keeps its existing cache ABI and ignores OTPU_PRECISIONS.

`tools/fp16_model_check.py` accepts the same precision arguments and checks each
layer residual, persistent state and logits against the selected numerical
oracle, then reports differences against the independent FP32 reference.

1. Numerical oracle and rules: exhaustive decoding of every half scale encoding
   with all 16 codes, independent rounding/packing checks, short groups, odd N/K,
   large-row chunk boundaries, exceptional values and policy/cache identities.
2. ISA and RTL: explicit capability for packed INT4 decode, operand bounds and
   footprints, backpressure, cancellation and old-image rejection. Do not change
   existing HMM/HCVT semantics or the qualified `v80:fp16` target's identity.
3. Compiler/kernels: packed weight placement, on-device tile decode and W4A16
   multiplication; W8A8 projection dispatch preserves QACT/MM semantics. Exercise
   transitions between all three modes and distinguish cache/layout identities.
4. Model: explicit per-kind/layer policy, full Qwen3.5 prefill/decode/reset,
   layer/state/logit comparisons and independent FP32 quality comparisons.
5. V80: complete route/audit/export, real-card diagnostics and legacy regressions,
   numerical agreement, image/cache sizes and cold/warm performance per policy.

Tests and generated reports follow the existing `tests/`, `tools/` and ignored
`build/mixed_precision/` layout. Record quality separately from ISA/RTL agreement;
a bit-exact implementation does not prove that a quantization policy preserves
model accuracy.
