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
meaning. No mixed policy is selected automatically; deployment choices need
workload-specific quality checks.
W4A16 reduces weight storage but adds device decode work and still uses HMM;
it is not a promise of higher throughput. Measure device cycles separately from
program compilation, image loading and the staged host launch overhead.
The policy identity records resolved choices, so equivalent rule spellings share
an identity while different operator choices do not alias.

## Integration and acceptance

The numerical contract, HDQ instruction, compiler/runtime placement, kernels and
staged model dispatch are implemented. `v80:mixed` enables FP16 and W4A16;
`v80:fp16` retains its qualified architecture hash. The new variant is
**board-qualified on 2026-11-15**, including complete Qwen3.5-0.8B checks and
explicit mixed-policy execution. The qualified FP16 artifacts remain available.

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
For example, after loading a matching `v80:mixed` image:

```bash
export OTPU_TARGET=v80:mixed
otpu-diag --dev qdma:BDF@Q --only isa,system --json diag.json
PYTHONPATH=. python tools/fp16_model_check.py /path/to/Qwen3.5-0.8B \
  --backend board --dev qdma:BDF@Q --precision w16a16 \
  --precisions 'mlp=w4a16,attn=w8a8' --head-precision w16a16 \
  --output mixed-board.json
```

The model report's `step_wall_s` includes host work and device execution;
`device_s` derives from measured cycles and the card's reported clock. ISA runs
have no measured hardware clock and report `device_s: null`.

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

## Component timing evidence

The `0c31d07d` HDQ/HMM/HCVT component was routed on `xcv80-lsva4737-2MHP-e-S`
with Vivado 2026.1.1, N=8 and W4A16=1. At a 10 ns period its registered OOC
boundary has WNS +3.808 ns and WHS +0.032 ns, using 9,896 LUTs, 5,201 FFs and
16 DSPs, with no BRAM/URAM. This passes the component timing/DRC gate; it is
not a whole-board timing or deployment result. The reproducible invocation is:

```bash
vivado -mode batch -source rtl/exp/fp16/ooc.tcl \
  -tclargs /path/to/output xcv80-lsva4737-2MHP-e-S 8 10 mxu 1
```

The full `v80:mixed` build and real-model checks use a separate build tree. The
qualified `feat/fp16` artifacts and original V80 flash are retained.

The implementation tests pass through test revision `b12bba4c`: 67 mixed-precision
numerical/compiler/model/RTL tests, plus two native-memory odd-shape/group-tail
checks and one legacy-scoreboard RTL check, plus 221 host/QDMA/target/build-flow tests. The
latter suite skips one unrelated LFM2.5 test because its checkpoint is absent.
The native-memory model test includes independent read/write backpressure,
consecutive tokens and reset/replay; the HDQ unit additionally sweeps 100 cancel
offsets for each of N=2 and N=8. Full-model FP16 ISA/RTL regressions also pass.
These software checks supplement the physical-board qualification below.

The complete `v80:mixed` implementation at hardware revision `0c31d07d`
passes the project route gate on 2026-11-15: WNS +0.444 ns, WHS +0.010 ns,
WPWS 0.000 ns, and worst bus-skew slack +9.197 ns across 84 reported paths.
There are no DRC errors or critical warnings. Core/TMEM clocks remain 100/200 MHz.
Routed use is 220,805 LUTs, 196,628 FFs, 1,040 RAMB36 plus 40 RAMB18
(1,060 BRAM36 equivalents), 1,356 DSPs and no URAM. Relative to the qualified
FP16 implementation, this adds eight DSPs and leaves BRAM use unchanged.
Export review, independent constraint audit and physical-board qualification
are separate gates.

## Quantization quality checks

The Qwen3.5-0.8B checks use fixed input tokens `[198, 17]`, including all 24 layer
residuals, recurrent states and final logits. Uniform W8A8, uniform W4A16 and
`w16a16` with `mlp=w4a16,attn=w8a8` and a W16A16 head each match their staged
numerical reference exactly in the ISA simulator for both steps.

That arithmetic agreement does not remove quantization error. Uniform W4A16
has logit cosine similarity 0.7840/0.9323 against the independent FP32 reference;
uniform W8A8 has 0.9953/0.9989. The mixed policies below were separately compared
with Hugging Face FP32 (`Qwen3_5ForCausalLM`, Transformers 5.22.2, PyTorch 2.15.0).
Each has a W16A16 head; image sizes include the staged model's state at capacity 8.

| Base mode and overrides | Image bytes | HF logit cosine, steps 0/1 | KL(HF || policy), steps 0/1 |
| --- | ---: | --- | --- |
| W16A16; `mlp=w4a16,attn=w8a8` | 1,092,804,096 | 0.8402 / 0.9734 | 1.0894 / 0.0316 |
| W8A8; `gateup=w4a16` | 949,108,736 | 0.9843 / 0.9904 | 0.1354 / 0.0051 |
| W8A8; `gateup=w4a16,delta=w16a16` | 1,136,267,264 | 0.9838 / 0.9901 | 0.1235 / 0.0038 |

All three mixed policies retain the HF greedy token for these two positions.
Two positions are insufficient to recommend a policy or establish corpus-level
accuracy. In this model, retaining the large vocabulary head in W16A16 can also
make a mixed image larger than uniform W8A8. The original automatic/INT8 default
is retained; full W4A16 is an explicit experimental choice with visible quality
loss. Assess a representative prompt/task corpus before selecting a deployment
policy, and measure throughput independently of the packed weight size.

## V80 deployment and validation (2026-11-15)

The complete build uses frozen hardware revision `0c31d07d`, target `v80:mixed`
and CFG_HASH `154f92b2`. Board/model checks use runtime `edaa60bf`; subsequent
test/documentation commits preserve all 53 recorded hardware inputs. Vivado and
Vitis are 2026.1.1, AMR is `c21a01b618be35511b2d88a8946bd8859639c713`, and
`amr_vivado_designs` is `4b7d244758c683ec4c45f357e65511c25cff8f38`.
Use the existing [V80 build flow](../boards/v80/README.md#platform-flow) with
`TARGET=v80:mixed` and a fresh `OUT_DIR`. Local Tcl and the input manifest are
generated as before; no dependency lock file is introduced.

Independent review passed XSA CRCs, embedded/standalone PDI hash agreement,
NoC files and BIF references. The routed constraint audit passed all 24 scoped
RAM exceptions and 16 read-FIFO Gray groups, including actual fan-in coverage.
The AMC firmware was built from this XSA and loaded with the PL by volatile JTAG
on qfpga0. The loaded card reports build `0c31d07d`, FP16 and W4A16 capabilities,
100 MHz, and the expected target geometry. AMC is READY and QDMA queue 4 is online.

| Artifact | Bytes | SHA-256 |
| --- | ---: | --- |
| `otpu_v80.xsa` | 26,520,217 | `901f8b97d99d4ac41a0e7c343f447342ebb26c6cd6c1ac4cd3920881d9919654` |
| `otpu_v80_pld.pdi` | 18,011,904 | `3d766daa8f5d33e34ea0026842022147437cc76e2c57ada804b25a11f92f7354` |
| `otpu_v80_amc_boot.pdi` | 4,095,568 | `ada8ca64aac9d0823382b85080a1675f79bb5fda7a7cccd85682e0ccd8812015` |

Physical checks passed seven register tests, 49 memory tests, 105 ISA tests and
five system tests. Memory testing includes March C- over all eight 4 GiB channels
(32 GiB total), with no bad bytes. ISA checks include all three FP16 cases and
two HDQ signed-code/half-scale cases, including odd output counts. The system
soak ran 20 iterations. HDQ was interrupted at four delays from 0.1 to 7 ms;
every cancellation was followed by exact FP16/HDQ recovery.

The unchanged legacy INT8 path passed eight prompts and 128 generated steps
against the retained ISA baseline: identical tokens and bit-exact logits,
maximum error 0 ULP. Separately, all five staged policies below passed both
fixed input tokens `[198,17]` at capacity 8, including every layer residual,
all recurrent/KV state, finite full logits and recorded logit hashes:

- uniform W16A16, W8A8 and W4A16;
- W16A16 with `mlp=w4a16,attn=w8a8` and a W16A16 head;
- W8A8 with `gateup=w4a16` and a W16A16 head.

W8A8, W4A16 and the first mixed policy also reproduce the full ISA-run hashes;
W16A16 reproduces the independently retained FP16 reference, and both mixed
policies reproduce the independent quality-check reference arrays. Numerical
agreement is separate from the quality limitations above.

The physical/model checks, fixed-token performance records, CLI conversation and
final management/FPT audit have passed. After testing, AMC remained READY,
both PCI functions retained the established Gen4 x8 link, the data function used
`qdma-pf`, and queue 4 remained online in both directions. The final card status
was idle/calibrated without ERROR, AXI_ERR, ECC_DED or WAIT timeout. Original
Flash/FPT contents were unchanged; deployment used volatile JTAG only.
The diagnostic CLI skips its XDMA-only PCIe checks on QDMA; PCIe/driver health
was checked independently through sysfs. Temperature and power were not measured.

### Loading and execution baseline

Each policy was loaded at capacity 2048 first with a fresh image-cache directory
and then with a verified cache hit. Cold here means image-cache miss; the
checkpoint's operating-system file cache was not flushed. Loads include checkpoint
loading, image construction/cache access and board initialization/upload. These
are individual measurements, not repeated-trial distributions.

For step timing, capacity is 8. Each engine first executes `[198,17]`, resets,
then repeats those tokens with its in-process programs warm. The table reports
the second measured position. Both outputs reproduce their qualification hashes,
and monitoring reports the whole model's cycle count and correct KV usage;
the status file is removed on close. Device time is cycles / 100 MHz.

| Policy | Image bytes (cap 2048) | Cold / warm load (s) | Device / step (s) | Wall / step (s) |
| --- | ---: | --- | ---: | ---: |
| W16A16 | 1,553,745,920 | 31.97 / 1.37 | 13.0215 | 13.3939 |
| W8A8 | 810,366,976 | 10.55 / 1.52 | 0.1043 | 0.2569 |
| W4A16 | 437,652,480 | 8.64 / 1.11 | 18.3582 | 18.8665 |
| W16A16; MLP W4A16, attention W8A8, head W16A16 | 1,117,964,544 | 30.34 / 1.69 | 14.1494 | 14.5508 |
| W8A8; gate/up W4A16, head W16A16 | 974,274,560 | 29.31 / 1.33 | 8.7482 | 8.9995 |

Cold image construction/cache writing takes 8.32–29.50 s in these measurements;
board initialization/upload takes 0.21–1.07 s. Image caching removes most repeated
loading work. It does not eliminate first-use program generation or the device's
arithmetic cost. W4A16 reduces image size but is slower than W16A16 in this first
implementation because it adds tile decoding to the existing HMM path. These
fixed-token results do not measure long-context or batched throughput.

All rows use the explicit staged engine and its FP16 KV/attention boundaries.
The W8A8 row is not the unchanged legacy `--wformat int8` engine's benchmark.
There is no automatic selection of one of these policies; the quality tradeoffs
above must be assessed independently of memory size and speed.

### Chat and retained evidence

The capacity-2048 CLI run used W8A8 with `gateup=w4a16`, a W16A16 head, greedy
selection, and the prompt `用一句话介绍自己。`. It loaded in 32.1 s with a fresh
image cache (28.3 s building/caching), then took 268.41 s to the first token with
a cold program cache and 17 prefill tokens. It generated 16 tokens before the
requested `max_new` limit, producing:

> 我是一个人工智能助手，擅长通过逻辑推理、代码编写、数据分析及创意写作

This is a bounded CLI smoke test, not a complete-answer or accuracy benchmark.
The CLI reports 0.10 wall tok/s during decode and 875.68 Mcycles/token at 100 MHz
(about 8.76 device seconds/token); its one-decimal device-rate display rounds to
0.1 tok/s. TTFT excludes model loading and includes first-use program generation
and sequential prefill. It is separate from the warm capacity-8 table above.

On the already programmed qfpga0, the retained test entry point is:

```bash
~/workspace/openTPU-mixed-build/full-v1/run-chat.sh \
  --precision w8a8 --precisions 'gateup=w4a16' --head-precision w16a16 \
  --greedy --prompt '用一句话介绍自己。' --max-new 16
```

The wrapper selects the final runtime, `OTPU_TARGET=v80:mixed`, QDMA queue 4
and persistent image/program caches. Without precision arguments it retains the
existing automatic/INT8 chat path. Explicit mixed execution remains a functional
baseline; arithmetic/memory parallelism, program preparation, clocks and scaling
remain work for `perf/v80`.

Evidence is retained on qfpga0 under
`/home/pqq/workspace/openTPU-mixed-build/full-v1/board-qualification/20261115_014903/`,
with local copies under `build/mixed_precision/full-v1/`. The matching `deploy/`
contains `QUALIFIED.json`, artifact hashes, the input manifest, route audit and
XSA/PL/AMC images. `ACCEPTED.json` records the earlier implementation-only gate;
`QUALIFIED.json` closes the board/model/cache/health gates and records their exact
runtime revision and evidence hashes. Later documentation/test-only commits do
not replace this qualification provenance.
