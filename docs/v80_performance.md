# V80 performance development

`perf/v80` starts at the qualified mixed-precision source `a9209340`.
The retained board image is build `0c31d07d`, architecture `154f92b2`, at
100 MHz core / 200 MHz TMEM. Its full-model timings and quality limitations are
in [mixed_precision.md](mixed_precision.md). This branch is under development;
the measurements below distinguish that physical baseline from candidate RTL.

## Scope and acceptance

Optimize all three projection paths and their transitions while preserving the
existing ISA, packing, quantization, ordered accumulation and reset contracts.
The baseline remains recoverable. A faster operator or isolated timing result
does not qualify a complete overlay or establish model throughput.

The work proceeds in this order:

1. Record comparable operator, model, host-time and hierarchical resource
   baselines. Identify unnecessary transfers before increasing compute lanes.
2. Reuse existing buffers and reduce HMM/HDQ conversion and memory overhead.
   Keep the storage-format handling, matrix arithmetic and command ownership
   separate. Check every mode, including transitions to the integer MXU.
3. Measure queue/buffer sizing and technology mapping. Compare generic and
   xilinxversal implementations with the same workload, clocks and numerical
   checks. Reduced DSP use alone is not an acceptance criterion: record LUT,
   FF, BRAM, timing and sustained throughput together.
4. Increase frequency and then useful parallelism, based on measured operand
   supply and full implementation timing. Select settings from measured tradeoffs,
   not a device-family clock estimate. Requalify memory, reset, legacy INT8 and
   all five retained model policies on the physical V80.

Every accepted change needs bit-exact results against the appropriate numerical
oracle and regression evidence for the other modes. Board results include build
ID, core clock, device cycles and host wall time. First compilation, cold image
construction, warm loading and steady execution are separate measurements.

## Shared hardware boundaries

| Resource | Current ownership | Optimization boundary |
|---|---|---|
| HBM/NoC, DMA, TMEM | All modes use the common memory subsystem | Reduce transfers and size queues without starving concurrent users |
| MXU command slot and TMEM ports | Integer MM and HMM/HCVT/HDQ serialize through the slice adapter | Preserve pending writes, read replay and mode-switch hazards |
| Half dot-product lanes | W16A16 and W4A16 share HMM and ordered FP32 accumulators | Improve operand supply without changing reduction order |
| HMXU beat register | Conversion/write staging; reused for HMM activation blocks in the candidate | Exclusive ownership by command/state; no extra activation RAM |
| Integer and half arithmetic | Separate datapaths; HDQ also has its own scale multipliers | Share arithmetic only when scheduling, rounding and timing costs justify it |
| Build features | `FP16=0` removes the half path; `W4A16=0` removes HDQ | Runtime policy switching requires the corresponding hardware to remain present |

The large buffers are not all TMEM. In the retained mixed build's **synthesis**
hierarchy, the integer MXU owns 548 RAMB36 + 5 RAMB18, TMEM owns 256 RAMB36,
and activation RAM owns 136 RAMB36. The integer MXU also owns 1,194 DSP blocks.
These are inclusive module counts from synthesis, not routed totals or additive
counts of every child instance. The full **routed** baseline uses 1,040 RAMB36 +
40 RAMB18 and 1,356 DSP blocks. Buffer experiments should therefore include the
integer weight-stream queues, not only TMEM and the half extension.

## Reproducible projection measurements

`tools/perf_precision.py` compiles the same FP32 input/weight arrays through the
three explicit paths, compares every output bit to the independent numerical
reference, and records each run separately. Shape order is M,N,K. The default
tile is `(1,128,128)`; W4 groups contain 128 K elements. The complete program
includes activation conversion, weight loads/dequantization and output stores.

```bash
PYTHONPATH=. python tools/perf_precision.py --backend rtl --repeat 1 \
  --output build/perf_v80/projection-rtl.json
PYTHONPATH=. python tools/perf_precision.py --backend board \
  --dev qdma:0000:41:00.1@4 --output build/perf_v80/projection-board.json
```

The RTL run uses the target's native AXI path with reduced storage capacities
(recorded in the JSON) and an uncalibrated memory model. Compare candidate and
baseline with identical parameters; do not extrapolate its time to HBM bandwidth
or tokens per second. RTL wall time includes simulator build/startup. The board
run uses the same reduced compiler capacities on the actual overlay; those
compiler limits do not shrink its physical memories. Its first run includes
program upload in wall time; repeats can reuse the resident program. Device
cycles exclude host compilation/upload/readback.

The JSON contains source hashes, revision/dirty state, configuration, inputs'
seed, output hashes, clock/build information on the board, and all repeated
samples. `complete` becomes true only after all requested cases pass.

Physical baseline, 2026-11-15, three repeats, **median device cycles**:

| M,N,K | W16A16 | W8A8 | W4A16 |
|---|---:|---:|---:|
| 1,128,128 | 28,392 | 356 | 40,092 |
| 4,128,128 | 113,607 | 1,639 | 160,414 |

These small operators expose control/arithmetic differences; they are not a
model benchmark. All 18 physical runs matched their mode's reference exactly.
Evidence: `build/perf_v80/baseline-board.json`, also retained on qfpga0 at
`/home/pqq/workspace/openTPU-perf-build/baseline/projection.json`.

The same board was also profiled by complete model stage using Qwen3.5-0.8B,
tokens `[198,17]`, cap 8, two initial steps followed by reset and a warm replay.
All 20 outputs matched the retained qualification logits bit for bit. The warm
second step gives the following **device seconds at 100 MHz**; stage categories
include their normalization, state/cache updates and other surrounding work.

| Policy | DeltaNet stages | Attention stages | MLP stages | Output head | Total device | Host wall |
|---|---:|---:|---:|---:|---:|---:|
| W16A16 | 3.30138 | 0.77314 | 4.55409 | 4.39296 | 13.02159 | 13.37342 |
| W8A8 | 0.03727 | 0.01864 | 0.02542 | 0.02301 | 0.10433 | 0.24792 |
| W4A16 | 4.64349 | 1.08698 | 6.43701 | 6.19075 | 18.35822 | 18.85166 |
| mixed | 3.30163 | 0.01861 | 6.43703 | 4.39289 | 14.15015 | 14.55977 |
| gateup4 | 0.03723 | 0.01863 | 4.29955 | 4.39273 | 8.74814 | 8.98492 |

`mixed` is W16A16 with `mlp=w4a16,attn=w8a8`; `gateup4` is W8A8 with
`gateup=w4a16`; both use a W16A16 head, as in the previous qualification. The
gateup4 case spends about half its device time in that head. This motivates
improving the common half execution path before selecting more aggressive
quantization. Conversely, W8A8's substantial host overhead needs a separate
program dispatch/transfer investigation. No policy is automatically changed.

The per-stage sums equal the engine's complete-token cycle counter. Evidence:
`build/perf_v80/baseline-model.json` and qfpga0's
`/home/pqq/workspace/openTPU-perf-build/baseline/model-profile.json`. These are
single warm samples per position, not a statistically sampled throughput test
or a longer-context benchmark.

## First candidate: reuse the HMXU staging beat

Previously HMM fetched one activation word for every K step, reading the same
packed word twice. The candidate fetches up to `ULANES` consecutive words into
the existing `data_q` register and consumes their low/high halves in increasing
K order. At eight lanes that is one activation request per 16 K steps instead
of 16 separate requests. It still performs every original multiply/add.

The staging beat belongs to the activation block during HMM; after the final
dot result it belongs to the output write. Accumulating HMM keeps its seed in
the existing separate seed registers. K=0 never enters the activation path.
HCVT and HDQ continue using the same register for their output beats. This
changes the controller and its selection logic without adding another RAM,
multiplier array, instruction or storage format.

Each output tile starts with a new activation read. Refills request only words
inside the logical row, and an odd K tail never consumes the unused high half.
Read/write backpressure must keep requests stable; reset cancels ownership so
a following command cannot consume an old activation block.

`tb_hmm` checks those boundaries at 2, 8 and 16 lanes, including 100 reset offsets
per width, poisoned cancelled operands, an odd K just past the refill boundary,
multiple rows/column tiles, accumulator seeds and untouched output padding.
The full instruction and model tests additionally compare arbitrary FP16
encodings and mode transitions against the ISA/numerical references.

Isolated HMXU synthesis/place/route on the same V80 part, Vivado 2026.1.1,
8 lanes, W4A16 enabled, 10 ns clock and the existing registered OOC wrapper:

| Metric | Retained mixed baseline | Activation reuse candidate |
|---|---:|---:|
| LUT | 9,896 | 10,527 |
| FF | 5,201 | 5,219 |
| DSP | 16 | 16 |
| BRAM / URAM | 0 / 0 | 0 / 0 |
| Setup slack, ns | +3.808 | +3.690 |
| Hold slack, ns | +0.032 | +0.031 |

The control/address/selection logic costs 631 LUT and 18 FF in this isolated
comparison. Timing and DRC pass, but these results do not establish full-overlay
timing or resources. Evidence: `build/perf_v80/ooc/`; the checkpoints remain at
`/home/pqq/workspace/openTPU-perf-build/hmxu-ooc/`.

Native RTL projection cycles, the same parameters and seed on both revisions:

| M,N,K | Mode | Baseline | Activation reuse | Cycle reduction |
|---|---|---:|---:|---:|
| 1,128,128 | W16A16 | 24,273 | 20,433 | 15.82% |
| 1,128,128 | W8A8 | 252 | 252 | 0% |
| 1,128,128 | W4A16 | 39,960 | 36,120 | 9.61% |
| 4,128,128 | W16A16 | 96,933 | 81,532 | 15.89% |
| 4,128,128 | W8A8 | 940 | 940 | 0% |
| 4,128,128 | W4A16 | 159,820 | 144,462 | 9.61% |

Every output matches its independent oracle and the corresponding baseline
output hash. These are whole-program simulation cycles, not measured board
speedups. Evidence: `build/perf_v80/baseline-rtl-final.json` and
`build/perf_v80/candidate-rtl.json`. The unchanged dot-product feedback interval
and the separate W4 decode work limit the gain from activation reuse alone.

## Combined candidate: overlap multiplication and accumulation

`otpu_hdot` now selects the FP32 accumulator when the product reaches the adder,
instead of delaying an earlier accumulator value through the multiplier pipe.
The next multiplication can run while the preceding addition completes. Every
sum still commits before the next product enters its adder, so increasing K
order, rounding and the first-result latency are unchanged. The acceptance
interval becomes `LA + 1`, five clocks at the defaults, instead of
`LM + LA + 1`, seven clocks. Captured seeds and pipelined first/last tags preserve
input gaps, held outputs and reset cancellation with overlapping beats.

This is the current candidate, including the activation reuse above. The same
isolated HMXU flow passes at 10 ns with WNS **+3.748 ns**, WHS **+0.038 ns**,
10,741 LUT, 5,041 FF, 16 DSP and no BRAM/URAM. Against activation reuse alone,
it trades 214 more LUT for 178 fewer FF and a shorter arithmetic feedback wait.
Against the original mixed baseline, the combined cost is +845 LUT and -160 FF,
with the same DSP/RAM counts. These remain component-level measurements.

| M,N,K | Mode | Original baseline cycles | Combined cycles | Cycle reduction |
|---|---|---:|---:|---:|
| 1,128,128 | W16A16 | 24,273 | 16,593 | 31.64% |
| 1,128,128 | W8A8 | 252 | 252 | 0% |
| 1,128,128 | W4A16 | 39,960 | 32,276 | 19.23% |
| 4,128,128 | W16A16 | 96,933 | 66,165 | 31.74% |
| 4,128,128 | W8A8 | 940 | 940 | 0% |
| 4,128,128 | W4A16 | 159,820 | 129,102 | 19.22% |

All six native AXI RTL projection results remain bit-exact, with unchanged
programs and output hashes. `tb_hdot` also checks the new acceptance interval
and cancellation with two accepted beats in flight, for multiple lane counts
and multiplier/adder latencies. The instruction/kernel/model regressions and
full V80 implementation are separate gates. Physical results for this combined
candidate follow below.

Evidence: `build/perf_v80/pipeline-rtl.json`, `pipeline/hdot-overlap.log` and
`pipeline-ooc/`. RTL/runtime source hashes in the isolated benchmark match the
main checkout. The matching OOC sources/reports/checkpoints are retained on
qfpga0 under `pipeline-src/` and `pipeline-ooc/` in the performance build root.

## Candidate validation and full implementation

The combined hardware source is `913a298d`. Its focused regression passes all
44 tests: FP16 instructions, HMM reset/backpressure at three lane counts,
mixed-precision execution, a complete small mixed model on native AXI including
reset/replay, a larger native FP16 matrix tile, HDQ reset, and floating-point
arithmetic. The isolated test checkout matches all 475 tracked RTL, runtime and
test files in this revision. Evidence: `build/perf_v80/pipeline-regression.log`
and `pipeline-source-review.json`.

```bash
python -m pytest -q tests/test_fp16_isa.py tests/test_mixed.py \
  tests/test_qwen35_mixed.py::test_model_native_rtl \
  tests/test_fp16_tiled.py::test_model_tile_native_memory \
  tests/test_int4_isa.py::test_decode_reset_rtl tests/test_fp16.py
```

The broader final sweep passes **151 tests**, covering FP16/INT4 instructions,
mixed projections and models, tiled FP16, compiler, attention, full FP16 models,
precision policy and arithmetic. An initial sweep exposed `ask_plain` retaining
the standard-output stream captured when the module was imported; a later CLI
invocation could write to an already closed capture stream. Runtime `8c8f6450`
resolves the default stream at each call. Both the 12-test model/CLI subset and
the complete 151-test sweep pass after that fix. Evidence:
`build/perf_v80/chat-regression.log`, `final-regression-fixed.log` and
`regression-review.json`. No hardware input changed with the CLI fix.

The full V80 build uses a clean, detached copy of `913a298d` at
`/home/pqq/workspace/openTPU-perf-build/full-v1/src`. All 53 hardware inputs match
the generated manifest; only `otpu_hmxu.sv` and `otpu_hdot.sv` differ from the
qualified mixed image's hardware inputs. Clocks remain 100/200 MHz and the
architecture hash remains `154f92b2`; the new build ID is `913a298d`.

Full implementation, export/CDC review and matching AMC packaging pass. The
independent route audit finds the expected 24 RAM exceptions and 16 Gray-code
groups. XSA/PDI hashes and BIF references agree, and implementation reports no
DRC errors or critical warnings.

| Full routed design | Mixed baseline | Combined candidate |
|---|---:|---:|
| LUT | 220,805 | 220,253 |
| FF | 196,628 | 196,638 |
| DSP | 1,356 | 1,356 |
| RAMB36 / RAMB18 | 1,040 / 40 | 1,040 / 40 |
| URAM | 0 | 0 |
| Setup / hold slack, ns | +0.444 / +0.010 | +0.224 / +0.010 |

Whole-design optimization makes these deltas different from isolated HMXU
results; do not add the component delta to the previous overlay total. Evidence:
`build/perf_v80/full-v1/evidence/`, and the matching server `platform/reports/`,
`platform/artifact_review.json`, `route_audit.log` and `deploy/ACCEPTED.json`.

The candidate was loaded through volatile JTAG, with AMC READY and QDMA queue 4
online. Physical numerical qualification passes: all eight 4 GiB channels pass
full March C-, 105 ISA checks and five system checks pass with 20 soak iterations,
and HMM/HDQ each recover exactly after four interrupted runs. Reset tests finish
operand setup before starting the long-running operator, so the interruption
does not merely cancel setup. Legacy INT8 matches all 128 steps across eight
prompts. All five model policies match intermediate outputs, persistent states
and final logits for both retained tokens, with the final hashes independently
checked against the preceding branch's references.

Evidence is retained on qfpga0 at
`full-v1/board-qualification/20261115_101346/` under the performance build root,
including `qualification_review.json`; JSON copies are in
`build/perf_v80/full-v1/board-qualification/`. The performance gate below fails;
the candidate has **not** received complete board acceptance or a
`deploy/QUALIFIED.json`. Flash/FPT have not been written.

## Physical performance and the open W8A8 regression

At the same 100/200 MHz clocks, the combined candidate produces exact outputs
for all 18 projection runs and all 20 model profiling steps. The warm second
model step, with the same tokens, cap and policies as the baseline above:

| Policy | Baseline device seconds | Candidate device seconds | Device cycle reduction | Candidate host wall seconds |
|---|---:|---:|---:|---:|
| W16A16 | 13.02159 | 9.44715 | 27.45% | 9.83678 |
| W8A8 | 0.10433 | 0.10926 | **-4.72%** | 0.28498 |
| W4A16 | 18.35822 | 14.83279 | 19.20% | 15.37066 |
| mixed | 14.15015 | 10.79681 | 23.70% | 11.19586 |
| gateup4 | 8.74814 | 6.71254 | 23.27% | 6.97794 |

These cap-8 measurements establish neither long-context throughput nor model
quality. Half-path improvements are substantial, but the W8A8 regression
exceeds the 2% acceptance limit. The performance review exits with failure;
the limit is unchanged and final acceptance remains open. Evidence is under
`/home/pqq/workspace/openTPU-perf-build/full-v1/performance/`.

A physical new/old/new comparison reproduces the W8A8 difference using exactly
the same runtime `8c8f6450`, image/program caches, tokens and compiler settings.
Each load runs one initial token pair and eight reset/replay pairs. All 54
outputs match the preceding branch's logits bit for bit. Warm second-token
medians are 10,924,173.5 cycles on the first new load, 10,430,709.5 on the old
image, and 10,925,740 on the second new load. This rules out the runtime revision
and a single noisy baseline sample as explanations for the measured difference.

Hardware input hashes differ only for the two half-path RTL files; exported
board parameters differ only in build ID. The NoC traffic specification and
logical instance description are identical, but the physical NoC solution
changes NMU placement and routes. For example, the S08 input moves from
`NOC_NMU512_X1Y5` to `NOC_NMU512_X2Y6`, and its reported NoC path latency changes
from 111 to 115. These report values are not core-clock cycles. This is a
candidate explanation requiring a controlled implementation comparison, not
an established root cause. Evidence: `full-v1/ab/repeat-summary.json`,
`platform-comparison.json` and `noc-path-review.json` on qfpga0.

The existing `tools/board_perf.py` now opens QDMA through the shared transport
factory and records the physical build ID. Its ISA/board/QDMA regression passes
27 tests. Three physical repetitions on each image pass every data comparison:

| Memory-path measurement | Mixed baseline median cycles | Combined candidate median cycles |
|---|---:|---:|
| INT8 weight stream, 8 MiB | 50,408 | 51,322 |
| INT8 weight stream, 32 MiB | 197,384 | 203,185 |
| INT8 weight stream, 128 MiB | 789,967 | 811,365 |
| Pointer chase, 1,040 hops | 84,689 | 84,786 |
| Port B loads, 64 MiB | 2,212,941 | 2,212,951 |
| Port B stores, 64 MiB | 2,218,669 | 2,218,334 |

The weight stream regresses by 2.71% at 128 MiB, while ordinary load/store
throughput is effectively unchanged. This narrows the investigation; it does
not establish which physical change causes the full-model regression. Records
and hardware counters are in `full-v1/ab/{new-1,old-2}/memory-*.json` and
`full-v1/ab/memory-summary.json`. The card has been restored to qualified build
`0c31d07d`; AMC READY, QDMA queue 4, the established Gen4 x8 link and unchanged
FPT were checked after restoration.

A separate `exp/v80-noc` experiment at `4ebbec16` pins the eight PL NMUs and the
control NSU to their locations in the qualified baseline. It uses the unchanged
combined RTL, FIFO depth 2048 and 100/200 MHz clocks. This follows AMD's
[NoC endpoint placement constraints](https://docs.amd.com/r/en-US/pg313-network-on-chip/Constraining-NMU/NSU-Locations).
The configuration/build tests pass 102 cases. Full implementation is running at
`openTPU-perf-build/full-noc`; its package check will verify the actual endpoint
locations in the exported NoC solution. Fixing endpoints does not guarantee
identical routes or improved performance; those still need board measurements.

The routed timing report identifies two useful targets for subsequent resource
and clock work. The worst setup path is `rst_mem_reg` in the 100 MHz domain to a
TMEM replay register's reset in the related 200 MHz domain: 5 ns requirement,
2.268 ns data delay and -2.166 ns clock skew, leaving +0.224 ns. The integer MXU's
`pop_q[KBm1]` to posted FIFO read-address path has 9.446 ns data delay, 93.6% in
routing, with +0.369 ns slack. These paths motivate measuring queue size and
local control distribution before increasing frequency. They remain timed;
removing their timing checks would not demonstrate a faster design.

## Integer MXU resource experiments

`rtl/exp/v80_mxu/ooc.tcl` runs synthesis out of context with the current V80
integer MXU parameters, including the board's posted FIFO quarter writes
(`XPOST=1,QB=2`). Only FIFO depth and arithmetic cell family vary:

```bash
vivado -mode batch -source rtl/exp/v80_mxu/ooc.tcl \
  -tclargs build/perf_v80/mxu-512 512 xilinxversal
```

The study compares `xilinxversal` at depths 2048, 1024 and 512, then `generic`
at the same depth 512. `inputs.txt` records the complete generic list, source
revision and hashes of every source and the Tcl script. Each case retains its
utilization reports and synthesis checkpoint. These are component synthesis
counts with exposed ports, not routed overlay resources or timing results.
No default target setting changes as part of this study.

Completed synthesis on the same V80 part and Vivado 2026.1.1:

| Family / FIFO depth | LUT | FF | DSP | RAMB36 | RAMB18 | BRAM36 equivalents |
|---|---:|---:|---:|---:|---:|---:|
| xilinxversal / 2048 | 56,626 | 45,409 | 1,194 | 548 | 5 | 550.5 |
| xilinxversal / 1024 | 56,376 | 45,319 | 1,194 | 274 | 5 | 276.5 |
| xilinxversal / 512 | 57,498 | 45,353 | 1,194 | 119 | 41 | 139.5 |
| generic / 512 | 157,914 | 123,809 | 1,178 | 119 | 41 | 139.5 |

FIFO depth 512 saves 411 BRAM36 equivalents (74.66%) in this component, at a
cost of 872 LUT and with unchanged DSP use. At depth 512, generic saves only
16 DSP while adding 100,416 LUT and 78,456 FF. The next full-build experiment
therefore retains xilinxversal and reduces FIFO depth alone. Neither result
demonstrates sustained throughput or full-design timing.

The isolated `exp/v80-fifo512` source `a16301b8` changes only the target's FIFO
depth relative to `51db044d`. It passes 87 configuration/build tests, the native
integer-MM check across all four integer weight formats, and three exact
mixed-model steps including reset/replay under AXI stalls. Source hashes match
the resource study and test inputs. Its full implementation is running at
100/200 MHz under `openTPU-perf-build/full-fifo512` on qfpga0. This is an
experiment: the preceding candidate's failed W8 gate remains recorded, and
any replacement must independently pass against the qualified mixed baseline.
The default `perf/v80` target remains at FIFO depth 2048 until those checks pass.

The existing resource parser now recognizes Versal's `DSP Slices`, `RAMB36E5`
and `RAMB18E5*` summary rows. Previously those names silently yielded zero.
The half controller is also registered as its own `otpu_hmxu` component with
FP16/INT4 and native mixed-model gates. The component/report toolchain's 71
tests pass; the legacy INT8 performance proxy is not used to score the half
controller. Evidence: `build/perf_v80/resource-parser-tests-fixed.log`.
