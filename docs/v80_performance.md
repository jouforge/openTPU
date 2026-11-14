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
full V80 implementation are separate gates; no new physical-board speedup is
claimed here.

Evidence: `build/perf_v80/pipeline-rtl.json`, `pipeline/hdot-overlap.log` and
`pipeline-ooc/`. RTL/runtime source hashes in the isolated benchmark match the
main checkout. The matching OOC sources/reports/checkpoints are retained on
qfpga0 under `pipeline-src/` and `pipeline-ooc/` in the performance build root.
