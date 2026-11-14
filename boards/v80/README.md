# openTPU on the AMD Alveo V80

The V80 port on `port/v80` has passed real-card qualification with the AMR platform,
QDMA transport and `xilinxversal` DSP58 cells. The qualified image meets routed timing
and CDC checks at 100 MHz core / 200 MHz pumped-memory clock. Matching segmented PDI/XSA
and AMC firmware were loaded through JTAG. All 32 GiB of HBM, 100 ISA checks, 20 repetitions
of the kernel set, and Qwen3.5-0.8B INT8 passed: eight prompts and 128 generated steps have
bit-exact logits against the ISA reference. See [the dated qualification](../../docs/v80.md#qualified-v80-image--2026-11-14)
for artifact hashes, reports and remaining limits. Deployment is currently volatile JTAG.

The board identifier is `v80`, following `bcu1525` and `ypcb-00338`. Use this name for
`boards/v80/`, `rtl/boards/v80/`, `targets/v80.toml` and `OTPU_TARGET=v80`.
Use **AMD Alveo V80** in prose. Model checkpoints, AMR releases and build hosts do not change
this identifier. See [the port plan](../../docs/v80.md), [target configuration](../../docs/targets.md)
and [the existing Qwen3.5 implementation](../../docs/qwen35.md).

## Target and tests

[`targets/v80.toml`](../../targets/v80.toml) defines one slice, D 128, eight MXU columns,
WG 8, eight logical 128-byte memory channels, AW 36 and platform ID 8. Its 32 GiB HBM map
passed the final full-capacity hardware test. The `xilinxversal` DSP58 cells preserve the generic
arithmetic and pipeline contract. `opentpu.target.sources("v80")` includes the board RTL;
Verilator simulations explicitly select generic cells, as for other FPGA targets.

Run from the repository root with Python 3.11+ and the project's dependencies plus pytest.
Model checks also require torch and transformers with Qwen3.5 support. RTL checks need
recent Verilator 5, make and a C++20 compiler. System Verilator 5.020 cannot parse the current
MXU RTL; see [the validated environment](../../docs/v80.md#simulation-baseline-validation).
Tiny-model checks use random weights and do not download checkpoints.

```sh
make -C boards/v80 config
make -C boards/v80 test
make -C boards/v80 test-platform
make -C boards/v80 test-model
make -C boards/v80 test-rtl
make -C boards/v80 test-bridge
make -C boards/v80 test-system
make -C boards/v80 test-tech
```

Set `PYTHON=/absolute/path/to/python` on make to select an environment; put its Verilator
on `PATH`. `test` checks all targets, `test-platform` checks the QDMA transport and build
input handling, `test-model` runs the V80 ISA kernel/model matrix, and `test-rtl` compares
native-memory matrix multiplication and tiny Qwen3.5 prefill/decode against the ISA simulator.
RTL tests retain target arithmetic and channel geometry with reduced simulated DRAM; they
do not exercise physical HBM, QDMA or Versal primitives. `test-bridge` covers the actual
V80 AXI4 wrapper and read-only startup probes; `test-system` uses the full target parameters
to check CSR, memory-ready gating, reset/error recovery and an AXI-loaded HALT instruction.
`test-tech` requires Vivado's `bin/` on `PATH` and compares the official DSP58 model to
generic in XSim, including all PE and pumped-PE variants. Results go to `build/tech_dsp/`.

## Platform flow

Start with the versions in the [integration baseline](../../docs/v80.md#platform-baseline).
AMR and `amr_vivado_designs` are external, tracked-clean checkouts. Generated files already present
in them are allowed; the flow writes its own outputs outside those trees. It does not fetch,
patch or build inside either dependency. Preparation creates `amr.pins.xdc` in `OUT_DIR`,
preserving the vendor pin file except for four reversed HBM reference-clock assignments.
Those four pins are corrected in the local XDC using PG313 and the device package database;
the generated copy's hash is recorded in the manifest.

`scripts/prepare.py` generates local Tcl from the selected target and paths below; there
is no dependency lock file. It records the actual external Git commits, Vivado version/build
and executable, device/board parts and input hashes in `OUT_DIR/manifest.json`. Other clean
revisions can be selected for a fresh build, but must satisfy the platform's interface and
pin checks. Recording a version does not establish compatibility or board qualification.

```sh
export AMR_ROOT=/path/to/amr
export AMR_DESIGN_ROOT=/path/to/amr_vivado_designs
export VIVADO=/path/to/Vivado/bin/vivado
export OUT_DIR=/path/to/openTPU/build/vivado_v80/platform
make -C boards/v80 prepare PYTHON=/path/to/python
make -C boards/v80 preflight PYTHON=/path/to/python
make -C boards/v80 project PYTHON=/path/to/python
make -C boards/v80 generate PYTHON=/path/to/python
make -C boards/v80 elaborate PYTHON=/path/to/python
make -C boards/v80 synth PYTHON=/path/to/python
make -C boards/v80 impl PYTHON=/path/to/python
make -C boards/v80 pdi PYTHON=/path/to/python
```

Python needs NumPy to load the existing target module. The runner also needs Bash and
`flock`. `OUT_DIR` defaults to `build/vivado_v80/platform`; select a fresh directory for a
new project or changed inputs. The manifest records the target, CFG_HASH, actual revisions,
all core/board RTL and local build-source hashes, external platform-source hashes, resolved
paths and the exact target-derived `OTPU_BOARD` macro. A different
manifest is rejected when a project already exists, including changes to the tool version.
Generated `config.tcl`, `sources.tcl` and `target_params.tcl` stay in the output directory;
change the target or environment and regenerate them instead of editing generated files.
Build commands do not access the card.

- `prepare`: record inputs and generate local Tcl; query `vivado -version` without creating
  a Vivado project or starting synthesis.
- `preflight`: verify pins, the recorded tool version, device/board parts and required IP catalog entries.
- `project`: source the unmodified AMD `v80_base` design, add the openTPU CSR/HBM boundary,
  validate the block design and NoC, and save `project/otpu_v80.xpr`.
- `generate`: validate the saved design and generate IP output products and its HDL wrapper.
  `otpu_fpga_top_v80` connects it to `otpu_v80_sys` and the shared `otpu_board` core.
- `elaborate`: generate products, synthesize the platform IP out of context, then elaborate
  the core/physical top with IP stubs (`synth_design -rtl -rtl_skip_ip`).
- `synth`: generate products, launch/wait for synthesis, and save a checkpoint plus
  utilization, clock, timing and CDC reports. Synthesis reports do not establish routed timing.
- `impl`: synthesize as needed, optimize/place/physically optimize/route, and save reports
  and a checkpoint. Require all eight bridges and pumped TMEM read paths to have their
  upstream timing constraints before optimization; check structural DRC before placement.
  Check posted-beat count and status synchronizers outside the bridges as well.
  Reject incomplete/error routes, negative setup/hold/pulse-width or bus-skew slack,
  DRC errors/critical warnings or HBM clock pins that differ from the board pinout.
- `pdi`: run the same implementation checks before generating the segmented boot/PL
  device images and exporting/validating `otpu_v80.xsa`, including the implemented NoC
  solution. A failed implementation stops export. Image export does not establish board
  or model qualification.

The earlier generic synthesis snapshot had positive estimated setup slack but failing hold
checks. The flow now replaces AMR's stale hierarchy/floorplan XDC with local constraints,
while retaining the vendor's other physical pins. Its HBM clock-pin assignments are corrected
and checked after routing. The shared `otpu_hbm_ch.tcl` and `otpu_clk2x.tcl` are reused from
the BCU1525/YPCB flows and included in the input manifest. `otpu_v80_top.tcl` applies
the BCU1525 posted-count crossing contract to the V80 hierarchy and bounds its status inputs.
See [the validation record](../../docs/v80.md#versal-cells-and-implementation).

The overlay retains CIPS, AMR management, DDR, HBM, clocks and PF1's existing GPIO. Its
initial 100 MHz boundary has these ports:

| Interface | Contract |
|---|---|
| `M_AXI_CTRL` | AXI4-Lite, 32-bit data, 64-bit address; PF1 user BAR2 + `0x100000`, size 1 MiB |
| `S_AXI_HBM0` ... `S_AXI_HBM7` | AXI4, 256-bit data, 64-bit address; channel c reaches four 1 GiB HBM pseudo-channels at `0x4000000000 + c * 0x100000000` |
| `axi_clk`, `axi_resetn` | CIPS PL0 at 100 MHz and its synchronized active-low reset |
| `core_clk`, `core_clk2x`, `core_resetn` | One MMCM's phase-aligned 100/200 MHz clocks and lock-qualified reset |

AMR's PF1 BAR2 translation is `0x20200000000`; the CSR's NoC address is therefore
`0x20200100000`. AMR GPIO at BAR2 + `0x20000` is preserved. PF1 SmartConnect crosses CSR
traffic into the new core clock domain. The memory bridges use the existing asynchronous
FIFO/reset-drain design; the core's 1x/2x clocks remain related for `TMEM_PMP=1`.

`otpu_v80_mem` reuses `otpu_hbm_ch` with 25-bit native beat addresses, widens its channel
address to 64 bits and adds the HBM base once. Its single-ID bursts are AXI4-compatible:
8-bit AxLEN, 32-byte transfers, INCR, at most 16 transfers, no 4 KiB crossing. Partial byte
strobes, write completion after B, ordered reads, posted reads and reset draining retain
the upstream implementation. Unexpected nonzero response IDs latch `n_err` until AXI reset.

The baseline AMR boundary has no PL calibration output. `otpu_v80_ready` reserves each AR/R
port at startup and reads one 32-byte word from each of its four HBM pseudo-channels.
Only four OKAY, ID-zero, single-beat responses enable that channel. STATUS CALIB0/CALIB1
mean all even/odd channels passed these probes; this is a route-readiness check, not PHY
telemetry or a capacity/bandwidth qualification. No response leaves readiness low; a bad
response latches the AXI error. A core reset cannot clear it; a coordinated PMC PL/NoC reset retries
all probes. NoC has no user-routable AXI reset: the physical top preserves AXI state across
later PL-only resets, using the bridge's core-side drain rather than discarding responses.
Probes do not write HBM. V80 sets HAS_TEMP=0 because no compatible temperature source is
wired to openTPU; use AMR management telemetry for board temperature.

Device-image export alone does not qualify board reset/reload behavior, HBM contents or
QDMA. Build matching AMC firmware with the selected AMR release's script. It needs a Git
checkout and performs source-tree cleanup, so use an isolated clone and a separate working
directory. With absolute paths for `OUT_DIR`, `AMR_ROOT`, `AMR_FW_ROOT` and `VITIS_ROOT`:

```sh
git clone --no-hardlinks "$AMR_ROOT" "$AMR_FW_ROOT"
git -C "$AMR_FW_ROOT" checkout --detach "$(git -C "$AMR_ROOT" rev-parse HEAD)"
source "$VITIS_ROOT/settings64.sh"
mkdir -p "$OUT_DIR/amc"
(
  cd "$OUT_DIR/amc"
  "$AMR_FW_ROOT/fw/AMC/scripts/build.sh" -xsa "$OUT_DIR/otpu_v80.xsa" -profile v80
)
```

For the recorded AMR baseline this produces `amc.elf`, the boot PDI with AMC firmware
(`build_amc/amr_ospi.bin`), and a separate FPT setup image (`amr_ospi_fpt.bin`). Pair the AMC
boot image with `otpu_v80_pld.pdi` from the same XSA. The FPT setup image changes the flash
layout and is not an ordinary application image. Do not concatenate boot and PL segments
for this Versal HBM SSI device; use the supported segmented programming flow described in
AMD's [segmented configuration tutorial](https://github.com/Xilinx/Vivado-Design-Tutorials/tree/2026.1/Versal/Boot_and_Config/Segmented_Configuration).

The first bring-up package uses volatile JTAG loading after quiescing/removing the card's
PCI functions, followed by PCI rescan and QDMA provisioning. The current PL PDI is
17,565,344 bytes, exceeding the inspected card's 16 MiB user flash partition. Persistent
deployment therefore needs a separate layout/packaging review. No general programming
target is provided yet; host-specific administration scripts stay in the local build output.

The tested Vivado 2026.1.1 sequence assigns each segment to `PROGRAM.FILE`. After opening
the intended JTAG cable and selecting its `xcv80` device as `dev`, with both PCI functions
removed and no DMA client running:

```tcl
set_property PROGRAM.FILE $amc_boot_pdi $dev
program_hw_devices $dev
set_property PROGRAM.FILE $pl_pdi $dev
program_hw_devices -skip_reset $dev
```

Check both programming results and DONE, then rescan the card's parent PCI bridge, wait
for AMC READY and provision the bidirectional QDMA MM queue. `-skip_reset` preserves the
boot configuration while loading PL. `PROGRAM.BOOT_FILE` / `PROGRAM.PLD_FILE` alone did
not select a programming file for this command in the tested version. Keep the matching
boot/PL hashes, programming log and post-load diagnostics with the local build record.

## Host transport

[`QdmaTransport`](../../opentpu/host/qdma.py) is selected explicitly with
`qdma:0000:41:00.1@4`: a full PCI BDF for PF1 and an already started bidirectional MM queue.
The BDF/queue are examples, not defaults. Only PCI domain `0000` is currently supported,
matching the driver's domain-less character-device names. PF1 must be `10ee:50b5`, bound
to `qdma-pf`. The process needs access to the selected queue node and PF1 `resource2`.
Queue provisioning and device permissions remain platform administration tasks.

After an openTPU device image has been built and loaded, existing tools can select it:

```sh
OTPU_TARGET=v80 otpu-smi --dev qdma:0000:41:00.1@4 --json
OTPU_TARGET=v80 otpu-diag --dev qdma:0000:41:00.1@4
```

These commands require the openTPU overlay: the stock AMR image has no openTPU CSRs.
The transport performs native 32-bit MMIO and synchronous QDMA `pread`/`pwrite`, preserving
64-bit HBM addresses, short I/O and partial-beat writes. All queues of one PF share the
existing openTPU exclusive device lock. `otpu-smi` opens only the register BAR and permits
SNAP writes to latch counters; it does not open the DMA queue. Host/core streaming is disabled pending concurrency qualification; automatic
QDMA device discovery, queue creation and driver setup are not implemented.

## Run Qwen3.5 with caches

With the overlay loaded, its queue provisioned, and the Python environment activated,
run from the repository root. Set `MODEL` to a local Qwen3.5-0.8B checkpoint and
`OTPU_DEV` to the provisioned PF1/queue. Use a writable cache directory on a disk with
enough free space; the existing caches enforce their own free-space floors.

```sh
export OTPU_TARGET=v80
export OTPU_DEV=qdma:0000:41:00.1@4
export MODEL=/absolute/path/to/Qwen3.5-0.8B
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
export OTPU_IMAGE_CACHE="$PWD/build/vivado_v80/cache/mxu"
export OTPU_IMG_CACHE="$PWD/build/vivado_v80/cache/img"
export OTPU_PROG_CACHE="$PWD/build/vivado_v80/cache/prog"
mkdir -p "$OTPU_IMAGE_CACHE" "$OTPU_IMG_CACHE" "$OTPU_PROG_CACHE"

otpu-smi --dev "$OTPU_DEV"
otpu-chat --model "$MODEL" --backend board --dev "$OTPU_DEV" \
  --wformat int8 --max-new 256 --plain
```

`OTPU_IMAGE_CACHE` caches quantized matrices where supported, `OTPU_IMG_CACHE` caches
the complete model memory image, and `OTPU_PROG_CACHE` caches device programs, prefill
fit results and instruction wait tables. Explicit paths also work on a host without
the default `~/otpu-build` directory. Keep the exports in the shell used to launch
each process. The caches are local runtime outputs, separate from build input records.

The first run builds missing entries. Later loads should report `read from the image
cache`; programs are reused when their keys match. Changed model layouts, source,
formats or newly reached context buckets can require more preparation. Cache reuse
does not eliminate loading the model onto the card.

To compare runs, ask a short question, enter `/reset`, and ask the same question again.
Then exit with an empty line or Ctrl-D, restart the same command, and repeat. Record
loading time, TTFT and both wall/device token rates; a cold reply's average decode rate
includes any program preparation after its first token. For an automated single prompt,
append `--max-new 64 --seed 42 --prompt "请用中文介绍一下你自己。"`.
The [measured cold/warm comparison](../../docs/v80.md#chat-performance-and-caches--2026-11-14)
records this workload's results and timing scope. Stop the chat before running diagnostics
that execute programs or overwrite model memory.

## Source layout and outputs

Implemented board sources:

```text
boards/v80/
  README.md, Makefile
  run_vivado.sh                # batch entry point; one process per output directory
  scripts/prepare.py           # dependency checks, manifest and target-derived Tcl
  scripts/preflight.tcl        # tool, part, board and IP checks
  vivado/run.tcl               # project/generate/elaborate/synth/impl dispatch and failure exit status
  vivado/create_project.tcl    # external AMR design plus local overlay
  vivado/bd.tcl                # CSR and eight HBM boundary ports
  vivado/check_bd.tcl          # assert address map and actual MMCM frequencies/phases
  vivado/clocks.tcl            # related core clocks, reset and CSR clock crossing
  vivado/report_cells.tcl      # inferred BRAM attributes and DSP allocation
  vivado/check_timing.tcl      # require bridge, posted-count and pumped-memory constraints
  vivado/check_opt.tcl         # structural DRC acceptance before placement
  vivado/check_impl.tcl        # routed timing/bus skew, DRC and HBM pin acceptance
  vivado/export.tcl            # boot/PL device images and implemented hardware handoff
  constraints/otpu_v80.xdc     # corrected HBM pins and asynchronous SMBus exceptions
  constraints/otpu_v80_top.tcl # posted-count and status crossings outside the bridges
```

Add bring-up scripts and lint stand-ins with
their implementations. Core/bridge RTL belongs in `rtl/boards/v80/`, reusable technology
cells in `rtl/tech/`, host code in `opentpu/host/`, and tests in `tests/` and `sim/`.

Generated projects, IP products, wrappers, logs, reports and manifests belong under `build/`
or the chosen `OUT_DIR`; do not commit them as board source. V80's deployable
artifacts are matching Versal boot/PL segments, built and packaged with AMC according to
the selected AMR design.
