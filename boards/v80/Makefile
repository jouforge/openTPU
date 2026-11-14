# AMD Alveo V80: simulation and integrated AMR/core flow (docs/v80.md).
# PYTHON selects the repository environment. See README.md for the qualified configuration.
ROOT := $(abspath ../..)
PYTHON ?= python3

.PHONY: help config test test-model test-rtl test-platform test-bridge test-system test-tech prepare preflight project generate elaborate synth impl pdi
help:
	@echo 'make config       show the V80 target'
	@echo 'make test         target/configuration regression (all targets)'
	@echo 'make test-model   V80 ISA kernel and tiny-model matrix (torch, transformers)'
	@echo 'make test-rtl     native-memory and Qwen3.5 RTL/ISA checks (Verilator 5)'
	@echo 'make test-platform  host transport and platform-flow tests'
	@echo 'make test-bridge  native/AXI4 bridge and HBM startup-probe tests'
	@echo 'make test-system  full-target CSR, memory readiness and instruction-load tests'
	@echo 'make test-tech    DSP58/generic equivalence in Vivado XSim (bin on PATH)'
	@echo 'make prepare      generate local Tcl and record actual source/tool versions'
	@echo 'make preflight    source inputs, Vivado part/board/IP catalog checks'
	@echo 'make project      AMR platform, core sources, CSR/HBM ports and related clocks'
	@echo 'make generate     generate platform IP and wrapper for otpu_fpga_top_v80'
	@echo 'make elaborate    elaborate the full physical top in Vivado'
	@echo 'make synth        synthesize and report utilization, clocks, timing and CDC'
	@echo 'make impl         place/route; require timing, routing, DRC and HBM pin checks'
	@echo 'make pdi          accepted implementation to boot/PL images and XSA'
	@echo 'PYTHON=/path/to/python selects the test environment'

config:
	cd $(ROOT) && $(PYTHON) -m opentpu.target show v80

test:
	cd $(ROOT) && $(PYTHON) -m pytest -q tests/test_target.py

test-model:
	cd $(ROOT) && $(PYTHON) -m pytest -q tests/test_config_matrix.py -k v80

test-rtl:
	cd $(ROOT) && $(PYTHON) -m pytest -q tests/test_v80.py

test-platform:
	cd $(ROOT) && $(PYTHON) -m pytest -q tests/test_qdma.py tests/test_v80_build.py

test-bridge:
	cd $(ROOT) && $(PYTHON) -m pytest -q tests/test_v80_mem.py

test-system:
	cd $(ROOT) && $(PYTHON) -m pytest -q tests/test_v80_sys.py

test-tech:
	cd $(ROOT) && $(PYTHON) tools/tech_dsp.py --family xilinxversal

prepare preflight project generate elaborate synth impl pdi:
	PYTHON=$(PYTHON) bash ./run_vivado.sh $@
