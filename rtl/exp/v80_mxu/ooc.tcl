# EXPERIMENTAL: integer MXU resource comparison at V80's current parameters.
#   vivado -mode batch -source rtl/exp/v80_mxu/ooc.tcl -tclargs OUT DEPTH TECH
# DEPTH: 256, 512, 1024 or 2048; TECH: xilinxversal or generic. Only these two
# choices vary. This is synthesis out of context, with all top-level ports kept:
# no board timing, clock-rate, bandwidth or functional qualification is implied.
# The board connects the posted FIFO's p_clk separately, with QB=2 in otpu_v80_sys.
if {$argc != 3} {error "expected: OUT DEPTH TECH"}
lassign $argv out depth tech
if {$depth ni {256 512 1024 2048}} {error "unsupported DEPTH: $depth"}
if {$tech ni {xilinxversal generic}} {error "unsupported TECH: $tech"}
set out [file normalize $out]
set root [file normalize [file join [file dirname [info script]] ../../..]]
set part xcv80-lsva4737-2MHP-e-S
file mkdir $out
set_param general.maxThreads 4
set sources [list $root/rtl/vpu/otpu_fp.sv $root/rtl/vpu/otpu_fpipe.sv \
  $root/rtl/top/otpu_pkg.sv $root/rtl/mxu/otpu_mxu.sv \
  $root/rtl/tech/generic/otpu_tech_ram.sv $root/rtl/tech/$tech/otpu_tech_dsp.sv]
set gen [list D=128 MCOLS=8 ROWS=16 DEPTH=$depth LANES=8 TLANES=8 IMPL=2 CL=16 E=8 \
  WG=8 AB=128 XPOST=1 QB=2 CHASH=1 TQ=1 RF_STYLE="distributed" FIFO_STYLE="block" \
  RF_ROWS=0 BRAW=1 IQ=0 QD=4 WASK=1 AASK=1 KT=1 SPL=0 PUMP=0 PHOLD=1 SID=0 DAW=36]
set f [open $out/inputs.txt w]
puts $f "part=$part depth=$depth tech=$tech"
puts $f "revision=[exec git -C $root rev-parse HEAD]"
puts $f "generics=$gen"
puts $f [exec sha256sum {*}$sources [info script]]
close $f
create_project -in_memory -part $part
set_property source_mgmt_mode None [current_project]
read_verilog -sv $sources
set ga {}
foreach g $gen {lappend ga -generic $g}
synth_design -top otpu_mxu -part $part -mode out_of_context -flatten_hierarchy rebuilt \
  -verilog_define SYNTHESIS {*}$ga
report_utilization -file $out/util.txt
report_utilization -hierarchical -hierarchical_depth 5 -file $out/util_hier.txt
write_checkpoint -force $out/synth.dcp
set f [open $out/result.txt w]
puts $f "synthesis_complete part=$part depth=$depth tech=$tech"
puts $f "scope=isolated integer MXU resources; no timing or throughput qualification"
close $f
puts "OTPU_V80_MXU_SYNTH_DONE depth=$depth tech=$tech"
close_project
