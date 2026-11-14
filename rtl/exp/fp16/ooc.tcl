# FP16 dot-product baseline, isolated synthesis and routing (docs/fp16.md).
# vivado -mode batch -source rtl/exp/fp16/ooc.tcl -tclargs OUT PART [N] [PERIOD_NS] [dot|mxu] [W4A16]
# This checks one component, not the complete V80 timing or its memory integration.
if {$argc < 2 || $argc > 6} { error "usage: OUT PART ?N? ?PERIOD_NS? ?dot|mxu? ?W4A16?" }
lassign $argv out part n period unit w4a16
set out [file normalize $out]
set root [file normalize [file join [file dirname [info script]] ../../..]]
if {$n eq ""} { set n 8 }
if {$period eq ""} { set period 10.0 }
if {$unit eq ""} { set unit dot }
if {$unit ni {dot mxu}} { error "unit must be dot or mxu" }
if {$w4a16 eq ""} { set w4a16 0 }
if {$w4a16 ni {0 1} || ($w4a16 && $unit ne "mxu")} { error "W4A16 requires mxu and 0 or 1" }
set generics [list N=$n]
if {$unit eq "mxu"} { lappend generics W4A16=$w4a16 }
set top otpu_h${unit}_ooc
if {![string is integer -strict $n] || $n < 1} { error "N must be positive" }
if {![string is double -strict $period] || $period <= 0} { error "period must be positive" }
file mkdir $out
create_project -in_memory -part $part
read_verilog -sv [list $root/rtl/top/otpu_pkg.sv $root/rtl/vpu/otpu_fp.sv \
                      $root/rtl/vpu/otpu_fpipe.sv $root/rtl/mxu/otpu_hdot.sv \
                      $root/rtl/mxu/otpu_hmxu.sv $root/rtl/exp/fp16/$top.sv]
# The wrapper supplies same-clock launch/capture registers. Bare OOC ports have
# no launch clock tree and cannot model an internal block's hold requirement.
set xdc [open $out/boundary.xdc w]
puts $xdc "create_clock -period $period -name clk \[get_ports clk\]"
puts $xdc {set_false_path -from [get_ports -filter {DIRECTION == IN && NAME != clk}]}
puts $xdc {set_false_path -to [all_outputs]}
close $xdc
read_xdc $out/boundary.xdc
synth_design -top $top -part $part -mode out_of_context -flatten_hierarchy rebuilt \
             -verilog_define SYNTHESIS -generic $generics
report_utilization -hierarchical -file $out/synth_utilization.rpt
write_checkpoint -force $out/synth.dcp
opt_design
place_design
phys_opt_design
route_design
report_timing_summary -report_unconstrained -file $out/timing.rpt
report_timing -delay_type max -max_paths 10 -file $out/setup.rpt
report_timing -delay_type min -max_paths 10 -file $out/hold.rpt
report_utilization -hierarchical -file $out/utilization.rpt
report_drc -file $out/drc.rpt
report_route_status -file $out/route.rpt
write_checkpoint -force $out/routed.dcp
set setup [get_timing_paths -delay_type max -max_paths 1]
set hold [get_timing_paths -delay_type min -max_paths 1]
if {![llength $setup] || ![llength $hold]} { error "missing timing paths" }
set wns [get_property SLACK $setup]
set whs [get_property SLACK $hold]
set f [open $out/result.txt w]
puts $f "part=$part unit=$unit W4A16=$w4a16 N=$n period_ns=$period WNS=$wns WHS=$whs"
puts $f "scope=isolated FP16 $unit component; not a qualified V80 image"
close $f
puts "OTPU_FP16_OOC UNIT=$unit N=$n PERIOD=$period WNS=$wns WHS=$whs"
if {$wns < 0 || $whs < 0} { error "isolated FP16 timing failed" }
if {[llength [get_drc_violations -filter {SEVERITY == Error}]]} {
  error "isolated FP16 DRC failed"
}
