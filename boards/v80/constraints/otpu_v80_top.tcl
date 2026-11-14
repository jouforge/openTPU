# Crossings outside otpu_hbm_cs, following bcu1525_wg_top.tcl's posted-count
# contract. Unmanaged Tcl, implementation only, after the IP clocks exist.
# Do not group the clocks asynchronously: that would override the datapath bounds.
namespace eval ::otpu {}
proc ::otpu::v80_fail {message} {
  puts "CRITICAL WARNING: \[otpu_v80_top.tcl\] $message"
  lappend ::otpu::fail "otpu_v80_top.tcl: $message"
}
set ca [get_clocks -quiet -of_objects [get_pins -quiet {u_sys/g_ch[0].u_mem/u_ch/u_cs/p_g_reg[0]/C}]]
set cc [get_clocks -quiet -of_objects [get_pins -quiet {u_sys/u_board/cal_s1_reg[0]/C}]]
if {[llength $ca] != 1 || [llength $cc] != 1} {
  ::otpu::v80_fail "expected one AXI and one core clock: '$ca', '$cc'"
  return
}
set ta [get_property -quiet -min PERIOD $ca]
set tc [get_property -quiet -min PERIOD $cc]

# Each registered gray count announces beats already written into the posted RAMs.
# Only its first ASYNC_REG stage is a crossing; the second stage and decoding stay
# normally timed. Bound delay by one source period, skew by the smaller period.
set npc 0
for {set j 0} {$j < 8} {incr j} {
  set g [get_cells -quiet "u_sys/g_ch?$j?.u_mem/u_ch/u_cs/p_g_reg\[*\]"]
  set d [get_cells -quiet "u_sys/u_board/u_mem/g_pc.pg1_reg?$j?\[*\]"]
  if {![llength $g] || [llength $g] != [llength $d]} {
    ::otpu::v80_fail "channel $j: [llength $g] gray registers, [llength $d] synchronizer registers"
    continue
  }
  set_max_delay -datapath_only -from $g -to $d $ta
  set_bus_skew -from $g -to $d [expr {min($ta, $tc)}]
  incr npc
}

# Startup calibration and sticky probe/AXI-ID errors are independent levels on
# aclk, sampled by two-flop ASYNC_REG synchronizers on clk. Bound each first stage
# by one source period; keep the synchronizer's following stages normally timed.
set ns 0
foreach pattern {{u_sys/u_board/cal_s1_reg[*]} {u_sys/err_s1_reg[*]} {u_sys/g_ch*.u_mem/id_s1_reg}} {
  set cells [get_cells -quiet $pattern]
  if {[llength $cells] != 8} {
    ::otpu::v80_fail "$pattern: expected eight status synchronizers, got [llength $cells]"
    continue
  }
  set_max_delay -datapath_only -from $ca -to $cells $ta
  incr ns [llength $cells]
}
if {$npc == 8 && $ns == 24} {set ::otpu::ok(otpu_v80_top) 1}
puts "otpu_v80_top.tcl: posted_counts=$npc status_inputs=$ns"
