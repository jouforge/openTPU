# Check the elaborated address map, not just the requested configuration properties.
# NoC validation may succeed with an unconnected/unmapped input, so check every channel.
set fp [open $otpu(out)/reports/boundary.txt w]
for {set c 0} {$c < $otpu(nch)} {incr c} {
  set port [get_bd_intf_ports S_AXI_HBM$c]
  foreach {prop expected} {PROTOCOL AXI4 ADDR_WIDTH 64 DATA_WIDTH 256} {
    if {[get_property CONFIG.$prop $port] ne $expected} {error "S_AXI_HBM$c: unexpected $prop"}
  }
  set segments [get_bd_addr_segs -of_objects [get_bd_addr_spaces S_AXI_HBM$c]]
  if {[llength $segments] != 4} {error "S_AXI_HBM$c: expected four mapped HBM segments"}
  set offsets {}
  foreach seg $segments {
    set offset [get_property OFFSET $seg]
    set range [get_property RANGE $seg]
    if {$range != 0x40000000} {error "$seg: unexpected range $range"}
    lappend offsets [expr {$offset}]
    puts $fp "$seg offset=$offset range=$range"
  }
  set offsets [lsort -integer $offsets]
  for {set pc 0} {$pc < 4} {incr pc} {
    set expected [expr {$otpu(mem_base) + $c * $otpu(channel_bytes) + $pc * 0x40000000}]
    if {[lindex $offsets $pc] != $expected} {error "S_AXI_HBM$c: HBM map differs from the target"}
  }
}
set csr [get_bd_addr_segs -of_objects [get_bd_addr_spaces cips/CPM_PCIE_NOC_1] -filter {NAME =~ *M_AXI_CTRL*}]
if {[llength $csr] != 1 || [get_property OFFSET $csr] != $otpu(csr_addr) ||
    [get_property RANGE $csr] != $otpu(csr_bytes)} {error "PF1 openTPU CSR map differs from the target"}
puts $fp "$csr offset=[get_property OFFSET $csr] range=[get_property RANGE $csr]"
puts $fp "Platform boundary address checks passed; accelerator and hardware remain unqualified."
close $fp

set cw [get_bd_cells otpu_clocks]
set fp [open $otpu(out)/reports/clocks.txt w]
set input [get_property CONFIG.PRIM_IN_FREQ $cw]
set mult [get_property CONFIG.CLKFBOUT_MULT $cw]
set div [get_property CONFIG.DIVCLK_DIVIDE $cw]
foreach {n expected} {1 100.0 2 200.0} {
  set od [get_property CONFIG.CLKOUT${n}_DIVIDE $cw]
  set phase [get_property CONFIG.CLKOUT${n}_PHASE $cw]
  set actual [expr {$input * $mult / $div / $od}]
  if {abs($actual - $expected) > 0.001 || abs($phase) > 0.001} {
    error "core clock $n differs from the phase-aligned 100/200 MHz contract"
  }
  puts $fp "clk_out$n: $actual MHz, phase $phase degrees, input $input, multiply $mult, divide $div/$od"
}
if {[get_property CONFIG.PRIMITIVE_TYPE $cw] ne "MMCM" ||
    ![get_property CONFIG.USE_LOCKED $cw]} {error "core clocks must share a lock-qualified MMCM"}
puts $fp "AXI NoC remains on CIPS PL0 (100 MHz); CSR crosses through PF1 SmartConnect."
close $fp
if {![get_property CONFIG.C_EXT_RESET_HIGH [get_bd_cells otpu_reset]]} {
  error "core reset input must match the active-high PL peripheral_reset"
}
