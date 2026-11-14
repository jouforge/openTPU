# A completed route command alone is not timing/DRC acceptance. Write all reports and
# the checkpoint before this gate, so a failed build remains available for diagnosis.
set failures {}
if {[llength [get_ports -quiet {smbus_0_scl_io smbus_0_sda_io}]] != 2} {
  lappend failures "V80 SMBus pin contract changed"
}
if {![report_route_status -boolean_check ROUTED_FULLY] ||
    [report_route_status -boolean_check ERRORS_IN_ROUTES]} {
  lappend failures "routing is incomplete or contains errors"
}
set fp [open $otpu(out)/reports/impl_timing.txt r]
set timing [read $fp]
close $fp
set header 0
set summary {}
foreach line [split $timing \n] {
  if {[string first "WNS(ns)" $line] >= 0} {set header 1; continue}
  set values [regexp -all -inline {\S+} $line]
  if {$header && [llength $values] == 12 && [string is double -strict [lindex $values 0]]} {
    set summary $values
    break
  }
}
if {![llength $summary]} {error "missing setup/hold/pulse-width timing summary"}
set fp [open $otpu(out)/reports/impl_checks.txt w]
foreach {name index} {WNS 0 WHS 4 WPWS 8} {
  set slack [lindex $summary $index]
  puts $fp "$name $slack ns"
  if {![string is double -strict $slack] || $slack < 0} {lappend failures "$name $slack"}
}
# Bus skew is a separate timing assertion, absent from report_timing_summary.
set sf [open $otpu(out)/reports/impl_bus_skew.txt r]
set skew [read $sf]
close $sf
set count 0
set worst Inf
foreach line [split $skew \n] {
  if {[regexp {^Slack \((MET|VIOLATED)\)\s*:\s*(-?[0-9]+(?:\.[0-9]+)?)ns} $line match status slack]} {
    incr count
    set worst [expr {min($worst, $slack)}]
    if {$status ne "MET" || $slack < 0} {lappend failures "bus skew $slack"}
  }
}
puts $fp "Bus skew paths $count, worst slack $worst ns"
if {!$count} {lappend failures "missing bus skew timing paths"}
set cf [open $otpu(out)/reports/impl_cdc.txt r]
set cdc [read $cf]
close $cf
set count 0
foreach line [split $cdc \n] {
  set fields [regexp -all -inline {\S+} $line]
  if {[lindex $fields 0] ni {Critical Warning Info}} {continue}
  set counts [lrange $fields end-4 end]
  if {[llength $counts] != 5} {error "invalid CDC summary row: $line"}
  foreach value $counts {
    if {![string is integer -strict $value]} {error "invalid CDC summary row: $line"}
  }
  lassign $counts endpoints safe unsafe unknown no_async
  if {$unsafe || $unknown || $no_async} {lappend failures "CDC: $line"}
  incr count
}
puts $fp "CDC clock-pair rows $count"
if {!$count} {lappend failures "missing CDC clock-pair summary"}
set drc [get_drc_violations -filter {SEVERITY == "Error" || SEVERITY == "Critical Warning"}]
puts $fp "DRC errors/critical warnings [llength $drc]"
if {[llength $drc]} {lappend failures "DRC: $drc"}
# PG313: the dedicated C4CCIO_PAD0/P and PAD1/N pins, in banks 800 and 801.
foreach {port pin} {hbm_ref_clk_0_clk_p[0] N19 hbm_ref_clk_0_clk_n[0] N18 hbm_ref_clk_1_clk_p[0] N37 hbm_ref_clk_1_clk_n[0] N38} {
  set actual [get_property PACKAGE_PIN [get_ports $port]]
  puts $fp "$port $actual"
  if {$actual ne $pin} {lappend failures "$port: expected $pin, got $actual"}
}
puts $fp "Failures: $failures"
close $fp
if {[llength $failures]} {error "implementation checks failed: $failures"}
