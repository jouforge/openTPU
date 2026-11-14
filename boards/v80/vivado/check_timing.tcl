# Unmanaged constraints can emit warnings and continue on a missed query. Stop
# before optimization unless the shared openTPU timing contracts were applied.
if {[info exists ::otpu::fail] && [llength $::otpu::fail]} {
  error "openTPU timing constraints failed: $::otpu::fail"
}
if {![info exists ::otpu::ok(otpu_v80_top)] || $::otpu::ok(otpu_v80_top) != 1} {
  error "V80 posted-count/status timing constraints are missing"
}
set bridges [get_cells -quiet -hierarchical -filter {ORIG_REF_NAME == otpu_hbm_cs || REF_NAME == otpu_hbm_cs}]
if {[llength $bridges] != 8 || ![info exists ::otpu::ok(otpu_hbm_ch)] ||
    $::otpu::ok(otpu_hbm_ch) != [llength $bridges]} {
  error "expected timing constraints on all eight V80 memory bridges"
}
foreach v {c2x_pb c2x_rp c2x_qp} {
  if {![info exists ::$v] || ![llength [set ::$v]]} {
    error "missing pumped TMEM timing objects: $v"
  }
}
# link_design invalidates generated-clock timing data after reading the XDC.
# Refresh it before querying PERIOD (otherwise Vivado can return an empty value).
update_timing
set slow [get_clocks -quiet $c2x_cs]
set fast [get_clocks -quiet $c2x_cf]
if {[llength $slow] != 1 || [llength $fast] != 1} {
  error "pumped TMEM clock objects are missing"
}
set ts [get_property PERIOD $slow]
set tf [get_property PERIOD $fast]
if {![string is double -strict $ts] || ![string is double -strict $tf] ||
    abs($ts - 2.0 * $tf) > 0.001} {
  error "pumped TMEM requires related 1x/2x clocks"
}
# The shared MXU section currently selects DSP48E2 pins. This target uses the
# unpumped MXU; extending its configuration requires a DSP58 constraint review.
if {[llength [get_cells -quiet -hierarchical -filter {ORIG_REF_NAME == otpu_mxpc || REF_NAME == otpu_mxpc}]]} {
  error "pumped Versal MXU timing constraints are not qualified"
}
puts "OTPU_V80_TIMING bridges=[llength $bridges] pump_banks=[llength $c2x_pb] read_pins=[llength $c2x_rp] word_pins=[llength $c2x_qp]"
