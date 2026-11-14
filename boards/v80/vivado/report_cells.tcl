# Bulk attribute reads avoid repeated list_property calls over large primitive arrays.
# Preserve the inferred BRAM modes and DSP allocation for warning/cascade review.
set fp [open $otpu(out)/reports/${step}_cells.txt w]
set groups [dict create]
set all [get_cells -hierarchical -filter {REF_NAME =~ RAMB* || REF_NAME =~ DSP*}]
foreach cell $all ref [get_property REF_NAME $all] {dict lappend groups $ref $cell}
# Versal's saved netlist uses expanded types such as RAMB36E5_INT. Group by the
# actual reference name; report_utilization separately counts physical RAM/DSP tiles.
dict for {ref cells} $groups {
  puts $fp "$ref count [llength $cells]"
  if {![llength $cells]} {continue}
  set props [list_property [lindex $cells 0]]
  foreach prop {BWE_MODE_A BWE_MODE_B READ_WIDTH_A READ_WIDTH_B WRITE_WIDTH_A WRITE_WIDTH_B DSP_MODE AREG BREG ADREG MREG PREG} {
    if {$prop ni $props} {continue}
    set counts [dict create]
    foreach value [get_property $prop $cells] {
      if {$value ne ""} {dict incr counts $value}
    }
    if {[dict size $counts]} {puts $fp "$prop $counts"}
  }
}
close $fp
