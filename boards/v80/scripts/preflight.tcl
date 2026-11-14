# Run inside the new project (or an in-memory project for make preflight).
if {[version -short] ne $otpu(version)} {
  error "Vivado [version -short] differs from the recorded $otpu(version); rerun prepare"
}
if {[llength [get_parts -quiet $otpu(part)]] != 1} {error "V80 part not installed: $otpu(part)"}
if {[llength [get_board_parts -quiet $otpu(board_part)]] != 1} {
  error "V80 board file not installed: $otpu(board_part)"
}
set_property board_part $otpu(board_part) [current_project]
set_property ip_repo_paths [list $otpu(base)/ip] [current_project]
update_ip_catalog
set fp [open $otpu(out)/reports/preflight.txt w]
puts $fp "Vivado [version -short]\nPart $otpu(part)\nBoard $otpu(board_part)"
foreach name {versal_cips axi_noc proc_sys_reset smartconnect axi_gpio smbus clk_wizard} {
  set ips [get_ipdefs -all -quiet xilinx.com:ip:$name:*]
  if {[llength $ips] == 0} {close $fp; error "IP missing: $name"}
  puts $fp "IP $ips"
}
puts $fp "Catalog checks passed; implementation licenses and timing are not qualified."
close $fp
