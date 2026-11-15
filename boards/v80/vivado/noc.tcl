# Reproduce the qualified mixed image's PL NoC endpoint placement (0c31d07d).
# Keep these independent of the arithmetic and FIFO experiments. Unconstrained
# endpoints moved between builds with identical traffic requirements.
foreach {port site} {
  S08_AXI NOC_NMU512_X1Y5
  S09_AXI NOC_NMU512_X2Y5
  S10_AXI NOC_NMU512_X2Y4
  S11_AXI NOC_NMU512_X2Y6
  S12_AXI NOC_NMU512_X1Y4
  S13_AXI NOC_NMU512_X2Y3
  S14_AXI NOC_NMU512_X2Y2
  S15_AXI NOC_NMU512_X3Y5
  M00_AXI NOC_NSU512_X0Y4
} {
  set pin [get_bd_intf_pins axi_noc_cips/$port]
  if {[llength $pin] != 1} {error "expected one NoC endpoint: $port"}
  set_property CONFIG.PHYSICAL_LOC $site $pin
  if {[get_property CONFIG.PHYSICAL_LOC $pin] ne $site} {
    error "$port: NoC placement constraint was not retained"
  }
}
