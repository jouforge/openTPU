# Platform boundary for the openTPU core, with the AMR management design retained.
# No accelerator is instantiated here: the ports are connected by the physical top.
current_bd_instance /

# A dedicated PF1 BAR2 window, preserving AMR's GPIO at BAR2 + 0x20000.
set_property CONFIG.NUM_MI 2 [get_bd_cells PF1/smartconnect_pf1]
current_bd_instance /PF1
set csr [create_bd_intf_pin -mode Master -vlnv xilinx.com:interface:aximm_rtl:1.0 OTPU_CTRL]
connect_bd_intf_net [get_bd_intf_pins smartconnect_pf1/M01_AXI] $csr
current_bd_instance /
set csr [create_bd_intf_port -mode Master -vlnv xilinx.com:interface:aximm_rtl:1.0 M_AXI_CTRL]
set_property -dict [list CONFIG.PROTOCOL AXI4LITE CONFIG.ADDR_WIDTH 64 CONFIG.DATA_WIDTH 32 \
  CONFIG.FREQ_HZ $otpu(core_hz)] $csr
connect_bd_intf_net [get_bd_intf_pins PF1/OTPU_CTRL] $csr
set seg [get_bd_addr_segs M_AXI_CTRL/Reg]
assign_bd_address -offset $otpu(csr_addr) -range $otpu(csr_bytes) \
  -target_address_space [get_bd_addr_spaces cips/CPM_PCIE_NOC_1] $seg -force

foreach c {0 1 2 3} {
  exclude_bd_addr_seg -target_address_space [get_bd_addr_spaces cips/FPD_CCI_NOC_$c] $seg
}

# Eight PL AXI4 inputs, one per logical channel. Channel c spans four 1 GiB HBM PCs
# (two physical channels); the host already reaches those exact global address segments.
set noc [get_bd_cells axi_noc_cips]
set_property CONFIG.NUM_SI 16 $noc
set busifs {M00_AXI}
for {set c 0} {$c < $otpu(nch)} {incr c} {
  set si [format S%02d_AXI [expr {8 + $c}]]
  set port [create_bd_intf_port -mode Slave -vlnv xilinx.com:interface:aximm_rtl:1.0 S_AXI_HBM$c]
  set_property -dict [list CONFIG.PROTOCOL AXI4 CONFIG.ADDR_WIDTH 64 CONFIG.DATA_WIDTH 256 \
    CONFIG.ID_WIDTH 1 CONFIG.FREQ_HZ $otpu(core_hz) CONFIG.HAS_BURST 1 \
    CONFIG.NUM_READ_OUTSTANDING 16 CONFIG.NUM_WRITE_OUTSTANDING 16] $port
  set conns {}
  for {set h [expr {2 * $c}]} {$h < 2 * $c + 2} {incr h} {
    foreach p {0 2} {
      # NoC 1.1 in Vivado 2026.1.1 requires the braced scalar spelling used by AMR.
      # A list of bare numbers is accepted as a property but creates no HBM segments.
      lappend conns HBM${h}_PORT$p {read_bw {250} write_bw {250} read_avg_burst {128} write_avg_burst {128}}
    }
  }
  set pin [get_bd_intf_pins axi_noc_cips/$si]
  set_property -dict [list CONFIG.DATA_WIDTH 256 CONFIG.CATEGORY pl CONFIG.CONNECTIONS $conns] $pin
  connect_bd_intf_net $port $pin
  lappend busifs $si

}
# The existing PL NoC clock already runs from CIPS PL0 at 100 MHz.
set_property CONFIG.ASSOCIATED_BUSIF [join $busifs :] [get_bd_pins axi_noc_cips/aclk4]
source $otpu(root)/boards/v80/vivado/clocks.tcl
source $otpu(root)/boards/v80/vivado/noc.tcl

for {set c 0} {$c < $otpu(nch)} {incr c} {
  set si [format S%02d_AXI [expr {8 + $c}]]
  for {set pc 0} {$pc < 4} {incr pc} {
    set h [expr {2 * $c + $pc / 2}]
    set half [expr {$pc % 2}]
    set addr [expr {$otpu(mem_base) + $c * $otpu(channel_bytes) + $pc * 0x40000000}]
    assign_bd_address -offset $addr -range 0x40000000 -target_address_space \
      [get_bd_addr_spaces S_AXI_HBM$c] [get_bd_addr_segs axi_noc_cips/$si/HBM${h}_PC$half] -force
  }
}
