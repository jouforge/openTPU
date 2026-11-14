# Versal uses clk_wizard, with vector-valued output configuration properties.
# TMEM_PMP=1 needs phase-aligned 1x/2x clocks from the same MMCM.
set cw [create_bd_cell -type ip -vlnv xilinx.com:ip:clk_wizard otpu_clocks]
set_property -dict [list CONFIG.PRIM_IN_FREQ 100.000 CONFIG.PRIM_SOURCE No_buffer \
  CONFIG.CLKOUT_USED {true,true,false,false,false,false,false} \
  CONFIG.CLKOUT_REQUESTED_OUT_FREQUENCY {100.000,200.000,100.000,100.000,100.000,100.000,100.000} \
  CONFIG.CLKOUT_REQUESTED_PHASE {0.000,0.000,0.000,0.000,0.000,0.000,0.000} \
  CONFIG.USE_LOCKED true CONFIG.USE_RESET true] $cw
connect_bd_net [get_bd_pins cips/pl0_ref_clk] [get_bd_pins otpu_clocks/clk_in1]
connect_bd_net [get_bd_pins proc_sys_reset_0/peripheral_reset] [get_bd_pins otpu_clocks/reset]
set rs [create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset otpu_reset]
connect_bd_net [get_bd_pins otpu_clocks/clk_out1] [get_bd_pins otpu_reset/slowest_sync_clk]
connect_bd_net [get_bd_pins otpu_clocks/locked] [get_bd_pins otpu_reset/dcm_locked]
connect_bd_net [get_bd_pins proc_sys_reset_0/peripheral_reset] [get_bd_pins otpu_reset/ext_reset_in]

# SmartConnect performs the CSR clock crossing. Its original output (AMR GPIO) stays
# on PL0; the second output follows the clock associated with M_AXI_CTRL.
current_bd_instance /PF1
set cpin [create_bd_pin -dir I -type clk otpu_clk]
set rpin [create_bd_pin -dir I -type rst otpu_resetn]
set_property CONFIG.NUM_CLKS 2 [get_bd_cells smartconnect_pf1]
connect_bd_net $cpin [get_bd_pins smartconnect_pf1/aclk1]
set sr [get_bd_pins smartconnect_pf1/aresetn]
disconnect_bd_net [get_bd_nets -of_objects $sr] $sr
connect_bd_net $rpin $sr
current_bd_instance /
connect_bd_net [get_bd_pins otpu_clocks/clk_out1] [get_bd_pins PF1/otpu_clk]
connect_bd_net [get_bd_pins otpu_reset/interconnect_aresetn] [get_bd_pins PF1/otpu_resetn]

foreach {name pin hz assoc reset} {
  core_clk otpu_clocks/clk_out1 100000000 M_AXI_CTRL core_resetn
  core_clk2x otpu_clocks/clk_out2 200000000 {} core_resetn
  axi_clk cips/pl0_ref_clk 100000000 S_AXI_HBM0:S_AXI_HBM1:S_AXI_HBM2:S_AXI_HBM3:S_AXI_HBM4:S_AXI_HBM5:S_AXI_HBM6:S_AXI_HBM7 axi_resetn
} {
  set port [create_bd_port -dir O -type clk $name]
  set_property -dict [list CONFIG.FREQ_HZ $hz CONFIG.ASSOCIATED_RESET $reset] $port
  if {$assoc ne ""} {set_property CONFIG.ASSOCIATED_BUSIF $assoc $port}
  connect_bd_net [get_bd_pins $pin] $port
}
foreach {name pin} {core_resetn otpu_reset/peripheral_aresetn axi_resetn proc_sys_reset_0/peripheral_aresetn} {
  set port [create_bd_port -dir O -type rst $name]
  set_property CONFIG.POLARITY ACTIVE_LOW $port
  connect_bd_net [get_bd_pins $pin] $port
}
