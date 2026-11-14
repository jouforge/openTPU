# Keep AMD's CIPS, HBM, DDR, SMBus and pin setup; extend the block design in bd.tcl.
set_property segmented_configuration true [current_project]
set xdc [list $otpu(out)/amr.pins.xdc $otpu(root)/boards/v80/constraints/otpu_v80.xdc]
add_files -fileset constrs_1 $xdc
set_property used_in_synthesis false [get_files $xdc]
# Reuse the same module-scoped bridge and pumped-memory timing contracts as BCU1525.
set bridge $otpu(root)/boards/bcu1525/constraints/otpu_hbm_ch.tcl
set late [list $bridge $otpu(root)/boards/ypcb-00338/constraints/otpu_clk2x.tcl \
               $otpu(root)/boards/v80/constraints/otpu_v80_top.tcl]
add_files -fileset constrs_1 -norecurse $late
set_property SCOPED_TO_REF otpu_hbm_cs [get_files $bridge]
set_property PROCESSING_ORDER LATE [get_files $late]
set_property USED_IN_SYNTHESIS false [get_files $late]
foreach f $late {
  if {![string match -nocase *tcl* [get_property FILE_TYPE [get_files $f]]]} {
    error "$f: expected unmanaged TCL constraints"
  }
}
set_property STEPS.PHYS_OPT_DESIGN.IS_ENABLED true [get_runs impl_1]
set_property STEPS.OPT_DESIGN.TCL.PRE $otpu(root)/boards/v80/vivado/check_timing.tcl [get_runs impl_1]
set_property STEPS.OPT_DESIGN.TCL.POST $otpu(root)/boards/v80/vivado/check_opt.tcl [get_runs impl_1]
set proj_name otpu_v80
source $otpu(base)/scripts/config_bd.tcl
if {[llength [get_bd_cells -quiet cips]] != 1} {error "AMR base did not create CIPS"}
source $otpu(root)/boards/v80/vivado/bd.tcl
validate_bd_design
source $otpu(root)/boards/v80/vivado/check_bd.tcl
save_bd_design
report_ip_status -file $otpu(out)/reports/ip_status.txt
write_bd_tcl -force $otpu(out)/reports/otpu_v80_bd.tcl

# One parameter source for simulations and synthesis; hash it with every RTL input.
source $otpu(out)/sources.tcl
add_files -norecurse $otpu_rtl
add_files -norecurse $otpu(out)/otpu_v80_defines.vh
set_property file_type {Verilog Header} [get_files otpu_v80_defines.vh]
set_property is_global_include true [get_files otpu_v80_defines.vh]
set_property top otpu_fpga_top_v80 [current_fileset]
update_compile_order -fileset sources_1
