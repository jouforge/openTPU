# The shell runner owns logs/output paths. Explicit error exit: batch Tcl errors must fail make.
if {$argc != 2} {puts stderr "usage: run.tcl config.tcl step"; exit 2}
source [lindex $argv 0]
set step [lindex $argv 1]
set requested_step $step
# PDI export uses the same implementation reports and acceptance checks.
if {$step eq "pdi"} {set step impl}
if {[catch {
  if {$step in {generate elaborate synth impl}} {
    if {[version -short] ne $otpu(version)} {error "Vivado differs from the recorded version; rerun prepare"}
    open_project $otpu(out)/project/otpu_v80.xpr
    open_bd_design [get_files otpu_v80.bd]
    validate_bd_design
    source $otpu(root)/boards/v80/vivado/check_bd.tcl
    generate_target all [get_files otpu_v80.bd]
    set wrappers [make_wrapper -files [get_files otpu_v80.bd] -top]
    add_files -norecurse $wrappers
    set_property top otpu_fpga_top_v80 [current_fileset]
    update_compile_order -fileset sources_1
    report_ip_status -file $otpu(out)/reports/ip_status.txt
    if {$step eq "elaborate"} {
      # Hierarchical BD IP needs its OOC DCP/stub even for RTL elaboration.
      create_ip_run [get_files otpu_v80.bd]
      set pending {}
      foreach run [get_runs -filter {IS_SYNTHESIS == 1 && NAME != synth_1}] {
        if {[get_property PROGRESS $run] ne "100%"} {lappend pending $run}
      }
      if {[llength $pending]} {
        launch_runs $pending -jobs 4
        foreach run $pending {
          wait_on_run $run
          if {[get_property PROGRESS $run] ne "100%"} {error "IP synthesis failed: $run"}
        }
      }
      synth_design -rtl -rtl_skip_ip -name rtl_1 -top otpu_fpga_top_v80
      set fp [open $otpu(out)/reports/elaborated.txt w]
      puts $fp "Top [get_property NAME [current_design]], [llength [get_cells -hierarchical]] elaborated cells"
      close $fp
    } elseif {$step in {synth impl}} {
      if {[get_property PROGRESS [get_runs synth_1]] ne "100%"} {
        launch_runs synth_1 -jobs 4
        wait_on_run synth_1
      }
      if {[get_property PROGRESS [get_runs synth_1]] ne "100%"} {
        error "synthesis failed: [get_property STATUS [get_runs synth_1]]"
      }
      if {$step eq "impl"} {
        if {[get_property PROGRESS [get_runs impl_1]] ne "100%"} {
          launch_runs impl_1 -to_step route_design -jobs 4
          wait_on_run impl_1
        }
        if {[get_property PROGRESS [get_runs impl_1]] ne "100%"} {
          error "implementation failed: [get_property STATUS [get_runs impl_1]]"
        }
        open_run impl_1
      } else {open_run synth_1}
      report_utilization -hierarchical -file $otpu(out)/reports/${step}_utilization.txt
      report_utilization -file $otpu(out)/reports/${step}_resources.txt
      report_clocks -file $otpu(out)/reports/${step}_clocks.txt
      report_timing_summary -check_timing_verbose -file $otpu(out)/reports/${step}_timing.txt
      report_cdc -file $otpu(out)/reports/${step}_cdc.txt
      report_cdc -details -severity Critical -file $otpu(out)/reports/${step}_cdc_critical.txt
      source $otpu(root)/boards/v80/vivado/report_cells.tcl
      write_checkpoint -force $otpu(out)/otpu_v80_${step}.dcp
      if {$step eq "impl"} {
        report_route_status -file $otpu(out)/reports/impl_route.txt
        report_bus_skew -sort_by_slack -file $otpu(out)/reports/impl_bus_skew.txt
        report_drc -file $otpu(out)/reports/impl_drc.txt
        report_methodology -file $otpu(out)/reports/impl_methodology.txt
        report_io -file $otpu(out)/reports/impl_io.txt
        source $otpu(root)/boards/v80/vivado/check_impl.tcl
        if {$requested_step eq "pdi"} {source $otpu(root)/boards/v80/vivado/export.tcl}
      }
    }
  } else {
    if {$step eq "preflight"} {
      create_project -in_memory -part $otpu(part)
    } elseif {$step eq "project"} {
      if {[file exists $otpu(out)/project]} {error "project exists; select a new OUT_DIR"}
      create_project otpu_v80 $otpu(out)/project -part $otpu(part)
    } else {error "unknown step: $step"}
    source $otpu(root)/boards/v80/scripts/preflight.tcl
    if {$step eq "project"} {source $otpu(root)/boards/v80/vivado/create_project.tcl}
  }
  close_project
} message options]} {
  puts stderr "V80 $step failed: $message"
  puts stderr [dict get $options -errorinfo]
  exit 1
}
puts "OTPU_V80_PASS $requested_step"
exit 0
