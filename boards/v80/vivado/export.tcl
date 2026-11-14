# Called with the accepted implemented design open. Keep the device-image files
# in the implementation run directory so write_hw_platform includes their BIFs,
# CDOs and both segments, together with the implemented NoC solution.
set run_dir [get_property DIRECTORY [get_runs impl_1]]
set top [get_property TOP [current_fileset]]
set cwd [pwd]
cd $run_dir
write_device_image -force $top.pdi
write_hw_platform -fixed -include_bit -force $otpu(out)/otpu_v80.xsa
validate_hw_platform -verbose $otpu(out)/otpu_v80.xsa
cd $cwd
foreach segment {boot pld} {
  set src $run_dir/${top}_${segment}.pdi
  if {![file exists $src] || [file size $src] == 0} {error "missing device image: $src"}
  file copy -force $src $otpu(out)/otpu_v80_${segment}.pdi
}
