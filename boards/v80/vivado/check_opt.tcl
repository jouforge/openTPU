# The built-in placer precheck omits some primitive checks (including DPCA-12).
# Reject structural DRC failures before spending time placing an invalid netlist.
report_drc -file otpu_opt_drc.rpt
set violations [get_drc_violations -filter {SEVERITY == "Error" || SEVERITY == "Critical Warning"}]
if {[llength $violations]} {error "post-opt DRC failed: $violations"}
