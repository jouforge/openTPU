# The AMR base's impl.xdc targets an absent top_i/clock_reset + top_i/base_logic
# hierarchy. Its empty SLR pblocks and NSU LOC do not constrain this BD. Keep the
# platform's other physical pins, and let the placer choose the fabric floorplan.
# IP-generated XDC owns CIPS/NoC/DDR and MMCM clocks. Core 1x/2x and CIPS PL0 stay
# related and timed; do not false-path the accelerator's clock or reset domains.

# PG313: C4CCIO_PAD0 is P, PAD1 is N. The pinned AMR file reverses these pairs;
# prepare.py omits its four HBM assignments from the generated amr.pins.xdc.
set_property PACKAGE_PIN N19 [get_ports {hbm_ref_clk_0_clk_p[0]}]
set_property PACKAGE_PIN N18 [get_ports {hbm_ref_clk_0_clk_n[0]}]
set_property PACKAGE_PIN N37 [get_ports {hbm_ref_clk_1_clk_p[0]}]
set_property PACKAGE_PIN N38 [get_ports {hbm_ref_clk_1_clk_n[0]}]

# SMBus is asynchronous to the fabric clock. The pinned AMR PHY samples SCL/SDA
# through xpm_cdc_single (DEST_SYNC_FF=3, SRC_INPUT_REG=0); XPM constrains the
# synchronizer. Open-drain output timing is determined by the PHY's bus counters,
# not by a synchronous receiver clock at these board pins.
set smb [get_ports {smbus_0_scl_io smbus_0_sda_io}]
set_false_path -from $smb
set_false_path -to $smb
