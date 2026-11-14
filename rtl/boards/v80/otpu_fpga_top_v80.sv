// Physical V80 top: AMR's pins/management, the openTPU CSR window and eight HBM ports.
// Port names match the pinned AMR XDC. No changes to AMD's external source tree are needed.
module otpu_fpga_top_v80 (
  output wire  [0:0]  CH0_DDR4_0_0_act_n,
  output wire  [16:0] CH0_DDR4_0_0_adr,
  output wire  [1:0]  CH0_DDR4_0_0_ba,
  output wire  [0:0]  CH0_DDR4_0_0_bg,
  output wire  [0:0]  CH0_DDR4_0_0_ck_c,
  output wire  [0:0]  CH0_DDR4_0_0_ck_t,
  output wire  [0:0]  CH0_DDR4_0_0_cke,
  output wire  [0:0]  CH0_DDR4_0_0_cs_n,
  inout  wire  [8:0]  CH0_DDR4_0_0_dm_n,
  inout  wire  [71:0] CH0_DDR4_0_0_dq,
  inout  wire  [8:0]  CH0_DDR4_0_0_dqs_c,
  inout  wire  [8:0]  CH0_DDR4_0_0_dqs_t,
  output wire  [0:0]  CH0_DDR4_0_0_odt,
  output wire  [0:0]  CH0_DDR4_0_0_reset_n,
  output wire  [0:0]  CH0_DDR4_0_1_act_n,
  output wire  [17:0] CH0_DDR4_0_1_adr,
  input  wire  [0:0]  CH0_DDR4_0_1_alert_n,
  output wire  [1:0]  CH0_DDR4_0_1_ba,
  output wire  [1:0]  CH0_DDR4_0_1_bg,
  output wire  [0:0]  CH0_DDR4_0_1_ck_c,
  output wire  [0:0]  CH0_DDR4_0_1_ck_t,
  output wire  [0:0]  CH0_DDR4_0_1_cke,
  output wire  [0:0]  CH0_DDR4_0_1_cs_n,
  inout  wire  [71:0] CH0_DDR4_0_1_dq,
  inout  wire  [17:0] CH0_DDR4_0_1_dqs_c,
  inout  wire  [17:0] CH0_DDR4_0_1_dqs_t,
  output wire  [0:0]  CH0_DDR4_0_1_odt,
  output wire  [0:0]  CH0_DDR4_0_1_par,
  output wire  [0:0]  CH0_DDR4_0_1_reset_n,
  input  wire         gt_pcie_refclk_clk_n,
  input  wire         gt_pcie_refclk_clk_p,
  input  wire  [7:0]  gt_pciea1_grx_n,
  input  wire  [7:0]  gt_pciea1_grx_p,
  output wire  [7:0]  gt_pciea1_gtx_n,
  output wire  [7:0]  gt_pciea1_gtx_p,
  input  wire  [0:0]  hbm_ref_clk_0_clk_n,
  input  wire  [0:0]  hbm_ref_clk_0_clk_p,
  input  wire  [0:0]  hbm_ref_clk_1_clk_n,
  input  wire  [0:0]  hbm_ref_clk_1_clk_p,
  inout  wire         smbus_0_scl_io,
  inout  wire         smbus_0_sda_io,
  input  wire  [0:0]  sys_clk0_0_clk_n,
  input  wire  [0:0]  sys_clk0_0_clk_p,
  input  wire  [0:0]  sys_clk0_1_clk_n,
  input  wire  [0:0]  sys_clk0_1_clk_p
);
  wire clk, clk_f, rstn, aclk, arstn;
  // NoC has no user-routable AXI reset (PG313 Resets). Release the bridges' AXI
  // state once after configuration. A later PL reset resets/drains the core side
  // while preserving in-flight NoC transactions. Retry failed probes by a coordinated
  // PMC power-on/DFX reset of PL and NoC, never by pulsing a PL-only reset.
  logic axi_boot_rst = 1'b1;
  always_ff @(posedge aclk) if (arstn) axi_boot_rst <= 1'b0;
  wire  [63:0]       s_ctl_awaddr;
  wire               s_ctl_awvalid;
  wire               s_ctl_awready;
  wire  [31:0]       s_ctl_wdata;
  wire  [3:0]        s_ctl_wstrb;
  wire               s_ctl_wvalid;
  wire               s_ctl_wready;
  wire  [1:0]        s_ctl_bresp;
  wire               s_ctl_bvalid;
  wire               s_ctl_bready;
  wire  [63:0]       s_ctl_araddr;
  wire               s_ctl_arvalid;
  wire               s_ctl_arready;
  wire  [31:0]       s_ctl_rdata;
  wire  [1:0]        s_ctl_rresp;
  wire               s_ctl_rvalid;
  wire               s_ctl_rready;
  wire  [7:0]        m_awvalid;
  wire  [7:0]        m_awready;
  wire  [7:0][63:0]  m_awaddr;
  wire  [7:0]        m_awid;
  wire  [7:0][7:0]   m_awlen;
  wire  [7:0][2:0]   m_awsize;
  wire  [7:0][1:0]   m_awburst;
  wire  [7:0]        m_awlock;
  wire  [7:0][3:0]   m_awcache, m_awqos, m_awregion;
  wire  [7:0][2:0]   m_awprot;
  wire  [7:0]        m_wvalid;
  wire  [7:0]        m_wready;
  wire  [7:0][255:0] m_wdata;
  wire  [7:0][31:0]  m_wstrb;
  wire  [7:0]        m_wlast;
  wire  [7:0]        m_bvalid, m_bid;
  wire  [7:0]        m_bready;
  wire  [7:0][1:0]   m_bresp;
  wire  [7:0]        m_arvalid;
  wire  [7:0]        m_arready;
  wire  [7:0][63:0]  m_araddr;
  wire  [7:0]        m_arid;
  wire  [7:0][7:0]   m_arlen;
  wire  [7:0][2:0]   m_arsize;
  wire  [7:0][1:0]   m_arburst;
  wire  [7:0]        m_arlock;
  wire  [7:0][3:0]   m_arcache, m_arqos, m_arregion;
  wire  [7:0][2:0]   m_arprot;
  wire  [7:0]        m_rvalid, m_rid;
  wire  [7:0]        m_rready;
  wire  [7:0][255:0] m_rdata;
  wire  [7:0][1:0]   m_rresp;
  wire  [7:0]        m_rlast;

  // The bridge supplies every AXI4 sideband. A macro keeps the eight identical NoC
  // port connections together, as other openTPU board tops do.
`define OTPU_V80_HBM(C) \
    .S_AXI_HBM``C``_araddr(m_araddr[C]), \
    .S_AXI_HBM``C``_arburst(m_arburst[C]), \
    .S_AXI_HBM``C``_arcache(m_arcache[C]), \
    .S_AXI_HBM``C``_arid(m_arid[C]), \
    .S_AXI_HBM``C``_arlen(m_arlen[C]), \
    .S_AXI_HBM``C``_arlock(m_arlock[C]), \
    .S_AXI_HBM``C``_arprot(m_arprot[C]), \
    .S_AXI_HBM``C``_arqos(m_arqos[C]), \
    .S_AXI_HBM``C``_arready(m_arready[C]), \
    .S_AXI_HBM``C``_arregion(m_arregion[C]), \
    .S_AXI_HBM``C``_arsize(m_arsize[C]), \
    .S_AXI_HBM``C``_arvalid(m_arvalid[C]), \
    .S_AXI_HBM``C``_awaddr(m_awaddr[C]), \
    .S_AXI_HBM``C``_awburst(m_awburst[C]), \
    .S_AXI_HBM``C``_awcache(m_awcache[C]), \
    .S_AXI_HBM``C``_awid(m_awid[C]), \
    .S_AXI_HBM``C``_awlen(m_awlen[C]), \
    .S_AXI_HBM``C``_awlock(m_awlock[C]), \
    .S_AXI_HBM``C``_awprot(m_awprot[C]), \
    .S_AXI_HBM``C``_awqos(m_awqos[C]), \
    .S_AXI_HBM``C``_awready(m_awready[C]), \
    .S_AXI_HBM``C``_awregion(m_awregion[C]), \
    .S_AXI_HBM``C``_awsize(m_awsize[C]), \
    .S_AXI_HBM``C``_awvalid(m_awvalid[C]), \
    .S_AXI_HBM``C``_bid(m_bid[C]), \
    .S_AXI_HBM``C``_bready(m_bready[C]), \
    .S_AXI_HBM``C``_bresp(m_bresp[C]), \
    .S_AXI_HBM``C``_bvalid(m_bvalid[C]), \
    .S_AXI_HBM``C``_rdata(m_rdata[C]), \
    .S_AXI_HBM``C``_rid(m_rid[C]), \
    .S_AXI_HBM``C``_rlast(m_rlast[C]), \
    .S_AXI_HBM``C``_rready(m_rready[C]), \
    .S_AXI_HBM``C``_rresp(m_rresp[C]), \
    .S_AXI_HBM``C``_rvalid(m_rvalid[C]), \
    .S_AXI_HBM``C``_wdata(m_wdata[C]), \
    .S_AXI_HBM``C``_wlast(m_wlast[C]), \
    .S_AXI_HBM``C``_wready(m_wready[C]), \
    .S_AXI_HBM``C``_wstrb(m_wstrb[C]), \
    .S_AXI_HBM``C``_wvalid(m_wvalid[C])

  otpu_v80_wrapper u_platform (
    .CH0_DDR4_0_0_act_n(CH0_DDR4_0_0_act_n),
    .CH0_DDR4_0_0_adr(CH0_DDR4_0_0_adr),
    .CH0_DDR4_0_0_ba(CH0_DDR4_0_0_ba),
    .CH0_DDR4_0_0_bg(CH0_DDR4_0_0_bg),
    .CH0_DDR4_0_0_ck_c(CH0_DDR4_0_0_ck_c),
    .CH0_DDR4_0_0_ck_t(CH0_DDR4_0_0_ck_t),
    .CH0_DDR4_0_0_cke(CH0_DDR4_0_0_cke),
    .CH0_DDR4_0_0_cs_n(CH0_DDR4_0_0_cs_n),
    .CH0_DDR4_0_0_dm_n(CH0_DDR4_0_0_dm_n),
    .CH0_DDR4_0_0_dq(CH0_DDR4_0_0_dq),
    .CH0_DDR4_0_0_dqs_c(CH0_DDR4_0_0_dqs_c),
    .CH0_DDR4_0_0_dqs_t(CH0_DDR4_0_0_dqs_t),
    .CH0_DDR4_0_0_odt(CH0_DDR4_0_0_odt),
    .CH0_DDR4_0_0_reset_n(CH0_DDR4_0_0_reset_n),
    .CH0_DDR4_0_1_act_n(CH0_DDR4_0_1_act_n),
    .CH0_DDR4_0_1_adr(CH0_DDR4_0_1_adr),
    .CH0_DDR4_0_1_alert_n(CH0_DDR4_0_1_alert_n),
    .CH0_DDR4_0_1_ba(CH0_DDR4_0_1_ba),
    .CH0_DDR4_0_1_bg(CH0_DDR4_0_1_bg),
    .CH0_DDR4_0_1_ck_c(CH0_DDR4_0_1_ck_c),
    .CH0_DDR4_0_1_ck_t(CH0_DDR4_0_1_ck_t),
    .CH0_DDR4_0_1_cke(CH0_DDR4_0_1_cke),
    .CH0_DDR4_0_1_cs_n(CH0_DDR4_0_1_cs_n),
    .CH0_DDR4_0_1_dq(CH0_DDR4_0_1_dq),
    .CH0_DDR4_0_1_dqs_c(CH0_DDR4_0_1_dqs_c),
    .CH0_DDR4_0_1_dqs_t(CH0_DDR4_0_1_dqs_t),
    .CH0_DDR4_0_1_odt(CH0_DDR4_0_1_odt),
    .CH0_DDR4_0_1_par(CH0_DDR4_0_1_par),
    .CH0_DDR4_0_1_reset_n(CH0_DDR4_0_1_reset_n),
    .gt_pcie_refclk_clk_n(gt_pcie_refclk_clk_n),
    .gt_pcie_refclk_clk_p(gt_pcie_refclk_clk_p),
    .gt_pciea1_grx_n(gt_pciea1_grx_n),
    .gt_pciea1_grx_p(gt_pciea1_grx_p),
    .gt_pciea1_gtx_n(gt_pciea1_gtx_n),
    .gt_pciea1_gtx_p(gt_pciea1_gtx_p),
    .hbm_ref_clk_0_clk_n(hbm_ref_clk_0_clk_n),
    .hbm_ref_clk_0_clk_p(hbm_ref_clk_0_clk_p),
    .hbm_ref_clk_1_clk_n(hbm_ref_clk_1_clk_n),
    .hbm_ref_clk_1_clk_p(hbm_ref_clk_1_clk_p),
    .smbus_0_scl_io(smbus_0_scl_io),
    .smbus_0_sda_io(smbus_0_sda_io),
    .sys_clk0_0_clk_n(sys_clk0_0_clk_n),
    .sys_clk0_0_clk_p(sys_clk0_0_clk_p),
    .sys_clk0_1_clk_n(sys_clk0_1_clk_n),
    .sys_clk0_1_clk_p(sys_clk0_1_clk_p),
    .core_clk(clk),
    .core_clk2x(clk_f),
    .core_resetn(rstn),
    .axi_clk(aclk),
    .axi_resetn(arstn),
    .M_AXI_CTRL_araddr(s_ctl_araddr),
    .M_AXI_CTRL_arprot(),
    .M_AXI_CTRL_arready(s_ctl_arready),
    .M_AXI_CTRL_arvalid(s_ctl_arvalid),
    .M_AXI_CTRL_awaddr(s_ctl_awaddr),
    .M_AXI_CTRL_awprot(),
    .M_AXI_CTRL_awready(s_ctl_awready),
    .M_AXI_CTRL_awvalid(s_ctl_awvalid),
    .M_AXI_CTRL_bready(s_ctl_bready),
    .M_AXI_CTRL_bresp(s_ctl_bresp),
    .M_AXI_CTRL_bvalid(s_ctl_bvalid),
    .M_AXI_CTRL_rdata(s_ctl_rdata),
    .M_AXI_CTRL_rready(s_ctl_rready),
    .M_AXI_CTRL_rresp(s_ctl_rresp),
    .M_AXI_CTRL_rvalid(s_ctl_rvalid),
    .M_AXI_CTRL_wdata(s_ctl_wdata),
    .M_AXI_CTRL_wready(s_ctl_wready),
    .M_AXI_CTRL_wstrb(s_ctl_wstrb),
    .M_AXI_CTRL_wvalid(s_ctl_wvalid),
    `OTPU_V80_HBM(0),
    `OTPU_V80_HBM(1),
    `OTPU_V80_HBM(2),
    `OTPU_V80_HBM(3),
    `OTPU_V80_HBM(4),
    `OTPU_V80_HBM(5),
    `OTPU_V80_HBM(6),
    `OTPU_V80_HBM(7));
`undef OTPU_V80_HBM

  otpu_v80_sys u_sys (
    .clk, .clk_f, .rst(!rstn), .ctl_mrst(!rstn), .aclk, .arst(axi_boot_rst), .led(),
    .s_ctl_awaddr(s_ctl_awaddr[11:0]),
    .s_ctl_awvalid,
    .s_ctl_awready,
    .s_ctl_wdata,
    .s_ctl_wstrb,
    .s_ctl_wvalid,
    .s_ctl_wready,
    .s_ctl_bresp,
    .s_ctl_bvalid,
    .s_ctl_bready,
    .s_ctl_araddr(s_ctl_araddr[11:0]),
    .s_ctl_arvalid,
    .s_ctl_arready,
    .s_ctl_rdata,
    .s_ctl_rresp,
    .s_ctl_rvalid,
    .s_ctl_rready,
    .m_awvalid,
    .m_awready,
    .m_awaddr,
    .m_awid,
    .m_awlen,
    .m_awsize,
    .m_awburst,
    .m_awlock,
    .m_awcache,
    .m_awqos,
    .m_awregion,
    .m_awprot,
    .m_wvalid,
    .m_wready,
    .m_wdata,
    .m_wstrb,
    .m_wlast,
    .m_bvalid,
    .m_bid,
    .m_bready,
    .m_bresp,
    .m_arvalid,
    .m_arready,
    .m_araddr,
    .m_arid,
    .m_arlen,
    .m_arsize,
    .m_arburst,
    .m_arlock,
    .m_arcache,
    .m_arqos,
    .m_arregion,
    .m_arprot,
    .m_rvalid,
    .m_rid,
    .m_rready,
    .m_rdata,
    .m_rresp,
    .m_rlast);
endmodule
