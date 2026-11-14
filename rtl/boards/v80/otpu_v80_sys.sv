// V80's shared openTPU board shell and eight native-to-NoC bridges.
// clk/clk_f are related 100/200 MHz clocks; aclk is AMR's 100 MHz PL0 clock.
// The read-only startup probes own AR/R until all four PCs in each channel respond.
// AXI reset must also reset the endpoint; a core reset only drains the existing bridges.
`ifndef OTPU_BOARD
`define OTPU_BOARD
`define OTPU_BOARD_UNSET
`endif
`ifndef OTPU_WPOST                // the board's NAT_WPOST (opentpu.host.boardmacro board_defines)
`define OTPU_WPOST 0
`endif
`ifndef OTPU_XPOST                // the board's NAT_XPOST
`define OTPU_XPOST 0
`endif
`ifndef OTPU_RHB                  // the board's NAT_RHB / NAT_WHB
`define OTPU_RHB 0
`endif
`ifndef OTPU_WHB
`define OTPU_WHB 0
`endif
`ifndef OTPU_XSW                  // log2 of the board's FIFO_DEPTH
`define OTPU_XSW 10
`endif
module otpu_v80_sys (
  input  logic              clk, clk_f, rst, ctl_mrst, aclk, arst,
  output logic [2:0]        led,
  input  logic [11:0]       s_ctl_awaddr,
  input  logic              s_ctl_awvalid,
  output logic              s_ctl_awready,
  input  logic [31:0]       s_ctl_wdata,
  input  logic [3:0]        s_ctl_wstrb,
  input  logic              s_ctl_wvalid,
  output logic              s_ctl_wready,
  output logic [1:0]        s_ctl_bresp,
  output logic              s_ctl_bvalid,
  input  logic              s_ctl_bready,
  input  logic [11:0]       s_ctl_araddr,
  input  logic              s_ctl_arvalid,
  output logic              s_ctl_arready,
  output logic [31:0]       s_ctl_rdata,
  output logic [1:0]        s_ctl_rresp,
  output logic              s_ctl_rvalid,
  input  logic              s_ctl_rready,
  output logic [7:0]        m_awvalid,
  input  logic [7:0]        m_awready,
  output logic [7:0][63:0]  m_awaddr,
  output logic [7:0]        m_awid,
  output logic [7:0][7:0]   m_awlen,
  output logic [7:0][2:0]   m_awsize,
  output logic [7:0][1:0]   m_awburst,
  output logic [7:0]        m_awlock,
  output logic [7:0][3:0]   m_awcache, m_awqos, m_awregion,
  output logic [7:0][2:0]   m_awprot,
  output logic [7:0]        m_wvalid,
  input  logic [7:0]        m_wready,
  output logic [7:0][255:0] m_wdata,
  output logic [7:0][31:0]  m_wstrb,
  output logic [7:0]        m_wlast,
  input  logic [7:0]        m_bvalid, m_bid,
  output logic [7:0]        m_bready,
  input  logic [7:0][1:0]   m_bresp,
  output logic [7:0]        m_arvalid,
  input  logic [7:0]        m_arready,
  output logic [7:0][63:0]  m_araddr,
  output logic [7:0]        m_arid,
  output logic [7:0][7:0]   m_arlen,
  output logic [7:0][2:0]   m_arsize,
  output logic [7:0][1:0]   m_arburst,
  output logic [7:0]        m_arlock,
  output logic [7:0][3:0]   m_arcache, m_arqos, m_arregion,
  output logic [7:0][2:0]   m_arprot,
  input  logic [7:0]        m_rvalid, m_rid,
  output logic [7:0]        m_rready,
  input  logic [7:0][255:0] m_rdata,
  input  logic [7:0][1:0]   m_rresp,
  input  logic [7:0]        m_rlast
);
  localparam int NT = 8, CB = 128, CAW = 25, XSW = `OTPU_XSW;
`ifdef OTPU_BOARD_UNSET
  initial $fatal(1, "otpu_v80_sys: no OTPU_BOARD target parameters");
`endif
  logic [NT-1:0]                 n_cvalid, n_cready, n_cwe, n_wvalid, n_wready, n_rvalid, n_err;
  logic [NT-1:0][5:0]            n_wfree;
  logic [NT-1:0][CAW-1:0]        n_caddr;
  logic [NT-1:0][CB*8-1:0]       n_wdata, n_rdata;
  logic [NT-1:0][CB-1:0]         n_wmask;
  logic [NT-1:0][15:0]           n_wdone;
  logic [NT-1:0][1:0]            n_csrc;
  logic [NT-1:0][XSW-1:0]        n_cslot;
  logic [NT-1:0][4:0][XSW-1:0]   p_slot;
  logic [NT-1:0]                 p_we;
  logic [NT-1:0][1:0][CB/32-1:0] p_q;
  logic [NT-1:0][255:0]          p_d;
  logic [NT-1:0][XSW+1:0]        p_cntg;
  logic [7:0]                    calib, probe_err;
  (* ASYNC_REG = "TRUE" *) logic [7:0] err_s1 = '0, err_s2 = '0;
  always_ff @(posedge clk) begin err_s1 <= probe_err; err_s2 <= err_s1; end
  logic [7:0] mem_err;
  assign n_err = mem_err | err_s2;
  otpu_board #(.S(1), .NCH(8), .CB(128), .CORE_KHZ(100000), .DDR_MTS(0),
               .HAS_TEMP(1'b0), .HAS_I2C(1'b0), .HOSTCAL(1'b0), .NAT_XQB(2),
               .DRAM_LOG2(35), .CH_LOG2(32) `OTPU_BOARD) u_board (
    .clk, .clk_f, .rst, .ctl_mrst, .calib, .ded('0), .temp(12'd0), .led,
    .i2c_lo(), .i2c_pin(5'h1F),
    .x2_s_tdata(), .x2_s_tvalid(), .x2_s_tlast(), .x2_s_tready('0),
    .x2_m_tdata('0), .x2_m_tvalid('0), .x2_m_tlast('0), .x2_m_thdr('0), .x2_m_tready(),
    .x2_up('0), .x2_rx_err('0), .x2_abort_in('0), .x2_abort(), .x2_rst(), .x2_card(), .x2_ncards(),
    .x2_c_we(), .x2_c_addr(), .x2_c_wd(), .x2_c_rd('0),
    .s_ctl_awaddr, .s_ctl_awvalid, .s_ctl_awready, .s_ctl_wdata, .s_ctl_wstrb, .s_ctl_wvalid,
    .s_ctl_wready, .s_ctl_bresp, .s_ctl_bvalid, .s_ctl_bready, .s_ctl_araddr, .s_ctl_arvalid,
    .s_ctl_arready, .s_ctl_rdata, .s_ctl_rresp, .s_ctl_rvalid, .s_ctl_rready,
    .n_cvalid, .n_cready, .n_cwe, .n_caddr, .n_wvalid, .n_wready, .n_wdata, .n_wmask,
    .n_rvalid, .n_rdata, .n_wdone, .n_wfree, .n_err,
    .n_csrc, .n_cslot, .p_clk(aclk), .p_we, .p_q, .p_slot, .p_d, .p_cntg);

  for (genvar j = 0; j < 8; j++) begin : g_ch
    logic av, rr, pv, pr;
    logic [63:0] aa, pa;
    logic [7:0] al;
    otpu_v80_ready #(.CH(j)) u_ready (.clk(aclk), .rst(arst), .ready(calib[j]),
      .err(probe_err[j]), .arvalid(pv), .arready(m_arready[j]), .araddr(pa),
      .rvalid(m_rvalid[j]), .rid(m_rid[j]), .rresp(m_rresp[j]), .rlast(m_rlast[j]), .rready(pr));
    assign m_arvalid[j] = calib[j] ? av : pv;
    assign m_araddr[j] = calib[j] ? aa : pa;
    assign m_arlen[j] = calib[j] ? al : 8'd0;
    assign m_rready[j] = calib[j] ? rr : pr;
    otpu_v80_mem #(.CH(j), .XSW(XSW), .WPOST(`OTPU_WPOST), .RHB(`OTPU_RHB),
                   .WHB(`OTPU_WHB)) u_mem (.clk, .rst, .aclk, .arst,
      .n_cvalid(n_cvalid[j]), .n_cready(n_cready[j]), .n_cwe(n_cwe[j]),
      .n_caddr(n_caddr[j]), .n_csrc(n_csrc[j]), .n_cslot(n_cslot[j]),
      .n_wvalid(n_wvalid[j]), .n_wready(n_wready[j]), .n_wfree(n_wfree[j]),
      .n_wdata(n_wdata[j]), .n_wmask(n_wmask[j]), .n_rvalid(n_rvalid[j]),
      .n_rdata(n_rdata[j]), .n_wdone(n_wdone[j]), .n_err(mem_err[j]),
      .p_we(p_we[j]), .p_q(p_q[j]), .p_slot(p_slot[j]),
      .p_d(p_d[j]), .p_cntg(p_cntg[j]), .m_awvalid(m_awvalid[j]),
      .m_awready(m_awready[j]), .m_awaddr(m_awaddr[j]), .m_awid(m_awid[j]),
      .m_awlen(m_awlen[j]), .m_awsize(m_awsize[j]), .m_awburst(m_awburst[j]),
      .m_awlock(m_awlock[j]), .m_awcache(m_awcache[j]), .m_awqos(m_awqos[j]),
      .m_awregion(m_awregion[j]), .m_awprot(m_awprot[j]), .m_wvalid(m_wvalid[j]),
      .m_wready(m_wready[j]), .m_wdata(m_wdata[j]), .m_wstrb(m_wstrb[j]),
      .m_wlast(m_wlast[j]), .m_bvalid(m_bvalid[j]), .m_bid(m_bid[j]),
      .m_bready(m_bready[j]), .m_bresp(m_bresp[j]), .m_arvalid(av),
      .m_arready(m_arready[j] && calib[j]), .m_araddr(aa), .m_arid(m_arid[j]),
      .m_arlen(al), .m_arsize(m_arsize[j]), .m_arburst(m_arburst[j]),
      .m_arlock(m_arlock[j]), .m_arcache(m_arcache[j]), .m_arqos(m_arqos[j]),
      .m_arregion(m_arregion[j]), .m_arprot(m_arprot[j]), .m_rvalid(m_rvalid[j] && calib[j]),
      .m_rid(m_rid[j]), .m_rready(rr), .m_rdata(m_rdata[j]),
      .m_rresp(m_rresp[j]), .m_rlast(m_rlast[j]));
  end
endmodule
