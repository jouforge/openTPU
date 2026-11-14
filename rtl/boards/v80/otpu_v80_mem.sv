// One V80 logical channel: the existing native bridge on a 256-bit AXI4 NoC port.
// otpu_hbm_ch owns ordering, credits, partial-byte strobes, posted reads and reset draining.
// Its single-ID, <=16-beat INCR bursts are also legal AXI4: extend AxLEN to eight bits,
// supply the sidebands and add AMR's global HBM base after widening the relative address.
// Channel CH is four consecutive 1 GiB pseudo-channels, starting at MEM_BASE + CH * 4 GiB.
// No burst crosses 4 KiB (nor a pseudo-channel boundary). n_wdone advances only after B.
//
// rst requests a core reset; the bridge drains issued AXI traffic before clearing its FIFOs.
// arst must reset the AXI endpoint as well as the bridge. Responses with a nonzero ID set a
// sticky error, synchronized to clk and cleared only by arst, like the bridge's response errors.
module otpu_v80_mem #(
  parameter int CH = 0,
  parameter int XSW = 11,
  parameter int ARD = 32,
  parameter bit WPOST = 1'b1,
  parameter bit RHB = 1'b1,
  parameter bit WHB = 1'b0,
  parameter logic [63:0] MEM_BASE = 64'h0000004000000000
) (
  input  logic                clk, rst, aclk, arst,
  input  logic                n_cvalid, n_cwe,
  output logic                n_cready,
  input  logic [24:0]         n_caddr,
  input  logic [1:0]          n_csrc,
  input  logic [XSW-1:0]      n_cslot,
  input  logic                n_wvalid,
  output logic                n_wready,
  output logic [5:0]          n_wfree,
  input  logic [1023:0]       n_wdata,
  input  logic [127:0]        n_wmask,
  output logic                n_rvalid,
  output logic [1023:0]       n_rdata,
  output logic [15:0]         n_wdone,
  output logic                n_err,
  output logic                p_we,
  output logic [1:0][3:0]     p_q,
  output logic [4:0][XSW-1:0] p_slot,
  output logic [255:0]        p_d,
  output logic [XSW+1:0]      p_cntg,
  output logic                m_awvalid,
  input  logic                m_awready,
  output logic [63:0]         m_awaddr,
  output logic                m_awid,
  output logic [7:0]          m_awlen,
  output logic [2:0]          m_awsize,
  output logic [1:0]          m_awburst,
  output logic                m_awlock,
  output logic [3:0]          m_awcache, m_awqos, m_awregion,
  output logic [2:0]          m_awprot,
  output logic                m_wvalid,
  input  logic                m_wready,
  output logic [255:0]        m_wdata,
  output logic [31:0]         m_wstrb,
  output logic                m_wlast,
  input  logic                m_bvalid, m_bid,
  output logic                m_bready,
  input  logic [1:0]          m_bresp,
  output logic                m_arvalid,
  input  logic                m_arready,
  output logic [63:0]         m_araddr,
  output logic                m_arid,
  output logic [7:0]          m_arlen,
  output logic [2:0]          m_arsize,
  output logic [1:0]          m_arburst,
  output logic                m_arlock,
  output logic [3:0]          m_arcache, m_arqos, m_arregion,
  output logic [2:0]          m_arprot,
  input  logic                m_rvalid, m_rid,
  output logic                m_rready,
  input  logic [255:0]        m_rdata,
  input  logic [1:0]          m_rresp,
  input  logic                m_rlast
);
  initial if (CH < 0 || CH > 7 || MEM_BASE[31:0] != 0 || MEM_BASE > 64'hFFFFFFF800000000)
    $fatal(1, "otpu_v80_mem: CH 0..7, MEM_BASE aligned to 4 GiB with room for 32 GiB");
  logic [63:0] awaddr, araddr;
  logic [3:0]  awlen, arlen;
  logic        bridge_err, id_err = 1'b0;
  (* ASYNC_REG = "TRUE" *) logic id_s1 = 1'b0, id_s2 = 1'b0;
  always_ff @(posedge aclk) begin
    if (arst) id_err <= 1'b0;
    else if ((m_bvalid && m_bready && m_bid) || (m_rvalid && m_rready && m_rid))
      id_err <= 1'b1;
  end
  always_ff @(posedge clk) begin
    id_s1 <= id_err;
    id_s2 <= id_s1;
  end
  assign n_err = bridge_err | id_s2;
  assign m_awaddr = MEM_BASE + awaddr;
  assign m_araddr = MEM_BASE + araddr;
  assign m_awlen = {4'b0000, awlen};
  assign m_arlen = {4'b0000, arlen};
  assign m_awid = 1'b0;      assign m_arid = 1'b0;
  assign m_awsize = 3'd5;    assign m_arsize = 3'd5;
  assign m_awburst = 2'b01;  assign m_arburst = 2'b01;
  assign m_awlock = 1'b0;    assign m_arlock = 1'b0;
  assign m_awcache = 4'b0011; assign m_arcache = 4'b0011;
  assign m_awprot = 3'b000;  assign m_arprot = 3'b000;
  assign m_awqos = 4'b0000;  assign m_arqos = 4'b0000;
  assign m_awregion = 4'b0000; assign m_arregion = 4'b0000;

  otpu_hbm_ch #(.CB(128), .CAW(25), .PC(5'(CH)), .OAW(64), .ARD(ARD), .WDD(32),
                .WPOST(WPOST), .XSW(XSW), .RHB(RHB), .WHB(WHB)) u_ch (
    .clk, .rst, .n_cvalid, .n_cready, .n_cwe, .n_caddr, .n_csrc, .n_cslot,
    .n_wvalid, .n_wready, .n_wfree, .n_wdata, .n_wmask, .n_rvalid, .n_rdata, .n_wdone,
    .n_err(bridge_err), .p_we, .p_q, .p_slot, .p_d, .p_cntg, .aclk, .arst, .mwait(6'd8),
    .m_awvalid, .m_awready, .m_awaddr(awaddr), .m_awlen(awlen),
    .m_wvalid, .m_wready, .m_wdata, .m_wstrb, .m_wlast, .m_bvalid, .m_bready, .m_bresp,
    .m_arvalid, .m_arready, .m_araddr(araddr), .m_arlen(arlen),
    .m_rvalid, .m_rready, .m_rdata, .m_rresp, .m_rlast, .ev(), .ev_n());
endmodule
