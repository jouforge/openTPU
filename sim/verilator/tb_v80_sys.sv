// Full target board shell: CSR contract, startup gating, probe failure/reset and HALT load.
module tb_v80_sys;
  logic clk = 0, clk_f = 1, aclk = 0, rst = 1, arst = 1;
  always #5000 clk = ~clk;
  always #2500 clk_f = ~clk_f;
  always #3701 aclk = ~aclk;
  logic [11:0]       s_ctl_awaddr;
  logic              s_ctl_awvalid;
  logic              s_ctl_awready;
  logic [31:0]       s_ctl_wdata;
  logic [3:0]        s_ctl_wstrb;
  logic              s_ctl_wvalid;
  logic              s_ctl_wready;
  logic [1:0]        s_ctl_bresp;
  logic              s_ctl_bvalid;
  logic              s_ctl_bready;
  logic [11:0]       s_ctl_araddr;
  logic              s_ctl_arvalid;
  logic              s_ctl_arready;
  logic [31:0]       s_ctl_rdata;
  logic [1:0]        s_ctl_rresp;
  logic              s_ctl_rvalid;
  logic              s_ctl_rready;
  logic [7:0]        m_awvalid;
  logic [7:0]        m_awready;
  logic [7:0][63:0]  m_awaddr;
  logic [7:0]        m_awid;
  logic [7:0][7:0]   m_awlen;
  logic [7:0][2:0]   m_awsize;
  logic [7:0][1:0]   m_awburst;
  logic [7:0]        m_awlock;
  logic [7:0][3:0]   m_awcache, m_awqos, m_awregion;
  logic [7:0][2:0]   m_awprot;
  logic [7:0]        m_wvalid;
  logic [7:0]        m_wready;
  logic [7:0][255:0] m_wdata;
  logic [7:0][31:0]  m_wstrb;
  logic [7:0]        m_wlast;
  logic [7:0]        m_bvalid, m_bid;
  logic [7:0]        m_bready;
  logic [7:0][1:0]   m_bresp;
  logic [7:0]        m_arvalid;
  logic [7:0]        m_arready;
  logic [7:0][63:0]  m_araddr;
  logic [7:0]        m_arid;
  logic [7:0][7:0]   m_arlen;
  logic [7:0][2:0]   m_arsize;
  logic [7:0][1:0]   m_arburst;
  logic [7:0]        m_arlock;
  logic [7:0][3:0]   m_arcache, m_arqos, m_arregion;
  logic [7:0][2:0]   m_arprot;
  logic [7:0]        m_rvalid, m_rid;
  logic [7:0]        m_rready;
  logic [7:0][255:0] m_rdata;
  logic [7:0][1:0]   m_rresp;
  logic [7:0]        m_rlast;
  otpu_v80_sys u_sys (.clk, .clk_f, .rst, .ctl_mrst(rst), .aclk, .arst, .led(),
    .s_ctl_awaddr, .s_ctl_awvalid, .s_ctl_awready, .s_ctl_wdata, .s_ctl_wstrb, .s_ctl_wvalid, .s_ctl_wready, .s_ctl_bresp, .s_ctl_bvalid, .s_ctl_bready, .s_ctl_araddr, .s_ctl_arvalid, .s_ctl_arready, .s_ctl_rdata, .s_ctl_rresp, .s_ctl_rvalid, .s_ctl_rready, .m_awvalid, .m_awready, .m_awaddr, .m_awid, .m_awlen, .m_awsize, .m_awburst, .m_awlock, .m_awcache, .m_awqos, .m_awregion, .m_awprot, .m_wvalid, .m_wready, .m_wdata, .m_wstrb, .m_wlast, .m_bvalid, .m_bid, .m_bready, .m_bresp, .m_arvalid, .m_arready, .m_araddr, .m_arid, .m_arlen, .m_arsize, .m_arburst, .m_arlock, .m_arcache, .m_arqos, .m_arregion, .m_arprot, .m_rvalid, .m_rid, .m_rready, .m_rdata, .m_rresp, .m_rlast);
  bit go = 0;
  int fault = 0;
  int counts[8], left[8], delay_r[8];
  logic [7:0] pending = '0;
  int axi_cycle = 0;
  always @(posedge aclk) begin
    axi_cycle <= axi_cycle + 1;
    for (int c = 0; c < 8; c++) begin
      if (arst) begin
        counts[c] <= 0; left[c] <= 0; delay_r[c] <= 0; pending[c] <= 0;
        m_rvalid[c] <= 0; m_arready[c] <= 0;
      end else begin
        m_arready[c] <= go && !pending[c] && axi_cycle % 5 != 0;
        if (m_arvalid[c] && m_arready[c]) begin
          if (pending[c]) $fatal(1, "multiple reads in flight in simple model");
          if (counts[c] < 4) begin
            if (m_araddr[c] != 64'h4000000000 + (64'(c) << 32) + (64'(counts[c]) << 30) ||
                m_arlen[c] != 0 || m_arsize[c] != 5 || m_arid[c] != 0)
              $fatal(1, "wrong probe %0d/%0d address %h", c, counts[c], m_araddr[c]);
          end else if (m_araddr[c] != 64'h4000000000 + (64'(c) << 32))
            $fatal(1, "unexpected instruction fetch address %h", m_araddr[c]);
          pending[c] <= 1; m_arready[c] <= 0;
          left[c] <= int'(m_arlen[c]) + 1; delay_r[c] <= 5 + c;
        end
        if (pending[c] && delay_r[c] > 0) delay_r[c] <= delay_r[c] - 1;
        if (pending[c] && delay_r[c] == 0 && !m_rvalid[c]) begin
          m_rvalid[c] <= 1;
          m_rlast[c] <= left[c] == 1;
          m_rid[c] <= fault == 1 && c == 7 && counts[c] == 3;
          m_rresp[c] <= (fault == 2 && c == 7 && counts[c] == 3) ? 2'b10 : 2'b00;
          // One 256-bit instruction per AXI beat: HALT (opcode 1).
          m_rdata[c] <= 256'd1;
        end
        if (m_rvalid[c] && m_rready[c]) begin
          m_rvalid[c] <= 0;
          left[c] <= left[c] - 1;
          if (m_rlast[c]) begin pending[c] <= 0; counts[c] <= counts[c] + 1; end
        end
        if (m_awvalid[c] || m_wvalid[c]) $fatal(1, "probe/HALT test must not write HBM");
      end
    end
  end
  task automatic rd(input logic [11:0] addr, output logic [31:0] data);
    @(negedge clk); s_ctl_araddr = addr; s_ctl_arvalid = 1;
    do @(posedge clk); while (!s_ctl_arready);
    @(negedge clk); s_ctl_arvalid = 0;
    wait (s_ctl_rvalid);
    repeat (3) @(negedge clk);
    data = s_ctl_rdata;
    if (s_ctl_rresp != 0) $fatal(1, "CSR read response");
    s_ctl_rready = 1;
    @(negedge clk); s_ctl_rready = 0;
  endtask
  task automatic wr(input logic [11:0] addr, input logic [31:0] data, input logic [3:0] mask = 15);
    @(negedge clk); s_ctl_awaddr = addr; s_ctl_awvalid = 1;
    // AXI masters must send W independently of AWREADY. The shared CSR slave
    // accepts AW/W together, so hold AW across the intentional W delay.
    repeat (3) @(negedge clk);
    s_ctl_wdata = data; s_ctl_wstrb = mask; s_ctl_wvalid = 1;
    do @(posedge clk); while (!(s_ctl_awready && s_ctl_wready));
    @(negedge clk); s_ctl_awvalid = 0; s_ctl_wvalid = 0;
    wait (s_ctl_bvalid);
    if (s_ctl_bresp != 0) $fatal(1, "CSR write response");
    repeat (2) @(negedge clk);
    s_ctl_bready = 1;
    @(negedge clk); s_ctl_bready = 0;
  endtask
  task automatic check(input logic [11:0] addr, input logic [31:0] expected);
    logic [31:0] d;
    rd(addr, d);
    if (d !== expected) $fatal(1, "CSR %h = %h expected %h", addr, d, expected);
  endtask
  initial begin
    logic [31:0] d;
    void'($value$plusargs("fault=%d", fault));
    s_ctl_awvalid = 0; s_ctl_wvalid = 0; s_ctl_bready = 0;
    s_ctl_arvalid = 0; s_ctl_rready = 0;
    m_awready = '0; m_wready = '0; m_bvalid = '0; m_bid = '0; m_bresp = '0;
    repeat (30) @(negedge clk); rst = 0;
    repeat (10) @(negedge aclk); arst = 0;
    check(12'h000, 32'h4f545055);
    check(12'h044, 100000);
    check(12'h090, 32'h01200823);
    check(12'h098, 8);
    check(12'h09c, 32'hd49b52f5);
    wr(12'h038, 32'h12345678);
    check(12'h038, 32'h12345678);
    check(12'h04c, 0);  // No XADC-compatible temperature sample from this platform.
    rd(12'h040, d);
    if (d[1]) $fatal(1, "V80 must not advertise an unwired temperature sensor");
    wr(12'h008, 1);     // RUN cannot rise before the HBM probes complete.
    check(12'h008, 0);
    go = 1;
    repeat (300) @(negedge clk);
    rd(12'h00c, d);
    if (fault != 0) begin
      // Channel 7 fails: CALIB1 (odd channels) stays low, CALIB0 is high.
      if ((d & 32'h70) != 32'h30) $fatal(1, "probe failure not reported: %h", d);
      // A core reset cannot claim that a failed HBM route recovered.
      rst = 1; repeat (12) @(negedge clk); rst = 0;
      repeat (20) @(negedge clk);
      rd(12'h00c, d);
      if ((d & 32'h70) != 32'h30) $fatal(1, "core reset lost probe error: %h", d);
      fault = 0;
      @(negedge aclk); arst = 1;
      repeat (12) @(negedge aclk); arst = 0;
      repeat (300) @(negedge clk);
    end
    rd(12'h00c, d);
    if ((d & 32'h70) != 32'h60) $fatal(1, "memory not ready: %h", d);
    // Load one real instruction through the native adapter, V80 bridge and AXI R path.
    wr(12'h010, 0); wr(12'h014, 1); wr(12'h008, 2);
    repeat (300) @(negedge clk);
    rd(12'h00c, d);
    if (d & 32'h16) $fatal(1, "loader did not finish: %h", d);
    wr(12'h008, 1);
    repeat (300) @(negedge clk);
    rd(12'h00c, d);
    if ((d & 32'h13) != 1) $fatal(1, "HALT failed: %h", d);
    check(12'h020, 1);
    $display("PASS v80 system"); $finish;
  end
  initial begin #100000000; $fatal(1, "timeout"); end
endmodule
