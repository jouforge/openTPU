// V80's AXI4 bridge under the upstream random native master and memory scoreboard.
// Reuse tb_hbmch_nat and otpu_hbm_model; check 64-bit NoC addresses before normalizing
// the model's bounded memory window. +offset=hex selects that window in the 4 GiB channel.
module tb_v80_mem #(parameter int CH = 7);
  localparam int CB = 128, CAW = 25, XSW = 11, WDD = 32, K = 4;
  localparam bit WPOST = 1'b1, RHB = 1'b1, WHB = 1'b0;
  int fault = 0;
  initial void'($value$plusargs("fault=%d", fault));
  int cp = 5000, ap = 5000;
  logic clk = 1'b0, aclk = 1'b0;
  initial begin
    int seed;
    void'($value$plusargs("cp=%d", cp));
    void'($value$plusargs("ap=%d", ap));
    if ($value$plusargs("seed=%d", seed)) void'($urandom(seed));
  end
  always #(cp) clk = ~clk;
  always #(ap) aclk = ~aclk;

  // resets
  logic rst = 1'b1, arst = 1'b1;
  longint ccyc = 0, acyc = 0;
  longint areset = -1, arep = 0, arlen = 40, hreset = -1, hrep = 0, hrlen = 2, hr_n = 0;
  longint unsigned offset = 0;
  localparam logic [63:0] BASE = 64'h4000000000 + (64'(CH) << 32);
  logic [63:0] window_base;
  assign window_base = BASE + offset;
  initial begin
    void'($value$plusargs("areset=%d", areset));
    void'($value$plusargs("arep=%d", arep));
    void'($value$plusargs("arlen=%d", arlen));
    void'($value$plusargs("hreset=%d", hreset));
    void'($value$plusargs("hrep=%d", hrep));
    void'($value$plusargs("hrlen=%d", hrlen));
    void'($value$plusargs("offset=%h", offset));
    if (offset[6:0] != 0 || offset + 1048576 > 64'h100000000)
      $fatal(1, "invalid test window offset");
  end
  function automatic bit at(input longint c, input longint s, input longint p);
    return s > 0 && c >= s && (p > 0 ? (c - s) % p == 0 : c == s);
  endfunction
  always @(posedge aclk) begin
    acyc <= acyc + 1;
    if (acyc == 60) arst <= 1'b0;
    if (at(acyc, hreset, hrep)) begin arst <= 1'b1; hr_n = hrlen; end
    else if (hr_n > 0) begin hr_n--; if (hr_n == 0) arst <= 1'b0; end
  end
  always @(posedge clk) begin
    ccyc <= ccyc + 1;
    if (ccyc == 30) rst <= 1'b0;
    if (at(ccyc, areset, arep)) rst <= 1'b1;
    if (at(ccyc, areset + arlen, arep)) rst <= 1'b0;
  end

  logic                 ncv, ncr, ncwe, nwv, nwr, nrv, nerr;
  logic [CAW-1:0]       nca;
  logic [1:0]           ncsrc;
  logic [XSW-1:0]       ncslot;
  logic [4:0][XSW-1:0]  pslot;                   // (otpu_hbm_ch's copies)
  logic                 pwe;
  logic [1:0][K-1:0]    pq;
  logic [255:0]         pd;
  logic [XSW+1:0]       pcntg;
  logic [CB*8-1:0]      nwd, nrd;
  logic [CB-1:0]        nwm;
  logic [15:0]          nwdone;
  logic [$clog2(WDD):0] nwfree;
  logic                 awv, awr, wv, wr, wl, bv, br, arv, arr, rv, rr, rl;
  logic [63:0]          awa, ara;
  logic                 awid, arid, awlock, arlock;
  logic [2:0]           awsize, arsize, awprot, arprot;
  logic [1:0]           awburst, arburst;
  logic [3:0]           awcache, arcache, awqos, arqos, awregion, arregion;
  logic [7:0]           awl, arl;
  logic [255:0]         wd, rd;
  logic [31:0]          ws;
  logic [1:0]           bre, rre;
  longint wap, wap_q = 0, wap_all = 0;      // the model's landed beats, and summed over its resets
  int     merr;
  logic   done, bad;

  tb_hbmch_nat #(.CB(CB), .CAW(CAW), .POST(WPOST), .XSW(XSW), .RHB(RHB), .WHB(WHB)) u_acc (
    .clk, .rst, .n_cvalid(ncv), .n_cready(ncr), .n_cwe(ncwe), .n_caddr(nca), .n_csrc(ncsrc),
    .n_cslot(ncslot), .n_wvalid(nwv), .n_wready(nwr), .n_wfree(8'(nwfree)), .n_wdata(nwd), .n_wmask(nwm),
    .n_rvalid(nrv), .n_rdata(nrd), .n_wdone(nwdone), .aclk, .p_we(pwe), .p_q(pq), .p_slot(pslot),
    .p_d(pd), .p_cntg(pcntg), .crst(u_ch.u_ch.a_crst), .wap(wap_all), .done, .bad);

  otpu_v80_mem #(.CH(CH), .XSW(XSW)) u_ch (
    .clk, .rst, .n_cvalid(ncv), .n_cready(ncr), .n_cwe(ncwe), .n_caddr(nca + 25'(offset >> 7)), .n_csrc(ncsrc),
    .n_cslot(ncslot), .n_wvalid(nwv), .n_wready(nwr), .n_wfree(nwfree), .n_wdata(nwd), .n_wmask(nwm),
    .n_rvalid(nrv), .n_rdata(nrd), .n_wdone(nwdone), .n_err(nerr),
    .p_we(pwe), .p_q(pq), .p_slot(pslot), .p_d(pd), .p_cntg(pcntg), .aclk, .arst,
    .m_awvalid(awv), .m_awready(awr), .m_awaddr(awa), .m_awlen(awl), .m_awid(awid), .m_awsize(awsize), .m_awburst(awburst),
    .m_awlock(awlock), .m_awcache(awcache), .m_awqos(awqos), .m_awregion(awregion), .m_awprot(awprot),
    .m_wvalid(wv), .m_wready(wr), .m_wdata(wd), .m_wstrb(ws), .m_wlast(wl),
    .m_bvalid(bv), .m_bready(br), .m_bresp(bre), .m_bid(fault == 1),
    .m_arvalid(arv), .m_arready(arr), .m_araddr(ara), .m_arlen(arl), .m_arid(arid), .m_arsize(arsize), .m_arburst(arburst),
    .m_arlock(arlock), .m_arcache(arcache), .m_arqos(arqos), .m_arregion(arregion), .m_arprot(arprot),
    .m_rvalid(rv), .m_rready(rr), .m_rdata(rd), .m_rresp(rre), .m_rid(fault == 2), .m_rlast(rl));

  otpu_hbm_model #(.BEATS(1 << 16), .PC(5'd0)) u_mem (
    .clk(aclk), .rst(arst), .dump(1'b0),
    .awvalid(awv), .awready(awr), .awid(6'd0), .awaddr(34'(awa - window_base)), .awlen(awl[3:0]),
    .wvalid(wv), .wready(wr), .wdata(wd), .wstrb(ws), .wlast(wl),
    .bvalid(bv), .bready(br), .bid(), .bresp(bre),
    .arvalid(arv), .arready(arr), .arid(6'd0), .araddr(34'(ara - window_base)), .arlen(arl[3:0]),
    .rvalid(rv), .rready(rr), .rid(), .rdata(rd), .rresp(rre), .rlast(rl), .wap, .nerr(merr));

  always @(posedge aclk) begin
    wap_all += (wap >= wap_q) ? wap - wap_q : wap;
    wap_q = wap;
  end

  // Check the full NoC address before translating a small window for the shared model.
  // The window can straddle a pseudo-channel boundary or occupy the top of channel 7.
  task automatic check_addr(input logic [63:0] addr, input logic [7:0] len);
    if (addr < window_base || addr + 32 * (64'(len) + 1) > window_base + 1048576 ||
        addr < BASE || addr + 32 * (64'(len) + 1) > BASE + 64'h100000000 ||
        addr[4:0] != 0 || len > 15 || int'(addr[11:0]) + 32 * (int'(len) + 1) > 4096)
      $fatal(1, "invalid full NoC address %h len %d, base %h", addr, len, window_base);
  endtask
  logic        awheld = 0, arheld = 0;
  logic [63:0] awprev, arprev;
  logic [7:0]  awlprev, arlprev;
  always @(posedge aclk) begin
    if (!arst) begin
      if (awv) begin
        check_addr(awa, awl);
        if ({awid, awsize, awburst, awlock, awcache, awprot, awqos, awregion} !=
            {1'b0, 3'd5, 2'b01, 1'b0, 4'b0011, 3'b0, 4'b0, 4'b0})
          $fatal(1, "AW sidebands");
      end
      if (arv) begin
        check_addr(ara, arl);
        if ({arid, arsize, arburst, arlock, arcache, arprot, arqos, arregion} !=
            {1'b0, 3'd5, 2'b01, 1'b0, 4'b0011, 3'b0, 4'b0, 4'b0})
          $fatal(1, "AR sidebands");
      end
      if (awheld && (!awv || {awa, awl} != {awprev, awlprev})) $fatal(1, "unstable AW");
      if (arheld && (!arv || {ara, arl} != {arprev, arlprev})) $fatal(1, "unstable AR");
    end
    awheld <= !arst && awv && !awr; arheld <= !arst && arv && !arr;
    awprev <= awa; awlprev <= awl; arprev <= ara; arlprev <= arl;
  end

  // p_cntg: one gray code step at a time, each after a beat's last quarter (pc_pend: such beats
  // not yet counted)
  logic [XSW+1:0] pcg_q = '0;
  int pcg_err = 0, pc_pend = 0;
  function automatic logic [XSW+1:0] g2b(input logic [XSW+1:0] g);
    logic [XSW+1:0] b;
    b[XSW+1] = g[XSW+1];
    for (int i = XSW; i >= 0; i--) b[i] = b[i + 1] ^ g[i];
    return b;
  endfunction
  always @(posedge aclk) begin
    if (pcntg != pcg_q && g2b(pcntg) != g2b(pcg_q) + (XSW + 2)'(1)) begin
      if (pcg_err < 10) $display("ERROR p_cntg %h -> %h: not one gray code step", pcg_q, pcntg);
      pcg_err++;
    end
    if (pcntg != pcg_q) begin
      if (pc_pend == 0) begin
        if (pcg_err < 10) $display("ERROR p_cntg steps before a posted beat's last quarter is written");
        pcg_err++;
      end else pc_pend--;
    end
    if (pwe && pq[0][K-1]) pc_pend++;
    pcg_q <= pcntg;
  end

  longint tmax = 4000000;
  initial void'($value$plusargs("tmax=%d", tmax));
  always @(posedge clk) begin
    if (nerr && fault == 0) begin
      $display("FAIL n_err: the bridge saw a response it did not expect");
      $finish;
    end
    if (done) begin
      repeat (4) @(posedge clk);
      if (fault != 0) begin
        if (!nerr) $fatal(1, "bad response ID did not set n_err");
        rst <= 1'b1;
        repeat (12) @(posedge clk);
        if (!nerr) $fatal(1, "core reset cleared the sticky ID error");
        arst <= 1'b1;
        repeat (12) @(posedge clk);
        if (nerr) $fatal(1, "AXI reset did not clear the ID error");
      end
      $display("%s cycles=%0d", (bad || merr != 0 || pcg_err != 0) ? "FAIL" : "PASS", ccyc);
      $finish;
    end
    if (ccyc > tmax) begin
      $display("TIMEOUT");
      $finish;
    end
  end
endmodule
