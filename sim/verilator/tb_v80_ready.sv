// Startup probe failures must not enable RUN or write memory; only AXI reset retries.
module tb_v80_ready;
  logic        clk = 0, rst = 1, ready, err, av, ar = 0, rv = 0, rid = 0, rl = 1, rr;
  logic [63:0] aa;
  logic [1:0]  resp = 0;
  always #5000 clk = ~clk;
  otpu_v80_ready #(.CH(7)) u_dut (.clk, .rst, .ready, .err, .arvalid(av), .arready(ar),
    .araddr(aa), .rvalid(rv), .rid, .rresp(resp), .rlast(rl), .rready(rr));
  task automatic reset_probe;
    @(negedge clk); rst = 1; ar = 0; rv = 0; rid = 0; resp = 0; rl = 1;
    repeat (3) @(negedge clk);
    if (ready || err || av || rr) $fatal(1, "reset did not clear probe");
    rst = 0;
  endtask
  task automatic address(input int pc);
    repeat (5) begin
      @(negedge clk);
      if (!av || aa != 64'h4700000000 + (64'(pc) << 30) || ready)
        $fatal(1, "probe address/valid changed while stalled");
    end
    ar = 1;
    @(negedge clk); ar = 0;
    if (av || !rr) $fatal(1, "probe did not wait for response");
  endtask
  task automatic response(input int bad);
    repeat (5) @(negedge clk);
    rid = bad == 1; resp = bad == 2 ? 2'b10 : 2'b00; rl = bad != 3; rv = 1;
    @(negedge clk); rv = 0;
    if (bad == 3) begin
      if (!err || ready || !rr || av) $fatal(1, "missing RLAST must drain and fail");
      repeat (3) @(negedge clk);
      rl = 1; rv = 1;
      @(negedge clk); rv = 0;
    end
  endtask
  initial begin
    reset_probe();
    for (int pc = 0; pc < 4; pc++) begin address(pc); response(0); end
    if (!ready || err || av || rr) $fatal(1, "four good PCs did not enable memory");
    for (int bad = 1; bad <= 3; bad++) begin
      reset_probe(); address(0); response(bad);
      repeat (20) @(negedge clk);
      if (ready || !err || av || rr) $fatal(1, "bad response was not sticky");
      // Even after three good PCs, a bad final response must never set ready.
      reset_probe();
      for (int pc = 0; pc < 3; pc++) begin address(pc); response(0); end
      address(3); response(bad);
      repeat (20) @(negedge clk);
      if (ready || !err || av || rr) $fatal(1, "bad final PC enabled memory");
    end
    reset_probe(); address(0);
    repeat (100) @(negedge clk);
    if (ready || err || av || !rr) $fatal(1, "no response must leave memory unavailable");
    // Reset while R is outstanding, and then while AR is held by backpressure.
    reset_probe();
    repeat (5) @(negedge clk);
    reset_probe();
    for (int pc = 0; pc < 4; pc++) begin address(pc); response(0); end
    if (!ready || err) $fatal(1, "reset retry failed");
    $display("PASS v80 ready"); $finish;
  end
  initial begin #10000000; $fatal(1, "timeout"); end
endmodule
