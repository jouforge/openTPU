// HMM activation reuse across refill/tail boundaries, with delayed reads,
// independent backpressure and reset cancellation. No out-of-row prefetches.
module tb_hmm;
  import otpu_pkg::*;
  import otpu_fp::*;
  parameter int N = 8;
  localparam int K = 2*N+1, NC = N+3, ARS = N+3, BRS = (NC+1)/2+1, ORS = NC+2;
  logic clk = 0, rst = 1, start = 0, rdy, done;
  always #5 clk = ~clk;
  cmd_t cmd;
  logic [N-1:0] r_req, w_req;
  logic [N-1:0][31:0] r_addr, r_data, w_addr, w_data;
  logic r_ready, r_valid = 0, w_ready;
  logic [31:0] mem [1024];
  logic [N-1:0][31:0] pending, held_addr, held_data, held_raddr;
  logic [N-1:0] held_req, held_rreq;
  bit held = 0, held_r = 0;
  int cycle = 0, delay_left = 0, a_reads = 0;
  assign r_ready = !rst && delay_left == 0 && !r_valid && cycle % 5 == 0;
  assign w_ready = !rst && cycle % 11 == 0;
  otpu_hmxu #(.N(N), .W4A16(1'b1)) dut (.*);

  always @(posedge clk) begin
    cycle <= cycle + 1;
    r_valid <= 0;
    if (rst) begin
      delay_left <= 0; held <= 0; held_r <= 0; a_reads <= 0;
      if (|r_req || |w_req || rdy) $fatal(1, "request during reset");
    end else begin
      if (held && (held_req != w_req || held_addr != w_addr || held_data != w_data))
        $fatal(1, "write changed while stalled");
      if (held_r && (held_rreq != r_req || held_raddr != r_addr))
        $fatal(1, "read changed while stalled");
      held <= |w_req && !w_ready;
      held_req <= w_req; held_addr <= w_addr; held_data <= w_data;
      held_r <= |r_req && !r_ready;
      held_rreq <= r_req; held_raddr <= r_addr;
      if (delay_left > 0) begin
        delay_left <= delay_left - 1;
        if (delay_left == 1) begin r_data <= pending; r_valid <= 1; end
      end
      if (|r_req && r_ready) begin
        if (r_addr[0] < 64) a_reads <= a_reads + 1;
        for (int j = 0; j < N; j++) begin
          if (r_req[j] && r_addr[j] < 64 &&
              !(r_addr[j] <= N || (r_addr[j] >= ARS && r_addr[j] <= ARS+N)))
            $fatal(1, "read beyond activation row");
          pending[j] <= r_req[j] ? mem[r_addr[j]] : 32'h7E007E00;
        end
        delay_left <= 3;
      end
      for (int j = 0; j < N; j++) if (w_req[j] && w_ready) mem[w_addr[j]] <= w_data[j];
    end
  end

  task automatic launch();
    @(negedge clk);
    if (!rdy) $fatal(1, "not ready after reset/completion");
    start = 1;
    @(negedge clk); start = 0;
  endtask
  task automatic reset_unit();
    @(negedge clk); rst = 1;
    repeat (2) @(negedge clk);
    if (done) $fatal(1, "completion survived reset");
    rst = 0;
    @(negedge clk);
  endtask
  task automatic initialize(input bit poison);
    for (int r = 0; r < 2; r++) begin
      for (int j = 0; j <= N; j++)
        mem[r*ARS+j] = poison ? 32'h7E007E00 : r == 0 ? 32'h3C004000 : 32'h4000BC00;
      mem[r*ARS+N][31:16] = 16'h7E00;
      for (int j = 0; j < ORS; j++) mem[768+r*ORS+j] = j < NC ? F_ONE : 32'hDEADBEEF;
    end
  endtask
  initial begin
    cmd = '0; cmd.op = OP_HMM; cmd.flags = 2;
    cmd.w1 = 768; cmd.w2 = 0; cmd.w3 = 64;
    cmd.w4 = (NC << 16) | 2; cmd.w5 = (ARS << 16) | K; cmd.w6 = (BRS << 16) | ORS;
    for (int i = 0; i < 1024; i++) mem[i] = 32'hDEADBEEF;
    for (int r = 0; r < K; r++) begin
      for (int j = 0; j < (NC+1)/2; j++) mem[64+r*BRS+j] = 32'h3C003C00;
      mem[64+r*BRS+(NC-1)/2][31:16] = 16'h7E00;
    end
    reset_unit();
    for (int cut = 0; cut < 100; cut++) begin
      initialize(1); launch();
      repeat ((cut*29) % 1024) @(negedge clk);
      reset_unit();
      initialize(0); launch();
      wait (done);
      @(negedge clk);
      for (int r = 0; r < 2; r++)
        for (int j = 0; j < ORS; j++)
          if (mem[768+r*ORS+j] != (j < NC ? i2f(r == 0 ? 3*N+3 : N) : 32'hDEADBEEF))
            $fatal(1, "restart mismatch cut=%0d row=%0d col=%0d got=%h", cut, r, j, mem[768+r*ORS+j]);
      // Two A blocks per output tile; the second holds only the last low half.
      if (a_reads != 2*((NC+N-1)/N)*2) $fatal(1, "activation reuse: reads=%0d", a_reads);
    end
    $display("HMM reset/restart cases=100 errors=0");
    $finish;
  end
  initial begin #10000000; $fatal(1, "timeout"); end
endmodule
