// HDQ reset/restart with read latency and independent read/write backpressure.
module tb_hdq;
  import otpu_pkg::*;
  parameter int N = 8;
  logic clk = 0, rst = 1, start = 0, rdy, done;
  always #5 clk = ~clk;
  cmd_t cmd;
  logic [N-1:0] r_req, w_req;
  logic [N-1:0][31:0] r_addr, r_data, w_addr, w_data;
  logic r_ready, r_valid = 0, w_ready;
  logic [31:0] mem [256];
  logic [N-1:0][31:0] pending, held_addr, held_data;
  logic [N-1:0] held_req;
  bit held = 0;
  int cycle = 0, delay_left = 0;
  assign r_ready = !rst && delay_left == 0 && !r_valid && cycle % 7 == 0;
  assign w_ready = !rst && cycle % 11 == 0;
  otpu_hmxu #(.N(N), .W4A16(1'b1)) dut (.*);
  always @(posedge clk) begin
    cycle <= cycle + 1;
    r_valid <= 0;
    if (rst) begin
      delay_left <= 0;
      held <= 0;
      if (|r_req || |w_req || rdy) $fatal(1, "request during reset");
    end else begin
      if (held && (held_req != w_req || held_addr != w_addr || held_data != w_data))
        $fatal(1, "write changed while stalled");
      held <= |w_req && !w_ready;
      held_req <= w_req; held_addr <= w_addr; held_data <= w_data;
      if (delay_left > 0) begin
        delay_left <= delay_left - 1;
        if (delay_left == 1) begin r_data <= pending; r_valid <= 1; end
      end
      if (|r_req && r_ready) begin
        for (int j = 0; j < N; j++) pending[j] <= r_req[j] ? mem[r_addr[j]] : 0;
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
  initial begin
    cmd = '0; cmd.op = OP_HDQ; cmd.w1 = 128; cmd.w2 = 0; cmd.w3 = 32; cmd.w4 = 65;
    // Code 1 times half 1; odd last output's padding must be zero.
    for (int i = 0; i < 256; i++) mem[i] = 0;
    for (int i = 0; i < 9; i++) mem[i] = 32'h11111111;
    for (int i = 0; i < 33; i++) mem[32+i] = 32'h3C003C00;
    reset_unit();
    // Different cancel offsets sweep request, delayed-response, multiply and
    // blocked-write states. Each cancellation is followed by a complete command.
    for (int cut = 0; cut < 100; cut++) begin
      launch();
      repeat (cut) @(negedge clk);
      reset_unit();
      for (int i = 0; i < 33; i++) mem[128+i] = 32'hDEADBEEF;
      launch();
      wait (done);
      @(negedge clk);
      for (int i = 0; i < 33; i++)
        if (mem[128+i] != (i == 32 ? 32'h00003C00 : 32'h3C003C00))
          $fatal(1, "restart mismatch cut=%0d word=%0d got=%h", cut, i, mem[128+i]);
    end
    $display("HDQ reset/restart cases=100 errors=0");
    $finish;
  end
  initial begin #2000000; $fatal(1, "timeout"); end
endmodule
