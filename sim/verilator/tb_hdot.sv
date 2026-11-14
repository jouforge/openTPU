// Stream matrix rows through the FP16 dot-product array, with input gaps, held
// outputs and reset cancellation at every arithmetic pipeline position.
module tb_hdot
  import otpu_fp::*;
#(parameter int N = 8, parameter int LM = 2, parameter int LA = 4);
  localparam int LAT = LM + LA;
  logic clk = 1'b0;
  always #5 clk = ~clk;
  logic rst = 1'b1, in_valid = 1'b0, in_ready, first = 1'b0, last = 1'b0;
  logic out_valid, out_ready = 1'b0;
  f16_t a, b [N];
  f32_t seed [N], y [N], expected [N], held [N];
  int fd, got, nfirst, nlast, av, word_, cases = 0, rows = 0, errors = 0, waited;
  string fname;
  otpu_hdot #(.N(N), .LM(LM), .LA(LA)) u_dut (.*);

  task automatic check(input f32_t value, input f32_t want);
    if (value !== want) begin
      if (errors < 20) $display("HDOT case=%0d want=%08h got=%08h", cases, want, value);
      errors++;
    end
  endtask

  task automatic tick;
    @(posedge clk); #1;
  endtask

  initial begin
    tick(); @(negedge clk); rst = 1'b0;
    // The next multiplication may start before the preceding sum commits.
    // Check the feedback interval and cancel with two accepted beats in flight.
    for (int pos = 0; pos <= LAT + 2; pos++) begin
      @(negedge clk);
      in_valid = 1'b1; first = 1'b1; last = 1'b0; a = 16'h3C00;
      for (int j = 0; j < N; j++) begin b[j] = 16'h3C00; seed[j] = F_ZERO; end
      tick(); @(negedge clk); in_valid = 1'b0;
      repeat (LA - 1) begin tick(); check({31'd0, in_ready}, F_ZERO); end
      tick(); check({31'd0, in_ready}, 32'd1);
      @(negedge clk); in_valid = 1'b1; first = 1'b0; last = 1'b1;
      tick(); @(negedge clk); in_valid = 1'b0;
      repeat (pos) tick();
      @(negedge clk); rst = 1'b1; tick();
      check({31'd0, out_valid}, F_ZERO);
      @(negedge clk); rst = 1'b0;
      repeat (LAT + 2) begin tick(); check({31'd0, out_valid}, F_ZERO); end
    end
    // Cancel an unfinished row or a final beat at every point, including a
    // completed result held by out_ready=0. A fresh first must work after reset.
    for (int final_ = 0; final_ < 2; final_++)
    for (int pos = 0; pos <= LAT + 2; pos++) begin
      @(negedge clk);
      in_valid = 1'b1; first = 1'b1; last = 1'(final_); a = 16'h3C00;
      for (int j = 0; j < N; j++) begin b[j] = 16'h3C00; seed[j] = F_ZERO; end
      tick(); @(negedge clk); in_valid = 1'b0;
      repeat (pos) tick();
      @(negedge clk); rst = 1'b1; tick();
      check({31'd0, out_valid}, F_ZERO);
      @(negedge clk); rst = 1'b0;
      repeat (LAT + 2) begin tick(); check({31'd0, out_valid}, F_ZERO); end
    end
    if (!$value$plusargs("vec=%s", fname)) $fatal(1, "need +vec=");
    fd = $fopen(fname, "r");
    if (fd == 0) $fatal(1, "cannot open %s", fname);
    while (!$feof(fd)) begin
      got = $fscanf(fd, "%h %h %h", nfirst, nlast, av);
      if (got != 3) break;
      @(negedge clk);
      in_valid = 1'b0;
      waited = 0;
      while (!in_ready) begin
        tick(); @(negedge clk); waited++;
        if (waited > LAT + 4) $fatal(1, "input stuck");
      end
      // Input bubbles after the unit has become ready.
      repeat (cases % 3) begin tick(); @(negedge clk); end
      a = 16'(av); first = 1'(nfirst); last = 1'(nlast);
      for (int j = 0; j < N; j++) begin
        got = $fscanf(fd, "%h", word_); if (got != 1) $fatal(1, "short weights"); b[j] = 16'(word_);
      end
      for (int j = 0; j < N; j++) begin
        got = $fscanf(fd, "%h", word_); if (got != 1) $fatal(1, "short seed"); seed[j] = 32'(word_);
      end
      for (int j = 0; j < N; j++) begin
        got = $fscanf(fd, "%h", word_); if (got != 1) $fatal(1, "short result"); expected[j] = 32'(word_);
      end
      in_valid = 1'b1; tick(); @(negedge clk); in_valid = 1'b0;
      // Change the payload while busy: the accepted operands/seed alone determine the sum.
      a = 16'h7E00;
      for (int j = 0; j < N; j++) begin b[j] = 16'h7C00; seed[j] = F_NAN; end
      if (last) begin
        waited = 0;
        while (!out_valid) begin tick(); waited++; if (waited > LAT + 2) $fatal(1, "result stuck"); end
        for (int j = 0; j < N; j++) begin check(y[j], expected[j]); held[j] = y[j]; end
        repeat (rows % 5 + 1) begin
          tick();
          check({31'd0, out_valid}, 32'd1);
          check({31'd0, in_ready}, F_ZERO);
          for (int j = 0; j < N; j++) check(y[j], held[j]);
        end
        @(negedge clk); out_ready = 1'b1; tick();
        @(negedge clk); out_ready = 1'b0;
        check({31'd0, out_valid}, F_ZERO);
        rows++;
      end
      cases++;
    end
    $fclose(fd);
    $display("HDOT beats=%0d rows=%0d errors=%0d", cases, rows, errors);
    if (errors != 0) $fatal(1, "HDOT mismatch");
    $finish;
  end
endmodule
