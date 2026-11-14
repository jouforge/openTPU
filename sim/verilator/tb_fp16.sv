// Binary16 conversions and native half-product/fp32-add pipelines (docs/fp16.md).
// Vectors: op a b c expected, hexadecimal. Payload registers have no reset; only
// compare after the documented number of enabled edges. Stalls freeze every stage.
module tb_fp16
  import otpu_fp::*;
#(parameter int LM = 2, parameter int LA = 4);
  logic clk = 1'b0;
  logic en = 1'b0;
  always #5 clk = ~clk;
  f16_t a, b;
  f32_t c, prod, sum, old_prod, old_sum, x, y, z, expected, got;
  int fd, n, op, cases, errors;
  string fname;
  otpu_hmul #(.LAT(LM)) u_mul (.clk, .en, .a, .b, .y(prod));
  otpu_hmadd #(.LM(LM), .LA(LA)) u_mac (.clk, .en, .a, .b, .c, .y(sum));

  task automatic check(input f32_t value, input f32_t want);
    if (value !== want) begin
      if (errors < 20)
        $display("FP16 mismatch case=%0d op=%0d a=%08h b=%08h c=%08h want=%08h got=%08h",
                 cases, op, x, y, z, want, value);
      errors++;
    end
  endtask

  initial begin
    if (!$value$plusargs("vec=%s", fname)) $fatal(1, "need +vec=");
    fd = $fopen(fname, "r");
    if (fd == 0) $fatal(1, "cannot open %s", fname);
    cases = 0; errors = 0;
    while (!$feof(fd)) begin
      n = $fscanf(fd, "%h %h %h %h %h\n", op, x, y, z, expected);
      if (n != 5) break;
      case (op)
        0: check(h2f(x[15:0]), expected);
        1: check({16'd0, f2h(x)}, expected);
        2, 3: begin
          if (op == 2) check(fp_hmul_s2(fp_hmul_s1(x[15:0], y[15:0])), expected);
          for (int j = 0; j < LM + LA; j++) begin
            @(negedge clk);
            if (j == 0) begin a = x[15:0]; b = y[15:0]; c = z; end
            else begin a = 16'(cases + j); b = 16'h3C00; c = F_ONE; end
            if ((cases + j) % 3 == 0) begin
              en = 1'b0; old_prod = prod; old_sum = sum;
              @(posedge clk); #1;
              check(prod, old_prod); check(sum, old_sum);
              @(negedge clk);
            end
            en = 1'b1;
            @(posedge clk); #1;
            if (op == 2 && j == LM - 1) check(prod, expected);
            if (op == 3 && j == LM + LA - 1) check(sum, expected);
          end
        end
        default: $fatal(1, "unknown op %0d", op);
      endcase
      cases++;
    end
    $fclose(fd);
    $display("FP16 cases=%0d errors=%0d", cases, errors);
    if (errors != 0) $fatal(1, "FP16 mismatch");
    $finish;
  end
endmodule
