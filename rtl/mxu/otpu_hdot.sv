// Native binary16 dot products, N independent outputs (docs/fp16.md). Each accepted
// beat supplies one activation a and N weights b, in increasing K order. first
// starts each output from seed; last returns the accumulated FP32 values. Products
// are exact FP32, with one RNE/FTZ FP32 addition per K. No quantization or scale rows.
//
// Multiplication overlaps the previous addition: in_ready returns LA+1 clocks
// after an accepted non-final beat. The accumulator is selected when the product
// reaches the adder, after the preceding sum commits. No reduction is reordered.
// The arithmetic payload is not reset; the validity pipe
// and row/output ownership are. rst cancels a partial row or held result. A result
// remains valid and stable until out_ready, and blocks the next row meanwhile.
// The caller must mark the first beat of each row and supply at least one beat.
module otpu_hdot
  import otpu_fp::*;
#(parameter int N = 8, parameter int LM = 2, parameter int LA = 4) (
  input  logic clk,
  input  logic rst,
  input  logic in_valid,
  output logic in_ready,
  input  logic first,
  input  logic last,
  input  f16_t a,
  input  f16_t b [N],
  input  f32_t seed [N],
  output logic out_valid,
  input  logic out_ready,
  output f32_t y [N]
);
  localparam int LAT = LM + LA;
  initial if (N < 1 || LM < 2 || LA < 4) $fatal(1, "otpu_hdot: N >= 1, LM >= 2, LA >= 4");
  logic active, final_;
  logic [LAT-1:0] valid, lasts;
  logic [LM-1:0] firsts;
  f32_t acc [N], sum [N], seed_q [N];
  wire take = in_valid && in_ready;
  assign in_ready = !rst && !(|valid[LA-1:0]) && !final_ && !out_valid;
  for (genvar j = 0; j < N; j++) begin : g_col
    f32_t c, prod;
    assign c = firsts[LM-1] ? seed_q[j] : acc[j];
    otpu_hmul #(.LAT(LM)) u_mul (.clk, .en(1'b1), .a, .b(b[j]), .y(prod));
    otpu_fadd #(.LAT(LA)) u_add (.clk, .en(1'b1), .a(prod), .b(c), .y(sum[j]));
  end

  always_ff @(posedge clk) begin
    if (rst) begin
      active <= 1'b0; final_ <= 1'b0;
      valid <= '0; lasts <= '0; firsts <= '0; out_valid <= 1'b0;
    end else begin
      valid <= {valid[LAT-2:0], take};
      lasts <= {lasts[LAT-2:0], take && last};
      firsts <= {firsts[LM-2:0], take && first};
      if (out_valid && out_ready) begin out_valid <= 1'b0; final_ <= 1'b0; end
      if (take) begin
        active <= 1'b1; final_ <= last;
        if (first) for (int j = 0; j < N; j++) seed_q[j] <= seed[j];
      end
      if (valid[LAT-1]) begin
        for (int j = 0; j < N; j++) acc[j] <= sum[j];
        if (lasts[LAT-1]) begin
          for (int j = 0; j < N; j++) y[j] <= sum[j];
          active <= 1'b0; out_valid <= 1'b1;
        end
      end
    end
  end
`ifndef SYNTHESIS
  always_ff @(posedge clk) if (!rst && take) begin
    if (!active && !first) $fatal(1, "otpu_hdot: a new row needs first");
    if (active && first) $fatal(1, "otpu_hdot: first in an unfinished row");
  end
`endif
endmodule
