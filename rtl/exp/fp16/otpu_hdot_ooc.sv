// Registered boundary for the isolated dot-product implementation check. Only
// the paths outside these registers are excluded; every DUT path is timed.
module otpu_hdot_ooc
  import otpu_fp::*;
#(parameter int N = 8) (
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
  logic rst_q, valid_q, first_q, last_q, ready_q;
  logic in_ready_d, out_valid_d;
  f16_t a_q, b_q [N];
  f32_t seed_q [N], y_d [N];
  always_ff @(posedge clk) begin
    rst_q <= rst; valid_q <= in_valid;
    first_q <= first; last_q <= last; ready_q <= out_ready;
    a_q <= a;
    for (int j = 0; j < N; j++) begin
      b_q[j] <= b[j]; seed_q[j] <= seed[j]; y[j] <= y_d[j];
    end
    in_ready <= in_ready_d; out_valid <= out_valid_d;
  end
  otpu_hdot #(.N(N)) u_dot (
    .clk, .rst(rst_q), .in_valid(valid_q), .in_ready(in_ready_d),
    .first(first_q), .last(last_q), .a(a_q), .b(b_q), .seed(seed_q),
    .out_valid(out_valid_d), .out_ready(ready_q), .y(y_d)
  );
endmodule
