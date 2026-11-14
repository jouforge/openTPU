// Registered boundary for HMM/HCVT synthesis and timing. This includes the
// instruction controller and converters, but not the slice's TMEM arbiter.
module otpu_hmxu_ooc
  import otpu_pkg::*;
#(parameter int N = 8) (
  input  logic clk,
  input  logic rst,
  input  logic start,
  input  cmd_t cmd,
  output logic rdy,
  output logic done,
  output logic [N-1:0] r_req,
  output logic [N-1:0][31:0] r_addr,
  input  logic r_ready,
  input  logic r_valid,
  input  logic [N-1:0][31:0] r_data,
  output logic [N-1:0] w_req,
  output logic [N-1:0][31:0] w_addr,
  output logic [N-1:0][31:0] w_data,
  input  logic w_ready
);
  logic rst_q, start_q, rr_q, rv_q, wr_q, rdy_d, done_d;
  cmd_t cmd_q;
  logic [N-1:0] r_req_d, w_req_d;
  logic [N-1:0][31:0] r_data_q, r_addr_d, w_addr_d, w_data_d;
  always_ff @(posedge clk) begin
    rst_q <= rst; start_q <= start; cmd_q <= cmd;
    rr_q <= r_ready; rv_q <= r_valid; wr_q <= w_ready; r_data_q <= r_data;
    rdy <= rdy_d; done <= done_d;
    r_req <= r_req_d; r_addr <= r_addr_d;
    w_req <= w_req_d; w_addr <= w_addr_d; w_data <= w_data_d;
  end
  otpu_hmxu #(.N(N)) u_half (
    .clk, .rst(rst_q), .start(start_q), .cmd(cmd_q), .rdy(rdy_d), .done(done_d),
    .r_req(r_req_d), .r_addr(r_addr_d), .r_ready(rr_q), .r_valid(rv_q), .r_data(r_data_q),
    .w_req(w_req_d), .w_addr(w_addr_d), .w_data(w_data_d), .w_ready(wr_q)
  );
endmodule
