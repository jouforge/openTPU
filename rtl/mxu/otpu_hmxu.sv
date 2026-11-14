// HMM / HCVT / optional HDQ execution on packed TMEM operands (docs/fp16.md,
// docs/mixed_precision.md). One command and
// one memory read in flight. The enclosing slice arbitrates complete read/write
// beats; responses may be delayed, and writes remain stable until accepted.
// Reads of a packed row start at its low halfword; odd padding is never used.
module otpu_hmxu
  import otpu_pkg::*;
  import otpu_fp::*;
#(parameter int N = 8, parameter bit W4A16 = 1'b0) (
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
  initial if (N < 2 || (N & (N - 1)) != 0) $fatal(1, "otpu_hmxu: N must be a power of two >= 2");
  typedef enum logic [4:0] {
    IDLE, INIT, SEED_R, SEED_D, A_R, A_D, B_R, B_D, DOT, RESULT, WRITE, CVT_R, CVT_D,
    DQ_R, DQ_D, SC_R, SC_D, DQ_M
  } state_t;
  state_t st;
  logic cvt, unpack, acc, deq;
  logic [31:0] dst, abase, bbase, m, n, k, ars, brs, ors;
  logic [31:0] row, col, kk, idx, cm, nw;
  f16_t a, b [N];
  f32_t seed [N], y [N];
  logic [N-1:0][31:0] data_q;
  logic [N-1:0][3:0] code_q;
  hmul_mid_t dq_mid [N];
  logic di_ready, do_valid;
  // INT4 is exact in half. Keep the integer sign and multiply the half scale
  // before a single final half rounding; -8 is defined although packing emits -7..7.
  function automatic f16_t i4h(input logic [3:0] q);
    logic [3:0] mag;
    logic [14:0] h;
    mag = q[3] ? 4'd0 - q : q;
    case (mag)
      1: h = 15'h3C00; 2: h = 15'h4000; 3: h = 15'h4200; 4: h = 15'h4400;
      5: h = 15'h4500; 6: h = 15'h4600; 7: h = 15'h4700; 8: h = 15'h4800;
      default: h = '0;
    endcase
    return {q[3], h};
  endfunction
  assign rdy = st == IDLE && !rst;
  always_comb begin
    cm = cvt ? n - idx : n - col;
    if (cm > N) cm = N;
    nw = cvt && !unpack ? (cm + 1) >> 1 : cm;
    r_req = '0; r_addr = '0;
    case (st)
      SEED_R: for (int j = 0; j < N; j++) begin
        r_req[j] = 32'(j) < cm; r_addr[j] = dst + col + 32'(j);
      end
      A_R: begin r_req[0] = 1'b1; r_addr[0] = abase + (kk >> 1); end
      B_R: for (int j = 0; j < N; j++) begin
        r_req[j] = 32'(j) < ((cm + 1) >> 1);
        r_addr[j] = bbase + (col >> 1) + 32'(j);
      end
      CVT_R: for (int j = 0; j < N; j++) begin
        r_req[j] = 32'(j) < (unpack ? (cm + 1) >> 1 : cm);
        r_addr[j] = abase + (unpack ? idx >> 1 : idx) + 32'(j);
      end
      DQ_R: for (int j = 0; j < N; j++) begin
        r_req[j] = 32'(j) < ((32'(idx[2:0]) + cm + 7) >> 3);
        r_addr[j] = abase + (idx >> 3) + 32'(j);
      end
      SC_R: for (int j = 0; j < N; j++) begin
        r_req[j] = 32'(j) < ((cm + 1) >> 1);
        r_addr[j] = bbase + (idx >> 1) + 32'(j);
      end
      default: ;
    endcase
    w_req = '0; w_addr = '0; w_data = data_q;
    for (int j = 0; j < N; j++) begin
      w_req[j] = st == WRITE && 32'(j) < nw && !rst;
      w_addr[j] = dst + (cvt ? (unpack ? idx : idx >> 1) : col) + 32'(j);
    end
    if (rst) r_req = '0;
  end
  otpu_hdot #(.N(N)) u_dot (
    .clk, .rst, .in_valid(st == DOT && !rst), .in_ready(di_ready),
    .first(kk == 0), .last(kk + 1 == k), .a, .b, .seed,
    .out_valid(do_valid), .out_ready(st == RESULT && !rst), .y
  );

  // B's K-row address advances by brs. The original base is retained for the
  // next output tile; A and dst advance only after every column of their row.
  logic [31:0] bfirst;
  always_ff @(posedge clk) begin
    if (rst) begin st <= IDLE; done <= 1'b0; end
    else begin
      done <= 1'b0;
      case (st)
        IDLE: if (start) begin
          cvt <= cmd.op == OP_HCVT || (W4A16 && cmd.op == OP_HDQ);
          deq <= W4A16 && cmd.op == OP_HDQ;
          unpack <= cmd.op == OP_HCVT && cmd.flags[0]; acc <= cmd.flags[1];
          dst <= cmd.w1; abase <= cmd.w2; bbase <= cmd.w3; bfirst <= cmd.w3;
          m <= 32'(cmd.w4[15:0]);
          n <= cmd.op == OP_HDQ ? cmd.w4 : cmd.op == OP_HCVT ? cmd.w3 : 32'(cmd.w4[31:16]);
          k <= 32'(cmd.w5[15:0]); ars <= 32'(cmd.w5[31:16]);
          brs <= 32'(cmd.w6[31:16]); ors <= 32'(cmd.w6[15:0]);
          row <= 0; col <= 0; kk <= 0; idx <= 0; st <= INIT;
        end
        INIT: begin
          if (n == 0 || (!cvt && m == 0)) begin st <= IDLE; done <= 1'b1; end
          else if (cvt) st <= deq ? DQ_R : CVT_R;
          else begin
            kk <= 0; bbase <= bfirst;
            for (int j = 0; j < N; j++) begin seed[j] <= F_ZERO; data_q[j] <= F_ZERO; end
            st <= acc ? SEED_R : k == 0 ? WRITE : A_R;
          end
        end
        SEED_R: if (r_ready) st <= SEED_D;
        SEED_D: if (r_valid) begin
          for (int j = 0; j < N; j++) begin
            seed[j] <= 32'(j) < cm ? r_data[j] : F_ZERO;
            data_q[j] <= r_data[j];
          end
          st <= k == 0 ? WRITE : A_R;
        end
        A_R: if (r_ready) st <= A_D;
        A_D: if (r_valid) begin a <= kk[0] ? r_data[0][31:16] : r_data[0][15:0]; st <= B_R; end
        B_R: if (r_ready) st <= B_D;
        B_D: if (r_valid) begin
          for (int j = 0; j < N; j++)
            b[j] <= 32'(j) < cm ? (j % 2 == 0 ? r_data[j/2][15:0] : r_data[j/2][31:16]) : 16'd0;
          st <= DOT;
        end
        DOT: if (di_ready) begin
          if (kk + 1 == k) st <= RESULT;
          else begin kk <= kk + 1; bbase <= bbase + brs; st <= A_R; end
        end
        RESULT: if (do_valid) begin
          for (int j = 0; j < N; j++) data_q[j] <= y[j];
          st <= WRITE;
        end
        CVT_R: if (r_ready) st <= CVT_D;
        CVT_D: if (r_valid) begin
          for (int j = 0; j < N; j++) begin
            if (unpack) data_q[j] <= h2f(j % 2 == 0 ? r_data[j/2][15:0] : r_data[j/2][31:16]);
            else if (j < N / 2)
              data_q[j] <= {32'(2*j+1) < cm ? f2h(r_data[2*j+1]) : 16'd0, f2h(r_data[2*j])};
          end
          st <= WRITE;
        end
        DQ_R: if (r_ready) st <= DQ_D;
        DQ_D: if (r_valid) begin
          for (int j = 0; j < N; j++)
            code_q[j] <= 4'(r_data[(int'(idx[2:0]) + j) / 8] >> (4 * ((int'(idx[2:0]) + j) % 8)));
          st <= SC_R;
        end
        SC_R: if (r_ready) st <= SC_D;
        SC_D: if (r_valid) begin
          for (int j = 0; j < N; j++)
            dq_mid[j] <= fp_hmul_s1(i4h(code_q[j]), j % 2 == 0 ? r_data[j/2][15:0] : r_data[j/2][31:16]);
          st <= DQ_M;
        end
        DQ_M: begin
          for (int j = 0; j < N / 2; j++)
            data_q[j] <= {32'(2*j+1) < cm ? f2h(fp_hmul_s2(dq_mid[2*j+1])) : 16'd0,
                         f2h(fp_hmul_s2(dq_mid[2*j]))};
          st <= WRITE;
        end
        WRITE: if (w_ready) begin
          if (cvt) begin
            if (idx + N >= n) begin st <= IDLE; done <= 1'b1; end
            else begin idx <= idx + N; st <= deq ? DQ_R : CVT_R; end
          end else if (col + N < n) begin col <= col + N; st <= INIT; end
          else if (row + 1 < m) begin
            row <= row + 1; col <= 0; abase <= abase + ars; dst <= dst + ors; st <= INIT;
          end else begin st <= IDLE; done <= 1'b1; end
        end
        default: st <= IDLE;
      endcase
    end
  end
endmodule
