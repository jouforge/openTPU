// Pipelined fp32 operators built from the staged functions of otpu_fp. Every register advances
// with `en`: a unit freezes as a whole while its TMEM grant is withheld.
//
//   otpu_fmul    y = a * b                LAT >= 2 (s1 | s2, extra cycles appended)
//   otpu_fadd    y = a + b                LAT >= 4 (s1 | s2 | s3 | s4, extra cycles appended)
//   otpu_fmadd   y = (a * b) + c          two roundings, as in the ISA; LAT = LM + LA
//   otpu_hmul    y = half(a) * half(b)   exact fp32 product; LAT >= 2
//   otpu_hmadd   y = half(a) * half(b) + fp32(c), one addition rounding; LAT = LM + LA
//   otpu_fmma    y = (a * b) + (c * e)    three roundings; with e = 1.0 it is otpu_fmadd
//                                         (c * 1.0 == c bit for bit after the adder's flush)
//   otpu_delay   N-cycle delay line (SRL-friendly: no reset)
//   otpu_vdelay  N-cycle delay line whose every stage resets: the valids beside an otpu_delay's
//                payload

module otpu_delay #(parameter int W = 32, parameter int N = 1) (
  input  logic         clk,
  input  logic         en,
  input  logic [W-1:0] d,
  output logic [W-1:0] q
);
  // The last stage is a flip-flop, not part of a shift-register LUT: an SRL's clock-to-out is
  // ~1.5 ns, too slow in front of the multipliers and adders these lines feed.
  if (N == 0) begin : g_wire
    assign q = d;
  end else if (N == 1) begin : g_one
    (* shreg_extract = "no", keep *) logic [W-1:0] qr;
    always_ff @(posedge clk) if (en) qr <= d;
    assign q = qr;
  end else begin : g_regs
    logic [W-1:0] r [N - 1];
    (* shreg_extract = "no", keep *) logic [W-1:0] qr;
    always_ff @(posedge clk) if (en) begin
      r[0] <= d;
      for (int k = 1; k < N - 1; k++) r[k] <= r[k-1];
      qr <= r[N-2];
    end
    assign q = qr;
  end
endmodule

// A line's valids (control: docs/asic.md "Power-up state"). An otpu_delay that advances only with
// its unit's enable keeps its power-up stages, or a stopped run's, until the unit's next
// operation, where a stray valid writes or counts; here every stage resets (synchronous, to 0:
// the R pin, so flip-flops, not a shift-register LUT). The payload beside it stays in an
// otpu_delay.
module otpu_vdelay #(parameter int W = 1, parameter int N = 1) (
  input  logic         clk,
  input  logic         rst,
  input  logic         en,
  input  logic [W-1:0] d,
  output logic [W-1:0] q
);
  if (N == 0) begin : g_wire
    assign q = d;
  end else begin : g_regs
    logic [N-1:0][W-1:0] vq;
    always_ff @(posedge clk)
      if (rst) vq <= '0;
      else if (en) begin
        vq[0] <= d;
        for (int k = 1; k < N; k++) vq[k] <= vq[k-1];
      end
    assign q = vq[N-1];
  end
endmodule

module otpu_fmul
  import otpu_fp::*;
#(parameter int LAT = 2) (
  input  logic  clk,
  input  logic  en,
  input  f32_t  a,
  input  f32_t  b,
  output f32_t  y
);
  fmul_mid_t m;
  f32_t r;
  always_ff @(posedge clk) if (en) begin
    m <= fp_mul_s1(a, b);
    r <= fp_mul_s2(m);
  end
  otpu_delay #(.W(32), .N(LAT - 2)) u_pad (.clk, .en, .d(r), .q(y));
endmodule

module otpu_hmul
  import otpu_fp::*;
#(parameter int LAT = 2) (
  input  logic clk,
  input  logic en,
  input  f16_t a,
  input  f16_t b,
  output f32_t y
);
  initial if (LAT < 2) $fatal(1, "otpu_hmul: LAT must be >= 2");
  hmul_mid_t m;
  f32_t r;
  always_ff @(posedge clk) if (en) begin
    m <= fp_hmul_s1(a, b);
    r <= fp_hmul_s2(m);
  end
  otpu_delay #(.W(32), .N(LAT - 2)) u_pad (.clk, .en, .d(r), .q(y));
endmodule

module otpu_hmadd
  import otpu_fp::*;
#(parameter int LM = 2, parameter int LA = 4) (
  input  logic clk,
  input  logic en,
  input  f16_t a,
  input  f16_t b,
  input  f32_t c,
  output f32_t y
);
  initial if (LM < 2 || LA < 4) $fatal(1, "otpu_hmadd: LM >= 2, LA >= 4");
  f32_t p, cd;
  otpu_hmul #(.LAT(LM)) u_mul (.clk, .en, .a, .b, .y(p));
  otpu_delay #(.W(32), .N(LM)) u_c (.clk, .en, .d(c), .q(cd));
  otpu_fadd #(.LAT(LA)) u_add (.clk, .en, .a(p), .b(cd), .y);
endmodule

module otpu_fadd
  import otpu_fp::*;
#(parameter int LAT = 4) (
  input  logic  clk,
  input  logic  en,
  input  f32_t  a,
  input  f32_t  b,
  output f32_t  y
);
  fadd_p1_t s1;
  fadd_p2_t s2;
  fadd_nm_t s3;
  f32_t r;
  always_ff @(posedge clk) if (en) begin
    s1 <= fp_add_s1(a, b);
    s2 <= fp_add_s2(s1);
    s3 <= fp_add_s3(s2);
    r  <= fp_add_s4(s3);
  end
  otpu_delay #(.W(32), .N(LAT - 4)) u_pad (.clk, .en, .d(r), .q(y));
endmodule

// otpu_fadd (LAT 4) with a second copy of its result register (yo), kept apart from y's: for a
// load far from the others
module otpu_fadd2
  import otpu_fp::*;
(
  input  logic  clk,
  input  logic  en,
  input  f32_t  a,
  input  f32_t  b,
  output f32_t  y,
  output f32_t  yo
);
  fadd_p1_t s1;
  fadd_p2_t s2;
  fadd_nm_t s3;
  f32_t r;
  (* keep *) f32_t ro;
  always_ff @(posedge clk) if (en) begin
    s1 <= fp_add_s1(a, b);
    s2 <= fp_add_s2(s1);
    s3 <= fp_add_s3(s2);
    r  <= fp_add_s4(s3);
    ro <= fp_add_s4(s3);
  end
  assign y = r;
  assign yo = ro;
endmodule

module otpu_fmadd
  import otpu_fp::*;
#(parameter int LM = 2, parameter int LA = 4) (
  input  logic  clk,
  input  logic  en,
  input  f32_t  a,
  input  f32_t  b,
  input  f32_t  c,
  output f32_t  y
);
  f32_t p, cd;
  otpu_fmul #(.LAT(LM)) u_m (.clk, .en, .a, .b, .y(p));
  otpu_delay #(.W(32), .N(LM)) u_c (.clk, .en, .d(c), .q(cd));
  otpu_fadd #(.LAT(LA)) u_a (.clk, .en, .a(p), .b(cd), .y);
endmodule

module otpu_fmma
  import otpu_fp::*;
#(parameter int LM = 2, parameter int LA = 4) (
  input  logic  clk,
  input  logic  en,
  input  f32_t  a,
  input  f32_t  b,
  input  f32_t  c,
  input  f32_t  e,
  output f32_t  y
);
  f32_t p, q;
  otpu_fmul #(.LAT(LM)) u_m (.clk, .en, .a, .b, .y(p));
  otpu_fmul #(.LAT(LM)) u_c (.clk, .en, .a(c), .b(e), .y(q));
  otpu_fadd #(.LAT(LA)) u_a (.clk, .en, .a(p), .b(q), .y);
endmodule
