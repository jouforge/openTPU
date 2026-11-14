// The pumped PE's four FIRST/BF variants, including independently stalled weight and
// pre-adder registers. Compare the full 48-bit cascade and low 32-bit packed result
// with rtl/tech/generic on identical stimulus (tools/tech_dsp.py).
`timescale 1ns/1ps
module tb_tech_ppe;
  logic clk_f = 1'b0, ce_w, ce_ad;
  logic [7:0] wlo, whi, act;
  logic [47:0] pcin;
  logic [3:0][47:0] pcout;
  logic [3:0][31:0] p;
  integer seed, cyc;
  for (genvar v = 0; v < 4; v++) begin : g_pe
    otpu_ppe #(.FIRST(v[0]), .BF(v[1])) u_pe (
      .clk_f, .ce_w, .ce_ad, .wlo, .whi, .act, .pcin, .pcout(pcout[v]), .p(p[v]));
  end
  initial begin
    #200;                                      // UNISIM startup GSR
    if (!$value$plusargs("seed=%d", seed)) seed = 1;
    for (cyc = 0; cyc < 4000; cyc++) begin
      ce_w = (cyc < 16) || ($random(seed) % 4 != 0);
      ce_ad = (cyc < 16) || ($random(seed) % 4 != 0);
      wlo = $random(seed); whi = $random(seed); act = $random(seed);
      pcin = {16'($random(seed)), 32'($random(seed))};
      if (cyc >= 16 && cyc < 272) begin
        act = cyc[0] ? 8'h80 : 8'h7f;
        wlo = cyc[1] ? 8'h80 : 8'h7f;
        whi = cyc[2] ? 8'h80 : 8'h7f;
        pcin = cyc[3] ? 48'h800000000000 : 48'h7fffffffffff;
        ce_w = (cyc % 32) < 20;
        ce_ad = (cyc % 16) < 12;
      end
      #1 clk_f = 1'b1;
      #1 clk_f = 1'b0;
      if (cyc >= 16) begin
        $write("%0d", cyc);
        for (int v = 0; v < 4; v++) $write(" %012x %08x", pcout[v], p[v]);
        $write("\n");
      end
    end
    $finish;
  end
endmodule
