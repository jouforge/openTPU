// The systolic MXU's arithmetic cells on Versal DSP58s, with the same interfaces and
// enabled-edge pipelines as rtl/tech/generic and rtl/tech/xilinxus. The file list selects
// this family (opentpu/target.py). DSP_MODE INT24 uses the signed 27 x 24 multiplier;
// USE_SIMD ONE58 uses the full post-adder. NEGATE and all resets are inactive.
//
// DSP58 has 34-bit A, 24-bit B and 58-bit P / PC ports. Extend signed operands at the
// primitive boundary; keep the common 24/32-bit PE result and 48-bit accumulator contract.
// PE links use P and C through fabric: the common 48-bit interface cannot carry DSP58
// PCOUT/PCIN, which requires all 58 bits connected (DPCA-12). CREG 0 preserves the pipeline.
// The column end keeps all 58 cascade bits internally. Taking its final low 48 bits gives
// the same modulo-2^48 sum as the generic model, including row accumulation wraparound.
//
// One PE: BREG 1, INMODEREG 1, optional weight A/D registers, MREG 1 and PREG except LAST.
// PK packs whi at bit 16; FIRST adds 2^15 under the registered select. The incoming PC
// sign extension cannot affect the low result bits. LAST's pcout is unused by the MXU.
module otpu_pe #(parameter bit FIRST = 1'b0, parameter bit LAST = 1'b0, parameter bit WREG = 1'b0,
                 parameter bit PK = 1'b0) (
  input  logic               clk,
  input  logic               en,
  input  logic [7:0]         act,
  input  logic [7:0]         wlo,               // WREG: the weights from the next cycle on
  input  logic [7:0]         whi,
  input  logic               sel,               // the select from the next cycle on
  input  logic [47:0]        pcin,
  output logic [47:0]        pcout,
  output logic signed [(PK ? 32 : 24)-1:0] p
);
  logic [57:0] pf;
  DSP58 #(
    .A_INPUT("DIRECT"), .B_INPUT("DIRECT"), .AMULTSEL("AD"), .BMULTSEL("B"),
    .PREADDINSEL("A"), .USE_MULT("MULTIPLY"), .USE_WIDEXOR("FALSE"),
    .DSP_MODE("INT24"), .RESET_MODE("SYNC"), .USE_SIMD("ONE58"),
    .AREG(WREG ? 1 : 0), .ACASCREG(WREG ? 1 : 0), .BREG(1), .BCASCREG(1),
    .CREG(0), .DREG(WREG ? 1 : 0),
    .ADREG(0), .MREG(1), .PREG(LAST ? 0 : 1), .INMODEREG(1), .OPMODEREG((FIRST && PK) ? 1 : 0),
    .ALUMODEREG(0), .CARRYINREG(0), .CARRYINSELREG(0), .USE_PATTERN_DETECT("NO_PATDET"),
    .AUTORESET_PATDET("NO_RESET"), .MASK(58'h0ffffffffffffff), .PATTERN(58'h0),
    .RND((FIRST && PK) ? 58'h8000 : 58'h0), .SEL_MASK("MASK"), .SEL_PATTERN("PATTERN")
  ) u_dsp (
    .CLK(clk), .ASYNC_RST(1'b0), .NEGATE(3'b000),
    .A({{26{wlo[7]}}, wlo}), .B({{16{act[7]}}, act}), .C({{10{pcin[47]}}, pcin}),
    .D(PK ? {{3{whi[7]}}, whi, 16'd0} : {{19{whi[7]}}, whi}),
    .INMODE({2'b00, sel, PK ? 1'b0 : sel, 1'b0}),
    .OPMODE(FIRST ? {(PK && sel) ? 2'b10 : 2'b00, 7'b000_01_01} : 9'b00_011_01_01),
    .ALUMODE(4'b0000), .CARRYIN(1'b0), .CARRYINSEL(3'b000),
    .CEA1(1'b0), .CEA2(WREG ? en : 1'b0), .CEB1(en), .CEB2(en), .CEC(1'b0),
    .CED(WREG ? en : 1'b0), .CEAD(1'b0),
    .CEM(en), .CEP(LAST ? 1'b0 : en), .CEALUMODE(1'b0), .CECTRL((FIRST && PK) ? en : 1'b0),
    .CECARRYIN(1'b0),
    .CEINMODE(en),
    .RSTA(1'b0), .RSTB(1'b0), .RSTC(1'b0), .RSTD(1'b0), .RSTM(1'b0), .RSTP(1'b0),
    .RSTALLCARRYIN(1'b0), .RSTALUMODE(1'b0), .RSTCTRL(1'b0), .RSTINMODE(1'b0),
    .ACIN(34'd0), .BCIN(24'd0), .PCIN(58'd0), .CARRYCASCIN(1'b0), .MULTSIGNIN(1'b0),
    .ACOUT(), .BCOUT(), .PCOUT(), .P(pf), .CARRYCASCOUT(), .MULTSIGNOUT(), .CARRYOUT(),
    .OVERFLOW(), .UNDERFLOW(), .PATTERNDETECT(), .PATTERNBDETECT(), .XOROUT());
  assign pcout = pf[47:0];
  assign p = pf[(PK ? 32 : 24)-1:0];
endmodule

// The column end: eight multipliers and one accumulator, as in the UltraScale family.
// Groups 0..3 multiply at E+1 and register their cascade sum at E+2. Groups 4..7 add one
// input stage and join that sum at E+3. The accumulator adds it at E+4, with first captured
// in OPMODEREG. CPG 2 enables D in the pre-adder; CPG 1 uses A's second register instead.
module otpu_mxce #(parameter int CPG = 1) (
  input  logic                clk,
  input  logic [7:0][23:0]    a,        // group g's chain end: the pre-adder's A ...
  input  logic [7:0][23:0]    d,        // ... and D (CPG 2: the group's second chain, or
                                        // otpu_mxu's packed int8 high halves; else unused)
  input  logic                den,      // D into the pre-adder (CPG 2; constant while it streams)
  input  logic [7:0][16:0]    bm,       // B = m << sh, with a / d
  input  logic                first,    // the row's first block (two cycles after a / d / bm)
  output logic [47:0]         acc
);
  initial if (CPG != 1 && CPG != 2) $fatal(1, "otpu_mxce: CPG %0d", CPG);
  logic [8:0][57:0] pc, pf;
  for (genvar g = 0; g < 8; g++) begin : g_m
    // groups 4..7: a register more before the multiplier (the pre-adder's, or A's second)
    localparam bit LATE = (g >= 4);
    localparam int AR = (LATE && CPG == 1) ? 2 : 1;
    DSP58 #(
      .A_INPUT("DIRECT"), .B_INPUT("DIRECT"), .AMULTSEL(CPG == 2 ? "AD" : "A"), .BMULTSEL("B"),
      .PREADDINSEL("A"), .USE_MULT("MULTIPLY"), .USE_WIDEXOR("FALSE"),
      .DSP_MODE("INT24"), .RESET_MODE("SYNC"), .USE_SIMD("ONE58"),
      .AREG(AR), .ACASCREG(AR), .BREG(LATE ? 2 : 1),
      .BCASCREG(LATE ? 2 : 1), .CREG(0), .DREG(CPG == 2 ? 1 : 0),
      .ADREG((LATE && CPG == 2) ? 1 : 0), .MREG(1),
      .PREG((g == 3 || g == 7) ? 1 : 0), .INMODEREG(0), .OPMODEREG(0), .ALUMODEREG(0),
      .CARRYINREG(0), .CARRYINSELREG(0), .USE_PATTERN_DETECT("NO_PATDET"),
      .AUTORESET_PATDET("NO_RESET"), .MASK(58'h0ffffffffffffff), .PATTERN(58'h0),
      .SEL_MASK("MASK"), .SEL_PATTERN("PATTERN")
    ) u_dsp (
      .CLK(clk), .ASYNC_RST(1'b0), .NEGATE(3'b000),
      .A({{10{a[g][23]}}, a[g]}), .B({7'd0, bm[g]}), .C(58'd0),
      .D((CPG == 2) ? {{3{d[g][23]}}, d[g]} : 27'd0),
      .INMODE((CPG == 2) ? {2'b00, den, 2'b00} : 5'b00000),
      .OPMODE(g == 0 ? 9'b00_000_01_01 : 9'b00_001_01_01),
      .ALUMODE(4'b0000), .CARRYIN(1'b0), .CARRYINSEL(3'b000),
      .CEA1(AR == 2 ? 1'b1 : 1'b0), .CEA2(1'b1), .CEB1(LATE ? 1'b1 : 1'b0), .CEB2(1'b1),
      .CEC(1'b0), .CED(CPG == 2 ? 1'b1 : 1'b0), .CEAD((LATE && CPG == 2) ? 1'b1 : 1'b0),
      .CEM(1'b1), .CEP((g == 3 || g == 7) ? 1'b1 : 1'b0), .CEALUMODE(1'b0), .CECTRL(1'b0),
      .CECARRYIN(1'b0), .CEINMODE(1'b0),
      .RSTA(1'b0), .RSTB(1'b0), .RSTC(1'b0), .RSTD(1'b0), .RSTM(1'b0), .RSTP(1'b0),
      .RSTALLCARRYIN(1'b0), .RSTALUMODE(1'b0), .RSTCTRL(1'b0), .RSTINMODE(1'b0),
      .ACIN(34'd0), .BCIN(24'd0), .PCIN(g == 0 ? 58'd0 : pc[g == 0 ? 0 : g - 1]),
      .CARRYCASCIN(1'b0), .MULTSIGNIN(1'b0),
      .ACOUT(), .BCOUT(), .PCOUT(pc[g]), .P(pf[g]), .CARRYCASCOUT(), .MULTSIGNOUT(),
      .CARRYOUT(), .OVERFLOW(), .UNDERFLOW(), .PATTERNDETECT(), .PATTERNBDETECT(), .XOROUT());
  end
  // the accumulator: X = P (0 for a row's first block), Y = 0, Z = PCIN
  DSP58 #(
    .A_INPUT("DIRECT"), .B_INPUT("DIRECT"), .AMULTSEL("A"), .BMULTSEL("B"), .PREADDINSEL("A"),
    .USE_MULT("NONE"), .USE_WIDEXOR("FALSE"),
    .DSP_MODE("INT24"), .RESET_MODE("SYNC"), .USE_SIMD("ONE58"),
    .AREG(0), .ACASCREG(0), .BREG(0), .BCASCREG(0), .CREG(0), .DREG(0),
    .ADREG(0), .MREG(0), .PREG(1), .INMODEREG(0), .OPMODEREG(1), .ALUMODEREG(0),
    .CARRYINREG(0), .CARRYINSELREG(0), .USE_PATTERN_DETECT("NO_PATDET"),
    .AUTORESET_PATDET("NO_RESET"), .MASK(58'h0ffffffffffffff), .PATTERN(58'h0),
    .SEL_MASK("MASK"), .SEL_PATTERN("PATTERN")
  ) u_acc (
    .CLK(clk), .ASYNC_RST(1'b0), .NEGATE(3'b000),
    .A(34'd0), .B(24'd0), .C(58'd0), .D(27'd0),
    .INMODE(5'b00000), .OPMODE(first ? 9'b00_001_00_00 : 9'b00_001_00_10),
    .ALUMODE(4'b0000), .CARRYIN(1'b0), .CARRYINSEL(3'b000),
    .CEA1(1'b0), .CEA2(1'b0), .CEB1(1'b0), .CEB2(1'b0), .CEC(1'b0), .CED(1'b0), .CEAD(1'b0),
    .CEM(1'b0), .CEP(1'b1), .CEALUMODE(1'b0), .CECTRL(1'b1), .CECARRYIN(1'b0),
    .CEINMODE(1'b0),
    .RSTA(1'b0), .RSTB(1'b0), .RSTC(1'b0), .RSTD(1'b0), .RSTM(1'b0), .RSTP(1'b0),
    .RSTALLCARRYIN(1'b0), .RSTALUMODE(1'b0), .RSTCTRL(1'b0), .RSTINMODE(1'b0),
    .ACIN(34'd0), .BCIN(24'd0), .PCIN(pc[7]), .CARRYCASCIN(1'b0), .MULTSIGNIN(1'b0),
    .ACOUT(), .BCOUT(), .PCOUT(pc[8]), .P(pf[8]), .CARRYCASCOUT(), .MULTSIGNOUT(),
    .CARRYOUT(), .OVERFLOW(), .UNDERFLOW(), .PATTERNDETECT(), .PATTERNBDETECT(), .XOROUT());
  assign acc = pf[8][47:0];
endmodule

// A pumped packed PE (otpu_mxpc): weights load on ce_w, their pre-adder on ce_ad; the
// activation, multiplier and accumulator advance on every clk_f edge. BF selects one or
// two activation registers. FIRST adds 2^15. Keep the external 48-bit cascade and 32-bit
// packed result, with the same enabled-edge behavior as rtl/tech/generic.
module otpu_ppe #(parameter bit FIRST = 1'b0, parameter bit BF = 1'b1) (
  input  logic        clk_f,
  input  logic        ce_w,
  input  logic        ce_ad,
  input  logic [7:0]  wlo,
  input  logic [7:0]  whi,
  input  logic [7:0]  act,
  input  logic [47:0] pcin,
  output logic [47:0] pcout,
  output logic [31:0] p
);
  logic [57:0] pf;
  DSP58 #(
    .A_INPUT("DIRECT"), .B_INPUT("DIRECT"), .AMULTSEL("AD"), .BMULTSEL("B"),
    .PREADDINSEL("A"), .USE_MULT("MULTIPLY"), .USE_WIDEXOR("FALSE"),
    .DSP_MODE("INT24"), .RESET_MODE("SYNC"), .USE_SIMD("ONE58"),
    .AREG(1), .ACASCREG(1), .BREG(BF ? 1 : 2), .BCASCREG(BF ? 1 : 2),
    .CREG(0), .DREG(1), .ADREG(1), .MREG(1), .PREG(1), .INMODEREG(0), .OPMODEREG(0),
    .ALUMODEREG(0), .CARRYINREG(0), .CARRYINSELREG(0), .USE_PATTERN_DETECT("NO_PATDET"),
    .AUTORESET_PATDET("NO_RESET"), .MASK(58'h0ffffffffffffff), .PATTERN(58'h0),
    .RND(FIRST ? 58'h8000 : 58'h0), .SEL_MASK("MASK"), .SEL_PATTERN("PATTERN")
  ) u_dsp (
    .CLK(clk_f), .ASYNC_RST(1'b0), .NEGATE(3'b000),
    .A({{26{wlo[7]}}, wlo}), .B({{16{act[7]}}, act}), .C({{10{pcin[47]}}, pcin}), .D({{3{whi[7]}}, whi, 16'd0}),
    .INMODE(5'b00100), .OPMODE(FIRST ? 9'b10_000_01_01 : 9'b00_011_01_01),
    .ALUMODE(4'b0000), .CARRYIN(1'b0), .CARRYINSEL(3'b000),
    .CEA1(1'b0), .CEA2(ce_w), .CEB1(BF ? 1'b0 : 1'b1), .CEB2(1'b1), .CEC(1'b0), .CED(ce_w),
    .CEAD(ce_ad), .CEM(1'b1), .CEP(1'b1), .CEALUMODE(1'b0), .CECTRL(1'b0), .CECARRYIN(1'b0),
    .CEINMODE(1'b0),
    .RSTA(1'b0), .RSTB(1'b0), .RSTC(1'b0), .RSTD(1'b0), .RSTM(1'b0), .RSTP(1'b0),
    .RSTALLCARRYIN(1'b0), .RSTALUMODE(1'b0), .RSTCTRL(1'b0), .RSTINMODE(1'b0),
    .ACIN(34'd0), .BCIN(24'd0), .PCIN(58'd0), .CARRYCASCIN(1'b0), .MULTSIGNIN(1'b0),
    .ACOUT(), .BCOUT(), .PCOUT(), .P(pf), .CARRYCASCOUT(), .MULTSIGNOUT(), .CARRYOUT(),
    .OVERFLOW(), .UNDERFLOW(), .PATTERNDETECT(), .PATTERNBDETECT(), .XOROUT());
  assign pcout = pf[47:0];
  assign p = pf[31:0];
endmodule
