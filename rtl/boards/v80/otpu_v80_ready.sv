// Read one 32-byte word from each of a channel's four 1 GiB HBM pseudo-channels.
// The pinned AMR design exposes no PL calibration flag. CALIB therefore means that all
// four routes responded OKAY since the AXI reset, not a synthetic clock-lock indication.
// This probe never writes memory. No response leaves ready low; a bad response is sticky.
// Until ready, the system reserves this AXI read port for the probe. rst requires a
// coordinated reset of the AXI endpoint; the physical top releases it once after configuration.
module otpu_v80_ready #(parameter int CH = 0) (
  input  logic        clk, rst,
  output logic        ready, err,
  output logic        arvalid,
  input  logic        arready,
  output logic [63:0] araddr,
  input  logic        rvalid, rid,
  input  logic [1:0]  rresp,
  input  logic        rlast,
  output logic        rready
);
  typedef enum logic [1:0] {SEND, WAIT_R, DONE, FAILED} state_t;
  state_t state = SEND;
  logic [1:0] pc = '0;
  logic       done = 1'b0, failed = 1'b0;
  // A registered level crosses into the core's calibration synchronizer. Decoding
  // state at that boundary can glitch when its bits change on different delays.
  assign ready = done;
  assign err = failed;
  assign arvalid = state == SEND && !rst;
  assign araddr = 64'h4000000000 + (64'(CH) << 32) + (64'(pc) << 30);
  assign rready = state == WAIT_R && !rst;
  always_ff @(posedge clk) begin
    if (rst) begin
      state <= SEND;
      pc <= '0;
      done <= 1'b0;
      failed <= 1'b0;
    end else begin
      if (arvalid && arready) state <= WAIT_R;
      if (rvalid && rready) begin
        // ARLEN=0: exactly one beat, with ID 0, RLAST and an OKAY response.
        if (rid || rresp != 0 || !rlast) failed <= 1'b1;
        if (rlast) begin
          if (failed || rid || rresp != 0) state <= FAILED;
          else if (pc == 3) begin state <= DONE; done <= 1'b1; end
          else begin pc <= pc + 1'b1; state <= SEND; end
        end
      end
    end
  end
endmodule
