// Format-selecting front end for three independent integer-only IQ block/dot
// engines. No VPU/ISA integration, bus protocol, or device evidence.
//
// Handshake contract (plain valid/ready on both ports, nothing else):
//   * A beat is accepted exactly on (valid && ready) at a rising edge. Ready is
//     never a function of valid, so a source may hold valid until it is taken;
//     no beat is accepted on any other condition, and w_ready/a_ready are never
//     high together, so weight and activation payloads never overlap.
//   * In IDLE the presented {w_fmt,w_data} selects the child and pulses its
//     synchronous reset for one cycle; w_ready stays low, so that first
//     presented beat is a selection probe and is NOT accepted. The source keeps
//     it asserted and it is accepted in LOAD exactly once. Cost: one
//     unaccepted cycle; a block is 17/19/15 ACCEPTED beats (9/11/7 weight plus
//     8 activations), counted by `count`, which advances only on an accept and
//     runs 0..weights+7 without resetting at the weight/activation phase change.
//   * w_ready in LOAD is qualified by the live w_fmt: a beat only enters the
//     child that its tag names. An unknown tag (3) or a tag change while a
//     block is in flight therefore stalls with w_ready low instead of being
//     silently absorbed; the wrapper latches the sticky C_ERROR state, drops the
//     child reset and waits for `rst`. Without the tag qualifier the capture in
//     the FSM would be a hidden accept: the beat would vanish with w_ready high.
//   * C_ERROR is sticky: w_ready and a_ready are low until rst.
//   * The selected format is stable through group 7, then the wrapper returns to
//     IDLE. Reset and child reset are synchronous, active high.
//
// IQ1_M contributes two separate integer terms (grid and delta), so grp_terms
// says which ones the group carries (1 = grid only, 2 = grid + delta).
// grp_index always has four 11-bit lanes; IQ2_S keeps its standalone 40-bit
// 10-bit-lane port and the shell zero-extends each lane.
module otpu_iq_stream (
  // Control and weight-stream port.  w_ready is high only in the load phase and
  // only for the child the presented w_fmt names; see the header.
  input  logic        clk,
  input  logic        rst,
  input  logic        w_valid,
  output logic        w_ready,
  input  logic [63:0] w_data,
  input  logic [1:0]  w_fmt,
  // Activation-stream port, one 256-bit beat per group.
  input  logic         a_valid,
  output logic         a_ready,
  input  logic [255:0] a_data,
  // Per-group result record, extended for the delta/term carrying format.
  output logic        grp_valid,
  output logic [2:0]  grp_idx,
  output logic [1:0]  grp_fmt,
  output logic [31:0] grp_bsum,
  output logic [31:0] grp_aux,
  output logic [1:0]  grp_terms,
  output logic [4:0]  grp_ls1,
  output logic [4:0]  grp_ls2,
  output logic [31:0] grp_signs,
  output logic [43:0] grp_index,
  output logic [3:0]  grp_delta,
  // Block side channel, asserted with group 0 only.
  output logic        blk_valid,
  output logic [1:0]  blk_fmt,
  output logic [15:0] blk_d,
  output logic [63:0] blk_scales
);
  localparam logic [1:0] C_IDLE=2'd0, C_LOAD=2'd1, C_ACT=2'd2, C_ERROR=2'd3;
  logic [1:0] ctl;
  logic [1:0] fmt;
  logic [3:0] count;      // last zero-based accepted-beat index of current block
  logic [3:0] last_count; // last accepted weight index, i.e. weights-1, per format
  logic [3:0] act_i;      // activation beats accepted so far, 0..7
  logic [2:0] child_rst;
  logic [2:0] cw_ready, ca_ready;
  logic [2:0] cv, bv;
  logic [2:0] c_grp_idx [3];
  logic [31:0] c_bsum [3], c_aux [3], c_signs [3];
  logic [63:0] c_scales [3];
  logic [1:0] c_terms [3];
  logic [4:0] c_ls1 [3], c_ls2 [3];
  logic [43:0] c_index [3];
  logic [3:0] c_delta [3];
  logic [15:0] c_d [3];
  logic [39:0] native_s_index;
  logic [63:0] native_s_scales;
  logic child_xxs_rst, child_s_rst, child_m_rst;
  logic vw_xxs, vw_s, vw_m, va_xxs, va_s, va_m;

  // ---- handshake -----------------------------------------------------------------
  // Ready is decoded from the registered state and the live format tag only, so
  // it never depends on valid and never hides an accept.
  always_comb begin
    case (fmt)
      2'd0:    last_count = 4'd8;   //  9 weight beats
      2'd1:    last_count = 4'd10;  // 11 weight beats
      default: last_count = 4'd6;   //  7 weight beats
    endcase
    child_rst = {rst, rst, rst};
    // The first presented beat selects the child and is NOT accepted: it arms
    // `fmt` and resets that child so the block starts from a known state.
    if (ctl==C_IDLE && w_valid && w_fmt!=2'd3) child_rst[w_fmt] = 1'b1;
    w_ready = !rst && ctl==C_LOAD && w_fmt!=2'd3 && cw_ready[w_fmt];
    a_ready = !rst && ctl==C_ACT && ca_ready[fmt];
    // child payload strobes: only a real transfer reaches the child
    vw_xxs = w_valid && w_ready && fmt==2'd0;
    vw_s   = w_valid && w_ready && fmt==2'd1;
    vw_m   = w_valid && w_ready && fmt==2'd2;
    va_xxs = a_valid && a_ready && fmt==2'd0;
    va_s   = a_valid && a_ready && fmt==2'd1;
    va_m   = a_valid && a_ready && fmt==2'd2;
  end
  assign child_xxs_rst = child_rst[0];
  assign child_s_rst   = child_rst[1];
  assign child_m_rst   = child_rst[2];

  // ---- control -------------------------------------------------------------------
  // One accept = one count step. In IDLE the probe beat is not accepted; a valid
  // tag arms the child. An illegal tag, or a tag that disagrees with the block in
  // flight, is a sticky protocol error.
  always_ff @(posedge clk) begin
    if (rst) begin
      ctl <= C_IDLE; fmt <= 2'd0; count <= 4'd0; act_i <= 4'd0;
    end else begin
      case (ctl)
        C_IDLE: begin
          if (w_valid) begin
            if (w_fmt==2'd3) ctl <= C_ERROR;
            // The probe beat selects the child but is not an accept, so the
            // block's accepted-beat count still starts at zero here.
            else begin fmt <= w_fmt; count <= 4'd0; act_i <= 4'd0; ctl <= C_LOAD; end
          end
        end
        C_LOAD: begin
          if (w_valid && w_fmt!=fmt) ctl <= C_ERROR;   // tag change mid block
          // a beat only exists when valid is asserted; ready alone must never
          // advance the block or the phase (no accept without a transfer)
          else if (w_valid && w_ready) begin
            // count keeps running across the phase change: it is the block's
            // accepted-beat index, not a per-phase counter.
            count <= count + 4'd1;
            if (count==last_count) begin ctl <= C_ACT; act_i <= 4'd0; end
          end
        end
        C_ACT: begin
          if (a_valid && a_ready) begin
            count <= count + 4'd1;        // continues past the weight beats
            if (act_i==4'd7) ctl <= C_IDLE;
            else act_i <= act_i + 4'd1;
          end
        end
        default: ctl <= C_ERROR;
      endcase
    end
  end

  otpu_iq2xxs xxs(.clk(clk), .rst(child_xxs_rst),
    .w_valid(vw_xxs), .w_ready(cw_ready[0]), .w_data(w_data),
    .a_valid(va_xxs), .a_ready(ca_ready[0]), .a_data(a_data),
    .grp_valid(cv[0]), .grp_idx(c_grp_idx[0]), .grp_bsum(c_bsum[0]),
    .grp_aux(c_aux[0]), .grp_terms(c_terms[0]), .grp_ls1(c_ls1[0]),
    .grp_ls2(c_ls2[0]), .grp_signs(c_signs[0]), .grp_index(c_index[0]),
    .grp_delta(c_delta[0]), .blk_valid(bv[0]), .blk_d(c_d[0]), .blk_scales(c_scales[0]));
  otpu_iq2s iq2s(.clk(clk), .rst(child_s_rst),
    .w_valid(vw_s), .w_ready(cw_ready[1]), .w_data(w_data),
    .a_valid(va_s), .a_ready(ca_ready[1]), .a_data(a_data),
    .grp_valid(cv[1]), .grp_idx(c_grp_idx[1]), .grp_bsum(c_bsum[1]),
    .grp_aux(c_aux[1]), .grp_ls1(c_ls1[1]), .grp_ls2(c_ls2[1]), .grp_signs(c_signs[1]),
    .grp_index(native_s_index), .blk_valid(bv[1]), .blk_d(c_d[1]),
    .blk_scales(native_s_scales));
  otpu_iq1m iq1m(.clk(clk), .rst(child_m_rst),
    .w_valid(vw_m), .w_ready(cw_ready[2]), .w_data(w_data),
    .a_valid(va_m), .a_ready(ca_ready[2]), .a_data(a_data),
    .grp_valid(cv[2]), .grp_idx(c_grp_idx[2]), .grp_bsum(c_bsum[2]),
    .grp_aux(c_aux[2]), .grp_terms(c_terms[2]), .grp_ls1(c_ls1[2]),
    .grp_ls2(c_ls2[2]), .grp_signs(c_signs[2]), .grp_index(c_index[2]),
    .grp_delta(c_delta[2]), .blk_valid(bv[2]), .blk_d(c_d[2]), .blk_scales(c_scales[2]));

  // ---- output mux ----------------------------------------------------------------
  // IQ2_S has no delta/terms outputs; its auxiliary output is the second
  // scaled half ls2*sumi2 (not an eighth-weight offset). The shell completes
  // the group record without changing the native combined partial.
  always_comb begin
    c_terms[1] = 2'd2;
    c_delta[1] = 4'd0;
    c_index[1] = {1'b0, native_s_index[39:30], 1'b0, native_s_index[29:20],
                  1'b0, native_s_index[19:10], 1'b0, native_s_index[9:0]};
    c_scales[1] = native_s_scales;
    grp_valid = cv[fmt];   grp_idx = c_grp_idx[fmt]; grp_fmt = fmt;
    grp_bsum  = c_bsum[fmt]; grp_aux = c_aux[fmt];   grp_terms = c_terms[fmt];
    grp_ls1   = c_ls1[fmt];  grp_ls2 = c_ls2[fmt];   grp_signs = c_signs[fmt];
    grp_index = c_index[fmt]; grp_delta = c_delta[fmt];
    blk_valid = bv[fmt];     blk_fmt = fmt;
    blk_d = c_d[fmt];        blk_scales = c_scales[fmt];
  end
endmodule