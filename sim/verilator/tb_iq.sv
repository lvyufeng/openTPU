// Testbench for rtl/vpu/otpu_iq2s.sv (tools/iq_rtl_check.py, docs/iq_dot.md): streams blocks and
// their activations, checks the handshake and the per-group metadata against the block's own bytes,
// and dumps every group's partial for the driver to compare with its reference. Verilator 5.x
// --binary:
//   build/iq_rtl/Vtb_iq +vectors=<file> +out=<file> [+cases=<n>]
// +out is optional; +cases limits the cases run (0 is all).
//
// Vectors are two lines a case, hexadecimal numbers separated by whitespace:
//   B w0 .. w21   the 82-byte block padded to 88, byte 4k + j is word k's byte j
//   A a0 .. a63   the 256 int8 activations, byte 4k + j is word k's byte j; activation byte
//                 32g + 8l + j is sub-block l's element j of group g
// The dump is, per case:
//   case <i> d <4 hex> scales <16 hex>
//   g <g> <bsum 8 hex> <ls1 2 hex> <ls2 2 hex> <signs 8 hex> <index 10 hex>
// bsum is the int32 partial, ls1/ls2 the group's multipliers, signs the four sign bytes and index
// the four 10-bit grid indices with sub-block l = 0 in the low bits.
//
// The checks here are protocol and wiring, not arithmetic: 19 accepted cycles a case (11 weight
// beats then 8 activation beats), one result per activation beat with grp_idx 0..7 in order, no
// results while loading, and d, scales, sign bytes, scale multipliers and grid indices equal to the
// same fields decoded here from the block's bytes. The partials are compared by the driver against
// its own model (the arithmetic claim is docs/iq_dot.md's, checked there).
module tb_iq;
  logic clk = 1'b0;
  logic rst = 1'b1;
  always #5 clk = ~clk;

  logic         w_valid = 1'b0;
  logic         w_ready;
  logic [63:0]  w_data  = '0;
  logic         a_valid = 1'b0;
  logic         a_ready;
  logic [255:0] a_data  = '0;
  logic         grp_valid, blk_valid;
  logic [2:0]   grp_idx;
  logic [31:0]  grp_bsum, grp_signs;
  logic [4:0]   grp_ls1, grp_ls2;
  logic [39:0]  grp_index;
  logic [15:0]  blk_d;
  logic [63:0]  blk_scales;

  otpu_iq2s dut (
    .clk(clk), .rst(rst),
    .w_valid(w_valid), .w_ready(w_ready), .w_data(w_data),
    .a_valid(a_valid), .a_ready(a_ready), .a_data(a_data),
    .grp_valid(grp_valid), .grp_idx(grp_idx), .grp_bsum(grp_bsum),
    .grp_ls1(grp_ls1), .grp_ls2(grp_ls2), .grp_signs(grp_signs), .grp_index(grp_index),
    .blk_valid(blk_valid), .blk_d(blk_d), .blk_scales(blk_scales)
  );

  longint cyc = 0;
  always @(posedge clk) cyc <= cyc + 1;

  // ---- vectors and results
  integer fd_in, fd_out, n, ci, cases, prot_err;
  int     ncases;
  string  vname, oname, op;
  logic [31:0] t;
  logic [31:0] blk_word [0:21];
  logic [31:0] act_word [0:63];
  logic [7:0]  blk      [0:83];
  logic [31:0] o_bsum   [0:7];
  logic [4:0]  o_ls1    [0:7];
  logic [4:0]  o_ls2    [0:7];
  logic [31:0] o_signs  [0:7];
  logic [39:0] o_idx    [0:7];
  logic [9:0]  ei       [0:3];
  logic [15:0] o_d;
  logic [63:0] o_scales;

  // ---- one weight beat: wait for w_ready, hold valid for the accepting edge
  task automatic w_beat(input logic [63:0] d);
    int tmo;
    begin
      @(negedge clk);
      tmo = 0;
      while (!w_ready) begin
        @(negedge clk);
        if (tmo++ > 32) $fatal(1, "w_ready timeout at case %0d", ci);
      end
      if (a_ready !== 1'b0) prot_err++;
      w_valid = 1'b1;
      w_data  = d;
      @(posedge clk);
      #1;
      w_valid = 1'b0;
      // the load phase never has results to hand out
      if (grp_valid !== 1'b0 || blk_valid !== 1'b0) prot_err++;
    end
  endtask

  // ---- one activation beat, and the group result it produces on its accepting edge
  task automatic a_beat(input logic [255:0] d, input int g);
    int tmo;
    begin
      @(negedge clk);
      tmo = 0;
      while (!a_ready) begin
        @(negedge clk);
        if (tmo++ > 32) $fatal(1, "a_ready timeout at case %0d", ci);
      end
      if (w_ready !== 1'b0) prot_err++;
      a_valid = 1'b1;
      a_data  = d;
      @(posedge clk);
      #1;
      a_valid = 1'b0;
      if (grp_valid !== 1'b1) prot_err++;
      if (grp_idx !== 3'(g)) prot_err++;
      if (g == 0) begin
        if (blk_valid !== 1'b1) prot_err++;
        o_d      = blk_d;
        o_scales = blk_scales;
      end else if (blk_valid !== 1'b0) begin
        prot_err++;
      end
      o_bsum[g]  = grp_bsum;
      o_ls1[g]   = grp_ls1;
      o_ls2[g]   = grp_ls2;
      o_signs[g] = grp_signs;
      o_idx[g]   = grp_index;
    end
  endtask

  // ---- a case: 11 weight beats, 8 activation beats, then the structural checks and the dump
  task automatic run_case;
    int         g, l;
    longint     c0;
    logic [39:0]  exp_idx;
    logic [31:0]  exp_sign;
    logic [4:0]   exp_ls1, exp_ls2;
    begin
      c0 = cyc;
      for (int k = 0; k < 11; k++)
        w_beat({blk_word[2 * k + 1], blk_word[2 * k]});
      for (int k = 0; k < 8; k++)
        a_beat({act_word[8 * k + 7], act_word[8 * k + 6], act_word[8 * k + 5], act_word[8 * k + 4],
                act_word[8 * k + 3], act_word[8 * k + 2], act_word[8 * k + 1], act_word[8 * k]}, k);
      // 11 + 8 accepted cycles, no wait except the handshake's own
      if ((cyc - c0) != 19) prot_err++;

      // the metadata the module must have taken from the block's bytes
      if (o_d !== {blk[1], blk[0]}) prot_err++;
      for (g = 0; g < 8; g++) begin
        if (o_scales[8 * g +: 8] !== blk[74 + g]) prot_err++;
        exp_sign = {blk[34 + 4 * g + 3], blk[34 + 4 * g + 2],
                    blk[34 + 4 * g + 1], blk[34 + 4 * g]};
        if (o_signs[g] !== exp_sign) prot_err++;
        exp_ls1 = 5'(1 + 2 * (blk[74 + g] & 8'h0f));
        exp_ls2 = 5'(1 + 2 * ((blk[74 + g] >> 4) & 8'h0f));
        if (o_ls1[g] !== exp_ls1 || o_ls2[g] !== exp_ls2) prot_err++;
        for (l = 0; l < 4; l++)
          ei[l] = {2'b00, blk[2 + 4 * g + l]} |
                  (({2'b00, blk[66 + g]} << (8 - 2 * l)) & 10'h300);
        exp_idx = {ei[3], ei[2], ei[1], ei[0]};
        if (o_idx[g] !== exp_idx) prot_err++;
      end

      if (fd_out != 0) begin
        $fdisplay(fd_out, "case %0d d %04h scales %016h", ci, o_d, o_scales);
        for (g = 0; g < 8; g++)
          $fdisplay(fd_out, "g %0d %08h %02h %02h %08h %010h",
                    g, o_bsum[g], o_ls1[g], o_ls2[g], o_signs[g], o_idx[g]);
      end
      cases++;
    end
  endtask

  initial begin
    int k, j;
    if (!$value$plusargs("vectors=%s", vname)) $fatal(1, "+vectors= missing");
    if ($value$plusargs("out=%s", oname)) begin
      fd_out = $fopen(oname, "w");
      if (fd_out == 0) $fatal(1, "cannot write %s", oname);
    end else begin
      fd_out = 0;
    end
    if (!$value$plusargs("cases=%d", ncases)) ncases = 0;
    fd_in = $fopen(vname, "r");
    if (fd_in == 0) $fatal(1, "cannot open %s", vname);

    ci = 0;
    cases = 0;
    prot_err = 0;
    repeat (3) @(negedge clk);
    rst = 1'b0;

    while (!$feof(fd_in)) begin
      n = $fscanf(fd_in, "%s", op);
      if (n != 1) break;
      case (op)
        "B": begin
          for (k = 0; k < 22; k++) begin
            n = $fscanf(fd_in, "%h", t);
            if (n != 1) $fatal(1, "short B line at case %0d", ci);
            blk_word[k] = t;
          end
          for (k = 0; k < 21; k++)
            for (j = 0; j < 4; j++)
              blk[4 * k + j] = blk_word[k][8 * j +: 8];
        end
        "A": begin
          for (k = 0; k < 64; k++) begin
            n = $fscanf(fd_in, "%h", t);
            if (n != 1) $fatal(1, "short A line at case %0d", ci);
            act_word[k] = t;
          end
          if (ncases == 0 || ci < ncases) run_case();
          ci++;
        end
        default: $fatal(1, "bad vector op %s at case %0d", op, ci);
      endcase
    end
    $fclose(fd_in);
    if (fd_out != 0) $fclose(fd_out);

    $display("IQ2STEST cases=%0d beats=%0d cycles=%0d errors=%0d",
             cases, 19 * cases, cyc, prot_err);
    if (prot_err != 0) $fatal(1, "IQ2_S testbench checks failed: %0d", prot_err);
    $finish;
  end
endmodule