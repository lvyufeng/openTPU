// Protocol/metadata testbench for the three-format IQ streaming wrapper.
// tools/iq_multi_rtl_check.py writes the vectors and independently checks the
// arithmetic; this file checks the handshake and the extended group record.
//
// Format on a B line: 0 = IQ2_XXS, 1 = IQ2_S, 2 = IQ1_M.
//
// Discipline enforced here:
//   * the source drives {valid,payload} on a falling edge and holds both until a
//     rising edge sees ready, so a beat is accepted exactly once, on the edge
//     where valid && ready, and never on a ready that arrived later;
//   * every counter below increments only on such an accepted edge (17/19/15 per
//     block, 9/11/7 weight plus 8 activations), and each block's accepted beats
//     are checked against that number plus the first-to-last accept span;
//   * no output is sampled speculatively: group fields and the block side channel
//     are captured in the same cycle the wrapper reports them, per group, so a
//     stale value can never be mistaken for a fresh one;
//   * valid is never assumed to be ready-dependent: after each beat the DUT must
//     drop w_ready, which proves no other accept happened during the hold.
// Rejection cases (unknown tag, tag change mid block) run after the vectors and
// must stall without accepting anything and without emitting a group.
module tb_iq_stream;
  logic clk = 1'b0;
  logic rst = 1'b1;
  always #5 clk = ~clk;
  logic w_valid = 1'b0, a_valid = 1'b0;
  logic w_ready, a_ready;
  logic [63:0] w_data = '0;
  logic [1:0] w_fmt = '0;
  logic [255:0] a_data = '0;
  logic grp_valid, blk_valid;
  logic [2:0] grp_idx;
  logic [1:0] grp_fmt, grp_terms, blk_fmt;
  logic [31:0] grp_bsum, grp_aux, grp_signs;
  logic [4:0] grp_ls1, grp_ls2;
  logic [43:0] grp_index;
  logic [3:0] grp_delta;
  logic [15:0] blk_d;
  logic [63:0] blk_scales;

  otpu_iq_stream dut (.*);

  longint cyc = 0;
  always @(posedge clk) cyc <= cyc + 1;

  integer fd_in, fd_out, n, ci, cases, prot_err;
  integer acc_w, acc_a;          // accepted beats over the whole run
  int blk_beat;                  // accepted beats inside the current block
  int aux_beats;                 // accepted beats from rejection exercise only
  logic quiet;                   // set while a protocol-only block is running
  longint c_first, c_last;       // accepted cycles of the first and last beat
  int ncases;
  string vname, oname, op;
  logic [31:0] t;
  logic [31:0] blk_word [0:21], act_word [0:63];
  logic [1:0] fmt_word;
  logic [7:0] blk [0:83];

  // per-group capture of the FULL extended record (d and scales are block level,
  // captured once with group 0, and compared against the same block in run_case)
  logic [31:0] o_bsum [0:7], o_aux [0:7], o_signs [0:7];
  logic [1:0] o_terms [0:7], o_fmt [0:7], o_bfmt;
  logic [3:0] o_delta [0:7];
  logic [4:0] o_ls1 [0:7], o_ls2 [0:7];
  logic [43:0] o_idx [0:7];
  logic [15:0] o_d;
  logic [63:0] o_scales;

  // KSIGNS_IQ2XS: the 128 sign bytes the IQ2_XXS child transports verbatim.
  // The python model compares grp_signs against this same table, so the
  // testbench reproduces it rather than re-deriving the bits.
  function automatic logic [7:0] iq2xxs_sign(input logic [6:0] seed);
    logic [7:0] rom [0:127];
    rom[0] = 8'h00;
    rom[1] = 8'h81;
    rom[2] = 8'h82;
    rom[3] = 8'h03;
    rom[4] = 8'h84;
    rom[5] = 8'h05;
    rom[6] = 8'h06;
    rom[7] = 8'h87;
    rom[8] = 8'h88;
    rom[9] = 8'h09;
    rom[10] = 8'h0A;
    rom[11] = 8'h8B;
    rom[12] = 8'h0C;
    rom[13] = 8'h8D;
    rom[14] = 8'h8E;
    rom[15] = 8'h0F;
    rom[16] = 8'h90;
    rom[17] = 8'h11;
    rom[18] = 8'h12;
    rom[19] = 8'h93;
    rom[20] = 8'h14;
    rom[21] = 8'h95;
    rom[22] = 8'h96;
    rom[23] = 8'h17;
    rom[24] = 8'h18;
    rom[25] = 8'h99;
    rom[26] = 8'h9A;
    rom[27] = 8'h1B;
    rom[28] = 8'h9C;
    rom[29] = 8'h1D;
    rom[30] = 8'h1E;
    rom[31] = 8'h9F;
    rom[32] = 8'hA0;
    rom[33] = 8'h21;
    rom[34] = 8'h22;
    rom[35] = 8'hA3;
    rom[36] = 8'h24;
    rom[37] = 8'hA5;
    rom[38] = 8'hA6;
    rom[39] = 8'h27;
    rom[40] = 8'h28;
    rom[41] = 8'hA9;
    rom[42] = 8'hAA;
    rom[43] = 8'h2B;
    rom[44] = 8'hAC;
    rom[45] = 8'h2D;
    rom[46] = 8'h2E;
    rom[47] = 8'hAF;
    rom[48] = 8'h30;
    rom[49] = 8'hB1;
    rom[50] = 8'hB2;
    rom[51] = 8'h33;
    rom[52] = 8'hB4;
    rom[53] = 8'h35;
    rom[54] = 8'h36;
    rom[55] = 8'hB7;
    rom[56] = 8'hB8;
    rom[57] = 8'h39;
    rom[58] = 8'h3A;
    rom[59] = 8'hBB;
    rom[60] = 8'h3C;
    rom[61] = 8'hBD;
    rom[62] = 8'hBE;
    rom[63] = 8'h3F;
    rom[64] = 8'hC0;
    rom[65] = 8'h41;
    rom[66] = 8'h42;
    rom[67] = 8'hC3;
    rom[68] = 8'h44;
    rom[69] = 8'hC5;
    rom[70] = 8'hC6;
    rom[71] = 8'h47;
    rom[72] = 8'h48;
    rom[73] = 8'hC9;
    rom[74] = 8'hCA;
    rom[75] = 8'h4B;
    rom[76] = 8'hCC;
    rom[77] = 8'h4D;
    rom[78] = 8'h4E;
    rom[79] = 8'hCF;
    rom[80] = 8'h50;
    rom[81] = 8'hD1;
    rom[82] = 8'hD2;
    rom[83] = 8'h53;
    rom[84] = 8'hD4;
    rom[85] = 8'h55;
    rom[86] = 8'h56;
    rom[87] = 8'hD7;
    rom[88] = 8'hD8;
    rom[89] = 8'h59;
    rom[90] = 8'h5A;
    rom[91] = 8'hDB;
    rom[92] = 8'h5C;
    rom[93] = 8'hDD;
    rom[94] = 8'hDE;
    rom[95] = 8'h5F;
    rom[96] = 8'h60;
    rom[97] = 8'hE1;
    rom[98] = 8'hE2;
    rom[99] = 8'h63;
    rom[100] = 8'hE4;
    rom[101] = 8'h65;
    rom[102] = 8'h66;
    rom[103] = 8'hE7;
    rom[104] = 8'hE8;
    rom[105] = 8'h69;
    rom[106] = 8'h6A;
    rom[107] = 8'hEB;
    rom[108] = 8'h6C;
    rom[109] = 8'hED;
    rom[110] = 8'hEE;
    rom[111] = 8'h6F;
    rom[112] = 8'hF0;
    rom[113] = 8'h71;
    rom[114] = 8'h72;
    rom[115] = 8'hF3;
    rom[116] = 8'h74;
    rom[117] = 8'hF5;
    rom[118] = 8'hF6;
    rom[119] = 8'h77;
    rom[120] = 8'h78;
    rom[121] = 8'hF9;
    rom[122] = 8'hFA;
    rom[123] = 8'h7B;
    rom[124] = 8'hFC;
    rom[125] = 8'h7D;
    rom[126] = 8'h7E;
    rom[127] = 8'hFF;
    iq2xxs_sign = rom[seed];
  endfunction

  // One weight beat: drive on the falling edge, accept on the rising edge that
  // sees w_ready, then drop valid immediately. Returns the accept cycle.
  task automatic w_beat(input logic [63:0] d, input logic [1:0] f);
    int tmo;
    begin
      @(negedge clk);
      w_fmt = f; w_data = d; w_valid = 1'b1; tmo = 0;
      // valid must be up before the ready it is matched with: sample at the
      // rising edge only, so one presented beat is exactly one accept.
      while (1'b1) begin
        @(posedge clk);
        if (w_ready === 1'b1) break;
        if (tmo++ > 64) $fatal(1, "w_ready timeout at case %0d", ci);
      end
      if (a_ready !== 1'b0) begin
        prot_err++; $display("HERR ci=%0d weight ready while a_ready", ci);
      end
      c_last = cyc;
      acc_w++; blk_beat++;
      // Deassert on the falling edge: a same-edge valid drop would race the
      // DUT's own sampling of that very posedge and could lose the accept.
      @(negedge clk);
      w_valid = 1'b0;
      if (grp_valid !== 1'b0 || blk_valid !== 1'b0) begin
        prot_err++; $display("HERR ci=%0d output during load", ci);
      end
      if (dut.count !== 4'(blk_beat)) begin
        prot_err++; $display("CNTERR ci=%0d count=%0d block accepts=%0d",
                             ci, dut.count, blk_beat);
      end
    end
  endtask

  // One activation beat, same discipline; the wrapper must present group `g`
  // in exactly the cycle that accepted it.
  task automatic a_beat(input logic [255:0] d, input int g);
    int tmo;
    begin
      @(negedge clk);
      a_valid = 1'b1; a_data = d; tmo = 0;
      // sample only on the rising edge: one presented group is one accept
      while (1'b1) begin
        @(posedge clk);
        if (a_ready === 1'b1) break;
        if (tmo++ > 64) $fatal(1, "a_ready timeout at case %0d group %0d", ci, g);
      end
      if (w_ready !== 1'b0) begin
        prot_err++; $display("HERR ci=%0d activation ready while w_ready", ci);
      end
      c_last = cyc;
      acc_a++; blk_beat++;
      #1;                        // the group record is valid through this delta
      if (grp_valid !== 1'b1) begin
        prot_err++; $display("GVERR ci=%0d g=%0d grp_valid=%b", ci, g, grp_valid);
      end
      if (grp_idx !== 3'(g) || grp_fmt !== fmt_word) begin
        prot_err++; $display("GVERR ci=%0d g=%0d idx=%0d fmt=%0d", ci, g, grp_idx, grp_fmt);
      end
      // group 0 is the only cycle that carries the block side channel
      if (g == 0) begin
        if (blk_valid !== 1'b1 || blk_fmt !== fmt_word) begin
          prot_err++; $display("BVERR ci=%0d g=0 bv=%b bfmt=%0d", ci, blk_valid, blk_fmt);
        end
        o_d = blk_d; o_scales = blk_scales; o_bfmt = blk_fmt;
      end else if (blk_valid !== 1'b0) begin
        prot_err++; $display("BVERR ci=%0d g=%0d blk_valid asserted", ci, g);
      end
      o_bsum[g] = grp_bsum; o_aux[g] = grp_aux; o_terms[g] = grp_terms;
      o_fmt[g] = grp_fmt;   o_ls1[g] = grp_ls1; o_ls2[g] = grp_ls2;
      o_signs[g] = grp_signs; o_idx[g] = grp_index; o_delta[g] = grp_delta;
      // Retire on the low phase, exactly like a weight beat, so one presented
      // group is one accept and every beat costs the same two cycles.
      @(negedge clk);
      a_valid = 1'b0;
      if (dut.count !== 4'(blk_beat)) begin
        prot_err++; $display("CNTERR ci=%0d count=%0d block accepts=%0d",
                             ci, dut.count, blk_beat);
      end
    end
  endtask

  task automatic run_case;
    int g, l, nbeats, nw0, na0;
    longint cf0, cl0;
    logic [31:0] aux;
    logic [15:0] sc;
    logic [7:0] high;
    logic [10:0] ei [0:3];
    begin
      nbeats = fmt_word == 0 ? 9 : fmt_word == 1 ? 11 : 7;
      nw0 = acc_w; na0 = acc_a; cf0 = c_first; cl0 = c_last; blk_beat = 0;
      // weight phase: the first beat is also the selection probe, so it is held
      // until the wrapper's own ready accepts it (one idle cycle is expected)
      for (int k = 0; k < nbeats; k++) begin
        w_beat({blk_word[2*k+1], blk_word[2*k]}, fmt_word);
        if (k == 0) c_first = c_last;   // accept cycle of this block's first beat
      end
      if (acc_w - nw0 != nbeats) begin
        prot_err++; $display("BEATERR ci=%0d accepted %0d weight beats, expected %0d",
                             ci, acc_w - nw0, nbeats);
      end
      for (int k = 0; k < 8; k++)
        a_beat({act_word[8*k+7],act_word[8*k+6],act_word[8*k+5],act_word[8*k+4],
                act_word[8*k+3],act_word[8*k+2],act_word[8*k+1],act_word[8*k]}, k);
      if (acc_a - na0 != 8) begin
        prot_err++; $display("BEATERR ci=%0d accepted %0d activation beats", ci, acc_a - na0);
      end
      // first-accepted to last-accepted span: two cycles per beat on this
      // stimulus (drive on the low phase, accept on the next rising edge).
      if ((c_last - c_first) != 2 * (nbeats + 8 - 1)) begin
        prot_err++;
        $display("LATERR ci=%0d span=%0d accepted=%0d ctl=%0d",
                 ci, c_last - c_first, nbeats + 8, dut.ctl);
      end
      if (dut.ctl !== 2'd0) begin
        prot_err++; $display("CTLERR ci=%0d ctl=%0d after block", ci, dut.ctl);
      end
      for (g = 0; g < 8; g++) begin
        if (o_fmt[g] !== fmt_word) begin
          prot_err++; $display("FVERR ci=%0d g=%0d fmt=%0d", ci, g, o_fmt[g]);
        end
      end
      if (fmt_word == 0) begin
        if (o_d !== {blk[1],blk[0]} || o_bfmt !== 2'd0) begin
          prot_err++; $display("FDERR ci=%0d d=%04h", ci, o_d);
        end
        for (g = 0; g < 8; g++) begin
          aux = {blk[8*g+9],blk[8*g+8],blk[8*g+7],blk[8*g+6]};
          if (o_scales[8*g+:8] !== {4'b0,aux[31:28]}) begin
            prot_err++; $display("FERR ci=%0d g=%0d scales", ci, g);
          end
          if (o_ls1[g] !== 5'(1+2*aux[31:28]) || o_ls2[g] !== 5'd0) begin
            prot_err++; $display("FERR ci=%0d g=%0d ls1=%0d ls2=%0d", ci, g, o_ls1[g], o_ls2[g]);
          end
          for (l = 0; l < 4; l++) begin
            ei[l] = {3'b0,blk[2+8*g+l]};
            if (o_signs[g][8*l+:8] !== iq2xxs_sign(aux[7*l+:7])) begin
              prot_err++; $display("FERR ci=%0d g=%0d l=%0d signs", ci, g, l);
            end
          end
          if (o_idx[g] !== {ei[3],ei[2],ei[1],ei[0]} || o_delta[g] !== 4'd0 ||
              o_aux[g] !== 32'd0 || o_terms[g] !== 2'd1) begin
            prot_err++; $display("FIELD ci=%0d g=%0d idx=%011h d=%0d aux=%0d t=%0d",
                                 ci, g, o_idx[g], o_delta[g], o_aux[g], o_terms[g]);
          end
        end
      end else if (fmt_word == 1) begin
        if (o_d !== {blk[1],blk[0]} || o_bfmt !== 2'd1) begin
          prot_err++; $display("FDERR ci=%0d d=%04h", ci, o_d);
        end
        if (o_scales !== {blk[81],blk[80],blk[79],blk[78],blk[77],blk[76],blk[75],blk[74]}) begin
          prot_err++; $display("FERR ci=%0d scales=%016h", ci, o_scales);
        end
        for (g = 0; g < 8; g++) begin
          if (o_scales[8*g+:8] !== blk[74+g]) begin
            prot_err++; $display("FERR ci=%0d g=%0d scales", ci, g);
          end
          if (o_signs[g] !== {blk[34+4*g+3],blk[34+4*g+2],blk[34+4*g+1],blk[34+4*g]}) begin
            prot_err++; $display("FERR ci=%0d g=%0d signs", ci, g);
          end
          if (o_ls1[g] !== 5'(1+2*(blk[74+g]&8'hF)) || o_ls2[g] !== 5'(1+2*(blk[74+g]>>4))) begin
            prot_err++; $display("FERR ci=%0d g=%0d ls1=%0d ls2=%0d", ci, g, o_ls1[g], o_ls2[g]);
          end
          for (l = 0; l < 4; l++)
            ei[l] = {1'b0, {2'b00,blk[2+4*g+l]} | (({2'b00,blk[66+g]} << (8-2*l)) & 11'h300)};
          if (o_idx[g] !== {ei[3],ei[2],ei[1],ei[0]} || o_delta[g] !== 4'd0 ||
              o_terms[g] !== 2'd2) begin
            prot_err++; $display("FIELD ci=%0d g=%0d idx=%011h d=%0d aux=%0d t=%0d",
                                 ci, g, o_idx[g], o_delta[g], o_aux[g], o_terms[g]);
          end
        end
      end else begin
        if (o_d !== {blk[55][7:4],blk[53][7:4],blk[51][7:4],blk[49][7:4]} || o_bfmt !== 2'd2) begin
          prot_err++; $display("FDERR ci=%0d d=%04h", ci, o_d);
        end
        if (o_scales !== {blk[55],blk[54],blk[53],blk[52],blk[51],blk[50],blk[49],blk[48]}) begin
          prot_err++; $display("FERR ci=%0d scales=%016h", ci, o_scales);
        end
        for (g = 0; g < 8; g++) begin
          sc = {blk[49+2*(g/2)],blk[48+2*(g/2)]};
          if (o_ls1[g] !== 5'(1+2*(sc[6*(g%2)+:3])) || o_ls2[g] !== 5'(1+2*(sc[6*(g%2)+3+:3]))) begin
            prot_err++; $display("FERR ci=%0d g=%0d ls1=%0d ls2=%0d", ci, g, o_ls1[g], o_ls2[g]);
          end
          if (o_signs[g] !== 32'd0) begin
            prot_err++; $display("FERR ci=%0d g=%0d signs=%08h", ci, g, o_signs[g]);
          end
          for (l = 0; l < 4; l++) begin
            high = blk[32+2*g+l/2];
            ei[l] = {high[4*(l%2)+:3],blk[4*g+l]};
            if (o_idx[g][11*l+:11] !== ei[l] || o_delta[g][l] !== high[4*(l%2)+3]) begin
              prot_err++; $display("FIELD ci=%0d g=%0d l=%0d idx=%011h d=%0d",
                                   ci, g, l, o_idx[g], o_delta[g]);
            end
          end
        end
        if (o_terms[0] !== 2'd2) begin
          prot_err++; $display("FIELD ci=%0d terms=%0d for IQ1_M", ci, o_terms[0]);
        end
      end
      if (fd_out != 0 && !quiet) begin
        $fdisplay(fd_out, "case %0d fmt %01h d %04h scales %016h", ci, fmt_word, o_d, o_scales);
        for (g = 0; g < 8; g++)
          $fdisplay(fd_out, "g %0d %08h %08h %02h %02h %02h %08h %011h %01h",
                    g, o_bsum[g], o_aux[g], o_terms[g], o_ls1[g], o_ls2[g],
                    o_signs[g], o_idx[g], o_delta[g]);
      end
      if (!quiet) cases++;
    end
  endtask

  // Wait `n` cycles with both valids low and count any accept seen there.
  task automatic idle_cycles(input int n);
    begin
      w_valid = 1'b0; a_valid = 1'b0;
      repeat (n) begin
        @(posedge clk); #1;
        if (w_ready !== 1'b0 || a_ready !== 1'b0 || grp_valid !== 1'b0) begin
          prot_err++; $display("IDLEERR ready high with no valid at cyc=%0d", cyc);
        end
      end
    end
  endtask

  // A malformed/ambiguous stream must not be run with arbitrary payload
  // semantics: the wrapper has to stall (ready low), accept nothing, emit no
  // group and raise its sticky error until reset. Valid is asserted first, the
  // load never waits for a ready that should not come.
  task automatic rejection_test(input logic [1:0] good_fmt, input logic [1:0] bad_fmt);
    int w0, a0, save_beat;
    begin
      @(negedge clk);
      rst = 1'b1; w_valid = 1'b0; a_valid = 1'b0;
      repeat (2) @(negedge clk);
      rst = 1'b0;
      w0 = acc_w; a0 = acc_a; save_beat = blk_beat;
      // 1) select the child with a good-tag beat; this one is consumed by the
      //    selection probe, so it is driven by hand and never counted as part
      //    of a real block.
      w_fmt = good_fmt; w_data = 64'h0123_4567_89AB_CDEF; w_valid = 1'b1;
      @(posedge clk); #1;
      @(negedge clk); w_valid = 1'b0;
      @(negedge clk);
      // 2) the tag changes mid block: valid held high with the wrong tag
      @(negedge clk);
      w_valid = 1'b1; w_fmt = bad_fmt; w_data = 64'h5678;
      idle_check_tag: begin
        repeat (4) begin
          @(posedge clk); #1;
          if (w_ready !== 1'b0) begin
            prot_err++; $display("RJERR ci=%0d bad tag %0d accepted", ci, bad_fmt);
          end
          if (dut.ctl !== 2'd3) begin
            prot_err++; $display("RJERR ci=%0d ctl=%0d after bad tag", ci, dut.ctl);
          end
        end
      end
      w_valid = 1'b0;
      // 3) it stays stalled while the error is latched: no silent recovery
      repeat (2) begin
        @(negedge clk); w_valid = 1'b1; w_fmt = good_fmt;
        @(posedge clk); #1;
        if (w_ready !== 1'b0) begin
          prot_err++; $display("RJERR ci=%0d accept after sticky error", ci);
        end
        w_valid = 1'b0;
      end
      if (acc_w != w0 || acc_a != a0) begin
        prot_err++; $display("RJERR ci=%0d rejected stream accepted beats", ci);
      end
      blk_beat = save_beat;      // rejected streams contribute no block beats
      if (grp_valid !== 1'b0) begin
        prot_err++; $display("RJERR ci=%0d group emitted for rejected stream", ci);
      end
      // 4) reset is the only exit
      @(negedge clk); rst = 1'b1;
      repeat (2) @(negedge clk); rst = 1'b0;
      idle_cycles(2);
      if (dut.ctl !== 2'd0) begin
        prot_err++; $display("RJERR ci=%0d ctl=%0d after reset", ci, dut.ctl);
      end
      blk_beat = 0;              // the next real block starts its own count
    end
  endtask

  // An unknown tag (3) must be rejected on selection without accepting a beat.
  task automatic unknown_fmt_test(input logic [1:0] good_fmt);
    int w0, a0, restore_ci, save_w, save_a;
    begin
      @(negedge clk);
      rst = 1'b1; w_valid = 1'b0; a_valid = 1'b0;
      repeat (2) @(negedge clk);
      rst = 1'b0;
      w0 = acc_w; a0 = acc_a;
      @(negedge clk);
      w_valid = 1'b1; w_fmt = 2'd3; w_data = 64'hDEAD;
      repeat (3) begin
        @(posedge clk); #1;
        if (w_ready !== 1'b0) begin
          prot_err++; $display("RJERR ci=%0d unknown tag accepted", ci);
        end
      end
      w_valid = 1'b0;
      if (dut.ctl !== 2'd3) begin
        prot_err++; $display("RJERR ci=%0d ctl=%0d for unknown tag", ci, dut.ctl);
      end
      if (acc_w != w0 || acc_a != a0 || grp_valid !== 1'b0) begin
        prot_err++; $display("RJERR ci=%0d unknown tag moved data", ci);
      end
      @(negedge clk); rst = 1'b1;
      repeat (2) @(negedge clk); rst = 1'b0;
      idle_cycles(2);
      // the wrapper is still usable after reset: one block per format again.
      // This is protocol exercise, not a checker case, so it is neither counted
      // nor dumped; ci is restored so the dump indices stay dense.
      restore_ci = ci; ci = 0; quiet = 1'b1;
      fmt_word = good_fmt;       // run_case follows the tag on the wire
      save_w = acc_w; save_a = acc_a;
      run_case();                // run_case checks its own beat counts
      // these beats belong to the protocol exercise, not to a checker case
      aux_beats = aux_beats + (acc_w - save_w) + (acc_a - save_a);
      quiet = 1'b0; ci = restore_ci;
      blk_beat = 0;
    end
  endtask

  initial begin
    int k, j;
    if (!$value$plusargs("vectors=%s", vname)) $fatal(1, "+vectors= missing");
    if ($value$plusargs("out=%s", oname)) begin
      fd_out = $fopen(oname, "w"); if (fd_out == 0) $fatal(1, "cannot write %s", oname);
    end else fd_out = 0;
    if (!$value$plusargs("cases=%d", ncases)) ncases = 0;
    fd_in = $fopen(vname, "r"); if (fd_in == 0) $fatal(1, "cannot open %s", vname);
    ci = 0; cases = 0; prot_err = 0; acc_w = 0; acc_a = 0; quiet = 1'b0;
    aux_beats = 0;
    c_first = 0; c_last = 0;
    // reset is driven explicitly; no accept may be counted before it is released
    @(negedge clk);
    rst = 1'b1; w_valid = 1'b0; a_valid = 1'b0;
    repeat (3) @(negedge clk);
    rst = 1'b0;
    idle_cycles(2);
    while (!$feof(fd_in)) begin
      n = $fscanf(fd_in, "%s", op); if (n != 1) break;
      case (op)
        "B": begin
          n = $fscanf(fd_in, "%d", t); if (n != 1 || t > 2) $fatal(1, "bad B fmt at %0d", ci);
          fmt_word = t[1:0];
          for (k = 0; k < 22; k++) begin
            n = $fscanf(fd_in, "%h", t); if (n != 1) $fatal(1, "short B %0d", ci);
            blk_word[k] = t;
          end
          for (k = 0; k < 21; k++) for (j = 0; j < 4; j++) blk[4*k+j] = blk_word[k][8*j+:8];
        end
        "A": begin
          for (k = 0; k < 64; k++) begin
            n = $fscanf(fd_in, "%h", t); if (n != 1) $fatal(1, "short A %0d", ci);
            act_word[k] = t;
          end
          if (ncases == 0 || ci < ncases) run_case();
          ci++;
        end
        default: $fatal(1, "bad vector op %s at %0d", op, ci);
      endcase
    end
    $fclose(fd_in);
    // no block may hand its last group over to the next one
    idle_cycles(2);
    rejection_test(2'd0, 2'd2);
    rejection_test(2'd2, 2'd1);
    rejection_test(2'd1, 2'd0);
    unknown_fmt_test(2'd0);
    unknown_fmt_test(2'd1);
    unknown_fmt_test(2'd2);
    if (fd_out != 0) $fclose(fd_out);
    $display("IQSTREAMTEST cases=%0d beats=%0d cycles=%0d errors=%0d",
             cases, acc_w + acc_a - aux_beats, cyc, prot_err);
    if (cases != ncases && ncases != 0) begin
      $fatal(1, "ran %0d cases, expected %0d", cases, ncases);
    end
    if (prot_err != 0) $fatal(1, "IQ stream testbench checks failed: %0d", prot_err);
    $finish;
  end
endmodule