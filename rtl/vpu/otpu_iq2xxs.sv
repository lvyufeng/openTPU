// SPDX-License-Identifier: MIT for the transcribed ROM data; the decoder is Apache-2.0.
// ROM data: Copyright (c) 2026 Niko1221 and the Strata contributors;
// Copyright (c) 2023-2026 The ggml authors and (c) 2023 Georgi Gerganov and the ggml contributors.
// The data and full notice, not a change of license: tools/packed_oracle/licenses/{STRATA,GGML}-MIT.txt.
// Sources pinned at Strata 3142b66b96e9c0e775e30f620b11b191a1427c2a, and identical
// in the source inventory 97affb754d298b2cce1fc7b12338797ac155aa9b.  See THIRD_PARTY.md.

// IQ2_XXS, 66 bytes / 256 weights.  Each group is four 8-bit indexes followed by
// a 32-bit word: four 7-bit sign-table indexes in bits 7l..7l+6, and one 4-bit
// group scale in bits 28..31.  Signs are KSIGNS lookups, including bit 7; they
// are not raw sign bytes.  bsum = (2 * magnitude + 1) * sum of all four signed
// grid rows.  The caller folds 0.125 * fp16(d) * the sum of group bsums.
// Nine 64-bit weight beats (66 used bytes plus six pad), eight 256-bit activation
// beats.  Fixed single-buffered load/dot phases: 17 accepted cycles per block.
// Integer bound: 31 * 2 * 43 * 128 * 32 = 10919936, so int26 (one headroom bit).
// Scalar format specification: ggml-cpu/quants.c:906-946 and ggml-quants.c:2488-2512;
// layers: ggml-common.h:371-376 and :552-617.  Device/VPU integration is not present.
module otpu_iq2xxs (
  input  logic         clk,
  input  logic         rst,
  input  logic         w_valid,
  output logic         w_ready,
  input  logic [63:0]  w_data,
  input  logic         a_valid,
  output logic         a_ready,
  input  logic [255:0] a_data,

  output logic         grp_valid,
  output logic [2:0]   grp_idx,
  output logic [31:0]  grp_bsum,
  output logic [31:0]  grp_aux,
  output logic [1:0]   grp_terms,
  output logic [4:0]   grp_ls1,
  output logic [4:0]   grp_ls2,
  output logic [31:0]  grp_signs,
  output logic [43:0]  grp_index,
  output logic [3:0]   grp_delta,
  output logic         blk_valid,
  output logic [15:0]  blk_d,
  output logic [63:0]  blk_scales
);

  localparam logic [3:0] LAST_BEAT = 4'd8;
  logic state;
  logic [3:0] w_cnt;
  logic [2:0] a_cnt;
  logic [575:0] blk;
  localparam logic S_LOAD = 1'b0, S_ACT = 1'b1;
  assign w_ready = !rst && (state == S_LOAD);
  assign a_ready = !rst && (state == S_ACT);

  logic [63:0] iq2xxs_rom [0:255];
  logic [7:0] ksigns_rom [0:127];
  initial begin
    iq2xxs_rom[   0] = 64'h0808080808080808;
    iq2xxs_rom[   1] = 64'h080808080808082B;
    iq2xxs_rom[   2] = 64'h0808080808081919;
    iq2xxs_rom[   3] = 64'h0808080808082B08;
    iq2xxs_rom[   4] = 64'h0808080808082B2B;
    iq2xxs_rom[   5] = 64'h0808080808190819;
    iq2xxs_rom[   6] = 64'h0808080808191908;
    iq2xxs_rom[   7] = 64'h08080808082B0808;
    iq2xxs_rom[   8] = 64'h08080808082B082B;
    iq2xxs_rom[   9] = 64'h08080808082B2B08;
    iq2xxs_rom[  10] = 64'h08080808082B2B2B;
    iq2xxs_rom[  11] = 64'h0808080819080819;
    iq2xxs_rom[  12] = 64'h0808080819081908;
    iq2xxs_rom[  13] = 64'h0808080819190808;
    iq2xxs_rom[  14] = 64'h0808080819192B08;
    iq2xxs_rom[  15] = 64'h08080808192B0819;
    iq2xxs_rom[  16] = 64'h08080808192B1908;
    iq2xxs_rom[  17] = 64'h080808082B080808;
    iq2xxs_rom[  18] = 64'h080808082B08082B;
    iq2xxs_rom[  19] = 64'h080808082B082B2B;
    iq2xxs_rom[  20] = 64'h080808082B2B082B;
    iq2xxs_rom[  21] = 64'h0808081908080819;
    iq2xxs_rom[  22] = 64'h0808081908081908;
    iq2xxs_rom[  23] = 64'h0808081908190808;
    iq2xxs_rom[  24] = 64'h0808081908191919;
    iq2xxs_rom[  25] = 64'h0808081919080808;
    iq2xxs_rom[  26] = 64'h080808192B081908;
    iq2xxs_rom[  27] = 64'h080808192B192B08;
    iq2xxs_rom[  28] = 64'h0808082B08080808;
    iq2xxs_rom[  29] = 64'h0808082B0808082B;
    iq2xxs_rom[  30] = 64'h0808082B082B082B;
    iq2xxs_rom[  31] = 64'h0808082B2B08082B;
    iq2xxs_rom[  32] = 64'h0808190808080819;
    iq2xxs_rom[  33] = 64'h0808190808081908;
    iq2xxs_rom[  34] = 64'h0808190808190808;
    iq2xxs_rom[  35] = 64'h08081908082B0819;
    iq2xxs_rom[  36] = 64'h08081908082B1908;
    iq2xxs_rom[  37] = 64'h0808190819080808;
    iq2xxs_rom[  38] = 64'h080819081908082B;
    iq2xxs_rom[  39] = 64'h0808190819082B08;
    iq2xxs_rom[  40] = 64'h08081908192B0808;
    iq2xxs_rom[  41] = 64'h080819082B080819;
    iq2xxs_rom[  42] = 64'h080819082B081908;
    iq2xxs_rom[  43] = 64'h080819082B190808;
    iq2xxs_rom[  44] = 64'h080819082B2B1908;
    iq2xxs_rom[  45] = 64'h0808191908080808;
    iq2xxs_rom[  46] = 64'h080819190808082B;
    iq2xxs_rom[  47] = 64'h0808191908082B08;
    iq2xxs_rom[  48] = 64'h08081919082B0808;
    iq2xxs_rom[  49] = 64'h080819191908192B;
    iq2xxs_rom[  50] = 64'h08081919192B2B19;
    iq2xxs_rom[  51] = 64'h080819192B080808;
    iq2xxs_rom[  52] = 64'h080819192B190819;
    iq2xxs_rom[  53] = 64'h0808192B08082B19;
    iq2xxs_rom[  54] = 64'h0808192B08190808;
    iq2xxs_rom[  55] = 64'h0808192B19080808;
    iq2xxs_rom[  56] = 64'h0808192B2B081908;
    iq2xxs_rom[  57] = 64'h0808192B2B2B1908;
    iq2xxs_rom[  58] = 64'h08082B0808080808;
    iq2xxs_rom[  59] = 64'h08082B0808081919;
    iq2xxs_rom[  60] = 64'h08082B0808082B08;
    iq2xxs_rom[  61] = 64'h08082B0808191908;
    iq2xxs_rom[  62] = 64'h08082B08082B2B08;
    iq2xxs_rom[  63] = 64'h08082B0819080819;
    iq2xxs_rom[  64] = 64'h08082B0819081908;
    iq2xxs_rom[  65] = 64'h08082B0819190808;
    iq2xxs_rom[  66] = 64'h08082B081919082B;
    iq2xxs_rom[  67] = 64'h08082B082B082B08;
    iq2xxs_rom[  68] = 64'h08082B1908081908;
    iq2xxs_rom[  69] = 64'h08082B1919080808;
    iq2xxs_rom[  70] = 64'h08082B2B0808082B;
    iq2xxs_rom[  71] = 64'h08082B2B08191908;
    iq2xxs_rom[  72] = 64'h0819080808080819;
    iq2xxs_rom[  73] = 64'h0819080808081908;
    iq2xxs_rom[  74] = 64'h0819080808190808;
    iq2xxs_rom[  75] = 64'h08190808082B0819;
    iq2xxs_rom[  76] = 64'h0819080819080808;
    iq2xxs_rom[  77] = 64'h08190808192B0808;
    iq2xxs_rom[  78] = 64'h081908082B081908;
    iq2xxs_rom[  79] = 64'h081908082B190808;
    iq2xxs_rom[  80] = 64'h081908082B191919;
    iq2xxs_rom[  81] = 64'h0819081908080808;
    iq2xxs_rom[  82] = 64'h0819081908082B08;
    iq2xxs_rom[  83] = 64'h08190819082B0808;
    iq2xxs_rom[  84] = 64'h0819081919190808;
    iq2xxs_rom[  85] = 64'h0819081919192B2B;
    iq2xxs_rom[  86] = 64'h081908192B080808;
    iq2xxs_rom[  87] = 64'h0819082B082B1908;
    iq2xxs_rom[  88] = 64'h0819082B19081919;
    iq2xxs_rom[  89] = 64'h0819190808080808;
    iq2xxs_rom[  90] = 64'h0819190808082B08;
    iq2xxs_rom[  91] = 64'h08191908082B0808;
    iq2xxs_rom[  92] = 64'h08191908082B1919;
    iq2xxs_rom[  93] = 64'h0819190819082B19;
    iq2xxs_rom[  94] = 64'h081919082B080808;
    iq2xxs_rom[  95] = 64'h0819191908192B08;
    iq2xxs_rom[  96] = 64'h08191919192B082B;
    iq2xxs_rom[  97] = 64'h0819192B08080808;
    iq2xxs_rom[  98] = 64'h0819192B0819192B;
    iq2xxs_rom[  99] = 64'h08192B0808080819;
    iq2xxs_rom[ 100] = 64'h08192B0808081908;
    iq2xxs_rom[ 101] = 64'h08192B0808190808;
    iq2xxs_rom[ 102] = 64'h08192B0819080808;
    iq2xxs_rom[ 103] = 64'h08192B082B080819;
    iq2xxs_rom[ 104] = 64'h08192B1908080808;
    iq2xxs_rom[ 105] = 64'h08192B1908081919;
    iq2xxs_rom[ 106] = 64'h08192B192B2B0808;
    iq2xxs_rom[ 107] = 64'h08192B2B19190819;
    iq2xxs_rom[ 108] = 64'h082B080808080808;
    iq2xxs_rom[ 109] = 64'h082B08080808082B;
    iq2xxs_rom[ 110] = 64'h082B080808082B2B;
    iq2xxs_rom[ 111] = 64'h082B080819081908;
    iq2xxs_rom[ 112] = 64'h082B0808192B0819;
    iq2xxs_rom[ 113] = 64'h082B08082B080808;
    iq2xxs_rom[ 114] = 64'h082B08082B08082B;
    iq2xxs_rom[ 115] = 64'h082B0819082B2B19;
    iq2xxs_rom[ 116] = 64'h082B081919082B08;
    iq2xxs_rom[ 117] = 64'h082B082B08080808;
    iq2xxs_rom[ 118] = 64'h082B082B0808082B;
    iq2xxs_rom[ 119] = 64'h082B190808080819;
    iq2xxs_rom[ 120] = 64'h082B190808081908;
    iq2xxs_rom[ 121] = 64'h082B190808190808;
    iq2xxs_rom[ 122] = 64'h082B190819080808;
    iq2xxs_rom[ 123] = 64'h082B19081919192B;
    iq2xxs_rom[ 124] = 64'h082B191908080808;
    iq2xxs_rom[ 125] = 64'h082B191919080819;
    iq2xxs_rom[ 126] = 64'h082B1919192B1908;
    iq2xxs_rom[ 127] = 64'h082B192B2B190808;
    iq2xxs_rom[ 128] = 64'h082B2B0808082B08;
    iq2xxs_rom[ 129] = 64'h082B2B08082B0808;
    iq2xxs_rom[ 130] = 64'h082B2B082B191908;
    iq2xxs_rom[ 131] = 64'h082B2B2B19081908;
    iq2xxs_rom[ 132] = 64'h1908080808080819;
    iq2xxs_rom[ 133] = 64'h1908080808081908;
    iq2xxs_rom[ 134] = 64'h1908080808190808;
    iq2xxs_rom[ 135] = 64'h1908080808192B08;
    iq2xxs_rom[ 136] = 64'h19080808082B0819;
    iq2xxs_rom[ 137] = 64'h19080808082B1908;
    iq2xxs_rom[ 138] = 64'h1908080819080808;
    iq2xxs_rom[ 139] = 64'h1908080819082B08;
    iq2xxs_rom[ 140] = 64'h190808081919192B;
    iq2xxs_rom[ 141] = 64'h19080808192B0808;
    iq2xxs_rom[ 142] = 64'h190808082B080819;
    iq2xxs_rom[ 143] = 64'h190808082B081908;
    iq2xxs_rom[ 144] = 64'h190808082B190808;
    iq2xxs_rom[ 145] = 64'h1908081908080808;
    iq2xxs_rom[ 146] = 64'h19080819082B0808;
    iq2xxs_rom[ 147] = 64'h19080819192B0819;
    iq2xxs_rom[ 148] = 64'h190808192B080808;
    iq2xxs_rom[ 149] = 64'h190808192B081919;
    iq2xxs_rom[ 150] = 64'h1908082B08080819;
    iq2xxs_rom[ 151] = 64'h1908082B08190808;
    iq2xxs_rom[ 152] = 64'h1908082B19082B08;
    iq2xxs_rom[ 153] = 64'h1908082B1919192B;
    iq2xxs_rom[ 154] = 64'h1908082B192B2B08;
    iq2xxs_rom[ 155] = 64'h1908190808080808;
    iq2xxs_rom[ 156] = 64'h1908190808082B08;
    iq2xxs_rom[ 157] = 64'h19081908082B0808;
    iq2xxs_rom[ 158] = 64'h190819082B080808;
    iq2xxs_rom[ 159] = 64'h190819082B192B19;
    iq2xxs_rom[ 160] = 64'h190819190819082B;
    iq2xxs_rom[ 161] = 64'h19081919082B1908;
    iq2xxs_rom[ 162] = 64'h1908192B08080808;
    iq2xxs_rom[ 163] = 64'h19082B0808080819;
    iq2xxs_rom[ 164] = 64'h19082B0808081908;
    iq2xxs_rom[ 165] = 64'h19082B0808190808;
    iq2xxs_rom[ 166] = 64'h19082B0819080808;
    iq2xxs_rom[ 167] = 64'h19082B0819081919;
    iq2xxs_rom[ 168] = 64'h19082B1908080808;
    iq2xxs_rom[ 169] = 64'h19082B1919192B08;
    iq2xxs_rom[ 170] = 64'h19082B19192B0819;
    iq2xxs_rom[ 171] = 64'h19082B192B08082B;
    iq2xxs_rom[ 172] = 64'h19082B2B19081919;
    iq2xxs_rom[ 173] = 64'h19082B2B2B190808;
    iq2xxs_rom[ 174] = 64'h1919080808080808;
    iq2xxs_rom[ 175] = 64'h1919080808082B08;
    iq2xxs_rom[ 176] = 64'h1919080808190819;
    iq2xxs_rom[ 177] = 64'h1919080808192B19;
    iq2xxs_rom[ 178] = 64'h19190808082B0808;
    iq2xxs_rom[ 179] = 64'h191908082B080808;
    iq2xxs_rom[ 180] = 64'h191908082B082B08;
    iq2xxs_rom[ 181] = 64'h1919081908081908;
    iq2xxs_rom[ 182] = 64'h191908191908082B;
    iq2xxs_rom[ 183] = 64'h191908192B2B1908;
    iq2xxs_rom[ 184] = 64'h1919082B2B190819;
    iq2xxs_rom[ 185] = 64'h191919082B190808;
    iq2xxs_rom[ 186] = 64'h191919082B19082B;
    iq2xxs_rom[ 187] = 64'h1919191908082B2B;
    iq2xxs_rom[ 188] = 64'h1919192B08080819;
    iq2xxs_rom[ 189] = 64'h1919192B19191908;
    iq2xxs_rom[ 190] = 64'h19192B0808080808;
    iq2xxs_rom[ 191] = 64'h19192B0808190819;
    iq2xxs_rom[ 192] = 64'h19192B0808192B19;
    iq2xxs_rom[ 193] = 64'h19192B08192B1908;
    iq2xxs_rom[ 194] = 64'h19192B1919080808;
    iq2xxs_rom[ 195] = 64'h19192B2B08082B08;
    iq2xxs_rom[ 196] = 64'h192B080808081908;
    iq2xxs_rom[ 197] = 64'h192B080808190808;
    iq2xxs_rom[ 198] = 64'h192B080819080808;
    iq2xxs_rom[ 199] = 64'h192B0808192B2B08;
    iq2xxs_rom[ 200] = 64'h192B081908080808;
    iq2xxs_rom[ 201] = 64'h192B081919191919;
    iq2xxs_rom[ 202] = 64'h192B082B08192B08;
    iq2xxs_rom[ 203] = 64'h192B082B192B0808;
    iq2xxs_rom[ 204] = 64'h192B190808080808;
    iq2xxs_rom[ 205] = 64'h192B190808081919;
    iq2xxs_rom[ 206] = 64'h192B191908190808;
    iq2xxs_rom[ 207] = 64'h192B19190819082B;
    iq2xxs_rom[ 208] = 64'h192B19192B081908;
    iq2xxs_rom[ 209] = 64'h192B2B081908082B;
    iq2xxs_rom[ 210] = 64'h2B08080808080808;
    iq2xxs_rom[ 211] = 64'h2B0808080808082B;
    iq2xxs_rom[ 212] = 64'h2B08080808082B2B;
    iq2xxs_rom[ 213] = 64'h2B08080819080819;
    iq2xxs_rom[ 214] = 64'h2B0808082B08082B;
    iq2xxs_rom[ 215] = 64'h2B08081908081908;
    iq2xxs_rom[ 216] = 64'h2B08081908192B08;
    iq2xxs_rom[ 217] = 64'h2B08081919080808;
    iq2xxs_rom[ 218] = 64'h2B08082B08190819;
    iq2xxs_rom[ 219] = 64'h2B08190808080819;
    iq2xxs_rom[ 220] = 64'h2B08190808081908;
    iq2xxs_rom[ 221] = 64'h2B08190808190808;
    iq2xxs_rom[ 222] = 64'h2B08190808191919;
    iq2xxs_rom[ 223] = 64'h2B08190819080808;
    iq2xxs_rom[ 224] = 64'h2B081908192B0808;
    iq2xxs_rom[ 225] = 64'h2B08191908080808;
    iq2xxs_rom[ 226] = 64'h2B0819191908192B;
    iq2xxs_rom[ 227] = 64'h2B0819192B191908;
    iq2xxs_rom[ 228] = 64'h2B08192B08082B19;
    iq2xxs_rom[ 229] = 64'h2B08192B19080808;
    iq2xxs_rom[ 230] = 64'h2B08192B192B0808;
    iq2xxs_rom[ 231] = 64'h2B082B080808082B;
    iq2xxs_rom[ 232] = 64'h2B082B1908081908;
    iq2xxs_rom[ 233] = 64'h2B082B2B08190819;
    iq2xxs_rom[ 234] = 64'h2B19080808081908;
    iq2xxs_rom[ 235] = 64'h2B19080808190808;
    iq2xxs_rom[ 236] = 64'h2B190808082B1908;
    iq2xxs_rom[ 237] = 64'h2B19080819080808;
    iq2xxs_rom[ 238] = 64'h2B1908082B2B0819;
    iq2xxs_rom[ 239] = 64'h2B1908190819192B;
    iq2xxs_rom[ 240] = 64'h2B1908192B080808;
    iq2xxs_rom[ 241] = 64'h2B19082B19081919;
    iq2xxs_rom[ 242] = 64'h2B19190808080808;
    iq2xxs_rom[ 243] = 64'h2B191908082B082B;
    iq2xxs_rom[ 244] = 64'h2B19190819081908;
    iq2xxs_rom[ 245] = 64'h2B19191919190819;
    iq2xxs_rom[ 246] = 64'h2B192B082B080819;
    iq2xxs_rom[ 247] = 64'h2B192B19082B0808;
    iq2xxs_rom[ 248] = 64'h2B2B08080808082B;
    iq2xxs_rom[ 249] = 64'h2B2B080819190808;
    iq2xxs_rom[ 250] = 64'h2B2B08082B081919;
    iq2xxs_rom[ 251] = 64'h2B2B081908082B19;
    iq2xxs_rom[ 252] = 64'h2B2B082B08080808;
    iq2xxs_rom[ 253] = 64'h2B2B190808192B08;
    iq2xxs_rom[ 254] = 64'h2B2B2B0819190808;
    iq2xxs_rom[ 255] = 64'h2B2B2B1908081908;
    ksigns_rom[   0] = 8'h00;
    ksigns_rom[   1] = 8'h81;
    ksigns_rom[   2] = 8'h82;
    ksigns_rom[   3] = 8'h03;
    ksigns_rom[   4] = 8'h84;
    ksigns_rom[   5] = 8'h05;
    ksigns_rom[   6] = 8'h06;
    ksigns_rom[   7] = 8'h87;
    ksigns_rom[   8] = 8'h88;
    ksigns_rom[   9] = 8'h09;
    ksigns_rom[  10] = 8'h0A;
    ksigns_rom[  11] = 8'h8B;
    ksigns_rom[  12] = 8'h0C;
    ksigns_rom[  13] = 8'h8D;
    ksigns_rom[  14] = 8'h8E;
    ksigns_rom[  15] = 8'h0F;
    ksigns_rom[  16] = 8'h90;
    ksigns_rom[  17] = 8'h11;
    ksigns_rom[  18] = 8'h12;
    ksigns_rom[  19] = 8'h93;
    ksigns_rom[  20] = 8'h14;
    ksigns_rom[  21] = 8'h95;
    ksigns_rom[  22] = 8'h96;
    ksigns_rom[  23] = 8'h17;
    ksigns_rom[  24] = 8'h18;
    ksigns_rom[  25] = 8'h99;
    ksigns_rom[  26] = 8'h9A;
    ksigns_rom[  27] = 8'h1B;
    ksigns_rom[  28] = 8'h9C;
    ksigns_rom[  29] = 8'h1D;
    ksigns_rom[  30] = 8'h1E;
    ksigns_rom[  31] = 8'h9F;
    ksigns_rom[  32] = 8'hA0;
    ksigns_rom[  33] = 8'h21;
    ksigns_rom[  34] = 8'h22;
    ksigns_rom[  35] = 8'hA3;
    ksigns_rom[  36] = 8'h24;
    ksigns_rom[  37] = 8'hA5;
    ksigns_rom[  38] = 8'hA6;
    ksigns_rom[  39] = 8'h27;
    ksigns_rom[  40] = 8'h28;
    ksigns_rom[  41] = 8'hA9;
    ksigns_rom[  42] = 8'hAA;
    ksigns_rom[  43] = 8'h2B;
    ksigns_rom[  44] = 8'hAC;
    ksigns_rom[  45] = 8'h2D;
    ksigns_rom[  46] = 8'h2E;
    ksigns_rom[  47] = 8'hAF;
    ksigns_rom[  48] = 8'h30;
    ksigns_rom[  49] = 8'hB1;
    ksigns_rom[  50] = 8'hB2;
    ksigns_rom[  51] = 8'h33;
    ksigns_rom[  52] = 8'hB4;
    ksigns_rom[  53] = 8'h35;
    ksigns_rom[  54] = 8'h36;
    ksigns_rom[  55] = 8'hB7;
    ksigns_rom[  56] = 8'hB8;
    ksigns_rom[  57] = 8'h39;
    ksigns_rom[  58] = 8'h3A;
    ksigns_rom[  59] = 8'hBB;
    ksigns_rom[  60] = 8'h3C;
    ksigns_rom[  61] = 8'hBD;
    ksigns_rom[  62] = 8'hBE;
    ksigns_rom[  63] = 8'h3F;
    ksigns_rom[  64] = 8'hC0;
    ksigns_rom[  65] = 8'h41;
    ksigns_rom[  66] = 8'h42;
    ksigns_rom[  67] = 8'hC3;
    ksigns_rom[  68] = 8'h44;
    ksigns_rom[  69] = 8'hC5;
    ksigns_rom[  70] = 8'hC6;
    ksigns_rom[  71] = 8'h47;
    ksigns_rom[  72] = 8'h48;
    ksigns_rom[  73] = 8'hC9;
    ksigns_rom[  74] = 8'hCA;
    ksigns_rom[  75] = 8'h4B;
    ksigns_rom[  76] = 8'hCC;
    ksigns_rom[  77] = 8'h4D;
    ksigns_rom[  78] = 8'h4E;
    ksigns_rom[  79] = 8'hCF;
    ksigns_rom[  80] = 8'h50;
    ksigns_rom[  81] = 8'hD1;
    ksigns_rom[  82] = 8'hD2;
    ksigns_rom[  83] = 8'h53;
    ksigns_rom[  84] = 8'hD4;
    ksigns_rom[  85] = 8'h55;
    ksigns_rom[  86] = 8'h56;
    ksigns_rom[  87] = 8'hD7;
    ksigns_rom[  88] = 8'hD8;
    ksigns_rom[  89] = 8'h59;
    ksigns_rom[  90] = 8'h5A;
    ksigns_rom[  91] = 8'hDB;
    ksigns_rom[  92] = 8'h5C;
    ksigns_rom[  93] = 8'hDD;
    ksigns_rom[  94] = 8'hDE;
    ksigns_rom[  95] = 8'h5F;
    ksigns_rom[  96] = 8'h60;
    ksigns_rom[  97] = 8'hE1;
    ksigns_rom[  98] = 8'hE2;
    ksigns_rom[  99] = 8'h63;
    ksigns_rom[ 100] = 8'hE4;
    ksigns_rom[ 101] = 8'h65;
    ksigns_rom[ 102] = 8'h66;
    ksigns_rom[ 103] = 8'hE7;
    ksigns_rom[ 104] = 8'hE8;
    ksigns_rom[ 105] = 8'h69;
    ksigns_rom[ 106] = 8'h6A;
    ksigns_rom[ 107] = 8'hEB;
    ksigns_rom[ 108] = 8'h6C;
    ksigns_rom[ 109] = 8'hED;
    ksigns_rom[ 110] = 8'hEE;
    ksigns_rom[ 111] = 8'h6F;
    ksigns_rom[ 112] = 8'hF0;
    ksigns_rom[ 113] = 8'h71;
    ksigns_rom[ 114] = 8'h72;
    ksigns_rom[ 115] = 8'hF3;
    ksigns_rom[ 116] = 8'h74;
    ksigns_rom[ 117] = 8'hF5;
    ksigns_rom[ 118] = 8'hF6;
    ksigns_rom[ 119] = 8'h77;
    ksigns_rom[ 120] = 8'h78;
    ksigns_rom[ 121] = 8'hF9;
    ksigns_rom[ 122] = 8'hFA;
    ksigns_rom[ 123] = 8'h7B;
    ksigns_rom[ 124] = 8'hFC;
    ksigns_rom[ 125] = 8'h7D;
    ksigns_rom[ 126] = 8'h7E;
    ksigns_rom[ 127] = 8'hFF;
  end

  logic [31:0] index_cur, aux_cur;
  logic [6:0] sign_index [4];
  logic [31:0] sign_cur;
  logic [63:0] mag [4];
  logic [4:0] scale_cur;
  logic signed [17:0] dot_l [4];
  logic signed [19:0] sum_c, sum_ext;
  logic signed [25:0] bsum_c, ls_s;
  always_comb begin
    index_cur = blk[8 * (2 + 8 * a_cnt) +: 32];
    aux_cur = blk[8 * (2 + 8 * a_cnt + 4) +: 32];
    scale_cur = 5'd1 + 5'd2 * {1'b0, aux_cur[31:28]};
    for (int l = 0; l < 4; l++) begin
      sign_index[l] = aux_cur[7 * l +: 7];
      sign_cur[8 * l +: 8] = ksigns_rom[sign_index[l]];
      mag[l] = iq2xxs_rom[index_cur[8 * l +: 8]];
    end
  end

  function automatic logic signed [17:0] dot8(input logic [63:0] mag_v,
                                              input logic [7:0] sign_v,
                                              input logic [63:0] act_v);
    logic signed [15:0] m, x, p;
    logic signed [17:0] acc;
    acc = 18'sd0;
    for (int j = 0; j < 8; j++) begin
      m = signed'({8'b0, mag_v[8 * j +: 8]});
      if (sign_v[j]) m = -m;
      x = signed'({{8{act_v[8 * j + 7]}}, act_v[8 * j +: 8]});
      p = m * x;
      acc = acc + 18'(p);
    end
    dot8 = acc;
  endfunction
  always_comb begin
    for (int l = 0; l < 4; l++)
      dot_l[l] = dot8(mag[l], sign_cur[8 * l +: 8], a_data[64 * l +: 64]);
    sum_c = dot_l[0] + dot_l[1] + dot_l[2] + dot_l[3];
    sum_ext = {{6{sum_c[19]}}, sum_c};
    ls_s = signed'({21'b0, scale_cur});
    bsum_c = ls_s * sum_ext;
  end
  always_ff @(posedge clk) begin
    if (rst) begin
      state <= S_LOAD;
      w_cnt <= '0;
      a_cnt <= '0;
      grp_valid <= 1'b0;
      grp_idx <= '0;
      grp_bsum <= '0;
      grp_aux <= '0;
      grp_terms <= '0;
      grp_ls1 <= '0;
      grp_ls2 <= '0;
      grp_signs <= '0;
      grp_index <= '0;
      grp_delta <= '0;
      blk_valid <= 1'b0;
      blk_d <= '0;
      blk_scales <= '0;
    end else begin
      grp_valid <= 1'b0;
      blk_valid <= 1'b0;
      if (state == S_LOAD) begin
        if (w_valid) begin
          blk[64 * w_cnt +: 64] <= w_data;
          if (w_cnt == LAST_BEAT) begin
            state <= S_ACT;
            a_cnt <= '0;
          end else begin
            w_cnt <= w_cnt + 1'b1;
          end
        end
      end else if (a_valid) begin
        grp_valid <= 1'b1;
        grp_idx <= a_cnt;
        grp_bsum <= {{6{bsum_c[25]}}, bsum_c};
        grp_aux <= 32'd0;
        grp_terms <= 2'd1;
        grp_ls1 <= scale_cur;
        grp_ls2 <= 5'd0;
        grp_signs <= sign_cur;
        grp_index <= {3'b000, index_cur[31:24], 3'b000, index_cur[23:16],
                      3'b000, index_cur[15:8], 3'b000, index_cur[7:0]};
        grp_delta <= 4'd0;
        if (a_cnt == 3'd0) begin
          blk_valid <= 1'b1;
          blk_d <= blk[15:0];
          for (int g = 0; g < 8; g++)
            blk_scales[8 * g +: 8] <= {4'b0000, blk[8 * (2 + 8 * g + 7) + 4 +: 4]};
        end
        if (a_cnt == 3'd7) begin
          state <= S_LOAD;
          w_cnt <= '0;
        end else begin
          a_cnt <= a_cnt + 3'd1;
        end
      end
    end
  end
endmodule
