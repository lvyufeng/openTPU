// IQ2_S block decode and dot, native: a 256-weight block's eight groups of 32 2-bit weights, each
// group dotted with its 32 int8 activations, to eight int32 partials beside the block's fp16 scale
// and eight raw scale bytes (docs/iq_dot.md; the format is ggml's, block_iq2_s in ggml-common.h:
// 396, pinned at 97affb754).
//
// A block is 82 bytes: d (fp16, bytes 0..1), qs (bytes 2..65: four index bytes then four sign bytes
// per group), qh (bytes 66..73, two index bits per sub-block), scales (bytes 74..81, the two 4-bit
// multipliers of a group's sub-block pairs). Group g has four sub-blocks l = 0..3 of eight weights
// each. The sub-block's grid entry is
//   idx = qs[4g + l] | ((qh[g] << (8 - 2l)) & 0x300)
// into iq2s_grid: 1024 x 64 bit, eight unsigned magnitudes in {8, 25, 43} per entry (ggml-common.h:
// 758), highest element in the entry's top byte. The signs are the bits of the sign byte
// qs[32 + 4g + l]: bit j set means element j is negative. With sumi1 the dot of l = 0, 1 and sumi2
// the dot of l = 2, 3, the group's partial is
//   bsum[g] = ls1 * sumi1 + ls2 * sumi2,  ls1 = 1 + 2 * (scales[g] & 15),
//                                         ls2 = 1 + 2 * (scales[g] >> 4).
// That is the arithmetic of the generic dot (ggml-cpu/quants.c:998) and of the dequant
// (ggml-quants.c:2543, grid at 2561) element for element: no rounding anywhere, and the block's
// value is 0.125 * fp16_to_fp32(d) * sum_g bsum[g], where the 0.125 is the dequant's
// (0.5 + n) * 0.25 = (2n + 1) / 8 written into the grid element's scale. This module stops at the
// integer partials; scaling d and summing them is the caller's (float) work.
//
// Every partial here is exact in integer: the magnitudes are at most 43 (six bits), |int8| <= 128,
// so an element's product is at most 43 * 128 = 5504, a sub-block's eight at most 44,032, a pair of
// sub-blocks at most 88,064 (int18) and a group's at most 31 * 2 * 88,064 = 5,459,968: inside this
// module's int24 with 1.53x to spare and far inside the output's int32. The datapath is two's
// complement with no saturation; a format with more than 32 weights a group would need the partial
// widened, not the arithmetic changed.
//
// Streaming: eleven 64-bit weight beats (88 bytes, the block's 82 and six pad; byte 8b + j is lane
// j of beat b), then eight 256-bit activation beats, one 32-weight group each (byte 8l + j is
// sub-block l's element j, i.e. activation word bytes 8l..8l+7 dotted with sub-block l). A beat is
// accepted on the edge where *valid and *ready are high; a group's result is registered on its
// accepting edge, so grp_valid, grp_idx, grp_bsum, grp_ls1, grp_ls2, grp_signs and grp_index are
// valid one cycle after the beat. grp_aux separately exposes ls2 * sumi2 without changing the
// combined grp_bsum contract. blk_valid, blk_d and blk_scales (bytes 74..81, group g in byte g)
// come with grp_idx = 0. A block is 11 load cycles and 8 dot cycles: 82 bytes in 19 cycles, 4.32
// bytes a cycle, a group every cycle in the dot phase. The partial is integer two's complement.
//
// The grid is a 1024 x 64 ROM, four asynchronous reads a cycle (one per sub-block). The four reads
// are why an FPGA implementation may replicate it four times (32 KiB) or bank it in four; none of
// that, the multiplier count (32 int8 x 6-bit products a cycle, 4:2 reduce trees) or the timing is
// measured here: there is no synthesis or timing run in this work (docs/iq_dot.md says what is not
// done). The ROM contents are a simulation-visible initial block, the portable way to write a
// table; a technology wrapper may replace it later.
//
// The ROM table is transcribed from ggml's iq2s_grid (ggml-common.h:758 at the revision pinned
// above): Copyright (c) 2023 Georgi Gerganov and the ggml contributors, MIT licence. The complete
// notice is copied at tools/packed_oracle/licenses/GGML-MIT.txt and recorded in THIRD_PARTY.md;
// openTPU's Apache-2.0 licence does not replace it.
//
// Reset is synchronous, active high; the block store (88 bytes) has no reset: a beat writes it
// before any of it is read, and the first block after reset writes all of it.
module otpu_iq2s (
  input  logic         clk,
  input  logic         rst,          // synchronous, active high
  // weight block: 11 beats of 64 bit (88 bytes, 82 used)
  input  logic         w_valid,
  output logic         w_ready,
  input  logic [63:0]  w_data,
  // activations: 8 beats of 256 bit, one 32-weight group per beat
  input  logic         a_valid,
  output logic         a_ready,
  input  logic [255:0] a_data,
  // per-group partial (valid one cycle after its activation beat)
  output logic         grp_valid,
  output logic [2:0]   grp_idx,      // 0..7
  output logic [31:0]  grp_bsum,     // complete scaled group partial (unchanged)
  output logic [31:0]  grp_aux,      // separate second scaled half: ls2 * sumi2
  output logic [4:0]   grp_ls1,      // 1 + 2 * (scales[g] & 15)
  output logic [4:0]   grp_ls2,      // 1 + 2 * (scales[g] >> 4)
  output logic [31:0]  grp_signs,    // raw sign bytes, l = 0 in bits 7:0
  output logic [39:0]  grp_index,    // 4 x 10 bit grid indices, l = 0 in bits 9:0
  // block metadata (valid with grp_idx = 0)
  output logic         blk_valid,
  output logic [15:0]  blk_d,        // raw fp16 bits, bytes 0..1
  output logic [63:0]  blk_scales    // raw scales bytes 74..81, group g in byte g
);
  // ---- the block being decoded, byte 8b + j at blk[64b + 8j +: 8]
  logic [703:0] blk;

  // ---- block fields: constant extract of the loaded block, one mux stage for the current group
  logic [31:0] grp_qs    [8];        // qs[4g + l] in byte l
  logic [31:0] grp_sign  [8];        // qs[32 + 4g + l] in byte l
  logic [7:0]  grp_qh    [8];
  logic [7:0]  grp_scale [8];
  for (genvar g = 0; g < 8; g++) begin : g_field
    assign grp_qs[g]    = blk[8 * (2 + 4 * g)  +: 32];
    assign grp_sign[g]  = blk[8 * (34 + 4 * g) +: 32];
    assign grp_qh[g]    = blk[8 * (66 + g)     +: 8];
    assign grp_scale[g] = blk[8 * (74 + g)     +: 8];
  end

  // ---- the grid: 1024 x 64 bit, the magnitudes in {8, 25, 43}; see the header for its provenance
  logic [63:0] iq2s_rom [0:1023];
  initial begin
    iq2s_rom[   0] = 64'h0808080808080808;
    iq2s_rom[   1] = 64'h080808080808082b;
    iq2s_rom[   2] = 64'h0808080808081919;
    iq2s_rom[   3] = 64'h0808080808082b08;
    iq2s_rom[   4] = 64'h0808080808082b2b;
    iq2s_rom[   5] = 64'h0808080808190819;
    iq2s_rom[   6] = 64'h0808080808191908;
    iq2s_rom[   7] = 64'h080808080819192b;
    iq2s_rom[   8] = 64'h0808080808192b19;
    iq2s_rom[   9] = 64'h08080808082b0808;
    iq2s_rom[  10] = 64'h08080808082b082b;
    iq2s_rom[  11] = 64'h08080808082b1919;
    iq2s_rom[  12] = 64'h08080808082b2b08;
    iq2s_rom[  13] = 64'h0808080819080819;
    iq2s_rom[  14] = 64'h0808080819081908;
    iq2s_rom[  15] = 64'h080808081908192b;
    iq2s_rom[  16] = 64'h0808080819082b19;
    iq2s_rom[  17] = 64'h0808080819190808;
    iq2s_rom[  18] = 64'h080808081919082b;
    iq2s_rom[  19] = 64'h0808080819191919;
    iq2s_rom[  20] = 64'h0808080819192b08;
    iq2s_rom[  21] = 64'h08080808192b0819;
    iq2s_rom[  22] = 64'h08080808192b1908;
    iq2s_rom[  23] = 64'h08080808192b192b;
    iq2s_rom[  24] = 64'h08080808192b2b19;
    iq2s_rom[  25] = 64'h080808082b080808;
    iq2s_rom[  26] = 64'h080808082b08082b;
    iq2s_rom[  27] = 64'h080808082b081919;
    iq2s_rom[  28] = 64'h080808082b082b08;
    iq2s_rom[  29] = 64'h080808082b190819;
    iq2s_rom[  30] = 64'h080808082b191908;
    iq2s_rom[  31] = 64'h080808082b2b0808;
    iq2s_rom[  32] = 64'h080808082b2b1919;
    iq2s_rom[  33] = 64'h080808082b2b2b2b;
    iq2s_rom[  34] = 64'h0808081908080819;
    iq2s_rom[  35] = 64'h0808081908081908;
    iq2s_rom[  36] = 64'h080808190808192b;
    iq2s_rom[  37] = 64'h0808081908082b19;
    iq2s_rom[  38] = 64'h0808081908190808;
    iq2s_rom[  39] = 64'h080808190819082b;
    iq2s_rom[  40] = 64'h0808081908191919;
    iq2s_rom[  41] = 64'h0808081908192b08;
    iq2s_rom[  42] = 64'h08080819082b0819;
    iq2s_rom[  43] = 64'h08080819082b1908;
    iq2s_rom[  44] = 64'h0808081919080808;
    iq2s_rom[  45] = 64'h080808191908082b;
    iq2s_rom[  46] = 64'h0808081919081919;
    iq2s_rom[  47] = 64'h0808081919082b08;
    iq2s_rom[  48] = 64'h0808081919190819;
    iq2s_rom[  49] = 64'h0808081919191908;
    iq2s_rom[  50] = 64'h080808191919192b;
    iq2s_rom[  51] = 64'h0808081919192b19;
    iq2s_rom[  52] = 64'h08080819192b0808;
    iq2s_rom[  53] = 64'h08080819192b1919;
    iq2s_rom[  54] = 64'h08080819192b2b08;
    iq2s_rom[  55] = 64'h080808192b080819;
    iq2s_rom[  56] = 64'h080808192b081908;
    iq2s_rom[  57] = 64'h080808192b190808;
    iq2s_rom[  58] = 64'h080808192b19082b;
    iq2s_rom[  59] = 64'h080808192b191919;
    iq2s_rom[  60] = 64'h080808192b2b0819;
    iq2s_rom[  61] = 64'h080808192b2b1908;
    iq2s_rom[  62] = 64'h0808082b08080808;
    iq2s_rom[  63] = 64'h0808082b0808082b;
    iq2s_rom[  64] = 64'h0808082b08081919;
    iq2s_rom[  65] = 64'h0808082b08082b08;
    iq2s_rom[  66] = 64'h0808082b08190819;
    iq2s_rom[  67] = 64'h0808082b08191908;
    iq2s_rom[  68] = 64'h0808082b082b0808;
    iq2s_rom[  69] = 64'h0808082b082b2b2b;
    iq2s_rom[  70] = 64'h0808082b19080819;
    iq2s_rom[  71] = 64'h0808082b19081908;
    iq2s_rom[  72] = 64'h0808082b1908192b;
    iq2s_rom[  73] = 64'h0808082b19082b19;
    iq2s_rom[  74] = 64'h0808082b19190808;
    iq2s_rom[  75] = 64'h0808082b19191919;
    iq2s_rom[  76] = 64'h0808082b2b080808;
    iq2s_rom[  77] = 64'h0808082b2b081919;
    iq2s_rom[  78] = 64'h0808082b2b082b2b;
    iq2s_rom[  79] = 64'h0808082b2b191908;
    iq2s_rom[  80] = 64'h0808082b2b2b082b;
    iq2s_rom[  81] = 64'h0808190808080819;
    iq2s_rom[  82] = 64'h0808190808081908;
    iq2s_rom[  83] = 64'h080819080808192b;
    iq2s_rom[  84] = 64'h0808190808082b19;
    iq2s_rom[  85] = 64'h0808190808190808;
    iq2s_rom[  86] = 64'h080819080819082b;
    iq2s_rom[  87] = 64'h0808190808191919;
    iq2s_rom[  88] = 64'h0808190808192b08;
    iq2s_rom[  89] = 64'h08081908082b0819;
    iq2s_rom[  90] = 64'h08081908082b1908;
    iq2s_rom[  91] = 64'h08081908082b192b;
    iq2s_rom[  92] = 64'h08081908082b2b19;
    iq2s_rom[  93] = 64'h0808190819080808;
    iq2s_rom[  94] = 64'h080819081908082b;
    iq2s_rom[  95] = 64'h0808190819081919;
    iq2s_rom[  96] = 64'h0808190819082b08;
    iq2s_rom[  97] = 64'h0808190819082b2b;
    iq2s_rom[  98] = 64'h0808190819190819;
    iq2s_rom[  99] = 64'h0808190819191908;
    iq2s_rom[ 100] = 64'h080819081919192b;
    iq2s_rom[ 101] = 64'h0808190819192b19;
    iq2s_rom[ 102] = 64'h08081908192b0808;
    iq2s_rom[ 103] = 64'h08081908192b082b;
    iq2s_rom[ 104] = 64'h08081908192b1919;
    iq2s_rom[ 105] = 64'h080819082b080819;
    iq2s_rom[ 106] = 64'h080819082b081908;
    iq2s_rom[ 107] = 64'h080819082b08192b;
    iq2s_rom[ 108] = 64'h080819082b082b19;
    iq2s_rom[ 109] = 64'h080819082b190808;
    iq2s_rom[ 110] = 64'h080819082b191919;
    iq2s_rom[ 111] = 64'h080819082b192b08;
    iq2s_rom[ 112] = 64'h080819082b2b0819;
    iq2s_rom[ 113] = 64'h080819082b2b1908;
    iq2s_rom[ 114] = 64'h0808191908080808;
    iq2s_rom[ 115] = 64'h080819190808082b;
    iq2s_rom[ 116] = 64'h0808191908081919;
    iq2s_rom[ 117] = 64'h0808191908082b08;
    iq2s_rom[ 118] = 64'h0808191908082b2b;
    iq2s_rom[ 119] = 64'h0808191908190819;
    iq2s_rom[ 120] = 64'h0808191908191908;
    iq2s_rom[ 121] = 64'h080819190819192b;
    iq2s_rom[ 122] = 64'h0808191908192b19;
    iq2s_rom[ 123] = 64'h08081919082b0808;
    iq2s_rom[ 124] = 64'h08081919082b1919;
    iq2s_rom[ 125] = 64'h08081919082b2b08;
    iq2s_rom[ 126] = 64'h0808191919080819;
    iq2s_rom[ 127] = 64'h0808191919081908;
    iq2s_rom[ 128] = 64'h080819191908192b;
    iq2s_rom[ 129] = 64'h0808191919082b19;
    iq2s_rom[ 130] = 64'h0808191919190808;
    iq2s_rom[ 131] = 64'h080819191919082b;
    iq2s_rom[ 132] = 64'h0808191919191919;
    iq2s_rom[ 133] = 64'h0808191919192b08;
    iq2s_rom[ 134] = 64'h08081919192b0819;
    iq2s_rom[ 135] = 64'h08081919192b1908;
    iq2s_rom[ 136] = 64'h080819192b080808;
    iq2s_rom[ 137] = 64'h080819192b08082b;
    iq2s_rom[ 138] = 64'h080819192b081919;
    iq2s_rom[ 139] = 64'h080819192b082b08;
    iq2s_rom[ 140] = 64'h080819192b190819;
    iq2s_rom[ 141] = 64'h080819192b191908;
    iq2s_rom[ 142] = 64'h080819192b2b0808;
    iq2s_rom[ 143] = 64'h0808192b08080819;
    iq2s_rom[ 144] = 64'h0808192b08081908;
    iq2s_rom[ 145] = 64'h0808192b0808192b;
    iq2s_rom[ 146] = 64'h0808192b08082b19;
    iq2s_rom[ 147] = 64'h0808192b08190808;
    iq2s_rom[ 148] = 64'h0808192b08191919;
    iq2s_rom[ 149] = 64'h0808192b19080808;
    iq2s_rom[ 150] = 64'h0808192b19081919;
    iq2s_rom[ 151] = 64'h0808192b19082b08;
    iq2s_rom[ 152] = 64'h0808192b19190819;
    iq2s_rom[ 153] = 64'h0808192b19191908;
    iq2s_rom[ 154] = 64'h0808192b192b0808;
    iq2s_rom[ 155] = 64'h0808192b2b080819;
    iq2s_rom[ 156] = 64'h0808192b2b081908;
    iq2s_rom[ 157] = 64'h0808192b2b190808;
    iq2s_rom[ 158] = 64'h08082b0808080808;
    iq2s_rom[ 159] = 64'h08082b080808082b;
    iq2s_rom[ 160] = 64'h08082b0808081919;
    iq2s_rom[ 161] = 64'h08082b0808082b08;
    iq2s_rom[ 162] = 64'h08082b0808190819;
    iq2s_rom[ 163] = 64'h08082b0808191908;
    iq2s_rom[ 164] = 64'h08082b080819192b;
    iq2s_rom[ 165] = 64'h08082b0808192b19;
    iq2s_rom[ 166] = 64'h08082b08082b0808;
    iq2s_rom[ 167] = 64'h08082b08082b1919;
    iq2s_rom[ 168] = 64'h08082b08082b2b2b;
    iq2s_rom[ 169] = 64'h08082b0819080819;
    iq2s_rom[ 170] = 64'h08082b0819081908;
    iq2s_rom[ 171] = 64'h08082b081908192b;
    iq2s_rom[ 172] = 64'h08082b0819082b19;
    iq2s_rom[ 173] = 64'h08082b0819190808;
    iq2s_rom[ 174] = 64'h08082b081919082b;
    iq2s_rom[ 175] = 64'h08082b0819191919;
    iq2s_rom[ 176] = 64'h08082b0819192b08;
    iq2s_rom[ 177] = 64'h08082b08192b0819;
    iq2s_rom[ 178] = 64'h08082b08192b1908;
    iq2s_rom[ 179] = 64'h08082b082b080808;
    iq2s_rom[ 180] = 64'h08082b082b081919;
    iq2s_rom[ 181] = 64'h08082b082b191908;
    iq2s_rom[ 182] = 64'h08082b082b2b2b2b;
    iq2s_rom[ 183] = 64'h08082b1908080819;
    iq2s_rom[ 184] = 64'h08082b1908081908;
    iq2s_rom[ 185] = 64'h08082b1908190808;
    iq2s_rom[ 186] = 64'h08082b190819082b;
    iq2s_rom[ 187] = 64'h08082b1908191919;
    iq2s_rom[ 188] = 64'h08082b1908192b08;
    iq2s_rom[ 189] = 64'h08082b19082b0819;
    iq2s_rom[ 190] = 64'h08082b1919080808;
    iq2s_rom[ 191] = 64'h08082b1919081919;
    iq2s_rom[ 192] = 64'h08082b1919082b08;
    iq2s_rom[ 193] = 64'h08082b1919190819;
    iq2s_rom[ 194] = 64'h08082b1919191908;
    iq2s_rom[ 195] = 64'h08082b19192b0808;
    iq2s_rom[ 196] = 64'h08082b192b080819;
    iq2s_rom[ 197] = 64'h08082b192b190808;
    iq2s_rom[ 198] = 64'h08082b2b08080808;
    iq2s_rom[ 199] = 64'h08082b2b08190819;
    iq2s_rom[ 200] = 64'h08082b2b08191908;
    iq2s_rom[ 201] = 64'h08082b2b082b082b;
    iq2s_rom[ 202] = 64'h08082b2b082b2b08;
    iq2s_rom[ 203] = 64'h08082b2b082b2b2b;
    iq2s_rom[ 204] = 64'h08082b2b19190808;
    iq2s_rom[ 205] = 64'h08082b2b2b192b19;
    iq2s_rom[ 206] = 64'h0819080808080819;
    iq2s_rom[ 207] = 64'h0819080808081908;
    iq2s_rom[ 208] = 64'h081908080808192b;
    iq2s_rom[ 209] = 64'h0819080808082b19;
    iq2s_rom[ 210] = 64'h0819080808190808;
    iq2s_rom[ 211] = 64'h081908080819082b;
    iq2s_rom[ 212] = 64'h0819080808191919;
    iq2s_rom[ 213] = 64'h0819080808192b08;
    iq2s_rom[ 214] = 64'h08190808082b0819;
    iq2s_rom[ 215] = 64'h08190808082b1908;
    iq2s_rom[ 216] = 64'h08190808082b192b;
    iq2s_rom[ 217] = 64'h0819080819080808;
    iq2s_rom[ 218] = 64'h081908081908082b;
    iq2s_rom[ 219] = 64'h0819080819081919;
    iq2s_rom[ 220] = 64'h0819080819082b08;
    iq2s_rom[ 221] = 64'h0819080819190819;
    iq2s_rom[ 222] = 64'h0819080819191908;
    iq2s_rom[ 223] = 64'h081908081919192b;
    iq2s_rom[ 224] = 64'h0819080819192b19;
    iq2s_rom[ 225] = 64'h08190808192b0808;
    iq2s_rom[ 226] = 64'h08190808192b082b;
    iq2s_rom[ 227] = 64'h08190808192b1919;
    iq2s_rom[ 228] = 64'h08190808192b2b08;
    iq2s_rom[ 229] = 64'h081908082b080819;
    iq2s_rom[ 230] = 64'h081908082b081908;
    iq2s_rom[ 231] = 64'h081908082b08192b;
    iq2s_rom[ 232] = 64'h081908082b190808;
    iq2s_rom[ 233] = 64'h081908082b191919;
    iq2s_rom[ 234] = 64'h081908082b192b08;
    iq2s_rom[ 235] = 64'h081908082b2b0819;
    iq2s_rom[ 236] = 64'h081908082b2b1908;
    iq2s_rom[ 237] = 64'h0819081908080808;
    iq2s_rom[ 238] = 64'h081908190808082b;
    iq2s_rom[ 239] = 64'h0819081908081919;
    iq2s_rom[ 240] = 64'h0819081908082b08;
    iq2s_rom[ 241] = 64'h0819081908082b2b;
    iq2s_rom[ 242] = 64'h0819081908190819;
    iq2s_rom[ 243] = 64'h0819081908191908;
    iq2s_rom[ 244] = 64'h081908190819192b;
    iq2s_rom[ 245] = 64'h0819081908192b19;
    iq2s_rom[ 246] = 64'h08190819082b0808;
    iq2s_rom[ 247] = 64'h08190819082b082b;
    iq2s_rom[ 248] = 64'h08190819082b1919;
    iq2s_rom[ 249] = 64'h08190819082b2b08;
    iq2s_rom[ 250] = 64'h0819081919080819;
    iq2s_rom[ 251] = 64'h0819081919081908;
    iq2s_rom[ 252] = 64'h081908191908192b;
    iq2s_rom[ 253] = 64'h0819081919082b19;
    iq2s_rom[ 254] = 64'h0819081919190808;
    iq2s_rom[ 255] = 64'h081908191919082b;
    iq2s_rom[ 256] = 64'h0819081919191919;
    iq2s_rom[ 257] = 64'h0819081919192b08;
    iq2s_rom[ 258] = 64'h08190819192b0819;
    iq2s_rom[ 259] = 64'h08190819192b1908;
    iq2s_rom[ 260] = 64'h081908192b080808;
    iq2s_rom[ 261] = 64'h081908192b08082b;
    iq2s_rom[ 262] = 64'h081908192b081919;
    iq2s_rom[ 263] = 64'h081908192b082b08;
    iq2s_rom[ 264] = 64'h081908192b190819;
    iq2s_rom[ 265] = 64'h081908192b191908;
    iq2s_rom[ 266] = 64'h0819082b08080819;
    iq2s_rom[ 267] = 64'h0819082b08081908;
    iq2s_rom[ 268] = 64'h0819082b08082b19;
    iq2s_rom[ 269] = 64'h0819082b08190808;
    iq2s_rom[ 270] = 64'h0819082b08191919;
    iq2s_rom[ 271] = 64'h0819082b082b0819;
    iq2s_rom[ 272] = 64'h0819082b082b1908;
    iq2s_rom[ 273] = 64'h0819082b19080808;
    iq2s_rom[ 274] = 64'h0819082b19081919;
    iq2s_rom[ 275] = 64'h0819082b19190819;
    iq2s_rom[ 276] = 64'h0819082b19191908;
    iq2s_rom[ 277] = 64'h0819082b2b080819;
    iq2s_rom[ 278] = 64'h0819082b2b081908;
    iq2s_rom[ 279] = 64'h0819082b2b190808;
    iq2s_rom[ 280] = 64'h0819190808080808;
    iq2s_rom[ 281] = 64'h081919080808082b;
    iq2s_rom[ 282] = 64'h0819190808081919;
    iq2s_rom[ 283] = 64'h0819190808082b08;
    iq2s_rom[ 284] = 64'h0819190808190819;
    iq2s_rom[ 285] = 64'h0819190808191908;
    iq2s_rom[ 286] = 64'h081919080819192b;
    iq2s_rom[ 287] = 64'h0819190808192b19;
    iq2s_rom[ 288] = 64'h08191908082b0808;
    iq2s_rom[ 289] = 64'h08191908082b1919;
    iq2s_rom[ 290] = 64'h08191908082b2b08;
    iq2s_rom[ 291] = 64'h0819190819080819;
    iq2s_rom[ 292] = 64'h0819190819081908;
    iq2s_rom[ 293] = 64'h081919081908192b;
    iq2s_rom[ 294] = 64'h0819190819082b19;
    iq2s_rom[ 295] = 64'h0819190819190808;
    iq2s_rom[ 296] = 64'h081919081919082b;
    iq2s_rom[ 297] = 64'h0819190819191919;
    iq2s_rom[ 298] = 64'h0819190819192b08;
    iq2s_rom[ 299] = 64'h08191908192b0819;
    iq2s_rom[ 300] = 64'h08191908192b1908;
    iq2s_rom[ 301] = 64'h081919082b080808;
    iq2s_rom[ 302] = 64'h081919082b08082b;
    iq2s_rom[ 303] = 64'h081919082b081919;
    iq2s_rom[ 304] = 64'h081919082b082b08;
    iq2s_rom[ 305] = 64'h081919082b190819;
    iq2s_rom[ 306] = 64'h081919082b191908;
    iq2s_rom[ 307] = 64'h081919082b2b0808;
    iq2s_rom[ 308] = 64'h0819191908080819;
    iq2s_rom[ 309] = 64'h0819191908081908;
    iq2s_rom[ 310] = 64'h081919190808192b;
    iq2s_rom[ 311] = 64'h0819191908082b19;
    iq2s_rom[ 312] = 64'h0819191908190808;
    iq2s_rom[ 313] = 64'h081919190819082b;
    iq2s_rom[ 314] = 64'h0819191908191919;
    iq2s_rom[ 315] = 64'h0819191908192b08;
    iq2s_rom[ 316] = 64'h08191919082b0819;
    iq2s_rom[ 317] = 64'h08191919082b1908;
    iq2s_rom[ 318] = 64'h0819191919080808;
    iq2s_rom[ 319] = 64'h081919191908082b;
    iq2s_rom[ 320] = 64'h0819191919081919;
    iq2s_rom[ 321] = 64'h0819191919082b08;
    iq2s_rom[ 322] = 64'h0819191919190819;
    iq2s_rom[ 323] = 64'h0819191919191908;
    iq2s_rom[ 324] = 64'h08191919192b0808;
    iq2s_rom[ 325] = 64'h081919192b080819;
    iq2s_rom[ 326] = 64'h081919192b081908;
    iq2s_rom[ 327] = 64'h081919192b190808;
    iq2s_rom[ 328] = 64'h0819192b08080808;
    iq2s_rom[ 329] = 64'h0819192b08081919;
    iq2s_rom[ 330] = 64'h0819192b08082b08;
    iq2s_rom[ 331] = 64'h0819192b08190819;
    iq2s_rom[ 332] = 64'h0819192b08191908;
    iq2s_rom[ 333] = 64'h0819192b082b0808;
    iq2s_rom[ 334] = 64'h0819192b19080819;
    iq2s_rom[ 335] = 64'h0819192b19081908;
    iq2s_rom[ 336] = 64'h0819192b19190808;
    iq2s_rom[ 337] = 64'h0819192b2b080808;
    iq2s_rom[ 338] = 64'h0819192b2b2b2b2b;
    iq2s_rom[ 339] = 64'h08192b0808080819;
    iq2s_rom[ 340] = 64'h08192b0808081908;
    iq2s_rom[ 341] = 64'h08192b080808192b;
    iq2s_rom[ 342] = 64'h08192b0808082b19;
    iq2s_rom[ 343] = 64'h08192b0808190808;
    iq2s_rom[ 344] = 64'h08192b0808191919;
    iq2s_rom[ 345] = 64'h08192b0808192b08;
    iq2s_rom[ 346] = 64'h08192b08082b0819;
    iq2s_rom[ 347] = 64'h08192b0819080808;
    iq2s_rom[ 348] = 64'h08192b081908082b;
    iq2s_rom[ 349] = 64'h08192b0819081919;
    iq2s_rom[ 350] = 64'h08192b0819082b08;
    iq2s_rom[ 351] = 64'h08192b0819190819;
    iq2s_rom[ 352] = 64'h08192b0819191908;
    iq2s_rom[ 353] = 64'h08192b08192b0808;
    iq2s_rom[ 354] = 64'h08192b082b080819;
    iq2s_rom[ 355] = 64'h08192b082b081908;
    iq2s_rom[ 356] = 64'h08192b1908080808;
    iq2s_rom[ 357] = 64'h08192b190808082b;
    iq2s_rom[ 358] = 64'h08192b1908081919;
    iq2s_rom[ 359] = 64'h08192b1908082b08;
    iq2s_rom[ 360] = 64'h08192b1908190819;
    iq2s_rom[ 361] = 64'h08192b1908191908;
    iq2s_rom[ 362] = 64'h08192b19082b0808;
    iq2s_rom[ 363] = 64'h08192b1919080819;
    iq2s_rom[ 364] = 64'h08192b1919081908;
    iq2s_rom[ 365] = 64'h08192b1919190808;
    iq2s_rom[ 366] = 64'h08192b19192b2b19;
    iq2s_rom[ 367] = 64'h08192b192b2b082b;
    iq2s_rom[ 368] = 64'h08192b2b08081908;
    iq2s_rom[ 369] = 64'h08192b2b08190808;
    iq2s_rom[ 370] = 64'h08192b2b19080808;
    iq2s_rom[ 371] = 64'h08192b2b1919192b;
    iq2s_rom[ 372] = 64'h082b080808080808;
    iq2s_rom[ 373] = 64'h082b08080808082b;
    iq2s_rom[ 374] = 64'h082b080808081919;
    iq2s_rom[ 375] = 64'h082b080808082b08;
    iq2s_rom[ 376] = 64'h082b080808190819;
    iq2s_rom[ 377] = 64'h082b080808191908;
    iq2s_rom[ 378] = 64'h082b08080819192b;
    iq2s_rom[ 379] = 64'h082b080808192b19;
    iq2s_rom[ 380] = 64'h082b0808082b0808;
    iq2s_rom[ 381] = 64'h082b0808082b1919;
    iq2s_rom[ 382] = 64'h082b0808082b2b2b;
    iq2s_rom[ 383] = 64'h082b080819080819;
    iq2s_rom[ 384] = 64'h082b080819081908;
    iq2s_rom[ 385] = 64'h082b080819190808;
    iq2s_rom[ 386] = 64'h082b08081919082b;
    iq2s_rom[ 387] = 64'h082b080819191919;
    iq2s_rom[ 388] = 64'h082b0808192b1908;
    iq2s_rom[ 389] = 64'h082b08082b080808;
    iq2s_rom[ 390] = 64'h082b08082b082b2b;
    iq2s_rom[ 391] = 64'h082b08082b191908;
    iq2s_rom[ 392] = 64'h082b08082b2b2b2b;
    iq2s_rom[ 393] = 64'h082b081908080819;
    iq2s_rom[ 394] = 64'h082b081908081908;
    iq2s_rom[ 395] = 64'h082b081908190808;
    iq2s_rom[ 396] = 64'h082b08190819082b;
    iq2s_rom[ 397] = 64'h082b081908191919;
    iq2s_rom[ 398] = 64'h082b0819082b0819;
    iq2s_rom[ 399] = 64'h082b081919080808;
    iq2s_rom[ 400] = 64'h082b08191908082b;
    iq2s_rom[ 401] = 64'h082b081919081919;
    iq2s_rom[ 402] = 64'h082b081919190819;
    iq2s_rom[ 403] = 64'h082b081919191908;
    iq2s_rom[ 404] = 64'h082b0819192b0808;
    iq2s_rom[ 405] = 64'h082b08192b080819;
    iq2s_rom[ 406] = 64'h082b08192b081908;
    iq2s_rom[ 407] = 64'h082b08192b190808;
    iq2s_rom[ 408] = 64'h082b082b08080808;
    iq2s_rom[ 409] = 64'h082b082b08082b2b;
    iq2s_rom[ 410] = 64'h082b082b082b082b;
    iq2s_rom[ 411] = 64'h082b082b082b2b08;
    iq2s_rom[ 412] = 64'h082b082b082b2b2b;
    iq2s_rom[ 413] = 64'h082b082b19081908;
    iq2s_rom[ 414] = 64'h082b082b19190808;
    iq2s_rom[ 415] = 64'h082b082b2b082b08;
    iq2s_rom[ 416] = 64'h082b082b2b082b2b;
    iq2s_rom[ 417] = 64'h082b082b2b2b2b08;
    iq2s_rom[ 418] = 64'h082b190808080819;
    iq2s_rom[ 419] = 64'h082b190808081908;
    iq2s_rom[ 420] = 64'h082b19080808192b;
    iq2s_rom[ 421] = 64'h082b190808082b19;
    iq2s_rom[ 422] = 64'h082b190808190808;
    iq2s_rom[ 423] = 64'h082b190808191919;
    iq2s_rom[ 424] = 64'h082b190808192b08;
    iq2s_rom[ 425] = 64'h082b1908082b0819;
    iq2s_rom[ 426] = 64'h082b1908082b1908;
    iq2s_rom[ 427] = 64'h082b190819080808;
    iq2s_rom[ 428] = 64'h082b19081908082b;
    iq2s_rom[ 429] = 64'h082b190819081919;
    iq2s_rom[ 430] = 64'h082b190819082b08;
    iq2s_rom[ 431] = 64'h082b190819190819;
    iq2s_rom[ 432] = 64'h082b190819191908;
    iq2s_rom[ 433] = 64'h082b1908192b0808;
    iq2s_rom[ 434] = 64'h082b19082b080819;
    iq2s_rom[ 435] = 64'h082b19082b081908;
    iq2s_rom[ 436] = 64'h082b19082b190808;
    iq2s_rom[ 437] = 64'h082b191908080808;
    iq2s_rom[ 438] = 64'h082b191908081919;
    iq2s_rom[ 439] = 64'h082b191908082b08;
    iq2s_rom[ 440] = 64'h082b191908190819;
    iq2s_rom[ 441] = 64'h082b191908191908;
    iq2s_rom[ 442] = 64'h082b1919082b0808;
    iq2s_rom[ 443] = 64'h082b191919080819;
    iq2s_rom[ 444] = 64'h082b191919081908;
    iq2s_rom[ 445] = 64'h082b191919190808;
    iq2s_rom[ 446] = 64'h082b1919192b192b;
    iq2s_rom[ 447] = 64'h082b19192b080808;
    iq2s_rom[ 448] = 64'h082b192b08080819;
    iq2s_rom[ 449] = 64'h082b192b08081908;
    iq2s_rom[ 450] = 64'h082b192b08190808;
    iq2s_rom[ 451] = 64'h082b192b19080808;
    iq2s_rom[ 452] = 64'h082b192b19192b19;
    iq2s_rom[ 453] = 64'h082b2b0808080808;
    iq2s_rom[ 454] = 64'h082b2b0808081919;
    iq2s_rom[ 455] = 64'h082b2b0808190819;
    iq2s_rom[ 456] = 64'h082b2b0808191908;
    iq2s_rom[ 457] = 64'h082b2b0819080819;
    iq2s_rom[ 458] = 64'h082b2b0819081908;
    iq2s_rom[ 459] = 64'h082b2b0819190808;
    iq2s_rom[ 460] = 64'h082b2b082b082b2b;
    iq2s_rom[ 461] = 64'h082b2b082b2b2b2b;
    iq2s_rom[ 462] = 64'h082b2b1908080819;
    iq2s_rom[ 463] = 64'h082b2b1908081908;
    iq2s_rom[ 464] = 64'h082b2b1908190808;
    iq2s_rom[ 465] = 64'h082b2b192b191919;
    iq2s_rom[ 466] = 64'h082b2b2b08082b2b;
    iq2s_rom[ 467] = 64'h082b2b2b082b082b;
    iq2s_rom[ 468] = 64'h082b2b2b192b1908;
    iq2s_rom[ 469] = 64'h082b2b2b2b082b08;
    iq2s_rom[ 470] = 64'h082b2b2b2b082b2b;
    iq2s_rom[ 471] = 64'h1908080808080819;
    iq2s_rom[ 472] = 64'h1908080808081908;
    iq2s_rom[ 473] = 64'h190808080808192b;
    iq2s_rom[ 474] = 64'h1908080808082b19;
    iq2s_rom[ 475] = 64'h1908080808190808;
    iq2s_rom[ 476] = 64'h190808080819082b;
    iq2s_rom[ 477] = 64'h1908080808191919;
    iq2s_rom[ 478] = 64'h1908080808192b08;
    iq2s_rom[ 479] = 64'h1908080808192b2b;
    iq2s_rom[ 480] = 64'h19080808082b0819;
    iq2s_rom[ 481] = 64'h19080808082b1908;
    iq2s_rom[ 482] = 64'h19080808082b192b;
    iq2s_rom[ 483] = 64'h1908080819080808;
    iq2s_rom[ 484] = 64'h190808081908082b;
    iq2s_rom[ 485] = 64'h1908080819081919;
    iq2s_rom[ 486] = 64'h1908080819082b08;
    iq2s_rom[ 487] = 64'h1908080819082b2b;
    iq2s_rom[ 488] = 64'h1908080819190819;
    iq2s_rom[ 489] = 64'h1908080819191908;
    iq2s_rom[ 490] = 64'h190808081919192b;
    iq2s_rom[ 491] = 64'h1908080819192b19;
    iq2s_rom[ 492] = 64'h19080808192b0808;
    iq2s_rom[ 493] = 64'h19080808192b082b;
    iq2s_rom[ 494] = 64'h19080808192b1919;
    iq2s_rom[ 495] = 64'h190808082b080819;
    iq2s_rom[ 496] = 64'h190808082b081908;
    iq2s_rom[ 497] = 64'h190808082b190808;
    iq2s_rom[ 498] = 64'h190808082b191919;
    iq2s_rom[ 499] = 64'h190808082b192b08;
    iq2s_rom[ 500] = 64'h190808082b2b0819;
    iq2s_rom[ 501] = 64'h190808082b2b1908;
    iq2s_rom[ 502] = 64'h1908081908080808;
    iq2s_rom[ 503] = 64'h190808190808082b;
    iq2s_rom[ 504] = 64'h1908081908081919;
    iq2s_rom[ 505] = 64'h1908081908082b08;
    iq2s_rom[ 506] = 64'h1908081908190819;
    iq2s_rom[ 507] = 64'h1908081908191908;
    iq2s_rom[ 508] = 64'h190808190819192b;
    iq2s_rom[ 509] = 64'h1908081908192b19;
    iq2s_rom[ 510] = 64'h19080819082b0808;
    iq2s_rom[ 511] = 64'h19080819082b082b;
    iq2s_rom[ 512] = 64'h19080819082b1919;
    iq2s_rom[ 513] = 64'h1908081919080819;
    iq2s_rom[ 514] = 64'h1908081919081908;
    iq2s_rom[ 515] = 64'h190808191908192b;
    iq2s_rom[ 516] = 64'h1908081919082b19;
    iq2s_rom[ 517] = 64'h1908081919190808;
    iq2s_rom[ 518] = 64'h190808191919082b;
    iq2s_rom[ 519] = 64'h1908081919191919;
    iq2s_rom[ 520] = 64'h1908081919192b08;
    iq2s_rom[ 521] = 64'h19080819192b0819;
    iq2s_rom[ 522] = 64'h19080819192b1908;
    iq2s_rom[ 523] = 64'h190808192b080808;
    iq2s_rom[ 524] = 64'h190808192b08082b;
    iq2s_rom[ 525] = 64'h190808192b081919;
    iq2s_rom[ 526] = 64'h190808192b082b08;
    iq2s_rom[ 527] = 64'h190808192b190819;
    iq2s_rom[ 528] = 64'h190808192b191908;
    iq2s_rom[ 529] = 64'h190808192b2b0808;
    iq2s_rom[ 530] = 64'h1908082b08080819;
    iq2s_rom[ 531] = 64'h1908082b08081908;
    iq2s_rom[ 532] = 64'h1908082b08190808;
    iq2s_rom[ 533] = 64'h1908082b0819082b;
    iq2s_rom[ 534] = 64'h1908082b08191919;
    iq2s_rom[ 535] = 64'h1908082b08192b08;
    iq2s_rom[ 536] = 64'h1908082b082b1908;
    iq2s_rom[ 537] = 64'h1908082b19080808;
    iq2s_rom[ 538] = 64'h1908082b19081919;
    iq2s_rom[ 539] = 64'h1908082b19082b08;
    iq2s_rom[ 540] = 64'h1908082b19190819;
    iq2s_rom[ 541] = 64'h1908082b19191908;
    iq2s_rom[ 542] = 64'h1908082b192b0808;
    iq2s_rom[ 543] = 64'h1908082b2b080819;
    iq2s_rom[ 544] = 64'h1908082b2b081908;
    iq2s_rom[ 545] = 64'h1908190808080808;
    iq2s_rom[ 546] = 64'h190819080808082b;
    iq2s_rom[ 547] = 64'h1908190808081919;
    iq2s_rom[ 548] = 64'h1908190808082b08;
    iq2s_rom[ 549] = 64'h1908190808082b2b;
    iq2s_rom[ 550] = 64'h1908190808190819;
    iq2s_rom[ 551] = 64'h1908190808191908;
    iq2s_rom[ 552] = 64'h190819080819192b;
    iq2s_rom[ 553] = 64'h1908190808192b19;
    iq2s_rom[ 554] = 64'h19081908082b0808;
    iq2s_rom[ 555] = 64'h19081908082b082b;
    iq2s_rom[ 556] = 64'h19081908082b1919;
    iq2s_rom[ 557] = 64'h19081908082b2b08;
    iq2s_rom[ 558] = 64'h1908190819080819;
    iq2s_rom[ 559] = 64'h1908190819081908;
    iq2s_rom[ 560] = 64'h190819081908192b;
    iq2s_rom[ 561] = 64'h1908190819082b19;
    iq2s_rom[ 562] = 64'h1908190819190808;
    iq2s_rom[ 563] = 64'h190819081919082b;
    iq2s_rom[ 564] = 64'h1908190819191919;
    iq2s_rom[ 565] = 64'h1908190819192b08;
    iq2s_rom[ 566] = 64'h19081908192b0819;
    iq2s_rom[ 567] = 64'h19081908192b1908;
    iq2s_rom[ 568] = 64'h190819082b080808;
    iq2s_rom[ 569] = 64'h190819082b08082b;
    iq2s_rom[ 570] = 64'h190819082b081919;
    iq2s_rom[ 571] = 64'h190819082b082b08;
    iq2s_rom[ 572] = 64'h190819082b190819;
    iq2s_rom[ 573] = 64'h190819082b191908;
    iq2s_rom[ 574] = 64'h190819082b2b0808;
    iq2s_rom[ 575] = 64'h1908191908080819;
    iq2s_rom[ 576] = 64'h1908191908081908;
    iq2s_rom[ 577] = 64'h190819190808192b;
    iq2s_rom[ 578] = 64'h1908191908082b19;
    iq2s_rom[ 579] = 64'h1908191908190808;
    iq2s_rom[ 580] = 64'h190819190819082b;
    iq2s_rom[ 581] = 64'h1908191908191919;
    iq2s_rom[ 582] = 64'h1908191908192b08;
    iq2s_rom[ 583] = 64'h19081919082b0819;
    iq2s_rom[ 584] = 64'h19081919082b1908;
    iq2s_rom[ 585] = 64'h1908191919080808;
    iq2s_rom[ 586] = 64'h190819191908082b;
    iq2s_rom[ 587] = 64'h1908191919081919;
    iq2s_rom[ 588] = 64'h1908191919082b08;
    iq2s_rom[ 589] = 64'h1908191919190819;
    iq2s_rom[ 590] = 64'h1908191919191908;
    iq2s_rom[ 591] = 64'h19081919192b0808;
    iq2s_rom[ 592] = 64'h19081919192b2b2b;
    iq2s_rom[ 593] = 64'h190819192b080819;
    iq2s_rom[ 594] = 64'h190819192b081908;
    iq2s_rom[ 595] = 64'h190819192b190808;
    iq2s_rom[ 596] = 64'h1908192b08080808;
    iq2s_rom[ 597] = 64'h1908192b0808082b;
    iq2s_rom[ 598] = 64'h1908192b08081919;
    iq2s_rom[ 599] = 64'h1908192b08082b08;
    iq2s_rom[ 600] = 64'h1908192b08190819;
    iq2s_rom[ 601] = 64'h1908192b08191908;
    iq2s_rom[ 602] = 64'h1908192b082b0808;
    iq2s_rom[ 603] = 64'h1908192b19080819;
    iq2s_rom[ 604] = 64'h1908192b19081908;
    iq2s_rom[ 605] = 64'h1908192b19190808;
    iq2s_rom[ 606] = 64'h1908192b2b080808;
    iq2s_rom[ 607] = 64'h1908192b2b2b1919;
    iq2s_rom[ 608] = 64'h19082b0808080819;
    iq2s_rom[ 609] = 64'h19082b0808081908;
    iq2s_rom[ 610] = 64'h19082b0808082b19;
    iq2s_rom[ 611] = 64'h19082b0808190808;
    iq2s_rom[ 612] = 64'h19082b080819082b;
    iq2s_rom[ 613] = 64'h19082b0808191919;
    iq2s_rom[ 614] = 64'h19082b0808192b08;
    iq2s_rom[ 615] = 64'h19082b08082b0819;
    iq2s_rom[ 616] = 64'h19082b08082b1908;
    iq2s_rom[ 617] = 64'h19082b0819080808;
    iq2s_rom[ 618] = 64'h19082b081908082b;
    iq2s_rom[ 619] = 64'h19082b0819081919;
    iq2s_rom[ 620] = 64'h19082b0819082b08;
    iq2s_rom[ 621] = 64'h19082b0819190819;
    iq2s_rom[ 622] = 64'h19082b0819191908;
    iq2s_rom[ 623] = 64'h19082b08192b0808;
    iq2s_rom[ 624] = 64'h19082b082b081908;
    iq2s_rom[ 625] = 64'h19082b082b190808;
    iq2s_rom[ 626] = 64'h19082b1908080808;
    iq2s_rom[ 627] = 64'h19082b190808082b;
    iq2s_rom[ 628] = 64'h19082b1908081919;
    iq2s_rom[ 629] = 64'h19082b1908082b08;
    iq2s_rom[ 630] = 64'h19082b1908190819;
    iq2s_rom[ 631] = 64'h19082b1908191908;
    iq2s_rom[ 632] = 64'h19082b19082b0808;
    iq2s_rom[ 633] = 64'h19082b1919080819;
    iq2s_rom[ 634] = 64'h19082b1919081908;
    iq2s_rom[ 635] = 64'h19082b1919190808;
    iq2s_rom[ 636] = 64'h19082b192b080808;
    iq2s_rom[ 637] = 64'h19082b192b19192b;
    iq2s_rom[ 638] = 64'h19082b2b08080819;
    iq2s_rom[ 639] = 64'h19082b2b08081908;
    iq2s_rom[ 640] = 64'h19082b2b08190808;
    iq2s_rom[ 641] = 64'h19082b2b19080808;
    iq2s_rom[ 642] = 64'h1919080808080808;
    iq2s_rom[ 643] = 64'h191908080808082b;
    iq2s_rom[ 644] = 64'h1919080808081919;
    iq2s_rom[ 645] = 64'h1919080808082b08;
    iq2s_rom[ 646] = 64'h1919080808190819;
    iq2s_rom[ 647] = 64'h1919080808191908;
    iq2s_rom[ 648] = 64'h191908080819192b;
    iq2s_rom[ 649] = 64'h1919080808192b19;
    iq2s_rom[ 650] = 64'h19190808082b0808;
    iq2s_rom[ 651] = 64'h19190808082b082b;
    iq2s_rom[ 652] = 64'h19190808082b1919;
    iq2s_rom[ 653] = 64'h19190808082b2b08;
    iq2s_rom[ 654] = 64'h1919080819080819;
    iq2s_rom[ 655] = 64'h1919080819081908;
    iq2s_rom[ 656] = 64'h191908081908192b;
    iq2s_rom[ 657] = 64'h1919080819082b19;
    iq2s_rom[ 658] = 64'h1919080819190808;
    iq2s_rom[ 659] = 64'h191908081919082b;
    iq2s_rom[ 660] = 64'h1919080819191919;
    iq2s_rom[ 661] = 64'h1919080819192b08;
    iq2s_rom[ 662] = 64'h19190808192b0819;
    iq2s_rom[ 663] = 64'h19190808192b1908;
    iq2s_rom[ 664] = 64'h191908082b080808;
    iq2s_rom[ 665] = 64'h191908082b08082b;
    iq2s_rom[ 666] = 64'h191908082b081919;
    iq2s_rom[ 667] = 64'h191908082b082b08;
    iq2s_rom[ 668] = 64'h191908082b190819;
    iq2s_rom[ 669] = 64'h191908082b191908;
    iq2s_rom[ 670] = 64'h1919081908080819;
    iq2s_rom[ 671] = 64'h1919081908081908;
    iq2s_rom[ 672] = 64'h191908190808192b;
    iq2s_rom[ 673] = 64'h1919081908082b19;
    iq2s_rom[ 674] = 64'h1919081908190808;
    iq2s_rom[ 675] = 64'h191908190819082b;
    iq2s_rom[ 676] = 64'h1919081908191919;
    iq2s_rom[ 677] = 64'h1919081908192b08;
    iq2s_rom[ 678] = 64'h19190819082b0819;
    iq2s_rom[ 679] = 64'h19190819082b1908;
    iq2s_rom[ 680] = 64'h1919081919080808;
    iq2s_rom[ 681] = 64'h191908191908082b;
    iq2s_rom[ 682] = 64'h1919081919081919;
    iq2s_rom[ 683] = 64'h1919081919082b08;
    iq2s_rom[ 684] = 64'h1919081919190819;
    iq2s_rom[ 685] = 64'h1919081919191908;
    iq2s_rom[ 686] = 64'h19190819192b0808;
    iq2s_rom[ 687] = 64'h191908192b080819;
    iq2s_rom[ 688] = 64'h191908192b081908;
    iq2s_rom[ 689] = 64'h191908192b190808;
    iq2s_rom[ 690] = 64'h1919082b08080808;
    iq2s_rom[ 691] = 64'h1919082b08081919;
    iq2s_rom[ 692] = 64'h1919082b08082b08;
    iq2s_rom[ 693] = 64'h1919082b08190819;
    iq2s_rom[ 694] = 64'h1919082b08191908;
    iq2s_rom[ 695] = 64'h1919082b082b0808;
    iq2s_rom[ 696] = 64'h1919082b19080819;
    iq2s_rom[ 697] = 64'h1919082b19081908;
    iq2s_rom[ 698] = 64'h1919082b19190808;
    iq2s_rom[ 699] = 64'h1919082b192b2b19;
    iq2s_rom[ 700] = 64'h1919082b2b080808;
    iq2s_rom[ 701] = 64'h1919190808080819;
    iq2s_rom[ 702] = 64'h1919190808081908;
    iq2s_rom[ 703] = 64'h191919080808192b;
    iq2s_rom[ 704] = 64'h1919190808082b19;
    iq2s_rom[ 705] = 64'h1919190808190808;
    iq2s_rom[ 706] = 64'h191919080819082b;
    iq2s_rom[ 707] = 64'h1919190808191919;
    iq2s_rom[ 708] = 64'h1919190808192b08;
    iq2s_rom[ 709] = 64'h19191908082b0819;
    iq2s_rom[ 710] = 64'h19191908082b1908;
    iq2s_rom[ 711] = 64'h1919190819080808;
    iq2s_rom[ 712] = 64'h191919081908082b;
    iq2s_rom[ 713] = 64'h1919190819081919;
    iq2s_rom[ 714] = 64'h1919190819082b08;
    iq2s_rom[ 715] = 64'h1919190819190819;
    iq2s_rom[ 716] = 64'h1919190819191908;
    iq2s_rom[ 717] = 64'h19191908192b0808;
    iq2s_rom[ 718] = 64'h191919082b080819;
    iq2s_rom[ 719] = 64'h191919082b081908;
    iq2s_rom[ 720] = 64'h191919082b190808;
    iq2s_rom[ 721] = 64'h1919191908080808;
    iq2s_rom[ 722] = 64'h191919190808082b;
    iq2s_rom[ 723] = 64'h1919191908081919;
    iq2s_rom[ 724] = 64'h1919191908082b08;
    iq2s_rom[ 725] = 64'h1919191908190819;
    iq2s_rom[ 726] = 64'h1919191908191908;
    iq2s_rom[ 727] = 64'h19191919082b0808;
    iq2s_rom[ 728] = 64'h1919191919080819;
    iq2s_rom[ 729] = 64'h1919191919081908;
    iq2s_rom[ 730] = 64'h1919191919190808;
    iq2s_rom[ 731] = 64'h191919192b080808;
    iq2s_rom[ 732] = 64'h1919192b08080819;
    iq2s_rom[ 733] = 64'h1919192b08081908;
    iq2s_rom[ 734] = 64'h1919192b08190808;
    iq2s_rom[ 735] = 64'h1919192b082b192b;
    iq2s_rom[ 736] = 64'h1919192b19080808;
    iq2s_rom[ 737] = 64'h19192b0808080808;
    iq2s_rom[ 738] = 64'h19192b080808082b;
    iq2s_rom[ 739] = 64'h19192b0808081919;
    iq2s_rom[ 740] = 64'h19192b0808082b08;
    iq2s_rom[ 741] = 64'h19192b0808190819;
    iq2s_rom[ 742] = 64'h19192b0808191908;
    iq2s_rom[ 743] = 64'h19192b08082b0808;
    iq2s_rom[ 744] = 64'h19192b0819080819;
    iq2s_rom[ 745] = 64'h19192b0819081908;
    iq2s_rom[ 746] = 64'h19192b0819190808;
    iq2s_rom[ 747] = 64'h19192b0819192b2b;
    iq2s_rom[ 748] = 64'h19192b082b080808;
    iq2s_rom[ 749] = 64'h19192b1908080819;
    iq2s_rom[ 750] = 64'h19192b1908081908;
    iq2s_rom[ 751] = 64'h19192b1908190808;
    iq2s_rom[ 752] = 64'h19192b1919080808;
    iq2s_rom[ 753] = 64'h19192b2b08080808;
    iq2s_rom[ 754] = 64'h19192b2b08192b19;
    iq2s_rom[ 755] = 64'h19192b2b2b081919;
    iq2s_rom[ 756] = 64'h19192b2b2b2b2b08;
    iq2s_rom[ 757] = 64'h192b080808080819;
    iq2s_rom[ 758] = 64'h192b080808081908;
    iq2s_rom[ 759] = 64'h192b08080808192b;
    iq2s_rom[ 760] = 64'h192b080808190808;
    iq2s_rom[ 761] = 64'h192b08080819082b;
    iq2s_rom[ 762] = 64'h192b080808191919;
    iq2s_rom[ 763] = 64'h192b080808192b08;
    iq2s_rom[ 764] = 64'h192b0808082b0819;
    iq2s_rom[ 765] = 64'h192b0808082b1908;
    iq2s_rom[ 766] = 64'h192b080819080808;
    iq2s_rom[ 767] = 64'h192b080819081919;
    iq2s_rom[ 768] = 64'h192b080819082b08;
    iq2s_rom[ 769] = 64'h192b080819190819;
    iq2s_rom[ 770] = 64'h192b080819191908;
    iq2s_rom[ 771] = 64'h192b0808192b0808;
    iq2s_rom[ 772] = 64'h192b08082b081908;
    iq2s_rom[ 773] = 64'h192b08082b190808;
    iq2s_rom[ 774] = 64'h192b081908080808;
    iq2s_rom[ 775] = 64'h192b08190808082b;
    iq2s_rom[ 776] = 64'h192b081908081919;
    iq2s_rom[ 777] = 64'h192b081908082b08;
    iq2s_rom[ 778] = 64'h192b081908190819;
    iq2s_rom[ 779] = 64'h192b081908191908;
    iq2s_rom[ 780] = 64'h192b0819082b0808;
    iq2s_rom[ 781] = 64'h192b081919080819;
    iq2s_rom[ 782] = 64'h192b081919081908;
    iq2s_rom[ 783] = 64'h192b081919190808;
    iq2s_rom[ 784] = 64'h192b08192b080808;
    iq2s_rom[ 785] = 64'h192b08192b192b19;
    iq2s_rom[ 786] = 64'h192b082b08081908;
    iq2s_rom[ 787] = 64'h192b082b08190808;
    iq2s_rom[ 788] = 64'h192b082b19080808;
    iq2s_rom[ 789] = 64'h192b082b1919192b;
    iq2s_rom[ 790] = 64'h192b082b2b2b0819;
    iq2s_rom[ 791] = 64'h192b190808080808;
    iq2s_rom[ 792] = 64'h192b190808081919;
    iq2s_rom[ 793] = 64'h192b190808082b08;
    iq2s_rom[ 794] = 64'h192b190808190819;
    iq2s_rom[ 795] = 64'h192b190808191908;
    iq2s_rom[ 796] = 64'h192b1908082b0808;
    iq2s_rom[ 797] = 64'h192b190819080819;
    iq2s_rom[ 798] = 64'h192b190819081908;
    iq2s_rom[ 799] = 64'h192b190819190808;
    iq2s_rom[ 800] = 64'h192b19082b080808;
    iq2s_rom[ 801] = 64'h192b191908080819;
    iq2s_rom[ 802] = 64'h192b191908081908;
    iq2s_rom[ 803] = 64'h192b191908190808;
    iq2s_rom[ 804] = 64'h192b191919080808;
    iq2s_rom[ 805] = 64'h192b191919082b2b;
    iq2s_rom[ 806] = 64'h192b1919192b2b08;
    iq2s_rom[ 807] = 64'h192b19192b19082b;
    iq2s_rom[ 808] = 64'h192b192b08080808;
    iq2s_rom[ 809] = 64'h192b192b2b191908;
    iq2s_rom[ 810] = 64'h192b2b0808080819;
    iq2s_rom[ 811] = 64'h192b2b0808081908;
    iq2s_rom[ 812] = 64'h192b2b0808190808;
    iq2s_rom[ 813] = 64'h192b2b08192b1919;
    iq2s_rom[ 814] = 64'h192b2b082b192b08;
    iq2s_rom[ 815] = 64'h192b2b1908080808;
    iq2s_rom[ 816] = 64'h192b2b19082b2b2b;
    iq2s_rom[ 817] = 64'h192b2b2b1908082b;
    iq2s_rom[ 818] = 64'h192b2b2b2b2b0819;
    iq2s_rom[ 819] = 64'h2b08080808080808;
    iq2s_rom[ 820] = 64'h2b0808080808082b;
    iq2s_rom[ 821] = 64'h2b08080808081919;
    iq2s_rom[ 822] = 64'h2b08080808082b08;
    iq2s_rom[ 823] = 64'h2b08080808190819;
    iq2s_rom[ 824] = 64'h2b08080808191908;
    iq2s_rom[ 825] = 64'h2b08080808192b19;
    iq2s_rom[ 826] = 64'h2b080808082b0808;
    iq2s_rom[ 827] = 64'h2b080808082b1919;
    iq2s_rom[ 828] = 64'h2b08080819080819;
    iq2s_rom[ 829] = 64'h2b08080819081908;
    iq2s_rom[ 830] = 64'h2b08080819190808;
    iq2s_rom[ 831] = 64'h2b0808081919082b;
    iq2s_rom[ 832] = 64'h2b08080819191919;
    iq2s_rom[ 833] = 64'h2b08080819192b08;
    iq2s_rom[ 834] = 64'h2b080808192b0819;
    iq2s_rom[ 835] = 64'h2b0808082b080808;
    iq2s_rom[ 836] = 64'h2b0808082b081919;
    iq2s_rom[ 837] = 64'h2b0808082b190819;
    iq2s_rom[ 838] = 64'h2b0808082b191908;
    iq2s_rom[ 839] = 64'h2b08081908080819;
    iq2s_rom[ 840] = 64'h2b08081908081908;
    iq2s_rom[ 841] = 64'h2b08081908082b19;
    iq2s_rom[ 842] = 64'h2b08081908190808;
    iq2s_rom[ 843] = 64'h2b0808190819082b;
    iq2s_rom[ 844] = 64'h2b08081908191919;
    iq2s_rom[ 845] = 64'h2b08081908192b08;
    iq2s_rom[ 846] = 64'h2b080819082b0819;
    iq2s_rom[ 847] = 64'h2b080819082b1908;
    iq2s_rom[ 848] = 64'h2b08081919080808;
    iq2s_rom[ 849] = 64'h2b0808191908082b;
    iq2s_rom[ 850] = 64'h2b08081919081919;
    iq2s_rom[ 851] = 64'h2b08081919082b08;
    iq2s_rom[ 852] = 64'h2b08081919190819;
    iq2s_rom[ 853] = 64'h2b08081919191908;
    iq2s_rom[ 854] = 64'h2b0808192b080819;
    iq2s_rom[ 855] = 64'h2b0808192b081908;
    iq2s_rom[ 856] = 64'h2b0808192b190808;
    iq2s_rom[ 857] = 64'h2b0808192b2b2b19;
    iq2s_rom[ 858] = 64'h2b08082b08080808;
    iq2s_rom[ 859] = 64'h2b08082b08081919;
    iq2s_rom[ 860] = 64'h2b08082b08082b2b;
    iq2s_rom[ 861] = 64'h2b08082b08190819;
    iq2s_rom[ 862] = 64'h2b08082b08191908;
    iq2s_rom[ 863] = 64'h2b08082b19080819;
    iq2s_rom[ 864] = 64'h2b08082b19081908;
    iq2s_rom[ 865] = 64'h2b08082b19190808;
    iq2s_rom[ 866] = 64'h2b08190808080819;
    iq2s_rom[ 867] = 64'h2b08190808081908;
    iq2s_rom[ 868] = 64'h2b0819080808192b;
    iq2s_rom[ 869] = 64'h2b08190808082b19;
    iq2s_rom[ 870] = 64'h2b08190808190808;
    iq2s_rom[ 871] = 64'h2b0819080819082b;
    iq2s_rom[ 872] = 64'h2b08190808191919;
    iq2s_rom[ 873] = 64'h2b08190808192b08;
    iq2s_rom[ 874] = 64'h2b081908082b0819;
    iq2s_rom[ 875] = 64'h2b08190819080808;
    iq2s_rom[ 876] = 64'h2b0819081908082b;
    iq2s_rom[ 877] = 64'h2b08190819081919;
    iq2s_rom[ 878] = 64'h2b08190819082b08;
    iq2s_rom[ 879] = 64'h2b08190819190819;
    iq2s_rom[ 880] = 64'h2b08190819191908;
    iq2s_rom[ 881] = 64'h2b081908192b0808;
    iq2s_rom[ 882] = 64'h2b0819082b080819;
    iq2s_rom[ 883] = 64'h2b0819082b081908;
    iq2s_rom[ 884] = 64'h2b0819082b190808;
    iq2s_rom[ 885] = 64'h2b08191908080808;
    iq2s_rom[ 886] = 64'h2b0819190808082b;
    iq2s_rom[ 887] = 64'h2b08191908081919;
    iq2s_rom[ 888] = 64'h2b08191908082b08;
    iq2s_rom[ 889] = 64'h2b08191908190819;
    iq2s_rom[ 890] = 64'h2b08191908191908;
    iq2s_rom[ 891] = 64'h2b081919082b0808;
    iq2s_rom[ 892] = 64'h2b08191919080819;
    iq2s_rom[ 893] = 64'h2b08191919081908;
    iq2s_rom[ 894] = 64'h2b08191919190808;
    iq2s_rom[ 895] = 64'h2b0819192b080808;
    iq2s_rom[ 896] = 64'h2b0819192b082b2b;
    iq2s_rom[ 897] = 64'h2b08192b08080819;
    iq2s_rom[ 898] = 64'h2b08192b08081908;
    iq2s_rom[ 899] = 64'h2b08192b08190808;
    iq2s_rom[ 900] = 64'h2b08192b082b2b19;
    iq2s_rom[ 901] = 64'h2b08192b19080808;
    iq2s_rom[ 902] = 64'h2b082b0808080808;
    iq2s_rom[ 903] = 64'h2b082b0808081919;
    iq2s_rom[ 904] = 64'h2b082b0808190819;
    iq2s_rom[ 905] = 64'h2b082b0808191908;
    iq2s_rom[ 906] = 64'h2b082b0819080819;
    iq2s_rom[ 907] = 64'h2b082b0819081908;
    iq2s_rom[ 908] = 64'h2b082b0819190808;
    iq2s_rom[ 909] = 64'h2b082b082b2b082b;
    iq2s_rom[ 910] = 64'h2b082b1908080819;
    iq2s_rom[ 911] = 64'h2b082b1908081908;
    iq2s_rom[ 912] = 64'h2b082b1919080808;
    iq2s_rom[ 913] = 64'h2b082b19192b1919;
    iq2s_rom[ 914] = 64'h2b082b2b082b082b;
    iq2s_rom[ 915] = 64'h2b082b2b19192b08;
    iq2s_rom[ 916] = 64'h2b082b2b19192b2b;
    iq2s_rom[ 917] = 64'h2b082b2b2b08082b;
    iq2s_rom[ 918] = 64'h2b082b2b2b2b082b;
    iq2s_rom[ 919] = 64'h2b19080808080819;
    iq2s_rom[ 920] = 64'h2b19080808081908;
    iq2s_rom[ 921] = 64'h2b19080808082b19;
    iq2s_rom[ 922] = 64'h2b19080808190808;
    iq2s_rom[ 923] = 64'h2b1908080819082b;
    iq2s_rom[ 924] = 64'h2b19080808191919;
    iq2s_rom[ 925] = 64'h2b19080808192b08;
    iq2s_rom[ 926] = 64'h2b190808082b1908;
    iq2s_rom[ 927] = 64'h2b19080819080808;
    iq2s_rom[ 928] = 64'h2b1908081908082b;
    iq2s_rom[ 929] = 64'h2b19080819081919;
    iq2s_rom[ 930] = 64'h2b19080819082b08;
    iq2s_rom[ 931] = 64'h2b19080819190819;
    iq2s_rom[ 932] = 64'h2b19080819191908;
    iq2s_rom[ 933] = 64'h2b190808192b0808;
    iq2s_rom[ 934] = 64'h2b1908082b080819;
    iq2s_rom[ 935] = 64'h2b1908082b081908;
    iq2s_rom[ 936] = 64'h2b1908082b190808;
    iq2s_rom[ 937] = 64'h2b19081908080808;
    iq2s_rom[ 938] = 64'h2b19081908081919;
    iq2s_rom[ 939] = 64'h2b19081908190819;
    iq2s_rom[ 940] = 64'h2b19081908191908;
    iq2s_rom[ 941] = 64'h2b19081919080819;
    iq2s_rom[ 942] = 64'h2b19081919081908;
    iq2s_rom[ 943] = 64'h2b19081919190808;
    iq2s_rom[ 944] = 64'h2b19081919192b2b;
    iq2s_rom[ 945] = 64'h2b19082b08080819;
    iq2s_rom[ 946] = 64'h2b19082b08081908;
    iq2s_rom[ 947] = 64'h2b19082b08190808;
    iq2s_rom[ 948] = 64'h2b19082b19080808;
    iq2s_rom[ 949] = 64'h2b19082b2b2b192b;
    iq2s_rom[ 950] = 64'h2b19190808080808;
    iq2s_rom[ 951] = 64'h2b1919080808082b;
    iq2s_rom[ 952] = 64'h2b19190808081919;
    iq2s_rom[ 953] = 64'h2b19190808082b08;
    iq2s_rom[ 954] = 64'h2b19190808190819;
    iq2s_rom[ 955] = 64'h2b19190808191908;
    iq2s_rom[ 956] = 64'h2b191908082b0808;
    iq2s_rom[ 957] = 64'h2b19190819080819;
    iq2s_rom[ 958] = 64'h2b19190819081908;
    iq2s_rom[ 959] = 64'h2b19190819190808;
    iq2s_rom[ 960] = 64'h2b1919082b080808;
    iq2s_rom[ 961] = 64'h2b1919082b19192b;
    iq2s_rom[ 962] = 64'h2b19191908080819;
    iq2s_rom[ 963] = 64'h2b19191908081908;
    iq2s_rom[ 964] = 64'h2b19191908190808;
    iq2s_rom[ 965] = 64'h2b19191919080808;
    iq2s_rom[ 966] = 64'h2b1919192b192b08;
    iq2s_rom[ 967] = 64'h2b1919192b2b0819;
    iq2s_rom[ 968] = 64'h2b19192b08080808;
    iq2s_rom[ 969] = 64'h2b19192b1908192b;
    iq2s_rom[ 970] = 64'h2b19192b192b1908;
    iq2s_rom[ 971] = 64'h2b192b0808080819;
    iq2s_rom[ 972] = 64'h2b192b0808081908;
    iq2s_rom[ 973] = 64'h2b192b0808190808;
    iq2s_rom[ 974] = 64'h2b192b08082b192b;
    iq2s_rom[ 975] = 64'h2b192b0819080808;
    iq2s_rom[ 976] = 64'h2b192b082b2b2b19;
    iq2s_rom[ 977] = 64'h2b192b1908080808;
    iq2s_rom[ 978] = 64'h2b192b1919082b19;
    iq2s_rom[ 979] = 64'h2b192b191919082b;
    iq2s_rom[ 980] = 64'h2b192b2b2b190808;
    iq2s_rom[ 981] = 64'h2b2b080808080808;
    iq2s_rom[ 982] = 64'h2b2b080808081919;
    iq2s_rom[ 983] = 64'h2b2b080808082b2b;
    iq2s_rom[ 984] = 64'h2b2b080808191908;
    iq2s_rom[ 985] = 64'h2b2b0808082b082b;
    iq2s_rom[ 986] = 64'h2b2b0808082b2b2b;
    iq2s_rom[ 987] = 64'h2b2b080819080819;
    iq2s_rom[ 988] = 64'h2b2b080819081908;
    iq2s_rom[ 989] = 64'h2b2b080819190808;
    iq2s_rom[ 990] = 64'h2b2b08082b2b082b;
    iq2s_rom[ 991] = 64'h2b2b08082b2b2b2b;
    iq2s_rom[ 992] = 64'h2b2b081919080808;
    iq2s_rom[ 993] = 64'h2b2b0819192b1919;
    iq2s_rom[ 994] = 64'h2b2b082b0808082b;
    iq2s_rom[ 995] = 64'h2b2b082b08082b2b;
    iq2s_rom[ 996] = 64'h2b2b082b082b082b;
    iq2s_rom[ 997] = 64'h2b2b082b082b2b08;
    iq2s_rom[ 998] = 64'h2b2b082b082b2b2b;
    iq2s_rom[ 999] = 64'h2b2b082b2b08082b;
    iq2s_rom[1000] = 64'h2b2b082b2b082b08;
    iq2s_rom[1001] = 64'h2b2b082b2b082b2b;
    iq2s_rom[1002] = 64'h2b2b082b2b2b2b08;
    iq2s_rom[1003] = 64'h2b2b190808080819;
    iq2s_rom[1004] = 64'h2b2b190808081908;
    iq2s_rom[1005] = 64'h2b2b190808190808;
    iq2s_rom[1006] = 64'h2b2b190819080808;
    iq2s_rom[1007] = 64'h2b2b19082b082b19;
    iq2s_rom[1008] = 64'h2b2b19082b2b1908;
    iq2s_rom[1009] = 64'h2b2b191908080808;
    iq2s_rom[1010] = 64'h2b2b191908192b19;
    iq2s_rom[1011] = 64'h2b2b192b19190819;
    iq2s_rom[1012] = 64'h2b2b2b0808082b2b;
    iq2s_rom[1013] = 64'h2b2b2b08082b2b08;
    iq2s_rom[1014] = 64'h2b2b2b082b2b082b;
    iq2s_rom[1015] = 64'h2b2b2b1919191908;
    iq2s_rom[1016] = 64'h2b2b2b192b08192b;
    iq2s_rom[1017] = 64'h2b2b2b2b08082b08;
    iq2s_rom[1018] = 64'h2b2b2b2b08082b2b;
    iq2s_rom[1019] = 64'h2b2b2b2b082b0808;
    iq2s_rom[1020] = 64'h2b2b2b2b082b082b;
    iq2s_rom[1021] = 64'h2b2b2b2b082b2b08;
    iq2s_rom[1022] = 64'h2b2b2b2b2b082b08;
    iq2s_rom[1023] = 64'h2b2b2b2b2b2b2b2b;
  end

  // ---- control: load 11 weight beats, then 8 activation beats
  localparam logic S_LOAD = 1'b0;
  localparam logic S_ACT  = 1'b1;
  logic        state;
  logic [3:0]  w_cnt;                // weight beat 0..10
  logic [2:0]  a_cnt;                // activation beat 0..7, the group being dotted
  assign w_ready = !rst && (state == S_LOAD);
  assign a_ready = !rst && (state == S_ACT);

  // ---- data path of one group: four sub-blocks, ROM read, products, integer partial
  logic [31:0]  qs_cur, sign_cur;
  logic [7:0]   qh_cur, scale_cur;
  logic [9:0]   grid_idx [4];
  logic [63:0]  mag      [4];
  logic signed [17:0] dot_l [4];     // a sub-block's eight products, exact in int18
  logic signed [17:0] sum1_c, sum2_c;
  logic signed [23:0] bsum_c;        // a group's partial, exact in int24
  logic signed [23:0] aux_c;         // second scaled half, exact in int23
  logic        [5:0]  ls1_w, ls2_w;
  logic        [4:0]  ls1_c, ls2_c;
  logic signed [23:0] ls1_s, ls2_s;

  always_comb begin
    qs_cur    = grp_qs[a_cnt];
    sign_cur  = grp_sign[a_cnt];
    qh_cur    = grp_qh[a_cnt];
    scale_cur = grp_scale[a_cnt];
  end

  // idx = qs[4g + l] | ((qh[g] << (8 - 2l)) & 0x300): qh bits 2l, 2l + 1 land in idx bits 8, 9
  always_comb
    for (int l = 0; l < 4; l++) begin
      grid_idx[l] = {qh_cur[2 * l +: 2], qs_cur[8 * l +: 8]};
      mag[l]      = iq2s_rom[grid_idx[l]];
    end

  // eight magnitude x int8 products, magnitude negated where the sign bit is set; the sum is exact.
  // The operands are widened to int16 before the multiply so the product is exact whatever the
  // tool's context-sizing rules are: the widest is 43 * 128 = 5504, far inside int16.
  // The result is assigned to the function name instead of using `return`, because the
  // conda-forge Yosys 0.71 build used for generic estimates does not parse `return`; the
  // construct is equivalent in SystemVerilog, and Verilator accepts both.
  function automatic logic signed [17:0] dot8(input logic [63:0] mag_v,
                                              input logic [7:0]  sign_v,
                                              input logic [63:0] act_v);
    logic signed [15:0] m, x, p;
    logic signed [17:0] acc;
    acc = 18'sd0;
    for (int j = 0; j < 8; j++) begin
      m = signed'({8'b0, mag_v[8 * j +: 8]});                   // magnitude <= 43, positive
      if (sign_v[j]) m = -m;
      x = signed'({{8{act_v[8 * j + 7]}}, act_v[8 * j +: 8]});  // int8, sign-extended to int16
      p = m * x;                                                // <= 5504, exact in int16
      acc = acc + 18'(p);
    end
    dot8 = acc;
  endfunction

  always_comb begin
    dot_l[0] = dot8(mag[0], sign_cur[7:0],   a_data[63:0]);
    dot_l[1] = dot8(mag[1], sign_cur[15:8],  a_data[127:64]);
    dot_l[2] = dot8(mag[2], sign_cur[23:16], a_data[191:128]);
    dot_l[3] = dot8(mag[3], sign_cur[31:24], a_data[255:192]);
    sum1_c = dot_l[0] + dot_l[1];
    sum2_c = dot_l[2] + dot_l[3];
    ls1_w  = 6'd1 + 6'd2 * {2'b00, scale_cur[3:0]};
    ls2_w  = 6'd1 + 6'd2 * {2'b00, scale_cur[7:4]};
    ls1_c  = ls1_w[4:0];
    ls2_c  = ls2_w[4:0];
    ls1_s  = signed'(24'(ls1_w));
    ls2_s  = signed'(24'(ls2_w));
    // Expose the two separately scaled halves expected by the frozen oracle too;
    // the original combined output remains bsum = scaled_half1 + scaled_half2.
    aux_c  = ls2_s * sum2_c;
    bsum_c = ls1_s * sum1_c + aux_c;
  end

  // ---- control and result registers
  always_ff @(posedge clk) begin
    if (rst) begin
      state      <= S_LOAD;
      w_cnt      <= '0;
      a_cnt      <= '0;
      grp_valid  <= 1'b0;
      grp_idx    <= '0;
      grp_bsum   <= '0;
      grp_aux    <= '0;
      grp_ls1    <= '0;
      grp_ls2    <= '0;
      grp_signs  <= '0;
      grp_index  <= '0;
      blk_valid  <= 1'b0;
      blk_d      <= '0;
      blk_scales <= '0;
    end else begin
      grp_valid <= 1'b0;
      blk_valid <= 1'b0;
      case (state)
        S_LOAD: if (w_valid) begin
          blk[64 * w_cnt +: 64] <= w_data;
          if (w_cnt == 4'd10) begin
            state <= S_ACT;
            a_cnt <= '0;
          end else begin
            w_cnt <= w_cnt + 4'd1;
          end
        end
        default: if (a_valid) begin    // S_ACT
          grp_valid <= 1'b1;
          grp_idx   <= a_cnt;
          grp_bsum  <= {{8{bsum_c[23]}}, bsum_c};
          grp_aux   <= {{8{aux_c[23]}}, aux_c};
          grp_ls1   <= ls1_c;
          grp_ls2   <= ls2_c;
          grp_signs <= sign_cur;
          grp_index <= {grid_idx[3], grid_idx[2], grid_idx[1], grid_idx[0]};
          if (a_cnt == 3'd0) begin
            blk_valid  <= 1'b1;
            blk_d      <= blk[15:0];
            blk_scales <= blk[592 +: 64];        // bytes 74..81
          end
          if (a_cnt == 3'd7) begin
            state <= S_LOAD;
            w_cnt <= '0;
          end else begin
            a_cnt <= a_cnt + 3'd1;
          end
        end
      endcase
    end
  end
endmodule