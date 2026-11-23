# Native IQ2_S dot

`rtl/vpu/otpu_iq2s.sv` decodes a ggml IQ2_S block and dots it with int8 activations as integers,
one 32-weight group at a time, and hands back the group partials beside the block's scale bytes.
It is the RTL half of the #8 "native IQ2_S dot" slice: the format's own arithmetic in hardware,
not a load-time expansion to 8 bits and not a call into the MXU. It is checked against an
independent model of the format by `tools/iq_rtl_check.py`, which builds the Verilator testbench
`sim/verilator/tb_iq.sv` and compares every dumped field exactly.

In short:

- **Format.** ggml IQ2_S, 2.5625 bits per weight: 82 bytes per 256-weight super-block — an fp16
  block scale, 64 index bytes and 8 high-index bytes addressing a 1024-entry grid of eight
  magnitudes from {8, 25, 43}, and 8 scale bytes with two 4-bit multipliers. Two sub-blocks
  share a multiplier, so a group of 32 weights is two scaled sums.
- **Hardware.** 11 weight beats of 64 bits (88 bytes, the block's 82 plus 6 pad), then 8
  activation beats of 256 bits (one group of 32 int8 each). 19 cycles a block, 4.32 bytes a
  cycle, one group's result per cycle in the dot phase; a result is registered on the edge its
  beat is accepted. The grid is a 1024 x 64 ROM, read four times a cycle (once per sub-block);
  the product path is 6-bit magnitudes times int8, so no multiplier wider than 8 x 8 is needed.
- **Exact.** The partial is an integer throughout: magnitudes <= 43 and |int8| <= 128 give
  <= 5,459,968 for a group, held in int24 with 1.53x to spare, sign-extended to int32 on the
  output. It equals the reference float dot exactly — `0.125 * d * sum_g bsum[g]` — because the
  dequantiser's sub-block scale `(0.5 + n) * 0.25` is exactly `(2n + 1) / 8` and the partial
  folds the odd `(2n + 1)` in. No rounding and no saturation anywhere in the partial path.
- **Verified.** `python3 tools/iq_rtl_check.py` ran clean against Verilator 5.054 on the `2080ti`
  simulation host: 256 cases at seeds `20261124`, `1` and `7`, and 1024 cases at `20261124` —
  14,386 and 54,322 exact checks, zero mismatches, exit 0, and the testbench's own protocol and
  cycle-count checks reported `errors=0`. That evidence is RTL simulation only: no synthesis,
  placement, timing or board run. The `--selftest` mode (table, RTL text, model against the float
  dequantiser in exact rationals) runs without a simulator and also passed.
- **Not done.** Not integrated into `otpu_vpu`, the ISA, the sequencer or `opentpu/rtlsim.py`;
  not in `RTL_SOURCES`; no synthesis or timing run; no device test. Sections below say what
  remains for the later #8 slices, #9 and #10.

## The format

A `block_iq2_s` is 82 bytes (`ggml-common.h:396`, pinned at `97affb754`):

| bytes | field | use in this module |
|---|---|---|
| 0..1 | `d`, fp16 | `blk_d` raw bits; the float fold `0.125 * d` is the caller's |
| 2..33 | `qs[0..31]` | index byte of sub-block `l` of group `g` at `2 + 4g + l` |
| 34..65 | `qs[32..63]` | sign byte of sub-block `l` of group `g` at `34 + 4g + l` |
| 66..73 | `qh[0..7]` | index bits 8, 9 of sub-block `l` of group `g` at `66 + g` |
| 74..81 | `scales[0..7]` | scale byte of group `g` at `74 + g` |
| 82..87 | pad | the 88-byte beat frame; written to the store, never read |

Group `g` of the 256 weights has four sub-blocks `l = 0..3` of eight weights. The grid entry is
a 64-bit word of eight unsigned magnitudes, element `j` in bits `8j +: 8` (`ggml-common.h:758`):

    idx[g][l] = qs[4g + l] | ((qh[g] << (8 - 2l)) & 0x300)      // 0..1023
    w[g][l][j] = iq2s_grid[idx[g][l]][j] * (sign[g][l] bit j set ? -1 : +1)
    bsum[g]   = ls1 * (dot of l = 0 + dot of l = 1)
              + ls2 * (dot of l = 2 + dot of l = 3)
    ls1 = 1 + 2 * (scales[g] & 15)
    ls2 = 1 + 2 * (scales[g] >> 4)

That is the arithmetic of the generic dot (`ggml-cpu/quants.c:998`,
`ggml_vec_dot_iq2_s_q8_K_generic`) and of the dequantiser (`ggml-quants.c:2543`, grid at
`:2561`) element for element. The AVX2 kernel (`ggml-cpu/arch/x86/quants.c:3075`), the NEON one
(`ggml-cpu/arch/arm/quants.c:3810`), the Metal dequantiser (`ggml-metal/kernels/dequantize.h:604`)
and Strata's own `dot_generic.cpp` (`engine/src/quant/dot_generic.cpp:800`) carry the same
arithmetic, so the integer form is the format's definition and not a hardware shortcut. The
quantiser (`quantize_row_iq2_s_impl`, `ggml-quants.c:5155`) writes magnitudes into the low eight
bits and the two index bits into `qh`, and marks a negative weight by setting its sign bit.

The dequantiser's value for a weight is `d * (0.5 + n) * 0.25 * grid_byte * sign`, and
`(0.5 + n) * 0.25 = (2n + 1) / 8`. Summing a block's dequantised weights against its int8
activations therefore gives exactly

    dot(block) = 0.125 * fp16_to_fp32(d) * sum_g bsum[g]

— which is why the module needs no float at all: the caller scales and sums the eight int32
partials. The activations' own block scale (`block_q8_K.d`) and the `0.125 * d` fold stay in the
caller's float path, as they do in the reference dot; `block_q8_K.bsums` is not used (those are
per-16 sums for a different kernel).

## The module

`rtl/vpu/otpu_iq2s.sv`, `module otpu_iq2s`:

| port | width | meaning |
|---|---|---|
| `clk`, `rst` | 1 | clock; synchronous active-high reset |
| `w_valid`, `w_ready`, `w_data` | 1, 1, 64 | weight beat, byte `8b + j` in lane `j` of beat `b` |
| `a_valid`, `a_ready`, `a_data` | 1, 1, 256 | activation beat, 32 int8, byte `8l + j` is sub-block `l`'s element `j` |
| `grp_valid`, `grp_idx` | 1, 3 | a group result this cycle; which group, 0..7 |
| `grp_bsum` | 32 | the group's partial, int32 two's complement |
| `grp_ls1`, `grp_ls2` | 5 each | `1 + 2 * (scales[g] & 15)`, `1 + 2 * (scales[g] >> 4)`, odd 1..31 |
| `grp_signs` | 32 | the group's raw sign bytes, `l = 0` in bits 7:0 |
| `grp_index` | 40 | the group's four 10-bit grid indices, `l = 0` in bits 9:0 |
| `blk_valid` | 1 | block metadata this cycle, with `grp_idx = 0` |
| `blk_d`, `blk_scales` | 16, 64 | raw fp16 bits; raw scale bytes 74..81, group `g` in byte `g` |

Handshake and timing:

- A beat is accepted on the clock edge where `*_valid` and `*_ready` are both high. `w_ready` is
  high only in the load phase and `a_ready` only in the dot phase, so the two streams cannot
  interleave; a source that holds `*_valid` low waits in place, and nothing is lost if either
  side stalls between beats of a block.
- Eleven weight beats fill the 88-byte store, then eight activation beats each prompt one
  group's result. The result registers and `grp_valid` update on the accepting edge, so a group's
  outputs are valid in the cycle after its beat; the last group's cycle is the block's last.
- A block is therefore 19 cycles from its first weight beat to its last group's result: 11 load
  plus 8 dot, 82 bytes in 19 cycles (4.32 bytes a cycle), one group a cycle in the dot phase.
  Blocks are serialised through the two phases (one store, no double buffering), so a steady
  stream also runs 19 cycles a block.
- `blk_valid`, `blk_d` and `blk_scales` come with `grp_idx = 0`; the block's `d` is raw fp16 bits
  and the scale bytes are raw, so the caller does all interpretation and float work.
- Reset is synchronous and active high. It clears the control state and the output registers;
  the 88-byte store has no reset, since a beat writes every byte of it before anything reads it.
- With `rst` high, both ready signals are low, so no beat is accepted while the outputs are being
  cleared.

## The check

`tools/iq_rtl_check.py` (stdlib only) generates seeded cases, writes a vector file, builds the
testbench with Verilator and compares its dump field by field with an independent model of the
format written in the script. The testbench checks the protocol (ready/valid exclusivity between
the two streams, result strobes, the 19 accepted cycles measured from the first weight beat's
accepting edge to the last activation beat's, and the structural fields it can see in the block
bytes); the script checks the arithmetic.

    python3 tools/iq_rtl_check.py --selftest          # no simulator: table, RTL text, model
    python3 tools/iq_rtl_check.py                     # build, run 256 cases, compare
    python3 tools/iq_rtl_check.py --cases 1024        # more cases
    python3 tools/iq_rtl_check.py --help

It needs Verilator 5.x (measured with 5.054) and a C++ compiler on `PATH` (or `--verilator
<path>`); the build goes to
the ignored `build/iq_rtl/` (`Vtb_iq`), with the vectors and the raw dump left beside it. No
Xilinx primitives are involved, so Verilator is the only simulator the module needs. Exit status
is 0 on success, 1 on a mismatch, and 2 when the tools are missing or the build or run failed.

Vectors, one case a line pair (`B` then `A`, hex, spaces):

    B <22 words>    the 82-byte block padded to 88 bytes, 32-bit little-endian words
    A <64 words>    the 256 int8 activations, 32-bit little-endian words

Dump, one line per group after the case's line:

    case <n> d <4 hex> scales <16 hex>
    g <0..7> <8 hex partial> <2 hex ls1> <2 hex ls2> <8 hex signs> <10 hex indices>

Cases cover every corner that changes the arithmetic: unconstrained blocks, zero weights, zero
activations, saturated scale and sign bytes with extreme int8 values, one `qh` bit pair per
group, and sparse blocks. The partials, multipliers, sign bytes and grid indices must match
exactly, not within a tolerance; the reference model recomputes them from the block bytes with
its own index, sign and scale code, sharing nothing with the RTL.

The `--selftest` mode also re-derives every group's partial from the float dequantisation in
exact rational arithmetic (no float rounding): the dequantised weights dotted with the int8
activations must equal `d * bsum / 8` for that group, and every partial must fit int24. That is
the integer-exactness claim above, checked rather than asserted. It checks the RTL text too: the
1024 ROM entries must equal the pinned table value for value, every port must be declared with
the intended width and connected by name in the testbench, and the module must contain no vendor
primitive, no delay and no simulation system task.

Measured on the `2080ti` simulation host with Verilator 5.054 (a user-local conda-forge build),
from a clean sync of this tree, one JSON object and exit status per run:

| command | `cases` | `checks` | `mismatches` | exit |
|---|---|---|---|---|
| `python3 tools/iq_rtl_check.py --cases 256` | 256 | 14,386 | none | 0 |
| `python3 tools/iq_rtl_check.py --seed 1 --cases 256` | 256 | 14,386 | none | 0 |
| `python3 tools/iq_rtl_check.py --seed 7 --cases 256` | 256 | 14,386 | none | 0 |
| `python3 tools/iq_rtl_check.py --seed 20261124 --cases 1024` | 1024 | 54,322 | none | 0 |

`checks` is `1071` static plus `52` per case plus `3`. Each run's testbench summary was
`IQ2STEST cases=<cases> beats=<19 x cases> cycles=<19 x cases + 4> errors=0` (for example
`IQ2STEST cases=256 beats=4864 cycles=4868 errors=0`), a JSON verdict of
`{"ok": true, "cases": ..., "checks": ..., "mismatches": [], "seed": ...,
"verilator": "Verilator 5.054 ...", "grid_sha256": "e1aa1473...", ...}`, and exit 0. The
independent model had earlier been cross-checked bit-exactly against the generic
`ggml_vec_dot_iq2_s_q8_K_generic` and against all 256 dequantised floats through a throw-away
host harness (256 blocks, five seeds); that harness is not committed, so the committed evidence
path is the script and the testbench.

This is RTL simulation evidence only: there is no synthesis, placement, timing or board evidence
for this module yet (see "What is not done").

## Integer exactness

| quantity | bound | fits |
|---|---|---|
| one product `mag * act` | 43 * 128 = 5,504 | int16 with 11x to spare |
| one sub-block, eight products | 8 * 5,504 = 44,032 | int18 |
| one sub-block pair `sumi1`, `sumi2` | 88,064 | int18 |
| one group `bsum = ls1 * sumi1 + ls2 * sumi2` | 31 * 2 * 88,064 = 5,459,968 | int24 with 1.53x to spare, int32 on the output |

The magnitudes are the largest grid byte (the table's alphabet is exactly {8, 25, 43}), `|int8|`
is at most 128, and the multipliers are odd values 1..31. Every step is integer addition and
8 x 8 multiplication in two's complement, so nothing rounds, overflows or saturates: a wider
format would need the partial register widened, not the arithmetic changed. The only float in the
reference path is the caller's `0.125 * d * d_act` fold, applied to a sum of exact integers.

## ROM and resources

- **Grid ROM.** 1024 entries of 64 bits = 8 KiB. Four sub-blocks are read in the same cycle, so
  an FPGA implementation may replicate the ROM four times (32 KiB, about 8 block RAMs of 36 Kb
  each) or bank it in four; how that maps is a later decision.
- **Multipliers.** 32 products of a 6-bit magnitude and an int8 a cycle in the product path,
  plus two 5 x 18 multipliers for the scale application. On 7-series a DSP48E1 could pack two
  8 x 8 products, suggesting roughly 16-18 DSPs, or about 1.5-2K LUTs if built in logic.
- **Estimated, not measured.** No synthesis or timing run was done for this module, so these are
  sizing estimates only; there is no area, frequency or utilisation report. The ROM is written as
  a simulation-visible `initial` block (plain `logic [63:0] rom [0:1023]` with asynchronous
  reads) because it is the portable way to write a table and Verilator can load it; a
  technology wrapper can replace it with a generated ROM later without touching the datapath.

## What is not done

- Not integrated: `otpu_vpu` does not instantiate the module, and the ISA, the sequencer, the
  compiler and `opentpu/rtlsim.py`/`RTL_SOURCES` do not know about it. That wiring, and the
  choice of who applies `0.125 * d` and how the eight partials are accumulated across blocks,
  are later #8 slices.
- No float: the caller must scale the partials and sum them; this module stops at integers.
- No synthesis, placement or timing run, and no measured area, frequency or power (the numbers
  above are estimates).
- No device test, no bitstream and no card measurement.
- One block at a time: the store is single-buffered and the two phases do not overlap, so there
  is no double-buffered weight stream and no pipelining of blocks. If the caller's activation
  stream cannot follow the dot phase, it stalls; a faster version would add a second store and
  overlap load with dot.
- The grid ROM's `initial` contents are a simulation form; a synthesis flow is expected to infer
  a ROM from it, which is plausible for this size and read pattern but unverified here.

## Reproduce

    python3 tools/iq_rtl_check.py --selftest                 # no simulator
    python3 tools/iq_rtl_check.py --cases 256                # needs Verilator 5.x
    python3 tools/iq_rtl_check.py --seed 1 --cases 256
    python3 tools/iq_rtl_check.py --seed 7 --cases 256
    python3 tools/iq_rtl_check.py --seed 20261124 --cases 1024

`--help` lists `--cases`, `--seed`, `--build`, `--verilator`; the script prints one JSON object
and exits non-zero on failure. Build only, by hand:

    verilator --binary --timing --top-module tb_iq -j 0 -Wno-fatal -O3 \
        --Mdir build/iq_rtl sim/verilator/tb_iq.sv rtl/vpu/otpu_iq2s.sv
    build/iq_rtl/Vtb_iq +vectors=build/iq_rtl/vectors.hex +out=build/iq_rtl/dump.txt

## Sources

Pinned at `97affb754`, in the `strata-ggml` checkout
(`engine/third_party/strata-ggml/ggml/src/`):

- `ggml-common.h:396` — `block_iq2_s`, 82 bytes.
- `ggml-common.h:758` — `iq2s_grid`, `GGML_TABLE_BEGIN(uint64_t, iq2s_grid, 1024)`; the embedded
  table's 8192 little-endian bytes hash to
  `e1aa1473412b0552c2174c30ef22ab4073f6a181b85a17056e8249bd2932fd88` and are byte-equal to
  `terminal/tools/strata_quant_tables.IQ2S_GRID`.
- `ggml-cpu/quants.c:998` — `ggml_vec_dot_iq2_s_q8_K_generic`, the integer arithmetic this module
  implements.
- `ggml-cpu/arch/x86/quants.c:3075`, `ggml-cpu/arch/arm/quants.c:3810` — the AVX2 and NEON
  kernels, same arithmetic.
- `ggml-metal/kernels/dequantize.h:604` — the Metal dequantiser.
- `ggml-quants.c:2543` (grid at `:2561`) — `dequantize_row_iq2_s`, the float form the integer
  form must equal; `ggml-quants.c:5155` — `quantize_row_iq2_s_impl`, the packer.
- `engine/src/quant/dot_generic.cpp:800` — Strata's own generic dot, same arithmetic.

The embedded grid table is a transcription of ggml's `iq2s_grid`, which is MIT-licensed
(Copyright (c) 2023 Georgi Gerganov and the ggml contributors); the complete notice is copied at
`tools/packed_oracle/licenses/GGML-MIT.txt` and recorded in `THIRD_PARTY.md`, and openTPU's
Apache-2.0 licence does not replace it.

In this repository: `rtl/vpu/otpu_iq2s.sv`, `sim/verilator/tb_iq.sv`, `tools/iq_rtl_check.py`.