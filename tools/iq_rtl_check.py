#!/usr/bin/env python3
"""Check the native IQ2_S dot primitive in rtl/vpu/otpu_iq2s.sv.

The module streams one 82-byte ggml IQ2_S block as 11 64-bit weight beats and
its 256 int8 activations as 8 256-bit beats, and hands back one result per
32-weight group: the int32 partial ls1*sumi1 + ls2*sumi2 plus the group's scale
nibbles, four sign bytes and four 10-bit grid indices.  This script generates
seeded blocks and activations, runs the Verilator testbench
sim/verilator/tb_iq.sv on them, and compares every dumped field with an
independent model of the format written here from the ggml sources - no shared
code with the RTL, no C compiler, no float in the partial path.  The partials
must agree exactly (not within a tolerance), and so must d, the scale bytes,
the multipliers, the sign bytes and the grid indices.

The 1024 x 64-bit iq2s_grid table is embedded verbatim, four values a line as
in the ggml source, and checked against its sha256.  Table, model and the RTL
text can be checked without a simulator:

    python3 tools/iq_rtl_check.py --selftest

The full check needs Verilator 5.x and a C++ compiler:

    python3 tools/iq_rtl_check.py

It builds build/iq_rtl/Vtb_iq (--binary, --timing), runs --cases cases
(default 256) and prints one JSON object:

    {"ok": true, "cases": 256, "checks": 14386, "mismatches": [], ...}

Exit status is 0 when ok, 1 on a model/testbench mismatch, and 2 when the
tools are missing or the build or run failed.  The generated vectors and the
raw dump stay in build/iq_rtl/ for inspection (build/ is ignored).

The selftest additionally re-derives every group partial from the float
dequantisation of the same block (blocks are dequantised to d * (2n+1)/8 *
grid * sign, dotted with the int8 activations in exact rational arithmetic,
and compared with 0.125 * d * bsum): that is the identity that makes the
integer path equal to the reference float path, checked exactly.

Reference sources, pinned at strata-gguf-inventory 97affb754:

    ggml/src/ggml-common.h:396          block_iq2_s layout (82 bytes)
    ggml/src/ggml-common.h:758          iq2s_grid (the table embedded below)
    ggml/src/ggml-cpu/quants.c:998      ggml_vec_dot_iq2_s_q8_K_generic
    ggml/src/ggml-quants.c:2543         dequantize_row_iq2_s (the float form)

The model here was additionally cross-checked bit-exactly against the generic
ggml kernel through a local host harness (not committed); this file is the
committed evidence path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import shutil
import struct
import subprocess
import sys
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RTL = ROOT / "rtl" / "vpu" / "otpu_iq2s.sv"
TB = ROOT / "sim" / "verilator" / "tb_iq.sv"
DEFAULT_BUILD = ROOT / "build" / "iq_rtl"

QK_K = 256          # super-block size
BLOCK_BYTES = 82    # sizeof(block_iq2_s)
N_GROUPS = 8        # QK_K / 32
N_CASES = 256
DEFAULT_SEED = 20261124

# iq2s_grid sha256 of the 8192 little-endian bytes, pinned at 97affb754
GRID_SHA256 = "e1aa1473412b0552c2174c30ef22ab4073f6a181b85a17056e8249bd2932fd88"

# magnitude alphabet of the table: every byte of every entry is 8, 25 or 43
GRID_MAGNITUDES = (8, 25, 43)

# the RTL keeps the partial in int24; the format's worst case is 5459968
PARTIAL_MIN = -(1 << 23)
PARTIAL_MAX = (1 << 23) - 1

# copied from opentpu/rtlsim.py VFLAGS, plus --timing (the testbench uses delays)
VERILATOR_FLAGS = [
    "--binary",
    "--timing",
    "--top-module", "tb_iq",
    "-j", "0",
    "--timescale", "1ns/1ps",
    "-Wno-fatal",
    "-Wno-WIDTHEXPAND",
    "-Wno-WIDTHTRUNC",
    "-Wno-UNUSEDSIGNAL",
    "-Wno-UNUSEDPARAM",
    "-O3",
    "--x-assign", "0",
    "--x-initial", "0",
]

# port name -> bit width, as rtl/vpu/otpu_iq2s.sv must declare them
PORTS = (
    ("clk", 1),
    ("rst", 1),
    ("w_valid", 1),
    ("w_ready", 1),
    ("w_data", 64),
    ("a_valid", 1),
    ("a_ready", 1),
    ("a_data", 256),
    ("grp_valid", 1),
    ("grp_idx", 3),
    ("grp_bsum", 32),
    ("grp_ls1", 5),
    ("grp_ls2", 5),
    ("grp_signs", 32),
    ("grp_index", 40),
    ("blk_valid", 1),
    ("blk_d", 16),
    ("blk_scales", 64),
)

ROM_LINE = re.compile(r"^\s*iq2s_rom\[\s*(\d+)\s*\]\s*=\s*64'h([0-9A-Fa-f]{16})\s*;\s*$", re.M)
CASE_LINE = re.compile(r"^case (\d+) d ([0-9A-Fa-f]+) scales ([0-9A-Fa-f]+)$")
GROUP_LINE = re.compile(
    r"^g (\d+) ([0-9A-Fa-f]+) ([0-9A-Fa-f]+) ([0-9A-Fa-f]+) ([0-9A-Fa-f]+) ([0-9A-Fa-f]+)$"
)
SUMMARY_LINE = re.compile(r"IQ2STEST cases=(\d+) beats=(\d+) cycles=(\d+) errors=(\d+)")
XILINX_PRIMITIVE = re.compile(
    r"\b(BUFG|BUFH|BUFIO|BUFR|FDRE|FDSE|FDPE|LUT[1-6]|RAMB18|RAMB36|DSP48\w*|"
    r"IBUF|OBUF|MMCM\w*|PLLE?\d?\w*|STARTUPE\w*)\b"
)
SIM_SYSTEM_TASK = re.compile(r"\$(display|finish|fatal|stop|write|monitor|dumpfile|fopen|fclose|value\$plusargs)")

# ---- iq2s_grid, embedded verbatim from ggml/src/ggml-common.h:758
# GGML_TABLE_BEGIN(uint64_t, iq2s_grid, 1024) ... GGML_TABLE_END(), four values a line
IQ2S_GRID = (
    0x0808080808080808, 0x080808080808082B, 0x0808080808081919, 0x0808080808082B08,
    0x0808080808082B2B, 0x0808080808190819, 0x0808080808191908, 0x080808080819192B,
    0x0808080808192B19, 0x08080808082B0808, 0x08080808082B082B, 0x08080808082B1919,
    0x08080808082B2B08, 0x0808080819080819, 0x0808080819081908, 0x080808081908192B,
    0x0808080819082B19, 0x0808080819190808, 0x080808081919082B, 0x0808080819191919,
    0x0808080819192B08, 0x08080808192B0819, 0x08080808192B1908, 0x08080808192B192B,
    0x08080808192B2B19, 0x080808082B080808, 0x080808082B08082B, 0x080808082B081919,
    0x080808082B082B08, 0x080808082B190819, 0x080808082B191908, 0x080808082B2B0808,
    0x080808082B2B1919, 0x080808082B2B2B2B, 0x0808081908080819, 0x0808081908081908,
    0x080808190808192B, 0x0808081908082B19, 0x0808081908190808, 0x080808190819082B,
    0x0808081908191919, 0x0808081908192B08, 0x08080819082B0819, 0x08080819082B1908,
    0x0808081919080808, 0x080808191908082B, 0x0808081919081919, 0x0808081919082B08,
    0x0808081919190819, 0x0808081919191908, 0x080808191919192B, 0x0808081919192B19,
    0x08080819192B0808, 0x08080819192B1919, 0x08080819192B2B08, 0x080808192B080819,
    0x080808192B081908, 0x080808192B190808, 0x080808192B19082B, 0x080808192B191919,
    0x080808192B2B0819, 0x080808192B2B1908, 0x0808082B08080808, 0x0808082B0808082B,
    0x0808082B08081919, 0x0808082B08082B08, 0x0808082B08190819, 0x0808082B08191908,
    0x0808082B082B0808, 0x0808082B082B2B2B, 0x0808082B19080819, 0x0808082B19081908,
    0x0808082B1908192B, 0x0808082B19082B19, 0x0808082B19190808, 0x0808082B19191919,
    0x0808082B2B080808, 0x0808082B2B081919, 0x0808082B2B082B2B, 0x0808082B2B191908,
    0x0808082B2B2B082B, 0x0808190808080819, 0x0808190808081908, 0x080819080808192B,
    0x0808190808082B19, 0x0808190808190808, 0x080819080819082B, 0x0808190808191919,
    0x0808190808192B08, 0x08081908082B0819, 0x08081908082B1908, 0x08081908082B192B,
    0x08081908082B2B19, 0x0808190819080808, 0x080819081908082B, 0x0808190819081919,
    0x0808190819082B08, 0x0808190819082B2B, 0x0808190819190819, 0x0808190819191908,
    0x080819081919192B, 0x0808190819192B19, 0x08081908192B0808, 0x08081908192B082B,
    0x08081908192B1919, 0x080819082B080819, 0x080819082B081908, 0x080819082B08192B,
    0x080819082B082B19, 0x080819082B190808, 0x080819082B191919, 0x080819082B192B08,
    0x080819082B2B0819, 0x080819082B2B1908, 0x0808191908080808, 0x080819190808082B,
    0x0808191908081919, 0x0808191908082B08, 0x0808191908082B2B, 0x0808191908190819,
    0x0808191908191908, 0x080819190819192B, 0x0808191908192B19, 0x08081919082B0808,
    0x08081919082B1919, 0x08081919082B2B08, 0x0808191919080819, 0x0808191919081908,
    0x080819191908192B, 0x0808191919082B19, 0x0808191919190808, 0x080819191919082B,
    0x0808191919191919, 0x0808191919192B08, 0x08081919192B0819, 0x08081919192B1908,
    0x080819192B080808, 0x080819192B08082B, 0x080819192B081919, 0x080819192B082B08,
    0x080819192B190819, 0x080819192B191908, 0x080819192B2B0808, 0x0808192B08080819,
    0x0808192B08081908, 0x0808192B0808192B, 0x0808192B08082B19, 0x0808192B08190808,
    0x0808192B08191919, 0x0808192B19080808, 0x0808192B19081919, 0x0808192B19082B08,
    0x0808192B19190819, 0x0808192B19191908, 0x0808192B192B0808, 0x0808192B2B080819,
    0x0808192B2B081908, 0x0808192B2B190808, 0x08082B0808080808, 0x08082B080808082B,
    0x08082B0808081919, 0x08082B0808082B08, 0x08082B0808190819, 0x08082B0808191908,
    0x08082B080819192B, 0x08082B0808192B19, 0x08082B08082B0808, 0x08082B08082B1919,
    0x08082B08082B2B2B, 0x08082B0819080819, 0x08082B0819081908, 0x08082B081908192B,
    0x08082B0819082B19, 0x08082B0819190808, 0x08082B081919082B, 0x08082B0819191919,
    0x08082B0819192B08, 0x08082B08192B0819, 0x08082B08192B1908, 0x08082B082B080808,
    0x08082B082B081919, 0x08082B082B191908, 0x08082B082B2B2B2B, 0x08082B1908080819,
    0x08082B1908081908, 0x08082B1908190808, 0x08082B190819082B, 0x08082B1908191919,
    0x08082B1908192B08, 0x08082B19082B0819, 0x08082B1919080808, 0x08082B1919081919,
    0x08082B1919082B08, 0x08082B1919190819, 0x08082B1919191908, 0x08082B19192B0808,
    0x08082B192B080819, 0x08082B192B190808, 0x08082B2B08080808, 0x08082B2B08190819,
    0x08082B2B08191908, 0x08082B2B082B082B, 0x08082B2B082B2B08, 0x08082B2B082B2B2B,
    0x08082B2B19190808, 0x08082B2B2B192B19, 0x0819080808080819, 0x0819080808081908,
    0x081908080808192B, 0x0819080808082B19, 0x0819080808190808, 0x081908080819082B,
    0x0819080808191919, 0x0819080808192B08, 0x08190808082B0819, 0x08190808082B1908,
    0x08190808082B192B, 0x0819080819080808, 0x081908081908082B, 0x0819080819081919,
    0x0819080819082B08, 0x0819080819190819, 0x0819080819191908, 0x081908081919192B,
    0x0819080819192B19, 0x08190808192B0808, 0x08190808192B082B, 0x08190808192B1919,
    0x08190808192B2B08, 0x081908082B080819, 0x081908082B081908, 0x081908082B08192B,
    0x081908082B190808, 0x081908082B191919, 0x081908082B192B08, 0x081908082B2B0819,
    0x081908082B2B1908, 0x0819081908080808, 0x081908190808082B, 0x0819081908081919,
    0x0819081908082B08, 0x0819081908082B2B, 0x0819081908190819, 0x0819081908191908,
    0x081908190819192B, 0x0819081908192B19, 0x08190819082B0808, 0x08190819082B082B,
    0x08190819082B1919, 0x08190819082B2B08, 0x0819081919080819, 0x0819081919081908,
    0x081908191908192B, 0x0819081919082B19, 0x0819081919190808, 0x081908191919082B,
    0x0819081919191919, 0x0819081919192B08, 0x08190819192B0819, 0x08190819192B1908,
    0x081908192B080808, 0x081908192B08082B, 0x081908192B081919, 0x081908192B082B08,
    0x081908192B190819, 0x081908192B191908, 0x0819082B08080819, 0x0819082B08081908,
    0x0819082B08082B19, 0x0819082B08190808, 0x0819082B08191919, 0x0819082B082B0819,
    0x0819082B082B1908, 0x0819082B19080808, 0x0819082B19081919, 0x0819082B19190819,
    0x0819082B19191908, 0x0819082B2B080819, 0x0819082B2B081908, 0x0819082B2B190808,
    0x0819190808080808, 0x081919080808082B, 0x0819190808081919, 0x0819190808082B08,
    0x0819190808190819, 0x0819190808191908, 0x081919080819192B, 0x0819190808192B19,
    0x08191908082B0808, 0x08191908082B1919, 0x08191908082B2B08, 0x0819190819080819,
    0x0819190819081908, 0x081919081908192B, 0x0819190819082B19, 0x0819190819190808,
    0x081919081919082B, 0x0819190819191919, 0x0819190819192B08, 0x08191908192B0819,
    0x08191908192B1908, 0x081919082B080808, 0x081919082B08082B, 0x081919082B081919,
    0x081919082B082B08, 0x081919082B190819, 0x081919082B191908, 0x081919082B2B0808,
    0x0819191908080819, 0x0819191908081908, 0x081919190808192B, 0x0819191908082B19,
    0x0819191908190808, 0x081919190819082B, 0x0819191908191919, 0x0819191908192B08,
    0x08191919082B0819, 0x08191919082B1908, 0x0819191919080808, 0x081919191908082B,
    0x0819191919081919, 0x0819191919082B08, 0x0819191919190819, 0x0819191919191908,
    0x08191919192B0808, 0x081919192B080819, 0x081919192B081908, 0x081919192B190808,
    0x0819192B08080808, 0x0819192B08081919, 0x0819192B08082B08, 0x0819192B08190819,
    0x0819192B08191908, 0x0819192B082B0808, 0x0819192B19080819, 0x0819192B19081908,
    0x0819192B19190808, 0x0819192B2B080808, 0x0819192B2B2B2B2B, 0x08192B0808080819,
    0x08192B0808081908, 0x08192B080808192B, 0x08192B0808082B19, 0x08192B0808190808,
    0x08192B0808191919, 0x08192B0808192B08, 0x08192B08082B0819, 0x08192B0819080808,
    0x08192B081908082B, 0x08192B0819081919, 0x08192B0819082B08, 0x08192B0819190819,
    0x08192B0819191908, 0x08192B08192B0808, 0x08192B082B080819, 0x08192B082B081908,
    0x08192B1908080808, 0x08192B190808082B, 0x08192B1908081919, 0x08192B1908082B08,
    0x08192B1908190819, 0x08192B1908191908, 0x08192B19082B0808, 0x08192B1919080819,
    0x08192B1919081908, 0x08192B1919190808, 0x08192B19192B2B19, 0x08192B192B2B082B,
    0x08192B2B08081908, 0x08192B2B08190808, 0x08192B2B19080808, 0x08192B2B1919192B,
    0x082B080808080808, 0x082B08080808082B, 0x082B080808081919, 0x082B080808082B08,
    0x082B080808190819, 0x082B080808191908, 0x082B08080819192B, 0x082B080808192B19,
    0x082B0808082B0808, 0x082B0808082B1919, 0x082B0808082B2B2B, 0x082B080819080819,
    0x082B080819081908, 0x082B080819190808, 0x082B08081919082B, 0x082B080819191919,
    0x082B0808192B1908, 0x082B08082B080808, 0x082B08082B082B2B, 0x082B08082B191908,
    0x082B08082B2B2B2B, 0x082B081908080819, 0x082B081908081908, 0x082B081908190808,
    0x082B08190819082B, 0x082B081908191919, 0x082B0819082B0819, 0x082B081919080808,
    0x082B08191908082B, 0x082B081919081919, 0x082B081919190819, 0x082B081919191908,
    0x082B0819192B0808, 0x082B08192B080819, 0x082B08192B081908, 0x082B08192B190808,
    0x082B082B08080808, 0x082B082B08082B2B, 0x082B082B082B082B, 0x082B082B082B2B08,
    0x082B082B082B2B2B, 0x082B082B19081908, 0x082B082B19190808, 0x082B082B2B082B08,
    0x082B082B2B082B2B, 0x082B082B2B2B2B08, 0x082B190808080819, 0x082B190808081908,
    0x082B19080808192B, 0x082B190808082B19, 0x082B190808190808, 0x082B190808191919,
    0x082B190808192B08, 0x082B1908082B0819, 0x082B1908082B1908, 0x082B190819080808,
    0x082B19081908082B, 0x082B190819081919, 0x082B190819082B08, 0x082B190819190819,
    0x082B190819191908, 0x082B1908192B0808, 0x082B19082B080819, 0x082B19082B081908,
    0x082B19082B190808, 0x082B191908080808, 0x082B191908081919, 0x082B191908082B08,
    0x082B191908190819, 0x082B191908191908, 0x082B1919082B0808, 0x082B191919080819,
    0x082B191919081908, 0x082B191919190808, 0x082B1919192B192B, 0x082B19192B080808,
    0x082B192B08080819, 0x082B192B08081908, 0x082B192B08190808, 0x082B192B19080808,
    0x082B192B19192B19, 0x082B2B0808080808, 0x082B2B0808081919, 0x082B2B0808190819,
    0x082B2B0808191908, 0x082B2B0819080819, 0x082B2B0819081908, 0x082B2B0819190808,
    0x082B2B082B082B2B, 0x082B2B082B2B2B2B, 0x082B2B1908080819, 0x082B2B1908081908,
    0x082B2B1908190808, 0x082B2B192B191919, 0x082B2B2B08082B2B, 0x082B2B2B082B082B,
    0x082B2B2B192B1908, 0x082B2B2B2B082B08, 0x082B2B2B2B082B2B, 0x1908080808080819,
    0x1908080808081908, 0x190808080808192B, 0x1908080808082B19, 0x1908080808190808,
    0x190808080819082B, 0x1908080808191919, 0x1908080808192B08, 0x1908080808192B2B,
    0x19080808082B0819, 0x19080808082B1908, 0x19080808082B192B, 0x1908080819080808,
    0x190808081908082B, 0x1908080819081919, 0x1908080819082B08, 0x1908080819082B2B,
    0x1908080819190819, 0x1908080819191908, 0x190808081919192B, 0x1908080819192B19,
    0x19080808192B0808, 0x19080808192B082B, 0x19080808192B1919, 0x190808082B080819,
    0x190808082B081908, 0x190808082B190808, 0x190808082B191919, 0x190808082B192B08,
    0x190808082B2B0819, 0x190808082B2B1908, 0x1908081908080808, 0x190808190808082B,
    0x1908081908081919, 0x1908081908082B08, 0x1908081908190819, 0x1908081908191908,
    0x190808190819192B, 0x1908081908192B19, 0x19080819082B0808, 0x19080819082B082B,
    0x19080819082B1919, 0x1908081919080819, 0x1908081919081908, 0x190808191908192B,
    0x1908081919082B19, 0x1908081919190808, 0x190808191919082B, 0x1908081919191919,
    0x1908081919192B08, 0x19080819192B0819, 0x19080819192B1908, 0x190808192B080808,
    0x190808192B08082B, 0x190808192B081919, 0x190808192B082B08, 0x190808192B190819,
    0x190808192B191908, 0x190808192B2B0808, 0x1908082B08080819, 0x1908082B08081908,
    0x1908082B08190808, 0x1908082B0819082B, 0x1908082B08191919, 0x1908082B08192B08,
    0x1908082B082B1908, 0x1908082B19080808, 0x1908082B19081919, 0x1908082B19082B08,
    0x1908082B19190819, 0x1908082B19191908, 0x1908082B192B0808, 0x1908082B2B080819,
    0x1908082B2B081908, 0x1908190808080808, 0x190819080808082B, 0x1908190808081919,
    0x1908190808082B08, 0x1908190808082B2B, 0x1908190808190819, 0x1908190808191908,
    0x190819080819192B, 0x1908190808192B19, 0x19081908082B0808, 0x19081908082B082B,
    0x19081908082B1919, 0x19081908082B2B08, 0x1908190819080819, 0x1908190819081908,
    0x190819081908192B, 0x1908190819082B19, 0x1908190819190808, 0x190819081919082B,
    0x1908190819191919, 0x1908190819192B08, 0x19081908192B0819, 0x19081908192B1908,
    0x190819082B080808, 0x190819082B08082B, 0x190819082B081919, 0x190819082B082B08,
    0x190819082B190819, 0x190819082B191908, 0x190819082B2B0808, 0x1908191908080819,
    0x1908191908081908, 0x190819190808192B, 0x1908191908082B19, 0x1908191908190808,
    0x190819190819082B, 0x1908191908191919, 0x1908191908192B08, 0x19081919082B0819,
    0x19081919082B1908, 0x1908191919080808, 0x190819191908082B, 0x1908191919081919,
    0x1908191919082B08, 0x1908191919190819, 0x1908191919191908, 0x19081919192B0808,
    0x19081919192B2B2B, 0x190819192B080819, 0x190819192B081908, 0x190819192B190808,
    0x1908192B08080808, 0x1908192B0808082B, 0x1908192B08081919, 0x1908192B08082B08,
    0x1908192B08190819, 0x1908192B08191908, 0x1908192B082B0808, 0x1908192B19080819,
    0x1908192B19081908, 0x1908192B19190808, 0x1908192B2B080808, 0x1908192B2B2B1919,
    0x19082B0808080819, 0x19082B0808081908, 0x19082B0808082B19, 0x19082B0808190808,
    0x19082B080819082B, 0x19082B0808191919, 0x19082B0808192B08, 0x19082B08082B0819,
    0x19082B08082B1908, 0x19082B0819080808, 0x19082B081908082B, 0x19082B0819081919,
    0x19082B0819082B08, 0x19082B0819190819, 0x19082B0819191908, 0x19082B08192B0808,
    0x19082B082B081908, 0x19082B082B190808, 0x19082B1908080808, 0x19082B190808082B,
    0x19082B1908081919, 0x19082B1908082B08, 0x19082B1908190819, 0x19082B1908191908,
    0x19082B19082B0808, 0x19082B1919080819, 0x19082B1919081908, 0x19082B1919190808,
    0x19082B192B080808, 0x19082B192B19192B, 0x19082B2B08080819, 0x19082B2B08081908,
    0x19082B2B08190808, 0x19082B2B19080808, 0x1919080808080808, 0x191908080808082B,
    0x1919080808081919, 0x1919080808082B08, 0x1919080808190819, 0x1919080808191908,
    0x191908080819192B, 0x1919080808192B19, 0x19190808082B0808, 0x19190808082B082B,
    0x19190808082B1919, 0x19190808082B2B08, 0x1919080819080819, 0x1919080819081908,
    0x191908081908192B, 0x1919080819082B19, 0x1919080819190808, 0x191908081919082B,
    0x1919080819191919, 0x1919080819192B08, 0x19190808192B0819, 0x19190808192B1908,
    0x191908082B080808, 0x191908082B08082B, 0x191908082B081919, 0x191908082B082B08,
    0x191908082B190819, 0x191908082B191908, 0x1919081908080819, 0x1919081908081908,
    0x191908190808192B, 0x1919081908082B19, 0x1919081908190808, 0x191908190819082B,
    0x1919081908191919, 0x1919081908192B08, 0x19190819082B0819, 0x19190819082B1908,
    0x1919081919080808, 0x191908191908082B, 0x1919081919081919, 0x1919081919082B08,
    0x1919081919190819, 0x1919081919191908, 0x19190819192B0808, 0x191908192B080819,
    0x191908192B081908, 0x191908192B190808, 0x1919082B08080808, 0x1919082B08081919,
    0x1919082B08082B08, 0x1919082B08190819, 0x1919082B08191908, 0x1919082B082B0808,
    0x1919082B19080819, 0x1919082B19081908, 0x1919082B19190808, 0x1919082B192B2B19,
    0x1919082B2B080808, 0x1919190808080819, 0x1919190808081908, 0x191919080808192B,
    0x1919190808082B19, 0x1919190808190808, 0x191919080819082B, 0x1919190808191919,
    0x1919190808192B08, 0x19191908082B0819, 0x19191908082B1908, 0x1919190819080808,
    0x191919081908082B, 0x1919190819081919, 0x1919190819082B08, 0x1919190819190819,
    0x1919190819191908, 0x19191908192B0808, 0x191919082B080819, 0x191919082B081908,
    0x191919082B190808, 0x1919191908080808, 0x191919190808082B, 0x1919191908081919,
    0x1919191908082B08, 0x1919191908190819, 0x1919191908191908, 0x19191919082B0808,
    0x1919191919080819, 0x1919191919081908, 0x1919191919190808, 0x191919192B080808,
    0x1919192B08080819, 0x1919192B08081908, 0x1919192B08190808, 0x1919192B082B192B,
    0x1919192B19080808, 0x19192B0808080808, 0x19192B080808082B, 0x19192B0808081919,
    0x19192B0808082B08, 0x19192B0808190819, 0x19192B0808191908, 0x19192B08082B0808,
    0x19192B0819080819, 0x19192B0819081908, 0x19192B0819190808, 0x19192B0819192B2B,
    0x19192B082B080808, 0x19192B1908080819, 0x19192B1908081908, 0x19192B1908190808,
    0x19192B1919080808, 0x19192B2B08080808, 0x19192B2B08192B19, 0x19192B2B2B081919,
    0x19192B2B2B2B2B08, 0x192B080808080819, 0x192B080808081908, 0x192B08080808192B,
    0x192B080808190808, 0x192B08080819082B, 0x192B080808191919, 0x192B080808192B08,
    0x192B0808082B0819, 0x192B0808082B1908, 0x192B080819080808, 0x192B080819081919,
    0x192B080819082B08, 0x192B080819190819, 0x192B080819191908, 0x192B0808192B0808,
    0x192B08082B081908, 0x192B08082B190808, 0x192B081908080808, 0x192B08190808082B,
    0x192B081908081919, 0x192B081908082B08, 0x192B081908190819, 0x192B081908191908,
    0x192B0819082B0808, 0x192B081919080819, 0x192B081919081908, 0x192B081919190808,
    0x192B08192B080808, 0x192B08192B192B19, 0x192B082B08081908, 0x192B082B08190808,
    0x192B082B19080808, 0x192B082B1919192B, 0x192B082B2B2B0819, 0x192B190808080808,
    0x192B190808081919, 0x192B190808082B08, 0x192B190808190819, 0x192B190808191908,
    0x192B1908082B0808, 0x192B190819080819, 0x192B190819081908, 0x192B190819190808,
    0x192B19082B080808, 0x192B191908080819, 0x192B191908081908, 0x192B191908190808,
    0x192B191919080808, 0x192B191919082B2B, 0x192B1919192B2B08, 0x192B19192B19082B,
    0x192B192B08080808, 0x192B192B2B191908, 0x192B2B0808080819, 0x192B2B0808081908,
    0x192B2B0808190808, 0x192B2B08192B1919, 0x192B2B082B192B08, 0x192B2B1908080808,
    0x192B2B19082B2B2B, 0x192B2B2B1908082B, 0x192B2B2B2B2B0819, 0x2B08080808080808,
    0x2B0808080808082B, 0x2B08080808081919, 0x2B08080808082B08, 0x2B08080808190819,
    0x2B08080808191908, 0x2B08080808192B19, 0x2B080808082B0808, 0x2B080808082B1919,
    0x2B08080819080819, 0x2B08080819081908, 0x2B08080819190808, 0x2B0808081919082B,
    0x2B08080819191919, 0x2B08080819192B08, 0x2B080808192B0819, 0x2B0808082B080808,
    0x2B0808082B081919, 0x2B0808082B190819, 0x2B0808082B191908, 0x2B08081908080819,
    0x2B08081908081908, 0x2B08081908082B19, 0x2B08081908190808, 0x2B0808190819082B,
    0x2B08081908191919, 0x2B08081908192B08, 0x2B080819082B0819, 0x2B080819082B1908,
    0x2B08081919080808, 0x2B0808191908082B, 0x2B08081919081919, 0x2B08081919082B08,
    0x2B08081919190819, 0x2B08081919191908, 0x2B0808192B080819, 0x2B0808192B081908,
    0x2B0808192B190808, 0x2B0808192B2B2B19, 0x2B08082B08080808, 0x2B08082B08081919,
    0x2B08082B08082B2B, 0x2B08082B08190819, 0x2B08082B08191908, 0x2B08082B19080819,
    0x2B08082B19081908, 0x2B08082B19190808, 0x2B08190808080819, 0x2B08190808081908,
    0x2B0819080808192B, 0x2B08190808082B19, 0x2B08190808190808, 0x2B0819080819082B,
    0x2B08190808191919, 0x2B08190808192B08, 0x2B081908082B0819, 0x2B08190819080808,
    0x2B0819081908082B, 0x2B08190819081919, 0x2B08190819082B08, 0x2B08190819190819,
    0x2B08190819191908, 0x2B081908192B0808, 0x2B0819082B080819, 0x2B0819082B081908,
    0x2B0819082B190808, 0x2B08191908080808, 0x2B0819190808082B, 0x2B08191908081919,
    0x2B08191908082B08, 0x2B08191908190819, 0x2B08191908191908, 0x2B081919082B0808,
    0x2B08191919080819, 0x2B08191919081908, 0x2B08191919190808, 0x2B0819192B080808,
    0x2B0819192B082B2B, 0x2B08192B08080819, 0x2B08192B08081908, 0x2B08192B08190808,
    0x2B08192B082B2B19, 0x2B08192B19080808, 0x2B082B0808080808, 0x2B082B0808081919,
    0x2B082B0808190819, 0x2B082B0808191908, 0x2B082B0819080819, 0x2B082B0819081908,
    0x2B082B0819190808, 0x2B082B082B2B082B, 0x2B082B1908080819, 0x2B082B1908081908,
    0x2B082B1919080808, 0x2B082B19192B1919, 0x2B082B2B082B082B, 0x2B082B2B19192B08,
    0x2B082B2B19192B2B, 0x2B082B2B2B08082B, 0x2B082B2B2B2B082B, 0x2B19080808080819,
    0x2B19080808081908, 0x2B19080808082B19, 0x2B19080808190808, 0x2B1908080819082B,
    0x2B19080808191919, 0x2B19080808192B08, 0x2B190808082B1908, 0x2B19080819080808,
    0x2B1908081908082B, 0x2B19080819081919, 0x2B19080819082B08, 0x2B19080819190819,
    0x2B19080819191908, 0x2B190808192B0808, 0x2B1908082B080819, 0x2B1908082B081908,
    0x2B1908082B190808, 0x2B19081908080808, 0x2B19081908081919, 0x2B19081908190819,
    0x2B19081908191908, 0x2B19081919080819, 0x2B19081919081908, 0x2B19081919190808,
    0x2B19081919192B2B, 0x2B19082B08080819, 0x2B19082B08081908, 0x2B19082B08190808,
    0x2B19082B19080808, 0x2B19082B2B2B192B, 0x2B19190808080808, 0x2B1919080808082B,
    0x2B19190808081919, 0x2B19190808082B08, 0x2B19190808190819, 0x2B19190808191908,
    0x2B191908082B0808, 0x2B19190819080819, 0x2B19190819081908, 0x2B19190819190808,
    0x2B1919082B080808, 0x2B1919082B19192B, 0x2B19191908080819, 0x2B19191908081908,
    0x2B19191908190808, 0x2B19191919080808, 0x2B1919192B192B08, 0x2B1919192B2B0819,
    0x2B19192B08080808, 0x2B19192B1908192B, 0x2B19192B192B1908, 0x2B192B0808080819,
    0x2B192B0808081908, 0x2B192B0808190808, 0x2B192B08082B192B, 0x2B192B0819080808,
    0x2B192B082B2B2B19, 0x2B192B1908080808, 0x2B192B1919082B19, 0x2B192B191919082B,
    0x2B192B2B2B190808, 0x2B2B080808080808, 0x2B2B080808081919, 0x2B2B080808082B2B,
    0x2B2B080808191908, 0x2B2B0808082B082B, 0x2B2B0808082B2B2B, 0x2B2B080819080819,
    0x2B2B080819081908, 0x2B2B080819190808, 0x2B2B08082B2B082B, 0x2B2B08082B2B2B2B,
    0x2B2B081919080808, 0x2B2B0819192B1919, 0x2B2B082B0808082B, 0x2B2B082B08082B2B,
    0x2B2B082B082B082B, 0x2B2B082B082B2B08, 0x2B2B082B082B2B2B, 0x2B2B082B2B08082B,
    0x2B2B082B2B082B08, 0x2B2B082B2B082B2B, 0x2B2B082B2B2B2B08, 0x2B2B190808080819,
    0x2B2B190808081908, 0x2B2B190808190808, 0x2B2B190819080808, 0x2B2B19082B082B19,
    0x2B2B19082B2B1908, 0x2B2B191908080808, 0x2B2B191908192B19, 0x2B2B192B19190819,
    0x2B2B2B0808082B2B, 0x2B2B2B08082B2B08, 0x2B2B2B082B2B082B, 0x2B2B2B1919191908,
    0x2B2B2B192B08192B, 0x2B2B2B2B08082B08, 0x2B2B2B2B08082B2B, 0x2B2B2B2B082B0808,
    0x2B2B2B2B082B082B, 0x2B2B2B2B082B2B08, 0x2B2B2B2B2B082B08, 0x2B2B2B2B2B2B2B2B,
)


# ---------------------------------------------------------------------------
# reference model


def grid_bytes() -> bytes:
    """The table as ggml stores it: 1024 uint64 little endian, 8192 bytes."""
    return b"".join(v.to_bytes(8, "little") for v in IQ2S_GRID)


def grid_sha256() -> str:
    return hashlib.sha256(grid_bytes()).hexdigest()


def check_grid() -> tuple[int, list[str]]:
    checks, bad = 3, []
    if len(IQ2S_GRID) != 1024:
        bad.append(f"embedded grid has {len(IQ2S_GRID)} entries, expected 1024")
    got = grid_sha256()
    if got != GRID_SHA256:
        bad.append(f"embedded grid sha256 {got} != {GRID_SHA256}")
    mags = sorted({(v >> (8 * j)) & 0xFF for v in IQ2S_GRID for j in range(8)})
    if tuple(mags) != GRID_MAGNITUDES:
        bad.append(f"grid magnitudes {mags} != {list(GRID_MAGNITUDES)}")
    return checks, bad


def fp16_to_fraction(raw: int) -> Fraction:
    """Exact value of a finite fp16: +/- 2**-24 * frac (subnormal) or mant*2**(e-25)."""
    sign = -1 if raw >> 15 else 1
    exp = (raw >> 10) & 0x1F
    frac = raw & 0x3FF
    if exp == 0:
        return Fraction(sign * frac, 1 << 24)
    return Fraction(sign * (1024 + frac) * (1 << (exp - 1)), 1 << 24)


def int8(value: int) -> int:
    return value - 256 if value >= 128 else value


def grid_index(blk: bytes, g: int, l: int) -> int:
    """The 10-bit grid row of sub-block l of group g (0..3, 0..7)."""
    qs = blk[2 + 4 * g + l]
    qh = blk[66 + g]
    return qs | ((qh << (8 - 2 * l)) & 0x300)


def reference_group(blk: bytes, act: bytes, g: int) -> tuple[int, list[int], int, int]:
    """The group's int32 partial, grid indices, ls1 and ls2, straight from the sources."""
    scale = blk[74 + g]
    ls1 = 1 + 2 * (scale & 0x0F)
    ls2 = 1 + 2 * ((scale >> 4) & 0x0F)
    idx = [grid_index(blk, g, l) for l in range(4)]
    sumi1 = 0
    sumi2 = 0
    for l in range(4):
        row = IQ2S_GRID[idx[l]]
        signs = blk[34 + 4 * g + l]
        acc = 0
        for j in range(8):
            w = (row >> (8 * j)) & 0xFF
            a = int8(act[32 * g + 8 * l + j])
            if (signs >> j) & 1:
                w = -w
            acc += w * a
        if l < 2:
            sumi1 += acc
        else:
            sumi2 += acc
    return ls1 * sumi1 + ls2 * sumi2, idx, ls1, ls2


def dequant_group_dot(blk: bytes, act: bytes, g: int) -> tuple[Fraction, int]:
    """The group's dot as the float dequantiser defines it, in exact rationals.

    Returns (dense dot with the row scale folded in, partial as int24) where the
    dense dot should equal d * bsum / 8 for that group's bsum.
    """
    d = fp16_to_fraction(int.from_bytes(blk[0:2], "little"))
    scale = blk[74 + g]
    dense = Fraction(0)
    for l in range(4):
        row = IQ2S_GRID[grid_index(blk, g, l)]
        nib = (scale & 0x0F) if l < 2 else ((scale >> 4) & 0x0F)
        dl = Fraction(2 * nib + 1, 8)
        signs = blk[34 + 4 * g + l]
        for j in range(8):
            w = (row >> (8 * j)) & 0xFF
            if (signs >> j) & 1:
                w = -w
            dense += d * dl * w * int8(act[32 * g + 8 * l + j])
    partial, _, _, _ = reference_group(blk, act, g)
    return dense, partial


# ---------------------------------------------------------------------------
# seeded cases


def gen_case(rng: random.Random, kind: int) -> tuple[bytes, bytes]:
    """One seeded (block, activations) pair; kind picks the interesting corner."""
    blk = bytearray(BLOCK_BYTES)
    act = bytearray(QK_K)

    # a finite fp16 scale: exponent 0..30, any sign and mantissa
    raw = (rng.getrandbits(1) << 15) | (rng.randrange(31) << 10) | rng.getrandbits(10)
    blk[0] = raw & 0xFF
    blk[1] = raw >> 8

    if kind == 0:  # unconstrained: every field random
        for i in range(2, BLOCK_BYTES):
            blk[i] = rng.getrandbits(8)
        for i in range(QK_K):
            act[i] = rng.getrandbits(8)
    elif kind == 1:  # zero weights: every partial must be zero
        for i in range(QK_K):
            act[i] = rng.getrandbits(8)
    elif kind == 2:  # zero activations: every partial must be zero
        for i in range(2, BLOCK_BYTES):
            blk[i] = rng.getrandbits(8)
    elif kind == 3:  # saturated scales and sign bytes, extreme activations
        for i in range(2, 66):
            blk[i] = rng.getrandbits(8)
        for g in range(N_GROUPS):
            blk[66 + g] = 0xFF          # every grid index in 768..1023
            blk[74 + g] = 0xFF          # ls1 = ls2 = 31, all signs negative
        for i in range(QK_K):
            act[i] = 0x80 if rng.getrandbits(1) else 0x7F
    elif kind == 4:  # one qh bit pair set per group, random everything else
        for i in range(2, 66):
            blk[i] = rng.getrandbits(8)
        for g in range(N_GROUPS):
            blk[66 + g] = (0x03 << (2 * (g % 4))) & 0xFF
            blk[74 + g] = rng.getrandbits(8)
        for i in range(QK_K):
            act[i] = rng.getrandbits(8)
    else:  # kind 5: sparse blocks and sparse activations
        for g in range(N_GROUPS):
            blk[74 + g] = rng.choice((0x00, 0x0F, 0xF0, 0xFF, rng.getrandbits(8)))
        for g in range(N_GROUPS):
            for l in range(4):
                s = 4 * g + l
                blk[2 + s] = rng.getrandbits(8) if s % 5 == 0 else 0
                blk[34 + s] = rng.getrandbits(8) if s % 3 == 0 else 0
        for _ in range(32):
            act[rng.randrange(QK_K)] = rng.getrandbits(8)

    return bytes(blk), bytes(act)


N_KINDS = 6


def write_vectors(path: Path, cases: list[tuple[bytes, bytes]]) -> None:
    """B lines: the block padded to 88 bytes as 22 little-endian words.

    A lines: the 256 activations as 64 little-endian words.
    """
    out = []
    for blk, act in cases:
        padded = blk + bytes(88 - len(blk))
        out.append("B " + " ".join(
            f"{int.from_bytes(padded[4 * k:4 * k + 4], 'little'):08X}" for k in range(22)))
        out.append("A " + " ".join(
            f"{int.from_bytes(act[4 * k:4 * k + 4], 'little'):08X}" for k in range(64)))
    path.write_text("\n".join(out) + "\n", encoding="ascii")


# ---------------------------------------------------------------------------
# static checks: the RTL text and the testbench wiring


def strip_sv_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    text = re.sub(r'"[^"\n]*"', '""', text)
    return re.sub(r"//[^\n]*", " ", text)


def static_checks() -> tuple[int, list[str]]:
    checks, bad = 0, []
    for path in (RTL, TB):
        checks += 1
        if not path.is_file():
            bad.append(f"missing {path.relative_to(ROOT)}")
    if bad:
        return checks, bad
    rtl = RTL.read_text(encoding="utf-8")
    tb = TB.read_text(encoding="utf-8")

    for text, needle, label in (
        (rtl, "module otpu_iq2s", "RTL declares module otpu_iq2s"),
        (tb, "module tb_iq", "testbench declares module tb_iq"),
    ):
        checks += 1
        if needle not in text:
            bad.append(f"{label}: not found")

    # every port declared with the intended width, and wired by name in the testbench
    for name, width in PORTS:
        checks += 1
        pat = rf"^\s*(?:input|output)\s+logic\s*(?:\[\s*(\d+)\s*:\s*0\s*\])?\s*{name}\b"
        m = re.search(pat, rtl, re.M)
        if not m:
            bad.append(f"RTL port {name} is not declared as input/output logic")
        else:
            want = None if width == 1 else str(width - 1)
            if m.group(1) != want:
                bad.append(f"RTL port {name} declared [{m.group(1)}:0], expected width {width}")
        checks += 1
        if not re.search(rf"\.{name}\s*\(", tb):
            bad.append(f"testbench does not connect .{name}")

    # the ROM is a copy of the pinned table, not a second source
    rom = {int(i): int(v, 16) for i, v in ROM_LINE.findall(rtl)}
    checks += 1
    if sorted(rom) != list(range(1024)):
        bad.append(f"RTL ROM has {len(rom)} entries, expected 1024 (0..1023)")
    for i in range(1024):
        checks += 1
        if i in rom and rom[i] != IQ2S_GRID[i]:
            bad.append(f"RTL rom[{i}] = 64'h{rom[i]:016x}, table has 64'h{IQ2S_GRID[i]:016x}")

    # no vendor cells or simulation-only constructs in the synthesizable module
    code = strip_sv_comments(rtl)
    checks += 1
    m = XILINX_PRIMITIVE.search(code)
    if m:
        bad.append(f"RTL references vendor primitive {m.group(1)}")
    checks += 1
    m = SIM_SYSTEM_TASK.search(code)
    if m:
        bad.append(f"RTL uses simulation system task {m.group(1)}")
    checks += 1
    if re.search(r"#\s*\d", code):
        bad.append("RTL contains a delay control")

    return checks, bad


# ---------------------------------------------------------------------------
# selftest: the integer path against the float dequantiser, in exact rationals


def selftest(cases: int, seed: int) -> tuple[int, list[str]]:
    checks, bad = 12, []
    # fp16 decoding agrees with struct.unpack for finite values, subnormals included
    for raw in (0x0000, 0x8000, 0x0001, 0x03FF, 0x0400, 0x3555, 0x7BFF, 0xFBFF, 0x0555):
        checks += 1
        want = Fraction(struct.unpack("<e", raw.to_bytes(2, "little"))[0])
        if fp16_to_fraction(raw) != want:
            bad.append(f"fp16 0x{raw:04x}: {fp16_to_fraction(raw)} != {want}")

    rng = random.Random(seed)
    n_blocks = 0
    for i in range(cases):
        blk, act = gen_case(rng, i % N_KINDS)
        n_blocks += 1
        for g in range(N_GROUPS):
            dense, partial = dequant_group_dot(blk, act, g)
            bsum, idx, ls1, ls2 = reference_group(blk, act, g)
            checks += 1
            d = fp16_to_fraction(int.from_bytes(blk[0:2], "little"))
            want = Fraction(bsum) * d / 8  # dense carries (2n + 1) / 8 per sub-block
            if dense != want:
                bad.append(
                    f"case {i} group {g}: dequant dot {dense} != d*bsum/8 {want} "
                    f"(bsum {bsum}, ls1 {ls1}, ls2 {ls2}, idx {idx})"
                )
            checks += 1
            if not PARTIAL_MIN <= partial <= PARTIAL_MAX:
                bad.append(f"case {i} group {g}: partial {partial} outside int24")
    return checks, bad


# ---------------------------------------------------------------------------
# simulator: build, generate, run, parse, compare


def parse_dump(text: str) -> list[dict]:
    cases: list[dict] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        m = CASE_LINE.match(line)
        if m:
            cases.append({
                "ci": int(m.group(1)),
                "d": int(m.group(2), 16),
                "scales": int(m.group(3), 16),
                "groups": [],
            })
            continue
        m = GROUP_LINE.match(line)
        if m:
            if not cases:
                raise ValueError(f"group line before any case line: {line!r}")
            cases[-1]["groups"].append({
                "g": int(m.group(1)),
                "bsum": int(m.group(2), 16),
                "ls1": int(m.group(3), 16),
                "ls2": int(m.group(4), 16),
                "signs": int(m.group(5), 16),
                "idx": int(m.group(6), 16),
            })
            continue
        raise ValueError(f"unexpected dump line: {line!r}")
    return cases


def compare(cases: list[tuple[bytes, bytes]], dump: list[dict]) -> tuple[int, list[str]]:
    checks, bad = 0, []
    checks += 1
    if len(dump) != len(cases):
        bad.append(f"dump has {len(dump)} cases, ran {len(cases)}")

    for ci, (blk, act) in enumerate(cases):
        if ci >= len(dump):
            break
        got = dump[ci]
        checks += 1
        if got["ci"] != ci:
            bad.append(f"case {ci}: dump says case {got['ci']}")
        checks += 1
        if got["d"] != int.from_bytes(blk[0:2], "little"):
            bad.append(f"case {ci}: d {got['d']:04x} != block {int.from_bytes(blk[0:2], 'little'):04x}")
        checks += 1
        if got["scales"] != int.from_bytes(blk[74:82], "little"):
            bad.append(
                f"case {ci}: scales {got['scales']:016x} != block "
                f"{int.from_bytes(blk[74:82], 'little'):016x}"
            )
        checks += 1
        if len(got["groups"]) != N_GROUPS:
            bad.append(f"case {ci}: dump has {len(got['groups'])} groups, expected {N_GROUPS}")

        for g in range(N_GROUPS):
            if g >= len(got["groups"]):
                break
            bg = got["groups"][g]
            bsum, idx, ls1, ls2 = reference_group(blk, act, g)
            checks += 1
            if bg["g"] != g:
                bad.append(f"case {ci}: group slot {g} says group {bg['g']}")
            checks += 1
            if bg["bsum"] != bsum & 0xFFFFFFFF:
                bad.append(
                    f"case {ci} group {g}: partial {bg['bsum'] & 0xFFFFFFFF:08x} != "
                    f"reference {bsum} (idx {idx}, ls1 {ls1}, ls2 {ls2})"
                )
            checks += 1
            if bg["ls1"] != ls1 or bg["ls2"] != ls2:
                bad.append(
                    f"case {ci} group {g}: ls ({bg['ls1']}, {bg['ls2']}) != ({ls1}, {ls2})"
                )
            checks += 1
            want_signs = int.from_bytes(blk[34 + 4 * g:38 + 4 * g], "little")
            if bg["signs"] != want_signs:
                bad.append(
                    f"case {ci} group {g}: signs {bg['signs']:08x} != block {want_signs:08x}"
                )
            checks += 1
            want_idx = sum(idx[l] << (10 * l) for l in range(4))
            if bg["idx"] != want_idx:
                bad.append(
                    f"case {ci} group {g}: index {bg['idx']:010x} != reference {want_idx:010x}"
                )
            checks += 1
            if not PARTIAL_MIN <= bsum <= PARTIAL_MAX:
                bad.append(f"case {ci} group {g}: reference partial {bsum} outside int24")

    return checks, bad


def report(ok: bool, cases: int, checks: int, bad: list[str], **extra) -> int:
    result = {
        "ok": ok,
        "cases": cases,
        "checks": checks,
        "mismatches": bad[:40],
        **extra,
    }
    if len(bad) > 40:
        result["mismatch_count"] = len(bad)
    print(json.dumps(result, indent=2))
    return 0 if ok else 1


def fail(status: str, message: str, checks: int = 0, **extra) -> int:
    print(json.dumps({
        "ok": False,
        "cases": 0,
        "checks": checks,
        "mismatches": [message],
        "status": status,
        **extra,
    }, indent=2), flush=True)
    return 2


def find_verilator(name: str) -> str | None:
    found = shutil.which(name)
    if found:
        return found
    path = Path(name)
    return str(path) if path.is_file() else None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Build and run the IQ2_S dot primitive testbench, compare with the reference model."
    )
    ap.add_argument("--cases", type=int, default=N_CASES, help=f"cases to generate (default {N_CASES})")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED, help=f"case seed (default {DEFAULT_SEED})")
    ap.add_argument("--build", type=Path, default=DEFAULT_BUILD, help="build directory (default build/iq_rtl)")
    ap.add_argument("--verilator", default="verilator", help="verilator executable (default: PATH)")
    ap.add_argument("--selftest", action="store_true", help="check table, RTL text and model without a simulator")
    args = ap.parse_args(argv)

    if args.cases < 1:
        return fail("args", "--cases must be at least 1")

    grid_checks, bad = check_grid()
    static_checked, static_bad = static_checks()
    checks = grid_checks + static_checked
    bad += static_bad

    if args.selftest:
        st_checks, st_bad = selftest(args.cases, args.seed)
        checks += st_checks
        bad += st_bad
        return report(not bad, args.cases, checks, bad, selftest=True,
                      seed=args.seed, grid_sha256=grid_sha256())

    verilator = find_verilator(args.verilator)
    if verilator is None:
        return fail("no_verilator", f"{args.verilator} not found; install Verilator 5.x "
                    f"or run --selftest", checks, seed=args.seed)
    try:
        proc = subprocess.run([verilator, "--version"], capture_output=True, text=True,
                              timeout=120, check=False)
        version = (proc.stdout.strip().splitlines() or ["unknown"])[0]
    except (OSError, subprocess.SubprocessError):
        version = "unknown"

    rng = random.Random(args.seed)
    cases = [gen_case(rng, i % N_KINDS) for i in range(args.cases)]
    mdir = Path(args.build)
    if not mdir.is_absolute():
        mdir = ROOT / mdir
    mdir.mkdir(parents=True, exist_ok=True)
    vectors = mdir / "vectors.hex"
    dump_path = mdir / "dump.txt"
    write_vectors(vectors, cases)

    cmd = [verilator, *VERILATOR_FLAGS, "--Mdir", str(mdir), str(TB), str(RTL)]
    print(f"building {mdir / 'Vtb_iq'}", file=sys.stderr, flush=True)
    try:
        proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, timeout=1800)
    except (OSError, subprocess.SubprocessError) as exc:
        return fail("build", f"verilator failed to run: {exc}", checks, seed=args.seed)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout[-4000:])
        sys.stderr.write(proc.stderr[-8000:])
        return fail("build", f"verilator exited {proc.returncode}", checks, seed=args.seed)

    exe = mdir / "Vtb_iq"
    if not exe.is_file():
        return fail("build", f"verilator produced no {exe}", checks, seed=args.seed)

    cmd = [str(exe), f"+vectors={vectors}", f"+out={dump_path}", f"+cases={len(cases)}"]
    try:
        proc = subprocess.run(cmd, cwd=str(mdir), capture_output=True, text=True, timeout=1800)
    except (OSError, subprocess.SubprocessError) as exc:
        return fail("run", f"testbench failed to run: {exc}", checks, seed=args.seed)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout[-4000:])
        sys.stderr.write(proc.stderr[-8000:])
        return fail("run", f"testbench exited {proc.returncode}", checks, seed=args.seed)

    m = SUMMARY_LINE.search(proc.stdout)
    if not m:
        return fail("run", "testbench printed no IQ2STEST summary", checks, seed=args.seed)
    tb_cases, _, _, tb_errors = (int(m.group(i)) for i in range(1, 5))
    checks += 1
    if tb_errors:
        bad.append(f"testbench protocol errors: {tb_errors}")
    checks += 1
    if tb_cases != len(cases):
        bad.append(f"testbench ran {tb_cases} cases, generated {len(cases)}")

    try:
        dump = parse_dump(dump_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return fail("run", f"cannot parse {dump_path}: {exc}", checks, seed=args.seed)

    cmp_checks, cmp_bad = compare(cases, dump)
    checks += cmp_checks
    bad += cmp_bad

    return report(not bad, len(cases), checks, bad, seed=args.seed, verilator=version,
                  grid_sha256=grid_sha256(), build=str(mdir.relative_to(ROOT)),
                  vectors=str(vectors.relative_to(ROOT)), dump=str(dump_path.relative_to(ROOT)))


if __name__ == "__main__":
    sys.exit(main())