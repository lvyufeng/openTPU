#!/usr/bin/env python3
"""Check the multi-format IQ2_XXS / IQ2_S / IQ1_M stream primitive.

``rtl/vpu/otpu_iq_stream.sv`` demultiplexes a mixed-format packed weight stream
into the three scalar dot tiles ``otpu_iq2xxs.sv``, ``otpu_iq2s.sv`` and
``otpu_iq1m.sv``.  One load is one ggml super-block - 66 bytes for IQ2_XXS
(type 16), 82 bytes for IQ2_S (type 22) or 56 bytes for IQ1_M (type 29) -
streamed as 64-bit weight beats, followed by the 256 int8 activations as eight
256-bit beats.  Each load yields one result per 32-weight group: the integer
partial ``bsum``, the Q8_K auxiliary column ``aux``, the term count, the two odd
multipliers, the sign bytes, the 44-bit grid index bundle and the IQ1_M
``delta`` flag.  A block takes 17 (IQ2_XXS), 19 (IQ2_S) or 15 (IQ1_M) cycles.

This script generates seeded blocks of all three formats, runs the Verilator
testbench ``sim/verilator/tb_iq_stream.sv`` on them, and compares every dumped
field with an independent model written here straight from the ggml byte
layout: no shared RTL, no C compiler, no floating point in the integer path.
Partials, aux columns, indices, signs, scale bytes, ``d`` and the ``delta``
flags must agree exactly, and the testbench must check the cycle counts itself.

The codebooks are copied, not read out of the RTL: ``tools/iq_tables.py``
carries ``iq2xxs_grid``, ``iq1s_grid``, ``ksigns_iq2xs`` and ``kmask_iq2xs``
verbatim under their MIT notices (with the pinned commit recorded there) and
reuses the ``iq2s_grid`` copy already in ``tools/iq_rtl_check.py``.  No expected
value is ever derived from testbench output.

    python3 tools/iq_multi_rtl_check.py --selftest   # tables, RTL text, model
    python3 tools/iq_multi_rtl_check.py              # needs Verilator 5.x

Both print one JSON object:

    {"ok": true, "cases": 768, "checks": 43010, "mismatches": [], ...}

``--cases`` counts blocks per format, so the default 256 runs 768 loads.  Exit
status is 0 when ok, 1 on a model/testbench mismatch, 2 when the tools are
missing or the build or run failed.

Reference sources, ggml revision 3cf03257f219afbe7334045ff7c6a06ac68c627d, as
recorded by the Strata distribution at 3142b66b:

    ggml/src/ggml-common.h:1323+   block_iq2_xxs, block_iq2_s, block_iq1_m
    ggml/src/ggml-common.h:1957+   iq2xxs_grid, iq2s_grid, iq1s_grid,
                                   ksigns_iq2xs, kmask_iq2xs
    ggml/src/ggml-cpu/quants.c:906     ggml_vec_dot_iq2_xxs_q8_K_generic
    ggml/src/ggml-cpu/quants.c:1048    ggml_vec_dot_iq2_s_q8_K_generic
    ggml/src/ggml-cpu/quants.c:1193    ggml_vec_dot_iq1_m_q8_K_generic
    ggml/src/ggml-quants.c:2488        dequantize_row_iq2_xxs
    ggml/src/ggml-quants.c:2577        dequantize_row_iq2_s
    ggml/src/ggml-quants.c:2650        dequantize_row_iq1_m

This file is Apache-2.0 (openTPU) and copies no MIT code: the group unpacking
below is an independent transcription of the byte layout and of the integer
sub-dot equations.  The codebooks live in tools/iq_tables.py.
"""

from __future__ import annotations

import argparse
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
RTL_DIR = ROOT / "rtl" / "vpu"
STREAM_RTL = RTL_DIR / "otpu_iq_stream.sv"
CHILD_RTL = {
    0: RTL_DIR / "otpu_iq2xxs.sv",
    1: RTL_DIR / "otpu_iq2s.sv",
    2: RTL_DIR / "otpu_iq1m.sv",
}
TB = ROOT / "sim" / "verilator" / "tb_iq_stream.sv"
DEFAULT_BUILD = ROOT / "build" / "iq_stream"

sys.path.insert(0, str(Path(__file__).resolve().parent))
import iq_tables  # noqa: E402  (data only; its own validation runs at import)

QK_K = 256          # super-block size
N_GROUPS = 8        # QK_K / 32
N_CASES = 256       # per format; one run therefore holds 3 * N_CASES blocks
DEFAULT_SEED = 20261124
N_KINDS = 6
EXTREME = (0x80, 0x7F)

#: format id carried on w_fmt2 / grp_fmt2 / blk_fmt2; also the child index
IQ2_XXS, IQ2_S, IQ1_M = 0, 1, 2
FMT_IDS = (IQ2_XXS, IQ2_S, IQ1_M)

#: name, packed bytes, weight beats, cycles (weight beats + 8 activation beats)
FORMATS = (
    ("IQ2_XXS", IQ2_XXS, 66, 9, 17),
    ("IQ2_S", IQ2_S, 82, 11, 19),
    ("IQ1_M", IQ1_M, 56, 7, 15),
)
FMT_NAME = {fmt: name for name, fmt, *_ in FORMATS}
FMT_BYTES = {fmt: size for name, fmt, size, _, _ in FORMATS}
FMT_BEATS = {fmt: beats for name, fmt, _, beats, _ in FORMATS}
FMT_CYCLES = {fmt: cycles for name, fmt, _, _, cycles in FORMATS}
MAX_BLOCK_BYTES = max(FMT_BYTES.values())
PAD_BYTES = 88                  # a B line is the widest block, zero padded
PAD_WORDS = PAD_BYTES // 4
ACT_WORDS = QK_K // 4           # 64 little-endian words per A line

#: the IQ2 grids store magnitudes 8, 25 and 43
GRID_MAGNITUDES = iq_tables.IQ2_GRID_MAGNITUDES
#: the IQ1_M grid stores signed ternary lanes
GRID_TERNARY = iq_tables.IQ1S_GRID_VALUES

#: bound on |bsum|: IQ2_XXS 8*43*31*128 < 2^19; IQ2_S 2*16*43*31*128 < 2^23;
#: IQ1_M 2 sub-blocks * 8 lanes * 1 * 15 * 128 < 2^16 for both bsum and aux.
#: worst-case width of a group's integer fields, so the selftest can prove the
#: RTL's 32-bit ports never truncate.  bsum = ls1*dot1 + ls2*dot2 where a row is
#: 8 weights against int8 activations: IQ2 magnitudes top out at 43 with ls <= 31
#: (4 rows of 8), IQ1_M lanes are +-1 with ls <= 15.  aux is IQ1_M only:
#: sum over 32 rows of ls * asum with asum <= 8*128, scaled by 8.
PARTIAL_BITS = {IQ2_XXS: 24, IQ2_S: 24, IQ1_M: 17}
AUX_BITS = {IQ2_XXS: 1, IQ2_S: 23, IQ1_M: 20}
IQ1M_DELTA_BITS = 17

MODULE_NAME = {0: "otpu_iq2xxs", 1: "otpu_iq2s", 2: "otpu_iq1m"}
#: the Verilator --binary output is named after the testbench's top module
TB_BINARY = "Vtb_iq_stream"

#: copied from opentpu/rtlsim.py VFLAGS, plus --timing (the testbench uses delays)
VERILATOR_FLAGS = [
    "--binary",
    "--timing",
    "--top-module", "tb_iq_stream",
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

#: the stream port list and the width each port must declare.  The names are
#: the contract's canonical spellings, which carry the width as a suffix for the
#: wide signals; the RTL may instead declare the plain name, because the width
#: is already checked from this table (see ``PORT_ALIASES``).
PORTS = (
    ("clk", 1),
    ("rst", 1),
    ("w_valid", 1),
    ("w_ready", 1),
    ("w_data64", 64),
    ("w_fmt2", 2),
    ("a_valid", 1),
    ("a_ready", 1),
    ("a_data256", 256),
    ("grp_valid", 1),
    ("grp_idx3", 3),
    ("grp_fmt2", 2),
    ("grp_bsum32", 32),
    ("grp_aux32", 32),
    ("grp_terms2", 2),
    ("grp_ls1", 5),
    ("grp_ls2", 5),
    ("grp_signs32", 32),
    ("grp_index", 44),
    ("grp_delta", 4),
    ("blk_valid", 1),
    ("blk_fmt2", 2),
    ("blk_d16", 16),
    ("blk_scales64", 64),
)

#: accepted alternative spellings: the spec suffix carries the width, the RTL
#: declares the same signal without it.  Both resolve to the same width check.
PORT_ALIASES = {
    "w_data64": "w_data",
    "w_fmt2": "w_fmt",
    "a_data256": "a_data",
    "grp_idx3": "grp_idx",
    "grp_fmt2": "grp_fmt",
    "grp_bsum32": "grp_bsum",
    "grp_aux32": "grp_aux",
    "grp_terms2": "grp_terms",
    "grp_signs32": "grp_signs",
    "blk_fmt2": "blk_fmt",
    "blk_d16": "blk_d",
    "blk_scales64": "blk_scales",
}

#: every child tile shares this per-block interface
CHILD_PORTS = (
    ("clk", 1), ("rst", 1),
    ("w_valid", 1), ("w_ready", 1), ("w_data64", 64),
    ("a_valid", 1), ("a_ready", 1), ("a_data256", 256),
    ("grp_valid", 1), ("grp_idx3", 3), ("grp_bsum32", 32),
    ("grp_ls1", 5), ("grp_ls2", 5), ("grp_signs32", 32),
    ("blk_valid", 1), ("blk_d16", 16), ("blk_scales64", 64),
)

#: extra per-format group ports.  otpu_iq2s.sv is the frozen pre-existing tile:
#: it exports no aux/terms/delta and packs four 10-bit indices into 40 bits,
#: which the stream zero-extends.  otpu_iq2xxs.sv and otpu_iq1m.sv follow the
#: full common contract with a 44-bit index.
CHILD_EXTRA_PORTS = {
    IQ2_XXS: (("grp_aux32", 32), ("grp_terms2", 2), ("grp_index", 44), ("grp_delta", 4)),
    IQ2_S: (("grp_index", 40),),
    IQ1_M: (("grp_aux32", 32), ("grp_terms2", 2), ("grp_index", 44), ("grp_delta", 4)),
}

#: the codebooks are data, so each child embeds the ROMs it needs verbatim
ROMS = (
    ("iq2xxs_rom", 256, lambda i: iq_tables.IQ2XXS_GRID[i]),
    ("iq2s_rom", 1024, lambda i: iq_tables.IQ2S_GRID[i]),
    ("iq1s_rom", 2048, lambda i: iq_tables.IQ1S_GRID[i]),
    ("ksigns_rom", 128, lambda i: iq_tables.KSIGNS_IQ2XS[i]),
)
ROM_REQUIRED = {
    0: ("iq2xxs_rom", "ksigns_rom"),
    1: ("iq2s_rom",),
    2: ("iq1s_rom",),
}

ROM_LINE = re.compile(
    r"\b(iq2xxs_rom|iq2s_rom|iq1s_rom|ksigns_rom)\[\s*(\d+)\s*\]\s*="
    r"\s*(?:64|8)'h([0-9a-fA-F]+)"
)
CASE_LINE = re.compile(
    r"^case (\d+) fmt ([0-9a-fA-F]+) d ([0-9a-fA-F]+) scales ([0-9a-fA-F]+)$"
)
GROUP_LINE = re.compile(
    r"^g (\d+) ([0-9a-fA-F]+) ([0-9a-fA-F]+) ([0-9a-fA-F]+) ([0-9a-fA-F]+) "
    r"([0-9a-fA-F]+) ([0-9a-fA-F]+) ([0-9a-fA-F]+) ([0-9a-fA-F]+)$"
)
SUMMARY_LINE = re.compile(
    r"IQSTREAMTEST cases=(\d+) beats=(\d+) cycles=(\d+) errors=(\d+)"
)
XILINX_PRIMITIVE = re.compile(
    r"\b(BUFG|BUFH|BUFIO|BUFR|FDRE|FDSE|FDPE|LUT[1-6]|RAMB18|RAMB36|DSP48\w*|"
    r"IBUF|OBUF|MMCM\w*|PLLE?\d?\w*|STARTUPE\w*)\b"
)
SIM_SYSTEM_TASK = re.compile(
    r"\$(display|finish|fatal|stop|write|monitor|dumpfile|fopen|fclose|value\$plusargs)"
)


# ---------------------------------------------------------------------------
# byte access (little-endian throughout, as the ggml structs store them)


def _u16(buf: bytes, off: int) -> int:
    return int.from_bytes(buf[off:off + 2], "little")


def _u32(buf: bytes, off: int) -> int:
    return int.from_bytes(buf[off:off + 4], "little")


def _s8(value: int) -> int:
    return value - 256 if value >= 128 else value


def block_size(fmt: int) -> int:
    """Packed size in bytes of one super-block of ``fmt`` (66 / 82 / 56)."""
    if fmt not in FMT_BYTES:
        raise ValueError(f"unknown weight format id {fmt!r}")
    return FMT_BYTES[fmt]


# ---------------------------------------------------------------------------
# the reference model: packed block -> per-group integer fields
#
# Every format's per-group structure is four eight-weight rows (32 weights).
# A weight is d*(2n+1)/8*grid*sign for IQ2, or d*ls*(grid + delta/8) for IQ1_M;
# the integer outputs preserve their scale and delta terms separately.


def _q8_act(act: bytes) -> tuple[int, ...]:
    """The 256 int8 activations of a Q8_K block in weight order."""
    return tuple(_s8(b) for b in act)


def _subblock_asum(signed_act: tuple[int, ...], sub: int) -> int:
    """Signed sum of the 16 activations of sub-block ``sub``."""
    lo = 16 * sub
    return sum(signed_act[lo:lo + 16])


def iq2_xxs_terms(fmt: int, blk: bytes, act: bytes, g: int) -> dict:
    """IQ2_XXS group ``g``: four 8-value grid rows, one sign seed each.

    qs holds sixteen 4-byte little-endian words, two per group: word[2g] packs
    the four 8-bit grid indices of the group's four lanes (lane l in byte l)
    and word[2g+1] is the aux word, whose bits 7*l .. 7*l+6 seed lane l's sign
    byte and whose bits 28..31 hold the group's 4-bit scale.  Lane l owns the
    weights 8*l .. 8*l+7, and ``KSIGNS_IQ2XS[seed]`` gives their sign bits
    (bit j set means the magnitude is negated).
    """
    if fmt != IQ2_XXS:
        raise ValueError("iq2_xxs_terms: wrong format")
    signed_act = _q8_act(act)
    base = 2 + 8 * g
    aux = _u32(blk, base + 4)
    ls1 = 2 * (aux >> 28) + 1
    index = 0
    signs = 0
    dot = 0
    for lane in range(4):
        idx = blk[base + lane]
        index |= idx << (11 * lane)
        sign_bits = iq_tables.KSIGNS_IQ2XS[(aux >> (7 * lane)) & 0x7F]
        signs |= sign_bits << (8 * lane)
        mags = iq_tables.grid_bytes(iq_tables.IQ2XXS_GRID[idx])
        for j in range(8):
            value = -mags[j] if (sign_bits >> j) & 1 else mags[j]
            dot += value * signed_act[32 * g + 8 * lane + j]
    return {
        "bsum": ls1 * dot,
        "aux": 0,
        "terms": 1,
        "ls1": ls1,
        "ls2": 0,
        "signs": signs,
        "index": index,
        "delta": 0,
    }


def iq2_s_terms(fmt: int, blk: bytes, act: bytes, g: int) -> dict:
    """IQ2_S group ``g``: four ten-bit-indexed 8-weight rows and scale pairs.

    layout: d 0:2, qs 2:34 (low 8 index bits, one per 8-weight row), sg 34:66
    (sign bits, one byte per row), qh 66:74 (two high index bits per row),
    scales 74:82 (low nibble ls1, high nibble ls2, one byte per group).
    Row r owns the weights 8r .. 8r+7, its index is
    ``qs[r] | ((qh[r//4] >> 2*(r%4)) & 3) << 8`` and its sign bit j comes from
    ``sg[r] >> j``.  Group g is the two 16-weight sub-blocks 2g and 2g+1, so
    it covers rows 4g .. 4g+3: two rows on ls1 and two on ls2.
    """
    if fmt != IQ2_S:
        raise ValueError("iq2_s_terms: wrong format")
    signed_act = _q8_act(act)
    ls_byte = blk[74 + g]
    ls1 = 2 * (ls_byte & 0xF) + 1
    ls2 = 2 * (ls_byte >> 4) + 1
    dot1 = 0
    dot2 = 0
    signs = 0
    index = 0
    for row in range(4 * g, 4 * g + 4):
        idx = blk[2 + row] | (((blk[66 + row // 4] >> (2 * (row % 4))) & 0x3) << 8)
        sign_byte = blk[34 + row]
        mags = iq_tables.grid_bytes(iq_tables.IQ2S_GRID[idx])
        for j in range(8):
            value = -mags[j] if (sign_byte >> j) & 1 else mags[j]
            # rows 2g and 2g+1 use ls1, rows 2g+2 and 2g+3 use ls2
            if row - 4 * g < 2:
                dot1 += value * signed_act[8 * row + j]
            else:
                dot2 += value * signed_act[8 * row + j]
            signs |= ((sign_byte >> j) & 1) << (8 * (row % 4) + j)
        # The common stream namespace always uses four 11-bit lanes; the frozen
        # child's 10-bit lanes are expanded here, not shifted by concatenation.
        index |= idx << (11 * (row % 4))
    return {
        "bsum": ls1 * dot1 + ls2 * dot2,
        # The frozen golden reports each scaled half separately.  Keep the
        # combined native partial and expose the second one, not a leftover
        # after subtracting a rounded or saturated first half.
        "aux": ls2 * dot2,
        "terms": 2,
        "ls1": ls1,
        "ls2": ls2,
        "signs": signs,
        "index": index,
        "delta": 0,
    }


def iq1_m_scales(blk: bytes, g: int) -> tuple[int, int]:
    """The two 3-bit scale fields of group ``g``: ``6*(g%2) + 0`` and ``+3``.

    The eight scale words at 48:56 are four uint16 words; each group takes its
    two fields from word ``g // 2``, so groups 0/2/4/6 use bits 0..2 and 3..5
    and groups 1/3/5/7 use bits 6..8 and 9..11.
    """
    word = _u16(blk, 48 + 2 * (g // 2))
    lo = (word >> (6 * (g % 2) + 0)) & 0x7
    hi = (word >> (6 * (g % 2) + 3)) & 0x7
    return 2 * lo + 1, 2 * hi + 1


def iq1_m_scale_bits(blk: bytes) -> int:
    """The fp16 scale of an IQ1_M block, reassembled from the scale nibbles.

    ``block_iq1_m`` has no contiguous ``d`` field: the four high nibbles of
    words 0..3 at 48, 50, 52, 54 carry the fp16 quarters, word 0 in bits 0..3
    (the nibbles of bytes 49, 51, 53 and 55).
    """
    return (
        (blk[49] >> 4)
        | ((blk[51] >> 4) << 4)
        | ((blk[53] >> 4) << 8)
        | ((blk[55] >> 4) << 12)
    )


def iq1_m_terms(fmt: int, blk: bytes, act: bytes, g: int) -> dict:
    """IQ1_M group ``g``: 32 eleven-bit ternary indices, two 3-bit scales.

    layout: qs 0:32 (low 8 index bits, one per 8-weight row), qh 32:48 (one
    nibble per row: bits 0..2 are the index's high bits, bit 3 is the delta
    sign), scales 48:56.  Row r owns weights 8r .. 8r+7, its index is
    ``qs[r] | ((qh[r//2] >> 4*(r%2)) & 7) << 8`` into ``iq1s_grid``, whose
    lanes are signed ternary values.  The delta bit picks the row offset: a
    clear bit gives +1/8 and a set bit gives -1/8, so every row contributes
    ``+-ls/8 * asum`` next to the grid dot.  ``aux`` carries that offset term,
    scaled by 8 so it stays integral: the group's dense dot is exactly
    ``d * (bsum + aux/8)``.  ``delta`` is the per-row delta flag, bit r.
    """
    if fmt != IQ1_M:
        raise ValueError("iq1_m_terms: wrong format")
    signed_act = _q8_act(act)
    ls1, ls2 = iq1_m_scales(blk, g)
    dot1 = 0
    dot2 = 0
    index = 0
    delta = 0
    aux = 0
    for row in range(4 * g, 4 * g + 4):
        nibble = (blk[32 + row // 2] >> (4 * (row % 2))) & 0xF
        idx = blk[row] | ((nibble & 0x7) << 8)
        grid = iq_tables.grid_signed(iq_tables.IQ1S_GRID[idx])
        first = row - 4 * g < 2
        ls = ls1 if first else ls2
        for j in range(8):
            value = grid[j] * signed_act[8 * row + j]
            if first:
                dot1 += value
            else:
                dot2 += value
        index |= idx << (11 * (row % 4))
        # the delta bit selects -1/8; a clear bit selects +1/8, so every row
        # contributes +-ls/8 * asum on top of the grid dot term
        row_asum = sum(signed_act[8 * row:8 * row + 8])
        if nibble & 0x8:
            delta |= 1 << (row % 4)
            aux -= ls * row_asum
        else:
            aux += ls * row_asum
    return {
        "bsum": ls1 * dot1 + ls2 * dot2,
        "aux": aux,
        "terms": 2,
        "ls1": ls1,
        "ls2": ls2,
        "signs": 0,
        "index": index,
        "delta": delta,
    }



def terms(fmt: int, blk: bytes, act: bytes, g: int) -> dict:
    """Reference integer terms of group ``g``: the public helper of this model."""
    _validate(fmt, blk, act)
    if not 0 <= g < N_GROUPS:
        raise ValueError("group index must be 0..7")
    if fmt == IQ2_XXS:
        return iq2_xxs_terms(fmt, blk, act, g)
    if fmt == IQ2_S:
        return iq2_s_terms(fmt, blk, act, g)
    return iq1_m_terms(fmt, blk, act, g)


def _validate(fmt: int, blk: bytes, act: bytes) -> None:
    if fmt not in FMT_IDS:
        raise ValueError(f"unknown weight format id {fmt!r}")
    if len(blk) != block_size(fmt):
        raise ValueError(f"expected a {block_size(fmt)}-byte block, got {len(blk)}")
    if len(act) != QK_K:
        raise ValueError(f"expected {QK_K} activations, got {len(act)}")


def block_metadata(fmt: int, blk: bytes) -> dict:
    """The per-block side channel: raw ``d`` bits and the dumped scale payload.

    IQ2_XXS and IQ2_S carry ``d`` in the first two bytes; IQ1_M carries it in
    the top nibbles of the four scale words.  ``scales`` is what ``blk_scales``
    must show: the eight 4-bit/3-bit scale bytes of IQ2_S and IQ1_M verbatim,
    and for IQ2_XXS ``{4'b0, aux[31:28]}`` per group (its whole scale field).
    """
    _validate(fmt, blk, b"\x00" * QK_K)
    if fmt == IQ2_XXS:
        scales = 0
        for g in range(N_GROUPS):
            scales |= ((_u32(blk, 2 + 8 * g + 4) >> 28) & 0xF) << (8 * g)
        return {"d": _u16(blk, 0), "scales": scales}
    if fmt == IQ2_S:
        return {"d": _u16(blk, 0), "scales": int.from_bytes(blk[74:82], "little")}
    return {"d": iq1_m_scale_bits(blk), "scales": int.from_bytes(blk[48:56], "little")}


# ---------------------------------------------------------------------------
# exact fp16 arithmetic for the selftest (no float ever touches the identity)


class NonFiniteScale(ValueError):
    """An fp16 ``d`` that is inf or NaN, so the identity cannot be evaluated."""


def fp16_fraction(raw: int) -> Fraction:
    """Exact value of a finite fp16 bit pattern (subnormals included)."""
    sign = -1 if raw >> 15 else 1
    exp = (raw >> 10) & 0x1F
    frac = raw & 0x3FF
    if exp == 0:
        return Fraction(sign * frac, 1 << 24)
    if exp == 0x1F:
        raise NonFiniteScale(f"fp16 0x{raw:04x} is inf or NaN")
    return Fraction(sign * (1024 + frac) * (1 << (exp - 1)), 1 << 24)


def fp16_is_finite(raw: int) -> bool:
    return (raw >> 10) & 0x1F != 0x1F


def _d_bits(fmt: int, blk: bytes) -> int:
    """The raw fp16 ``d`` bits of a block, in the layout the RTL must expose."""
    if fmt == IQ2_XXS or fmt == IQ2_S:
        return _u16(blk, 0)
    return iq1_m_scale_bits(blk)


# ---------------------------------------------------------------------------
# seeded cases: at least six kinds per format, all reproducible from the seed


def _fp16_scalar(rng: random.Random) -> int:
    """A finite fp16 bit pattern with a normal exponent: any sign, no inf/NaN."""
    return (rng.getrandbits(1) << 15) | (rng.randrange(1, 31) << 10) | rng.getrandbits(10)


def gen_case(rng: random.Random, kind: int, fmt: int) -> tuple[bytes, bytes]:
    """One seeded ``(block, activations)`` pair for ``fmt``; kind picks the corner.

    Kinds are the same six for every format so a failure at one kind can be
    compared across formats::

        0  unconstrained: every packed byte and every activation random
        1  codebook extremes: grid index 0 (all-minimum magnitudes 8 for the
           IQ2 grids, and for IQ1_M the 0xff all-negative-lane entry) in every
           slot, with random scales, signs and activations
        2  zero activations: the block is random, every activation is zero, so
           every ``bsum`` (and IQ1_M ``aux``) must come out exactly 0
        3  saturated: extreme scales, sign bytes and activations
        4  targeted: one high index bit / scale field / sign bit forced
        5  sparse: mostly zero block bytes and mostly zero activations

    There is no "zero weights" kind: the IQ2 codebooks carry magnitudes 8, 25
    and 43 and never 0, so every IQ2 weight is non-zero whenever ``d`` is; the
    zero-dot corner is reached through zero activations (kind 2) instead.  The
    fp16 ``d`` is always kept finite: random kinds draw a normal exponent and
    the pinned kinds use ``d = 1.0``, so a denorm never masks a broken field.
    """
    size = FMT_BYTES[fmt]
    blk = bytearray(size)
    act = bytearray(QK_K)

    def random_act() -> None:
        for i in range(QK_K):
            act[i] = rng.getrandbits(8)

    def saturated_act() -> None:
        for i in range(QK_K):
            act[i] = EXTREME[rng.getrandbits(1)]

    def random_body() -> None:
        if fmt == IQ2_XXS:
            for i in range(2, size):
                blk[i] = rng.getrandbits(8)
        elif fmt == IQ2_S:
            for i in range(2, size):
                blk[i] = rng.getrandbits(8)
        else:
            for i in range(0, size):
                blk[i] = rng.getrandbits(8)

    if kind == 0:                      # unconstrained
        raw = _fp16_scalar(rng)
        if fmt == IQ1_M:
            random_body()
            # keep the generated scale nibbles, only the sign bit is normalised
            pass
        else:
            blk[0] = raw & 0xFF
            blk[1] = raw >> 8
            random_body()
        random_act()
    elif kind == 1:                    # codebook entry 0 everywhere
        if fmt == IQ2_XXS:
            blk[0] = 0x00
            blk[1] = 0x3C              # d = 1.0
            for g in range(N_GROUPS):
                for lane in range(4):
                    blk[2 + 8 * g + lane] = 0x00          # grid index 0
                blk[2 + 8 * g + 4: 2 + 8 * g + 8] = rng.getrandbits(32).to_bytes(4, "little")
        elif fmt == IQ2_S:
            blk[0] = 0x00
            blk[1] = 0x3C              # d = 1.0
            for r in range(32):
                blk[2 + r] = 0x00      # low index bits: grid row 0
            for g in range(N_GROUPS):
                blk[66 + g] = 0x00     # high index bits clear -> still row 0
                blk[74 + g] = rng.getrandbits(8)
            for r in range(32):
                blk[34 + r] = rng.getrandbits(8)
        else:                          # IQ1_M: index 0 and all deltas set
            for i in range(32):
                blk[i] = 0x00          # qs = 0, qh = 0 -> index 0 (all -1 lanes)
            for w in range(4):
                blk[48 + 2 * w] = rng.getrandbits(8) & 0x3F
                blk[49 + 2 * w] = rng.getrandbits(8)
        random_act()
    elif kind == 2:                    # zero activations
        raw = _fp16_scalar(rng)
        if fmt != IQ1_M:
            blk[0] = raw & 0xFF
            blk[1] = raw >> 8
        random_body()
    elif kind == 3:                    # saturated scales, signs and activations
        saturated_act()
        if fmt == IQ2_XXS:
            blk[0] = 0x00
            blk[1] = 0x3C
            for g in range(N_GROUPS):
                for byte in range(4):
                    blk[2 + 8 * g + byte] = rng.getrandbits(8)
                # aux magnitude 15 in the highest nibble -> ls1 = 31, with
                # all four seven-bit sign seeds = 0x7f (including bit 7).
                blk[2 + 8 * g + 4] = 0xFF
                blk[2 + 8 * g + 5] = 0xFF
                blk[2 + 8 * g + 6] = 0xFF
                blk[2 + 8 * g + 7] = 0xFF
        elif fmt == IQ2_S:
            blk[0] = 0x00
            blk[1] = 0x3C
            for i in range(2, 66):
                blk[i] = rng.getrandbits(8)
            for r in range(32):
                blk[34 + r] = 0xFF     # every weight negative
            for g in range(N_GROUPS):
                blk[66 + g] = 0xFF     # every high index bit set
                blk[74 + g] = 0xFF     # ls1 = ls2 = 31
        else:                          # IQ1_M: both three-bit scales -> ls = 15
            for i in range(0, 48):
                blk[i] = rng.getrandbits(8)
            for w in range(4):
                # low six bits = 0b111111: two adjacent three-bit scale
                # fields, both seven -> both odd multipliers are 15.
                blk[48 + 2 * w] = 0x3F
                blk[49 + 2 * w] = 0x3F
            blk[49] = 0x0F             # preserve scale bits 6..11 below
            blk[51] = 0xCF
            blk[53] = 0x3F
            blk[55] = 0x0F             # d = 0x3c00, with every scale ls = 15
    elif kind == 4:                    # targeted high index / scale / sign bits
        raw = _fp16_scalar(rng)
        if fmt == IQ2_XXS:
            blk[0] = raw & 0xFF
            blk[1] = raw >> 8
            for i in range(2, size):
                blk[i] = 0
            for g in range(N_GROUPS):
                for lane in range(8):
                    blk[2 + 8 * g + lane] = rng.getrandbits(8)
                # one selected lane's sign group set to the 0x7f all-negative code
                lane = g % 4
                aux = 0
                for l in range(4):
                    aux |= (0x7F if l == lane else rng.randrange(0x80)) << (7 * l)
                aux = (aux & 0x0FFFFFFF) | ((rng.randrange(1, 16)) << 28)
                blk[2 + 8 * g + 4: 2 + 8 * g + 8] = aux.to_bytes(4, "little")
        elif fmt == IQ2_S:
            blk[0] = raw & 0xFF
            blk[1] = raw >> 8
            for i in range(2, size):
                blk[i] = 0
            for r in range(32):
                blk[2 + r] = rng.getrandbits(8)
            for g in range(N_GROUPS):
                # one 2-bit high field per group, alternating which row it hits
                blk[66 + g] = (0x03 << (2 * (g % 4))) & 0xFF
                blk[74 + g] = rng.getrandbits(8)
            for r in range(32):
                if r % 5 == 0:
                    blk[34 + r] = 0xFF
        else:
            for i in range(0, size):
                blk[i] = 0
            for r in range(32):
                blk[r] = rng.getrandbits(8)
            for r in range(32):
                # alternate a clear and a set delta bit in every row pair
                nib = rng.getrandbits(4)
                nib = (nib & 0x7) | (0x8 if r % 3 == 0 else 0)
                blk[32 + r // 2] |= nib << (4 * (r % 2))
            for w in range(4):
                blk[48 + 2 * w] = 0x3F
                blk[49 + 2 * w] = 0x3F
        random_act()
    else:                              # kind 5: sparse blocks and sparse activations
        raw = _fp16_scalar(rng)
        if fmt != IQ1_M:
            blk[0] = raw & 0xFF
            blk[1] = raw >> 8
        for i in range(2 if fmt != IQ1_M else 0, size):
            blk[i] = rng.getrandbits(8) if rng.randrange(4) == 0 else 0
        for _ in range(32):
            act[rng.randrange(QK_K)] = rng.getrandbits(8)

    blk = bytes(blk)
    if not fp16_is_finite(_d_bits(fmt, blk)):
        # never hand the RTL a block whose scale is inf/NaN: patch to d = 1.0
        if fmt == IQ1_M:
            # d = 1.0 = 0x3C00 as the four nibbles 0x0, 0xC, 0x3, 0x0, leaving
            # the low nibble of each scale byte (the 3-bit scale fields) alone
            fix = bytearray(blk)
            for byte, nib in ((49, 0x0), (51, 0xC), (53, 0x3), (55, 0x0)):
                fix[byte] = (nib << 4) | (fix[byte] & 0x0F)
            blk = bytes(fix)
            if not fp16_is_finite(_d_bits(fmt, blk)):
                raise AssertionError("cannot pin a finite IQ1_M scale")
        else:
            fix = bytearray(blk)
            fix[0] = 0x00
            fix[1] = 0x3C
            blk = bytes(fix)
    return blk, bytes(act)


def generate_cases(cases: int, seed: int) -> list[tuple[int, bytes, bytes]]:
    """``cases`` blocks of every format, round-robin over the kinds.

    The list order is the ``ci`` the testbench must echo: all IQ2_XXS blocks
    (kind 0, 1, 2, ...), then all IQ2_S, then all IQ1_M, so a mismatch can be
    read off the format and the kind directly.
    """
    rng = random.Random(seed)
    out: list[tuple[int, bytes, bytes]] = []
    for fmt in FMT_IDS:
        for i in range(cases):
            blk, act = gen_case(rng, i % N_KINDS, fmt)
            out.append((fmt, blk, act))
    return out


def case_kind(ci: int, cases: int) -> int:
    """The kind a case index was generated with (0..N_KINDS-1)."""
    return (ci % cases) % N_KINDS


def write_vectors(path: Path, cases: list[tuple[int, bytes, bytes]]) -> None:
    """Two lines per load: ``B`` the block padded to 88 bytes, ``A`` the activations.

    Both are little-endian 32-bit words so the testbench can load them with
    ``$readmemh``-style ``fscanf`` into a byte array without knowing the format:

        B <fmt> <22 words>      fmt is the w_fmt code, 0, 1 or 2
        A <64 words>            the last word's padding byte is zero

    The block is zero padded out to the widest format's 88 bytes so one reader
    serves all three; the padding is never consumed because the length is fixed
    by ``fmt``.
    """
    words_b = PAD_BYTES // 4
    lines: list[str] = []
    for fmt, blk, act in cases:
        padded = blk + bytes(PAD_BYTES - len(blk))
        lines.append(" ".join(
            [f"B {fmt}"] +
            [f"{int.from_bytes(padded[4 * k:4 * k + 4], 'little'):08X}"
             for k in range(words_b)]))
        lines.append(" ".join(
            [f"A"] +
            [f"{int.from_bytes(act[4 * k:4 * k + 4], 'little'):08X}"
             for k in range(ACT_WORDS)]))
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


# ---------------------------------------------------------------------------
# static checks: the RTL text, the ROMs and the testbench wiring


def strip_sv_comments(text: str) -> str:
    """Drop ``/* */``, ``//`` comments and string bodies before pattern checks."""
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    text = re.sub(r'"[^"\n]*"', '""', text)
    return re.sub(r"//[^\n]*", " ", text)


def _port_of(port: str) -> list[str]:
    """The names a port may be declared under: the canonical one first."""
    alias = PORT_ALIASES.get(port)
    return [port] if alias is None else [port, alias]


def _port_widths(text: str) -> dict[str, int | None]:
    """``name -> declared width`` for every ``input|output logic`` port."""
    found: dict[str, int | None] = {}
    for m in re.finditer(
        r"^\s*(?:input|output)\s+logic\s*(?:\[\s*(\d+)\s*:\s*0\s*\])?\s*(\w+)",
        text, re.M,
    ):
        msb, name = m.group(1), m.group(2)
        found[name] = 1 if msb is None else int(msb) + 1
    return found


def static_checks() -> tuple[int, list[str]]:
    """Text-level checks that need no simulator; missing files are notices only.

    Returns ``(checks, problems)``.  A problem whose text starts with
    ``missing`` is an absent-file notice the caller reports without failing:
    the RTL and the testbench are owned elsewhere and may lag this script.
    """
    checks, bad = 0, []
    paths = [STREAM_RTL, TB, CHILD_RTL[IQ2_XXS], CHILD_RTL[IQ2_S], CHILD_RTL[IQ1_M]]
    present = {}
    for path in paths:
        checks += 1
        if path.is_file():
            present[path] = path.read_text(encoding="utf-8")
        else:
            bad.append(f"missing {path.relative_to(ROOT)}")

    stream = present.get(STREAM_RTL)
    if stream is not None:
        checks += 1
        if "module otpu_iq_stream" not in stream:
            bad.append("RTL does not declare module otpu_iq_stream")
        widths = _port_widths(stream)
        for name, width in PORTS:
            checks += 1
            found = next((w for w in (_port_of(name)) if w in widths), None)
            if found is None:
                bad.append(f"otpu_iq_stream: port {name} missing")
            elif widths[found] != width:
                bad.append(
                    f"otpu_iq_stream: port {name} is {widths[found]} bits, "
                    f"expected {width}")
        code = strip_sv_comments(stream)
        checks += 1
        m = XILINX_PRIMITIVE.search(code)
        if m:
            bad.append(f"otpu_iq_stream: vendor primitive {m.group(1)}")
        checks += 1
        m = SIM_SYSTEM_TASK.search(code)
        if m:
            bad.append(f"otpu_iq_stream: simulation system task {m.group(1)}")
        checks += 1
        if re.search(r"(?<![#\w])#\s*[\d(]", code):
            bad.append("otpu_iq_stream: delay control in synthesizable code")

    for fmt in FMT_IDS:
        path = CHILD_RTL[fmt]
        text = present.get(path)
        if text is None:
            continue
        name = MODULE_NAME[fmt]
        checks += 1
        if f"module {name}" not in text:
            bad.append(f"{path.name}: does not declare module {name}")
        widths = _port_widths(text)
        for port, width in CHILD_PORTS + CHILD_EXTRA_PORTS[fmt]:
            checks += 1
            found = next((w for w in _port_of(port) if w in widths), None)
            if found is None:
                bad.append(f"{name}: port {port} missing")
            elif widths[found] != width:
                bad.append(
                    f"{name}: port {port} is {widths[found]} bits, expected {width}")
        code = strip_sv_comments(text)
        checks += 1
        m = XILINX_PRIMITIVE.search(code)
        if m:
            bad.append(f"{name}: vendor primitive {m.group(1)}")
        checks += 1
        m = SIM_SYSTEM_TASK.search(code)
        if m:
            bad.append(f"{name}: simulation system task {m.group(1)}")
        checks += 1
        if re.search(r"(?<![#\w])#\s*[\d(]", code):
            bad.append(f"{name}: delay control in synthesizable code")
        # the embedded ROMs must be byte-for-byte copies of the copied tables
        roms = {name: {} for name, _, _ in ROMS}
        for tag, index, value in ROM_LINE.findall(code):
            roms[tag][int(index)] = int(value, 16)
        for tag in ROM_REQUIRED[fmt]:
            depth = next(d for n, d, _ in ROMS if n == tag)
            getter = next(fn for n, _, fn in ROMS if n == tag)
            entries = roms[tag]
            checks += 1
            if sorted(entries) != list(range(depth)):
                missing = depth - len(entries)
                bad.append(
                    f"{name}: {tag} has {len(entries)} distinct entries, "
                    f"expected {depth} ({missing} unassigned)")
                continue
            for i in range(depth):
                checks += 1
                if entries[i] != getter(i):
                    bad.append(
                        f"{name}: {tag}[{i}] = 0x{entries[i]:x}, "
                        f"table has 0x{getter(i):x}")
                    break

    tb = present.get(TB)
    if tb is not None:
        checks += 1
        if "module tb_iq_stream" not in tb:
            bad.append("testbench does not declare module tb_iq_stream")
        # the testbench binds the DUT by wildcard (dut (.*)), so a port is
        # connected when a TB signal of that name exists; explicit .port(...)
        # bindings are accepted too
        wildcard = bool(re.search(r"\.\s*\*\s*\)", tb))
        # a declaration may bind several names: "logic a, b = '0, c;"
        tb_signals = set()
        for decl in re.findall(r"^\s*logic\b([^;]*);", tb, re.M):
            fn = re.findall(r"([A-Za-z_]\w*)\s*(?:=|,|$)", decl)
            tb_signals.update(n for n in fn if n not in ("logic",))
        for name, _ in PORTS:
            checks += 1
            if any(re.search(rf"\.{w}\s*\(", tb) for w in _port_of(name)):
                continue
            if wildcard and tb_signals & set(_port_of(name)):
                continue
            bad.append(f"testbench does not connect .{name}")
        for plus in ("vectors", "out", "cases"):
            checks += 1
            if plus not in tb:
                bad.append(f"testbench does not read +{plus}")
        checks += 1
        if not SUMMARY_LINE.search(tb) and "IQSTREAMTEST" not in tb:
            bad.append("testbench does not print the IQSTREAMTEST summary")
    return checks, bad


# ---------------------------------------------------------------------------
# selftest: the integer terms against the float dequantiser, in exact rationals
#
# For every group the dequantised weights are rebuilt from the block bytes with
# the ggml float formulas and dotted with the int8 activations in exact
# rational arithmetic; that dense dot must equal the integer identity:
#
#     IQ2_XXS, IQ2_S   dense = d * bsum / 8        (bsum already carries 2n+1)
#     IQ1_M            dense = d * (bsum + aux/8)  (aux carries the +-1/8 rows)


def dequant_group_dot(fmt: int, blk: bytes, act: bytes, g: int) -> Fraction:
    """The group's 32-weight dot as the float dequantiser defines it, exactly."""
    d = fp16_fraction(_d_bits(fmt, blk))
    signed_act = _q8_act(act)
    total = Fraction(0)
    for i in range(32):
        idx = 32 * g + i
        a = signed_act[idx]
        if fmt == IQ2_XXS:
            base = 2 + 8 * g
            lane, j = divmod(i, 8)
            aux = _u32(blk, base + 4)
            grid = iq_tables.IQ2XXS_GRID[blk[base + lane]]
            mag = (grid >> (8 * j)) & 0xFF
            sign = (iq_tables.KSIGNS_IQ2XS[(aux >> (7 * lane)) & 0x7F] >> j) & 1
            w = d * Fraction(2 * (aux >> 28) + 1, 8) * (-mag if sign else mag)
        elif fmt == IQ2_S:
            row = 4 * g + i // 8
            j = i % 8
            ls_byte = blk[74 + g]
            nib = (ls_byte & 0xF) if (i // 8) < 2 else (ls_byte >> 4)
            grid = iq_tables.IQ2S_GRID[
                blk[2 + row] | (((blk[66 + row // 4] >> (2 * (row % 4))) & 3) << 8)]
            mag = (grid >> (8 * j)) & 0xFF
            sign = (blk[34 + row] >> j) & 1
            w = d * Fraction(2 * nib + 1, 8) * (-mag if sign else mag)
        else:
            row = 4 * g + i // 8
            j = i % 8
            nibble = (blk[32 + row // 2] >> (4 * (row % 2))) & 0xF
            grid = iq_tables.grid_signed(
                iq_tables.IQ1S_GRID[blk[row] | ((nibble & 0x7) << 8)])
            ls1, ls2 = iq1_m_scales(blk, g)
            ls = ls1 if (row - 4 * g) < 2 else ls2
            offset = Fraction(-1, 8) if nibble & 0x8 else Fraction(1, 8)
            w = d * Fraction(ls) * (grid[j] + offset)
        total += w * a
    return total


def selftest(cases: int, seed: int) -> tuple[int, list[str]]:
    """Tables, RTL text and the model identity; runs without a simulator."""
    checks, bad = 0, []

    # the copied codebooks are intact and carry the documented alphabets
    for tag, depth in sorted(iq_tables.table_sizes().items()):
        checks += 1
        if depth <= 0:
            bad.append(f"table {tag} is empty")
    checks += 1
    if iq_tables.sha256_tables() != iq_tables.sha256_tables():
        bad.append("table digests are not stable between calls")
    checks += 1
    if iq_tables.IQ2S_GRID is not __import__("iq_rtl_check").IQ2S_GRID:
        bad.append("IQ2_S grid is not the object imported from iq_rtl_check")
    checks += 1
    if set(iq_tables.IQ2_GRID_MAGNITUDES) != set(GRID_MAGNITUDES):
        bad.append("IQ2 magnitude alphabet differs from the pinned (8, 25, 43)")
    checks += 1
    if tuple(sorted(set(iq_tables.IQ1S_GRID_VALUES))) != tuple(sorted(GRID_TERNARY)):
        bad.append("IQ1_M grid ternary alphabet differs from the pinned (-1, 0, 1)")

    # every finite fp16 decodes to the same value struct.unpack gives
    rng = random.Random(seed)
    for _ in range(64):
        raw = _fp16_scalar(rng) | (rng.getrandbits(1) << 15)
        checks += 1
        want = Fraction(struct.unpack("<e", raw.to_bytes(2, "little"))[0])
        if fp16_fraction(raw) != want:
            bad.append(f"fp16 0x{raw:04x} decoded wrong")
    for raw in (0x0000, 0x8000, 0x0001, 0x03FF, 0x0400, 0x7BFF, 0xFBFF, 0x3555):
        checks += 1
        try:
            fp16_fraction(raw)
        except NonFiniteScale:
            bad.append(f"finite fp16 0x{raw:04x} rejected")
    for raw in (0x7C00, 0xFC00, 0x7E00, 0x7FFF):
        checks += 1
        if fp16_is_finite(raw):
            bad.append(f"non-finite fp16 0x{raw:04x} accepted")

    # the integer path reproduces the float dequantiser in exact rationals
    groups = 0
    for ci, (fmt, blk, act) in enumerate(generate_cases(cases, seed)):
        if not fp16_is_finite(_d_bits(fmt, blk)):
            bad.append(f"case {ci}: generated a non-finite scale")
            continue
        d = fp16_fraction(_d_bits(fmt, blk))
        for g in range(N_GROUPS):
            groups += 1
            t = terms(fmt, blk, act, g)
            dense = dequant_group_dot(fmt, blk, act, g)
            if fmt == IQ1_M:
                want = d * (Fraction(t["bsum"]) + Fraction(t["aux"], 8))
            else:
                want = d * Fraction(t["bsum"]) / 8
            checks += 1
            if dense != want:
                kind = case_kind(ci, cases)
                bad.append(
                    f"{FMT_NAME[fmt]} case {ci} (kind {kind}) group {g}: "
                    f"dense {float(dense):.9g} != identity {float(want):.9g}"
                    f" [bsum {t['bsum']}, aux {t['aux']},"
                    f" ls1 {t['ls1']}, ls2 {t['ls2']}]")
                continue
            # the dumped fields must respect their declared widths
            for key, width in (("bsum", PARTIAL_BITS[fmt]), ("aux", AUX_BITS[fmt]),
                               ("terms", 2), ("ls1", 5), ("ls2", 5)):
                checks += 1
                value = t[key]
                if key in ("bsum", "aux"):
                    ok = -(1 << (width - 1)) <= value < (1 << (width - 1))
                else:
                    ok = 0 <= value < (1 << width)
                if not ok:
                    bad.append(
                        f"{FMT_NAME[fmt]} case {ci} group {g}: {key} {value} "
                        f"does not fit {width} bits")
            checks += 1
            if not 0 <= t["index"] < (1 << 44):
                bad.append(f"{FMT_NAME[fmt]} case {ci} group {g}: index {t['index']} > 44 bits")
            checks += 1
            if not 0 <= t["signs"] < (1 << 32):
                bad.append(f"{FMT_NAME[fmt]} case {ci} group {g}: signs {t['signs']} > 32 bits")
            checks += 1
            if not 0 <= t["delta"] < 16:
                bad.append(f"{FMT_NAME[fmt]} case {ci} group {g}: delta {t['delta']} > 4 bits")

    # the block side channel: d bits and the scale payload
    for ci, (fmt, blk, _act) in enumerate(generate_cases(cases, seed)):
        meta = block_metadata(fmt, blk)
        checks += 1
        if not 0 <= meta["d"] < (1 << 16):
            bad.append(f"{FMT_NAME[fmt]} case {ci}: d {meta['d']} > 16 bits")
        checks += 1
        if not 0 <= meta["scales"] < (1 << 64):
            bad.append(f"{FMT_NAME[fmt]} case {ci}: scales {meta['scales']} > 64 bits")

    # a zero-activation block has an exactly zero dot for every format
    for fmt in FMT_IDS:
        blk, _ = gen_case(random.Random(seed + fmt), 3, fmt)
        zero = bytes(QK_K)
        for g in range(N_GROUPS):
            checks += 1
            t = terms(fmt, blk, zero, g)
            if t["bsum"] != 0 or t["aux"] != 0:
                bad.append(f"{FMT_NAME[fmt]}: zero activations gave bsum {t['bsum']}, aux {t['aux']}")

    # the worst-case field widths stay inside the RTL's 32-bit group ports
    for fmt in FMT_IDS:
        for label, bound in (("bsum", PARTIAL_BITS[fmt]), ("aux", AUX_BITS[fmt])):
            checks += 1
            if bound > 32:
                bad.append(f"{FMT_NAME[fmt]}: {label} bound {bound} does not fit int32")
    # the analytic bound is at least the largest value the model produced
    seen = {fmt: {"bsum": 0, "aux": 0} for fmt in FMT_IDS}
    for fmt, blk, act in generate_cases(cases, seed):
        for g in range(N_GROUPS):
            t = terms(fmt, blk, act, g)
            for key in ("bsum", "aux"):
                seen[fmt][key] = max(seen[fmt][key], abs(t[key]))
    for fmt in FMT_IDS:
        for key in ("bsum", "aux"):
            checks += 1
            bound = (PARTIAL_BITS if key == "bsum" else AUX_BITS)[fmt]
            if seen[fmt][key] >= (1 << (bound - 1)):
                bad.append(
                    f"{FMT_NAME[fmt]}: observed |{key}| {seen[fmt][key]} exceeds the "
                    f"{bound}-bit bound")

    checks_static, static_bad = static_checks()
    checks += checks_static
    bad += static_bad
    return checks, bad, groups


# ---------------------------------------------------------------------------
# testbench dump parsing
# ---------------------------------------------------------------------------


def parse_dump(text: str) -> list[dict]:
    """Parse the testbench dump into ``[{ci, fmt, d, scales, groups: [...]}]``.

    Every line is either ``case <ci> fmt <2hex> d <4hex> scales <16hex>`` or
    ``g <g> <bsum 8hex> <aux 8hex> <terms 2hex> <ls1 2hex> <ls2 2hex>
    <signs 8hex> <indices 11hex> <delta 1hex>``.  The group index is explicit
    and the 44-bit index is written as 11 hex digits.  Anything else is an
    error, so a truncated or reordered dump cannot pass silently.
    """
    cases: list[dict] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        m = CASE_LINE.match(line)
        if m:
            cases.append({
                "ci": int(m.group(1)),
                "fmt": int(m.group(2), 16),
                "d": int(m.group(3), 16),
                "scales": int(m.group(4), 16),
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
                "aux": int(m.group(3), 16),
                "terms": int(m.group(4), 16),
                "ls1": int(m.group(5), 16),
                "ls2": int(m.group(6), 16),
                "signs": int(m.group(7), 16),
                "index": int(m.group(8), 16),
                "delta": int(m.group(9), 16),
            })
            continue
        raise ValueError(f"unexpected dump line: {line!r}")
    return cases


def _signed(value: int, bits: int) -> int:
    """Interpret ``value`` as a ``bits``-wide two's complement number."""
    value &= (1 << bits) - 1
    return value - (1 << bits) if value >> (bits - 1) else value


def compare(cases: list[tuple[int, bytes, bytes]], dump: list[dict]) -> tuple[int, list[str]]:
    """Compare the parsed dump against the model terms, field by field."""
    checks, bad = 0, []
    checks += 1
    if len(dump) != len(cases):
        bad.append(f"dump has {len(dump)} cases, ran {len(cases)}")

    for ci, (fmt, blk, act) in enumerate(cases):
        if ci >= len(dump):
            break
        got = dump[ci]
        checks += 1
        if got["ci"] != ci:
            bad.append(f"case {ci}: dump case index {got['ci']}")
        checks += 1
        if got["fmt"] != fmt:
            bad.append(f"{FMT_NAME[fmt]} case {ci}: dump fmt {got['fmt']}")
        meta = block_metadata(fmt, blk)
        checks += 1
        if got["d"] != meta["d"]:
            bad.append(
                f"{FMT_NAME[fmt]} case {ci}: blk_d {got['d']:04x} != {meta['d']:04x}")
        checks += 1
        if got["scales"] != meta["scales"]:
            bad.append(
                f"{FMT_NAME[fmt]} case {ci}: blk_scales {got['scales']:016x} != "
                f"{meta['scales']:016x}")
        checks += 1
        if len(got["groups"]) != N_GROUPS:
            bad.append(
                f"{FMT_NAME[fmt]} case {ci}: {len(got['groups'])} groups, expected {N_GROUPS}")
            continue
        for g in range(N_GROUPS):
            row = got["groups"][g]
            want = terms(fmt, blk, act, g)
            checks += 1
            if row["g"] != g:
                bad.append(f"{FMT_NAME[fmt]} case {ci}: group {g} tagged {row['g']}")
                continue
            # bsum and aux are signed 32-bit; the rest are plain bit fields
            checks += 1
            if _signed(row["bsum"], 32) != want["bsum"]:
                bad.append(
                    f"{FMT_NAME[fmt]} case {ci} group {g}: bsum "
                    f"{_signed(row['bsum'], 32)} != {want['bsum']}")
            checks += 1
            if _signed(row["aux"], 32) != want["aux"]:
                bad.append(
                    f"{FMT_NAME[fmt]} case {ci} group {g}: aux "
                    f"{_signed(row['aux'], 32)} != {want['aux']}")
            for key in ("terms", "ls1", "ls2", "signs", "index", "delta"):
                checks += 1
                if row[key] != want[key]:
                    bad.append(
                        f"{FMT_NAME[fmt]} case {ci} group {g}: {key} "
                        f"{row[key]} != {want[key]}")
    return checks, bad


# ---------------------------------------------------------------------------
# reporting
# ---------------------------------------------------------------------------


def report(ok: bool, cases: int, checks: int, bad: list[str], **extra) -> int:
    """Print the machine-readable result and return the process exit code."""
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
    """A tool or build failure: exit code 2, distinct from a mismatch."""
    print(json.dumps({
        "ok": False,
        "cases": 0,
        "checks": checks,
        "mismatches": [message],
        "status": status,
        **extra,
    }, indent=2), flush=True)
    return 2


def report_path(path: Path) -> str:
    """Keep in-tree reports relative; permit caller-owned external build trees."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def find_verilator(name: str) -> str | None:
    """Resolve the simulator, accepting a bare PATH name or an explicit path."""
    found = shutil.which(name)
    if found:
        return found
    path = Path(name)
    return str(path) if path.is_file() else None


def grid_sha256() -> dict[str, str]:
    """The digest of every codebook the three engines embed."""
    tables = dict(iq_tables.sha256_tables())
    tables["iq2s_grid"] = iq_tables.IQ2S_GRID_SHA256
    return tables


# ---------------------------------------------------------------------------
# entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Check the three-format IQ stream: model identities, RTL "
                    "text, ROM contents and (optionally) a Verilator run."
    )
    ap.add_argument("--cases", type=int, default=N_CASES,
                    help=f"cases per format (default {N_CASES}, {N_CASES * len(FMT_IDS)} blocks)")
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED,
                    help=f"case seed (default {DEFAULT_SEED})")
    ap.add_argument("--build", type=Path, default=DEFAULT_BUILD,
                    help="build directory (default build/iq_multi)")
    ap.add_argument("--verilator", default="verilator",
                    help="verilator executable (default: PATH)")
    ap.add_argument("--selftest", action="store_true",
                    help="check tables, RTL text and the model without a simulator")
    args = ap.parse_args(argv)

    if args.cases < 1:
        return fail("args", "--cases must be at least 1")

    static_checked, static_bad = static_checks()
    checks = static_checked
    bad = list(static_bad)

    if args.selftest:
        st_checks, st_bad, groups = selftest(args.cases, args.seed)
        checks += st_checks
        bad += st_bad
        return report(not bad, args.cases, checks, bad, selftest=True,
                      blocks=args.cases * len(FMT_IDS), groups=groups,
                      seed=args.seed, grid_sha256=grid_sha256())

    verilator = find_verilator(args.verilator)
    if verilator is None:
        return fail("no_verilator",
                    f"{args.verilator} not found; install Verilator 5.x "
                    f"or run --selftest", checks, seed=args.seed)
    try:
        proc = subprocess.run([verilator, "--version"], capture_output=True,
                              text=True, timeout=120, check=False)
        version = (proc.stdout.strip().splitlines() or ["unknown"])[0]
    except (OSError, subprocess.SubprocessError):
        version = "unknown"

    cases = generate_cases(args.cases, args.seed)
    mdir = Path(args.build)
    if not mdir.is_absolute():
        mdir = ROOT / mdir
    mdir.mkdir(parents=True, exist_ok=True)
    vectors = mdir / "vectors.hex"
    dump_path = mdir / "dump.txt"
    write_vectors(vectors, cases)
    ordered_cases = cases

    cmd = [verilator, *VERILATOR_FLAGS, "--Mdir", str(mdir), str(TB),
           str(STREAM_RTL), *CHILD_RTL.values()]
    print(f"building {mdir / 'Vtb_iq_stream'}", file=sys.stderr, flush=True)
    try:
        proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True,
                              text=True, timeout=3600)
    except (OSError, subprocess.SubprocessError) as exc:
        return fail("build", f"verilator failed to run: {exc}", checks, seed=args.seed)
    if proc.returncode != 0:
        sys.stderr.write(proc.stdout[-4000:])
        sys.stderr.write(proc.stderr[-8000:])
        return fail("build", f"verilator exited {proc.returncode}", checks,
                    seed=args.seed)

    exe = mdir / TB_BINARY
    if not exe.is_file():
        return fail("build", f"verilator produced no {exe}", checks, seed=args.seed)

    cmd = [str(exe), f"+vectors={vectors}", f"+out={dump_path}",
           f"+cases={len(ordered_cases)}"]
    try:
        proc = subprocess.run(cmd, cwd=str(mdir), capture_output=True,
                              text=True, timeout=3600)
    except (OSError, subprocess.SubprocessError) as exc:
        return fail("run", f"testbench failed to run: {exc}", checks, seed=args.seed)
    sys.stderr.write(proc.stdout[-4000:])
    if proc.returncode != 0:
        sys.stderr.write(proc.stderr[-8000:])
        return fail("run", f"testbench exited {proc.returncode}", checks,
                    seed=args.seed)

    cases = ordered_cases
    m = SUMMARY_LINE.search(proc.stdout)
    if not m:
        return fail("run", "testbench printed no IQSTREAMTEST summary", checks,
                    seed=args.seed)
    tb_cases, tb_beats, tb_cycles, tb_errors = (int(m.group(i)) for i in range(1, 5))
    checks += 1
    if tb_errors:
        bad.append(f"testbench protocol errors: {tb_errors}")
    checks += 1
    if tb_cases != len(cases):
        bad.append(f"testbench ran {tb_cases} cases, generated {len(cases)}")
    # The testbench counts every handshake it performed: one beat pre load plus the
    # eight activation beats, so a case costs w_beats + 8 beats.  It checks the
    # per-case latency itself (accepted cycles == w_beats + 7, i.e. 17/19/15) and
    # counts a cycle once per clock, so "cycles" is total wall time, not a sum of
    # the per-case latencies: only its being at least the summed latency is known.
    checks += 1
    want_beats = sum(FMT_BEATS[fmt] + 8 for fmt, _, _ in cases)
    if tb_beats != want_beats:
        bad.append(f"testbench saw {tb_beats} beats, expected {want_beats}")
    checks += 1
  # cyc is the TB's free-running global cycle counter, including reset/rejection
  # testing.  Per-case accepted latency is enforced in the testbench itself.
    min_cycles = sum(FMT_CYCLES[fmt] for fmt, _, _ in cases) + 16
    if tb_cycles < min_cycles:
        bad.append(
            f"testbench saw {tb_cycles} cycles, below the {min_cycles} "
            f"the per-case latencies require")

    try:
        dump = parse_dump(dump_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        return fail("run", f"cannot parse {dump_path}: {exc}", checks, seed=args.seed)

    cmp_checks, cmp_bad = compare(cases, dump)
    checks += cmp_checks
    bad += cmp_bad

    return report(not bad, len(cases), checks, bad, seed=args.seed,
                  verilator=version, grid_sha256=grid_sha256(),
                  beats=tb_beats, cycles=tb_cycles,
                  saturated_cases=sum(case_kind(ci, args.cases) == 3
                                      for ci in range(len(cases))),
                  build=report_path(mdir),
                  vectors=report_path(vectors), dump=report_path(dump_path))


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
