# SPDX-License-Identifier: MIT
# Copyright (c) 2026 lvyufeng and the openTPU contributors
#
# The decode and integer-dot formulas below are independent transcriptions of
# the MIT-licensed ggml scalar reference (see ATTRIBUTION.md); the codebook
# *data* they consume lives in ``iq_tables.py``.

"""Strict scalar reference for the IQ2_XXS, IQ2_S and IQ1_M weight formats.

This module is the production half of the packed-golden oracle: it is the code
path under test.  The independent reference implementation used to derive the
expected values in the JSON fixtures lives in ``iq_vectors.py``; the README
records which is which and why they are separate.

Every block holds 256 values, split into eight 32-element groups that ggml calls
``ib32``.  The three formats share the activation format ``Q8_K`` (292 bytes per
256 values: one FP32 scale, 256 signed bytes, sixteen int16 16-element sums).
``Q8_K`` blocks are checked for a finite ``d``; their ``bsums`` are *not*
consumed because the IQ dot kernels only read ``qs`` and ``d``.

Block layouts (little-endian, exactly the C structs in
``engine/include/strata/quant/strata-quant-common.h`` at the locked Strata
revision ``3142b66b96e9c0e775e30f620b11b191a1427c2a``)::

    Q8_K      d:fp32 @0 (4) | qs:int8[256] @4 | bsums:int16[16] @260     = 292
    IQ2_XXS   d:fp16 @0 (2) | qs:u8[64] @2                                =  66
    IQ2_S     d:fp16 @0 (2) | qs:u8[64] @2 | qh:u8[8] @66 | sc:u8[8] @74 =  82
    IQ1_M     qs:u8[32] @0 | qh:u8[16] @32 | sc:u8[8] @48                 =  56

``FORMATS`` / ``ACTIVATIONS`` expose the same descriptors as plain JSON-safe
data for fixtures and for the RTL vector package.

Decode semantics (ggml ``ggml-quants.c``, pinned ggml revision
``3cf03257f219afbe7334045ff7c6a06ac68c627d``; the same code is vendored at the
locked Strata revision):

``dequantize_row_iq2_xxs`` (:2488-2512)::

    db = d * (0.5f + (aux32[1] >> 28)) * 0.25f
    y[j] = db * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f)
    grid  = iq2xxs_grid[aux8[l]]                   aux8 = first 4 bytes of group
    signs = ksigns_iq2xs[(aux32[1] >> 7*l) & 127]

``dequantize_row_iq2_s`` (:2543-2577)::

    db[0] = d * (0.5f + (scales[ib32] & 0xf)) * 0.25f
    db[1] = d * (0.5f + (scales[ib32] >>  4)) * 0.25f
    y[j]  = db[l/2] * grid[j] * (signs[l] & kmask_iq2xs[j] ? -1.f : 1.f)
    grid  = iq2s_grid[qs[l] | (qh[ib32] << (8-2*l) & 0x300)]
    signs = qs[QK_K/8 + l]                          (qs advances by 4 per group)

``dequantize_row_iq1_m`` (:2675-2714)::

    d     = fp16(sc[0]>>12 | (sc[1]>>8 & 0x00f0) |
                 (sc[2]>>4 & 0x0f00) | (sc[3] & 0xf000))
    dl1   = d * (2*((sc[ib/2] >> (6*(ib%2)+0)) & 0x7) + 1)
    dl2   = d * (2*((sc[ib/2] >> (6*(ib%2)+3)) & 0x7) + 1)
    idx   = qs[l] | ((qh[l/2] << (8-4*(l%2))) & 0x700)
    delta = qh[l/2] & (0x08 or 0x80) ? -IQ1S_DELTA : IQ1S_DELTA
    y[j]  = (l < 2 ? dl1 : dl2) * (grid[j] + delta)   IQ1S_DELTA = 0.125f

The grid words are uint64 little-endian; the IQ1_M grid is read as eight int8
lanes (its bytes are only 0x00, 0x01 and 0xff, i.e. -1/0/+1).

Dot semantics transcribe the generic ggml CPU kernels in
``engine/third_party/strata-llama/ggml/src/ggml-cpu/quants.c`` at the same
pinned revision.  They return ``*s = d_weight * d_activation * 0.125f *
sum_ib32(ls * sumi)`` for IQ2_XXS/IQ2_S and
``*s = d_weight * d_activation * (sumi1 + 0.125f*sumi2)`` for IQ1_M; the exact
expression order is reproduced with separate binary32 rounding.
"""

from __future__ import annotations

from . import iq_tables as _tables
from .formats import (
    ARITHMETIC_ID,
    MAX_ELEMENTS,
    FormatError,
    _add32,
    _half32,
    _hex32,
    _int32,
    _mul32,
)

#: Exact binary32 constants used by the ggml reference expressions.
_HALF_BITS = 0x3F000000  # 0.5f
_QUARTER_BITS = 0x3E800000  # 0.25f
_EIGHTH_BITS = 0x3E000000  # 0.125f, also IQ1S_DELTA / IQ1M_DELTA
_NEG_ONE_BITS = 0xBF800000  # -1.0f, the decode sign multiply
_SIGN_BIT = 0x80000000

#: Maximum packed-activation payload in bytes (256 blocks of Q8_K).
MAX_IQ_PAYLOAD_BYTES = (MAX_ELEMENTS // 256) * 292

# Offsets/sizes are (offset, bytes) inside one block, little-endian.
_WEIGHT_SPECS = {
    "IQ2_XXS": {
        "ggml_type": 16,
        "block_elements": 256,
        "block_bytes": 66,
        "activation_format": "Q8_K",
        "activation_block_bytes": 292,
        "subblocks": 8,
        "subblock_elements": 32,
        "grid_entries": 256,
        "grid_index_bits": 8,
        "index_high_bits": 0,
        "sign_table_entries": 128,
        "group_scales": 8,
        "group_scale_bits": 4,
        "group_scale_levels": 16,
        "integer_subdot_fields": ["grid"],
        "delta_flags_per_group": 0,
        "block_scale": "fp16",
        "final_scale_fp32": "0x3e000000",
        "layout": (("d", 0, 2), ("qs", 2, 64)),
    },
    "IQ2_S": {
        "ggml_type": 22,
        "block_elements": 256,
        "block_bytes": 82,
        "activation_format": "Q8_K",
        "activation_block_bytes": 292,
        "subblocks": 8,
        "subblock_elements": 32,
        "grid_entries": 1024,
        "grid_index_bits": 10,
        "index_high_bits": 2,
        "sign_table_entries": 128,
        "group_scales": 8,
        "group_scale_bits": 4,
        "group_scale_levels": 16,
        "integer_subdot_fields": ["low16", "high16"],
        "delta_flags_per_group": 0,
        "block_scale": "fp16",
        "final_scale_fp32": "0x3e000000",
        "layout": (("d", 0, 2), ("qs", 2, 64), ("qh", 66, 8), ("scales", 74, 8)),
    },
    "IQ1_M": {
        "ggml_type": 29,
        "block_elements": 256,
        "block_bytes": 56,
        "activation_format": "Q8_K",
        "activation_block_bytes": 292,
        "subblocks": 8,
        "subblock_elements": 32,
        "grid_entries": _tables.NGRID_IQ1S,
        "grid_index_bits": 11,
        "index_high_bits": 3,
        "sign_table_entries": 0,
        "group_scales": 8,
        "group_scale_bits": 3,
        "group_scale_levels": 8,
        "integer_subdot_fields": ["grid", "delta"],
        "delta_flags_per_group": 4,
        "block_scale": "fp16-packed",
        "final_scale_fp32": None,
        "layout": (("qs", 0, 32), ("qh", 32, 16), ("scales", 48, 8)),
    },
}

_WEIGHT_SIZES = {
    name: (spec["block_elements"], spec["block_bytes"])
    for name, spec in _WEIGHT_SPECS.items()
}

_ACTIVATION_SPECS = {
    "Q8_K": {
        "block_elements": 256,
        "block_bytes": 292,
        "block_scale": "fp32",
        "subblock_elements": 16,
        "layout": (("d", 0, 4), ("qs", 4, 256), ("bsums", 260, 32)),
    },
}

_ACTIVATION_SIZES = {
    name: (spec["block_elements"], spec["block_bytes"])
    for name, spec in _ACTIVATION_SPECS.items()
}


def _public_spec(spec: dict) -> dict:
    result = {key: value for key, value in spec.items() if key != "layout"}
    result["layout"] = [list(item) for item in spec["layout"]]
    return result


#: JSON-safe descriptors for the three supported weight formats.
FORMATS = {name: _public_spec(spec) for name, spec in _WEIGHT_SPECS.items()}

#: JSON-safe descriptor for the only supported IQ activation format.
ACTIVATIONS = {
    name: _public_spec(spec) for name, spec in _ACTIVATION_SPECS.items()
}

#: Stable names/lengths of the integer partial sums reported per 32-value group.
_SUBDOT_FIELDS = {
    name: tuple(spec["integer_subdot_fields"])
    for name, spec in _WEIGHT_SPECS.items()
}


def _check_specs() -> None:
    for name, spec in _WEIGHT_SPECS.items():
        assert spec["block_elements"] == 256, name
        assert spec["subblocks"] * spec["subblock_elements"] == 256, name
        assert spec["subblocks"] == spec["group_scales"], name
        assert spec["block_bytes"] == sum(item[2] for item in spec["layout"]), name
        assert spec["grid_entries"] == 1 << spec["grid_index_bits"], name
        assert spec["activation_format"] == "Q8_K", name
        assert spec["activation_block_bytes"] == 292, name
        assert spec["block_scale"] in ("fp16", "fp16-packed"), name
        assert 1 <= len(spec["integer_subdot_fields"]) <= 2, name
        assert spec["delta_flags_per_group"] in (0, 4), name
    for spec in _ACTIVATION_SPECS.values():
        assert spec["block_bytes"] == sum(item[2] for item in spec["layout"])
        assert spec["block_elements"] == 256


_check_specs()


def parse_hex(text: str) -> bytes:
    """Parse bounded, contiguous, even-length ASCII hex; never strip input."""
    import re

    if type(text) is not str:
        raise FormatError("invalid_hex", "hex input must be a string")
    if len(text) > 2 * MAX_IQ_PAYLOAD_BYTES:
        raise FormatError("invalid_hex", "hex input exceeds the packed-row byte limit")
    pattern = re.compile(r"[0-9a-fA-F]+", flags=re.ASCII)
    if not text or len(text) % 2 or pattern.fullmatch(text) is None:
        raise FormatError(
            "invalid_hex", "hex must contain contiguous pairs of ASCII hex digits"
        )
    return bytes.fromhex(text)


def block_layout(format_name: str) -> list[dict]:
    """Return the field/offset/bytes descriptor of one block.

    Accepts the three IQ weight formats plus ``Q8_K`` (the activation format).
    """
    if type(format_name) is not str or (
        format_name not in _WEIGHT_SPECS and format_name not in _ACTIVATION_SPECS
    ):
        raise FormatError(
            "unsupported_format",
            "supported block layouts are IQ2_XXS, IQ2_S, IQ1_M and Q8_K",
        )
    spec = _WEIGHT_SPECS.get(format_name) or _ACTIVATION_SPECS[format_name]
    return [
        {"field": field, "offset": offset, "bytes": size}
        for field, offset, size in spec["layout"]
    ]


def _weight_size(format_name: str) -> tuple[int, int]:
    if type(format_name) is not str or format_name not in _WEIGHT_SIZES:
        raise FormatError(
            "unsupported_format",
            "supported IQ weight formats are IQ2_XXS, IQ2_S and IQ1_M",
        )
    return _WEIGHT_SIZES[format_name]


def _activation_size(activation_format: str) -> tuple[int, int]:
    if type(activation_format) is not str or activation_format not in _ACTIVATION_SIZES:
        raise FormatError(
            "unsupported_activation_format",
            "the only IQ activation format is Q8_K",
        )
    return _ACTIVATION_SIZES[activation_format]


def _validate_row(
    block_elements: int,
    block_bytes: int,
    payload: bytes,
    elements: int,
    label: str,
) -> int:
    if type(elements) is not int or not 1 <= elements <= MAX_ELEMENTS:
        raise FormatError(
            "invalid_elements", f"elements must be an integer in [1, {MAX_ELEMENTS}]"
        )
    if elements % block_elements:
        raise FormatError(
            "invalid_elements", f"elements must be divisible by {block_elements}"
        )
    if type(payload) is not bytes:
        raise FormatError("invalid_payload", f"{label} payload must be bytes")
    blocks = elements // block_elements
    expected = blocks * block_bytes
    if len(payload) != expected:
        raise FormatError(
            "invalid_length", f"{label} payload must contain exactly {expected} bytes"
        )
    return blocks


def _validate_weight_row(format_name: str, payload: bytes, elements: int) -> int:
    block_elements, block_bytes = _weight_size(format_name)
    return _validate_row(block_elements, block_bytes, payload, elements, "weight")


def _validate_activation_row(
    activation_format: str, payload: bytes, elements: int
) -> int:
    block_elements, block_bytes = _activation_size(activation_format)
    return _validate_row(
        block_elements, block_bytes, payload, elements, "activation"
    )


def _u16(data: bytes, offset: int) -> int:
    return data[offset] | (data[offset + 1] << 8)


def _u32(data: bytes, offset: int) -> int:
    return (
        data[offset]
        | (data[offset + 1] << 8)
        | (data[offset + 2] << 16)
        | (data[offset + 3] << 24)
    )


def _activate(activation_format: str, payload: bytes, blocks: int) -> list[tuple[int, list[int]]]:
    """Return one (fp32 scale bits, 256 signed int8 values) pair per block."""
    _activation_size(activation_format)
    result = []
    for block in range(blocks):
        offset = block * 292
        scale_bits = _u32(payload, offset)
        if (scale_bits >> 23) & 0xFF == 0xFF:
            raise FormatError(
                "nonfinite_scale", "Q8_K activation scale must not be infinity or NaN"
            )
        raw = payload[offset + 4 : offset + 260]
        result.append(
            (scale_bits, [byte - 256 if byte >= 128 else byte for byte in raw])
        )
    return result


def _iq2_xxs_terms(
    block: bytes,
) -> tuple[int, list[tuple[int, bytes, tuple[int, ...]]]]:
    """Unpack one IQ2_XXS block: scale bits plus eight (q, indices, sign bytes)."""
    scale_bits = _half32(_u16(block, 0))
    groups = []
    for ib32 in range(8):
        base = 2 + 8 * ib32
        indices = block[base : base + 4]
        high = _u32(block, base + 4)
        magnitude = high >> 28
        sign_bytes = tuple(
            _tables.KSIGNS_IQ2XS[(high >> (7 * lane)) & 127] for lane in range(4)
        )
        groups.append((magnitude, indices, sign_bytes))
    return scale_bits, groups


def _iq2_s_terms(
    block: bytes,
) -> tuple[int, list[tuple[int, int, tuple[tuple[int, int], ...]]]]:
    """Unpack one IQ2_S block: scale bits plus eight (low q, high q, rows)."""
    scale_bits = _half32(_u16(block, 0))
    qs = block[2:66]
    qh = block[66:74]
    scales = block[74:82]
    groups = []
    for ib32 in range(8):
        low = scales[ib32] & 0xF
        high = scales[ib32] >> 4
        rows = tuple(
            (
                qs[4 * ib32 + lane]
                | ((qh[ib32] << (8 - 2 * lane)) & 0x300),
                qs[32 + 4 * ib32 + lane],
            )
            for lane in range(4)
        )
        groups.append((low, high, rows))
    return scale_bits, groups


def _iq1_m_terms(
    block: bytes,
) -> tuple[int, list[tuple[int, int, tuple[tuple[int, int], ...]]]]:
    """Unpack one IQ1_M block: packed scale bits plus eight (ls1, ls2, rows).

    ``rows`` holds (11-bit grid index, delta sign) for the four 8-value lanes of
    the group.  ``sc`` is read as four little-endian uint16 words; the fp16
    scale is scattered over their top nibbles and each word also stores two
    6-bit group-scale fields.
    """
    scales = [_u16(block, 48 + 2 * index) for index in range(4)]
    packed = (
        (scales[0] >> 12)
        | ((scales[1] >> 8) & 0x00F0)
        | ((scales[2] >> 4) & 0x0F00)
        | (scales[3] & 0xF000)
    )
    scale_bits = _half32(packed)
    qs = block[0:32]
    qh = block[32:48]
    groups = []
    for ib32 in range(8):
        rows = []
        for lane in range(4):
            high_byte = qh[2 * ib32 + lane // 2]
            index = qs[4 * ib32 + lane] | ((high_byte << (8 - 4 * (lane % 2))) & 0x700)
            mask = 0x08 if lane % 2 == 0 else 0x80
            rows.append((index, -1 if high_byte & mask else 1))
        ls1 = 2 * ((scales[ib32 // 2] >> (6 * (ib32 % 2) + 0)) & 0x7) + 1
        ls2 = 2 * ((scales[ib32 // 2] >> (6 * (ib32 % 2) + 3)) & 0x7) + 1
        groups.append((ls1, ls2, tuple(rows)))
    return scale_bits, groups


def _decode_block(format_name: str, block: bytes) -> list[str]:
    """Decode the 256 values of one block to ``0x``-prefixed fp32 bit strings."""
    values: list[str] = []
    if format_name == "IQ2_XXS":
        scale_bits, groups = _iq2_xxs_terms(block)
        for magnitude, indices, sign_bytes in groups:
            db = _mul32(
                _mul32(scale_bits, _add32(_HALF_BITS, _int32(magnitude))),
                _QUARTER_BITS,
            )
            for lane in range(4):
                grid = _tables.grid_bytes(_tables.IQ2XXS_GRID[indices[lane]])
                sign = sign_bytes[lane]
                for j in range(8):
                    value = _mul32(db, _int32(grid[j]))
                    if sign & _tables.KMASK_IQ2XS[j]:
                        value = _mul32(value, _NEG_ONE_BITS)
                    values.append(_hex32(value))
    elif format_name == "IQ2_S":
        scale_bits, groups = _iq2_s_terms(block)
        for low, high, rows in groups:
            db_low = _mul32(
                _mul32(scale_bits, _add32(_HALF_BITS, _int32(low))), _QUARTER_BITS
            )
            db_high = _mul32(
                _mul32(scale_bits, _add32(_HALF_BITS, _int32(high))), _QUARTER_BITS
            )
            for lane, (index, sign) in enumerate(rows):
                db = db_low if lane < 2 else db_high
                grid = _tables.grid_bytes(_tables.IQ2S_GRID[index])
                for j in range(8):
                    value = _mul32(db, _int32(grid[j]))
                    if sign & _tables.KMASK_IQ2XS[j]:
                        value = _mul32(value, _NEG_ONE_BITS)
                    values.append(_hex32(value))
    else:
        scale_bits, groups = _iq1_m_terms(block)
        for ls1, ls2, rows in groups:
            dl1 = _mul32(scale_bits, _int32(ls1))
            dl2 = _mul32(scale_bits, _int32(ls2))
            for lane, (index, delta) in enumerate(rows):
                dl = dl1 if lane < 2 else dl2
                delta_bits = (
                    _EIGHTH_BITS | _SIGN_BIT if delta < 0 else _EIGHTH_BITS
                )
                grid = _tables.grid_signed(_tables.IQ1S_GRID[index])
                for j in range(8):
                    values.append(
                        _hex32(_mul32(dl, _add32(_int32(grid[j]), delta_bits)))
                    )
    return values


def decode(format_name: str, payload: bytes, elements: int) -> dict:
    """Decode a whole-block row of supplied IQ packed bytes to fp32 bits.

    ``payload`` must contain exactly ``elements / 256`` blocks; no trailing or
    missing bytes are tolerated.  The arithmetic is the exact separate-rounding
    binary32 contract (:data:`ARITHMETIC_ID`), matching the ggml dequantizers
    transcribed in the module docstring.  This is not a quantizer, a GGUF
    reader, a model loader, or a claim of native/RTL parity.
    """
    blocks = _validate_weight_row(format_name, payload, elements)
    _, block_bytes = _weight_size(format_name)
    values: list[str] = []
    for block in range(blocks):
        offset = block * block_bytes
        values.extend(_decode_block(format_name, payload[offset : offset + block_bytes]))
    return {
        "format": format_name,
        "elements": elements,
        "blocks": blocks,
        "fp32_bits": values,
    }


def _block_dot_iq2_xxs(
    block: bytes, q8: list[int], activation_scale: int
) -> tuple[int, list[list[int]]]:
    """One IQ2_XXS block: ``ggml_vec_dot_iq2_xxs_q8_K_generic`` (quants.c:906)."""
    scale_bits, groups = _iq2_xxs_terms(block)
    d = _mul32(scale_bits, activation_scale)
    bsum = 0
    subdots: list[list[int]] = []
    lane = 0
    for magnitude, indices, sign_bytes in groups:
        ls = 2 * magnitude + 1
        sumi = 0
        for l in range(4):
            grid = _tables.grid_bytes(_tables.IQ2XXS_GRID[indices[l]])
            sign = sign_bytes[l]
            for j in range(8):
                product = grid[j] * q8[lane + j]
                sumi += -product if sign & _tables.KMASK_IQ2XS[j] else product
            lane += 8
        bsum += sumi * ls
        subdots.append([sumi * ls])
    return _mul32(d, _int32(bsum)), subdots


def _block_dot_iq2_s(
    block: bytes, q8: list[int], activation_scale: int
) -> tuple[int, list[list[int]]]:
    """One IQ2_S block: ``ggml_vec_dot_iq2_s_q8_K_generic`` (quants.c:998)."""
    scale_bits, groups = _iq2_s_terms(block)
    d = _mul32(scale_bits, activation_scale)
    bsum = 0
    subdots: list[list[int]] = []
    lane = 0
    for low, high, rows in groups:
        ls1 = 1 + 2 * low
        ls2 = 1 + 2 * high
        sumi1 = 0
        sumi2 = 0
        for l in range(2):
            grid = _tables.grid_bytes(_tables.IQ2S_GRID[rows[l][0]])
            sign = rows[l][1]
            for j in range(8):
                product = grid[j] * q8[lane + j]
                sumi1 += -product if sign & _tables.KMASK_IQ2XS[j] else product
            lane += 8
        for l in range(2, 4):
            grid = _tables.grid_bytes(_tables.IQ2S_GRID[rows[l][0]])
            sign = rows[l][1]
            for j in range(8):
                product = grid[j] * q8[lane + j]
                sumi2 += -product if sign & _tables.KMASK_IQ2XS[j] else product
            lane += 8
        bsum += ls1 * sumi1 + ls2 * sumi2
        subdots.append([ls1 * sumi1, ls2 * sumi2])
    return _mul32(d, _int32(bsum)), subdots


def _block_dot_iq1_m(
    block: bytes, q8: list[int], activation_scale: int
) -> tuple[int, list[list[int]]]:
    """One IQ1_M block: ``ggml_vec_dot_iq1_m_q8_K_generic`` (quants.c:1193)."""
    scale_bits, groups = _iq1_m_terms(block)
    sumi1 = 0
    sumi2 = 0
    subdots: list[list[int]] = []
    lane = 0
    for ls1, ls2, rows in groups:
        sum1 = [0, 0]
        sum2 = [0, 0]
        for l in range(4):
            grid = _tables.grid_signed(_tables.IQ1S_GRID[rows[l][0]])
            lsum1 = 0
            lsum2 = 0
            for j in range(8):
                value = q8[lane + j]
                lsum1 += value * grid[j]
                lsum2 += value
            lane += 8
            sum1[l // 2] += lsum1
            sum2[l // 2] += lsum2 * rows[l][1]
        grid_total = sum1[0] * ls1 + sum1[1] * ls2
        delta_total = sum2[0] * ls1 + sum2[1] * ls2
        sumi1 += grid_total
        sumi2 += delta_total
        subdots.append([grid_total, delta_total])
    d = _mul32(scale_bits, activation_scale)
    term = _add32(_int32(sumi1), _mul32(_EIGHTH_BITS, _int32(sumi2)))
    return _mul32(d, term), subdots


def _block_dot(
    format_name: str, block: bytes, q8: list[int], activation_scale: int
) -> tuple[int, list[list[int]]]:
    if format_name == "IQ2_XXS":
        return _block_dot_iq2_xxs(block, q8, activation_scale)
    if format_name == "IQ2_S":
        return _block_dot_iq2_s(block, q8, activation_scale)
    return _block_dot_iq1_m(block, q8, activation_scale)


def dot(
    weight_format: str,
    weight_payload: bytes,
    activation_payload: bytes,
    elements: int,
    activation_format: str = "Q8_K",
) -> dict:
    """Integer-exact group dot of one IQ row against one Supplied Q8_K row.

    Integer partial sums are accumulated with Python ints, so they are exact and
    independent of any FP32 rounding; each 32-value group reports the terms
    named by ``FORMATS[weight_format]["integer_subdot_fields"]``:

    * ``IQ2_XXS``: one term ``(2q+1) * sum_i grid_i * q8_i`` per group;
    * ``IQ2_S``: ``ls1*sumi1`` and ``ls2*sumi2`` for the two 16-value halves;
    * ``IQ1_M``: ``grid`` and ``delta`` terms (``sumi1`` and ``sumi2``).

    The FP32 result reproduces the generic ggml CPU kernel expressions exactly,
    with one round-to-nearest-even per binary32 multiply/add, in this order:

    * ``IQ2_XXS`` (quants.c:906-946): ``d = fp16(x.d) * y.d``;
      ``sumf += d * bsum`` per block; ``s = 0.125f * sumf``;
    * ``IQ2_S`` (quants.c:998-1048): same shape, ``bsum = ls1*sumi1 + ls2*sumi2``;
    * ``IQ1_M`` (quants.c:1193-1252): no block ``d`` field; ``d = fp16`` rebuilt
      from the top scale nibbles times ``y.d``;
      ``sumf += d * (sumi1 + 0.125f*sumi2)`` and ``s = sumf`` (no final 0.125).

    ``Q8_K`` ``bsums`` are not read, exactly like the reference kernels.  The
    float path is kept separate from :func:`decode`; tests cross-check both on
    random data with a small tolerance.  This does not claim native SIMD, RTL,
    FPGA, or model parity.
    """
    if type(activation_format) is not str or activation_format not in _ACTIVATION_SIZES:
        raise FormatError(
            "unsupported_activation_format",
            "the only IQ activation format is Q8_K",
        )
    weight_blocks = _validate_weight_row(weight_format, weight_payload, elements)
    _validate_activation_row(activation_format, activation_payload, elements)
    _, block_bytes = _weight_size(weight_format)
    activations = _activate(activation_format, activation_payload, weight_blocks)

    row = 0  # +0; f32 accumulation starts from positive zero.
    integer_subdots: list[list[list[int]]] = []
    for block in range(weight_blocks):
        offset = block * block_bytes
        activation_scale, q8 = activations[block]
        contribution, subdots = _block_dot(
            weight_format, weight_payload[offset : offset + block_bytes], q8, activation_scale
        )
        integer_subdots.append(subdots)
        row = _add32(row, contribution)
    if weight_format != "IQ1_M":
        row = _mul32(_EIGHTH_BITS, row)
    return {
        "weight_format": weight_format,
        "activation_format": activation_format,
        "elements": elements,
        "weight_blocks": weight_blocks,
        "activation_blocks": weight_blocks,
        "integer_subdots": integer_subdots,
        "fp32_bits": _hex32(row),
        "arithmetic_id": ARITHMETIC_ID,
    }
