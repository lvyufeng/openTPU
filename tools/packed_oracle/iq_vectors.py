# SPDX-License-Identifier: MIT
"""Independent synthetic goldens for the IQ2_XXS/IQ2_S/IQ1_M packed oracle.

This module is the *independent* half of the IQ packed-golden oracle.  Its
``_reference_*`` functions are a second, deliberately straightforward
transcription of the ggml block layouts and the three integer group-dot
kernels.  They do not import, call, or share term-extraction helpers with
``tools/packed_oracle/iq_formats.py`` (the production path under test); the two
implementations were written separately from the same pinned sources and the
README records which file is which.  Shared inputs are limited to the pure
data module ``iq_tables.py`` (codebook bytes copied verbatim from upstream).

Reference arithmetic
    Every binary32/binary16 value is carried as an exact ``(negative, Fraction)``
    pair and each individual C ``float`` operation is rounded separately with
    round-to-nearest-even, signed zero, subnormals and overflow detection.
    Multiplicands are exact, so repeated ``Fraction`` products cannot pick up
    the rounding the DUT is supposed to perform.

Synthetic data
    Every payload is generated from a seeded splitmix64 stream over *raw block
    bytes* (never from real model weights).  Q8_K activations are generated from
    the documented synthetic patterns (int8 extremes, zeros, all-equal,
    alternating, ramp, random) with synthetic block scales.

Fixture contract
    ``generate_fixture()`` emits the canonical ``data/iq_synthetic.json``
    document; ``generate_rtl_bundle()`` emits the smaller
    ``data/iq_rtl_vectors.json`` bundle for an RTL integer datapath (raw block
    hex plus expected per-32-group integer partial sums).  Both documents are
    deterministic, carry a ``payload_sha256`` over their canonical JSON without
    the digest field, and contain no device, model, native or RTL results.
"""

import hashlib
import json
import re
from fractions import Fraction

from . import iq_tables as _tables
from .formats import ARITHMETIC_ID, MAX_ELEMENTS, FormatError

FIXTURE_ID = "iq-packed-supplied-v1"
RTL_FIXTURE_ID = "iq-packed-rtl-integer-v1"
MAX_CASES = 64
MAX_FIXTURE_BYTES = 8 * 1024 * 1024
PACKED_BLOCK_ELEMENTS = 256

# Frozen copies of the public metadata contract.  Expected values in the
# fixture are produced by the reference functions below, never by the DUT.
_IQ2_XXS = {
    "ggml_type": 16,
    "block_bytes": 66,
    "block_elements": 256,
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
    "delta_flags_per_group": 0,
    "block_scale": "fp16",
    "integer_subdot_fields": ["grid"],
    "final_scale_fp32": "0x3e000000",
    "layout": [
        {"field": "d", "offset": 0, "bytes": 2},
        {"field": "qs", "offset": 2, "bytes": 64},
    ],
}
_IQ2_S = {
    "ggml_type": 22,
    "block_bytes": 82,
    "block_elements": 256,
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
    "delta_flags_per_group": 0,
    "block_scale": "fp16",
    "integer_subdot_fields": ["low16", "high16"],
    "final_scale_fp32": "0x3e000000",
    "layout": [
        {"field": "d", "offset": 0, "bytes": 2},
        {"field": "qs", "offset": 2, "bytes": 64},
        {"field": "qh", "offset": 66, "bytes": 8},
        {"field": "scales", "offset": 74, "bytes": 8},
    ],
}
_IQ1_M = {
    "ggml_type": 29,
    "block_bytes": 56,
    "block_elements": 256,
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
    "delta_flags_per_group": 4,
    "block_scale": "fp16-packed",
    "integer_subdot_fields": ["grid", "delta"],
    "final_scale_fp32": None,
    "layout": [
        {"field": "qs", "offset": 0, "bytes": 32},
        {"field": "qh", "offset": 32, "bytes": 16},
        {"field": "scales", "offset": 48, "bytes": 8},
    ],
}
_EXPECTED_FORMATS = {"IQ2_XXS": _IQ2_XXS, "IQ2_S": _IQ2_S, "IQ1_M": _IQ1_M}

_Q8_K = {
    "block_bytes": 292,
    "block_elements": 256,
    "block_scale": "fp32",
    "subblock_elements": 16,
    "layout": [
        {"field": "d", "offset": 0, "bytes": 4},
        {"field": "qs", "offset": 4, "bytes": 256},
        {"field": "bsums", "offset": 260, "bytes": 32},
    ],
}
_EXPECTED_ACTIVATIONS = {"Q8_K": _Q8_K}

_WEIGHT_ORDER = ("IQ2_XXS", "IQ2_S", "IQ1_M")

_EXPECTED_TABLE_SHA256 = {
    "kmask_iq2xs": "5ac9831b2e30eb285ef34f8501620f878432d5c04331ad1ae47f977a83ba41a5",
    "ksigns_iq2xs": "a76742a603f8beca5212ecce0f1f02f11a4887fd2f7c8b15aca0ea3eb3380c31",
    "iq2xxs_grid": "05826b5d3e472a3a78f196be62ac78acf81df0f909626e12ab9fa2a5d490dd54",
    "iq2s_grid": "e1aa1473412b0552c2174c30ef22ab4073f6a181b85a17056e8249bd2932fd88",
    "iq1s_grid": "07540ffc1aeaf6ad4d97e96b0fcc765aae39671d4ae4a27bbd0e796fde167c6a",
}

_SCOPE_EXCLUSIONS = [
    "real model weights or tokenizer data",
    "native ggml/Strata/Metal execution",
    "RTL simulation, synthesis, or on-device results",
    "quantizer round-trip or perplexity evaluation",
    "timing, throughput, or hardware performance numbers",
]

# Frozen provenance.  Tests compare the fixture field against these literals;
# nothing here is derived from a DUT call.
_HEADER = "engine/include/strata/quant/strata-quant-common.h"
_GGML_COMMON = "engine/third_party/strata-llama/ggml/src/ggml-common.h"
_GGML_QUANTS_C = (
    "engine/third_party/strata-llama/ggml/src/ggml-cpu/quants.c"
)
_GGML_DEQUANT_C = "engine/third_party/strata-llama/ggml/src/ggml-quants.c"

_EXPECTED_PROVENANCE = {
    "strata": {
        "repository": "https://github.com/lvyufeng/Strata",
        "revision": "3142b66b96e9c0e775e30f620b11b191a1427c2a",
        "header": _HEADER,
        "header_sha256": "0709a7a826c1c8c73d814b8362818b6108d37c119257b97df9824b3053d08a1a",
        "anchors": [
            {"path": _HEADER, "start_line": 376, "end_line": 376},
            {"path": _HEADER, "start_line": 393, "end_line": 393},
            {"path": _HEADER, "start_line": 427, "end_line": 427},
            {"path": _HEADER, "start_line": 434, "end_line": 434},
            {"path": _HEADER, "start_line": 499, "end_line": 513},
            {"path": _HEADER, "start_line": 552, "end_line": 617},
            {"path": _HEADER, "start_line": 752, "end_line": 1009},
            {"path": _HEADER, "start_line": 1128, "end_line": 1133},
            {"path": _HEADER, "start_line": 1646, "end_line": 1646},
        ],
    },
    "ggml": {
        "repository": "https://github.com/ggml-org/llama.cpp",
        "revision": "3cf03257f219afbe7334045ff7c6a06ac68c627d",
        "header": _GGML_COMMON,
        "header_sha256": "0061131b615c5721fc88a78feeb22c1f8c450f1c2646a317d80796a653bf595c",
        "anchors": [
            {"path": _GGML_COMMON, "start_line": 1323, "end_line": 1337},
            {"path": _GGML_COMMON, "start_line": 1376, "end_line": 1441},
            {"path": _GGML_COMMON, "start_line": 1576, "end_line": 1833},
            {"path": _GGML_COMMON, "start_line": 1957, "end_line": 2470},
            {"path": _GGML_QUANTS_C, "start_line": 906, "end_line": 946},
            {"path": _GGML_QUANTS_C, "start_line": 998, "end_line": 1048},
            {"path": _GGML_QUANTS_C, "start_line": 1193, "end_line": 1252},
            {"path": _GGML_DEQUANT_C, "start_line": 2488, "end_line": 2512},
            {"path": _GGML_DEQUANT_C, "start_line": 2543, "end_line": 2577},
            {"path": _GGML_DEQUANT_C, "start_line": 2675, "end_line": 2714},
        ],
    },
    "opentpu_base": {
        "repository": "https://github.com/lvyufeng/openTPU",
        "revision": "d84db55712aba0d40811d7762cb052f05d55b7e5",
    },
    "tables": {
        "module": "tools/packed_oracle/iq_tables.py",
        "license": "MIT",
        "data_not_code": True,
        "sha256": _EXPECTED_TABLE_SHA256,
        "excluded": "iq1s_grid_gpu (GPU-only value+1 layout is not reference semantics)",
    },
    "generator": {
        "id": "iq-fraction-integer-reference",
        "version": 1,
        "production_path_under_test": "tools/packed_oracle/iq_formats.py",
        "independent_reference": "tools/packed_oracle/iq_vectors.py",
    },
    "expected_derivation": (
        "independent exact-rational binary32 RNE transcription of pinned ggml "
        "dequant and integer group-dot kernels; not a DUT output"
    ),
    "scope_exclusions": _SCOPE_EXCLUSIONS,
}

_TOP_KEYS = {
    "schema_version",
    "fixture_id",
    "synthetic",
    "arithmetic_id",
    "formats",
    "activations",
    "provenance",
    "cases",
    "payload_sha256",
}
_DECODE_KEYS = {"id", "operation", "format", "elements", "payload_hex", "expected_fp32_bits"}
_DOT_KEYS = {
    "id",
    "operation",
    "weight_format",
    "activation_format",
    "elements",
    "weight_hex",
    "activation_hex",
    "expected_fp32_bits",
    "expected_integer_subdots",
}
_RTL_TOP_KEYS = {
    "schema_version",
    "fixture_id",
    "synthetic",
    "arithmetic_id",
    "formats",
    "activations",
    "provenance",
    "cases",
    "payload_sha256",
}
_RTL_CASE_KEYS = {
    "id",
    "weight_format",
    "activation_format",
    "elements",
    "weight_hex",
    "activation_hex",
    "integer_subdot_fields",
    "expected_integer_subdots",
    "expected_fp32_bits",
}
_BITS_PATTERN = re.compile(r"0x[0-9a-f]{8}", re.ASCII)
_ID_PATTERN = re.compile(r"[a-z0-9][a-z0-9-]{0,63}", re.ASCII)
_HEX_PATTERN = re.compile(r"[0-9a-f]+", re.ASCII)
_SHA_PATTERN = re.compile(r"[0-9a-f]{64}", re.ASCII)

_HALF_ONE = 0x3C00
_HALF_NEG_ONE = 0xBC00
_HALF_MIN_SUBNORMAL = 0x0001
_HALF_MAX_FINITE = 0x7BFF
_F32_ONE = 0x3F800000
_F32_NEG_ONE = 0xBF800000
_F32_MIN_SUBNORMAL = 0x00000001
_F32_MAX_FINITE = 0x7F7FFFFF
_MASK64 = (1 << 64) - 1

# 2**0, 2**-1 ... as exact pairs are built lazily by _pow2.
_ZERO = (False, Fraction(0))
_EIGHTH = (False, Fraction(1, 8))
_QUARTER = (False, Fraction(1, 4))
_HALF = (False, Fraction(1, 2))


def _fail(code: str, message: str):
    raise FormatError(code, message)


def _pow2(exponent: int) -> Fraction:
    if exponent >= 0:
        return Fraction(1 << exponent)
    return Fraction(1, 1 << -exponent)


# ---------------------------------------------------------------------------
# Exact binary32 / binary16 reference arithmetic
# ---------------------------------------------------------------------------


def _round_ratio(numerator: int, denominator: int) -> int:
    quotient, remainder = divmod(numerator, denominator)
    twice = 2 * remainder
    if twice > denominator or (twice == denominator and quotient & 1):
        quotient += 1
    return quotient


def _round_scale(numerator: int, denominator: int, shift: int) -> int:
    if shift >= 0:
        return _round_ratio(numerator << shift, denominator)
    return _round_ratio(numerator, denominator << -shift)


def _at_least(num: int, den: int, exponent: int) -> bool:
    if exponent >= 0:
        return num >= den << exponent
    return num << -exponent >= den


def _magnitude_to_bits(value: Fraction) -> int:
    """Nonnegative exact rational -> positive binary32 bits (RNE, no rounding to inf)."""
    if value < 0:
        raise ValueError("magnitude must be nonnegative")
    if value == 0:
        return 0
    num, den = value.numerator, value.denominator
    exponent = num.bit_length() - den.bit_length()
    while not _at_least(num, den, exponent):
        exponent -= 1
    while _at_least(num, den, exponent + 1):
        exponent += 1
    if exponent < -126:
        mantissa = _round_scale(num, den, 149)
        if mantissa == 0:
            return 0
        if mantissa >= 1 << 23:
            return 0x00800000
        return mantissa
    significand = _round_scale(num, den, 23 - exponent)
    if significand == 1 << 24:
        significand >>= 1
        exponent += 1
    if exponent > 127:
        raise ValueError("binary32 overflow")
    return ((exponent + 127) << 23) | (significand - (1 << 23))


def _pack_f32(negative: bool, magnitude: Fraction) -> int:
    bits = _magnitude_to_bits(magnitude)
    if negative:
        bits |= 0x80000000
    return bits


def _unpack_f32(bits: int) -> tuple[bool, Fraction]:
    exponent = (bits >> 23) & 0xFF
    fraction = bits & 0x7FFFFF
    if exponent == 0xFF:
        raise ValueError("binary32 payload is not finite")
    if exponent == 0:
        magnitude = Fraction(fraction, 1 << 149)
    else:
        magnitude = Fraction((1 << 23) + fraction) * _pow2(exponent - 150)
    return (bool(bits >> 31), magnitude)


def _unpack_f16(raw: int) -> tuple[bool, Fraction]:
    raw &= 0xFFFF
    exponent = (raw >> 10) & 0x1F
    fraction = raw & 0x3FF
    if exponent == 0x1F:
        raise ValueError("binary16 payload is not finite")
    if exponent == 0:
        magnitude = Fraction(fraction, 1 << 24)
    else:
        magnitude = Fraction(1024 + fraction) * _pow2(exponent - 25)
    return (bool(raw >> 15), magnitude)


def _pair_bits(pair) -> int:
    return _pack_f32(pair[0], pair[1])


def _pair_value(pair) -> Fraction:
    return -pair[1] if pair[0] else pair[1]


def _int_pair(value: int):
    negative = value < 0
    return (negative, Fraction(-value if negative else value))


def _add(a, b):
    total = _pair_value(a) + _pair_value(b)
    negative = total < 0 or (total == 0 and a[0] and b[0])
    return (negative, -total if total < 0 else total)


def _mul(a, b):
    negative = a[0] != b[0]
    product = a[1] * b[1]
    if product == 0:
        return (negative, Fraction(0))
    bits = _pack_f32(negative, product)
    return _unpack_f32(bits)


def _neg(pair):
    return (not pair[0], pair[1])


def _hex32(bits: int) -> str:
    return "0x%08x" % bits


# ---------------------------------------------------------------------------
# Raw little-endian accessors shared by both sides of the fixture schema
# ---------------------------------------------------------------------------


def _u16(data: bytes, offset: int) -> int:
    return data[offset] | data[offset + 1] << 8


def _u32(data: bytes, offset: int) -> int:
    return (
        data[offset]
        | data[offset + 1] << 8
        | data[offset + 2] << 16
        | data[offset + 3] << 24
    )


def _grid_unsigned(word: int) -> tuple[int, ...]:
    return tuple((word >> (8 * lane)) & 0xFF for lane in range(8))


def _grid_signed(word: int) -> tuple[int, ...]:
    unsigned = _grid_unsigned(word)
    return tuple(byte - 256 if byte >= 128 else byte for byte in unsigned)


def _check_constants() -> None:
    for name, digest in _EXPECTED_TABLE_SHA256.items():
        if _tables.IQ_TABLE_SHA256.get(name) != digest:
            raise AssertionError("codebook digest mismatch for " + name)
    if len(_tables.IQ2XXS_GRID) != 256 or len(_tables.IQ2S_GRID) != 1024:
        raise AssertionError("IQ2 codebook sizes changed")
    if len(_tables.IQ1S_GRID) != _tables.NGRID_IQ1S or _tables.NGRID_IQ1S != 2048:
        raise AssertionError("IQ1 codebook sizes changed")
    if tuple(_tables.KMASK_IQ2XS) != tuple(1 << lane for lane in range(8)):
        raise AssertionError("IQ2 sign mask changed")
    if len(_tables.KSIGNS_IQ2XS) != 128:
        raise AssertionError("IQ2 sign table changed")
    if _grid_signed(_tables.IQ1S_GRID[1]) != (1, -1, -1, -1, -1, -1, -1, -1):
        raise AssertionError("IQ1 grid interpretation changed")


_check_constants()


# ---------------------------------------------------------------------------
# Independent block transcription: raw bytes -> per-32-group integer terms
# ---------------------------------------------------------------------------


def _xxs_terms(block: bytes):
    """IQ2_XXS: ggml-quants.c:2488-2512 and quants.c:906-946."""
    scale = _u16(block, 0)
    groups = []
    for group in range(8):
        base = 2 + 8 * group
        indices = [block[base + lane] for lane in range(4)]
        aux = _u32(block, base + 4)
        signs = [(aux >> (7 * lane)) & 0x7F for lane in range(4)]
        magnitude = (aux >> 28) & 0xF
        groups.append((magnitude, indices, signs))
    return scale, groups


def _iq2_s_terms(block: bytes):
    """IQ2_S: ggml-quants.c:2543-2577 and quants.c:998-1048."""
    scale = _u16(block, 0)
    qs = block[2:66]
    qh = block[66:74]
    scales = block[74:82]
    groups = []
    for group in range(8):
        low = scales[group] & 0xF
        high = scales[group] >> 4
        rows = []
        for lane in range(4):
            index = qs[4 * group + lane] | ((qh[group] << (8 - 2 * lane)) & 0x300)
            rows.append((index, qs[32 + 4 * group + lane]))
        groups.append((low, high, rows))
    return scale, groups


def _iq1_m_terms(block: bytes):
    """IQ1_M: ggml-quants.c:2675-2714 and quants.c:1193-1252."""
    words = [_u16(block, 48 + 2 * index) for index in range(4)]
    packed = (
        (words[0] >> 12)
        | ((words[1] >> 8) & 0x00F0)
        | ((words[2] >> 4) & 0x0F00)
        | (words[3] & 0xF000)
    )
    qs = block[0:32]
    qh = block[32:48]
    groups = []
    for group in range(8):
        lanes = []
        for lane in range(4):
            byte = qh[2 * group + lane // 2]
            index = qs[4 * group + lane] | ((byte << (8 - 4 * (lane % 2))) & 0x700)
            flag = 0x08 if lane % 2 == 0 else 0x80
            delta = -1 if byte & flag else 1
            lanes.append((index, delta))
        word = words[group // 2]
        shift = 6 * (group % 2)
        low_scale = 2 * ((word >> (shift + 0)) & 0x7) + 1
        high_scale = 2 * ((word >> (shift + 3)) & 0x7) + 1
        groups.append((low_scale, high_scale, lanes))
    return packed, groups


# ---------------------------------------------------------------------------
# Independent decode to binary32 bits
# ---------------------------------------------------------------------------


def _block_values(name: str, block: bytes):
    if name == "IQ2_XXS":
        scale_raw, groups = _xxs_terms(block)
        d = _unpack_f16(scale_raw)
        values = []
        for magnitude, indices, signs in groups:
            db = _mul(_mul(d, _add(_HALF, _int_pair(magnitude))), _QUARTER)
            for lane, index in enumerate(indices):
                table_signs = _tables.KSIGNS_IQ2XS[signs[lane]]
                grid = _grid_unsigned(_tables.IQ2XXS_GRID[index])
                for column in range(8):
                    value = _mul(db, _int_pair(grid[column]))
                    if table_signs & _tables.KMASK_IQ2XS[column]:
                        value = _neg(value)
                    values.append(value)
        return values
    if name == "IQ2_S":
        scale_raw, groups = _iq2_s_terms(block)
        d = _unpack_f16(scale_raw)
        values = []
        for low, high, rows in groups:
            low_db = _mul(_mul(d, _add(_HALF, _int_pair(low))), _QUARTER)
            high_db = _mul(_mul(d, _add(_HALF, _int_pair(high))), _QUARTER)
            for lane, (index, sign_byte) in enumerate(rows):
                db = low_db if lane < 2 else high_db
                grid = _grid_unsigned(_tables.IQ2S_GRID[index])
                for column in range(8):
                    value = _mul(db, _int_pair(grid[column]))
                    if sign_byte & _tables.KMASK_IQ2XS[column]:
                        value = _neg(value)
                    values.append(value)
        return values
    if name == "IQ1_M":
        packed, groups = _iq1_m_terms(block)
        d = _unpack_f16(packed)
        values = []
        for low_scale, high_scale, lanes in groups:
            low_dl = _mul(d, _int_pair(low_scale))
            high_dl = _mul(d, _int_pair(high_scale))
            for lane, (index, delta) in enumerate(lanes):
                dl = low_dl if lane < 2 else high_dl
                grid = _grid_signed(_tables.IQ1S_GRID[index])
                delta_pair = (True, Fraction(1, 8)) if delta < 0 else _EIGHTH
                for column in range(8):
                    values.append(_mul(dl, _add(_int_pair(grid[column]), delta_pair)))
        return values
    raise ValueError("unsupported packed format: " + name)


def _validate_weight_payload(name: str, payload: bytes) -> None:
    metadata = _EXPECTED_FORMATS[name]
    stride = metadata["block_bytes"]
    if not isinstance(payload, (bytes, bytearray)) or len(payload) == 0:
        raise ValueError("weight payload must be nonempty bytes")
    if len(payload) % stride:
        raise ValueError("weight payload is not a whole number of %s blocks" % name)
    for start in range(0, len(payload), stride):
        block = bytes(payload[start : start + stride])
        if name == "IQ1_M":
            raw = _iq1_m_terms(block)[0]
        else:
            raw = _u16(block, 0)
        if (raw >> 10) & 0x1F == 0x1F:
            raise ValueError("nonfinite binary16 block scale")


def _validate_activation_payload(payload: bytes) -> None:
    stride = _Q8_K["block_bytes"]
    if not isinstance(payload, (bytes, bytearray)) or len(payload) == 0:
        raise ValueError("activation payload must be nonempty bytes")
    if len(payload) % stride:
        raise ValueError("activation payload is not a whole number of Q8_K blocks")
    for start in range(0, len(payload), stride):
        if (_u32(payload, start) >> 23) & 0xFF == 0xFF:
            raise ValueError("nonfinite binary32 activation scale")


def _reference_decode(name: str, payload: bytes) -> list[str]:
    """Second transcription: packed block bytes -> expected binary32 bit strings."""
    _validate_weight_payload(name, payload)
    stride = _EXPECTED_FORMATS[name]["block_bytes"]
    bits = []
    for start in range(0, len(payload), stride):
        block = bytes(payload[start : start + stride])
        bits.extend(_hex32(_pair_bits(value)) for value in _block_values(name, block))
    return bits


# ---------------------------------------------------------------------------
# Independent integer group-dot transcription
# ---------------------------------------------------------------------------


def _q8_block(block: bytes):
    scale = _u32(block, 0)
    if (scale >> 23) & 0xFF == 0xFF:
        raise ValueError("nonfinite binary32 activation scale")
    values = [byte - 256 if byte >= 128 else byte for byte in block[4:260]]
    return scale, values


def _reference_dot(name: str, weight: bytes, activation: bytes):
    """Second transcription: integer group partial sums and a separate f32 combination.

    Mirrors exactly what the ggml kernels accumulate as integers (scaled integer
    codebook values times int8 activations) and then combines with the block
    scales using separate binary32 multiplies and adds.
    """
    _validate_weight_payload(name, weight)
    _validate_activation_payload(activation)
    weight_stride = _EXPECTED_FORMATS[name]["block_bytes"]
    activation_stride = _Q8_K["block_bytes"]
    if len(weight) // weight_stride != len(activation) // activation_stride:
        raise ValueError("weight and activation block counts differ")
    row = _ZERO
    subdots = []
    for index in range(len(weight) // weight_stride):
        wblock = bytes(weight[index * weight_stride : (index + 1) * weight_stride])
        ablock = bytes(activation[index * activation_stride : (index + 1) * activation_stride])
        activation_scale_bits, q8 = _q8_block(ablock)
        activation_scale = _unpack_f32(activation_scale_bits)
        block_subdots = []
        if name == "IQ2_XXS":
            scale_raw, groups = _xxs_terms(wblock)
            weight_scale = _unpack_f16(scale_raw)
            total = 0
            for group, (magnitude, indices, signs) in enumerate(groups):
                sumi = 0
                for lane, grid_index in enumerate(indices):
                    sign_bits = _tables.KSIGNS_IQ2XS[signs[lane]]
                    grid = _grid_unsigned(_tables.IQ2XXS_GRID[grid_index])
                    for column in range(8):
                        product = grid[column] * q8[group * 32 + lane * 8 + column]
                        sumi += -product if sign_bits & _tables.KMASK_IQ2XS[column] else product
                local = sumi * (2 * magnitude + 1)
                total += local
                block_subdots.append([local])
            scale = _mul(weight_scale, activation_scale)
            row = _add(row, _mul(scale, _int_pair(total)))
        elif name == "IQ2_S":
            scale_raw, groups = _iq2_s_terms(wblock)
            weight_scale = _unpack_f16(scale_raw)
            total = 0
            for group, (low, high, rows) in enumerate(groups):
                low_sumi = 0
                high_sumi = 0
                for lane, (grid_index, sign_byte) in enumerate(rows):
                    grid = _grid_unsigned(_tables.IQ2S_GRID[grid_index])
                    sumi = 0
                    for column in range(8):
                        product = grid[column] * q8[group * 32 + lane * 8 + column]
                        sumi += -product if sign_byte & _tables.KMASK_IQ2XS[column] else product
                    if lane < 2:
                        low_sumi += sumi
                    else:
                        high_sumi += sumi
                low_local = (2 * low + 1) * low_sumi
                high_local = (2 * high + 1) * high_sumi
                total += low_local + high_local
                block_subdots.append([low_local, high_local])
            scale = _mul(weight_scale, activation_scale)
            row = _add(row, _mul(scale, _int_pair(total)))
        elif name == "IQ1_M":
            packed, groups = _iq1_m_terms(wblock)
            weight_scale = _unpack_f16(packed)
            grid_total_all = 0
            delta_total_all = 0
            for group, (low_scale, high_scale, lanes) in enumerate(groups):
                grid_sum = [0, 0]
                delta_sum = [0, 0]
                for lane, (grid_index, delta) in enumerate(lanes):
                    grid = _grid_signed(_tables.IQ1S_GRID[grid_index])
                    grid_lane = 0
                    raw_lane = 0
                    for column in range(8):
                        q8_value = q8[group * 32 + lane * 8 + column]
                        grid_lane += q8_value * grid[column]
                        raw_lane += q8_value
                    grid_sum[lane // 2] += grid_lane
                    delta_sum[lane // 2] += raw_lane * delta
                grid_local = grid_sum[0] * low_scale + grid_sum[1] * high_scale
                delta_local = delta_sum[0] * low_scale + delta_sum[1] * high_scale
                grid_total_all += grid_local
                delta_total_all += delta_local
                block_subdots.append([grid_local, delta_local])
            combined = _add(_int_pair(grid_total_all), _mul(_EIGHTH, _int_pair(delta_total_all)))
            scale = _mul(weight_scale, activation_scale)
            row = _add(row, _mul(scale, combined))
        else:
            raise ValueError("unsupported packed format: " + name)
        subdots.append(block_subdots)
    if name != "IQ1_M":
        row = _mul(_EIGHTH, row)
    return _hex32(_pair_bits(row)), subdots


def _reference_decoded_dot(name: str, weight: bytes, activation: bytes) -> str:
    """Float reference path: decode both operands, then accumulate float products.

    This deliberately shares no algebra with the integer group-dot kernel; it is
    only used for a small-tolerance cross-check against the decoded float values.
    """
    _validate_weight_payload(name, weight)
    _validate_activation_payload(activation)
    weight_stride = _EXPECTED_FORMATS[name]["block_bytes"]
    activation_stride = _Q8_K["block_bytes"]
    if len(weight) // weight_stride != len(activation) // activation_stride:
        raise ValueError("weight and activation block counts differ")
    row = _ZERO
    for index in range(len(weight) // weight_stride):
        wblock = bytes(weight[index * weight_stride : (index + 1) * weight_stride])
        ablock = bytes(activation[index * activation_stride : (index + 1) * activation_stride])
        activation_scale_bits, q8 = _q8_block(ablock)
        activation_scale = _unpack_f32(activation_scale_bits)
        weights = _block_values(name, wblock)
        for lane in range(256):
            term = _mul(_mul(weights[lane], _int_pair(q8[lane])), activation_scale)
            row = _add(row, term)
    return _hex32(_pair_bits(row))


# ---------------------------------------------------------------------------
# Seeded synthetic generator over raw block bytes
# ---------------------------------------------------------------------------


class _Rng:
    """splitmix64; deterministic, dependency-free, and independent of Python's RNG."""

    def __init__(self, seed: int):
        self.state = seed & _MASK64

    def next(self) -> int:
        self.state = (self.state + 0x9E3779B97F4A7C15) & _MASK64
        value = self.state
        value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _MASK64
        value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _MASK64
        return (value ^ (value >> 31)) & _MASK64

    def below(self, bound: int) -> int:
        return self.next() % bound

    def byte(self) -> int:
        return self.below(256)


def _finite_f16(raw: int) -> int:
    raw &= 0xFFFF
    if (raw >> 10) & 0x1F == 0x1F:
        raw = (raw & 0x8000) | _HALF_MAX_FINITE
    return raw


def _finite_f32(raw: int) -> int:
    raw &= 0xFFFFFFFF
    if (raw >> 23) & 0xFF == 0xFF:
        raw = (raw & 0x80000000) | _F32_MAX_FINITE
    return raw


_HALF_PATTERNS = (
    "one",
    "negative-one",
    "min-subnormal",
    "max-finite",
    "random",
    "random-negative",
)


def _half_scale(pattern: str, rng: _Rng) -> int:
    if pattern == "one":
        return _HALF_ONE
    if pattern == "negative-one":
        return _HALF_NEG_ONE
    if pattern == "min-subnormal":
        return _HALF_MIN_SUBNORMAL
    if pattern == "max-finite":
        return _HALF_MAX_FINITE
    if pattern == "random":
        return _finite_f16(rng.below(0x7C00))
    if pattern == "random-negative":
        return _finite_f16(0x8000 | rng.below(0x7C00))
    raise ValueError("unknown half scale pattern: " + pattern)


_F32_PATTERNS = (
    "one",
    "negative-one",
    "half",
    "eighth",
    "large",
    "tiny",
    "random",
    "random-negative",
)


def _f32_scale(pattern: str, rng: _Rng) -> int:
    if pattern == "one":
        return _F32_ONE
    if pattern == "negative-one":
        return _F32_NEG_ONE
    if pattern == "half":
        return 0x3F000000
    if pattern == "eighth":
        return 0x3E000000
    if pattern == "large":
        return 0x42800000  # 64.0f
    if pattern == "tiny":
        return 0x35800000  # 2**-20
    if pattern == "random":
        return _finite_f32(rng.below(0x7F800000))
    if pattern == "random-negative":
        return _finite_f32(0x80000000 | rng.below(0x7F800000))
    raise ValueError("unknown binary32 scale pattern: " + pattern)


def _pack_iq2_xxs(scale: int, magnitudes, indices, sign_indices) -> bytes:
    out = bytearray(66)
    out[0] = scale & 0xFF
    out[1] = (scale >> 8) & 0xFF
    for group in range(8):
        base = 2 + 8 * group
        for lane in range(4):
            out[base + lane] = indices[4 * group + lane] & 0xFF
        aux = (magnitudes[group] & 0xF) << 28
        for lane in range(4):
            aux |= (sign_indices[4 * group + lane] & 0x7F) << (7 * lane)
        for byte in range(4):
            out[base + 4 + byte] = (aux >> (8 * byte)) & 0xFF
    return bytes(out)


def _pack_iq2_s(scale: int, low_scales, high_scales, index_low, index_high, sign_indices) -> bytes:
    out = bytearray(82)
    out[0] = scale & 0xFF
    out[1] = (scale >> 8) & 0xFF
    for group in range(8):
        qh = 0
        for lane in range(4):
            out[2 + 4 * group + lane] = index_low[4 * group + lane] & 0xFF
            qh |= (index_high[4 * group + lane] & 0x3) << (2 * lane)
            out[2 + 32 + 4 * group + lane] = sign_indices[4 * group + lane] & 0xFF
        out[66 + group] = qh
        out[74 + group] = (low_scales[group] & 0xF) | ((high_scales[group] & 0xF) << 4)
    return bytes(out)


def _pack_iq1_m(scale: int, index_low, index_high, delta_signs, group_fields) -> bytes:
    out = bytearray(56)
    qh = bytearray(16)
    for lane in range(32):
        out[lane] = index_low[lane] & 0xFF
    for group in range(8):
        for lane in range(4):
            slot = 2 * group + lane // 2
            high = index_high[4 * group + lane] & 0x7
            if lane % 2 == 0:
                qh[slot] |= high
                if delta_signs[4 * group + lane] < 0:
                    qh[slot] |= 0x08
            else:
                qh[slot] |= high << 4
                if delta_signs[4 * group + lane] < 0:
                    qh[slot] |= 0x80
    out[32:48] = qh
    # Inverse of the packed-half reconstruction in iq_tables/ggml:
    # packed = (w0 >> 12) | ((w1 >> 8) & 0x00f0) | ((w2 >> 4) & 0x0f00) | (w3 & 0xf000)
    words = [
        (scale & 0xF) << 12,
        ((scale >> 4) & 0xF) << 12,
        ((scale >> 8) & 0xF) << 12,
        scale & 0xF000,
    ]
    for group in range(8):
        word_index = group // 2
        shift = 6 * (group % 2)
        words[word_index] |= (group_fields[group][0] & 0x7) << shift
        words[word_index] |= (group_fields[group][1] & 0x7) << (shift + 3)
    for index in range(4):
        out[48 + 2 * index] = words[index] & 0xFF
        out[49 + 2 * index] = (words[index] >> 8) & 0xFF
    return bytes(out)


_Q8_PATTERNS = ("extremes", "zeros", "equal-positive", "equal-negative", "alternating", "ramp", "random")


def _q8_values(pattern: str, rng: _Rng) -> list[int]:
    if pattern == "extremes":
        cycle = (-128, 127, 0, -1, 1, 64, -64, -127, 126, 2, -2, 100, -100, 3, -3, 127)
        return [cycle[index % len(cycle)] for index in range(256)]
    if pattern == "zeros":
        return [0] * 256
    if pattern == "equal-positive":
        return [127] * 256
    if pattern == "equal-negative":
        return [-128] * 256
    if pattern == "alternating":
        return [127 if index % 2 == 0 else -128 for index in range(256)]
    if pattern == "ramp":
        return [(index * 37 + 11) % 256 - 128 for index in range(256)]
    if pattern == "random":
        return [rng.byte() - 128 for _ in range(256)]
    raise ValueError("unknown Q8_K pattern: " + pattern)


def _pack_q8_k(scale: int, values) -> bytes:
    out = bytearray(292)
    out[0:4] = scale.to_bytes(4, "little")
    for index, value in enumerate(values):
        out[4 + index] = value & 0xFF
    for sub_block in range(16):
        total = sum(values[16 * sub_block : 16 * sub_block + 16])
        out[260 + 2 * sub_block : 262 + 2 * sub_block] = total.to_bytes(2, "little", signed=True)
    return bytes(out)


# ---------------------------------------------------------------------------
# Case payload generators
# ---------------------------------------------------------------------------


def _sweep_xxs_blocks() -> list[bytes]:
    blocks = []
    for block in range(8):
        magnitudes = [(8 * block + group) % 16 for group in range(8)]
        indices = [(32 * block + slot) % 256 for slot in range(32)]
        signs = [(32 * block + slot) % 128 for slot in range(32)]
        blocks.append(_pack_iq2_xxs(_HALF_ONE, magnitudes, indices, signs))
    return blocks


def _corner_xxs_blocks() -> list[bytes]:
    blocks = []
    # Every sign nibble and every 4-bit magnitude is exercised above; the corner
    # blocks exercise the block-scale field and grid-index boundaries.
    blocks.append(
        _pack_iq2_xxs(
            _HALF_MIN_SUBNORMAL,
            [index % 16 for index in range(8)],
            [0 if index % 2 == 0 else 255 for index in range(32)],
            [0 if index % 2 == 0 else 127 for index in range(32)],
        )
    )
    blocks.append(
        _pack_iq2_xxs(
            _HALF_MAX_FINITE,
            [15] * 8,
            [255] * 32,
            [127] * 32,
        )
    )
    blocks.append(
        _pack_iq2_xxs(
            _HALF_NEG_ONE,
            [0] * 8,
            [0] * 32,
            [0] * 32,
        )
    )
    rng = _Rng(0x1F2C3D4E5A6B7C8D)
    for pattern in ("random", "random-negative"):
        blocks.append(
            _pack_iq2_xxs(
                _half_scale(pattern, rng),
                [rng.below(16) for _ in range(8)],
                [rng.below(256) for _ in range(32)],
                [rng.below(128) for _ in range(32)],
            )
        )
    return blocks


def _sweep_iq2_s_blocks() -> list[bytes]:
    # 32 blocks x 32 indices = every one of the 1024 ten-bit grid indices once.
    blocks = []
    for block in range(32):
        low_scales = [(8 * block + group) % 16 for group in range(8)]
        high_scales = [(8 * block + group + 7) % 16 for group in range(8)]
        index_low = []
        index_high = []
        for slot in range(32):
            index = 32 * block + slot
            index_low.append(index & 0xFF)
            index_high.append(index >> 8)
        signs = [(32 * block + slot) % 128 for slot in range(32)]
        blocks.append(_pack_iq2_s(_HALF_ONE, low_scales, high_scales, index_low, index_high, signs))
    return blocks


def _corner_iq2_s_blocks() -> list[bytes]:
    rng = _Rng(0x2A3B4C5D6E7F8091)
    blocks = []
    # All 10-bit indices at their boundaries with every high-bit combination.
    index_low = [0 if slot % 2 == 0 else 255 for slot in range(32)]
    index_high = [slot % 4 for slot in range(32)]
    blocks.append(
        _pack_iq2_s(
            _HALF_MIN_SUBNORMAL,
            [0] * 8,
            [15] * 8,
            index_low,
            index_high,
            [127] * 32,
        )
    )
    blocks.append(
        _pack_iq2_s(
            _HALF_MAX_FINITE,
            [15] * 8,
            [0] * 8,
            [255] * 32,
            [3] * 32,
            [0] * 32,
        )
    )
    blocks.append(
        _pack_iq2_s(
            _HALF_NEG_ONE,
            [0] * 8,
            [0] * 8,
            [0] * 32,
            [1] * 32,
            [0] * 32,
        )
    )
    for pattern in ("random", "random-negative"):
        blocks.append(
            _pack_iq2_s(
                _half_scale(pattern, rng),
                [rng.below(16) for _ in range(8)],
                [rng.below(16) for _ in range(8)],
                [rng.below(256) for _ in range(32)],
                [rng.below(4) for _ in range(32)],
                [rng.below(128) for _ in range(32)],
            )
        )
    return blocks


def _sweep_iq1_m_blocks() -> list[bytes]:
    # 64 blocks x 32 indices = every one of the 2048 eleven-bit grid indices once.
    blocks = []
    for block in range(64):
        index_low = []
        index_high = []
        for slot in range(32):
            index = 32 * block + slot
            index_low.append(index & 0xFF)
            index_high.append(index >> 8)
        delta_signs = [1 if (32 * block + slot) % 2 == 0 else -1 for slot in range(32)]
        group_fields = [((8 * block + group) % 8, (8 * block + group + 3) % 8) for group in range(8)]
        blocks.append(
            _pack_iq1_m(_HALF_ONE, index_low, index_high, delta_signs, group_fields)
        )
    return blocks


def _corner_iq1_m_blocks() -> list[bytes]:
    rng = _Rng(0x3B4C5D6E7F8091A2)
    blocks = []
    blocks.append(
        _pack_iq1_m(
            _HALF_MIN_SUBNORMAL,
            [0 if slot % 2 == 0 else 255 for slot in range(32)],
            [7] * 32,
            [-1] * 32,
            [(7, 7)] * 8,
        )
    )
    blocks.append(
        _pack_iq1_m(
            _HALF_MAX_FINITE,
            [255] * 32,
            [0] * 32,
            [1] * 32,
            [(0, 0)] * 8,
        )
    )
    blocks.append(
        _pack_iq1_m(
            _HALF_NEG_ONE,
            [0] * 32,
            [3] * 32,
            [-1, 1] * 16,
            [(1, 2)] * 8,
        )
    )
    for pattern in ("random", "random-negative"):
        blocks.append(
            _pack_iq1_m(
                _half_scale(pattern, rng),
                [rng.below(256) for _ in range(32)],
                [rng.below(8) for _ in range(32)],
                [1 if rng.below(2) else -1 for _ in range(32)],
                [(rng.below(8), rng.below(8)) for _ in range(8)],
            )
        )
    return blocks


def _random_xxs_blocks(count: int, seed: int) -> list[bytes]:
    rng = _Rng(seed)
    return [
        _pack_iq2_xxs(
            _finite_f16(rng.next()),
            [rng.below(16) for _ in range(8)],
            [rng.below(256) for _ in range(32)],
            [rng.below(128) for _ in range(32)],
        )
        for _ in range(count)
    ]


def _random_iq2_s_blocks(count: int, seed: int) -> list[bytes]:
    rng = _Rng(seed)
    return [
        _pack_iq2_s(
            _finite_f16(rng.next()),
            [rng.below(16) for _ in range(8)],
            [rng.below(16) for _ in range(8)],
            [rng.below(256) for _ in range(32)],
            [rng.below(4) for _ in range(32)],
            [rng.below(128) for _ in range(32)],
        )
        for _ in range(count)
    ]


def _random_iq1_m_blocks(count: int, seed: int) -> list[bytes]:
    rng = _Rng(seed)
    return [
        _pack_iq1_m(
            _finite_f16(rng.next()),
            [rng.below(256) for _ in range(32)],
            [rng.below(8) for _ in range(32)],
            [1 if rng.below(2) else -1 for _ in range(32)],
            [(rng.below(8), rng.below(8)) for _ in range(8)],
        )
        for _ in range(count)
    ]


def _activation(patterns, scales, seed: int) -> bytes:
    rng = _Rng(seed)
    blocks = []
    for pattern, scale in zip(patterns, scales):
        blocks.append(_pack_q8_k(scale, _q8_values(pattern, rng)))
    if len(blocks) != len(scales):
        raise AssertionError("activation pattern/scale count mismatch")
    return b"".join(blocks)


def _random_activation(patterns, seed: int) -> bytes:
    rng = _Rng(seed)
    return b"".join(
        _pack_q8_k(_finite_f32(rng.next()), _q8_values(pattern, rng)) for pattern in patterns
    )


def _cancellation_xxs_payloads() -> tuple[bytes, bytes]:
    # Block 0/1: d_weight = 1.0, d_activation = 1024.0, grid 8 everywhere,
    # mag 0 (ls = 1), q8 = +8/-8 -> bsum = +/-2**14 and a contribution of
    # +/-2**24 that cancels exactly.  Block 2: one nonzero activation element
    # gives bsum = 8 and a contribution of 8, so the final 0.125f multiply
    # yields exactly 1.0f with no rounding anywhere.
    weight_blocks = [
        _pack_iq2_xxs(_HALF_ONE, [0] * 8, [0] * 32, [0] * 32),
        _pack_iq2_xxs(_HALF_ONE, [0] * 8, [0] * 32, [0] * 32),
        _pack_iq2_xxs(_HALF_ONE, [0] * 8, [0] * 32, [0] * 32),
    ]
    all_eight = [8] * 256
    minus_eight = [-8] * 256
    sparse = [0] * 256
    sparse[0] = 1
    activation_blocks = [
        _pack_q8_k(0x44800000, all_eight),  # 1024.0f
        _pack_q8_k(0x44800000, minus_eight),
        _pack_q8_k(_F32_ONE, sparse),
    ]
    return b"".join(weight_blocks), b"".join(activation_blocks)


def _cases() -> list[dict]:
    cases = []
    cases.append(_decode_case("iq2-xxs-grid-sweep", "IQ2_XXS", b"".join(_sweep_xxs_blocks())))
    cases.append(_decode_case("iq2-xxs-scale-corners", "IQ2_XXS", b"".join(_corner_xxs_blocks())))
    cases.append(_decode_case("iq2-s-grid-sweep", "IQ2_S", b"".join(_sweep_iq2_s_blocks())))
    cases.append(_decode_case("iq2-s-scale-corners", "IQ2_S", b"".join(_corner_iq2_s_blocks())))
    cases.append(_decode_case("iq1-m-grid-sweep", "IQ1_M", b"".join(_sweep_iq1_m_blocks())))
    cases.append(_decode_case("iq1-m-scale-corners", "IQ1_M", b"".join(_corner_iq1_m_blocks())))

    cases.append(
        _dot_case(
            "iq2-xxs-dot-extremes",
            "IQ2_XXS",
            b"".join(_corner_xxs_blocks()[:2]),
            _activation(("extremes", "zeros"), (_F32_ONE, _F32_NEG_ONE), 0x51),
        )
    )
    cases.append(
        _dot_case(
            "iq2-xxs-dot-random",
            "IQ2_XXS",
            b"".join(_random_xxs_blocks(3, 0x5A2B3C4D)),
            _random_activation(("random", "ramp", "extremes"), 0x52),
        )
    )
    cases.append(
        _dot_case(
            "iq2-xxs-dot-scale-corners",
            "IQ2_XXS",
            b"".join(_corner_xxs_blocks()[2:5]),
            _activation(
                ("alternating", "equal-positive", "equal-negative"),
                (0x3E000000, 0x42800000, 0x35800000),
                0x53,
            ),
        )
    )
    cases.append(
        _dot_case(
            "iq2-xxs-dot-cancellation",
            "IQ2_XXS",
            *_cancellation_xxs_payloads(),
        )
    )
    cases.append(
        _dot_case(
            "iq2-s-dot-extremes",
            "IQ2_S",
            b"".join(_corner_iq2_s_blocks()[:2]),
            _activation(("extremes", "zeros"), (_F32_ONE, _F32_NEG_ONE), 0x54),
        )
    )
    cases.append(
        _dot_case(
            "iq2-s-dot-random",
            "IQ2_S",
            b"".join(_random_iq2_s_blocks(3, 0x6B3C4D5E)),
            _random_activation(("random", "alternating", "random"), 0x55),
        )
    )
    cases.append(
        _dot_case(
            "iq2-s-dot-scale-corners",
            "IQ2_S",
            b"".join(_corner_iq2_s_blocks()[2:4]),
            _activation(("ramp", "equal-positive"), (0x3F000000, 0x3D800000), 0x56),
        )
    )
    cases.append(
        _dot_case(
            "iq1-m-dot-extremes",
            "IQ1_M",
            b"".join(_corner_iq1_m_blocks()[:2]),
            _activation(("extremes", "zeros"), (_F32_ONE, _F32_NEG_ONE), 0x57),
        )
    )
    cases.append(
        _dot_case(
            "iq1-m-dot-random",
            "IQ1_M",
            b"".join(_random_iq1_m_blocks(3, 0x7C4D5E6F)),
            _random_activation(("random", "ramp", "extremes"), 0x58),
        )
    )
    cases.append(
        _dot_case(
            "iq1-m-dot-scale-corners",
            "IQ1_M",
            b"".join(_corner_iq1_m_blocks()[2:4]),
            _activation(("alternating", "equal-negative"), (0x3E000000, 0x42800000), 0x59),
        )
    )
    return cases


def _decode_case(case_id: str, name: str, payload: bytes) -> dict:
    metadata = _EXPECTED_FORMATS[name]
    return {
        "id": case_id,
        "operation": "decode",
        "format": name,
        "elements": len(payload) // metadata["block_bytes"] * PACKED_BLOCK_ELEMENTS,
        "payload_hex": payload.hex(),
        "expected_fp32_bits": _reference_decode(name, payload),
    }


def _dot_case(case_id: str, name: str, weight: bytes, activation: bytes) -> dict:
    bits, subdots = _reference_dot(name, weight, activation)
    return {
        "id": case_id,
        "operation": "dot",
        "weight_format": name,
        "activation_format": "Q8_K",
        "elements": len(weight) // _EXPECTED_FORMATS[name]["block_bytes"] * PACKED_BLOCK_ELEMENTS,
        "weight_hex": weight.hex(),
        "activation_hex": activation.hex(),
        "expected_fp32_bits": bits,
        "expected_integer_subdots": subdots,
    }


# ---------------------------------------------------------------------------
# Canonical JSON, generation, validation, and verification
# ---------------------------------------------------------------------------


def _canonical_bytes(document: dict, *, omit_digest: bool) -> bytes:
    payload = {
        key: value for key, value in document.items() if not omit_digest or key != "payload_sha256"
    }
    return json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode("ascii")


def _digest(document: dict) -> str:
    return hashlib.sha256(_canonical_bytes(document, omit_digest=True)).hexdigest()


def _base_document(fixture_id: str, cases: list[dict]) -> dict:
    return {
        "schema_version": 1,
        "fixture_id": fixture_id,
        "synthetic": True,
        "arithmetic_id": ARITHMETIC_ID,
        "formats": _EXPECTED_FORMATS,
        "activations": _EXPECTED_ACTIVATIONS,
        "provenance": _EXPECTED_PROVENANCE,
        "cases": cases,
    }


def generate_fixture() -> dict:
    """Build the deterministic IQ decode/dot fixture without querying a DUT."""
    document = _base_document(FIXTURE_ID, _cases())
    document["payload_sha256"] = _digest(document)
    return document


def _integer_subdot_bounds(name: str) -> tuple[int, int]:
    if name == "IQ1_M":
        return (-1000000, 1000000)
    return (-8000000, 8000000)


def _subdot_columns(name: str) -> int:
    return len(_EXPECTED_FORMATS[name]["integer_subdot_fields"])


def _check_keys(value, expected: set[str], path: str) -> None:
    if type(value) is not dict or set(value) != expected:
        _fail("fixture_schema", path + ": expected exact object keys")


def _check_bits(value, path: str) -> None:
    if type(value) is not str or not _BITS_PATTERN.fullmatch(value):
        _fail("fixture_schema", path + ": expected lowercase 0x and eight binary32 hex digits")
    if (int(value[2:], 16) >> 23) & 0xFF == 0xFF:
        _fail("fixture_schema", path + ": binary32 expectation must be finite")


def _check_elements(value, name: str, path: str) -> None:
    block = _EXPECTED_FORMATS[name]["block_elements"]
    if type(value) is not int or not 1 <= value <= MAX_ELEMENTS or value % block:
        _fail(
            "fixture_schema",
            path + ": expected positive whole %d-element blocks up to %d" % (block, MAX_ELEMENTS),
        )


def _check_payload_hex(value, name: str, elements: int, path: str) -> bytes:
    metadata = _EXPECTED_FORMATS[name]
    expected_bytes = elements // metadata["block_elements"] * metadata["block_bytes"]
    if type(value) is not str or len(value) != 2 * expected_bytes or not _HEX_PATTERN.fullmatch(value):
        _fail("fixture_schema", path + ": expected exactly %d bytes of contiguous hex" % expected_bytes)
    payload = bytes.fromhex(value)
    stride = metadata["block_bytes"]
    for start in range(0, len(payload), stride):
        block = payload[start : start + stride]
        raw = _iq1_m_terms(block)[0] if name == "IQ1_M" else _u16(block, 0)
        if (raw >> 10) & 0x1F == 0x1F:
            _fail("fixture_schema", path + ": nonfinite binary16 scale at block %d" % (start // stride))
    return payload


def _check_activation_hex(value, elements: int, path: str) -> bytes:
    expected_bytes = elements // PACKED_BLOCK_ELEMENTS * _Q8_K["block_bytes"]
    if type(value) is not str or len(value) != 2 * expected_bytes or not _HEX_PATTERN.fullmatch(value):
        _fail("fixture_schema", path + ": expected exactly %d bytes of contiguous hex" % expected_bytes)
    payload = bytes.fromhex(value)
    for start in range(0, len(payload), _Q8_K["block_bytes"]):
        if (_u32(payload, start) >> 23) & 0xFF == 0xFF:
            _fail("fixture_schema", path + ": nonfinite binary32 scale at block %d" % (start // _Q8_K["block_bytes"]))
    return payload


def _check_subdots(value, name: str, elements: int, path: str) -> None:
    columns = _subdot_columns(name)
    bounds = _integer_subdot_bounds(name)
    blocks = elements // PACKED_BLOCK_ELEMENTS
    groups = _EXPECTED_FORMATS[name]["subblocks"]
    if type(value) is not list or len(value) != blocks:
        _fail("fixture_schema", path + ": expected exactly %d block rows" % blocks)
    for block_index, block_row in enumerate(value):
        if type(block_row) is not list or len(block_row) != groups:
            _fail("fixture_schema", path + "[%d]: expected exactly %d group rows" % (block_index, groups))
        for group_index, group in enumerate(block_row):
            if type(group) is not list or len(group) != columns:
                _fail(
                    "fixture_schema",
                    path + "[%d][%d]: expected exactly %d integer terms" % (block_index, group_index, columns),
                )
            for term_index, entry in enumerate(group):
                where = path + "[%d][%d][%d]" % (block_index, group_index, term_index)
                if type(entry) is not int:
                    _fail("fixture_schema", where + ": expected integer")
                if not bounds[0] <= entry <= bounds[1]:
                    _fail("fixture_schema", where + ": outside integer partial-sum range")


def _check_common(document, fixture_id: str) -> None:
    if type(document) is not dict:
        _fail("fixture_schema", "fixture: expected an object")
    for key, expected in (
        ("schema_version", 1),
        ("fixture_id", fixture_id),
        ("synthetic", True),
        ("arithmetic_id", ARITHMETIC_ID),
        ("formats", _EXPECTED_FORMATS),
        ("activations", _EXPECTED_ACTIVATIONS),
        ("provenance", _EXPECTED_PROVENANCE),
    ):
        actual = document.get(key, object())
        if type(actual) is not type(expected) or actual != expected:
            _fail("fixture_schema", "fixture." + key + ": does not match frozen synthetic contract")
    cases = document.get("cases")
    if type(cases) is not list or not 1 <= len(cases) <= MAX_CASES:
        _fail("fixture_schema", "fixture.cases: expected 1..%d cases" % MAX_CASES)


def _canonical_and_digest(document: dict):
    digest = document.get("payload_sha256")
    if type(digest) is not str or not _SHA_PATTERN.fullmatch(digest):
        _fail("fixture_schema", "fixture.payload_sha256: expected 64 lowercase hex digits")
    try:
        canonical = _canonical_bytes(document, omit_digest=False)
        claimed = _canonical_bytes(document, omit_digest=True)
    except (ValueError, TypeError, RecursionError, OverflowError):
        _fail("fixture_schema", "fixture: cannot encode bounded finite canonical JSON")
    if len(canonical) > MAX_FIXTURE_BYTES:
        _fail("fixture_schema", "fixture: canonical JSON exceeds %d bytes" % MAX_FIXTURE_BYTES)
    if hashlib.sha256(claimed).hexdigest() != digest:
        _fail("fixture_digest", "fixture.payload_sha256: canonical payload digest mismatch")
    return canonical


def _validate_case_header(case, path: str, seen: set):
    if type(case) is not dict:
        _fail("fixture_schema", path + ": expected an object")
    case_id = case.get("id")
    if type(case_id) is not str or not _ID_PATTERN.fullmatch(case_id) or case_id in seen:
        _fail("fixture_schema", path + ".id: expected unique bounded lowercase case id")
    seen.add(case_id)
    return case_id


def _case_format(case: dict) -> str:
    return case.get("weight_format") if case.get("operation") == "dot" else case.get("format")


def _validate_fixture(document) -> tuple[bytes, list[tuple[dict, bytes, bytes | None]]]:
    _check_common(document, FIXTURE_ID)
    if set(document) != _TOP_KEYS:
        _fail("fixture_schema", "fixture: expected exact object keys")
    canonical = _canonical_and_digest(document)
    seen = set()
    prepared = []
    for index, case in enumerate(document["cases"]):
        path = "fixture.cases[%d]" % index
        _validate_case_header(case, path, seen)
        operation = case.get("operation")
        if type(operation) is not str or operation not in ("decode", "dot"):
            _fail("fixture_schema", path + ".operation: expected decode or dot")
        _check_keys(case, _DECODE_KEYS if operation == "decode" else _DOT_KEYS, path)
        name = _case_format(case)
        if type(name) is not str or name not in _EXPECTED_FORMATS:
            _fail("fixture_schema", path + ": unsupported IQ packed format")
        elements = case["elements"]
        _check_elements(elements, name, path + ".elements")
        weight = _check_payload_hex(case["payload_hex"] if operation == "decode" else case["weight_hex"], name, elements, path)
        if operation == "decode":
            bits = case["expected_fp32_bits"]
            if type(bits) is not list or len(bits) != elements:
                _fail("fixture_schema", path + ".expected_fp32_bits: expected exactly %d values" % elements)
            for lane, value in enumerate(bits):
                _check_bits(value, path + ".expected_fp32_bits[%d]" % lane)
            prepared.append((case, weight, None))
        else:
            if case["activation_format"] != "Q8_K":
                _fail("fixture_schema", path + ".activation_format: expected Q8_K")
            activation = _check_activation_hex(case["activation_hex"], elements, path + ".activation_hex")
            _check_bits(case["expected_fp32_bits"], path + ".expected_fp32_bits")
            _check_subdots(case["expected_integer_subdots"], name, elements, path + ".expected_integer_subdots")
            prepared.append((case, weight, activation))
    return canonical, prepared


def verify_fixture(document) -> dict:
    """Validate metadata strictly, then compare the production DUT against goldens."""
    _, prepared = _validate_fixture(document)
    from . import iq_formats as dut

    mismatches = []
    decode_count = 0
    dot_count = 0
    for case, first, second in prepared:
        if case["operation"] == "decode":
            decode_count += 1
            result = dut.decode(case["format"], first, case["elements"])
            comparisons = [("fp32_bits", case["expected_fp32_bits"], result["fp32_bits"])]
        else:
            dot_count += 1
            result = dut.dot(
                case["weight_format"], first, second, case["elements"], activation_format=case["activation_format"]
            )
            comparisons = [
                ("fp32_bits", case["expected_fp32_bits"], result["fp32_bits"]),
                ("integer_subdots", case["expected_integer_subdots"], result["integer_subdots"]),
            ]
        for field, expected, actual in comparisons:
            if type(actual) is not type(expected) or actual != expected:
                mismatches.append({"case_id": case["id"], "field": field, "expected": expected, "actual": actual})
    return {
        "schema_version": 1,
        "fixture_id": FIXTURE_ID,
        "arithmetic_id": ARITHMETIC_ID,
        "payload_sha256": document["payload_sha256"],
        "case_count": len(prepared),
        "decode_cases": decode_count,
        "dot_cases": dot_count,
        "synthetic": True,
        "verified": not mismatches,
        "mismatches": mismatches,
        "device_ready": False,
        "inference_ready": False,
    }


# ---------------------------------------------------------------------------
# RTL integer bundle: raw bytes plus per-32-group integer partial sums
# ---------------------------------------------------------------------------


def _rtl_case(case_id: str, name: str, weight: bytes, activation: bytes) -> dict:
    bits, subdots = _reference_dot(name, weight, activation)
    elements = len(weight) // _EXPECTED_FORMATS[name]["block_bytes"] * PACKED_BLOCK_ELEMENTS
    return {
        "id": case_id,
        "weight_format": name,
        "activation_format": "Q8_K",
        "elements": elements,
        "weight_hex": weight.hex(),
        "activation_hex": activation.hex(),
        "integer_subdot_fields": list(_EXPECTED_FORMATS[name]["integer_subdot_fields"]),
        "expected_integer_subdots": subdots,
        "expected_fp32_bits": bits,
    }


def _rtl_cases() -> list[dict]:
    return [
        _rtl_case(
            "iq2-xxs-extremes",
            "IQ2_XXS",
            b"".join(_corner_xxs_blocks()[:2]),
            _activation(("extremes", "zeros"), (_F32_ONE, _F32_NEG_ONE), 0x51),
        ),
        _rtl_case(
            "iq2-xxs-random",
            "IQ2_XXS",
            b"".join(_random_xxs_blocks(2, 0x5A2B3C4D)),
            _random_activation(("random", "ramp"), 0x52),
        ),
        _rtl_case("iq2-xxs-cancellation", "IQ2_XXS", *_cancellation_xxs_payloads()),
        _rtl_case(
            "iq2-s-extremes",
            "IQ2_S",
            b"".join(_corner_iq2_s_blocks()[:2]),
            _activation(("extremes", "zeros"), (_F32_ONE, _F32_NEG_ONE), 0x54),
        ),
        _rtl_case(
            "iq2-s-random",
            "IQ2_S",
            b"".join(_random_iq2_s_blocks(2, 0x6B3C4D5E)),
            _random_activation(("random", "alternating"), 0x55),
        ),
        _rtl_case(
            "iq1-m-extremes",
            "IQ1_M",
            b"".join(_corner_iq1_m_blocks()[:2]),
            _activation(("extremes", "zeros"), (_F32_ONE, _F32_NEG_ONE), 0x57),
        ),
        _rtl_case(
            "iq1-m-random",
            "IQ1_M",
            b"".join(_random_iq1_m_blocks(2, 0x7C4D5E6F)),
            _random_activation(("random", "ramp"), 0x58),
        ),
    ]


def generate_rtl_bundle() -> dict:
    """Build the deterministic RTL integer-vector bundle without querying a DUT."""
    document = _base_document(RTL_FIXTURE_ID, _rtl_cases())
    document["payload_sha256"] = _digest(document)
    return document


def validate_rtl_bundle(document) -> tuple[bytes, list[tuple[dict, bytes, bytes]]]:
    _check_common(document, RTL_FIXTURE_ID)
    if set(document) != _RTL_TOP_KEYS:
        _fail("fixture_schema", "fixture: expected exact object keys")
    canonical = _canonical_and_digest(document)
    seen = set()
    prepared = []
    for index, case in enumerate(document["cases"]):
        path = "fixture.cases[%d]" % index
        _validate_case_header(case, path, seen)
        _check_keys(case, _RTL_CASE_KEYS, path)
        name = case.get("weight_format")
        if type(name) is not str or name not in _EXPECTED_FORMATS:
            _fail("fixture_schema", path + ": unsupported IQ packed format")
        if case["activation_format"] != "Q8_K":
            _fail("fixture_schema", path + ".activation_format: expected Q8_K")
        elements = case["elements"]
        _check_elements(elements, name, path + ".elements")
        weight = _check_payload_hex(case["weight_hex"], name, elements, path + ".weight_hex")
        activation = _check_activation_hex(case["activation_hex"], elements, path + ".activation_hex")
        if case["integer_subdot_fields"] != _EXPECTED_FORMATS[name]["integer_subdot_fields"]:
            _fail("fixture_schema", path + ".integer_subdot_fields: does not match format contract")
        _check_bits(case["expected_fp32_bits"], path + ".expected_fp32_bits")
        _check_subdots(case["expected_integer_subdots"], name, elements, path + ".expected_integer_subdots")
        prepared.append((case, weight, activation))
    return canonical, prepared


def verify_rtl_bundle(document) -> dict:
    """Validate the RTL bundle, then compare the integer group sums to the DUT."""
    _, prepared = validate_rtl_bundle(document)
    from . import iq_formats as dut

    mismatches = []
    for case, weight, activation in prepared:
        result = dut.dot(
            case["weight_format"], weight, activation, case["elements"], activation_format=case["activation_format"]
        )
        for field, expected, actual in (
            ("expected_integer_subdots", case["expected_integer_subdots"], result["integer_subdots"]),
            ("expected_fp32_bits", case["expected_fp32_bits"], result["fp32_bits"]),
        ):
            if type(actual) is not type(expected) or actual != expected:
                mismatches.append({"case_id": case["id"], "field": field, "expected": expected, "actual": actual})
    return {
        "schema_version": 1,
        "fixture_id": RTL_FIXTURE_ID,
        "arithmetic_id": ARITHMETIC_ID,
        "payload_sha256": document["payload_sha256"],
        "case_count": len(prepared),
        "synthetic": True,
        "verified": not mismatches,
        "mismatches": mismatches,
        "device_ready": False,
        "inference_ready": False,
    }
