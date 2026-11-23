# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 lvyufeng and the openTPU contributors

"""Golden tests for the production IQ2_XXS / IQ2_S / IQ1_M scalar path.

``tools/packed_oracle/iq_formats.py`` is the code path under test (the
descriptor-driven production transcription).  Every expected value in this
module is derived independently from the pinned ggml formulas documented in the
module docstrings, using ``tools/packed_oracle/iq_tables.py`` as the only shared
*data* input.  ``tools/packed_oracle/iq_vectors.py`` (the separately written
reference transcription) is used only to cross-check the production path, never
to produce the expected values asserted here.

Test payloads are written from *semantic* fields (group scales, grid indices,
sign indices, delta signs) and then packed with the independent packers below,
so a bit-order mistake in the production decoder cannot cancel out against a
matching mistake in an expected-value helper.
"""

import ast
import hashlib
from pathlib import Path
import struct
import unittest

from tools.packed_oracle import iq_formats as dut
from tools.packed_oracle import iq_tables as t
from tools.packed_oracle import iq_vectors as ref


MAX_ELEMENTS = 65536
BLOCK_ELEMENTS = 256
HALF_ONE = 0x3C00
F32_ONE = 0x3F800000
F32_TWO = 0x40000000
F32_POINT_FIVE = 0x3F000000
_F32_FINITE_MAX = 3.4028234663852886e38
_HALF_FINITE_MAX = 65504.0

_FORMATS = ("IQ2_XXS", "IQ2_S", "IQ1_M")
_EXPECTED_LAYOUTS = {
    "IQ2_XXS": [
        {"field": "d", "offset": 0, "bytes": 2},
        {"field": "qs", "offset": 2, "bytes": 64},
    ],
    "IQ2_S": [
        {"field": "d", "offset": 0, "bytes": 2},
        {"field": "qs", "offset": 2, "bytes": 64},
        {"field": "qh", "offset": 66, "bytes": 8},
        {"field": "scales", "offset": 74, "bytes": 8},
    ],
    "IQ1_M": [
        {"field": "qs", "offset": 0, "bytes": 32},
        {"field": "qh", "offset": 32, "bytes": 16},
        {"field": "scales", "offset": 48, "bytes": 8},
    ],
    "Q8_K": [
        {"field": "d", "offset": 0, "bytes": 4},
        {"field": "qs", "offset": 4, "bytes": 256},
        {"field": "bsums", "offset": 260, "bytes": 32},
    ],
}
_EXPECTED_BLOCK_BYTES = {"IQ2_XXS": 66, "IQ2_S": 82, "IQ1_M": 56, "Q8_K": 292}
_EXPECTED_TYPE_IDS = {"IQ2_XXS": 16, "IQ2_S": 22, "IQ1_M": 29}
_EXPECTED_GRID_ENTRIES = {"IQ2_XXS": 256, "IQ2_S": 1024, "IQ1_M": 2048}
_EXPECTED_INDEX_BITS = {"IQ2_XXS": 8, "IQ2_S": 10, "IQ1_M": 11}
_EXPECTED_HIGH_BITS = {"IQ2_XXS": 0, "IQ2_S": 2, "IQ1_M": 3}
_EXPECTED_SCALE_BITS = {"IQ2_XXS": 4, "IQ2_S": 4, "IQ1_M": 3}
_EXPECTED_SCALE_LEVELS = {"IQ2_XXS": 16, "IQ2_S": 16, "IQ1_M": 8}
_EXPECTED_DELTAS = {"IQ2_XXS": 0, "IQ2_S": 0, "IQ1_M": 4}
_EXPECTED_BLOCK_SCALE = {
    "IQ2_XXS": "fp16",
    "IQ2_S": "fp16",
    "IQ1_M": "fp16-packed",
}
_EXPECTED_SUBDOT_FIELDS = {
    "IQ2_XXS": ["grid"],
    "IQ2_S": ["low16", "high16"],
    "IQ1_M": ["grid", "delta"],
}
_EXPECTED_TABLE_SHA256 = {
    "kmask_iq2xs": "5ac9831b2e30eb285ef34f8501620f878432d5c04331ad1ae47f977a83ba41a5",
    "ksigns_iq2xs": "a76742a603f8beca5212ecce0f1f02f11a4887fd2f7c8b15aca0ea3eb3380c31",
    "iq2xxs_grid": "05826b5d3e472a3a78f196be62ac78acf81df0f909626e12ab9fa2a5d490dd54",
    "iq2s_grid": "e1aa1473412b0552c2174c30ef22ab4073f6a181b85a17056e8249bd2932fd88",
    "iq1s_grid": "07540ffc1aeaf6ad4d97e96b0fcc765aae39671d4ae4a27bbd0e796fde167c6a",
}

_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------- exact floats


def _f32_value(bits):
    """Exact binary32 bit pattern -> Python float (always exact)."""
    return struct.unpack("<f", struct.pack("<I", bits))[0]


def _half_value(bits):
    """Exact binary16 bit pattern -> Python float (always exact)."""
    return struct.unpack("<e", struct.pack("<H", bits))[0]


def _f32_bits_exact(value):
    """Return binary32 bits, asserting the Python value is exactly representable.

    Every analytic expectation in this file is a short dyadic rational, so the
    assertion is a genuine self-check of the test vector, not a tolerance.
    """
    packed = struct.pack("<f", value)
    bits = struct.unpack("<I", packed)[0]
    round_trip = struct.unpack("<f", packed)[0]
    if round_trip != value:
        raise AssertionError(
            "test expectation %r is not exactly representable as binary32" % (value,)
        )
    return bits


# --------------------------------------------------------- payload packers


def _pack_xxs(scale_bits, magnitudes, indices, sign_indices):
    """Pack one IQ2_XXS block from semantic fields (66 bytes)."""
    out = bytearray(66)
    out[0:2] = struct.pack("<H", scale_bits)
    for group in range(8):
        base = 2 + 8 * group
        for lane in range(4):
            out[base + lane] = indices[4 * group + lane] & 0xFF
        aux = (magnitudes[group] & 0xF) << 28
        for lane in range(4):
            aux |= (sign_indices[4 * group + lane] & 0x7F) << (7 * lane)
        out[base + 4 : base + 8] = struct.pack("<I", aux & 0xFFFFFFFF)
    return bytes(out)


def _pack_iq2_s(scale_bits, low_scales, high_scales, index_low, index_high, sign_indices):
    """Pack one IQ2_S block from semantic fields (82 bytes)."""
    out = bytearray(82)
    out[0:2] = struct.pack("<H", scale_bits)
    for group in range(8):
        high_bits = 0
        for lane in range(4):
            out[2 + 4 * group + lane] = index_low[4 * group + lane] & 0xFF
            out[2 + 32 + 4 * group + lane] = sign_indices[4 * group + lane] & 0xFF
            high_bits |= (index_high[4 * group + lane] & 0x3) << (2 * lane)
        out[66 + group] = high_bits & 0xFF
        out[74 + group] = (low_scales[group] & 0xF) | ((high_scales[group] & 0xF) << 4)
    return bytes(out)


def _pack_iq1_m(scale_bits, group_fields, index_low, index_high, delta_signs):
    """Pack one IQ1_M block from semantic fields (56 bytes)."""
    out = bytearray(56)
    for lane in range(32):
        out[lane] = index_low[lane] & 0xFF
    high_bytes = bytearray(16)
    for group in range(8):
        for lane in range(4):
            slot = 2 * group + lane // 2
            high = index_high[4 * group + lane] & 0x7
            if lane % 2 == 0:
                high_bytes[slot] |= high
                if delta_signs[4 * group + lane] < 0:
                    high_bytes[slot] |= 0x08
            else:
                high_bytes[slot] |= high << 4
                if delta_signs[4 * group + lane] < 0:
                    high_bytes[slot] |= 0x80
    out[32:48] = high_bytes
    words = [
        (scale_bits & 0xF) << 12,
        ((scale_bits >> 4) & 0xF) << 12,
        ((scale_bits >> 8) & 0xF) << 12,
        scale_bits & 0xF000,
    ]
    for group in range(8):
        index = group // 2
        shift = 6 * (group % 2)
        words[index] |= (group_fields[group][0] & 0x7) << shift
        words[index] |= (group_fields[group][1] & 0x7) << (shift + 3)
    for index in range(4):
        out[48 + 2 * index : 50 + 2 * index] = struct.pack("<H", words[index])
    return bytes(out)


def _q8_block(values, scale_bits=F32_ONE):
    """Pack one Q8_K block (292 bytes) with zero ``bsums``."""
    if len(values) != 256:
        raise AssertionError("Q8_K block needs 256 activations")
    out = bytearray(292)
    out[0:4] = struct.pack("<I", scale_bits)
    for index, value in enumerate(values):
        if not -128 <= value <= 127:
            raise AssertionError("Q8_K value outside int8 range: %r" % (value,))
        out[4 + index] = value & 0xFF
    return bytes(out)


def _q8_values(pattern):
    if pattern == "extremes":
        return [-128 if index % 2 == 0 else 127 for index in range(256)]
    if pattern == "zeros":
        return [0] * 256
    if pattern == "equal-positive":
        return [127] * 256
    if pattern == "equal-negative":
        return [-128] * 256
    if pattern == "alternating":
        return [127 if index % 2 == 0 else -128 for index in range(256)]
    if pattern == "ramp":
        return [index - 128 for index in range(256)]
    if pattern == "random":
        return [((index * 97 + 13) % 256) - 128 for index in range(256)]
    raise AssertionError("unknown Q8_K pattern: " + pattern)


_Q8_PATTERNS = (
    "extremes",
    "zeros",
    "equal-positive",
    "equal-negative",
    "alternating",
    "ramp",
    "random",
)


# --------------------------------------------- analytic expected-value formulas


def _expected_xxs_values(scale_bits, magnitudes, indices, sign_indices):
    """ggml dequantize_row_iq2_xxs, evaluated from semantic block fields."""
    d = _half_value(scale_bits)
    values = []
    for group in range(8):
        db = d * (2 * magnitudes[group] + 1) / 8.0
        for lane in range(4):
            grid = t.grid_bytes(t.IQ2XXS_GRID[indices[4 * group + lane]])
            mask = t.KSIGNS_IQ2XS[sign_indices[4 * group + lane]]
            for column in range(8):
                value = db * grid[column]
                if mask & t.KMASK_IQ2XS[column]:
                    value = -value
                values.append(_f32_bits_exact(value))
    return values


def _expected_iq2_s_values(scale_bits, low, high, index_low, index_high, sign_indices):
    """ggml dequantize_row_iq2_s, evaluated from semantic block fields."""
    d = _half_value(scale_bits)
    values = []
    for group in range(8):
        group_scales = (low[group], high[group])
        for lane in range(4):
            db = d * (2 * group_scales[lane // 2] + 1) / 8.0
            grid_index = (index_low[4 * group + lane] & 0xFF) | (
                (index_high[4 * group + lane] & 0x3) << 8
            )
            grid = t.grid_bytes(t.IQ2S_GRID[grid_index])
            mask = sign_indices[4 * group + lane] & 0xFF
            for column in range(8):
                value = db * grid[column]
                if mask & t.KMASK_IQ2XS[column]:
                    value = -value
                values.append(_f32_bits_exact(value))
    return values


def _expected_iq1_m_values(scale_bits, group_fields, index_low, index_high, delta_signs):
    """ggml dequantize_row_iq1_m, evaluated from semantic block fields."""
    d = _half_value(scale_bits)
    values = []
    for group in range(8):
        low_scale = 2 * group_fields[group][0] + 1
        high_scale = 2 * group_fields[group][1] + 1
        for lane in range(4):
            dl = d * (low_scale if lane < 2 else high_scale)
            grid_index = (index_low[4 * group + lane] & 0xFF) | (
                (index_high[4 * group + lane] & 0x7) << 8
            )
            grid = t.grid_signed(t.IQ1S_GRID[grid_index])
            delta = delta_signs[4 * group + lane] / 8.0
            for column in range(8):
                values.append(_f32_bits_exact(dl * (grid[column] + delta)))
    return values


# ---------------------------------------------------- analytic integer subdots


def _xxs_subdots(magnitudes, indices, sign_indices, q8):
    subdots = []
    for group in range(8):
        total = 0
        for lane in range(4):
            grid = t.grid_bytes(t.IQ2XXS_GRID[indices[4 * group + lane]])
            mask = t.KSIGNS_IQ2XS[sign_indices[4 * group + lane]]
            base = group * 32 + lane * 8
            for column in range(8):
                product = grid[column] * q8[base + column]
                total += -product if mask & t.KMASK_IQ2XS[column] else product
        subdots.append([(2 * magnitudes[group] + 1) * total])
    return subdots


def _iq2_s_subdots(low, high, index_low, index_high, sign_indices, q8):
    subdots = []
    for group in range(8):
        low_total = 0
        high_total = 0
        for lane in range(4):
            grid_index = (index_low[4 * group + lane] & 0xFF) | (
                (index_high[4 * group + lane] & 0x3) << 8
            )
            grid = t.grid_bytes(t.IQ2S_GRID[grid_index])
            mask = sign_indices[4 * group + lane] & 0xFF
            base = group * 32 + lane * 8
            total = 0
            for column in range(8):
                product = grid[column] * q8[base + column]
                total += -product if mask & t.KMASK_IQ2XS[column] else product
            if lane < 2:
                low_total += total
            else:
                high_total += total
        subdots.append([(2 * low[group] + 1) * low_total, (2 * high[group] + 1) * high_total])
    return subdots


def _iq1_m_subdots(group_fields, index_low, index_high, delta_signs, q8):
    subdots = []
    for group in range(8):
        grid_sum = [0, 0]
        delta_sum = [0, 0]
        for lane in range(4):
            grid_index = (index_low[4 * group + lane] & 0xFF) | (
                (index_high[4 * group + lane] & 0x7) << 8
            )
            grid = t.grid_signed(t.IQ1S_GRID[grid_index])
            base = group * 32 + lane * 8
            grid_lane = 0
            raw_lane = 0
            for column in range(8):
                grid_lane += grid[column] * q8[base + column]
                raw_lane += q8[base + column]
            grid_sum[lane // 2] += grid_lane
            delta_sum[lane // 2] += raw_lane * delta_signs[4 * group + lane]
        low_scale = 2 * group_fields[group][0] + 1
        high_scale = 2 * group_fields[group][1] + 1
        subdots.append(
            [
                grid_sum[0] * low_scale + grid_sum[1] * high_scale,
                delta_sum[0] * low_scale + delta_sum[1] * high_scale,
            ]
        )
    return subdots


def _dot_bits_xxs(blocks, q8, activation_scale_bits=F32_ONE):
    """Analytic IQ2_XXS dot: only used where every step is exactly representable."""
    d_activation = _f32_value(activation_scale_bits)
    row = 0.0
    for scale_bits, magnitudes, indices, sign_indices in blocks:
        subdots = _xxs_subdots(magnitudes, indices, sign_indices, q8)
        bsum = sum(term for group in subdots for term in group)
        d = _half_value(scale_bits) * d_activation
        row += d * bsum
    return _f32_bits_exact(0.125 * row)


def _dot_bits_iq2_s(blocks, q8, activation_scale_bits=F32_ONE):
    d_activation = _f32_value(activation_scale_bits)
    row = 0.0
    for scale_bits, low, high, index_low, index_high, signs in blocks:
        subdots = _iq2_s_subdots(low, high, index_low, index_high, signs, q8)
        bsum = sum(term for group in subdots for term in group)
        d = _half_value(scale_bits) * d_activation
        row += d * bsum
    return _f32_bits_exact(0.125 * row)


def _dot_bits_iq1_m(blocks, q8, activation_scale_bits=F32_ONE):
    d_activation = _f32_value(activation_scale_bits)
    row = 0.0
    for scale_bits, group_fields, index_low, index_high, delta_signs in blocks:
        subdots = _iq1_m_subdots(group_fields, index_low, index_high, delta_signs, q8)
        grid_total = sum(term[0] for term in subdots)
        delta_total = sum(term[1] for term in subdots)
        d = _half_value(scale_bits) * d_activation
        row += d * (grid_total + 0.125 * delta_total)
    return _f32_bits_exact(row)


# ------------------------------------------------------------- misc helpers


def _flat_terms(integer_subdots):
    return [term for block in integer_subdots for group in block for term in group]


def _mag_key(bits):
    """Monotone integer key for finite binary32 (sign-magnitude ordering)."""
    magnitude = bits & 0x7FFFFFFF
    return -magnitude if bits & 0x80000000 else magnitude


def _ulp_distance(left_bits, right_bits):
    return abs(_mag_key(left_bits) - _mag_key(right_bits))


def _decode_bits(format_name, payload, elements):
    result = dut.decode(format_name, payload, elements)
    return [int(bits, 16) for bits in result["fp32_bits"]]


def _random_xxs_blocks(count, seed):
    return ref._random_xxs_blocks(count, seed)


def _random_iq2_s_blocks(count, seed):
    return ref._random_iq2_s_blocks(count, seed)


def _random_iq1_m_blocks(count, seed):
    return ref._random_iq1_m_blocks(count, seed)


def _descriptor_layout(spec):
    """Normalise a JSON-safe ``[field, offset, bytes]`` descriptor to dict rows."""
    return [
        {"field": field, "offset": offset, "bytes": size}
        for field, offset, size in spec["layout"]
    ]


class BlockLayoutTests(unittest.TestCase):
    def test_weight_descriptors_match_the_pinned_ggml_fields(self):
        self.assertEqual(set(dut.FORMATS), set(_FORMATS))
        for name in _FORMATS:
            spec = dut.FORMATS[name]
            with self.subTest(format=name):
                self.assertEqual(spec["ggml_type"], _EXPECTED_TYPE_IDS[name])
                self.assertEqual(spec["block_bytes"], _EXPECTED_BLOCK_BYTES[name])
                self.assertEqual(spec["block_elements"], BLOCK_ELEMENTS)
                self.assertEqual(spec["subblocks"], 8)
                self.assertEqual(spec["subblock_elements"], 32)
                self.assertEqual(spec["activation_format"], "Q8_K")
                self.assertEqual(spec["activation_block_bytes"], 292)
                self.assertEqual(spec["grid_entries"], _EXPECTED_GRID_ENTRIES[name])
                self.assertEqual(spec["grid_index_bits"], _EXPECTED_INDEX_BITS[name])
                self.assertEqual(spec["index_high_bits"], _EXPECTED_HIGH_BITS[name])
                self.assertEqual(spec["group_scales"], 8)
                self.assertEqual(spec["group_scale_bits"], _EXPECTED_SCALE_BITS[name])
                self.assertEqual(spec["group_scale_levels"], _EXPECTED_SCALE_LEVELS[name])
                self.assertEqual(spec["delta_flags_per_group"], _EXPECTED_DELTAS[name])
                self.assertEqual(spec["block_scale"], _EXPECTED_BLOCK_SCALE[name])
                self.assertEqual(
                    spec["integer_subdot_fields"], _EXPECTED_SUBDOT_FIELDS[name]
                )
                self.assertEqual(_descriptor_layout(spec), _EXPECTED_LAYOUTS[name])
                self.assertEqual(
                    sum(field["bytes"] for field in _descriptor_layout(spec)),
                    _EXPECTED_BLOCK_BYTES[name],
                )
                if name == "IQ1_M":
                    self.assertIsNone(spec["final_scale_fp32"])
                    self.assertEqual(spec["sign_table_entries"], 0)
                else:
                    self.assertEqual(spec["final_scale_fp32"], "0x3e000000")
                    self.assertEqual(spec["sign_table_entries"], 128)

    def test_q8_k_activation_descriptor(self):
        self.assertEqual(set(dut.ACTIVATIONS), {"Q8_K"})
        spec = dut.ACTIVATIONS["Q8_K"]
        self.assertEqual(
            set(spec),
            {"block_elements", "block_bytes", "block_scale", "subblock_elements", "layout"},
        )
        self.assertEqual(spec["block_elements"], 256)
        self.assertEqual(spec["block_bytes"], 292)
        self.assertEqual(spec["block_scale"], "fp32")
        self.assertEqual(spec["subblock_elements"], 16)
        self.assertEqual(_descriptor_layout(spec), _EXPECTED_LAYOUTS["Q8_K"])
        self.assertEqual(sum(field["bytes"] for field in _descriptor_layout(spec)), 292)

    def test_block_layout_reports_contiguous_fields(self):
        for name in _FORMATS + ("Q8_K",):
            rows = dut.block_layout(name)
            with self.subTest(format=name):
                self.assertEqual(rows, _EXPECTED_LAYOUTS[name])
                cursor = 0
                for row in rows:
                    self.assertEqual(set(row), {"field", "offset", "bytes"})
                    self.assertEqual(row["offset"], cursor)
                    cursor += row["bytes"]
                self.assertEqual(cursor, _EXPECTED_BLOCK_BYTES[name])

    def test_block_layout_rejects_unknown_names(self):
        for bad in ("Q2_0", "IQ1_S", "iq2_xxs", "", None, 22, b"IQ2_S"):
            with self.subTest(value=bad):
                with self.assertRaises(dut.FormatError) as context:
                    dut.block_layout(bad)
                self.assertEqual(context.exception.code, "unsupported_format")

    def test_payload_limits(self):
        self.assertEqual(dut.MAX_ELEMENTS, MAX_ELEMENTS)
        self.assertEqual(dut.MAX_IQ_PAYLOAD_BYTES, (MAX_ELEMENTS // 256) * 292)
        self.assertEqual(dut.MAX_IQ_PAYLOAD_BYTES, 74752)

    def test_codebooks_are_byte_identical_to_the_pinned_sources(self):
        streams = {
            "kmask_iq2xs": bytes(t.KMASK_IQ2XS),
            "ksigns_iq2xs": bytes(t.KSIGNS_IQ2XS),
            "iq2xxs_grid": b"".join(struct.pack("<Q", word) for word in t.IQ2XXS_GRID),
            "iq2s_grid": b"".join(struct.pack("<Q", word) for word in t.IQ2S_GRID),
            "iq1s_grid": b"".join(struct.pack("<Q", word) for word in t.IQ1S_GRID),
        }
        for name, data in streams.items():
            with self.subTest(table=name):
                self.assertEqual(
                    hashlib.sha256(data).hexdigest(), _EXPECTED_TABLE_SHA256[name]
                )
                self.assertEqual(t.IQ_TABLE_SHA256[name], _EXPECTED_TABLE_SHA256[name])

    def test_codebook_domains_and_signedness(self):
        self.assertEqual(len(t.KMASK_IQ2XS), 8)
        self.assertEqual(list(t.KMASK_IQ2XS), [1 << column for column in range(8)])
        self.assertEqual(len(t.KSIGNS_IQ2XS), 128)
        # ggml folds the eighth column sign into bit 7 as the parity of the
        # 7-bit lane code, so the table spans the full byte domain.
        self.assertEqual(len(set(t.KSIGNS_IQ2XS)), 128)
        for index, value in enumerate(t.KSIGNS_IQ2XS):
            parity = bin(index).count("1") & 1
            self.assertEqual(value, index | (parity << 7))
        self.assertTrue(all(0 <= value <= 255 for value in t.KSIGNS_IQ2XS))
        self.assertEqual(len(t.IQ2XXS_GRID), 256)
        self.assertEqual(len(t.IQ2S_GRID), 1024)
        self.assertEqual(len(t.IQ1S_GRID), t.NGRID_IQ1S)
        byte_domains = set()
        for word in list(t.IQ2XXS_GRID) + list(t.IQ2S_GRID):
            byte_domains.update(t.grid_bytes(word))
        self.assertEqual(byte_domains, {8, 25, 43})
        iq1_domains = set()
        for word in t.IQ1S_GRID:
            iq1_domains.update(t.grid_bytes(word))
        self.assertEqual(iq1_domains, {0, 1, 255})
        for word in t.IQ1S_GRID:
            signed = t.grid_signed(word)
            self.assertTrue(all(value in (-1, 0, 1) for value in signed))
        self.assertEqual(t.grid_bytes(t.IQ2XXS_GRID[0]), (8,) * 8)
        self.assertEqual(t.grid_signed(t.IQ1S_GRID[0]), (-1,) * 8)
        self.assertEqual(t.grid_signed(t.IQ1S_GRID[1]), (1, -1, -1, -1, -1, -1, -1, -1))


class DecodeIq2XxsTests(unittest.TestCase):
    def test_every_grid_index_and_every_block_scale(self):
        for scale_bits in (HALF_ONE, 0x3800):
            magnitudes = [(index * 5) % 16 for index in range(8)]
            indices = list(range(256))
            sign_indices = [(index * 37) % 128 for index in range(32)]
            payload = _pack_xxs(scale_bits, magnitudes, indices, sign_indices)
            expected = _expected_xxs_values(scale_bits, magnitudes, indices, sign_indices)
            self.assertEqual(_decode_bits("IQ2_XXS", payload, 256), expected)

    def test_all_sixteen_magnitudes_and_all_lane_scales(self):
        magnitudes = list(range(8)) * 2
        payload = _pack_xxs(HALF_ONE, magnitudes, [0] * 32, [0] * 32)
        expected = _expected_xxs_values(HALF_ONE, magnitudes, [0] * 32, [0] * 32)
        self.assertEqual(_decode_bits("IQ2_XXS", payload, 256), expected)

    def test_all_128_sign_patterns_flip_only_the_kmask_columns(self):
        # Grid 0 is (8, ..., 8); d = 1 and magnitude 0 make db = 1/8, so every
        # element is exactly +1 or -1 and the sign table is the only variable.
        for sign_index in range(128):
            signs = [0] * 32
            signs[5] = sign_index
            payload = _pack_xxs(HALF_ONE, [0] * 8, [0] * 32, signs)
            values = _decode_bits("IQ2_XXS", payload, 256)
            mask = t.KSIGNS_IQ2XS[sign_index]
            for column in range(8):
                expected = 0xBF800000 if mask & t.KMASK_IQ2XS[column] else 0x3F800000
                self.assertEqual(
                    values[5 * 8 + column],
                    expected,
                    "sign index %d column %d" % (sign_index, column),
                )

    def test_every_one_of_the_32_sign_byte_positions_is_read(self):
        for lane_position in range(32):
            for sign_index in (1, 64, 127):
                signs = [0] * 32
                signs[lane_position] = sign_index
                payload = _pack_xxs(HALF_ONE, [0] * 8, [0] * 32, signs)
                values = _decode_bits("IQ2_XXS", payload, 256)
                mask = t.KSIGNS_IQ2XS[sign_index]
                for column in range(8):
                    expected = (
                        0xBF800000 if mask & t.KMASK_IQ2XS[column] else 0x3F800000
                    )
                    self.assertEqual(values[lane_position * 8 + column], expected)

    def test_scale_corners_include_subnormals_and_signed_zero(self):
        for scale_bits in (
            0x0001,
            0x03FF,
            0x0400,
            0x3555,
            0x3C00,
            0xBC00,
            0x7BFF,
            0xFBFF,
            0x8000,
            0x0000,
        ):
            payload = _pack_xxs(scale_bits, [0] * 8, [0] * 32, [0] * 32)
            expected = _expected_xxs_values(scale_bits, [0] * 8, [0] * 32, [0] * 32)
            with self.subTest(scale=hex(scale_bits)):
                self.assertEqual(_decode_bits("IQ2_XXS", payload, 256), expected)

    def test_nonfinite_half_scale_is_rejected(self):
        for scale_bits in (0x7C00, 0xFC00, 0x7E00, 0x7FFF, 0xFFFF):
            payload = _pack_xxs(scale_bits, [0] * 8, [0] * 32, [0] * 32)
            with self.subTest(scale=hex(scale_bits)):
                with self.assertRaises(dut.FormatError) as context:
                    dut.decode("IQ2_XXS", payload, 256)
                self.assertEqual(context.exception.code, "nonfinite_scale")

    def test_multi_block_concatenation(self):
        blocks = [_pack_xxs(HALF_ONE, [index % 16] * 8, [index] * 32, [0] * 32) for index in range(3)]
        expected = []
        for index in range(3):
            expected.extend(
                _expected_xxs_values(HALF_ONE, [index % 16] * 8, [index] * 32, [0] * 32)
            )
        result = dut.decode("IQ2_XXS", b"".join(blocks), 768)
        self.assertEqual(result["elements"], 768)
        self.assertEqual(result["blocks"], 3)
        self.assertEqual([int(bits, 16) for bits in result["fp32_bits"]], expected)


class DecodeIq2STests(unittest.TestCase):
    def test_every_ten_bit_grid_index(self):
        for high_part in (0, 1, 2, 3):
            indices = [high_part * 256 + index for index in range(256)]
            index_low = [index & 0xFF for index in indices]
            index_high = [(index >> 8) & 0x3 for index in indices]
            payload = _pack_iq2_s(
                HALF_ONE, [0] * 8, [0] * 8, index_low, index_high, [0] * 32
            )
            expected = _expected_iq2_s_values(
                HALF_ONE, [0] * 8, [0] * 8, index_low, index_high, [0] * 32
            )
            with self.subTest(high=high_part):
                self.assertEqual(_decode_bits("IQ2_S", payload, 256), expected)

    def test_high_index_bits_are_private_to_each_lane(self):
        index_low = [0] * 32
        index_high = [0] * 32
        for lane_position in range(32):
            index_high[lane_position] = 0x3
            payload = _pack_iq2_s(
                HALF_ONE, [0] * 8, [0] * 8, index_low, index_high, [0] * 32
            )
            expected = _expected_iq2_s_values(
                HALF_ONE, [0] * 8, [0] * 8, index_low, index_high, [0] * 32
            )
            self.assertEqual(_decode_bits("IQ2_S", payload, 256), expected)
            index_high[lane_position] = 0

    def test_all_low_and_high_scale_nibbles(self):
        for nibble in range(16):
            low = [nibble] * 8
            high = [15 - nibble] * 8
            payload = _pack_iq2_s(
                HALF_ONE, low, high, [0] * 32, [0] * 32, [0] * 32
            )
            expected = _expected_iq2_s_values(
                HALF_ONE, low, high, [0] * 32, [0] * 32, [0] * 32
            )
            with self.subTest(nibble=nibble):
                self.assertEqual(_decode_bits("IQ2_S", payload, 256), expected)

    def test_all_256_sign_byte_masks_drive_the_columns(self):
        # IQ2_S stores a raw sign *mask* per 8-value lane (unlike IQ2_XXS,
        # whose 7-bit value indexes ksigns_iq2xs), so every byte value must
        # independently control the eight columns.
        for sign_byte in range(256):
            signs = [0] * 32
            signs[0] = sign_byte
            payload = _pack_iq2_s(
                HALF_ONE, [0] * 8, [0] * 8, [0] * 32, [0] * 32, signs
            )
            values = _decode_bits("IQ2_S", payload, 256)
            for column in range(8):
                expected = (
                    0xBF800000 if sign_byte & t.KMASK_IQ2XS[column] else 0x3F800000
                )
                self.assertEqual(values[column], expected)

    def test_each_of_the_32_sign_bytes_is_independent(self):
        for lane_position in range(32):
            for sign_byte in (1, 64, 128, 255):
                signs = [0] * 32
                signs[lane_position] = sign_byte
                payload = _pack_iq2_s(
                    HALF_ONE, [0] * 8, [0] * 8, [0] * 32, [0] * 32, signs
                )
                values = _decode_bits("IQ2_S", payload, 256)
                base = lane_position * 8
                for column in range(8):
                    expected = (
                        0xBF800000 if sign_byte & t.KMASK_IQ2XS[column] else 0x3F800000
                    )
                    self.assertEqual(values[base + column], expected)

    def test_scale_corners_and_nonfinite_rejection(self):
        for scale_bits in (0x0001, 0x0400, 0x3C00, 0xBC00, 0x7BFF, 0x8000):
            payload = _pack_iq2_s(
                scale_bits, [0] * 8, [0] * 8, [0] * 32, [0] * 32, [0] * 32
            )
            expected = _expected_iq2_s_values(
                scale_bits, [0] * 8, [0] * 8, [0] * 32, [0] * 32, [0] * 32
            )
            with self.subTest(scale=hex(scale_bits)):
                self.assertEqual(_decode_bits("IQ2_S", payload, 256), expected)
        for scale_bits in (0x7C00, 0x7E00, 0xFFFF):
            payload = _pack_iq2_s(
                scale_bits, [0] * 8, [0] * 8, [0] * 32, [0] * 32, [0] * 32
            )
            with self.subTest(nonfinite=hex(scale_bits)):
                with self.assertRaises(dut.FormatError) as context:
                    dut.decode("IQ2_S", payload, 256)
                self.assertEqual(context.exception.code, "nonfinite_scale")

    def test_multi_block_stride_is_82_bytes(self):
        blocks = [
            _pack_iq2_s(HALF_ONE, [0] * 8, [0] * 8, [index] * 32, [0] * 32, [0] * 32)
            for index in (0, 255)
        ]
        expected = _expected_iq2_s_values(
            HALF_ONE, [0] * 8, [0] * 8, [0] * 32, [0] * 32, [0] * 32
        ) + _expected_iq2_s_values(
            HALF_ONE, [0] * 8, [0] * 8, [255] * 32, [0] * 32, [0] * 32
        )
        result = dut.decode("IQ2_S", b"".join(blocks), 512)
        self.assertEqual(result["blocks"], 2)
        self.assertEqual([int(bits, 16) for bits in result["fp32_bits"]], expected)


class DecodeIq1MTests(unittest.TestCase):
    def test_every_eleven_bit_grid_index(self):
        for high_part in range(8):
            indices = [high_part * 256 + index for index in range(256)]
            index_low = [index & 0xFF for index in indices]
            index_high = [(index >> 8) & 0x7 for index in indices]
            payload = _pack_iq1_m(
                HALF_ONE, [(0, 0)] * 8, index_low, index_high, [1] * 32
            )
            expected = _expected_iq1_m_values(
                HALF_ONE, [(0, 0)] * 8, index_low, index_high, [1] * 32
            )
            with self.subTest(high=high_part):
                self.assertEqual(_decode_bits("IQ1_M", payload, 256), expected)

    def test_all_eight_group_scale_fields_per_half(self):
        for level in range(8):
            group_fields = [(level, 7 - level)] * 8
            payload = _pack_iq1_m(
                HALF_ONE, group_fields, [0] * 32, [0] * 32, [1] * 32
            )
            expected = _expected_iq1_m_values(
                HALF_ONE, group_fields, [0] * 32, [0] * 32, [1] * 32
            )
            with self.subTest(level=level):
                self.assertEqual(_decode_bits("IQ1_M", payload, 256), expected)

    def test_group_fields_only_affect_their_own_group(self):
        for target in range(8):
            group_fields = [(0, 0)] * 8
            group_fields[target] = (7, 6)
            payload = _pack_iq1_m(
                HALF_ONE, group_fields, [0] * 32, [0] * 32, [1] * 32
            )
            values = _decode_bits("IQ1_M", payload, 256)
            expected = _expected_iq1_m_values(
                HALF_ONE, group_fields, [0] * 32, [0] * 32, [1] * 32
            )
            self.assertEqual(values, expected)

    def test_all_32_delta_flag_positions(self):
        for lane_position in range(32):
            delta_signs = [1] * 32
            delta_signs[lane_position] = -1
            payload = _pack_iq1_m(
                HALF_ONE, [(0, 0)] * 8, [0] * 32, [0] * 32, delta_signs
            )
            values = _decode_bits("IQ1_M", payload, 256)
            expected = _expected_iq1_m_values(
                HALF_ONE, [(0, 0)] * 8, [0] * 32, [0] * 32, delta_signs
            )
            self.assertEqual(values, expected)
            # Grid 0 is (-1, ..., -1), d = 1, ls = 1 and the set flag makes
            # delta -1/8, so y = 1 * (-1 - 1/8) = -1.125.
            self.assertEqual(values[lane_position * 8], 0xBF900000)

    def test_delta_flag_does_not_alias_the_index_high_bits(self):
        # qh nibble bits are: b0..b2 index high bits, b3 delta (low nibble) /
        # b4..b6 index, b7 delta (high nibble).  Setting only a delta bit must
        # keep the grid index at 0.
        for delta_position in (0, 1, 2, 3):
            delta_signs = [1] * 32
            delta_signs[delta_position] = -1
            payload = _pack_iq1_m(
                HALF_ONE, [(0, 0)] * 8, [0] * 32, [0] * 32, delta_signs
            )
            values = _decode_bits("IQ1_M", payload, 256)
            # Set delta flag -> -1/8 -> y = 1 * (-1 - 1/8) = -1.125; a leaked
            # index bit would move the magnitude away from exactly this value.
            self.assertEqual(values[delta_position * 8], 0xBF900000)

    def test_packed_scale_nibbles_reconstruct_finite_halves(self):
        halves = [
            0x0000,
            0x8000,
            0x0001,
            0x03FF,
            0x0400,
            0x3555,
            0x3C00,
            0xBC00,
            0x4B00,
            0x7BFF,
            0xFBFF,
            0x1234,
            0x7B00,
        ]
        for half_bits in halves:
            payload = _pack_iq1_m(
                half_bits, [(0, 0)] * 8, [0] * 32, [0] * 32, [1] * 32
            )
            expected = _expected_iq1_m_values(
                half_bits, [(0, 0)] * 8, [0] * 32, [0] * 32, [1] * 32
            )
            with self.subTest(scale=hex(half_bits)):
                self.assertEqual(_decode_bits("IQ1_M", payload, 256), expected)
            # The packed nibbles must round-trip through the four scale words.
            words = [payload[48 + 2 * index] | (payload[49 + 2 * index] << 8) for index in range(4)]
            rebuilt = (
                (words[0] >> 12)
                | ((words[1] >> 8) & 0x00F0)
                | ((words[2] >> 4) & 0x0F00)
                | (words[3] & 0xF000)
            )
            self.assertEqual(rebuilt, half_bits)
            self.assertEqual(payload[0:48], bytes(48))

    def test_nonfinite_packed_halves_are_rejected(self):
        for half_bits in (0x7C00, 0x7FFF, 0xFC00, 0xFFFF):
            payload = _pack_iq1_m(
                half_bits, [(0, 0)] * 8, [0] * 32, [0] * 32, [1] * 32
            )
            with self.subTest(scale=hex(half_bits)):
                with self.assertRaises(dut.FormatError) as context:
                    dut.decode("IQ1_M", payload, 256)
                self.assertEqual(context.exception.code, "nonfinite_scale")

    def test_multi_block_stride_is_56_bytes(self):
        blocks = [
            _pack_iq1_m(
                HALF_ONE,
                [(0, 0)] * 8,
                [index] * 32,
                [0] * 32,
                [1] * 32,
            )
            for index in (0, 1, 2)
        ]
        expected = []
        for index in (0, 1, 2):
            expected.extend(
                _expected_iq1_m_values(
                    HALF_ONE, [(0, 0)] * 8, [index] * 32, [0] * 32, [1] * 32
                )
            )
        result = dut.decode("IQ1_M", b"".join(blocks), 768)
        self.assertEqual(result["blocks"], 3)
        self.assertEqual([int(bits, 16) for bits in result["fp32_bits"]], expected)


class DotIq2XxsTests(unittest.TestCase):
    def test_analytic_uniform_grid_against_every_activation_pattern(self):
        magnitudes = [0] * 8
        indices = [0] * 32
        signs = [0] * 32
        weight = _pack_xxs(HALF_ONE, magnitudes, indices, signs)
        for pattern in _Q8_PATTERNS:
            q8 = _q8_values(pattern)
            result = dut.dot("IQ2_XXS", weight, _q8_block(q8), 256)
            expected_subdots = _xxs_subdots(magnitudes, indices, signs, q8)
            expected_bits = _dot_bits_xxs([(HALF_ONE, magnitudes, indices, signs)], q8)
            with self.subTest(pattern=pattern):
                self.assertEqual(result["integer_subdots"], [expected_subdots])
                self.assertEqual(int(result["fp32_bits"], 16), expected_bits)

    def test_scale_combination_is_binary32_exact(self):
        magnitudes = [3] * 8
        indices = [37] * 32
        signs = [91] * 32
        weight = _pack_xxs(HALF_ONE, magnitudes, indices, signs)
        q8 = _q8_values("ramp")
        activation = _q8_block(q8, F32_ONE)
        result = dut.dot("IQ2_XXS", weight, activation, 256)
        expected = _dot_bits_xxs([(HALF_ONE, magnitudes, indices, signs)], q8)
        self.assertEqual(int(result["fp32_bits"], 16), expected)
        # Non-power-of-two activation scale: compare against the exact reference.
        for activation_bits in (0x40490FDB, 0x3F490FDB):
            activation = _q8_block(q8, activation_bits)
            result = dut.dot("IQ2_XXS", weight, activation, 256)
            bits, subdots = ref._reference_dot("IQ2_XXS", weight, activation)
            self.assertEqual(result["integer_subdots"], subdots)
            self.assertEqual(result["fp32_bits"], bits)

    def test_multiblock_exact_accumulation(self):
        blocks = []
        payloads = []
        for index in range(8):
            magnitudes = [0] * 8
            indices = [0] * 32
            signs = [0] * 32
            blocks.append((HALF_ONE, magnitudes, indices, signs))
            payloads.append(_pack_xxs(HALF_ONE, magnitudes, indices, signs))
        q8 = _q8_values("equal-positive")
        activation = _q8_block(q8) * 8
        result = dut.dot("IQ2_XXS", b"".join(payloads), activation, 8 * 256)
        self.assertEqual(result["weight_blocks"], 8)
        expected_subdots = [_xxs_subdots(m, i, s, q8) for (_, m, i, s) in blocks]
        self.assertEqual(result["integer_subdots"], expected_subdots)
        self.assertEqual(int(result["fp32_bits"], 16), _dot_bits_xxs(blocks, q8))

    def test_activation_bsums_are_not_consumed(self):
        payload = _pack_xxs(HALF_ONE, [0] * 8, [0] * 32, [0] * 32)
        q8 = _q8_values("extremes")
        base = bytearray(_q8_block(q8))
        changed = bytearray(base)
        for index in range(16):
            changed[260 + 2 * index : 262 + 2 * index] = struct.pack("<h", 12345 - index)
        self.assertEqual(
            dut.dot("IQ2_XXS", payload, bytes(base), 256)["fp32_bits"],
            dut.dot("IQ2_XXS", payload, bytes(changed), 256)["fp32_bits"],
        )

    def test_subdot_shape_and_metadata(self):
        payload = _pack_xxs(HALF_ONE, [0] * 8, [0] * 32, [0] * 32)
        result = dut.dot("IQ2_XXS", payload, _q8_block(_q8_values("zeros")), 256)
        self.assertEqual(result["weight_format"], "IQ2_XXS")
        self.assertEqual(result["activation_format"], "Q8_K")
        self.assertEqual(result["elements"], 256)
        self.assertEqual(result["weight_blocks"], 1)
        self.assertEqual(result["activation_blocks"], 1)
        self.assertEqual(result["arithmetic_id"], "strata-generic-f32-rne-separate-v1")
        self.assertEqual(len(result["integer_subdots"]), 1)
        self.assertEqual(len(result["integer_subdots"][0]), 8)
        self.assertTrue(all(len(group) == 1 for group in result["integer_subdots"][0]))
        self.assertRegex(result["fp32_bits"], r"^0x[0-9a-f]{8}$")

    def test_nonfinite_activation_scale_is_rejected(self):
        payload = _pack_xxs(HALF_ONE, [0] * 8, [0] * 32, [0] * 32)
        for scale_bits in (0x7F800000, 0xFF800000, 0x7FC00000, 0x7FFFFFFF):
            with self.subTest(scale=hex(scale_bits)):
                with self.assertRaises(dut.FormatError) as context:
                    dut.dot("IQ2_XXS", payload, _q8_block([0] * 256, scale_bits), 256)
                self.assertEqual(context.exception.code, "nonfinite_scale")


class DotIq2STests(unittest.TestCase):
    def test_analytic_dot_against_every_activation_pattern(self):
        # Grid 0 is (8, ..., 8) and d = 1 in this case, so every integer group
        # partial sum times 0.125 is exactly representable in binary32.
        low = [2] * 8
        high = [5] * 8
        index_low = [0] * 32
        index_high = [0] * 32
        signs = [3] * 32
        weight = _pack_iq2_s(HALF_ONE, low, high, index_low, index_high, signs)
        for pattern in _Q8_PATTERNS:
            q8 = _q8_values(pattern)
            result = dut.dot("IQ2_S", weight, _q8_block(q8), 256)
            expected_subdots = _iq2_s_subdots(low, high, index_low, index_high, signs, q8)
            expected_bits = _dot_bits_iq2_s(
                [(HALF_ONE, low, high, index_low, index_high, signs)], q8
            )
            with self.subTest(pattern=pattern):
                self.assertEqual(result["integer_subdots"], [expected_subdots])
                self.assertEqual(int(result["fp32_bits"], 16), expected_bits)

    def test_high_scale_half_only_weights_lanes_two_and_three(self):
        index_low = [0] * 32
        index_high = [0] * 32
        signs = [0] * 32
        for group in range(8):
            low = [0] * 8
            high = [0] * 8
            high[group] = 7
            payload = _pack_iq2_s(HALF_ONE, low, high, index_low, index_high, signs)
            q8 = _q8_values("equal-positive")
            result = dut.dot("IQ2_S", payload, _q8_block(q8), 256)
            expected = _iq2_s_subdots(low, high, index_low, index_high, signs, q8)
            self.assertEqual(result["integer_subdots"], [expected])

    def test_multiblock_and_subdot_shape(self):
        low = [1] * 8
        high = [2] * 8
        index_low = [200] * 32
        index_high = [3] * 32
        signs = [120] * 32
        block = _pack_iq2_s(HALF_ONE, low, high, index_low, index_high, signs)
        q8 = _q8_values("alternating")
        result = dut.dot("IQ2_S", block * 3, _q8_block(q8) * 3, 768)
        self.assertEqual(result["weight_blocks"], 3)
        expected_subdots = _iq2_s_subdots(low, high, index_low, index_high, signs, q8)
        self.assertEqual(result["integer_subdots"], [expected_subdots] * 3)
        self.assertTrue(all(len(group) == 2 for group in result["integer_subdots"][0]))

    def test_nonfinite_activation_scale_is_rejected(self):
        block = _pack_iq2_s(HALF_ONE, [0] * 8, [0] * 8, [0] * 32, [0] * 32, [0] * 32)
        with self.assertRaises(dut.FormatError) as context:
            dut.dot("IQ2_S", block, _q8_block([0] * 256, 0x7F800000), 256)
        self.assertEqual(context.exception.code, "nonfinite_scale")


class DotIq1MTests(unittest.TestCase):
    def test_analytic_dot_against_every_activation_pattern(self):
        # Grid 0 is (-1, ..., -1) and d = 1, so the analytic float path below is
        # exactly representable (ls1 = 3, ls2 = 9, grids are unit magnitudes).
        group_fields = [(1, 4)] * 8
        index_low = [0] * 32
        index_high = [0] * 32
        delta_signs = [1, -1] * 16
        weight = _pack_iq1_m(HALF_ONE, group_fields, index_low, index_high, delta_signs)
        for pattern in _Q8_PATTERNS:
            q8 = _q8_values(pattern)
            result = dut.dot("IQ1_M", weight, _q8_block(q8), 256)
            expected_subdots = _iq1_m_subdots(
                group_fields, index_low, index_high, delta_signs, q8
            )
            expected_bits = _dot_bits_iq1_m(
                [(HALF_ONE, group_fields, index_low, index_high, delta_signs)], q8
            )
            with self.subTest(pattern=pattern):
                self.assertEqual(result["integer_subdots"], [expected_subdots])
                self.assertEqual(int(result["fp32_bits"], 16), expected_bits)

    def test_delta_term_matches_signed_activation_sums(self):
        group_fields = [(0, 0)] * 8
        index_low = [0] * 32
        index_high = [0] * 32
        for lane_position in range(32):
            delta_signs = [1] * 32
            delta_signs[lane_position] = -1
            weight = _pack_iq1_m(
                HALF_ONE, group_fields, index_low, index_high, delta_signs
            )
            q8 = _q8_values("ramp")
            result = dut.dot("IQ1_M", weight, _q8_block(q8), 256)
            expected = _iq1_m_subdots(
                group_fields, index_low, index_high, delta_signs, q8
            )
            self.assertEqual(result["integer_subdots"], [expected])

    def test_multiblock_and_subdot_shape(self):
        group_fields = [(7, 0)] * 8
        index_low = [255] * 32
        index_high = [7] * 32
        delta_signs = [-1] * 32
        block = _pack_iq1_m(HALF_ONE, group_fields, index_low, index_high, delta_signs)
        q8 = _q8_values("zeros")
        result = dut.dot("IQ1_M", block * 2, _q8_block(q8) * 2, 512)
        self.assertEqual(result["weight_blocks"], 2)
        expected_subdots = _iq1_m_subdots(
            group_fields, index_low, index_high, delta_signs, q8
        )
        self.assertEqual(result["integer_subdots"], [expected_subdots] * 2)
        self.assertTrue(all(len(group) == 2 for group in result["integer_subdots"][0]))
        self.assertEqual(int(result["fp32_bits"], 16), 0x00000000)

    def test_nonfinite_activation_scale_is_rejected(self):
        block = _pack_iq1_m(HALF_ONE, [(0, 0)] * 8, [0] * 32, [0] * 32, [1] * 32)
        with self.assertRaises(dut.FormatError) as context:
            dut.dot("IQ1_M", block, _q8_block([0] * 256, 0x7FC00000), 256)
        self.assertEqual(context.exception.code, "nonfinite_scale")


def _executable_code(source):
    """Return the module source with every docstring removed (comments are not
    part of the AST), so prose that merely *names* the other half of the oracle
    cannot trip the independence checks."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        body = getattr(node, "body", None)
        if not isinstance(body, list) or not body:
            continue
        first = body[0]
        if (
            isinstance(first, ast.Expr)
            and isinstance(first.value, ast.Constant)
            and isinstance(first.value.value, str)
        ):
            body[0] = ast.Pass()
    return ast.unparse(tree)


def _function_scoped_imports(source, target):
    """Names of the functions that import ``target`` lazily."""
    found = set()
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for child in ast.walk(node):
            if isinstance(child, ast.ImportFrom):
                if any(alias.name == target for alias in child.names):
                    found.add(node.name)
            elif isinstance(child, ast.Import):
                if any(alias.name.split(".")[-1] == target for alias in child.names):
                    found.add(node.name)
    return found


def _module_level_imports(source):
    """Import names bound at import time (lazy in-function imports excluded)."""
    names = set()
    for node in ast.parse(source).body:
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            prefix = node.module or ""
            names.update(
                prefix + "." + alias.name if prefix else alias.name
                for alias in node.names
            )
    return names


class CrossCheckTests(unittest.TestCase):
    """Production path vs the independent reference transcription."""

    def _activation_payload(self, blocks, seed):
        # One Q8_K block per weight block; cycle the deterministic finite
        # activation patterns so consecutive blocks stay dissimilar.
        return b"".join(
            ref._random_activation(
                (_Q8_PATTERNS[(seed + index) % len(_Q8_PATTERNS)],), seed + index
            )
            for index in range(blocks)
        )

    def test_decode_matches_independent_reference_on_random_payloads(self):
        payloads = [
            ("IQ2_XXS", b"".join(_random_xxs_blocks(4, 1001))),
            ("IQ2_S", b"".join(_random_iq2_s_blocks(4, 1002))),
            ("IQ1_M", b"".join(_random_iq1_m_blocks(4, 1003))),
        ]
        for name, payload in payloads:
            elements = (len(payload) // _EXPECTED_BLOCK_BYTES[name]) * 256
            with self.subTest(format=name):
                self.assertEqual(
                    dut.decode(name, payload, elements)["fp32_bits"],
                    ref._reference_decode(name, payload),
                )

    def test_integer_dot_matches_independent_reference_on_random_payloads(self):
        cases = [
            ("IQ2_XXS", b"".join(_random_xxs_blocks(3, 3001))),
            ("IQ2_S", b"".join(_random_iq2_s_blocks(3, 3002))),
            ("IQ1_M", b"".join(_random_iq1_m_blocks(3, 3003))),
        ]
        for name, weight in cases:
            blocks = len(weight) // _EXPECTED_BLOCK_BYTES[name]
            elements = blocks * 256
            activation = self._activation_payload(blocks, 3000)
            with self.subTest(format=name):
                result = dut.dot(name, weight, activation, elements)
                bits, subdots = ref._reference_dot(name, weight, activation)
                self.assertEqual(result["integer_subdots"], subdots)
                self.assertEqual(result["fp32_bits"], bits)

    def test_float_dot_agrees_with_decoded_float_dot(self):
        for name, blocks in (
            ("IQ2_XXS", _random_xxs_blocks(2, 5001)),
            ("IQ2_S", _random_iq2_s_blocks(2, 5002)),
            ("IQ1_M", _random_iq1_m_blocks(2, 5003)),
        ):
            weight = b"".join(blocks)
            for seed in (4001, 4002, 4003):
                activation = self._activation_payload(2, seed)
                with self.subTest(format=name, seed=seed):
                    integer_bits = int(
                        dut.dot(name, weight, activation, 512)["fp32_bits"], 16
                    )
                    decoded_bits = int(
                        ref._reference_decoded_dot(name, weight, activation), 16
                    )
                    ulps = _ulp_distance(integer_bits, decoded_bits)
                    self.assertLessEqual(
                        ulps,
                        4,
                        "%s: integer path 0x%08x vs decoded path 0x%08x (%d ulp)"
                        % (name, integer_bits, decoded_bits, ulps),
                    )

    def test_expected_values_are_generated_without_the_production_path(self):
        """The reference half must not consult the code path under test."""
        probes = (
            "_iq2_xxs_terms",
            "_iq2_s_terms",
            "_iq1_m_terms",
            "_decode_block",
            "_block_dot",
            "_block_dot_iq2_xxs",
            "_block_dot_iq2_s",
            "_block_dot_iq1_m",
            "dot",
            "decode",
        )

        def _explode(*args, **kwargs):
            raise AssertionError("reference generation called a production helper")

        saved = {}
        try:
            for name in probes:
                saved[name] = getattr(dut, name)
                setattr(dut, name, _explode)
            cases = ref._cases()
            self.assertEqual(len(cases), 16)
            for case in cases:
                self.assertIn("expected_fp32_bits", case)
                if case["operation"] == "dot":
                    self.assertIn("expected_integer_subdots", case)
        finally:
            for name, function in saved.items():
                setattr(dut, name, function)

    def test_production_and_reference_are_separate_transcriptions(self):
        production = (_ROOT / "iq_formats.py").read_text(encoding="utf-8")
        reference = (_ROOT / "iq_vectors.py").read_text(encoding="utf-8")
        # Compare executable code only: the production docstring legitimately
        # names iq_vectors.py to document which module holds the reference.
        self.assertNotIn("iq_vectors", _executable_code(production))
        # The reference may import the production module lazily, inside
        # verify_fixture/verify_rtl_bundle (the *verification* direction is
        # allowed; the generation direction is not).  No import-time binding.
        module_imports = _module_level_imports(reference)
        self.assertNotIn("iq_formats", module_imports)
        self.assertNotIn(".iq_formats", module_imports)
        # Nothing may bind the production module at import time; the lazy
        # imports are allowed only inside the two verification functions.
        self.assertEqual(
            _function_scoped_imports(reference, "iq_formats"),
            {"verify_fixture", "verify_rtl_bundle"},
        )
        for helper in (
            ref._cases,
            ref._reference_decode,
            ref._reference_dot,
            ref._reference_decoded_dot,
            ref.generate_fixture,
            ref.generate_rtl_bundle,
        ):
            self.assertNotIn("iq_formats", helper.__code__.co_names)
        self.assertIs(dut._tables, t)
        self.assertIs(ref._tables, t)
        # The reference golden generator must be self-contained.
        self.assertNotIn("dut", ref._cases.__code__.co_names)
        for helper in (ref._reference_decode, ref._reference_dot):
            self.assertNotIn("dut", helper.__code__.co_names)


class ValidationTests(unittest.TestCase):
    def test_unsupported_weight_format(self):
        for name in ("Q2_0", "Q8_0", "IQ1_S", "iq2_s", "", None, 16, b"IQ2_S"):
            with self.subTest(value=name):
                with self.assertRaises(dut.FormatError) as context:
                    dut.decode(name, b"", 256)
                self.assertEqual(context.exception.code, "unsupported_format")

    def test_invalid_elements(self):
        payload = _pack_xxs(HALF_ONE, [0] * 8, [0] * 32, [0] * 32)
        for elements in (0, -256, 255, 257, MAX_ELEMENTS + 256, True, 256.0, "256", None):
            with self.subTest(elements=elements):
                with self.assertRaises(dut.FormatError) as context:
                    dut.decode("IQ2_XXS", payload, elements)
                self.assertEqual(context.exception.code, "invalid_elements")

    def test_invalid_payload_type_or_length(self):
        payload = _pack_xxs(HALF_ONE, [0] * 8, [0] * 32, [0] * 32)
        for bad in (bytearray(payload), memoryview(payload), payload.decode("latin-1"), None):
            with self.subTest(value=type(bad).__name__):
                with self.assertRaises(dut.FormatError) as context:
                    dut.decode("IQ2_XXS", bad, 256)
                self.assertEqual(context.exception.code, "invalid_payload")
        for bad in (payload[:-1], payload + b"\x00", b"", payload * 2):
            with self.subTest(length=len(bad)):
                with self.assertRaises(dut.FormatError) as context:
                    dut.decode("IQ2_XXS", bad, 256)
                self.assertEqual(context.exception.code, "invalid_length")

    def test_parse_hex_contract(self):
        self.assertEqual(dut.parse_hex("00010203ff"), b"\x00\x01\x02\x03\xff")
        self.assertEqual(dut.parse_hex("AABB"), b"\xaa\xbb")
        for bad in ("0", "000", "zz", "00 11", " 00", "00\n", "", "0x00", "00.11", None, 12, b"00"):
            with self.subTest(value=bad):
                with self.assertRaises(dut.FormatError) as context:
                    dut.parse_hex(bad)
                self.assertEqual(context.exception.code, "invalid_hex")

    def test_dot_rejects_other_activation_formats(self):
        payload = _pack_xxs(HALF_ONE, [0] * 8, [0] * 32, [0] * 32)
        activation = _q8_block([0] * 256)
        for name in ("Q8_0", "Q2_0", "q8_k", "", None, 15, b"Q8_K"):
            with self.subTest(value=name):
                with self.assertRaises(dut.FormatError) as context:
                    dut.dot("IQ2_XXS", payload, activation, 256, activation_format=name)
                self.assertEqual(context.exception.code, "unsupported_activation_format")

    def test_dot_requires_matching_block_counts(self):
        payload = _pack_xxs(HALF_ONE, [0] * 8, [0] * 32, [0] * 32)
        for activation_elements in (256, 512):
            with self.subTest(activation_elements=activation_elements):
                with self.assertRaises(dut.FormatError) as context:
                    dut.dot(
                        "IQ2_XXS",
                        payload * 2,
                        _q8_block([0] * 256) * activation_elements,
                        512,
                    )
                self.assertEqual(context.exception.code, "invalid_length")

    def test_decode_result_keys(self):
        payload = _pack_xxs(HALF_ONE, [0] * 8, [0] * 32, [0] * 32)
        result = dut.decode("IQ2_XXS", payload, 256)
        self.assertEqual(
            set(result), {"format", "elements", "blocks", "fp32_bits"}
        )
        self.assertEqual(result["format"], "IQ2_XXS")
        self.assertEqual(result["elements"], 256)
        self.assertEqual(result["blocks"], 1)
        self.assertEqual(len(result["fp32_bits"]), 256)


if __name__ == "__main__":
    unittest.main()
