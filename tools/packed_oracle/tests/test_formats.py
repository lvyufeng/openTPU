# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 lvyufeng and the openTPU contributors

"""Public scalar API, raw lane/scale, bound, and malformed-input tests."""

from fractions import Fraction
import unittest

from tools.packed_oracle import formats as dut
from tools.packed_oracle import vectors as ref


class StringSubclass(str):
    pass


class BytesSubclass(bytes):
    pass


class IntegerSubclass(int):
    pass


class ScalarFormatTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = ref.generate_fixture()
        cls.q2 = ref._q2_block(0x3C00, [0, 1, 2, 3] * 16)
        cls.q8 = ref._q8_block(0x3C00, [-128, -1, 0, 1, 127] * 6 + [0, 1])
        cls.activation = ref._q8_block(0x3C00, [1] * 32)

    def assert_format_error(self, code, function, *args, **kwargs):
        with self.assertRaises(dut.FormatError) as raised:
            function(*args, **kwargs)
        self.assertEqual(raised.exception.code, code)
        self.assertIsInstance(raised.exception.message, str)
        self.assertTrue(raised.exception.message)

    def test_public_format_metadata_exact_and_arithmetic_id(self):
        self.assertEqual(dut.FORMATS, ref._EXPECTED_FORMATS)
        self.assertEqual(dut.ARITHMETIC_ID, "strata-generic-f32-rne-separate-v1")

    def test_parse_contiguous_even_ascii_hex(self):
        for text, expected in (("00", b"\x00"), ("aB09Ff", b"\xab\x09\xff"), ("00807f80", b"\x00\x80\x7f\x80")):
            with self.subTest(text=text):
                self.assertEqual(dut.parse_hex(text), expected)
                self.assertIs(type(dut.parse_hex(text)), bytes)

    def test_parse_hex_rejects_type_prefix_whitespace_unicode_and_odd(self):
        values = (None, True, 42, 1.0, b"00", bytearray(b"00"), [], {}, StringSubclass("00"), "", "0", "001", "0x00", "0X00", " 00", "00 ", "00\n", "0 0", "0\t0", "00:00", "00_00", "GG", "００", "\u200b00")
        for value in values:
            with self.subTest(value=repr(value)):
                self.assert_format_error("invalid_hex", dut.parse_hex, value)

    def test_parse_hex_enforces_row_byte_bound(self):
        largest = "00" * dut.MAX_PAYLOAD_BYTES
        self.assertEqual(len(dut.parse_hex(largest)), dut.MAX_PAYLOAD_BYTES)
        self.assert_format_error("invalid_hex", dut.parse_hex, largest + "00")

    def test_all_decode_fixtures_match_independent_goldens(self):
        count = 0
        for case in self.fixture["cases"]:
            if case["operation"] != "decode":
                continue
            with self.subTest(case=case["id"]):
                result = dut.decode(case["format"], bytes.fromhex(case["payload_hex"]), case["elements"])
                self.assertEqual(set(result), {"format", "elements", "blocks", "fp32_bits"})
                self.assertEqual(result["format"], case["format"])
                self.assertEqual(result["elements"], case["elements"])
                self.assertEqual(result["blocks"], case["elements"] // ref._EXPECTED_FORMATS[case["format"]]["block_elements"])
                self.assertEqual(result["fp32_bits"], case["expected_fp32_bits"])
                count += 1
        self.assertEqual(count, 9)

    def test_q2_signed_zero_and_negative_scale_hard_bits(self):
        probes = {
            0x0000: ["0x80000000", "0x00000000", "0x00000000", "0x00000000"],
            0x8000: ["0x00000000", "0x80000000", "0x80000000", "0x80000000"],
            0xBC00: ["0x3f800000", "0x80000000", "0xbf800000", "0xc0000000"],
        }
        for scale, expected in probes.items():
            with self.subTest(scale=scale):
                payload = ref._q2_block(scale, [0, 1, 2, 3] * 16)
                self.assertEqual(dut.decode("Q2_0", payload, 64)["fp32_bits"], expected * 16)

    def test_q8_signed_zero_and_negative_scale_hard_bits(self):
        lanes = ([-128, -1, 0, 1, 127] * 7)[:32]
        probes = {
            0x0000: (["0x80000000", "0x80000000", "0x00000000", "0x00000000", "0x00000000"] * 7)[:32],
            0x8000: (["0x00000000", "0x00000000", "0x80000000", "0x80000000", "0x80000000"] * 7)[:32],
            0xBC00: (["0x43000000", "0x3f800000", "0x80000000", "0xbf800000", "0xc2fe0000"] * 7)[:32],
        }
        for scale, expected in probes.items():
            with self.subTest(scale=scale):
                self.assertEqual(dut.decode("Q8_0", ref._q8_block(scale, lanes), 32)["fp32_bits"], expected)

    def test_minimum_fp16_subnormal_decode_hard_bits(self):
        q2 = dut.decode("Q2_0", ref._q2_block(1, [0, 1, 2, 3] * 16), 64)
        self.assertEqual(q2["fp32_bits"], ["0xb3800000", "0x00000000", "0x33800000", "0x34000000"] * 16)
        lanes = ([-128, -1, 0, 1, 127] * 7)[:32]
        q8 = dut.decode("Q8_0", ref._q8_block(1, lanes), 32)
        self.assertEqual(q8["fp32_bits"], (["0xb7000000", "0xb3800000", "0x00000000", "0x33800000", "0x36fe0000"] * 7)[:32])

    def test_every_finite_fp16_widens_identically_to_exact_reference(self):
        finite = 0
        nonfinite = 0
        for raw in range(65536):
            if (raw >> 10) & 31 == 31:
                self.assert_format_error("nonfinite_scale", dut._half32, raw)
                nonfinite += 1
            else:
                self.assertEqual(dut._half32(raw), ref._half_to_bits(raw), hex(raw))
                finite += 1
        self.assertEqual((finite, nonfinite), (63488, 2048))

    def test_all_dot_fixtures_match_independent_goldens(self):
        count = 0
        keys = {"weight_format", "activation_format", "elements", "weight_blocks", "activation_blocks", "integer_subdots", "fp32_bits", "arithmetic_id"}
        for case in self.fixture["cases"]:
            if case["operation"] != "dot":
                continue
            with self.subTest(case=case["id"]):
                result = dut.dot(case["weight_format"], bytes.fromhex(case["weight_hex"]), bytes.fromhex(case["activation_hex"]), case["elements"])
                self.assertEqual(set(result), keys)
                self.assertEqual(result["fp32_bits"], case["expected_fp32_bits"])
                self.assertEqual(result["integer_subdots"], case["expected_integer_subdots"])
                self.assertEqual(result["weight_format"], case["weight_format"])
                self.assertEqual(result["activation_format"], "Q8_0")
                self.assertEqual(result["elements"], case["elements"])
                self.assertEqual(result["weight_blocks"], case["elements"] // ref._EXPECTED_FORMATS[case["weight_format"]]["block_elements"])
                self.assertEqual(result["activation_blocks"], case["elements"] // 32)
                self.assertEqual(result["arithmetic_id"], ref.ARITHMETIC_ID)
                count += 1
        self.assertEqual(count, 137)

    def test_signed_int8_dot_extremal_bounds(self):
        q2 = ref._q2_block(0x3C00, [3] * 64)
        q8 = ref._q8_block(0x3C00, [-128] * 32)
        negative = ref._q8_block(0x3C00, [-128] * 32)
        positive = ref._q8_block(0x3C00, [127] * 32)
        self.assertEqual(dut.dot("Q2_0", q2, negative * 2, 64)["integer_subdots"], [[-8192, -8192]])
        self.assertEqual(dut.dot("Q2_0", q2, positive * 2, 64)["integer_subdots"], [[8128, 8128]])
        self.assertEqual(dut.dot("Q8_0", q8, positive, 32)["integer_subdots"], [[-520192]])
        self.assertEqual(dut.dot("Q8_0", q8, negative, 32)["integer_subdots"], [[524288]])

    def test_q2_maps_distinct_activation_halves_and_resets_subtotal(self):
        one = [1] + [0] * 31
        codes = [2] + [1] * 31 + [3] + [1] * 31
        weight = ref._q2_block(0x3800, codes) + ref._q2_block(0xBC00, codes)
        activation = b"".join(ref._q8_block(scale, one) for scale in (0x3400, 0x4000, 0x3C00, 0x3800))
        result = dut.dot("Q2_0", weight, activation, 128)
        # First subtotal=.25*1+2*2=4.25, row=.5*4.25=2.125.
        # Second subtotal=1*1+.5*2=2; new row=2.125-2=.125.
        self.assertEqual(result["integer_subdots"], [[1, 2], [1, 2]])
        self.assertEqual(result["fp32_bits"], "0x3e000000")

    def test_empty_or_partial_or_out_of_bound_dimensions_rejected(self):
        values = (None, True, False, "64", 64.0, IntegerSubclass(64), 0, -1, -64, 1, 31, 33, 63, 65, 65537, 65600)
        for name, payload, elements in (("Q2_0", self.q2, 64), ("Q8_0", self.q8, 32)):
            for value in values:
                with self.subTest(name=name, value=value):
                    self.assert_format_error("invalid_elements", dut.decode, name, payload, value)
                    self.assert_format_error("invalid_elements", dut.dot, name, payload, self.activation * (elements // 32), value)

    def test_payload_types_are_strict_bytes_only(self):
        for name, payload, elements in (("Q2_0", self.q2, 64), ("Q8_0", self.q8, 32)):
            activation = self.activation * (elements // 32)
            for value in (None, True, payload.hex(), list(payload), bytearray(payload), memoryview(payload), BytesSubclass(payload)):
                with self.subTest(name=name, value_type=type(value).__name__):
                    self.assert_format_error("invalid_payload", dut.decode, name, value, elements)
                    self.assert_format_error("invalid_payload", dut.dot, name, value, activation, elements)
                    self.assert_format_error("invalid_payload", dut.dot, name, payload, value, elements)

    def test_bad_lengths_include_trailing_bytes_on_both_dot_inputs(self):
        for name, payload, elements in (("Q2_0", self.q2, 64), ("Q8_0", self.q8, 32)):
            activation = self.activation * (elements // 32)
            for malformed in (b"", payload[:-1], payload[:-2], payload + b"\x00", payload + b"\x00\x00", payload * 2):
                with self.subTest(name=name, weight_length=len(malformed)):
                    self.assert_format_error("invalid_length", dut.decode, name, malformed, elements)
                    self.assert_format_error("invalid_length", dut.dot, name, malformed, activation, elements)
            for malformed in (b"", activation[:-1], activation + b"\x00", activation + b"\x00\x00", activation * 2):
                with self.subTest(name=name, activation_length=len(malformed)):
                    self.assert_format_error("invalid_length", dut.dot, name, payload, malformed, elements)

    def test_unsupported_format_and_non_q8_activation_rejected(self):
        for name in (None, True, 42, [], {}, "q2_0", "Q2_K", "Q4_0", "IQ2_XS", "FP4", StringSubclass("Q2_0")):
            with self.subTest(name=name):
                self.assert_format_error("unsupported_format", dut.decode, name, self.q2, 64)
                self.assert_format_error("unsupported_format", dut.dot, name, self.q2, self.activation * 2, 64)
        for activation_format in (None, True, 8, [], {}, "Q2_0", "Q8_K", "q8_0", StringSubclass("Q8_0")):
            with self.subTest(activation_format=activation_format):
                self.assert_format_error("unsupported_activation_format", dut.dot, "Q2_0", self.q2, self.activation * 2, 64, activation_format=activation_format)

    def test_nonfinite_scale_categories_in_decode_later_block_and_both_dot_inputs(self):
        for name, payload, block_elements, stride in (("Q2_0", self.q2, 64, 18), ("Q8_0", self.q8, 32, 34)):
            elements = block_elements * 2
            weight = payload * 2
            activation = self.activation * (elements // 32)
            for raw in (0x7C00, 0xFC00, 0x7C01, 0xFC01, 0x7E00, 0xFE00, 0x7FFF, 0xFFFF):
                bad_weight = bytearray(weight)
                bad_weight[stride:stride + 2] = raw.to_bytes(2, "little")
                bad_activation = bytearray(activation)
                bad_activation[-34:-32] = raw.to_bytes(2, "little")
                with self.subTest(name=name, raw=raw):
                    self.assert_format_error("nonfinite_scale", dut.decode, name, bytes(bad_weight), elements)
                    self.assert_format_error("nonfinite_scale", dut.dot, name, bytes(bad_weight), activation, elements)
                    self.assert_format_error("nonfinite_scale", dut.dot, name, weight, bytes(bad_activation), elements)

    def test_maximum_elements_is_accepted_without_trailing_bytes(self):
        for name, block in (("Q2_0", ref._q2_block(0x3C00, [1] * 64)), ("Q8_0", ref._q8_block(0x3C00, [0] * 32))):
            elements = 65536
            weight = block * (elements // ref._EXPECTED_FORMATS[name]["block_elements"])
            activation = ref._q8_block(0x3C00, [0] * 32) * (elements // 32)
            with self.subTest(name=name):
                decoded = dut.decode(name, weight, elements)
                self.assertEqual(len(decoded["fp32_bits"]), elements)
                self.assertEqual(set(decoded["fp32_bits"]), {"0x00000000"})
                result = dut.dot(name, weight, activation, elements)
                self.assertEqual(result["fp32_bits"], "0x00000000")
                self.assertEqual(len(result["integer_subdots"]), elements // ref._EXPECTED_FORMATS[name]["block_elements"])
                self.assert_format_error("invalid_length", dut.dot, name, weight + b"\x00", activation, elements)

    def test_bounded_supplied_rows_cannot_overflow_binary32(self):
        # Worst absolute sum uses finite maximum half scale and maximum products.
        maximum = Fraction(65504)
        q8_bound = (65536 // 32) * 524288 * maximum * maximum
        q2_bound = (65536 // 64) * maximum * (2 * 8192 * maximum)
        self.assertLess(q8_bound, Fraction(1 << 64))
        self.assertLess(q2_bound, Fraction(1 << 64))
        self.assertLess(q8_bound, Fraction((1 << 24) - 1) * (1 << 104))

    def test_scalar_binary32_sampled_operations_match_independent_fraction(self):
        probes = [0, 0x80000000, 1, 0x80000001, 0x007FFFFF, 0x00800000, 0x3F800000, 0xBF800000, 0x4B800000, 0x35800000, 0x7F7FFFFF]
        probes.extend((i * 32452843) & 0xFFFFFFFF for i in range(1, 601))
        finite = [bits for bits in probes if bits & 0x7F800000 != 0x7F800000]
        for index, left in enumerate(finite):
            right = finite[(index * 73 + 5) % len(finite)]
            for function, expected in ((dut._add32, ref._add(left, right)), (dut._mul32, ref._multiply(left, right))):
                with self.subTest(left=hex(left), right=hex(right), operation=function.__name__):
                    if expected & 0x7F800000 == 0x7F800000:
                        self.assert_format_error("nonfinite_result", function, left, right)
                    else:
                        self.assertEqual(function(left, right), expected)


if __name__ == "__main__":
    unittest.main()
