# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 lvyufeng and the openTPU contributors

"""Independent golden, binary32 RNE, provenance, and fail-closed fixture tests."""

from copy import deepcopy
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from tools.packed_oracle import formats
from tools.packed_oracle import vectors as v


FIXTURE_PATH = Path(__file__).resolve().parents[1] / "data" / "q2_q8_synthetic.json"
REPORT_KEYS = {
    "schema_version", "fixture_id", "arithmetic_id", "payload_sha256", "case_count",
    "decode_cases", "dot_cases", "synthetic", "verified", "mismatches",
    "device_ready", "inference_ready",
}


def resign(document):
    document["payload_sha256"] = hashlib.sha256(json.dumps(
        {k: value for k, value in document.items() if k != "payload_sha256"},
        sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False,
    ).encode("ascii")).hexdigest()
    return document


class ExactReferenceTests(unittest.TestCase):
    def test_integer_rne_ties_choose_even(self):
        for numerator, denominator, expected in ((1, 2, 0), (3, 2, 2), (5, 2, 2), (7, 2, 4), (10, 3, 3), (11, 3, 4)):
            with self.subTest(numerator=numerator, denominator=denominator):
                self.assertEqual(v._rne_ratio(numerator, denominator), expected)

    def test_binary32_positive_and_negative_zero(self):
        self.assertEqual(v._fraction_to_bits(Fraction(0)), 0)
        self.assertEqual(v._fraction_to_bits(Fraction(0), True), 0x80000000)
        self.assertEqual(v._add(0x80000000, 0x80000000), 0x80000000)
        self.assertEqual(v._add(0, 0x80000000), 0)
        self.assertEqual(v._add(0x3F800000, 0xBF800000), 0)
        self.assertEqual(v._multiply(0, 0xBF800000), 0x80000000)
        self.assertEqual(v._multiply(0x80000000, 0xBF800000), 0)

    def test_binary32_normal_ties_and_carry(self):
        probes = [
            (Fraction(1) + Fraction(1, 1 << 24), 0x3F800000),
            (Fraction(1) + Fraction(3, 1 << 24), 0x3F800002),
            (Fraction(1) + Fraction(5, 1 << 24), 0x3F800002),
            (Fraction(2) - Fraction(1, 1 << 24), 0x40000000),
            (Fraction((1 << 24) + 1), 0x4B800000),
            (Fraction((1 << 24) + 3), 0x4B800002),
        ]
        for value, expected in probes:
            with self.subTest(value=value):
                self.assertEqual(v._fraction_to_bits(value), expected)
                self.assertEqual(v._fraction_to_bits(-value), expected | 0x80000000)

    def test_binary32_gradual_underflow_ties(self):
        probes = [
            (Fraction(1, 1 << 150), 0),
            (Fraction(3, 1 << 150), 2),
            (Fraction(5, 1 << 150), 2),
            (Fraction((1 << 24) - 1, 1 << 150), 0x00800000),
            (Fraction(1, 1 << 126), 0x00800000),
        ]
        for value, expected in probes:
            with self.subTest(value=value):
                self.assertEqual(v._fraction_to_bits(value), expected)
                self.assertEqual(v._fraction_to_bits(-value), expected | 0x80000000)

    def test_binary32_overflow_rounding(self):
        maximum = Fraction((1 << 24) - 1) * (1 << 104)
        self.assertEqual(v._fraction_to_bits(maximum), 0x7F7FFFFF)
        self.assertEqual(v._fraction_to_bits(maximum + (1 << 102)), 0x7F7FFFFF)
        self.assertEqual(v._fraction_to_bits(maximum + (1 << 103)), 0x7F800000)

    def test_binary16_hand_widening(self):
        probes = {
            0x0000: 0x00000000, 0x8000: 0x80000000,
            0x0001: 0x33800000, 0x8001: 0xB3800000,
            0x03FF: 0x387FC000, 0x0400: 0x38800000,
            0x3C00: 0x3F800000, 0xBC00: 0xBF800000,
            0x7BFF: 0x477FE000, 0xFBFF: 0xC77FE000,
        }
        for raw, bits in probes.items():
            with self.subTest(raw=raw):
                self.assertEqual(v._half_to_bits(raw), bits)

    def test_binary16_all_finite_exact_magnitude_roundtrip(self):
        count = 0
        for raw in range(65536):
            if (raw >> 10) & 31 == 31:
                continue
            exponent, fraction = (raw >> 10) & 31, raw & 1023
            magnitude = Fraction(fraction, 1 << 24) if exponent == 0 else Fraction(1024 + fraction) * v._power_two(exponent - 25)
            if raw & 0x8000:
                magnitude = -magnitude
            bits = v._half_to_bits(raw)
            self.assertEqual(v._bits_to_fraction(bits), magnitude, hex(raw))
            self.assertEqual(bits >> 31, raw >> 15, hex(raw))
            count += 1
        self.assertEqual(count, 63488)

    def test_reference_nonfinite_scales_rejected(self):
        for raw in (0x7C00, 0xFC00, 0x7C01, 0x7E00, 0xFE00, 0x7FFF, 0xFFFF):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                v._half_to_bits(raw)

    def test_fixture_generator_does_not_call_dut(self):
        with patch.object(formats, "decode", side_effect=AssertionError("DUT forbidden during generation")), patch.object(formats, "dot", side_effect=AssertionError("DUT forbidden during generation")), patch.object(formats, "FORMATS", {}), patch.object(formats, "ARITHMETIC_ID", "not-golden-metadata"):
            first = v.generate_fixture()
            second = v.generate_fixture()
        self.assertEqual(first, second)
        self.assertEqual(first["arithmetic_id"], "strata-generic-f32-rne-separate-v1")
        self.assertEqual(first["formats"]["Q2_0"]["ggml_type"], 42)

    def test_binary32_roundtrip_sampled_exact_values(self):
        probes = [0, 0x80000000, 1, 0x80000001, 0x007FFFFF, 0x00800000, 0x3F800000, 0x7F7FFFFF]
        probes += [((i * 15485863) & 0x7FFFFFFF) for i in range(1, 2000)]
        for bits in probes:
            if bits & 0x7F800000 == 0x7F800000:
                continue
            with self.subTest(bits=bits):
                self.assertEqual(v._fraction_to_bits(v._bits_to_fraction(bits), bool(bits >> 31)), bits)


class GoldenFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = v.generate_fixture()
        cls.cases = {case["id"]: case for case in cls.fixture["cases"]}

    def test_hard_q2_minus_70_and_integer_subdots(self):
        case = self.cases["q2-hand-minus-70"]
        self.assertEqual(case["expected_integer_subdots"], [[80, -80]])
        self.assertEqual(case["expected_fp32_bits"], "0xc28c0000")
        self.assertEqual(Fraction(1, 2) * (Fraction(1, 4) * 80 + 2 * -80), -70)
        # Supplied lane byte E4 is codes [0,1,2,3], low two bits first.
        self.assertEqual(bytes.fromhex(case["weight_hex"])[2:], bytes([0xE4] * 16))

    def test_hard_q8_378_and_integer_subdot(self):
        case = self.cases["q8-hand-378"]
        self.assertEqual((-128 * -20) + (127 * 3) + (-7 * -9) + (5 * 4), 3024)
        self.assertEqual(case["expected_integer_subdots"], [[3024]])
        self.assertEqual(case["expected_fp32_bits"], "0x43bd0000")
        self.assertEqual(Fraction(1, 2) * Fraction(1, 4) * 3024, 378)

    def test_every_q2_lane_independently_exercised(self):
        cases = [case for case in self.fixture["cases"] if case["id"].startswith("q2-lane-impulse-")]
        self.assertEqual(len(cases), 64)
        for lane, case in enumerate(cases):
            expected_dot = 2 * (lane + 1)
            self.assertEqual(case["expected_integer_subdots"], [[expected_dot, 0] if lane < 32 else [0, expected_dot]])
            value = Fraction(3, 4) * (Fraction(1, 2) if lane < 32 else 2) * expected_dot
            self.assertEqual(case["expected_fp32_bits"], v._hex_bits(v._fraction_to_bits(value)))

    def test_every_q8_lane_and_all_int8_values(self):
        cases = [case for case in self.fixture["cases"] if case["id"].startswith("q8-lane-impulse-")]
        self.assertEqual(len(cases), 32)
        for lane, case in enumerate(cases):
            self.assertEqual(case["expected_integer_subdots"], [[(-128 + lane) * (lane + 1)]])
        signed = self.cases["q8-every-signed-int8"]
        self.assertEqual(signed["elements"], 256)
        expected = [v._hex_bits(v._fraction_to_bits(Fraction(i))) for i in range(-128, 128)]
        self.assertEqual(signed["expected_fp32_bits"], expected)

    def test_little_endian_scale_not_host_endian(self):
        for name in ("q2", "q8"):
            case = self.cases[f"{name}-little-endian-scale"]
            self.assertTrue(case["payload_hex"].startswith("3412"))
            wanted = v._half_to_bits(0x1234)
            wrong = v._half_to_bits(0x3412)
            self.assertNotEqual(wanted, wrong)
            self.assertEqual(case["expected_fp32_bits"], [v._hex_bits(wanted)] * case["elements"])

    def test_multiblock_order_and_combined_sum_change_expected_bits(self):
        lost = self.cases["q8-multiblock-lost-unit"]
        preserved = self.cases["q8-multiblock-preserved-unit"]
        twice = self.cases["q8-multiblock-two-separate-ties"]
        self.assertEqual(lost["expected_fp32_bits"], "0x00000000")
        self.assertEqual(preserved["expected_fp32_bits"], "0x3f800000")
        self.assertEqual(twice["expected_fp32_bits"], "0x4b800000")
        self.assertEqual(v._fraction_to_bits(Fraction((1 << 24) + 1 - (1 << 24))), 0x3F800000)
        self.assertEqual(v._fraction_to_bits(Fraction((1 << 24) + 2)), 0x4B800001)
        # Reverse the exact same packed block pairs; it restores the lost unit.
        weight = bytes.fromhex(lost["weight_hex"])
        activation = bytes.fromhex(lost["activation_hex"])
        order = (0, 2, 1)
        bits, _ = v._reference_dot("Q8_0", b"".join(weight[i * 34:(i + 1) * 34] for i in order), b"".join(activation[i * 34:(i + 1) * 34] for i in order))
        self.assertEqual(bits, "0x3f800000")
        self.assertNotEqual(bits, lost["expected_fp32_bits"])

    def test_q2_subtotal_and_multiblock_rounding_are_observable(self):
        lost = self.cases["q2-multiblock-lost-unit"]
        tie = self.cases["q2-half-subtotal-rne-tie"]
        self.assertEqual(lost["expected_fp32_bits"], "0x00000000")
        self.assertEqual(tie["expected_integer_subdots"], [[4096, 1]])
        self.assertEqual(tie["expected_fp32_bits"], "0x4b802000")
        # A merged exact weighted half-sum keeps a unit the contract discards.
        merged = Fraction(1025, 1024) * ((1 << 24) + 1)
        self.assertEqual(v._fraction_to_bits(merged), 0x4B802001)
        self.assertNotEqual(v._fraction_to_bits(merged), int(tie["expected_fp32_bits"], 16))

    def test_separate_multiply_and_add_are_not_fma(self):
        case = self.cases["q8-no-fma-witness"]
        self.assertEqual(case["expected_integer_subdots"], [[1], [28]])
        self.assertEqual(case["expected_fp32_bits"], "0x41153957")
        factor = v._multiply(v._half_to_bits(0x3555), v._half_to_bits(0x3BFF))
        fused = v._fraction_to_bits(Fraction(1, 1 << 20) + 28 * v._bits_to_fraction(factor))
        self.assertEqual(fused, 0x41153958)
        self.assertNotEqual(fused, int(case["expected_fp32_bits"], 16))

    def test_subnormal_scale_products_not_flushed(self):
        for name in ("q2_0", "q8_0"):
            case = self.cases[f"{name}-subnormal-scales-no-ftz"]
            self.assertEqual(case["expected_fp32_bits"], "0x27800000")  # 2**-48.
            self.assertNotEqual(case["expected_fp32_bits"], "0x00000000")

    def test_shipped_fixture_is_exact_deterministic_serialization(self):
        expected_text = json.dumps(self.fixture, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False) + "\n"
        self.assertEqual(FIXTURE_PATH.read_text(encoding="utf-8"), expected_text)
        self.assertEqual(json.loads(expected_text), self.fixture)
        self.assertLess(len(expected_text.encode("ascii")), v.MAX_FIXTURE_BYTES)

    def test_canonical_digest_is_not_pretty_file_digest(self):
        canonical = json.dumps({k: value for k, value in self.fixture.items() if k != "payload_sha256"}, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")
        self.assertEqual(hashlib.sha256(canonical).hexdigest(), self.fixture["payload_sha256"])
        self.assertNotEqual(hashlib.sha256(FIXTURE_PATH.read_bytes()).hexdigest(), self.fixture["payload_sha256"])

    def test_fixture_matches_dut_and_exact_report_schema(self):
        report = v.verify_fixture(self.fixture)
        self.assertEqual(set(report), REPORT_KEYS)
        self.assertEqual(report["case_count"], 146)
        self.assertEqual(report["decode_cases"], 9)
        self.assertEqual(report["dot_cases"], 137)
        self.assertTrue(report["verified"])
        self.assertEqual(report["mismatches"], [])
        self.assertIs(report["synthetic"], True)
        self.assertIs(report["device_ready"], False)
        self.assertIs(report["inference_ready"], False)

    def test_verifier_does_not_modify_input_document(self):
        snapshot = deepcopy(self.fixture)
        v.verify_fixture(self.fixture)
        self.assertEqual(self.fixture, snapshot)

    def test_digest_is_not_truth_bad_bits_are_mismatch(self):
        document = deepcopy(self.fixture)
        target = next(case for case in document["cases"] if case["id"] == "q8-hand-378")
        target["expected_fp32_bits"] = "0x00000000"
        resign(document)
        report = v.verify_fixture(document)
        self.assertFalse(report["verified"])
        self.assertEqual(report["mismatches"], [{"case_id": "q8-hand-378", "field": "fp32_bits", "expected": "0x00000000", "actual": "0x43bd0000"}])

    def test_digest_is_not_truth_bad_subdot_is_mismatch(self):
        document = deepcopy(self.fixture)
        target = next(case for case in document["cases"] if case["id"] == "q2-hand-minus-70")
        target["expected_integer_subdots"] = [[79, -80]]
        resign(document)
        report = v.verify_fixture(document)
        self.assertFalse(report["verified"])
        self.assertEqual(report["mismatches"], [{"case_id": "q2-hand-minus-70", "field": "integer_subdots", "expected": [[79, -80]], "actual": [[80, -80]]}])


class StrictFixtureValidationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        full = v.generate_fixture()
        cls.fixture = resign({**full, "cases": [deepcopy(full["cases"][0]), deepcopy(next(case for case in full["cases"] if case["id"] == "q2-hand-minus-70")), deepcopy(next(case for case in full["cases"] if case["id"] == "q8-hand-378"))]})

    def reject(self, document, *, sign=True):
        if sign:
            resign(document)
        with patch.object(formats, "decode", side_effect=AssertionError("invalid fixture reached DUT")), patch.object(formats, "dot", side_effect=AssertionError("invalid fixture reached DUT")):
            with self.assertRaises(formats.FormatError) as raised:
                v.verify_fixture(document)
        self.assertIn(raised.exception.code, ("fixture_schema", "fixture_digest"))

    def test_top_exact_keys_and_plain_object_type(self):
        for key in tuple(self.fixture):
            document = deepcopy(self.fixture)
            del document[key]
            with self.subTest(missing=key):
                self.reject(document, sign=False)
        for value in (None, [], (), True, 1, "fixture"):
            with self.subTest(value=value), self.assertRaises(formats.FormatError):
                v.verify_fixture(value)
        document = deepcopy(self.fixture)
        document["device_ready"] = True
        self.reject(document)

    def test_frozen_header_types_values(self):
        mutations = {
            "schema_version": (True, 1.0, 2, "1"),
            "synthetic": (False, 1, "true"),
            "fixture_id": ("wrong", None, True),
            "arithmetic_id": ("fused", None, True),
        }
        for key, values in mutations.items():
            for value in values:
                document = deepcopy(self.fixture)
                document[key] = value
                with self.subTest(key=key, value=value):
                    self.reject(document)

    def test_format_metadata_types_values_and_keys(self):
        for name in ("Q2_0", "Q8_0"):
            for key in self.fixture["formats"][name]:
                document = deepcopy(self.fixture)
                value = document["formats"][name][key]
                document["formats"][name][key] = float(value) if type(value) is int else "bad"
                with self.subTest(name=name, key=key):
                    self.reject(document)
        document = deepcopy(self.fixture)
        document["formats"]["Q2_0"]["unexpected"] = 0
        self.reject(document)
        document = deepcopy(self.fixture)
        document["formats"]["Q8_0"]["block_elements"] = True
        self.reject(document)

    def test_provenance_revision_repository_generator_and_exclusions(self):
        mutations = [
            ("strata", "repository", "https://invalid.example/Strata"),
            ("strata", "revision", "0" * 40),
            ("opentpu_base", "repository", "https://invalid.example/openTPU"),
            ("opentpu_base", "revision", "0" * 40),
            ("generator", "id", "from-dut"),
            ("generator", "version", True),
        ]
        for section, key, value in mutations:
            document = deepcopy(self.fixture)
            document["provenance"][section][key] = value
            with self.subTest(section=section, key=key):
                self.reject(document)
        for key, value in (("ggml_origin_revision", "0" * 40), ("expected_derivation", "DUT output"), ("scope_exclusions", [])):
            document = deepcopy(self.fixture)
            document["provenance"][key] = value
            with self.subTest(key=key):
                self.reject(document)
        document = deepcopy(self.fixture)
        document["provenance"]["scope_exclusions"].reverse()
        self.reject(document)

    def test_provenance_anchor_exact_paths_ranges_and_types(self):
        for index, anchor in enumerate(self.fixture["provenance"]["strata"]["anchors"]):
            for key in anchor:
                document = deepcopy(self.fixture)
                value = anchor[key]
                document["provenance"]["strata"]["anchors"][index][key] = float(value) if type(value) is int else "wrong/path"
                with self.subTest(index=index, key=key):
                    self.reject(document)
        document = deepcopy(self.fixture)
        document["provenance"]["strata"]["anchors"].reverse()
        self.reject(document)

    def test_digest_type_case_and_integrity(self):
        for value in (None, True, 123, "0" * 63, "g" * 64, "A" * 64, "0" * 64):
            document = deepcopy(self.fixture)
            document["payload_sha256"] = value
            with self.subTest(value=value):
                self.reject(document, sign=False)
        document = deepcopy(self.fixture)
        document["cases"][0]["expected_fp32_bits"][0] = "0x00000000"
        self.reject(document, sign=False)

    def test_case_container_count_bound(self):
        for value in (None, {}, (), [], True):
            document = deepcopy(self.fixture)
            document["cases"] = value
            with self.subTest(value=value):
                self.reject(document)
        document = deepcopy(self.fixture)
        document["cases"] = [document["cases"][0]] * 257
        self.reject(document)

    def test_case_id_operation_and_exact_case_keys(self):
        for value in ("", "UPPER", "has space", "x" * 81, None, True, 5, "abc\n"):
            document = deepcopy(self.fixture)
            document["cases"][0]["id"] = value
            with self.subTest(value=value):
                self.reject(document)
        document = deepcopy(self.fixture)
        document["cases"][1]["id"] = document["cases"][0]["id"]
        self.reject(document)
        for index in range(3):
            for key in self.fixture["cases"][index]:
                document = deepcopy(self.fixture)
                del document["cases"][index][key]
                with self.subTest(index=index, key=key):
                    self.reject(document)
            document = deepcopy(self.fixture)
            document["cases"][index]["extra"] = True
            self.reject(document)
        for operation in ("quantize", "Decode", True, None):
            document = deepcopy(self.fixture)
            document["cases"][0]["operation"] = operation
            self.reject(document)

    def test_element_type_range_and_whole_blocks(self):
        for index in range(3):
            for value in (True, False, None, "64", 64.0, -64, 0, 1, 31, 33, 63, 65, 65537, 65600):
                document = deepcopy(self.fixture)
                document["cases"][index]["elements"] = value
                with self.subTest(index=index, value=value):
                    self.reject(document)

    def test_format_and_activation_type_value(self):
        for index, key in ((0, "format"), (1, "weight_format"), (2, "weight_format"), (1, "activation_format")):
            for value in (True, None, 42, [], {}, "q2_0", "IQ2_XS", "FP4", "Q8_K"):
                document = deepcopy(self.fixture)
                document["cases"][index][key] = value
                with self.subTest(index=index, key=key, value=value):
                    self.reject(document)

    def test_hex_type_length_whitespace_prefix_unicode_and_trailing(self):
        for index, key in ((0, "payload_hex"), (1, "weight_hex"), (1, "activation_hex"), (2, "weight_hex"), (2, "activation_hex")):
            original = self.fixture["cases"][index][key]
            for value in (None, True, 0, [], "", original[:-1], original[:-2], original + "00", "0x" + original, " " + original, original + "\n", original[:2] + " " + original[2:], "z" + original[1:], "０" + original[1:]):
                document = deepcopy(self.fixture)
                document["cases"][index][key] = value
                with self.subTest(index=index, key=key, value=str(value)[:30]):
                    self.reject(document)

    def test_all_nonfinite_half_categories_and_later_block_are_rejected(self):
        for index, key, stride in ((0, "payload_hex", 18), (1, "weight_hex", 18), (1, "activation_hex", 34), (2, "weight_hex", 34), (2, "activation_hex", 34)):
            for raw in (0x7C00, 0xFC00, 0x7C01, 0x7E00, 0xFE00, 0x7FFF):
                document = deepcopy(self.fixture)
                payload = bytearray.fromhex(document["cases"][index][key])
                offset = stride if len(payload) > stride else 0
                payload[offset:offset + 2] = raw.to_bytes(2, "little")
                document["cases"][index][key] = payload.hex()
                with self.subTest(index=index, key=key, raw=raw):
                    self.reject(document)

    def test_expected_bits_exact_type_length_finite_and_lowercase(self):
        for value in (None, True, 5, "0x00000000", (), [], ["0x00000000"] * 63, ["0x00000000"] * 65):
            document = deepcopy(self.fixture)
            document["cases"][0]["expected_fp32_bits"] = value
            self.reject(document)
        bad_bits = (True, None, 0, "", "00000000", "0X00000000", "0x3F800000", "0x0000000", "0x000000000", "0x00000000\n", "0x7f800000", "0xff800000", "0x7fc00000", "0xffffffff")
        for value in bad_bits:
            for index in (0, 1, 2):
                document = deepcopy(self.fixture)
                if index == 0:
                    document["cases"][0]["expected_fp32_bits"][0] = value
                else:
                    document["cases"][index]["expected_fp32_bits"] = value
                with self.subTest(index=index, value=value):
                    self.reject(document)

    def test_integer_subdot_shape_type_and_mathematical_bounds(self):
        mutations = [None, True, 0, (), [], [80, -80], [[80]], [[80, -80, 0]], [[80, -80], [0, 0]], [[True, -80]], [[80.0, -80]], [["80", -80]], [[-8193, -80]], [[8129, -80]]]
        for value in mutations:
            document = deepcopy(self.fixture)
            document["cases"][1]["expected_integer_subdots"] = value
            with self.subTest(value=value):
                self.reject(document)
        for value in (None, [], [[3024, 0]], [[-520193]], [[524289]], [[True]], [[3024.0]]):
            document = deepcopy(self.fixture)
            document["cases"][2]["expected_integer_subdots"] = value
            with self.subTest(value=value):
                self.reject(document)

    def test_fixture_larger_than_8mib_is_rejected_before_dut(self):
        document = deepcopy(self.fixture)
        document["cases"][0]["payload_hex"] = "0" * (v.MAX_FIXTURE_BYTES + 1)
        self.reject(document)

    def test_non_json_nan_huge_integer_and_recursive_cases_fail_closed(self):
        for value in (float("nan"), float("inf"), float("-inf"), 10 ** 5000):
            document = deepcopy(self.fixture)
            document["cases"][0]["elements"] = value
            self.reject(document, sign=False)
        document = deepcopy(self.fixture)
        document["cases"][0]["expected_fp32_bits"] = document["cases"]
        self.reject(document, sign=False)

    def test_complete_validation_pass_precedes_any_dut_execution(self):
        document = deepcopy(self.fixture)
        document["cases"][-1]["expected_fp32_bits"] = "0x7f800000"
        self.reject(document)

    def test_uppercase_payload_hex_allowed_but_digest_changes(self):
        document = deepcopy(self.fixture)
        for case in document["cases"]:
            for key in ("payload_hex", "weight_hex", "activation_hex"):
                if key in case:
                    case[key] = case[key].upper()
        resign(document)
        report = v.verify_fixture(document)
        self.assertTrue(report["verified"])

    def test_public_mutable_dut_metadata_cannot_weaken_schema(self):
        document = deepcopy(self.fixture)
        document["formats"]["Q2_0"]["ggml_type"] = 123
        resign(document)
        with patch.object(formats, "FORMATS", document["formats"]):
            self.reject(document)


if __name__ == "__main__":
    unittest.main()
