# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 lvyufeng and the openTPU contributors

"""Frozen IQ fixture/RTL-bundle contracts and independent generation checks.

These tests cover the shipped JSON artifacts, the strict schema/verifier, and
the exact integer coverage of the synthetic grid sweeps.  They never accept a
DUT value as an expectation: every expected bit string in the shipped files is
produced by ``iq_vectors`` and compared against the independent transcription.
"""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from tools.packed_oracle import iq_formats as dut
from tools.packed_oracle import iq_tables as tables
from tools.packed_oracle import iq_vectors as ref


DATA = Path(__file__).resolve().parents[1] / "data"
FIXTURE_PATH = DATA / "iq_synthetic.json"
RTL_PATH = DATA / "iq_rtl_vectors.json"

FIXTURE_SHA256 = "b4bb7529b476e295ba92ecd1340ba35c81dc69f4f2a09a16a94e22b210cb9a4f"
RTL_SHA256 = "0cefcf13c276569790780ce7ab30026e666007269531c60b550e119f3ab0612f"

FORMAT_NAMES = ("IQ2_XXS", "IQ2_S", "IQ1_M")
GRID_SIZES = {"IQ2_XXS": 256, "IQ2_S": 1024, "IQ1_M": 2048}
SWEEP_BLOCKS = {"IQ2_XXS": 8, "IQ2_S": 32, "IQ1_M": 64}
SWEEP_IDS = {
    "IQ2_XXS": "iq2-xxs-grid-sweep",
    "IQ2_S": "iq2-s-grid-sweep",
    "IQ1_M": "iq1-m-grid-sweep",
}

VERIFY_KEYS = {
    "schema_version",
    "fixture_id",
    "arithmetic_id",
    "payload_sha256",
    "case_count",
    "decode_cases",
    "dot_cases",
    "synthetic",
    "verified",
    "mismatches",
    "device_ready",
    "inference_ready",
}
RTL_VERIFY_KEYS = {
    "schema_version",
    "fixture_id",
    "arithmetic_id",
    "payload_sha256",
    "case_count",
    "synthetic",
    "verified",
    "mismatches",
    "device_ready",
    "inference_ready",
}
MISMATCH_KEYS = {"case_id", "field", "expected", "actual"}
RTL_CASE_KEYS = {
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


def _load(path):
    return json.loads(path.read_bytes().decode("ascii"))


def _pretty(document):
    return (
        json.dumps(document, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False) + "\n"
    ).encode("ascii")


def _resign(document):
    document["payload_sha256"] = ref._digest(document)


def _sweep_indices(name, payload):
    """Collect every grid index the sweep case actually encodes."""
    block_bytes = ref._EXPECTED_FORMATS[name]["block_bytes"]
    indices = set()
    for start in range(0, len(payload), block_bytes):
        block = payload[start : start + block_bytes]
        if name == "IQ2_XXS":
            _, groups = ref._xxs_terms(block)
            for _, lane_indices, _ in groups:
                indices.update(lane_indices)
        elif name == "IQ2_S":
            _, groups = ref._iq2_s_terms(block)
            for _, _, rows in groups:
                indices.update(index for index, _ in rows)
        else:
            _, groups = ref._iq1_m_terms(block)
            for _, _, lanes in groups:
                indices.update(index for index, _ in lanes)
    return indices


class ShippedFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = FIXTURE_PATH.read_bytes()
        cls.document = _load(FIXTURE_PATH)
        cls.generated = ref.generate_fixture()
        cls.report = ref.verify_fixture(cls.document)
        cls.cases = {case["id"]: case for case in cls.document["cases"]}

    def test_fixture_is_canonical_pretty_json_with_trailing_newline(self):
        self.assertEqual(self.raw, _pretty(self.document))
        self.assertEqual(self.raw, _pretty(self.generated))

    def test_fixture_regeneration_is_byte_identical_and_deterministic(self):
        self.assertEqual(self.document, self.generated)
        self.assertEqual(ref._digest(self.document), ref._digest(self.generated))
        self.assertEqual(self.document["payload_sha256"], ref._digest(self.document))

    def test_fixture_payload_digest_is_frozen_literal(self):
        self.assertEqual(self.document["payload_sha256"], FIXTURE_SHA256)
        self.assertEqual(len(self.raw), 727402)
        canonical = ref._canonical_bytes(self.document, omit_digest=False)
        self.assertEqual(len(canonical), 440336)
        self.assertEqual(
            hashlib.sha256(ref._canonical_bytes(self.document, omit_digest=True)).hexdigest(),
            FIXTURE_SHA256,
        )

    def test_fixture_metadata_matches_frozen_contract(self):
        self.assertEqual(set(self.document), ref._TOP_KEYS)
        self.assertEqual(self.document["schema_version"], 1)
        self.assertEqual(self.document["fixture_id"], "iq-packed-supplied-v1")
        self.assertIs(self.document["synthetic"], True)
        self.assertEqual(self.document["arithmetic_id"], ref.ARITHMETIC_ID)
        self.assertEqual(self.document["formats"], ref._EXPECTED_FORMATS)
        self.assertEqual(self.document["activations"], ref._EXPECTED_ACTIVATIONS)
        self.assertEqual(self.document["provenance"], ref._EXPECTED_PROVENANCE)
        self.assertEqual(self.document["provenance"]["scope_exclusions"], ref._SCOPE_EXCLUSIONS)
        self.assertEqual(
            self.document["provenance"]["generator"]["production_path_under_test"],
            "tools/packed_oracle/iq_formats.py",
        )
        self.assertEqual(
            self.document["provenance"]["generator"]["independent_reference"],
            "tools/packed_oracle/iq_vectors.py",
        )

    def test_fixture_case_inventory_is_frozen(self):
        ids = [case["id"] for case in self.document["cases"]]
        self.assertEqual(len(ids), 16)
        self.assertEqual(len(set(ids)), 16)
        self.assertEqual(
            [case["operation"] for case in self.document["cases"]],
            ["decode"] * 6 + ["dot"] * 10,
        )
        for name in FORMAT_NAMES:
            self.assertEqual(sum(1 for case in self.document["cases"] if case.get("format") == name), 2)
            self.assertEqual(
                sum(1 for case in self.document["cases"] if case.get("weight_format") == name),
                4 if name == "IQ2_XXS" else 3,
            )

    def test_verify_report_is_exact(self):
        self.assertEqual(set(self.report), VERIFY_KEYS)
        self.assertIs(self.report["verified"], True)
        self.assertEqual(self.report["mismatches"], [])
        self.assertEqual(self.report["case_count"], 16)
        self.assertEqual(self.report["decode_cases"], 6)
        self.assertEqual(self.report["dot_cases"], 10)
        self.assertEqual(self.report["fixture_id"], "iq-packed-supplied-v1")
        self.assertEqual(self.report["payload_sha256"], FIXTURE_SHA256)
        self.assertIs(self.report["device_ready"], False)
        self.assertIs(self.report["inference_ready"], False)

    def test_sweep_cases_cover_every_grid_index(self):
        for name in FORMAT_NAMES:
            case = self.cases[SWEEP_IDS[name]]
            payload = bytes.fromhex(case["payload_hex"])
            self.assertEqual(len(payload), SWEEP_BLOCKS[name] * ref._EXPECTED_FORMATS[name]["block_bytes"])
            indices = _sweep_indices(name, payload)
            self.assertEqual(indices, set(range(GRID_SIZES[name])))
            self.assertEqual(len(case["expected_fp32_bits"]), len(payload) // ref._EXPECTED_FORMATS[name]["block_bytes"] * 256)

    def test_every_format_has_both_scale_corner_cases(self):
        for name in FORMAT_NAMES:
            ids = [case["id"] for case in self.document["cases"] if case.get("format") == name]
            self.assertEqual(len(ids), 2)
            self.assertIn("-grid-sweep", ids[0])
            self.assertIn("-scale-corners", ids[1])

    def test_integer_subdots_have_one_row_per_block_and_eight_groups(self):
        for case in self.document["cases"]:
            if case["operation"] != "dot":
                continue
            name = case["weight_format"]
            blocks = case["elements"] // 256
            subdots = case["expected_integer_subdots"]
            self.assertEqual(len(subdots), blocks)
            columns = len(ref._EXPECTED_FORMATS[name]["integer_subdot_fields"])
            for row in subdots:
                self.assertEqual(len(row), 8)
                for group in row:
                    self.assertEqual(len(group), columns)
                    for term in group:
                        self.assertIs(type(term), int)
                        low, high = ref._integer_subdot_bounds(name)
                        self.assertGreaterEqual(term, low)
                        self.assertLessEqual(term, high)


class FixtureTamperTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.document = _load(FIXTURE_PATH)

    def test_expected_bit_flip_is_outside_the_input_error_channel(self):
        document = deepcopy(self.document)
        case = document["cases"][0]
        case["expected_fp32_bits"][0] = "0xdeadbeef"
        _resign(document)
        report = ref.verify_fixture(document)
        self.assertIs(report["verified"], False)
        self.assertEqual(len(report["mismatches"]), 1)
        mismatch = report["mismatches"][0]
        self.assertEqual(set(mismatch), MISMATCH_KEYS)
        self.assertEqual(mismatch["case_id"], case["id"])
        self.assertEqual(mismatch["field"], "fp32_bits")
        self.assertEqual(mismatch["expected"][0], "0xdeadbeef")
        self.assertEqual(mismatch["actual"][0], self.document["cases"][0]["expected_fp32_bits"][0])

    def test_integer_subdot_tamper_is_reported(self):
        document = deepcopy(self.document)
        case = next(case for case in document["cases"] if case["operation"] == "dot")
        case["expected_integer_subdots"][0][0][0] += 1
        _resign(document)
        report = ref.verify_fixture(document)
        self.assertIs(report["verified"], False)
        self.assertEqual(report["mismatches"][0]["field"], "integer_subdots")
        self.assertEqual(report["mismatches"][0]["case_id"], case["id"])

    def test_unsound_digest_is_an_input_error_not_a_false_pass(self):
        document = deepcopy(self.document)
        document["cases"][0]["expected_fp32_bits"][0] = "0xdeadbeef"
        with self.assertRaises(ref.FormatError) as caught:
            ref.verify_fixture(document)
        self.assertEqual(caught.exception.code, "fixture_digest")

    def test_nonfinite_scale_is_rejected_after_resigning(self):
        document = deepcopy(self.document)
        case = document["cases"][0]
        payload = bytearray.fromhex(case["payload_hex"])
        payload[0:2] = b"\x00\x7c"  # binary16 +infinity
        case["payload_hex"] = bytes(payload).hex()
        _resign(document)
        with self.assertRaises(ref.FormatError) as caught:
            ref.verify_fixture(document)
        self.assertEqual(caught.exception.code, "fixture_schema")

    def test_duplicate_case_id_is_rejected(self):
        document = deepcopy(self.document)
        document["cases"][1]["id"] = document["cases"][0]["id"]
        _resign(document)
        with self.assertRaises(ref.FormatError) as caught:
            ref.verify_fixture(document)
        self.assertEqual(caught.exception.code, "fixture_schema")

    def test_wrong_operation_shape_is_rejected(self):
        document = deepcopy(self.document)
        document["cases"][0]["operation"] = "dot"
        _resign(document)
        with self.assertRaises(ref.FormatError) as caught:
            ref.verify_fixture(document)
        self.assertEqual(caught.exception.code, "fixture_schema")

    def test_fixture_case_count_bound_is_enforced(self):
        document = deepcopy(self.document)
        document["cases"] = document["cases"] * 5
        _resign(document)
        with self.assertRaises(ref.FormatError) as caught:
            ref.verify_fixture(document)
        self.assertEqual(caught.exception.code, "fixture_schema")


class RtlBundleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = RTL_PATH.read_bytes()
        cls.document = _load(RTL_PATH)
        cls.generated = ref.generate_rtl_bundle()
        cls.report = ref.verify_rtl_bundle(cls.document)

    def test_bundle_is_canonical_and_byte_identical_to_generation(self):
        self.assertEqual(self.raw, _pretty(self.document))
        self.assertEqual(self.raw, _pretty(self.generated))
        self.assertEqual(len(self.raw), 28187)
        self.assertEqual(self.document, self.generated)

    def test_bundle_digest_is_frozen_literal(self):
        self.assertEqual(self.document["payload_sha256"], RTL_SHA256)
        self.assertEqual(ref._digest(self.document), RTL_SHA256)
        self.assertEqual(set(self.document), ref._RTL_TOP_KEYS)
        self.assertEqual(self.document["fixture_id"], "iq-packed-rtl-integer-v1")

    def test_bundle_has_documented_case_shape(self):
        cases = self.document["cases"]
        self.assertEqual(len(cases), 7)
        self.assertEqual(len({case["id"] for case in cases}), 7)
        self.assertEqual([case["weight_format"] for case in cases].count("IQ2_XXS"), 3)
        self.assertEqual([case["weight_format"] for case in cases].count("IQ2_S"), 2)
        self.assertEqual([case["weight_format"] for case in cases].count("IQ1_M"), 2)
        for case in cases:
            self.assertEqual(set(case), RTL_CASE_KEYS)
            name = case["weight_format"]
            self.assertEqual(case["activation_format"], "Q8_K")
            self.assertEqual(case["integer_subdot_fields"], ref._EXPECTED_FORMATS[name]["integer_subdot_fields"])
            blocks = case["elements"] // 256
            self.assertEqual(len(case["expected_integer_subdots"]), blocks)
            for row in case["expected_integer_subdots"]:
                self.assertEqual(len(row), 8)
                for group in row:
                    self.assertEqual(len(group), len(case["integer_subdot_fields"]))
            self.assertRegex(case["expected_fp32_bits"], r"\A0x[0-9a-f]{8}\Z")

    def test_rtl_verify_report_is_exact(self):
        self.assertEqual(set(self.report), RTL_VERIFY_KEYS)
        self.assertIs(self.report["verified"], True)
        self.assertEqual(self.report["mismatches"], [])
        self.assertEqual(self.report["case_count"], 7)
        self.assertEqual(self.report["fixture_id"], "iq-packed-rtl-integer-v1")
        self.assertEqual(self.report["payload_sha256"], RTL_SHA256)
        self.assertIs(self.report["device_ready"], False)
        self.assertIs(self.report["inference_ready"], False)

    def test_rtl_integer_partials_are_not_scale_premultiplied(self):
        # IQ1_M group columns are raw integer sums; IQ2 columns are the two
        # 4-bit-scale integer sums.  All columns must stay integral and bounded.
        for case in self.document["cases"]:
            bounds = ref._integer_subdot_bounds(case["weight_format"])
            for row in case["expected_integer_subdots"]:
                for group in row:
                    for term in group:
                        self.assertIs(type(term), int)
                        self.assertGreaterEqual(term, bounds[0])
                        self.assertLessEqual(term, bounds[1])

    def test_rtl_tamper_is_detected(self):
        document = deepcopy(self.document)
        document["cases"][0]["expected_integer_subdots"][0][0][0] += 1
        _resign(document)
        report = ref.verify_rtl_bundle(document)
        self.assertIs(report["verified"], False)
        self.assertEqual(report["mismatches"][0]["field"], "expected_integer_subdots")

    def test_rtl_digest_tamper_is_an_input_error(self):
        document = deepcopy(self.document)
        document["cases"][0]["expected_fp32_bits"] = "0x00000000"
        with self.assertRaises(ref.FormatError) as caught:
            ref.verify_rtl_bundle(document)
        self.assertEqual(caught.exception.code, "fixture_digest")


class GenerationIndependenceTests(unittest.TestCase):
    def test_generation_never_calls_the_production_path(self):
        def explode(*_args, **_kwargs):
            raise AssertionError("generation must not call the production DUT")

        with patch.object(dut, "decode", explode), patch.object(dut, "dot", explode):
            fixture = ref.generate_fixture()
            bundle = ref.generate_rtl_bundle()
        self.assertEqual(fixture["payload_sha256"], FIXTURE_SHA256)
        self.assertEqual(bundle["payload_sha256"], RTL_SHA256)

    def test_generation_is_deterministic_across_calls(self):
        self.assertEqual(ref.generate_fixture(), ref.generate_fixture())
        self.assertEqual(ref.generate_rtl_bundle(), ref.generate_rtl_bundle())

    def test_tables_module_is_a_data_only_copy(self):
        self.assertIs(ref._tables, tables)
        self.assertIs(dut._tables, tables)
        for name, digest in ref._EXPECTED_TABLE_SHA256.items():
            self.assertEqual(tables.IQ_TABLE_SHA256[name], digest)
        self.assertEqual(len(tables.IQ2XXS_GRID), 256)
        self.assertEqual(len(tables.IQ2S_GRID), 1024)
        self.assertEqual(len(tables.IQ1S_GRID), 2048)
        self.assertEqual(tables.IQ1S_GRID[0], 0xFFFFFFFFFFFFFFFF)
        self.assertEqual(tables.IQ1S_GRID[1], 0xFFFFFFFFFFFFFF01)


if __name__ == "__main__":
    unittest.main()
