# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 lvyufeng and the openTPU contributors

"""Stdlib CLI contracts for the IQ2_XXS/IQ2_S/IQ1_M ``iq-*`` subcommands."""

from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from tools.packed_oracle import __main__ as cli
from tools.packed_oracle import iq_vectors as ref


ROOT = Path(__file__).resolve().parents[3]
DATA = Path(__file__).resolve().parents[1] / "data"
FIXTURE = DATA / "iq_synthetic.json"
RTL = DATA / "iq_rtl_vectors.json"

ENVELOPE_KEYS = {"schema_version", "scope", "result", "device_ready", "inference_ready"}
ERROR_KEYS = {"schema_version", "scope", "error", "device_ready", "inference_ready"}
REPORT_KEYS = {
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
RTL_REPORT_KEYS = {
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


def _resign(document):
    document["payload_sha256"] = ref._digest(document)


def _write(document, path):
    path.write_text(
        json.dumps(document, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False) + "\n",
        encoding="ascii",
    )


class IqCliTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = ref.generate_fixture()
        cls.cases = {case["id"]: case for case in cls.fixture["cases"]}
        cls.decode_case = next(case for case in cls.fixture["cases"] if case["operation"] == "decode")
        cls.dot_case = next(case for case in cls.fixture["cases"] if case["operation"] == "dot")

    def invoke(self, arguments):
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = cli.main(arguments)
        return code, stdout.getvalue(), stderr.getvalue()

    def assert_error(self, arguments, expected_code):
        code, stdout, stderr = self.invoke(arguments)
        self.assertEqual(code, 2)
        self.assertEqual(stdout, "")
        document = json.loads(stderr)
        self.assertEqual(set(document), ERROR_KEYS)
        self.assertEqual(document["schema_version"], 1)
        self.assertEqual(document["scope"], "synthetic-packed-reference")
        self.assertEqual(document["error"]["code"], expected_code)
        self.assertIsInstance(document["error"]["message"], str)
        self.assertIs(document["device_ready"], False)
        self.assertIs(document["inference_ready"], False)

    # -- result envelopes ---------------------------------------------------

    def test_iq_decode_envelope_matches_fixture_case(self):
        case = self.decode_case
        code, stdout, stderr = self.invoke(
            [
                "iq-decode",
                "--format",
                case["format"],
                "--elements",
                str(case["elements"]),
                "--hex",
                case["payload_hex"],
            ]
        )
        self.assertEqual((code, stderr), (0, ""))
        document = json.loads(stdout)
        self.assertEqual(set(document), ENVELOPE_KEYS)
        self.assertEqual(document["scope"], "synthetic-packed-reference")
        self.assertIs(document["device_ready"], False)
        self.assertIs(document["inference_ready"], False)
        result = document["result"]
        self.assertEqual(
            set(result),
            {"format", "elements", "blocks", "fp32_bits"},
        )
        self.assertEqual(result["format"], case["format"])
        self.assertEqual(result["elements"], case["elements"])
        self.assertEqual(result["blocks"], case["elements"] // 256)
        self.assertEqual(result["fp32_bits"], case["expected_fp32_bits"])

    def test_iq_dot_envelope_matches_fixture_case(self):
        case = self.dot_case
        code, stdout, stderr = self.invoke(
            [
                "iq-dot",
                "--weight-format",
                case["weight_format"],
                "--activation-format",
                case["activation_format"],
                "--elements",
                str(case["elements"]),
                "--weight-hex",
                case["weight_hex"],
                "--activation-hex",
                case["activation_hex"],
            ]
        )
        self.assertEqual((code, stderr), (0, ""))
        result = json.loads(stdout)["result"]
        self.assertEqual(
            set(result),
            {
                "weight_format",
                "activation_format",
                "elements",
                "weight_blocks",
                "activation_blocks",
                "integer_subdots",
                "fp32_bits",
                "arithmetic_id",
            },
        )
        self.assertEqual(result["weight_format"], case["weight_format"])
        self.assertEqual(result["activation_format"], "Q8_K")
        self.assertEqual(result["fp32_bits"], case["expected_fp32_bits"])
        self.assertEqual(result["integer_subdots"], case["expected_integer_subdots"])
        self.assertEqual(result["arithmetic_id"], ref.ARITHMETIC_ID)

    def test_iq_dot_activation_format_defaults_to_q8_k(self):
        case = self.dot_case
        arguments = [
            "iq-dot",
            "--weight-format",
            case["weight_format"],
            "--elements",
            str(case["elements"]),
            "--weight-hex",
            case["weight_hex"],
            "--activation-hex",
            case["activation_hex"],
        ]
        code, stdout, stderr = self.invoke(arguments)
        self.assertEqual((code, stderr), (0, ""))
        self.assertEqual(json.loads(stdout)["result"]["activation_format"], "Q8_K")

    # -- fixture / bundle lifecycle ----------------------------------------

    def test_iq_verify_default_is_the_shipped_fixture(self):
        code, stdout, stderr = self.invoke(["iq-verify"])
        self.assertEqual((code, stderr), (0, ""))
        report = json.loads(stdout)
        self.assertEqual(set(report), REPORT_KEYS)
        self.assertIs(report["verified"], True)
        self.assertEqual(report["mismatches"], [])
        self.assertEqual(report["case_count"], 16)
        self.assertEqual(report["decode_cases"], 6)
        self.assertEqual(report["dot_cases"], 10)
        self.assertEqual(report["payload_sha256"], self.fixture["payload_sha256"])

    def test_iq_verify_explicit_path(self):
        code, stdout, stderr = self.invoke(["iq-verify", str(FIXTURE)])
        self.assertEqual((code, stderr), (0, ""))
        self.assertIs(json.loads(stdout)["verified"], True)

    def test_iq_generate_stdout_matches_shipped_file_byte_for_byte(self):
        code, stdout, stderr = self.invoke(["iq-generate", "--stdout"])
        self.assertEqual((code, stderr), (0, ""))
        self.assertEqual(stdout.encode("ascii"), FIXTURE.read_bytes())

    def test_iq_rtl_generate_stdout_matches_shipped_file_byte_for_byte(self):
        code, stdout, stderr = self.invoke(["iq-rtl-generate", "--stdout"])
        self.assertEqual((code, stderr), (0, ""))
        self.assertEqual(stdout.encode("ascii"), RTL.read_bytes())

    def test_iq_rtl_verify_default_and_explicit(self):
        for arguments in (["iq-rtl-verify"], ["iq-rtl-verify", str(RTL)]):
            code, stdout, stderr = self.invoke(arguments)
            self.assertEqual((code, stderr), (0, ""))
            report = json.loads(stdout)
            self.assertEqual(set(report), RTL_REPORT_KEYS)
            self.assertIs(report["verified"], True)
            self.assertEqual(report["case_count"], 7)
            self.assertEqual(report["mismatches"], [])

    def test_validly_resigned_mismatch_exits_one(self):
        document = deepcopy(self.fixture)
        document["cases"][0]["expected_fp32_bits"][0] = "0xdeadbeef"
        _resign(document)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "tampered.json"
            _write(document, path)
            code, stdout, stderr = self.invoke(["iq-verify", str(path)])
        self.assertEqual((code, stderr), (1, ""))
        report = json.loads(stdout)
        self.assertFalse(report["verified"])
        self.assertEqual(report["mismatches"][0]["field"], "fp32_bits")

    def test_unsound_digest_is_an_input_error(self):
        document = deepcopy(self.fixture)
        document["cases"][0]["expected_fp32_bits"][0] = "0xdeadbeef"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "unsound.json"
            _write(document, path)
            self.assert_error(["iq-verify", str(path)], "fixture_digest")

    def test_wrong_fixture_id_is_a_schema_error(self):
        document = deepcopy(self.fixture)
        document["fixture_id"] = "not-iq"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "wrong-id.json"
            _write(document, path)
            self.assert_error(["iq-verify", str(path)], "fixture_schema")

    def test_malformed_json_is_an_input_error(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "broken.json"
            path.write_bytes(b'{"cases": [}')
            self.assert_error(["iq-verify", str(path)], "invalid_json")
            path.write_bytes(b"\xff\xfe")
            self.assert_error(["iq-rtl-verify", str(path)], "invalid_json")

    # -- argument and payload validation ------------------------------------

    def test_missing_extra_unknown_and_abbreviated_arguments(self):
        self.assert_error([], "invalid_arguments")
        self.assert_error(["iq-decode", "--format", "IQ2_XXS"], "invalid_arguments")
        self.assert_error(
            ["iq-decode", "--format", "IQ2_XXS", "--elements", "256", "--hex", "00", "--bogus"],
            "invalid_arguments",
        )
        self.assert_error(
            ["iq-generate"],
            "invalid_arguments",
        )

    def test_invalid_elements_and_formats_are_structured(self):
        block = "00" * 66
        self.assert_error(["iq-decode", "--format", "IQ2_XXS", "--elements", "0", "--hex", block], "invalid_elements")
        self.assert_error(
            ["iq-decode", "--format", "IQ2_XXS", "--elements", "0256", "--hex", block],
            "invalid_elements",
        )
        self.assert_error(["iq-decode", "--format", "Q2_0", "--elements", "256", "--hex", block], "unsupported_format")
        self.assert_error(
            ["iq-decode", "--format", "IQ2_S", "--elements", "256", "--hex", block],
            "invalid_length",
        )
        self.assert_error(
            ["iq-decode", "--format", "IQ2_XXS", "--elements", "256", "--hex", block + "00"],
            "invalid_length",
        )
        self.assert_error(
            ["iq-decode", "--format", "IQ2_XXS", "--elements", "256", "--hex", "00 3c" + "00" * 65],
            "invalid_hex",
        )

    def test_activation_format_and_length_are_validated(self):
        weight = self.dot_case["weight_hex"]
        activation = self.dot_case["activation_hex"]
        elements = str(self.dot_case["elements"])
        weight_format = self.dot_case["weight_format"]
        self.assert_error(
            [
                "iq-dot",
                "--weight-format",
                weight_format,
                "--activation-format",
                "Q8_0",
                "--elements",
                elements,
                "--weight-hex",
                weight,
                "--activation-hex",
                activation,
            ],
            "unsupported_activation_format",
        )
        self.assert_error(
            [
                "iq-dot",
                "--weight-format",
                weight_format,
                "--elements",
                elements,
                "--weight-hex",
                weight,
                "--activation-hex",
                activation[:-2],
            ],
            "invalid_length",
        )

    def test_nonfinite_scales_are_structured_errors(self):
        infinite_half = "007c" + "00" * 64
        self.assert_error(
            ["iq-decode", "--format", "IQ2_XXS", "--elements", "256", "--hex", infinite_half],
            "nonfinite_scale",
        )

    # -- independence at the CLI boundary -----------------------------------

    def test_invalid_input_never_calls_the_production_path(self):
        def explode(*_args, **_kwargs):
            raise AssertionError("invalid CLI input must not reach the DUT")

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "broken.json"
            path.write_bytes(b"not json")
            with patch("tools.packed_oracle.iq_formats.decode", explode), patch(
                "tools.packed_oracle.iq_formats.dot", explode
            ):
                self.assert_error(["iq-verify", str(path)], "invalid_json")
                self.assert_error(
                    ["iq-decode", "--format", "IQ2_XXS", "--elements", "0", "--hex", "00"],
                    "invalid_elements",
                )

    def test_subprocess_module_entry_point_is_source_only(self):
        command = [sys.executable, "-B", "-S", "-m", "tools.packed_oracle", "iq-rtl-verify"]
        process = subprocess.run(command, cwd=ROOT, check=False, capture_output=True, text=True, timeout=30)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(process.stderr, "")
        report = json.loads(process.stdout)
        self.assertIs(report["verified"], True)
        self.assertEqual(report["case_count"], 7)

    def test_no_real_model_or_device_subcommands(self):
        parser = cli._parser()
        actions = [action for action in parser._actions if getattr(action, "choices", None)]
        names = list(actions[0].choices)
        self.assertEqual(
            names,
            ["decode", "dot", "verify", "generate", "iq-decode", "iq-dot", "iq-verify", "iq-generate", "iq-rtl-verify", "iq-rtl-generate"],
        )
        for name in names:
            for word in ("model", "device", "flash", "program", "run", "bench", "serve"):
                self.assertNotIn(word, name)


if __name__ == "__main__":
    unittest.main()
