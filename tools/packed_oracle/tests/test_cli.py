# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 lvyufeng and the openTPU contributors

"""Stdlib CLI contracts: structured results, fail-closed JSON, and exit codes."""

from contextlib import redirect_stderr, redirect_stdout
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from tools.packed_oracle import __main__ as cli
from tools.packed_oracle import formats
from tools.packed_oracle import vectors


ROOT = Path(__file__).resolve().parents[3]
DATA = Path(__file__).resolve().parents[1] / "data" / "q2_q8_synthetic.json"
ENVELOPE_KEYS = {"schema_version", "scope", "result", "device_ready", "inference_ready"}
ERROR_KEYS = {"schema_version", "scope", "error", "device_ready", "inference_ready"}
REPORT_KEYS = {
    "schema_version", "fixture_id", "arithmetic_id", "payload_sha256", "case_count",
    "decode_cases", "dot_cases", "synthetic", "verified", "mismatches", "device_ready", "inference_ready",
}


def resign(document):
    payload = {key: value for key, value in document.items() if key != "payload_sha256"}
    document["payload_sha256"] = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")).hexdigest()


class CliTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = vectors.generate_fixture()
        cls.cases = {case["id"]: case for case in cls.fixture["cases"]}

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
        self.assertEqual(set(document["error"]), {"code", "message"})
        self.assertEqual(document["error"]["code"], expected_code)
        self.assertIsInstance(document["error"]["message"], str)
        self.assertTrue(document["error"]["message"])
        self.assertIs(document["device_ready"], False)
        self.assertIs(document["inference_ready"], False)
        self.assertNotIn("Traceback", stderr)
        return document

    def assert_result(self, arguments):
        code, stdout, stderr = self.invoke(arguments)
        self.assertEqual(code, 0)
        self.assertEqual(stderr, "")
        document = json.loads(stdout)
        self.assertEqual(set(document), ENVELOPE_KEYS)
        self.assertEqual(document["schema_version"], 1)
        self.assertEqual(document["scope"], "synthetic-packed-reference")
        self.assertIs(document["device_ready"], False)
        self.assertIs(document["inference_ready"], False)
        return document["result"]

    def test_generate_stdout_matches_shipped_file_byte_for_byte(self):
        code, stdout, stderr = self.invoke(["generate", "--stdout"])
        self.assertEqual(code, 0)
        self.assertEqual(stderr, "")
        self.assertEqual(stdout, DATA.read_text(encoding="utf-8"))
        self.assertEqual(json.loads(stdout), self.fixture)

    def test_default_verify_is_exact_frozen_report(self):
        code, stdout, stderr = self.invoke(["verify"])
        self.assertEqual(code, 0)
        self.assertEqual(stderr, "")
        report = json.loads(stdout)
        self.assertEqual(set(report), REPORT_KEYS)
        self.assertTrue(report["verified"])
        self.assertEqual(report["mismatches"], [])
        self.assertEqual(report["case_count"], 146)
        self.assertEqual(report["decode_cases"], 9)
        self.assertEqual(report["dot_cases"], 137)
        self.assertIs(report["device_ready"], False)
        self.assertIs(report["inference_ready"], False)

    def test_explicit_path_verify_and_no_write(self):
        before = DATA.read_bytes()
        code, stdout, stderr = self.invoke(["verify", str(DATA)])
        self.assertEqual(code, 0)
        self.assertEqual(stderr, "")
        self.assertTrue(json.loads(stdout)["verified"])
        self.assertEqual(DATA.read_bytes(), before)

    def test_decode_envelope_and_hard_raw_lanes(self):
        payload = (0x3C00).to_bytes(2, "little") + bytes([0xE4] * 16)
        result = self.assert_result(["decode", "--format", "Q2_0", "--elements", "64", "--hex", payload.hex()])
        self.assertEqual(set(result), {"format", "elements", "blocks", "fp32_bits"})
        self.assertEqual(result["fp32_bits"], ["0xbf800000", "0x00000000", "0x3f800000", "0x40000000"] * 16)
        self.assertEqual((result["format"], result["elements"], result["blocks"]), ("Q2_0", 64, 1))

    def test_hard_q2_dot_minus70_cli(self):
        case = self.cases["q2-hand-minus-70"]
        result = self.assert_result(["dot", "--weight-format", "Q2_0", "--elements", "64", "--weight-hex", case["weight_hex"], "--activation-hex", case["activation_hex"]])
        self.assertEqual(result["fp32_bits"], "0xc28c0000")
        self.assertEqual(result["integer_subdots"], [[80, -80]])
        self.assertEqual((result["weight_blocks"], result["activation_blocks"]), (1, 2))

    def test_hard_q8_dot378_explicit_activation_cli(self):
        case = self.cases["q8-hand-378"]
        result = self.assert_result(["dot", "--weight-format", "Q8_0", "--activation-format", "Q8_0", "--elements", "32", "--weight-hex", case["weight_hex"], "--activation-hex", case["activation_hex"]])
        self.assertEqual(result["fp32_bits"], "0x43bd0000")
        self.assertEqual(result["integer_subdots"], [[3024]])
        self.assertEqual((result["weight_blocks"], result["activation_blocks"]), (1, 1))

    def test_subprocess_module_import_and_cli_are_source_only(self):
        command = [sys.executable, "-B", "-S", "-m", "tools.packed_oracle", "verify"]
        process = subprocess.run(command, cwd=ROOT, check=False, capture_output=True, text=True, timeout=20)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(process.stderr, "")
        self.assertTrue(json.loads(process.stdout)["verified"])
        script = "import sys; import tools.packed_oracle; forbidden={'numpy','tensorflow','torch','opentpu','tpu','tpu_driver'}; assert not (forbidden & set(sys.modules)); print('stdlib-only')"
        process = subprocess.run([sys.executable, "-B", "-S", "-c", script], cwd=ROOT, check=False, capture_output=True, text=True, timeout=10)
        self.assertEqual(process.returncode, 0, process.stderr)
        self.assertEqual(process.stdout, "stdlib-only\n")
        self.assertEqual(process.stderr, "")

    def test_missing_extra_unknown_and_abbreviated_arguments(self):
        values = ([], ["not-a-command"], ["generate"], ["generate", "--std"], ["generate", "--stdout", "extra"], ["verify", "--unexpected"], ["verify", "a", "b"], ["decode"], ["decode", "--format", "Q2_0", "--hex", "00"], ["decode", "--format", "Q2_0", "--ele", "64", "--hex", "00"], ["dot"], ["dot", "--weight-format", "Q8_0", "--elements", "32", "--weight-hex", "00"], ["decode", "--format", "Q2_0", "--elements", "64", "--hex", "00", "--unknown"])
        for arguments in values:
            with self.subTest(arguments=arguments):
                self.assert_error(arguments, "invalid_arguments")

    def test_noncanonical_or_invalid_decimal_elements_are_stable_error(self):
        for text in ("true", "false", "0", "-64", "+64", "01", "064", "64.0", "6.4e1", " 64", "64 ", "６４", "65537", "100000", "1", "63"):
            arguments = ["decode", "--format", "Q2_0", "--elements", text, "--hex", self.cases["q2-hand-minus-70"]["weight_hex"]]
            with self.subTest(text=text):
                self.assert_error(arguments, "invalid_elements")

    def test_invalid_hex_length_format_and_activation_errors(self):
        weight = self.cases["q2-hand-minus-70"]["weight_hex"]
        activation = self.cases["q2-hand-minus-70"]["activation_hex"]
        self.assert_error(["decode", "--format", "Q2_K", "--elements", "64", "--hex", weight], "unsupported_format")
        self.assert_error(["decode", "--format", "Q2_0", "--elements", "64", "--hex", "0x" + weight], "invalid_hex")
        self.assert_error(["decode", "--format", "Q2_0", "--elements", "64", "--hex", weight[:-2]], "invalid_length")
        self.assert_error(["decode", "--format", "Q2_0", "--elements", "64", "--hex", weight + "00"], "invalid_length")
        self.assert_error(["dot", "--weight-format", "Q2_0", "--activation-format", "Q8_K", "--elements", "64", "--weight-hex", weight, "--activation-hex", activation], "unsupported_activation_format")
        self.assert_error(["dot", "--weight-format", "Q2_0", "--elements", "64", "--weight-hex", weight, "--activation-hex", activation[:-2]], "invalid_length")
        self.assert_error(["dot", "--weight-format", "Q2_0", "--elements", "64", "--weight-hex", weight, "--activation-hex", activation + "00"], "invalid_length")

    def test_nonfinite_scale_error_is_structured(self):
        payload = b"\x00\x7c" + bytes(32)
        self.assert_error(["decode", "--format", "Q8_0", "--elements", "32", "--hex", payload.hex()], "nonfinite_scale")

    def test_validly_resigned_false_expectation_exit_one_not_input_error(self):
        document = deepcopy(self.fixture)
        target = next(case for case in document["cases"] if case["id"] == "q8-hand-378")
        target["expected_fp32_bits"] = "0x00000000"
        resign(document)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "mismatch.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            code, stdout, stderr = self.invoke(["verify", str(path)])
        self.assertEqual(code, 1)
        self.assertEqual(stderr, "")
        report = json.loads(stdout)
        self.assertEqual(set(report), REPORT_KEYS)
        self.assertIs(report["verified"], False)
        self.assertEqual(report["mismatches"], [{"case_id": "q8-hand-378", "field": "fp32_bits", "expected": "0x00000000", "actual": "0x43bd0000"}])
        self.assertIs(report["device_ready"], False)
        self.assertIs(report["inference_ready"], False)

    def test_bad_digest_exit_two_not_false_verified_report(self):
        document = deepcopy(self.fixture)
        document["payload_sha256"] = "0" * 64
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bad-digest.json"
            path.write_text(json.dumps(document), encoding="utf-8")
            self.assert_error(["verify", str(path)], "fixture_digest")

    def test_json_invalid_utf8_duplicate_nonfinite_trailing_deep_hugeint(self):
        values = (b"\xff", b"", b"{", b"{} trailing", b'{"schema_version":1,"schema_version":1}', b'{"cases":NaN}', b'{"cases":Infinity}', b'{"cases":-Infinity}', b"[" * 2000 + b"0" + b"]" * 2000, b'{"elements":' + b"9" * 5000 + b"}")
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "invalid.json"
            for index, value in enumerate(values):
                path.write_bytes(value)
                with self.subTest(index=index):
                    self.assert_error(["verify", str(path)], "invalid_json")

    def test_json_valid_but_wrong_schema_is_input_error(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "wrong-schema.json"
            for document in (None, [], True, 1, "fixture", {}, {"device_ready": True}):
                path.write_text(json.dumps(document), encoding="utf-8")
                with self.subTest(document=document):
                    self.assert_error(["verify", str(path)], "fixture_schema")

    def test_json_file_size_bound_and_file_io_errors(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "oversized.json"
            path.write_bytes(b" " * (8 * 1024 * 1024 + 1))
            self.assert_error(["verify", str(path)], "invalid_fixture")
            self.assert_error(["verify", str(Path(temporary) / "missing.json")], "fixture_io")
            self.assert_error(["verify", temporary], "fixture_io")

    def test_json_depth_64_boundary_and_brackets_inside_strings(self):
        controls = [
            b"[" * 64 + b"0" + b"]" * 64,
            json.dumps({"quoted": "[" * 200 + "]" * 200 + '{\\\"escaped\\\"}' + '\\\\"'}).encode("utf-8"),
        ]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "depth-control.json"
            for index, payload in enumerate(controls):
                path.write_bytes(payload)
                with self.subTest(index=index):
                    self.assertEqual(cli._load_fixture(path), json.loads(payload))
            path.write_bytes(b"[" * 65 + b"0" + b"]" * 65)
            self.assert_error(["verify", str(path)], "invalid_json")

    def test_json_numeric_token_limit_and_overflow_are_interpreter_independent(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "numeric-boundary.json"
            accepted = ("9" * 64, "-" + "9" * 63, "0." + "1" * 62, "1e2")
            rejected = ("9" * 65, "-" + "9" * 64, "0." + "1" * 63, "1e999", "-1e999")
            for text in accepted:
                path.write_text('{"value":' + text + '}', encoding="utf-8")
                with self.subTest(accepted=text):
                    self.assertEqual(cli._load_fixture(path), json.loads('{"value":' + text + '}'))
            for text in rejected:
                path.write_text('{"value":' + text + '}', encoding="utf-8")
                with self.subTest(rejected=text):
                    self.assert_error(["verify", str(path)], "invalid_json")

    def test_invalid_json_never_calls_dut(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "invalid.json"
            path.write_bytes(b'{"cases":NaN}')
            with patch.object(formats, "decode", side_effect=AssertionError("DUT must not run")), patch.object(formats, "dot", side_effect=AssertionError("DUT must not run")):
                self.assert_error(["verify", str(path)], "invalid_json")

    def test_cli_deterministic_generation_is_independent_of_dut(self):
        with patch.object(formats, "decode", side_effect=AssertionError("DUT must not run")), patch.object(formats, "dot", side_effect=AssertionError("DUT must not run")):
            code, stdout, stderr = self.invoke(["generate", "--stdout"])
        self.assertEqual(code, 0)
        self.assertEqual(stderr, "")
        self.assertEqual(stdout, DATA.read_text(encoding="utf-8"))

    def test_no_real_model_or_device_subcommands(self):
        for command in ("quantize", "model", "gguf", "load", "dma", "device", "run", "infer", "rtl", "build"):
            with self.subTest(command=command):
                self.assert_error([command], "invalid_arguments")


if __name__ == "__main__":
    unittest.main()
