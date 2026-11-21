# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 lvyufeng and the openTPU contributors

"""Stdlib CLI for inert packed-byte reference inputs, not device operations."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import re
import sys

from .formats import FormatError, decode, dot, parse_hex


MAX_FIXTURE_BYTES = 8 * 1024 * 1024
MAX_JSON_DEPTH = 64
MAX_JSON_NUMBER_CHARS = 64
DEFAULT_FIXTURE = Path(__file__).with_name("data") / "q2_q8_synthetic.json"
_INTEGER = re.compile(r"[1-9][0-9]*", flags=re.ASCII)


class _Parser(argparse.ArgumentParser):
    def error(self, message: str) -> None:
        # Do not print argparse's prose or echo untrusted command-line values.
        raise FormatError("invalid_arguments", "invalid or missing command-line arguments")


def _elements(text: str) -> int:
    if len(text) > 5 or _INTEGER.fullmatch(text) is None:
        raise FormatError("invalid_elements", "elements must be a positive decimal integer")
    return int(text)


def _json_pairs(pairs: list[tuple[str, object]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise FormatError("invalid_json", "fixture JSON has a duplicate object key")
        result[key] = value
    return result


def _json_nonfinite(text: str) -> None:
    raise FormatError("invalid_json", "fixture JSON contains a non-finite number")


def _json_integer(text: str) -> int:
    if len(text) > MAX_JSON_NUMBER_CHARS:
        raise FormatError("invalid_json", "fixture JSON numeric literal exceeds 64 characters")
    return int(text)


def _json_float(text: str) -> float:
    if len(text) > MAX_JSON_NUMBER_CHARS:
        raise FormatError("invalid_json", "fixture JSON numeric literal exceeds 64 characters")
    value = float(text)
    if not math.isfinite(value):
        _json_nonfinite(text)
    return value


def _check_json_depth(text: str) -> None:
    # Check lexical depth before json.loads: its recursion behavior differs
    # between Python releases. Brackets inside strings do not count.
    depth = 0
    in_string = False
    escaped = False
    for character in text:
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
        elif character == '"':
            in_string = True
        elif character in "[{":
            depth += 1
            if depth > MAX_JSON_DEPTH:
                raise FormatError("invalid_json", "fixture JSON nesting exceeds 64 levels")
        elif character in "]}":
            depth -= 1


def _load_fixture(path: Path) -> object:
    try:
        with path.open("rb") as handle:
            payload = handle.read(MAX_FIXTURE_BYTES + 1)
        if len(payload) > MAX_FIXTURE_BYTES:
            raise FormatError("invalid_fixture", "fixture exceeds the 8 MiB input limit")
        text = payload.decode("utf-8")
        _check_json_depth(text)
        return json.loads(
            text,
            object_pairs_hook=_json_pairs,
            parse_constant=_json_nonfinite,
            parse_int=_json_integer,
            parse_float=_json_float,
        )
    except FormatError:
        raise
    except OSError:
        raise FormatError("fixture_io", "fixture file cannot be read") from None
    except (UnicodeError, ValueError, RecursionError):
        raise FormatError("invalid_json", "fixture must contain valid bounded UTF-8 JSON") from None


def _emit(document: object, stream: object = None) -> None:
    if stream is None:
        stream = sys.stdout
    text = json.dumps(
        document, sort_keys=True, indent=2, ensure_ascii=True, allow_nan=False
    )
    print(text, file=stream)


def _result_envelope(result: dict) -> dict:
    return {
        "schema_version": 1,
        "scope": "synthetic-packed-reference",
        "result": result,
        "device_ready": False,
        "inference_ready": False,
    }


def _parser() -> _Parser:
    parser = _Parser(
        prog="python -m tools.packed_oracle",
        description="Source-only Q2_0/Q8_0 supplied-packed scalar references; no devices.",
        allow_abbrev=False,
    )
    commands = parser.add_subparsers(dest="command", required=True)
    decode_parser = commands.add_parser("decode", allow_abbrev=False)
    decode_parser.add_argument("--format", required=True)
    decode_parser.add_argument("--elements", required=True)
    decode_parser.add_argument("--hex", required=True)
    dot_parser = commands.add_parser("dot", allow_abbrev=False)
    dot_parser.add_argument("--weight-format", required=True)
    dot_parser.add_argument("--activation-format", default="Q8_0")
    dot_parser.add_argument("--elements", required=True)
    dot_parser.add_argument("--weight-hex", required=True)
    dot_parser.add_argument("--activation-hex", required=True)
    verify_parser = commands.add_parser("verify", allow_abbrev=False)
    verify_parser.add_argument("fixture", nargs="?", type=Path, default=DEFAULT_FIXTURE)
    generate_parser = commands.add_parser("generate", allow_abbrev=False)
    generate_parser.add_argument("--stdout", action="store_true", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Return 0 for success, 1 for valid fixture mismatches, 2 for input errors."""
    try:
        options = _parser().parse_args(argv)
        if options.command in ("decode", "dot"):
            # FormatError subclasses ValueError, which argparse's type converter
            # would otherwise swallow and relabel as invalid_arguments.
            options.elements = _elements(options.elements)
        if options.command == "decode":
            result = decode(options.format, parse_hex(options.hex), options.elements)
            _emit(_result_envelope(result))
        elif options.command == "dot":
            result = dot(
                options.weight_format,
                parse_hex(options.weight_hex),
                parse_hex(options.activation_hex),
                options.elements,
                activation_format=options.activation_format,
            )
            _emit(_result_envelope(result))
        elif options.command == "verify":
            # Lazy import: ordinary decode/dot need no fixture generation code.
            from .vectors import verify_fixture

            report = verify_fixture(_load_fixture(options.fixture))
            _emit(report)
            return 0 if report["verified"] else 1
        else:
            from .vectors import generate_fixture

            _emit(generate_fixture())
        return 0
    except FormatError as error:
        _emit(
            {
                "schema_version": 1,
                "scope": "synthetic-packed-reference",
                "error": {"code": error.code, "message": error.message},
                "device_ready": False,
                "inference_ready": False,
            },
            sys.stderr,
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
