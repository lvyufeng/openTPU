# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 lvyufeng and the openTPU contributors

"""Independent, synthetic supplied-packed goldens and their strict verifier.

Generation uses only exact Fraction/integer binary32 round-to-nearest-even math.
It never calls the scalar implementation under test, a quantizer, or native code.
Verification checks the entire frozen schema before executing the scalar DUT.
A matching digest is integrity metadata, not a certificate of experimental truth.
"""

from __future__ import annotations

from copy import deepcopy
from fractions import Fraction
import hashlib
import json
import re


ARITHMETIC_ID = "strata-generic-f32-rne-separate-v1"
FIXTURE_ID = "q2-q8-supplied-packed-v1"
MAX_FIXTURE_BYTES = 8 * 1024 * 1024
MAX_CASES = 256
MAX_ELEMENTS = 65536

# Literals are intentional: expected metadata and provenance are not taken from DUT.
_EXPECTED_FORMATS = {
    "Q2_0": {
        "ggml_type": 42,
        "block_elements": 64,
        "block_bytes": 18,
        "activation_format": "Q8_0",
        "byte_order": "little",
    },
    "Q8_0": {
        "ggml_type": 8,
        "block_elements": 32,
        "block_bytes": 34,
        "activation_format": "Q8_0",
        "byte_order": "little",
    },
}
_SCOPE_EXCLUSIONS = [
    "real-model coverage",
    "native CPU or AVX parity",
    "RTL or FPGA parity",
    "DMA or device operations",
    "quantization or model loading",
    "performance or whole decode",
]
_EXPECTED_PROVENANCE = {
    "strata": {
        "repository": "https://github.com/lvyufeng/Strata",
        "revision": "3142b66b96e9c0e775e30f620b11b191a1427c2a",
        "anchors": [
            {"path": "engine/src/quant/dequant.cpp", "start_line": 25, "end_line": 43},
            {"path": "engine/src/quant/dequant.cpp", "start_line": 139, "end_line": 153},
            {"path": "engine/src/quant/dot_generic.cpp", "start_line": 22, "end_line": 63},
            {"path": "engine/src/quant/dot_generic.cpp", "start_line": 203, "end_line": 226},
            {
                "path": "engine/include/strata/quant/strata-quant-common.h",
                "start_line": 195,
                "end_line": 201,
            },
            {
                "path": "engine/include/strata/quant/strata-quant-common.h",
                "start_line": 266,
                "end_line": 272,
            },
        ],
    },
    "opentpu_base": {
        "repository": "https://github.com/lvyufeng/openTPU",
        "revision": "d84db55712aba0d40811d7762cb052f05d55b7e5",
    },
    "ggml_origin_revision": "3cf03257f219afbe7334045ff7c6a06ac68c627d",
    "generator": {"id": "fraction-integer-reference", "version": 1},
    "expected_derivation": "independent exact rational binary32 RNE; no DUT results",
    "scope_exclusions": _SCOPE_EXCLUSIONS,
}
_TOP_KEYS = {
    "schema_version", "fixture_id", "synthetic", "arithmetic_id", "formats",
    "provenance", "cases", "payload_sha256",
}
_DECODE_KEYS = {
    "id", "operation", "format", "elements", "payload_hex", "expected_fp32_bits",
}
_DOT_KEYS = {
    "id", "operation", "weight_format", "activation_format", "elements",
    "weight_hex", "activation_hex", "expected_fp32_bits", "expected_integer_subdots",
}
_ID_PATTERN = re.compile(r"[a-z0-9][a-z0-9_-]{0,79}\Z")
_HEX_PATTERN = re.compile(r"[0-9a-fA-F]+\Z")
_BITS_PATTERN = re.compile(r"0x[0-9a-f]{8}\Z")
_SHA_PATTERN = re.compile(r"[0-9a-f]{64}\Z")


def _fail(code: str, message: str):
    # Lazy import keeps independent fixture generation usable before the DUT exists.
    from .formats import FormatError
    raise FormatError(code, message)


def _rne_ratio(numerator: int, denominator: int) -> int:
    """Round a nonnegative exact rational to an integer; ties choose even."""
    quotient, remainder = divmod(numerator, denominator)
    twice_remainder = 2 * remainder
    if twice_remainder > denominator or (
        twice_remainder == denominator and quotient & 1
    ):
        quotient += 1
    return quotient


def _scaled_rne(numerator: int, denominator: int, shift: int) -> int:
    if shift >= 0:
        return _rne_ratio(numerator << shift, denominator)
    return _rne_ratio(numerator, denominator << -shift)


def _fraction_to_bits(value: Fraction, negative_zero: bool = False) -> int:
    """Exact rational -> IEEE binary32 bits, RNE, with no binary floating math."""
    sign = int(value < 0 or (value == 0 and negative_zero)) << 31
    if value == 0:
        return sign
    magnitude = abs(value)
    numerator, denominator = magnitude.numerator, magnitude.denominator
    exponent = numerator.bit_length() - denominator.bit_length()
    if exponent >= 0:
        if numerator < denominator << exponent:
            exponent -= 1
    elif numerator << -exponent < denominator:
        exponent -= 1

    if exponent < -126:
        # Includes underflow ties and rounding up into the minimum normal value.
        return sign | _scaled_rne(numerator, denominator, 149)
    significand = _scaled_rne(numerator, denominator, 23 - exponent)
    if significand == 1 << 24:
        significand >>= 1
        exponent += 1
    if exponent > 127:
        return sign | 0x7F800000
    return sign | ((exponent + 127) << 23) | (significand - (1 << 23))


def _power_two(exponent: int) -> Fraction:
    if exponent >= 0:
        return Fraction(1 << exponent)
    return Fraction(1, 1 << -exponent)


def _bits_to_fraction(bits: int) -> Fraction:
    exponent = (bits >> 23) & 255
    mantissa = bits & 0x7FFFFF
    if exponent == 255:
        raise ValueError("nonfinite binary32 reference operand")
    if exponent == 0:
        magnitude = Fraction(mantissa, 1 << 149)
    else:
        magnitude = Fraction((1 << 23) | mantissa) * _power_two(exponent - 150)
    return -magnitude if bits >> 31 else magnitude


def _half_to_bits(raw: int) -> int:
    exponent = (raw >> 10) & 31
    mantissa = raw & 1023
    if exponent == 31:
        raise ValueError("nonfinite binary16 reference scale")
    if exponent == 0:
        magnitude = Fraction(mantissa, 1 << 24)
    else:
        magnitude = Fraction(1024 + mantissa) * _power_two(exponent - 25)
    if raw & 0x8000:
        magnitude = -magnitude
    return _fraction_to_bits(magnitude, bool(raw & 0x8000))


def _multiply(left: int, right: int) -> int:
    value = _bits_to_fraction(left) * _bits_to_fraction(right)
    return _fraction_to_bits(value, bool((left ^ right) & 0x80000000))


def _add(left: int, right: int) -> int:
    value = _bits_to_fraction(left) + _bits_to_fraction(right)
    # Exact opposite-sign cancellation yields +0 under RNE. Only -0 + -0 is -0.
    negative_zero = value == 0 and left == 0x80000000 and right == 0x80000000
    return _fraction_to_bits(value, negative_zero)


def _hex_bits(bits: int) -> str:
    return f"0x{bits:08x}"


def _q2_block(scale_bits: int, codes: list[int]) -> bytes:
    if len(codes) != 64 or any(type(c) is not int or not 0 <= c <= 3 for c in codes):
        raise ValueError("reference Q2 block needs 64 supplied integer codes in 0..3")
    if type(scale_bits) is not int or not 0 <= scale_bits <= 65535:
        raise ValueError("reference scale must be supplied binary16 bits")
    _half_to_bits(scale_bits)
    packed = [sum(codes[4 * i + lane] << (2 * lane) for lane in range(4)) for i in range(16)]
    return scale_bits.to_bytes(2, "little") + bytes(packed)


def _q8_block(scale_bits: int, integers: list[int]) -> bytes:
    if len(integers) != 32 or any(type(q) is not int or not -128 <= q <= 127 for q in integers):
        raise ValueError("reference Q8 block needs 32 supplied signed int8 values")
    if type(scale_bits) is not int or not 0 <= scale_bits <= 65535:
        raise ValueError("reference scale must be supplied binary16 bits")
    _half_to_bits(scale_bits)
    return scale_bits.to_bytes(2, "little") + bytes(q & 255 for q in integers)


def _read_blocks(name: str, payload: bytes) -> list[tuple[int, list[int]]]:
    stride = 18 if name == "Q2_0" else 34
    result = []
    for start in range(0, len(payload), stride):
        raw = payload[start:start + stride]
        scale = _half_to_bits(int.from_bytes(raw[:2], "little"))
        if name == "Q2_0":
            # The reference is an independent list comprehension, not the DUT decoder.
            integers = [((byte // (4 ** lane)) % 4) - 1 for byte in raw[2:] for lane in range(4)]
        else:
            integers = [byte if byte < 128 else byte - 256 for byte in raw[2:]]
        result.append((scale, integers))
    return result


def _reference_decode(name: str, payload: bytes) -> list[str]:
    return [
        _hex_bits(_multiply(_fraction_to_bits(Fraction(integer)), scale))
        for scale, integers in _read_blocks(name, payload)
        for integer in integers
    ]


def _reference_dot(name: str, weight: bytes, activation: bytes) -> tuple[str, list[list[int]]]:
    weights = _read_blocks(name, weight)
    activations = _read_blocks("Q8_0", activation)
    row = 0
    integer_subdots = []
    for index, (weight_scale, weight_integers) in enumerate(weights):
        if name == "Q2_0":
            subtotal = 0
            subdots = []
            for half in range(2):
                activation_scale, activation_integers = activations[2 * index + half]
                integer_dot = sum(
                    x * y for x, y in zip(weight_integers[half * 32:(half + 1) * 32], activation_integers)
                )
                subdots.append(integer_dot)
                term = _multiply(activation_scale, _fraction_to_bits(Fraction(integer_dot)))
                subtotal = _add(subtotal, term)
            row = _add(row, _multiply(weight_scale, subtotal))
        else:
            activation_scale, activation_integers = activations[index]
            integer_dot = sum(x * y for x, y in zip(weight_integers, activation_integers))
            subdots = [integer_dot]
            factor = _multiply(weight_scale, activation_scale)
            term = _multiply(_fraction_to_bits(Fraction(integer_dot)), factor)
            row = _add(row, term)
        integer_subdots.append(subdots)
    return _hex_bits(row), integer_subdots


def _decode_case(case_id: str, name: str, payload: bytes) -> dict:
    block_elements = _EXPECTED_FORMATS[name]["block_elements"]
    block_bytes = _EXPECTED_FORMATS[name]["block_bytes"]
    return {
        "id": case_id,
        "operation": "decode",
        "format": name,
        "elements": (len(payload) // block_bytes) * block_elements,
        "payload_hex": payload.hex(),
        "expected_fp32_bits": _reference_decode(name, payload),
    }


def _dot_case(case_id: str, name: str, weight: bytes, activation: bytes) -> dict:
    bits, subdots = _reference_dot(name, weight, activation)
    return {
        "id": case_id,
        "operation": "dot",
        "weight_format": name,
        "activation_format": "Q8_0",
        "elements": len(weight) // _EXPECTED_FORMATS[name]["block_bytes"] * _EXPECTED_FORMATS[name]["block_elements"],
        "weight_hex": weight.hex(),
        "activation_hex": activation.hex(),
        "expected_fp32_bits": bits,
        "expected_integer_subdots": subdots,
    }


def _canonical_bytes(document: dict, *, omit_digest: bool) -> bytes:
    payload = {k: v for k, v in document.items() if not omit_digest or k != "payload_sha256"}
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("ascii")


def _digest(document: dict) -> str:
    return hashlib.sha256(_canonical_bytes(document, omit_digest=True)).hexdigest()


def generate_fixture() -> dict:
    """Build a deterministic synthetic fixture without querying any DUT outputs."""
    cases = []
    for rotation in range(4):
        codes = [((i // 4) * 3 + i % 4 + rotation) % 4 for i in range(64)]
        cases.append(_decode_case(f"q2-code-lanes-{rotation}", "Q2_0", _q2_block(0x3E00, codes)))
    scales = [0x0000, 0x8000, 0x0001, 0x8001, 0x03FF, 0x83FF, 0x0400, 0x8400, 0x7BFF, 0xFBFF, 0x3555, 0xB555]
    cases.append(_decode_case(
        "q2-finite-scale-sign-zero-subnormal", "Q2_0",
        b"".join(_q2_block(scale, [0, 1, 2, 3] * 16) for scale in scales),
    ))
    cases.append(_decode_case(
        "q8-every-signed-int8", "Q8_0",
        b"".join(_q8_block(0x3C00, list(range(-128 + 32 * i, -96 + 32 * i))) for i in range(8)),
    ))
    cases.append(_decode_case(
        "q8-finite-scale-sign-zero-subnormal", "Q8_0",
        b"".join(_q8_block(scale, ([-128, -1, 0, 1, 127] * 7)[:32]) for scale in scales),
    ))
    cases.append(_decode_case("q2-little-endian-scale", "Q2_0", _q2_block(0x1234, [2] * 64)))
    cases.append(_decode_case("q8-little-endian-scale", "Q8_0", _q8_block(0x1234, [1] * 32)))

    # Hand derivation: [-1,0,1,2] dot [-2,0,2,3] = 10, eight groups = 80.
    # Other half is -80. dw=.5, da=(.25,2): .5*(.25*80+2*(-80))=-70.
    cases.append(_dot_case(
        "q2-hand-minus-70", "Q2_0", _q2_block(0x3800, [0, 1, 2, 3] * 16),
        _q8_block(0x3400, [-2, 0, 2, 3] * 8) + _q8_block(0x4000, [2, 0, -2, -3] * 8),
    ))
    # Hand derivation: (-128)*(-20)+127*3+(-7)*(-9)+5*4=3024; .5*.25*3024=378.
    cases.append(_dot_case(
        "q8-hand-378", "Q8_0", _q8_block(0x3800, [-128, 127, -7, 5] + [0] * 28),
        _q8_block(0x3400, [-20, 3, -9, 4] + [0] * 28),
    ))

    for lane in range(64):
        codes = [1] * 64
        codes[lane] = 3
        activations = [0] * 64
        activations[lane] = lane + 1
        cases.append(_dot_case(
            f"q2-lane-impulse-{lane:02d}", "Q2_0", _q2_block(0x3A00, codes),
            _q8_block(0x3800, activations[:32]) + _q8_block(0x4000, activations[32:]),
        ))
    for lane in range(32):
        weight = [0] * 32
        activation = [0] * 32
        weight[lane] = -128 + lane
        activation[lane] = lane + 1
        cases.append(_dot_case(
            f"q8-lane-impulse-{lane:02d}", "Q8_0", _q8_block(0xB800, weight), _q8_block(0x3400, activation),
        ))

    one = [1] + [0] * 31
    q2_one = [2] + [1] * 63
    for name in ("Q2_0", "Q8_0"):
        if name == "Q2_0":
            weight = _q2_block(0x0001, q2_one)
            activation = _q8_block(0x0001, one) + _q8_block(0x8001, [0] * 32)
        else:
            weight = _q8_block(0x0001, one)
            activation = _q8_block(0x0001, one)
        cases.append(_dot_case(f"{name.lower()}-subnormal-scales-no-ftz", name, weight, activation))
        for negative in (False, True):
            if name == "Q2_0":
                weight = _q2_block(0x8000 if negative else 0, [0, 1, 2, 3] * 16)
                activation = _q8_block(0x8000, [-128] * 32) + _q8_block(0, [127] * 32)
            else:
                weight = _q8_block(0x8000 if negative else 0, [-128] * 32)
                activation = _q8_block(0x8000, [127] * 32)
            cases.append(_dot_case(f"{name.lower()}-signed-zero-{'negative' if negative else 'positive'}", name, weight, activation))

    cases.append(_dot_case(
        "q2-signed-int8-negative-bound", "Q2_0", _q2_block(0x3C00, [3] * 64),
        _q8_block(0x3C00, [-128] * 32) * 2,
    ))
    cases.append(_dot_case(
        "q2-signed-int8-positive-bound", "Q2_0", _q2_block(0x3C00, [3] * 64),
        _q8_block(0x3C00, [127] * 32) * 2,
    ))
    cases.append(_dot_case(
        "q8-signed-int8-positive-bound", "Q8_0", _q8_block(0x3800, [-128] * 32),
        _q8_block(0x3400, [-128] * 32),
    ))

    # Separate binary32 adds: 2**24 + 1 rounds to even 2**24; then -2**24 = +0.
    for label, scale_pairs in (
        ("lost-unit", [(0x6C00, 0x6C00), (0x3C00, 0x3C00), (0xEC00, 0x6C00)]),
        ("preserved-unit", [(0x6C00, 0x6C00), (0xEC00, 0x6C00), (0x3C00, 0x3C00)]),
        ("two-separate-ties", [(0x6C00, 0x6C00), (0x3C00, 0x3C00), (0x3C00, 0x3C00)]),
    ):
        cases.append(_dot_case(
            f"q8-multiblock-{label}", "Q8_0",
            b"".join(_q8_block(dw, one) for dw, _ in scale_pairs),
            b"".join(_q8_block(da, one) for _, da in scale_pairs),
        ))
    big_codes = [3] * 32 + [1] * 32
    big_activation = _q8_block(0x6C00, [64] * 32) + _q8_block(0x3C00, [0] * 32)
    unit_activation = _q8_block(0x3C00, one) + _q8_block(0x3C00, [0] * 32)
    cases.append(_dot_case(
        "q2-multiblock-lost-unit", "Q2_0",
        _q2_block(0x3C00, big_codes) + _q2_block(0x3C00, q2_one) + _q2_block(0xBC00, big_codes),
        big_activation + unit_activation + big_activation,
    ))
    tie_codes = [3] * 32 + [2] + [1] * 31
    cases.append(_dot_case(
        "q2-half-subtotal-rne-tie", "Q2_0", _q2_block(0x3C01, tie_codes),
        _q8_block(0x6C00, [64] * 32) + _q8_block(0x3C00, one),
    ))
    cases.append(_dot_case(
        "q8-no-fma-witness", "Q8_0",
        _q8_block(0x0010, one) + _q8_block(0x3555, [28] + [0] * 31),
        _q8_block(0x3C00, one) + _q8_block(0x3BFF, one),
    ))
    for case_number in range(12):
        codes = [((i * 13 + case_number * 7) ^ (i // 7)) % 4 for i in range(128)]
        integers = [((i * 41 + case_number * 19) % 256) - 128 for i in range(128)]
        weight_q2 = _q2_block(0x3555, codes[:64]) + _q2_block(0xBC01, codes[64:])
        activation = b"".join(
            _q8_block(scale, integers[32 * i:32 * (i + 1)])
            for i, scale in enumerate((0x1234, 0x3BFF, 0x8401, 0xBEAB))
        )
        cases.append(_dot_case(f"q2-irregular-rounding-{case_number:02d}", "Q2_0", weight_q2, activation))
        weight_q8 = b"".join(
            _q8_block(scale, list(reversed(integers[32 * i:32 * (i + 1)])))
            for i, scale in enumerate((0x3555, 0xBC01, 0x0401, 0x3EAB))
        )
        cases.append(_dot_case(f"q8-irregular-rounding-{case_number:02d}", "Q8_0", weight_q8, activation))

    document = {
        "schema_version": 1,
        "fixture_id": FIXTURE_ID,
        "synthetic": True,
        "arithmetic_id": ARITHMETIC_ID,
        "formats": deepcopy(_EXPECTED_FORMATS),
        "provenance": deepcopy(_EXPECTED_PROVENANCE),
        "cases": cases,
    }
    document["payload_sha256"] = _digest(document)
    return document


def _same_typed_value(actual, expected) -> bool:
    if type(actual) is not type(expected):
        return False
    if type(expected) is dict:
        return actual.keys() == expected.keys() and all(_same_typed_value(actual[k], v) for k, v in expected.items())
    if type(expected) is list:
        return len(actual) == len(expected) and all(_same_typed_value(a, e) for a, e in zip(actual, expected))
    return actual == expected


def _check_keys(value, expected: set[str], path: str):
    if type(value) is not dict or set(value) != expected:
        _fail("fixture_schema", f"{path}: expected exact object keys")


def _check_bits(value, path: str):
    if type(value) is not str or not _BITS_PATTERN.fullmatch(value):
        _fail("fixture_schema", f"{path}: expected lowercase 0x and eight binary32 hex digits")
    if (int(value[2:], 16) >> 23) & 255 == 255:
        _fail("fixture_schema", f"{path}: binary32 expectation must be finite")


def _check_elements(value, name: str, path: str):
    block = _EXPECTED_FORMATS[name]["block_elements"]
    if type(value) is not int or not 1 <= value <= MAX_ELEMENTS or value % block:
        _fail("fixture_schema", f"{path}: expected positive whole {block}-element blocks up to {MAX_ELEMENTS}")


def _check_payload(value, name: str, elements: int, path: str) -> bytes:
    metadata = _EXPECTED_FORMATS[name]
    expected_bytes = elements // metadata["block_elements"] * metadata["block_bytes"]
    if type(value) is not str or len(value) != 2 * expected_bytes or not _HEX_PATTERN.fullmatch(value):
        _fail("fixture_schema", f"{path}: expected exactly {expected_bytes} bytes of contiguous hex")
    payload = bytes.fromhex(value)
    stride = metadata["block_bytes"]
    for start in range(0, len(payload), stride):
        raw = int.from_bytes(payload[start:start + 2], "little")
        if (raw >> 10) & 31 == 31:
            _fail("fixture_schema", f"{path}: nonfinite binary16 scale at block {start // stride}")
    return payload


def _validate_fixture(document) -> tuple[bytes, list[tuple[dict, bytes, bytes | None]]]:
    _check_keys(document, _TOP_KEYS, "fixture")
    if type(document["schema_version"]) is not int or document["schema_version"] != 1:
        _fail("fixture_schema", "fixture.schema_version: expected integer 1")
    for key, expected in (
        ("fixture_id", FIXTURE_ID), ("synthetic", True), ("arithmetic_id", ARITHMETIC_ID),
        ("formats", _EXPECTED_FORMATS), ("provenance", _EXPECTED_PROVENANCE),
    ):
        if not _same_typed_value(document[key], expected):
            _fail("fixture_schema", f"fixture.{key}: does not match frozen synthetic contract")
    digest = document["payload_sha256"]
    if type(digest) is not str or not _SHA_PATTERN.fullmatch(digest):
        _fail("fixture_schema", "fixture.payload_sha256: expected 64 lowercase hex digits")
    cases = document["cases"]
    if type(cases) is not list or not 1 <= len(cases) <= MAX_CASES:
        _fail("fixture_schema", f"fixture.cases: expected 1..{MAX_CASES} cases")
    try:
        canonical = _canonical_bytes(document, omit_digest=False)
        claimed_payload = _canonical_bytes(document, omit_digest=True)
    except (ValueError, TypeError, RecursionError, OverflowError):
        _fail("fixture_schema", "fixture: cannot encode bounded finite canonical JSON")
    if len(canonical) > MAX_FIXTURE_BYTES:
        _fail("fixture_schema", f"fixture: canonical JSON exceeds {MAX_FIXTURE_BYTES} bytes")
    if hashlib.sha256(claimed_payload).hexdigest() != digest:
        _fail("fixture_digest", "fixture.payload_sha256: canonical payload digest mismatch")

    seen_ids = set()
    prepared = []
    for index, case in enumerate(cases):
        path = f"fixture.cases[{index}]"
        if type(case) is not dict:
            _fail("fixture_schema", f"{path}: expected an object")
        operation = case.get("operation")
        if type(operation) is not str or operation not in ("decode", "dot"):
            _fail("fixture_schema", f"{path}.operation: expected decode or dot")
        _check_keys(case, _DECODE_KEYS if operation == "decode" else _DOT_KEYS, path)
        case_id = case["id"]
        if type(case_id) is not str or not _ID_PATTERN.fullmatch(case_id) or case_id in seen_ids:
            _fail("fixture_schema", f"{path}.id: expected unique bounded lowercase case id")
        seen_ids.add(case_id)
        name = case["format"] if operation == "decode" else case["weight_format"]
        if type(name) is not str or name not in _EXPECTED_FORMATS:
            _fail("fixture_schema", f"{path}: unsupported supplied packed format")
        elements = case["elements"]
        _check_elements(elements, name, f"{path}.elements")
        if operation == "decode":
            payload = _check_payload(case["payload_hex"], name, elements, f"{path}.payload_hex")
            bits = case["expected_fp32_bits"]
            if type(bits) is not list or len(bits) != elements:
                _fail("fixture_schema", f"{path}.expected_fp32_bits: expected exactly {elements} values")
            for lane, value in enumerate(bits):
                _check_bits(value, f"{path}.expected_fp32_bits[{lane}]")
            prepared.append((case, payload, None))
        else:
            if type(case["activation_format"]) is not str or case["activation_format"] != "Q8_0":
                _fail("fixture_schema", f"{path}.activation_format: expected Q8_0")
            weight = _check_payload(case["weight_hex"], name, elements, f"{path}.weight_hex")
            activation = _check_payload(case["activation_hex"], "Q8_0", elements, f"{path}.activation_hex")
            _check_bits(case["expected_fp32_bits"], f"{path}.expected_fp32_bits")
            subdots = case["expected_integer_subdots"]
            expected_rows = elements // _EXPECTED_FORMATS[name]["block_elements"]
            expected_columns = 2 if name == "Q2_0" else 1
            bounds = (-8192, 8128) if name == "Q2_0" else (-520192, 524288)
            if type(subdots) is not list or len(subdots) != expected_rows:
                _fail("fixture_schema", f"{path}.expected_integer_subdots: wrong block count")
            for row_index, row in enumerate(subdots):
                if type(row) is not list or len(row) != expected_columns:
                    _fail("fixture_schema", f"{path}.expected_integer_subdots[{row_index}]: wrong half count")
                if any(type(value) is not int or not bounds[0] <= value <= bounds[1] for value in row):
                    _fail("fixture_schema", f"{path}.expected_integer_subdots[{row_index}]: invalid integer range/type")
            prepared.append((case, weight, activation))
    return canonical, prepared


def verify_fixture(document) -> dict:
    """Strictly validate supplied metadata, then compare DUT outputs to goldens.

    Raises formats.FormatError for malformed documents. Validly structured but
    false expectations produce verified=False and a bounded mismatch report.
    Input metadata never authorizes real-model, native, RTL, or device operations.
    """
    _, prepared = _validate_fixture(document)
    from . import formats as dut
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
            result = dut.dot(case["weight_format"], first, second, case["elements"], activation_format=case["activation_format"])
            comparisons = [
                ("fp32_bits", case["expected_fp32_bits"], result["fp32_bits"]),
                ("integer_subdots", case["expected_integer_subdots"], result["integer_subdots"]),
            ]
        for field, expected, actual in comparisons:
            if not _same_typed_value(actual, expected):
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
