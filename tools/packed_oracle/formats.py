# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Niko1221 and the Strata contributors
# Copyright (c) 2023-2026 The ggml authors
# Copyright (c) 2026 lvyufeng and the openTPU contributors
# Adapted from the pinned Strata scalar layout/dequant/dot references.
# See ATTRIBUTION.md and licenses/ for full notices and modification records.

"""Strict supplied-packed Q2_0/Q8_0 scalar references, without native code.

The public results contain binary32 *bits*, not JSON floating-point values.
Every multiplication and addition is rounded separately using integer dyadic
arithmetic: round to nearest, ties to even, with gradual underflow and signed
zero.  No Python floating-point operation determines a result bit.
"""

from __future__ import annotations

import re


ARITHMETIC_ID = "strata-generic-f32-rne-separate-v1"
MAX_ELEMENTS = 65536
MAX_PAYLOAD_BYTES = (MAX_ELEMENTS // 32) * 34

# The published JSON metadata is deliberately small and contains no codebooks.
FORMATS = {
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

# Internal sizes do not depend on mutation of the public metadata dictionary.
_BLOCK_SIZES = {"Q2_0": (64, 18), "Q8_0": (32, 34)}
_HEX = re.compile(r"[0-9a-fA-F]+", flags=re.ASCII)


class FormatError(ValueError):
    """A malformed or unsupported input, with a stable machine-readable code."""

    def __init__(self, code: str, message: str) -> None:
        self.code = code
        self.message = message
        super().__init__(message)


def parse_hex(text: str) -> bytes:
    """Parse bounded, contiguous, even-length ASCII hex; never strip input."""
    if type(text) is not str:
        raise FormatError("invalid_hex", "hex input must be a string")
    if len(text) > 2 * MAX_PAYLOAD_BYTES:
        raise FormatError("invalid_hex", "hex input exceeds the packed-row byte limit")
    if not text or len(text) % 2 or _HEX.fullmatch(text) is None:
        raise FormatError(
            "invalid_hex", "hex must contain contiguous pairs of ASCII hex digits"
        )
    # fullmatch above forbids the whitespace accepted by bytes.fromhex itself.
    return bytes.fromhex(text)


def _format_size(format_name: str) -> tuple[int, int]:
    if type(format_name) is not str or format_name not in _BLOCK_SIZES:
        raise FormatError(
            "unsupported_format", "supported supplied-packed formats are Q2_0 and Q8_0"
        )
    return _BLOCK_SIZES[format_name]


def _validate_row(
    format_name: str, payload: bytes, elements: int, label: str
) -> int:
    block_elements, block_bytes = _format_size(format_name)
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


def _round_shift_even(magnitude: int, shift: int) -> int:
    """Round a nonnegative integer divided by 2**shift, ties to even."""
    if shift <= 0:
        return magnitude << -shift
    quotient, remainder = divmod(magnitude, 1 << shift)
    halfway = 1 << (shift - 1)
    if remainder > halfway or (remainder == halfway and quotient & 1):
        quotient += 1
    return quotient


def _round_dyadic(magnitude: int, exponent: int, negative: bool = False) -> int:
    """Encode exact magnitude*2**exponent as finite RNE binary32 bits."""
    sign = 0x80000000 if negative else 0
    if not magnitude:
        return sign
    unbiased = magnitude.bit_length() - 1 + exponent
    if unbiased < -126:
        # Subnormal spacing is 2**-149.  A rounded subnormal can become the
        # smallest normal, whose bit pattern is also the integer 2**23.
        significand = _round_shift_even(magnitude, -149 - exponent)
        return sign | significand
    significand = _round_shift_even(magnitude, magnitude.bit_length() - 24)
    if significand == 1 << 24:
        significand >>= 1
        unbiased += 1
    if unbiased > 127:
        raise FormatError("nonfinite_result", "binary32 arithmetic overflowed")
    return sign | ((unbiased + 127) << 23) | (significand - (1 << 23))


def _dyadic(bits: int) -> tuple[int, int]:
    """Return an exact signed integer coefficient and binary exponent."""
    exponent_field = (bits >> 23) & 0xFF
    fraction = bits & 0x7FFFFF
    if exponent_field == 0:
        coefficient, exponent = fraction, -149
    else:
        coefficient, exponent = (1 << 23) | fraction, exponent_field - 150
    if bits & 0x80000000:
        coefficient = -coefficient
    return coefficient, exponent


def _int32(value: int) -> int:
    return _round_dyadic(abs(value), 0, value < 0)


def _mul32(left: int, right: int) -> int:
    left_coefficient, left_exponent = _dyadic(left)
    right_coefficient, right_exponent = _dyadic(right)
    return _round_dyadic(
        abs(left_coefficient * right_coefficient),
        left_exponent + right_exponent,
        bool((left ^ right) & 0x80000000),
    )


def _add32(left: int, right: int) -> int:
    left_coefficient, left_exponent = _dyadic(left)
    right_coefficient, right_exponent = _dyadic(right)
    exponent = min(left_exponent, right_exponent)
    coefficient = (left_coefficient << (left_exponent - exponent)) + (
        right_coefficient << (right_exponent - exponent)
    )
    if not coefficient:
        # RN cancellation is +0.  Only adding two negative zeros gives -0.
        negative_zero = (
            left_coefficient == right_coefficient == 0
            and bool(left & right & 0x80000000)
        )
        return 0x80000000 if negative_zero else 0
    return _round_dyadic(abs(coefficient), exponent, coefficient < 0)


def _half32(half_bits: int) -> int:
    """Exactly widen finite IEEE binary16 bits, including signed subnormals."""
    negative = bool(half_bits & 0x8000)
    exponent = (half_bits >> 10) & 0x1F
    fraction = half_bits & 0x03FF
    if exponent == 0x1F:
        raise FormatError("nonfinite_scale", "FP16 scale must not be infinity or NaN")
    if not exponent:
        return _round_dyadic(fraction, -24, negative)
    sign = 0x80000000 if negative else 0
    return sign | ((exponent + 112) << 23) | (fraction << 13)


def _unpack(
    format_name: str, payload: bytes, blocks: int
) -> list[tuple[int, list[int]]]:
    block_elements, block_bytes = _BLOCK_SIZES[format_name]
    result = []
    for block in range(blocks):
        offset = block * block_bytes
        half_bits = payload[offset] | (payload[offset + 1] << 8)
        scale = _half32(half_bits)
        packed = payload[offset + 2 : offset + block_bytes]
        if format_name == "Q2_0":
            values = [
                ((packed[index // 4] >> ((index % 4) * 2)) & 3) - 1
                for index in range(block_elements)
            ]
        else:
            values = [byte if byte < 128 else byte - 256 for byte in packed]
        result.append((scale, values))
    return result


def _hex32(bits: int) -> str:
    return f"0x{bits:08x}"


def decode(format_name: str, payload: bytes, elements: int) -> dict:
    """Decode one exact, whole-block row of supplied packed bytes.

    Q2_0 uses low-to-high two-bit lanes mapped to -1, 0, +1, +2.  Q8_0
    uses signed int8 lanes.  Each block starts with little-endian FP16 scale.
    This is not a quantizer, a GGUF reader, or an expert/model loader.
    """
    blocks = _validate_row(format_name, payload, elements, "decode")
    values = [
        _hex32(_mul32(_int32(value), scale))
        for scale, integers in _unpack(format_name, payload, blocks)
        for value in integers
    ]
    return {
        "format": format_name,
        "elements": elements,
        "blocks": blocks,
        "fp32_bits": values,
    }


def dot(
    weight_format: str,
    weight_payload: bytes,
    activation_payload: bytes,
    elements: int,
    activation_format: str = "Q8_0",
) -> dict:
    """Generic-contract row dot with exact integer subdots and separate FP32.

    A Q2_0 block pairs with two Q8_0 blocks.  Q8_0 weights pair one-to-one.
    This deliberately does not claim native SIMD, RTL, FPGA, or model parity.
    """
    if type(activation_format) is not str or activation_format != "Q8_0":
        raise FormatError(
            "unsupported_activation_format", "the only activation format is Q8_0"
        )
    weight_blocks = _validate_row(weight_format, weight_payload, elements, "weight")
    activation_blocks = _validate_row(
        activation_format, activation_payload, elements, "activation"
    )
    weights = _unpack(weight_format, weight_payload, weight_blocks)
    activations = _unpack(activation_format, activation_payload, activation_blocks)
    integer_subdots = []
    row = 0  # +0, fixed increasing block order.
    for block, (weight_scale, integers) in enumerate(weights):
        if weight_format == "Q2_0":
            subtotal = 0  # +0 independently for each 64-element weight block.
            subdots = []
            for half in range(2):
                activation_scale, activation_integers = activations[block * 2 + half]
                integer_dot = sum(
                    integers[half * 32 + lane] * activation_integers[lane]
                    for lane in range(32)
                )
                subdots.append(integer_dot)
                product = _mul32(activation_scale, _int32(integer_dot))
                subtotal = _add32(subtotal, product)
            integer_subdots.append(subdots)
            row = _add32(row, _mul32(weight_scale, subtotal))
        else:
            activation_scale, activation_integers = activations[block]
            integer_dot = sum(
                integers[lane] * activation_integers[lane] for lane in range(32)
            )
            integer_subdots.append([integer_dot])
            factor = _mul32(weight_scale, activation_scale)
            product = _mul32(_int32(integer_dot), factor)
            row = _add32(row, product)
    return {
        "weight_format": weight_format,
        "activation_format": activation_format,
        "elements": elements,
        "weight_blocks": weight_blocks,
        "activation_blocks": activation_blocks,
        "integer_subdots": integer_subdots,
        "fp32_bits": _hex32(row),
        "arithmetic_id": ARITHMETIC_ID,
    }
