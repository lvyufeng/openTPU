# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 lvyufeng and the openTPU contributors

"""Opt-in, stdlib-only supplied-packed scalar references.

Importing this package does not import openTPU runtime/driver modules, read
files, or acquire any hardware.  The adapted mathematical implementation in
formats.py is MIT-licensed; see ATTRIBUTION.md and licenses/.
"""

from .formats import ARITHMETIC_ID, FORMATS, FormatError, decode, dot, parse_hex

__all__ = ["ARITHMETIC_ID", "FORMATS", "FormatError", "decode", "dot", "parse_hex"]
