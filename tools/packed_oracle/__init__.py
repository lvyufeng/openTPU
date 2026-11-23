# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 lvyufeng and the openTPU contributors

"""Opt-in, stdlib-only supplied-packed scalar references.

Importing this package does not import openTPU runtime/driver modules, read
files, or acquire any hardware.  The adapted mathematical implementation in
formats.py is MIT-licensed; see ATTRIBUTION.md and licenses/.

Two independent families live here:

* ``decode``/``dot`` cover the supplied Q2_0/Q8_0 formats in ``formats.py``.
* ``iq_decode``/``iq_dot`` cover IQ2_XXS/IQ2_S/IQ1_M weights against Q8_K
  activations in ``iq_formats.py``.  The ``iq_*`` names are aliases so the
  original Q2_0/Q8_0 API above keeps its exact spelling and result keys.
"""

from .formats import ARITHMETIC_ID, FORMATS, FormatError, decode, dot, parse_hex
from .iq_formats import (
    ACTIVATIONS as IQ_ACTIVATIONS,
    FORMATS as IQ_FORMATS,
    block_layout as iq_block_layout,
    decode as iq_decode,
    dot as iq_dot,
)

__all__ = [
    "ARITHMETIC_ID",
    "FORMATS",
    "FormatError",
    "IQ_ACTIVATIONS",
    "IQ_FORMATS",
    "decode",
    "dot",
    "iq_block_layout",
    "iq_decode",
    "iq_dot",
    "parse_hex",
]
