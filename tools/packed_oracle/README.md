<!-- SPDX-License-Identifier: Apache-2.0 -->

# Packed-byte scalar oracle

This opt-in, Python-standard-library tool checks **supplied synthetic packed
bytes**, not model inference. It implements Strata's Q2_0 and Q8_0 scalar row
references. Run it with Python 3.11 or 3.14; no installation is needed.

**Nothing in the openTPU runtime, MXU, VPU, or existing nibble-packed format is
changed.** This package does not `import opentpu`, use native extensions, load a
model, quantize activations, open a device, issue DMA, or execute RTL. Q2_0 is
Strata's two-bit format, **not IQ2_XS, Q2_K, or FP4**. IQ2_XS and IQ3_S are
explicitly unsupported, not silently interpreted as Q2_0.

## Fixed formats and arithmetic

| Format | Strata/GGML type ID | Elements/block | Bytes/block | Payload after FP16 scale |
| --- | ---: | ---: | ---: | --- |
| Q2_0 | 42 | 64 | 18 | 16 bytes, four two-bit lanes/byte |
| Q8_0 | 8 | 32 | 34 | 32 signed int8 lanes |

Each block begins with a **little-endian IEEE binary16** scale. Negative
scales, both signed zeros, and finite subnormals are supported. Infinity and
NaN scales are rejected. Q2_0 lanes are read low-to-high within each byte and
mapped `00 -> -1`, `01 -> 0`, `10 -> +1`, `11 -> +2`.

The only dot activation format is Q8_0. One Q2_0 weight block pairs with two
Q8_0 activation blocks; one Q8_0 weight block pairs with one activation block.
All integer subdots are exact and exposed in the result.

The arithmetic identifier is `strata-generic-f32-rne-separate-v1`.
Every multiplication and addition rounds independently to IEEE binary32,
**round to nearest, ties to even**, without FMA, reassociation, FTZ, or DAZ.
Each row starts at +0 and processes blocks in increasing order. Q2_0 also
starts each 64-element subtotal at +0:

```text
Q2_0: for each of two 32-element subdots:
          subtotal = r32(subtotal + r32(activation_scale * integer_subdot))
      row = r32(row + r32(weight_scale * subtotal))
Q8_0: factor = r32(weight_scale * activation_scale)
      row = r32(row + r32(integer_subdot * factor))
```

The [scalar implementation](<formats.py>) uses exact signed integer dyadics
and a binary32 encoder. Python's floating-point intermediates and the host's
rounding/ISA environment do not determine result bits. Products and sums
are not combined. Gradual underflow and signed zeros are preserved; a
non-finite arithmetic result is an explicit error. This is a fixed generic
reference, **not a claim about a compiler's FMA settings, native AVX kernels,
RTL math, or FPGA math**.

## API

```python
from tools.packed_oracle import (
    ARITHMETIC_ID, FORMATS, FormatError, decode, dot, parse_hex,
)

weight = parse_hex("003c" + "e4" * 16)
activation = parse_hex("003c" + "01" * 32)
values = decode("Q2_0", weight, 64)
product = dot("Q2_0", weight, activation * 2, 64)
```

`decode(format_name, payload: bytes, elements: int)` returns exactly
`format`, `elements`, `blocks`, and `fp32_bits` (a list). `dot(weight_format,
weight_payload, activation_payload, elements, activation_format="Q8_0")`
returns exactly `weight_format`, `activation_format`, `elements`,
`weight_blocks`, `activation_blocks`, `integer_subdots`, `fp32_bits` (one
string), and `arithmetic_id`. Every result bit string is lowercase
`0x` plus eight hexadecimal digits. Q2_0 integer subdots are one `[a, b]`
pair per weight block; Q8_0 uses one `[a]` per weight block.

Dimensions must be integers (not booleans), positive, at most 65,536, and
whole blocks. The payload must be `bytes` of the exact size; trailing data,
short data, implicit padding, and bytearray coercion are rejected. Format
names are case-sensitive. `parse_hex` accepts only bounded, nonempty,
contiguous, even-length ASCII hex: no spaces, prefixes, or stripping.
`FormatError` exposes stable `code` and `message` fields. Codes include
`unsupported_format`, `unsupported_activation_format`, `invalid_elements`,
`invalid_payload`, `invalid_hex`, `invalid_length`, `nonfinite_scale`, and
`nonfinite_result`.

## CLI

Run from the openTPU checkout root. `-B -S` disables bytecode writes and site
initialization; the package itself has no third-party dependencies.

```sh
python3 -B -S -m tools.packed_oracle decode --format Q2_0 --elements 64 \
  --hex 003ce4e4e4e4e4e4e4e4e4e4e4e4e4e4e4e4
python3 -B -S -m tools.packed_oracle dot --weight-format Q2_0 \
  --activation-format Q8_0 --elements 64 \
  --weight-hex 003ce4e4e4e4e4e4e4e4e4e4e4e4e4e4e4e4 \
  --activation-hex 003c0101010101010101010101010101010101010101010101010101010101010101003c0101010101010101010101010101010101010101010101010101010101010101
python3 -B -S -m tools.packed_oracle verify
python3 -B -S -m tools.packed_oracle verify tools/packed_oracle/data/q2_q8_synthetic.json
python3 -B -S -m tools.packed_oracle generate --stdout
```

`decode` and `dot` print JSON with exactly `schema_version: 1`,
`scope: "synthetic-packed-reference"`, `result`, `device_ready: false`, and
`inference_ready: false`. Valid fixture verification prints its report.
Generation prints deterministically formatted JSON to stdout only and does
not overwrite files. Verification's default is the
[shipped fixture](<data/q2_q8_synthetic.json>) relative to this package, not the
current directory.

Exit **0** means success, **1** means a structurally valid fixture contains a
reference mismatch, and **2** means malformed/unsupported input. Errors are
one JSON object on stderr, with `error: {code, message}`, schema/scope and
both readiness flags false; no traceback or input payload is echoed. CLI
arguments are not abbreviated. `generate` requires `--stdout`. The fixture
reader bounds input to 8 MiB, JSON nesting to 64 levels, and numeric literals
to 64 characters. It rejects invalid UTF-8/JSON, duplicate object keys, and
non-finite numbers (including overflowing decimal exponents). Fixture
schema/provenance validation remains strict even if its digest is recalculated.

## Independent synthetic evidence

The [fixture generator and verifier](<vectors.py>) derive expected bits with
an independent exact rational binary32 implementation, not calls to the DUT
while generating. Cases cover supplied-packed mapping/sign/scale/rounding
and dot operation order; they are **not samples from actual model tensors**.
The fixture fixes its schema, formats, source revisions and line anchors,
generator identity, case fields, expected bits, and scope exclusions. Its
`payload_sha256` is SHA-256 of the entire canonical JSON document with only
that field removed (sorted keys, compact separators, ASCII escaping, no
non-finite numbers). The digest is an integrity check, not authentication.

The shipped fixture contains **146 cases: 9 decode and 137 dot cases**. Its
canonical payload digest is
`59a84c7e1884e50a31798cee3a8f171229e26b361d09d80b0f57ddc0c3c04efa`.
Hand-derived dots include Q2_0 = -70 (`0xc28c0000`, subdots `[80, -80]`)
and Q8_0 = 378 (`0x43bd0000`, subdot `3024`). Multiblock rounding witnesses
include `[2^24, 1, -2^24] -> +0` versus `[2^24, -2^24, 1] -> 1`, and
`[2^24, 1, 1] -> 2^24` rather than a combined-add result of `2^24 + 2`.

The report contains `schema_version`, `fixture_id`, `arithmetic_id`,
`payload_sha256`, `case_count`, `decode_cases`, `dot_cases`, `synthetic`,
`verified`, `mismatches`, `device_ready`, and `inference_ready`. A mismatch
records `case_id`, `field`, `expected`, and `actual`. Success always keeps
both readiness flags false.

Local source-only checks:

```sh
python3 -B -S -m unittest discover -s tools/packed_oracle/tests -v
python3 -B -S -m tools.packed_oracle verify
python3 -B -S -m tools.packed_oracle generate --stdout > /tmp/q2_q8_regenerated.json
cmp tools/packed_oracle/data/q2_q8_synthetic.json /tmp/q2_q8_regenerated.json
```

The [dedicated CI](<../../.github/workflows/packed-oracle.yml>) runs these
stdlib checks under Python 3.11 and 3.14 only. It does not build or run existing
openTPU native, simulator, runtime, RTL, or board paths. Passing this fixture
is **synthetic source-only arithmetic evidence**, not real-model coverage,
native CPU/AVX parity, RTL/FPGA parity, DMA/device behavior, quantizer parity,
performance, or full decode correctness/throughput. No >10 tok/s claim follows.

## Licenses and provenance

The adapted [mathematical module](<formats.py>) and its modifications are
MIT-licensed with full [Strata](<licenses/STRATA-MIT.txt>),
[ggml](<licenses/GGML-MIT.txt>), and
[modification](<licenses/MODIFICATIONS-MIT.txt>) notices. New CLI/package,
fixture/test scaffolding, documentation, and CI retain openTPU's Apache-2.0
license. See the [adaptation record](<ATTRIBUTION.md>) and the repository
[Apache-2.0 license](<../../LICENSE>). These software licenses do not grant
model-weight or vendor-IP rights.
