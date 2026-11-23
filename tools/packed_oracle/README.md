<!-- SPDX-License-Identifier: Apache-2.0 -->

# Packed-byte scalar oracle

This opt-in, Python-standard-library tool checks **supplied synthetic packed
bytes**, not model inference. It implements Strata's Q2_0 and Q8_0 scalar row
references, plus the IQ2_XXS, IQ2_S and IQ1_M packed-integer weight formats
against Q8_K activations. Run it with Python 3.11 or 3.14; no installation is
needed.

**Nothing in the openTPU runtime, MXU, VPU, or existing nibble-packed format is
changed.** This package does not `import opentpu`, use native extensions, load a
model, quantize activations, open a device, issue DMA, or execute RTL. Q2_0 is
Strata's two-bit format, **not IQ2_XS, Q2_K, or FP4**. The IQ support here is
limited to the ggml IQ2_XXS/IQ2_S/IQ1_M grid codebooks; IQ2_XS, IQ3_S, IQ4_NL,
Q2_K and FP4 remain explicitly unsupported and are never silently interpreted
as Q2_0 or as one of the supported IQ formats.

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

## IQ formats: block layout, decode, and integer group dots

Three ggml/Strata IQ weight formats pair with the supplied `Q8_K` activation
format. Every weight block holds 256 values in eight 32-element groups (ggml's
`ib32`); every activation block holds 256 values as one FP32 scale, 256 signed
int8 values, and sixteen int16 sums.

| Format | Type ID | Grid codebook | Bytes/block | Group scale bits | Index bits | Sign source |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| IQ2_XXS | 16 | 256 x 8-bit | 66 | 4-bit, 16 levels | 8 | `ksigns_iq2xs[...]` nibble-selected byte |
| IQ2_S | 22 | 1024 x 10-bit | 82 | 2 x 4-bit, 16 levels | 10 | raw stored sign byte |
| IQ1_M | 29 | 2048 x 11-bit | 56 | 3-bit, 8 levels, FP16 packed in high nibbles | 11 | four delta-flag bits |

Little-endian raw block layouts (the C structs in
`engine/include/strata/quant/strata-quant-common.h` at the locked Strata
revision `3142b66b96e9c0e775e30f620b11b191a1427c2a`):

```text
Q8_K      d:f32 @0 (4) | qs:int8[256] @4 | bsums:int16[16] @260            = 292
IQ2_XXS   d:fp16 @0 (2) | qs:u8[64] @2                                     =  66
IQ2_S     d:fp16 @0 (2) | qs:u8[64] @2 (first 32 = index bits, last 32 = signs)
          qh:u8[8] @66 | scales:u8[8] @74                                  =  82
IQ1_M     qs:u8[32] @0 | qh:u8[16] @32 | scales:u8[8] @48                  =  56
```

Decode formulas, transcribed from the pinned ggml dequantizers (ggml revision
`3cf03257f219afbe7334045ff7c6a06ac68c627d`, vendored unchanged in the locked
Strata tree):

```text
IQ2_XXS  dequantize_row_iq2_xxs  (ggml-quants.c:2488-2512)
    db    = d * (0.5f + (aux32[1] >> 28)) * 0.25f
    grid  = iq2xxs_grid[first 4 bytes of the group][l]
    signs = ksigns_iq2xs[(aux32[1] >> 7*l) & 127]
    y[j]  = db * grid[j] * ((signs & kmask_iq2xs[j]) ? -1.f : 1.f)

IQ2_S    dequantize_row_iq2_s    (ggml-quants.c:2543-2577)
    db[0] = d * (0.5f + (scales[ib32] & 0xf)) * 0.25f
    db[1] = d * (0.5f + (scales[ib32] >>  4)) * 0.25f
    index = qs[l] | ((qh[ib32] << (8 - 2*l)) & 0x300)   l = 0..3
    y[j]  = db[l/2] * iq2s_grid[index][j] *
            ((signs[l] & kmask_iq2xs[j]) ? -1.f : 1.f)  signs = qs + QK_K/8

IQ1_M    dequantize_row_iq1_m    (ggml-quants.c:2675-2714)
    d    = fp16((sc[0]>>12) | ((sc[1]>>8)&0x00f0) |
                ((sc[2]>>4)&0x0f00) | (sc[3]&0xf000))       sc = scales as 4 x uint16
    dl1  = d * (2*((sc[ib/2] >> (6*(ib%2)+0)) & 0x7) + 1)
    dl2  = d * (2*((sc[ib/2] >> (6*(ib%2)+3)) & 0x7) + 1)
    idx  = qs[l] | ((qh[l/2] << (8 - 4*(l%2))) & 0x700)
    delta = (qh[l/2] & (l%2 == 0 ? 0x08 : 0x80)) ? -0.125f : +0.125f
    y[j] = (l < 2 ? dl1 : dl2) * (iq1s_grid[idx][j] + delta)
```

IQ1_M stores four FP16 bits (the block scale) in the top nibble of each of its
four 16-bit scale words, packs six 3-bit group scales into the remaining bits,
and never has a standalone `d` field. IQ2_XXS carries one 4-bit group scale per
32 values; IQ2_S carries two (one per 16-value half). Its sign byte is used
verbatim, unlike IQ2_XXS which looks the byte up in `ksigns_iq2xs`.

Integer group-dot semantics (generic ggml CPU kernels in
`ggml/src/ggml-cpu/quants.c` at the same pinned revision):

```text
IQ2_XXS  vec_dot_iq2_xxs_q8_K  (quants.c:906-946)
    per group: ls = 2*(aux32[1] >> 28) + 1
               sumi = sum over 4 x 8 lanes of grid[j]*q8[j]*(sign ? -1 : 1)
               bsum += ls*sumi
IQ2_S    vec_dot_iq2_s_q8_K    (quants.c:998-1048)
    per group: ls1 = 1 + 2*(scales & 0xf), ls2 = 1 + 2*(scales >> 4)
               sumi1/lsum lanes 0-1, sumi2/lsum lanes 2-3
               bsum += ls1*sumi1 + ls2*sumi2
IQ1_M    vec_dot_iq1_m_q8_K    (quants.c:1193-1252)
    per group: sum1[h] = signed half-grid dot, sum2[h] = sum of q8 times delta
               bsum += ls1*sum1[0] + ls2*sum1[1]  (grid term)
               bsum2 += ls1*sum2[0] + ls2*sum2[1] (delta term)

FP32 combination (one RNE rounding per multiply/add, in this order)
    d = fp16(weight scale) * Q8_K.d          (IQ1_M: rebuilt fp16 scale)
    IQ2_XXS/IQ2_S: sumf += d * bsum per block;  result = 0.125f * sumf
    IQ1_M:         sumf += d * (bsum + 0.125f * bsum2);  result = sumf
```

`integer_subdots` is `[block][8 groups][terms]` and holds the already
scale-multiplied integer terms above: one term for IQ2_XXS, two
(`low16`, `high16`) for IQ2_S, two (`grid`, `delta`) for IQ1_M. They are exact
Python integers, so they are the integer-exact contract an RTL datapath must
reproduce *before* any FP32 scaling. Magnitudes fit signed 32 bits with wide
margin (worst-case group term is about 5.5 million; a whole IQ2 block sum is
about 44 million). No saturation, clamping, FMA, reassociation, FTZ, or DAZ is
implied anywhere.

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

The IQ family is exported under aliased names so the Q2_0/Q8_0 surface keeps
its exact spelling and result keys:

```python
from tools.packed_oracle import (
    IQ_ACTIVATIONS, IQ_FORMATS, iq_block_layout, iq_decode, iq_dot,
)

layout = iq_block_layout("IQ2_S")       # JSON-safe offset/size descriptors
values = iq_decode("IQ2_XXS", weight, 512)
product = iq_dot("IQ2_XXS", weight, q8k_activation, 512)  # Q8_K is the default
```

`iq_block_layout(format_name)` returns the block field descriptors
(`name`, `offset`, `bytes`) and the block size. `iq_decode` returns exactly
`format`, `elements`, `blocks`, `fp32_bits`. `iq_dot(weight_format,
weight_payload, activation_payload, elements, activation_format="Q8_K")`
returns exactly `weight_format`, `activation_format`, `elements`,
`weight_blocks`, `activation_blocks`, `integer_subdots`, `fp32_bits`, and
`arithmetic_id`; `integer_subdots` is `[block][8 groups][terms]` as described
above. IQ payload sizes are 66/82/56 bytes per weight block and 292 bytes per
Q8_K block; `Q8_K.bsums` are parsed but intentionally not consumed, exactly
like the reference kernels. IQ elements must be multiples of 256.

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
python3 -B -S -m tools.packed_oracle iq-decode --format IQ2_XXS --elements 256 \
  --hex 010000ff00ff803fe00f00ff00ff803fe01f00ff00ff803fe02f00ff00ff803fe03f00ff00ff803fe04f00ff00ff803fe05f00ff00ff803fe06f00ff00ff803fe07f
python3 -B -S -m tools.packed_oracle iq-dot --weight-format IQ2_XXS \
  --activation-format Q8_K --elements 256 --weight-hex 010000ff00ff803fe00f00ff00ff803fe01f00ff00ff803fe02f00ff00ff803fe03f00ff00ff803fe04f00ff00ff803fe05f00ff00ff803fe06f00ff00ff803fe07f \
  --activation-hex 0000803f807f00ff0140c0817e02fe649c03fd7f807f00ff0140c0817e02fe649c03fd7f807f00ff0140c0817e02fe649c03fd7f807f00ff0140c0817e02fe649c03fd7f807f00ff0140c0817e02fe649c03fd7f807f00ff0140c0817e02fe649c03fd7f807f00ff0140c0817e02fe649c03fd7f807f00ff0140c0817e02fe649c03fd7f807f00ff0140c0817e02fe649c03fd7f807f00ff0140c0817e02fe649c03fd7f807f00ff0140c0817e02fe649c03fd7f807f00ff0140c0817e02fe649c03fd7f807f00ff0140c0817e02fe649c03fd7f807f00ff0140c0817e02fe649c03fd7f807f00ff0140c0817e02fe649c03fd7f807f00ff0140c0817e02fe649c03fd7f7d007d007d007d007d007d007d007d007d007d007d007d007d007d007d007d00
python3 -B -S -m tools.packed_oracle iq-verify
python3 -B -S -m tools.packed_oracle iq-generate --stdout
python3 -B -S -m tools.packed_oracle iq-rtl-verify
python3 -B -S -m tools.packed_oracle iq-rtl-generate --stdout
```

`decode`/`iq-decode` and `dot`/`iq-dot` print JSON with exactly `schema_version: 1`,
`scope: "synthetic-packed-reference"`, `result`, `device_ready: false`, and
`inference_ready: false`. Valid fixture verification prints its report.
Generation prints deterministically formatted JSON to stdout only and does
not overwrite files. Verification's default is the
[shipped fixture](<data/q2_q8_synthetic.json>) relative to this package, not the
current directory; `iq-verify` defaults to
[iq_synthetic.json](<data/iq_synthetic.json>) and `iq-rtl-verify` to
[iq_rtl_vectors.json](<data/iq_rtl_vectors.json>). `iq-generate` and
`iq-rtl-generate` require `--stdout`.

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

## IQ independent synthetic evidence and RTL vector bundle

The IQ half keeps two implementations deliberately separate, and the README
names which is which:

* **production path under test** — [iq_formats.py](<iq_formats.py>): a
  descriptor-driven decoder and the integer/FP32 dot path, transcribed from
  ggml. Its integer partial sums are the contract, and the shipped fixture is
  never generated from it.
* **independent reference** — [iq_vectors.py](<iq_vectors.py>): a second,
  straightforward transcription using exact `fractions.Fraction` arithmetic.
  It derives every expected `fp32_bits` and `integer_subdots` value in the IQ
  fixtures. It imports `iq_formats` only lazily inside its two `verify_*`
  functions.
* **analytic test expectations** — [tests/test_iq_formats.py](<tests/test_iq_formats.py>)
  builds expected values from the semantic block fields using test-local
  packers, so the format tests use neither implementation as their oracle.

Codebook bytes are **data, not derived code**: [iq_tables.py](<iq_tables.py>)
copies `kmask_iq2xs`, `ksigns_iq2xs`, `iq2xxs_grid`, `iq2s_grid` and
`iq1s_grid` verbatim from the pinned ggml/Strata headers, with a provenance
header, per-table line anchors, raw-byte SHA-256 checks and a
data-not-code/exclusion note. `iq1s_grid_gpu` (a GPU-only 4-bit +1 encoding) is
deliberately excluded. See [ATTRIBUTION.md](<ATTRIBUTION.md>) and
[licenses/](<licenses/>).

Synthetic payloads come from a seeded splitmix64 generator over raw block bytes,
never from model weights. The IQ fixture contains **16 cases: 6 decode and 10
dot**. Decode cases exhaustively sweep the grid indices (IQ2_XXS 256/256,
IQ2_S 1024/1024, IQ1_M 2048/2048), all sign-byte positions, all 4-bit and
3-bit group scales, IQ1_M's packed-FP16 scale nibbles and delta bits, and
scale corner values. Dot cases pair those weights with synthetic Q8_K
activations (`int8` extremes -128/127, zeros, equal positive/negative,
alternating, random), so every 32-group partial sum and field is exercised.
Canonical payload digest:
`b4bb7529b476e295ba92ecd1340ba35c81dc69f4f2a09a16a94e22b210cb9a4f`
(440,252 canonical bytes; the pretty-printed shipped file is 727,402 bytes).

The [RTL vector bundle](<data/iq_rtl_vectors.json>) is a smaller
integer-only contract for hardware bring-up. Its canonical schema is:

```text
top:   schema_version, fixture_id ("iq-packed-rtl-integer-v1"), synthetic,
       arithmetic_id, provenance, formats, activations, cases, payload_sha256
case:  id, weight_format, activation_format, elements, weight_hex,
       activation_hex, integer_subdot_fields, expected_integer_subdots,
       expected_fp32_bits
```

`expected_integer_subdots` is `[blocks][8 groups][terms]`, so a datapath can
compare 32 group partial sums per block with no FP32 rounding involved. The
bundle has 7 cases (3 IQ2_XXS, 2 IQ2_S, 2 IQ1_M; extremes/random, plus an
IQ2_XXS cancellation case) for 512-768 elements each; the digest is
`0cefcf13c276569790780ce7ab30026e666007269531c60b550e119f3ab0612f`.
`iq-dot`/`iq-rtl-verify` also recompute `expected_fp32_bits` from the separate
reference path.

Local source-only checks:

```sh
python3 -B -S -m unittest discover -s tools/packed_oracle/tests -v
python3 -B -S -m tools.packed_oracle verify
python3 -B -S -m tools.packed_oracle generate --stdout > /tmp/q2_q8_regenerated.json
cmp tools/packed_oracle/data/q2_q8_synthetic.json /tmp/q2_q8_regenerated.json
python3 -B -S -m tools.packed_oracle iq-verify
python3 -B -S -m tools.packed_oracle iq-rtl-verify
python3 -B -S -m tools.packed_oracle iq-generate --stdout > /tmp/iq_regenerated.json
cmp tools/packed_oracle/data/iq_synthetic.json /tmp/iq_regenerated.json
python3 -B -S -m tools.packed_oracle iq-rtl-generate --stdout > /tmp/iq_rtl_regenerated.json
cmp tools/packed_oracle/data/iq_rtl_vectors.json /tmp/iq_rtl_regenerated.json
```

The [dedicated CI](<../../.github/workflows/packed-oracle.yml>) runs these
stdlib checks under Python 3.11 and 3.14 only, with byte-exact regeneration
`cmp` for both shipped fixtures. It does not build or run existing openTPU
native, simulator, runtime, RTL, or board paths. Passing these fixtures is
**synthetic source-only arithmetic evidence**, not real-model coverage, native
CPU/AVX parity, RTL/FPGA parity, DMA/device behavior, quantizer parity,
performance, or full decode correctness/throughput. No >10 tok/s claim follows.

## Licenses and provenance

The adapted [mathematical module](<formats.py>), the IQ production path
[iq_formats.py](<iq_formats.py>), the independent IQ reference
[iq_vectors.py](<iq_vectors.py>) and the copied IQ codebook data
[iq_tables.py](<iq_tables.py>), and their modifications are MIT-licensed with
full [Strata](<licenses/STRATA-MIT.txt>), [ggml](<licenses/GGML-MIT.txt>), and
[modification](<licenses/MODIFICATIONS-MIT.txt>) notices. New CLI/package,
fixture/test scaffolding, documentation, and CI retain openTPU's Apache-2.0
license. See the [adaptation record](<ATTRIBUTION.md>) and the repository
[Apache-2.0 license](<../../LICENSE>). These software licenses do not grant
model-weight or vendor-IP rights.
