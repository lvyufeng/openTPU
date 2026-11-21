<!-- SPDX-License-Identifier: Apache-2.0 -->

# Source attribution and adaptation record

## License scopes

- [formats.py](<formats.py>) adapts Strata's supplied-packed block/dequant/dot
  mathematics and carries **MIT** SPDX and original notices. Its new strict
  validation and exact integer/dyadic binary32 implementation are also MIT;
  the complete [modification notice](<licenses/MODIFICATIONS-MIT.txt>) applies.
- The complete [Strata MIT notice](<licenses/STRATA-MIT.txt>) and
  [ggml MIT notice](<licenses/GGML-MIT.txt>) must accompany copies or
  substantial portions of this mathematical module. These original MIT
  notices are not replaced by openTPU's Apache license.
- New package/CLI scaffolding, independent synthetic fixture/test scaffolding,
  documentation, and CI use **Apache-2.0**, matching the repository's
  [license](<../../LICENSE>). The module's separate MIT scope remains intact.
- No upstream native source, IQ codebook, quantizer, model weight, vendor IP,
  or bitstream is copied into this package. Software notices do not grant
  rights to model artifacts or vendor components.

## Pinned sources inspected

Reference repository: [lvyufeng/Strata](https://github.com/lvyufeng/Strata),
revision `3142b66b96e9c0e775e30f620b11b191a1427c2a`, mirrored from upstream
[Niko1221/Strata](https://github.com/Niko1221/Strata).
The following line ranges were read at that exact revision:

| Source anchor | Adapted contract |
| --- | --- |
| [dequant.cpp:25–43](https://github.com/lvyufeng/Strata/blob/3142b66b96e9c0e775e30f620b11b191a1427c2a/engine/src/quant/dequant.cpp#L25-L43) | Q2_0 low-to-high two-bit lanes, code-minus-one values, FP16 scale |
| [dequant.cpp:139–153](https://github.com/lvyufeng/Strata/blob/3142b66b96e9c0e775e30f620b11b191a1427c2a/engine/src/quant/dequant.cpp#L139-L153) | Q8_0 signed int8 lanes times FP16 scale |
| [dot_generic.cpp:22–63](https://github.com/lvyufeng/Strata/blob/3142b66b96e9c0e775e30f620b11b191a1427c2a/engine/src/quant/dot_generic.cpp#L22-L63) | Q2_0/Q8_0 exact 32-lane subdots, two activation scales, sequential subtotal/row accumulation |
| [dot_generic.cpp:203–226](https://github.com/lvyufeng/Strata/blob/3142b66b96e9c0e775e30f620b11b191a1427c2a/engine/src/quant/dot_generic.cpp#L203-L226) | Q8_0/Q8_0 integer dot, scale product, sequential row accumulation |
| [strata-quant-common.h:195–201](https://github.com/lvyufeng/Strata/blob/3142b66b96e9c0e775e30f620b11b191a1427c2a/engine/include/strata/quant/strata-quant-common.h#L195-L201) | Q2_0: 64 elements, FP16 + 16 packed bytes = 18 bytes |
| [strata-quant-common.h:266–272](https://github.com/lvyufeng/Strata/blob/3142b66b96e9c0e775e30f620b11b191a1427c2a/engine/include/strata/quant/strata-quant-common.h#L266-L272) | Q8_0: 32 elements, FP16 + 32 int8 bytes = 34 bytes |

Type IDs were additionally checked in
[types.hpp:13–16](https://github.com/lvyufeng/Strata/blob/3142b66b96e9c0e775e30f620b11b191a1427c2a/engine/include/strata/quant/types.hpp#L13-L16),
and activation-format metadata in
[types.cpp:19–40](https://github.com/lvyufeng/Strata/blob/3142b66b96e9c0e775e30f620b11b191a1427c2a/engine/src/quant/types.cpp#L19-L40).
The six fixture provenance anchors above are fixed in that order; these
additional static inspections do not alter the fixture anchor list.

Strata's dequant/dot file headers explicitly record derivation from
[ggml/llama.cpp](https://github.com/ggml-org/llama.cpp/tree/3cf03257f219afbe7334045ff7c6a06ac68c627d)
revision `3cf03257f219afbe7334045ff7c6a06ac68c627d`, ggml authors,
Copyright (c) 2023–2026, MIT. This is **recorded upstream provenance**;
this implementation inspected the locked Strata files, not a separately
acquired or built checkout of that ggml-origin revision.

Host repository base: [lvyufeng/openTPU](https://github.com/lvyufeng/openTPU),
revision `d84db55712aba0d40811d7762cb052f05d55b7e5`.
The tool is standalone/opt-in at that base. Existing runtime, MXU, VPU,
nibble-packed layouts, ISA, and device paths are unchanged. There is no
`import opentpu` or runtime/native dependency.

## Exact notice copies

| Included file | Locked source path | SHA-256 |
| --- | --- | --- |
| [STRATA-MIT.txt](<licenses/STRATA-MIT.txt>) | `LICENSE` at the pinned Strata revision | `949b9b0aada7476006469e58ee4ffb0e0f68c2d9137e90a356757b61a09d37f0` |
| [GGML-MIT.txt](<licenses/GGML-MIT.txt>) | `engine/third_party/strata-ggml/LICENSE` at the pinned Strata revision | `94f29bbed6a22c35b992c5c6ebf0e7c92f13b836b90f36f461c9cf2f0f1d010d` |

The copies preserve the full copyright, permission, inclusion obligation,
and warranty/liability disclaimer verbatim, including final newline.

## Modification record

Date: **2026-11-22 (Asia/Shanghai)**. Author: lvyufeng and the openTPU
contributors.

1. Adapt only the Q2_0/Q8_0 byte layouts and generic scalar equations from
   the pinned references into a standalone Python module. No upstream
   imports, pointer aliasing, C++ assertions, or architecture dispatch.
2. Replace implicit C++ float/compiler arithmetic with explicit exact dyadic
   integer multiplication/addition and per-operation binary32 RNE encoding.
   Preserve signed zero and gradual underflow; do not fuse operations.
3. Replace asserted dimensions with strict bounded whole-block validation,
   exact bytes/length checks, finite FP16 checks, and stable `FormatError`
   failures. Unsupported formats, including IQ2_XS and IQ3_S, are explicit.
4. Return exact binary32 bit strings and exact integer subdots; expose the
   fixed arithmetic identifier `strata-generic-f32-rne-separate-v1`.
5. Add stdlib JSON CLI/schema/provenance gates and independent synthetic
   Fraction/integer golden fixtures. The generator does not use DUT outputs
   as expected values; integrity digests are not authenticity claims.
6. Add source-only Python 3.11/3.14 test/fixture/regeneration CI. Do not modify
   runtime, MXU/VPU/RTL or nibble-packed formats. No native/FPGA/DMA/model/
   quantizer/performance validation is claimed.

This adaptation is a mathematical reference preparation artifact. It is not
real-model coverage, Strata native/AVX parity, openTPU RTL/FPGA parity, or
whole-decode correctness/throughput evidence.
