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
- [iq_tables.py](<iq_tables.py>) copies **data, not executable code**: the
  `kmask_iq2xs`, `ksigns_iq2xs`, `iq2xxs_grid`, `iq2s_grid` and `iq1s_grid`
  codebook tables, byte for byte, from the pinned ggml/Strata headers. It
  carries the MIT SPDX, the ggml and Strata copyright lines, a provenance
  header with per-table line anchors and raw-byte SHA-256 values, and the
  modification notice. The tables are not relicensed by openTPU's Apache
  license.
- [iq_formats.py](<iq_formats.py>) adapts ggml's IQ2_XXS/IQ2_S/IQ1_M
  dequantization and generic CPU dot mathematics into exact integer/dyadic
  binary32 Python, and [iq_vectors.py](<iq_vectors.py>) is a second independent
  transcription of the same block layouts and integer group-dot kernels used as
  the golden reference. Both are **MIT** under the same notices. The IQ
  fixtures, CLI, tests, RTL vector bundle, documentation and CI are new
  **Apache-2.0** work.
- New package/CLI scaffolding, independent synthetic fixture/test scaffolding,
  documentation, and CI use **Apache-2.0**, matching the repository's
  [license](<../../LICENSE>). The module's separate MIT scope remains intact.
- No upstream native source, quantizer, model weight, vendor IP, or bitstream
  is copied into this package. The only copied upstream artifacts are the
  codebook tables described above (and the two verbatim MIT notice files).
  Software notices do not grant rights to model artifacts or vendor
  components.

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

The IQ formats, IQ codebook data and IQ dot kernels were inspected at the same
pinned Strata revision in the vendored ggml tree:

| Source anchor | Adapted contract |
| --- | --- |
| [strata-quant-common.h:364–369](https://github.com/lvyufeng/Strata/blob/3142b66b96e9c0e775e30f620b11b191a1427c2a/engine/include/strata/quant/strata-quant-common.h#L364-L369) | `block_q8_K` activation layout = 292 bytes; `bsums` not consumed |
| [strata-quant-common.h:371–376](https://github.com/lvyufeng/Strata/blob/3142b66b96e9c0e775e30f620b11b191a1427c2a/engine/include/strata/quant/strata-quant-common.h#L371-L376) | `block_iq2_xxs` = 66 bytes, FP16 scale + 64 packed bytes |
| [strata-quant-common.h:384–393](https://github.com/lvyufeng/Strata/blob/3142b66b96e9c0e775e30f620b11b191a1427c2a/engine/include/strata/quant/strata-quant-common.h#L384-L393) | `block_iq2_s` = 82 bytes, indices/signs/qh/scales |
| [strata-quant-common.h:422–434](https://github.com/lvyufeng/Strata/blob/3142b66b96e9c0e775e30f620b11b191a1427c2a/engine/include/strata/quant/strata-quant-common.h#L422-L434) | `block_iq1_m` = 56 bytes, no `d` field; `iq1m_scale_t` FP16 union |
| [strata-quant-common.h:499–513](https://github.com/lvyufeng/Strata/blob/3142b66b96e9c0e775e30f620b11b191a1427c2a/engine/include/strata/quant/strata-quant-common.h#L499-L513) | `kmask_iq2xs` / `ksigns_iq2xs` byte tables |
| [strata-quant-common.h:552–617](https://github.com/lvyufeng/Strata/blob/3142b66b96e9c0e775e30f620b11b191a1427c2a/engine/include/strata/quant/strata-quant-common.h#L552-L617) | `iq2xxs_grid` 256 x uint64 codebook |
| [strata-quant-common.h:752–1009](https://github.com/lvyufeng/Strata/blob/3142b66b96e9c0e775e30f620b11b191a1427c2a/engine/include/strata/quant/strata-quant-common.h#L752-L1009) | `iq2s_grid` 1024 x uint64 codebook |
| [strata-quant-common.h:1128–1130](https://github.com/lvyufeng/Strata/blob/3142b66b96e9c0e775e30f620b11b191a1427c2a/engine/include/strata/quant/strata-quant-common.h#L1128-L1130) | `NGRID_IQ1S = 2048`, `IQ1S_DELTA`, `IQ1M_DELTA` |
| [strata-quant-common.h:1133–1646](https://github.com/lvyufeng/Strata/blob/3142b66b96e9c0e775e30f620b11b191a1427c2a/engine/include/strata/quant/strata-quant-common.h#L1133-L1646) | `iq1s_grid` 2048 x uint64 codebook |
| [ggml-quants.c:2488–2512](https://github.com/lvyufeng/Strata/blob/3142b66b96e9c0e775e30f620b11b191a1427c2a/engine/third_party/strata-llama/ggml/src/ggml-quants.c#L2488-L2512) | `dequantize_row_iq2_xxs` scale, grid and `ksigns_iq2xs` sign selection |
| [ggml-quants.c:2543–2577](https://github.com/lvyufeng/Strata/blob/3142b66b96e9c0e775e30f620b11b191a1427c2a/engine/third_party/strata-llama/ggml/src/ggml-quants.c#L2543-L2577) | `dequantize_row_iq2_s` two 4-bit scales, 10-bit indices, raw sign byte |
| [ggml-quants.c:2675–2714](https://github.com/lvyufeng/Strata/blob/3142b66b96e9c0e775e30f620b11b191a1427c2a/engine/third_party/strata-llama/ggml/src/ggml-quants.c#L2675-L2714) | `dequantize_row_iq1_m` packed FP16 scale, 11-bit indices, -0.125 delta |
| [ggml-cpu/quants.c:906–946](https://github.com/lvyufeng/Strata/blob/3142b66b96e9c0e775e30f620b11b191a1427c2a/engine/third_party/strata-llama/ggml/src/ggml-cpu/quants.c#L906-L946) | `ggml_vec_dot_iq2_xxs_q8_K_generic` integer group semantics |
| [ggml-cpu/quants.c:998–1048](https://github.com/lvyufeng/Strata/blob/3142b66b96e9c0e775e30f620b11b191a1427c2a/engine/third_party/strata-llama/ggml/src/ggml-cpu/quants.c#L998-L1048) | `ggml_vec_dot_iq2_s_q8_K_generic` integer group semantics |
| [ggml-cpu/quants.c:1193–1252](https://github.com/lvyufeng/Strata/blob/3142b66b96e9c0e775e30f620b11b191a1427c2a/engine/third_party/strata-llama/ggml/src/ggml-cpu/quants.c#L1193-L1252) | `ggml_vec_dot_iq1_m_q8_K_generic` integer group semantics |

The GPU-only `iq1s_grid_gpu` table (a +1-biased 4-bit encoding used by the
Metal/CUDA IQ1 paths) is deliberately **not** copied; it is not the scalar
reference. Raw-byte SHA-256 values for every copied table are fixed in
[iq_tables.py](<iq_tables.py>) and re-checked by its import-time validation.
`iq_formats.py` and `iq_vectors.py` were transcribed from the pinned scalar
sources; the fp16-to-fp32 widening, binary32 rounding and dot ordering are
documented as openTPU's own exact-arithmetic specification.

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
7. Copy the pinned ggml IQ codebook tables verbatim into `iq_tables.py` as
   data (not code), with provenance header, line anchors, raw-byte SHA-256
   checks and an explicit exclusion of the GPU-only `iq1s_grid_gpu` encoding.
8. Add `iq_formats.py` (descriptor-driven production path) and
   `iq_vectors.py` (independent exact `Fraction` reference) for IQ2_XXS,
   IQ2_S and IQ1_M decode and integer/FP32 group dots against Q8_K, keeping
   the two transcriptions separate and recording every ggml anchor above.
9. Add IQ synthetic fixture and RTL integer vector bundle generators,
   verifiers, CLI subcommands and tests. The RTL bundle stops at exact
   per-32-group integer partial sums; the FP32 combination is documented
   separately. This remains source-only arithmetic evidence, not silicon or
   native CPU parity.

This adaptation is a mathematical reference preparation artifact. It is not
real-model coverage, Strata native/AVX parity, openTPU RTL/FPGA parity, or
whole-decode correctness/throughput evidence.
