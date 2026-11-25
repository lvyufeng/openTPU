# Native IQ block/dot primitives and shared stream

This is a **source/simulation-only** M3 slice for the Strata/openTPU roadmap.
The shared stream supports three packed block formats:

| format / selector | packed bytes | 64-bit weight beats | 256-bit activation beats | total accepts |
| --- | ---: | ---: | ---: | ---: |
| IQ2_XXS / `0` | 66 (6 bytes zero-padded) | 9 | 8 | 17 |
| IQ2_S / `1` | 82 (6 bytes zero-padded) | 11 | 8 | 19 |
| IQ1_M / `2` | 56 | 7 | 8 | 15 |

Each block has 256 weights, eight int8 activation beats and eight result groups.
No real model checkpoint, native ggml executable or device is needed for these
checks.

## Data path

* **IQ2_XXS** (`rtl/vpu/otpu_iq2xxs.sv`): a 256 x 64-bit magnitude grid and
  128-entry sign-pattern table. Four 8-bit indices then a 32-bit auxiliary word
  start at byte `2 + 8*g`; sign selection uses all seven selected auxiliary bits.
  A group is `(2*scale + 1) * sum(signed_grid * int8_activation)`.
* **IQ2_S** (`rtl/vpu/otpu_iq2s.sv`): the existing combined `grp_bsum` arithmetic
  and stream protocol remain intact; one output is added for the separately
  scaled second half, `grp_aux = ls2 * sumi2`, to compare the frozen golden
  `[ls1 * sumi1, ls2 * sumi2]` without losing the complete partial. Its 40-bit
  index is zero-extended to four 11-bit lanes at the shared interface. This
  slice runs the primitive separately, not as a wrapper around IQ2_XXS.
* **IQ1_M** (`rtl/vpu/otpu_iq1m.sv`): signed-ternary, 2048 x 64-bit grid, separate
  scaled integer `grid` and `delta` terms. Four scale pairs and scattered fp16
  bits retain the exact packed-block layout. The caller computes
  `d * (sum_grid + 0.125 * sum_delta)`, **without an extra outer 0.125**.

The ROM contents are simulation-visible literal tables. They do not assert a
specific Xilinx RAM primitive or device implementation.

## Shared stream contract

`rtl/vpu/otpu_iq_stream.sv` leaves the three child engines independently in
place and selects exactly one. In idle, `w_fmt` probes the child without
accepting a beat. A successful selection transfers one reset to that child;
other children remain selected off. Weight format must remain unchanged until
the selected child has accepted its last weight beat. The next phase is one
activation beat per group, with `grp_valid` and group index. Group 0 also emits
`blk_valid`, `blk_fmt`, `blk_d` and `blk_scales`; other groups do not.

At the common interface, all indices are four zero-extended 11-bit lanes.
`grp_terms=1` for IQ2_XXS. IQ2_S has `grp_terms=2`: `grp_bsum` is the
**combined** scaled partial and `grp_aux` is the second term `ls2 * sumi2`,
so the caller can derive the first term as `grp_bsum - grp_aux` without changing
its old complete-partial contract. IQ1_M emits `grp_aux` as its separately scaled
delta term and `grp_terms=2`. IQ2_XXS `grp_signs`
uses the ksigns lookup output. IQ1_M `grp_delta[l]` has bit 1 for a negative
delta and bit 0 for a positive one. Integer outputs are two's complement 32-bit
ports; the checker bounds them against the actual integer model.

Format selector `3` or a mid-block format change makes `w_ready` deassert and
sets a sticky, visible error until reset. Outputs during a load are asserted not
to occur. The TB checks output strobes separately from data fields and stale
results do not count as successes. All handshakes and source counters are
observed at clock sampling edges; drivers release valid on falling edges to
avoid same-edge races.

## Verification

The modules remain **not connected** to production `RTL_SOURCES` or the VPU.
Use the following commands from the openTPU worktree with Verilator 5.x and a
C++ compiler on `PATH`:

```
# Exact table/model/reset/negative-control selftest; no simulator required.
python3 tools/iq_multi_rtl_check.py --selftest --cases 256 --seed 20261124

# Three independent instance streams, backpressure and sticky-error recovery.
python3 tools/iq_multi_rtl_check.py --cases 256 --seed 20261124

# The frozen synthetic oracle, read as data, plus generated/model checks.
python3 tools/iq_stream_verify.py --cases 256 --seed 20261124 \
  --oracle-root ../opentpu-iq-oracle

# Or only the existing frozen vectors; no oracle generator is rebuilt.
python3 tools/iq_stream_verify.py --cases 0 \
  --oracle-root ../opentpu-iq-oracle

# Reject corrupt payloads, canonical-digest changes and golden mutations in
# temporary fixture copies only; never regenerate the frozen oracle.
python3 tools/iq_stream_verify.py --cases 0 --fixture-selftest \
  --oracle-root ../opentpu-iq-oracle
```

The oracle directory is a **reviewed, read-only checkout** containing
`tools/packed_oracle/data/iq_rtl_vectors.json`, from openTPU IQ oracle commit
`f35763dd19e5e863c1bbf32be7d1d24ad0643b4a`. The checker verifies fixture
identity and exact pinned raw-file/canonical-payload SHA-256 digests, full packed
block sizes and per-group integer shapes. IQ2_XXS compares its full bias sum;
IQ2_S compares the separately scaled halves `[grp_bsum - grp_aux, grp_aux]` to
the frozen golden without changing the native complete-partial output; IQ1_M
compares its independent grid and delta terms. Both halves are independently
checked against the reference model through the combined `grp_bsum` and `grp_aux`.
The checker uses exact signed integers, not float tolerances, and records raw
fixture file and payload SHA-256 separately.

The pre-existing standalone IQ2_S stream test remains independently runnable:

```
python3 tools/iq_rtl_check.py --selftest --cases 64 --seed 20261107
python3 tools/iq_rtl_check.py --cases 64 --seed 20261107
```

Verilator flags are intentionally not hardware timing/frequency arguments;
`wall_cycles` is a TB simulation-cycle count. Vector SHA-256, dump SHA-256 and
RTL SHA-256 are recorded to identify what was actually checked. No output dump
should be read beyond its exact expected block/group count.

## Pinned table provenance and limits

Raw tables are copied without value modifications from Strata revision
`3142b66b96e9c0e775e30f620b11b191a1427c2a`, with source inventory
`97affb754d298b2cce1fc7b12338797ac155aa9b` and embedded ggml revision
`3cf03257f219afbe7334045ff7c6a06ac68c627d`. See `tools/iq_tables.py` and
`THIRD_PARTY.md` for anchors, raw table digests and complete MIT-scope references.
These tables and associated notice obligations are not silently relicensed by
openTPU's Apache-2.0.

This slice contains no VPU/ISA/sequencer/compiler integration, host ABI, expert
execution branch, double buffering, DMA, board synthesis/PnR or bitstreams. It
makes no end-to-end inference, Strata expert throughput or `>10 tok/s` claim.
Arithmetic models are equivalence checks for the supplied format data paths,
not independent validation by a native ggml implementation.

## Optional open-source generic mapping estimate

With a user-local open-source Yosys:

```
yosys -p 'read_verilog -sv rtl/vpu/otpu_iq2xxs.sv; synth_xilinx -family xc7 -top otpu_iq2xxs; stat'
yosys -p 'read_verilog -sv rtl/vpu/otpu_iq1m.sv; synth_xilinx -family xc7 -top otpu_iq1m; stat'
yosys -p 'read_verilog -sv rtl/vpu/otpu_iq2xxs.sv rtl/vpu/otpu_iq2s.sv rtl/vpu/otpu_iq1m.sv rtl/vpu/otpu_iq_stream.sv; synth_xilinx -family xc7 -top otpu_iq_stream; stat'
```

Any resulting `stat` values are generic, unplaced and unconstrained xc7 mapping
estimates. They are **not** board area, Fmax, power, timing signoff, Vivado
synthesis or programming results. This slice uses no board access or vendor
flow.
