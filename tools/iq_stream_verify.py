#!/usr/bin/env python3
"""Check the IQ stream RTL against a frozen, synthetic integer fixture.

Only a reviewed fixture's JSON data is read. No oracle executable is run, no
float arithmetic is used here and no real model weights are accessed. Use zero
generated cases for fixture-only validation (otherwise --cases is per format).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import copy
import subprocess
import tempfile
from pathlib import Path

import iq_multi_rtl_check as m

FORMATS = {"IQ2_XXS": 0, "IQ2_S": 1, "IQ1_M": 2}
FIXTURE_RELATIVE = Path("tools/packed_oracle/data/iq_rtl_vectors.json")
FROZEN_FILE_SHA256 = "31fb016201fc7b4bda2b941bb3ca6980fd204d3216b5b7ea77b6fd957f24e9b5"
FROZEN_PAYLOAD_SHA256 = "0cefcf13c276569790780ce7ab30026e666007269531c60b550e119f3ab0612f"


def _canonical_digest(doc):
    payload = {k: v for k, v in doc.items() if k != "payload_sha256"}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":"),
                                        ensure_ascii=True, allow_nan=False).encode("ascii")).hexdigest()


def frozen_cases(root: Path, *, expected_fixture_hashes=None):
    """Validate the pinned oracle; explicit copy hashes are for negative tests."""
    pinned = {"file_sha256": FROZEN_FILE_SHA256, "payload_sha256": FROZEN_PAYLOAD_SHA256}
    expected_hashes = pinned if expected_fixture_hashes is None else expected_fixture_hashes
    path = root / FIXTURE_RELATIVE
    blob = path.read_bytes()
    doc = json.loads(blob)
    raw_digest = hashlib.sha256(blob).hexdigest()
    if set(expected_hashes) != {"file_sha256", "payload_sha256"} or any(
            type(value) is not str or len(value) != 64 for value in expected_hashes.values()):
        raise ValueError("expected fixture hashes must be explicit file/payload SHA256 strings")
    if raw_digest != expected_hashes["file_sha256"]:
        raise ValueError("frozen fixture file SHA256 differs from the pinned read-only data")
    digest = _canonical_digest(doc)
    if (doc.get("fixture_id") != "iq-packed-rtl-integer-v1" or doc.get("schema_version") != 1 or
        not isinstance(doc.get("synthetic"), bool) or not doc["synthetic"] or
        not isinstance(doc.get("payload_sha256"), str) or doc["payload_sha256"] != digest
            or digest != expected_hashes["payload_sha256"]):
        raise ValueError("fixture identity or canonical payload digest is invalid")
    if not isinstance(doc.get("cases"), list) or not doc["cases"]:
        raise ValueError("fixture does not supply any cases")
    cases = []
    for case in doc["cases"]:
        if case["weight_format"] not in FORMATS or case["activation_format"] != "Q8_K":
            raise ValueError("fixture supplies a format outside this checked selection")
        fmt = FORMATS[case["weight_format"]]
        weights = bytes.fromhex(case["weight_hex"])
        activations = bytes.fromhex(case["activation_hex"])
        expected = case["expected_integer_subdots"]
        size = m.block_size(fmt)
        if len(weights) % size or len(activations) != 292 * (len(weights) // size) \
                or len(expected) != len(weights) // size:
            raise ValueError(f"{case['id']}: weight/activation/result block counts disagree")
        for b, groups in enumerate(expected):
            if not isinstance(groups, list) or len(groups) != 8:
                raise ValueError(f"{case['id']} block {b}: expected exactly eight groups")
            for g, terms in enumerate(groups):
                wanted = 1 if fmt == 0 else 2
                if not isinstance(terms, list) or len(terms) != wanted \
                        or any(type(value) is not int or abs(value) >= 2**31 for value in terms):
                    raise ValueError(f"{case['id']} block {b} group {g}: invalid integer terms")
            cases.append((case["id"], b, fmt, weights[b*size:(b+1)*size],
                          activations[b*292+4:b*292+260], groups))
    return cases, {"file_sha256": hashlib.sha256(blob).hexdigest(), "payload_sha256": digest}


def compare_frozen(frozen, dump):
    """Compare expected_integer_subdots independently, as signed int32 values."""
    checks, bad = 0, []
    for n, (case_id, b, fmt, _weights, _activations, expected) in enumerate(frozen):
        for g, terms in enumerate(expected):
            row = dump[-len(frozen)+n]["groups"][g]
            bsum = m._signed(row["bsum"], 32)
            aux = m._signed(row["aux"], 32)
            # IQ2_S retains its native combined bsum. The frozen oracle's first
            # term is the separately scaled first half, exactly bsum - aux.
            actual = [bsum - aux, aux] if fmt == 1 else [bsum] if fmt == 0 else [bsum, aux]
            checks += 1
            if actual != terms:
                bad.append(f"{case_id} block {b} group {g}: {actual} != {terms}")
    return checks, bad


def fixture_selftest(path: Path):
    """Test fixture integrity and golden mismatches on temporary copies only."""
    source_blob = path.read_bytes()
    with tempfile.TemporaryDirectory(prefix="iq-frozen-negative-") as tmp:
        target = Path(tmp) / "root/tools/packed_oracle/data/iq_rtl_vectors.json"
        target.parent.mkdir(parents=True)
        original = json.loads(source_blob)

        def mutate_first(doc, fmt, g=0, term=0):
            case = next(c for c in doc["cases"] if c["weight_format"] == m.FMT_NAME[fmt])
            case["expected_integer_subdots"][0][g][term] += 1

        def shorten_activation(doc):
            case = next(c for c in doc["cases"] if c["weight_format"] == "IQ1_M")
            case["activation_hex"] = case["activation_hex"][:-2]

        checks = 1
        frozen, hashes = frozen_cases(path.parent.parent.parent.parent)
        # Dump construction must remain independent from the mutated document.
        original_expected = {(row[0], row[1]): row[-1] for row in frozen}
        if len(frozen) != 15:
            raise ValueError("frozen selftest selected an unexpected block count")
        probes = [
            ("corrupt-payload-digest", lambda d: d.__setitem__("payload_sha256", "0" * 64), False, None),
            ("iq2s-first-golden-recanonicalized", lambda d: mutate_first(d, 1), True, 1),
            ("iq1m-short-activation-recanonicalized", shorten_activation, True, None),
            ("iq2s-second-golden-valid-digest", lambda d: mutate_first(d, 1, term=1), True, 1),
            ("iq1m-delta-golden-valid-digest", lambda d: mutate_first(d, 2, term=1), True, 2),
            ("iq2xxs-golden-valid-digest", lambda d: mutate_first(d, 0), True, 0),
        ]
        results = []
        for name, mutation, recanonicalized, kind in probes:
            doc = copy.deepcopy(original)
            mutation(doc)
            if recanonicalized:
                doc["payload_sha256"] = _canonical_digest(doc)
            target_blob = (json.dumps(doc, indent=2) + "\n").encode("utf-8")
            target.write_bytes(target_blob)
            copy_hashes = {"file_sha256": hashlib.sha256(target_blob).hexdigest(),
                           "payload_sha256": _canonical_digest(doc) if recanonicalized
                           else FROZEN_PAYLOAD_SHA256}
            checks += 1
            if kind is None:
                try:
                    frozen_cases(target.parent.parent.parent.parent,
                                 expected_fixture_hashes=copy_hashes)
                except ValueError:
                    rejected = True
                else:
                    rejected = False
                detail = "reader accepted corrupt fixture" if not rejected else "integrity rejected"
            else:
                selected, _ = frozen_cases(target.parent.parent.parent.parent,
                                           expected_fixture_hashes=copy_hashes)
                selected = sorted(selected, key=lambda row: (row[2], row[0], row[1]))
                dump = [{"groups": []} for _ in selected]
                for index, (_id, _b, fmt, _w, _a, expected) in enumerate(selected):
                    original_terms = original_expected[(_id, _b)]
                    for g, terms in enumerate(original_terms):
                        if fmt == 1:
                            bsum, aux = sum(terms), terms[1]
                        else:
                            bsum, aux = terms[0], terms[1] if fmt == 2 else 0
                        dump[index]["groups"].append({"bsum": bsum & 0xffffffff,
                                                       "aux": aux & 0xffffffff})
                count, bad = compare_frozen(selected, dump)
                checks += count
                rejected = len(bad) == 1
                detail = bad if rejected else f"expected one mismatch, got {len(bad)}"
            results.append({"probe": name, "rejected": rejected, "detail": detail,
                            "digest_recanonized": recanonicalized})
        checks += 1
        unchanged = source_blob == path.read_bytes() and hashes["file_sha256"] == hashlib.sha256(source_blob).hexdigest()
    ok = unchanged and all(probe["rejected"] for probe in results)
    return {"ok": ok, "checks": checks, "rejection_probes": results,
            "original_fixture_unchanged": unchanged, **hashes}, 0 if ok else 1


def simulate(verilator, vectors: Path, run_dir: Path):
    run_dir.mkdir(parents=True, exist_ok=True)
    command = [verilator, *m.VERILATOR_FLAGS, "--Mdir", str(run_dir),
               str(m.TB), str(m.STREAM_RTL), *m.CHILD_RTL.values()]
    built = subprocess.run(command, cwd=m.ROOT, text=True, capture_output=True, timeout=3600)
    if built.returncode:
        raise RuntimeError("Verilator build failed: " + (built.stderr or built.stdout)[-4000:])
    log_path = run_dir / "simulation.log"
    dump_path = run_dir / "dump.txt"
    result = subprocess.run([str(run_dir / "Vtb_iq_stream"), f"+vectors={vectors}",
                             f"+out={dump_path}", "+cases=0"], cwd=run_dir, text=True,
                            capture_output=True, timeout=3600)
    log_path.write_text(result.stdout + result.stderr, encoding="utf-8")
    summary = m.SUMMARY_LINE.search(result.stdout)
    if result.returncode or not summary:
        raise RuntimeError("Verilator run failed; inspect " + str(log_path))
    return m.parse_dump(dump_path.read_text()), [int(v) for v in summary.groups()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", type=int, default=256, metavar="N", help="per format; zero with oracle only")
    parser.add_argument("--seed", type=int, default=m.DEFAULT_SEED)
    parser.add_argument("--build", type=Path, default=Path("build/iq-stream-oracle"))
    parser.add_argument("--verilator", default="verilator")
    parser.add_argument("--oracle-root", type=Path,
                        help="reviewed checkout containing tools/packed_oracle/data/iq_rtl_vectors.json")
    parser.add_argument("--fixture-selftest", action="store_true",
                        help="test golden rejection on temporary fixture copies; no simulator")
    args = parser.parse_args()
    if args.cases < 0 or (args.cases == 0 and args.oracle_root is None):
        return m.fail("args", "--cases >= 0; zero requires an oracle fixture")
    if args.fixture_selftest:
        if args.oracle_root is None or args.cases != 0:
            return m.fail("args", "--fixture-selftest requires --oracle-root and --cases 0")
        try:
            report, status = fixture_selftest(args.oracle_root / FIXTURE_RELATIVE)
        except (OSError, ValueError, KeyError, TypeError) as exc:
            return m.fail("oracle_selftest", str(exc))
        print(json.dumps(report, indent=2))
        return status
    verilator = m.find_verilator(args.verilator)
    if verilator is None:
        return m.fail("no_verilator", "Verilator not on PATH")
    generated = m.generate_cases(args.cases, args.seed)
    frozen, oracle_hashes = [], None
    try:
        if args.oracle_root is not None:
            root = args.oracle_root.resolve()
            selected, oracle_hashes = frozen_cases(root)
            frozen = sorted(selected, key=lambda row: (row[2], row[0], row[1]))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return m.fail("oracle", str(exc), seed=args.seed)
    ordered = generated + [(fmt, weights, activation) for _id, _b, fmt, weights, activation, _e in frozen]
    run_dir = args.build if args.build.is_absolute() else m.ROOT / args.build
    run_dir.mkdir(parents=True, exist_ok=True)
    vectors = run_dir / "vectors.hex"
    m.write_vectors(vectors, ordered)
    version = subprocess.run([verilator, "--version"], text=True, capture_output=True,
                             timeout=120, check=True).stdout.strip().splitlines()[0]
    report = {
        "ok": False, "seed": args.seed, "generated_per_format": args.cases,
        "generated_blocks": len(generated), "run_blocks": len(ordered),
        "checks": 0, "mismatches": [], "expected_beats": sum(m.FMT_BEATS[f] + 8 for f, _, _ in ordered),
    }
    try:
        dump, summary = simulate(verilator, vectors, run_dir)
    except (OSError, RuntimeError, ValueError) as exc:
        return m.fail("simulation", str(exc), seed=args.seed)
    report.update({"tb_summary": {"cases": summary[0], "beats": summary[1],
                                  "wall_cycles": summary[2], "errors": summary[3]},
                   "vectors_sha256": hashlib.sha256(vectors.read_bytes()).hexdigest(),
                   "dump_sha256": hashlib.sha256((run_dir / "dump.txt").read_bytes()).hexdigest(),
                   "verilator": version,
                   "sources_sha256": {str(p.relative_to(m.ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                                      for p in [m.TB, m.STREAM_RTL, *m.CHILD_RTL.values()]}})
    if summary != [len(ordered), report["expected_beats"], summary[2], 0] or len(dump) != len(ordered):
        report["mismatches"].append("testbench did not consume the exact vector/beats without error")
    checks, bad = m.compare(ordered, dump)
    report["checks"] += checks
    report["mismatches"] += bad
    if oracle_hashes is not None:
        checks, bad = compare_frozen(frozen, dump)
        report.update({"oracle_blocks": len(frozen), "oracle_groups": len(frozen) * 8,
                       "oracle_golden_checks": checks, "oracle_mismatches": len(bad),
                       **oracle_hashes})
        report["mismatches"] += bad
    report["ok"] = not report["mismatches"]
    print(json.dumps(report, indent=2))
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
