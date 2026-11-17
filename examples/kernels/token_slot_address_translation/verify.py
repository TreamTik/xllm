# Copyright 2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     https://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Reproduce the task's exact integer, lifetime and paired wall-time checks."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import platform
import statistics
import subprocess
import sys
import time
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch_npu

sys.path.insert(0, str(Path(__file__).resolve().parent / "fixtures"))
import generate_data
import golden
from solution import prepare

_I32 = 2147483647
_SPEC = json.loads(Path(__file__).with_name("fixtures").joinpath("case.json").read_text())


def _tensor(values: list, offset: bool = False) -> torch.Tensor:
    cpu = torch.tensor(values, dtype=torch.int32)
    if not offset:
        return cpu.to("npu:0")
    base = torch.empty((cpu.numel() + 1,), dtype=torch.int32, device="npu:0")
    view = base[1:].view(cpu.shape)
    view.copy_(cpu)
    return view


def _inputs(case: dict) -> tuple[list[torch.Tensor], dict]:
    values, attrs = generate_data.generate(case)
    return [_tensor(v, case.get("offset", False)) for v in values], attrs


def _baseline(inputs: list[torch.Tensor], attrs: dict) -> Callable[[], torch.Tensor]:
    m = inputs[1]
    return golden.reference("baseline").prepare(
        inputs, attrs, {"layers": 1 if m.ndim == 1 else m.shape[0], "blocks": m.shape[-1], "single": m.ndim == 1}
    )


def _oracle(inputs: list[torch.Tensor], attrs: dict) -> torch.Tensor:
    values = [x.cpu().tolist() for x in inputs]
    result = golden.calculate(values, attrs, inputs[1].ndim == 1)
    shape = (inputs[0].numel(),) if inputs[1].ndim == 1 else (inputs[1].shape[0], inputs[0].numel())
    return torch.tensor(result, dtype=torch.int32).reshape(shape)


def _check(identifier: str, inputs: list[torch.Tensor], attrs: dict, call: Callable | None = None) -> dict:
    originals = [x.cpu().clone() for x in inputs]
    expected = _oracle(inputs, attrs)
    candidate = (prepare(inputs, attrs).run if call is None else call)()
    reference = _baseline(inputs, attrs)()
    torch.npu.synchronize()
    actual = candidate.cpu()
    checks = {
        "candidate_mismatches": int(torch.count_nonzero(actual != expected)),
        "baseline_mismatches": int(torch.count_nonzero(reference.cpu() != expected)),
        "shape": candidate.shape == expected.shape,
        "dtype": candidate.dtype == torch.int32,
        "npu": candidate.device == inputs[0].device,
        "contiguous": candidate.is_contiguous(),
        "inputs_unchanged": all(torch.equal(x.cpu(), before) for x, before in zip(inputs, originals)),
        "independent_storage": candidate.numel() == 0 or all(candidate.data_ptr() != x.data_ptr() for x in inputs),
    }
    assert checks["candidate_mismatches"] == checks["baseline_mismatches"] == 0, (identifier, checks)
    assert all(v for k, v in checks.items() if not k.endswith("mismatches")), (identifier, checks)
    record = {
        "id": identifier,
        "status": "PASS",
        "checks": checks,
        "shape": list(candidate.shape),
        "input_shapes": [list(x.shape) for x in inputs],
        "attributes": attrs,
        "output_sha256": hashlib.sha256(json.dumps(actual.tolist(), separators=(",", ":")).encode()).hexdigest(),
        "output_raw_sha256": hashlib.sha256(actual.numpy().tobytes()).hexdigest(),
    }
    print(json.dumps(record), flush=True)
    return record


def _negative_checks() -> list[dict]:
    case = next(c for c in _SPEC["input_cases"] if c["id"] == "slot_perf_small")
    records = []
    for identifier in [x["id"] for x in _SPEC["expected_checks"]["result.json"]["negative"]]:
        inputs, attrs = _inputs(case)
        if identifier == "input_count":
            inputs.pop()
        elif identifier == "unknown_attribute":
            attrs["unknown"] = 1
        elif identifier == "cpu_input":
            inputs[0] = inputs[0].cpu()
        elif identifier == "dtype":
            inputs[0] = inputs[0].to(torch.int64)
        elif identifier == "noncontiguous":
            inputs[0] = torch.zeros((inputs[0].numel() * 2,), device="npu:0", dtype=torch.int32)[::2]
        elif identifier == "negative_mapping":
            inputs[1].view(-1)[0] = -1
        elif identifier == "logical_address_out_of_range":
            inputs[0][0] = _I32
        elif identifier == "zero_block_size":
            attrs["block_size"] = 0
        elif identifier == "bool_block_size":
            attrs["block_size"] = True
        elif identifier == "output_int32_overflow":
            inputs[1].view(-1)[0] = _I32
        try:
            prepare(inputs, attrs)
        except (TypeError, ValueError) as error:
            records.append({"id": identifier, "status": "PASS", "exception": type(error).__name__})
        else:
            raise AssertionError(identifier + " accepted illegal input")
    return records


def _mutable_checks(case: dict) -> tuple[list[dict], list[dict]]:
    inputs, attrs = _inputs(case)
    call = prepare(inputs, attrs).run
    updates, rebindings = [], []
    inputs[0][0] = attrs["block_size"] + 1
    updates.append(_check("logical_values_changed", inputs, attrs, call))
    inputs[1].copy_(inputs[1].flip([-1]))
    updates.append(_check("mapping_values_changed", inputs, attrs, call))
    if case["id"] == "slot_perf_large":
        inputs[1].fill_((_I32 + 1) // attrs["block_size"] - 1)
        inputs[0][-1] = inputs[1].shape[-1] * attrs["block_size"] - 1
        updates.append(_check("large_path_int32_output_boundary", inputs, attrs, call))
    for index, name in [(1, "mapping_set_storage"), (0, "logical_set_storage")]:
        value = inputs[index]
        old = value.data_ptr()
        replacement = torch.empty((value.numel() + 1,), dtype=value.dtype, device=value.device)
        replacement[1:].view(value.shape).copy_(value)
        shape, stride = value.shape, value.stride()
        value.set_(replacement.untyped_storage(), 1, shape, stride)
        record = _check(name, inputs, attrs, call)
        record.update(
            {
                "storage_offset": value.storage_offset(),
                "pointer_changed": old != value.data_ptr(),
                "old_alignment_mod16": old % 16,
                "new_alignment_mod16": value.data_ptr() % 16,
            }
        )
        assert record["pointer_changed"] and record["storage_offset"] == 1
        rebindings.append(record)
    return updates, rebindings


def _boundary_checks() -> dict:
    records: dict[str, Any] = {"large_nonpower_boundaries": [], "power2_logical_i32_boundaries": []}
    for size in (3, 7, 31):
        x = torch.arange(65536, dtype=torch.int32).remainder(4096 * size)
        x[0] = -2147483648
        m = torch.full((32, 4096), (_I32 + 1) // size - 1, dtype=torch.int32)
        inputs = [x.to("npu:0"), m.to("npu:0")]
        records["large_nonpower_boundaries"].append(
            _check(f"large_nonpower_{size}_near_i32", inputs, {"block_size": size})
        )
    for size in (1073741823, 1073741824):
        pattern = [-2147483648, -1, 0, size - 1, size, 2 * size - 1, _I32]
        for large in (False, True):
            values = (pattern * 147)[:1024] if large else pattern
            m = [[0, 1, 0]] * 32 if large else [0, 1, 0]
            kind = "power2" if size == 1073741824 else "nonpower"
            name = f"{'large' if large else 'small'}_logical_i32_{kind}"
            record = _check(name, [_tensor(values), _tensor(m)], {"block_size": size})
            if kind == "power2":
                records["power2_logical_i32_boundaries"].append(record)
            else:
                records["large_logical_i32_boundary" if large else "small_logical_i32_boundary"] = record
    # General neighboring geometries exercise tiny, unaligned and large-table paths.
    records["additional_geometry"] = []
    for count, layers, blocks, size in [
        (1, 3, 3, 7),
        (7, 5, 17, 17),
        (31, 4, 11, 96),
        (33, 3, 8193, 128),
        (2049, 3, 33, 31),
    ]:
        case = {
            "id": f"geometry_{count}_{layers}_{blocks}",
            "tokens": count,
            "layers": layers,
            "blocks": blocks,
            "block_size": size,
        }
        inputs, attrs = _inputs(case)
        records["additional_geometry"].append(_check(case["id"], inputs, attrs))
    return records


def _peak(call: Callable[[], torch.Tensor]) -> dict:
    warm = call()
    torch.npu.synchronize()
    del warm
    gc.collect()
    torch.npu.synchronize()
    before = torch.npu.memory_allocated()
    torch.npu.reset_peak_memory_stats()
    output = call()
    torch.npu.synchronize()
    peak = torch.npu.max_memory_allocated() - before
    output_bytes = output.numel() * output.element_size()
    del output
    for _ in range(20):
        output = call()
        del output
    torch.npu.synchronize()
    gc.collect()
    after = torch.npu.memory_allocated()
    assert before == after, (before, after)
    return {
        "active_before_bytes": before,
        "active_after_bytes": after,
        "peak_increment_bytes": peak,
        "output_bytes": output_bytes,
        "repeat20_released": "PASS",
    }


def _engineering(case: dict) -> dict:
    inputs, attrs = _inputs(case)
    prepared = prepare(inputs, attrs)
    expected = _oracle(inputs, attrs)
    warm = prepared.run()
    torch.npu.synchronize()
    del warm
    streams = [torch.npu.Stream() for _ in range(2)]
    outputs = []
    for stream in streams:
        with torch.npu.stream(stream):
            outputs.extend(prepared.run() for _ in range(3))
    torch.npu.synchronize()
    assert len({x.data_ptr() for x in outputs}) == 6
    assert all(torch.equal(x.cpu(), expected) for x in outputs)
    del outputs
    candidate = _peak(prepared.run)
    baseline = _peak(_baseline(inputs, attrs))
    return {
        **candidate,
        "two_streams": "PASS",
        "retained_outputs_independent": "PASS",
        **{"baseline_" + k: v for k, v in baseline.items()},
        "candidate_minus_baseline_peak_bytes": candidate["peak_increment_bytes"] - baseline["peak_increment_bytes"],
        "peak_definition": "Warmed PyTorch active allocation increment for one complete prepared call; includes output and per-call temporaries; not total HBM/runtime scratch",
    }


def functional() -> dict:
    result: dict[str, Any] = {
        "cases": [],
        "negative": [],
        "dynamic_updates": {},
        "storage_rebindings": {},
        "engineering_by_case": {},
    }
    for case in _SPEC["input_cases"]:
        inputs, attrs = _inputs(case)
        record = _check(case["id"], inputs, attrs)
        digest = case.get("generalization", {}).get("expected_sha256")
        if digest is not None:
            assert record["output_sha256"] == digest, (case["id"], record["output_sha256"], digest)
        result["cases"].append(record)
    result["negative"] = _negative_checks()
    result.update(_boundary_checks())
    for case in [c for c in _SPEC["input_cases"] if c.get("performance")]:
        result["dynamic_updates"][case["id"]], result["storage_rebindings"][case["id"]] = _mutable_checks(case)
        result["engineering_by_case"][case["id"]] = _engineering(case)
    result["engineering"] = result["engineering_by_case"]["slot_perf_small"]
    return result


def _timed(call: Callable[[], torch.Tensor]) -> float:
    torch.npu.synchronize()
    begin = time.perf_counter_ns()
    for _ in range(200):
        output = call()
    torch.npu.synchronize()
    elapsed = (time.perf_counter_ns() - begin) / 200000.0
    del output
    return elapsed


def _paired(case: dict) -> dict:
    inputs, attrs = _inputs(case)
    candidate = prepare(inputs, attrs).run
    baseline = _baseline(inputs, attrs)
    for call in (baseline, candidate):
        for _ in range(50):
            output = call()
        torch.npu.synchronize()
        del output
    raw: dict[str, list] = {"baseline_rounds_us": [], "candidate_rounds_us": [], "round_order": []}
    for index in range(6):
        pairs = [("baseline", baseline), ("candidate", candidate)]
        if index % 2:
            pairs.reverse()
        raw["round_order"].append([name + "_us" for name, _ in pairs])
        for name, call in pairs:
            raw[name + "_rounds_us"].append(_timed(call))
    base, cand = np.array(raw["baseline_rounds_us"]), np.array(raw["candidate_rounds_us"])
    diff = base - cand
    sample = np.random.default_rng(20261031).integers(0, 6, size=(10000, 6))
    ci = np.percentile(diff[sample].mean(axis=1), [2.5, 97.5], method="linear").tolist()
    max_us = _SPEC["acceptance"]["max_mean_us"].get(case["id"])
    gates = {
        "positive_mean_saving": bool(diff.mean() > 0),
        "positive_median_saving": bool(np.median(base) - np.median(cand) > 0),
        "ci95_lower_positive": bool(ci[0] > 0),
        "mean_within_target": max_us is None or float(cand.mean()) <= max_us,
    }
    return {
        "id": case["id"],
        **raw,
        "baseline_mean_us": float(base.mean()),
        "candidate_mean_us": float(cand.mean()),
        "baseline_median_us": float(np.median(base)),
        "candidate_median_us": float(np.median(cand)),
        "saved_mean_us": float(diff.mean()),
        "saved_mean_ci95_us": ci,
        "iterations": 200,
        "warmup": 50,
        "rounds": 6,
        "resamples": 10000,
        "seed": 20261031,
        "max_mean_us": max_us,
        "gates": gates if max_us is not None else {},
        "status": "PASS" if max_us is None or all(gates.values()) else "FAIL",
        "measurement_protocol": _SPEC["acceptance"]["measurement_protocol"],
    }


def benchmark() -> dict:
    cases = [c for c in _SPEC["input_cases"] if c.get("performance") or c.get("generalization", {}).get("measure")]
    records = [_paired(case) for case in cases]
    return {"benchmarks": records[:2], "diagnostics": records[2:]}


def _environment() -> dict:
    root = Path(__file__).resolve().parents[3]
    path = root / "xllm/python/kernels_npu/triton/token_slot_address_translation.py"
    return {
        "hostname": platform.node(),
        "python": platform.python_version(),
        "device": torch.npu.get_device_name(0),
        "versions": {
            name: importlib.metadata.version(name) for name in ("torch", "torch-npu", "triton-ascend", "numpy")
        },
        "kernel_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "source_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        "source_dirty": subprocess.check_output(["git", "status", "--short"], cwd=root, text=True),
        "hardware_scope": "Ascend910B3 development; Atlas A3 acceptance pending",
        "cann": "9.1.0",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("functional", "benchmark", "all"), default="all")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    result: dict[str, Any] = {"status": "RUNNING", "environment": _environment()}
    try:
        if args.mode in ("all", "functional"):
            result.update(functional())
            ids = _SPEC["generalization"]["case_ids"]
            general = {"cases": [c for c in result["cases"] if c["id"] in ids]}
            (args.output_dir / "generalization.json").write_text(json.dumps(general, indent=2) + "\n")
        if args.mode in ("all", "benchmark"):
            result.update(benchmark())
        result["status"] = "PASS" if all(x["status"] == "PASS" for x in result.get("benchmarks", [])) else "FAIL"
    except Exception:
        result["status"] = "ERROR"
        result["error"] = traceback.format_exc()
        raise
    finally:
        (args.output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
