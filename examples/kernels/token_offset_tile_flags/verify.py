# Copyright 2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Task-specified functional, paired wall-time and allocator validation."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import inspect
import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
import torch_npu
from solution import prepare


def _inputs(values: list[list[int]], offset: bool = False) -> list[torch.Tensor]:
    result = []
    for value in values:
        tensor = torch.tensor(value, dtype=torch.int32, device="npu:0")
        if offset:
            storage = torch.empty(tensor.numel() + 1, dtype=torch.int32, device=tensor.device)
            view = storage[1:]
            view.copy_(tensor)
            tensor = view
        result.append(tensor)
    return result


def _check(output: torch.Tensor, expected: list[int], inputs: list[torch.Tensor]) -> dict[str, Any]:
    golden = torch.tensor(expected, dtype=torch.uint8)
    checks = {
        "candidate_mismatches": int((output.cpu() != golden).sum()),
        "shape": list(output.shape) == [len(expected)],
        "dtype": output.dtype == torch.uint8,
        "npu": output.device == inputs[0].device,
        "contiguous": output.is_contiguous(),
        "independent_storage": not output.numel()
        or all(output.untyped_storage().data_ptr() != x.untyped_storage().data_ptr() for x in inputs),
    }
    assert checks["candidate_mismatches"] == 0 and all(v for k, v in checks.items() if k != "candidate_mismatches"), (
        checks
    )
    return checks


def _functional(cases: list[dict], generator: Any, golden: Any) -> dict[str, Any]:
    records = []
    for case in cases:
        print(json.dumps({"case_started": case["id"]}), flush=True)
        values, attrs = generator.generate(case)
        inputs = _inputs(values, case.get("offset", False))
        snapshots = [x.clone() for x in inputs]
        prepared = prepare(inputs, attrs)
        out = prepared.run()
        expected = golden.calculate(values, attrs)
        checks = _check(out, expected, inputs)
        baseline = golden.reference("baseline").prepare(inputs, attrs, {"rows": len(values[2])})
        ref = baseline()
        checks["baseline_mismatches"] = _check(ref, expected, inputs)["candidate_mismatches"]
        checks["inputs_unchanged"] = all(torch.equal(x, before) for x, before in zip(inputs, snapshots))
        assert checks["inputs_unchanged"]
        records.append(
            {
                "id": case["id"],
                "status": "PASS",
                "checks": checks,
                "input_shapes": [list(x.shape) for x in inputs],
                "shape": list(out.shape),
                "attributes": attrs,
            }
        )
    return {"cases": records}


def _negative(case: dict, generator: Any) -> list[dict]:
    values, attrs = generator.generate(case)
    records = []
    for name in [
        "input_count",
        "unknown_attribute",
        "cpu_input",
        "dtype",
        "noncontiguous",
        "zero_tile",
        "oversize_tile",
        "non_dense_output_offsets",
        "length_exceeds_segment",
    ]:
        inputs, attributes = _inputs(values), dict(attrs)
        if name == "input_count":
            inputs.pop()
        elif name == "unknown_attribute":
            attributes["unknown"] = 1
        elif name == "cpu_input":
            inputs[0] = inputs[0].cpu()
        elif name == "dtype":
            inputs[0] = inputs[0].long()
        elif name == "noncontiguous":
            inputs[0] = torch.empty(inputs[0].numel() * 2, dtype=torch.int32, device="npu:0")[::2]
        elif name == "zero_tile":
            attributes["tile_size"] = 0
        elif name == "oversize_tile":
            attributes["tile_size"] = 4097
        elif name == "non_dense_output_offsets":
            inputs[3][1] = 0
        elif name == "length_exceeds_segment":
            inputs[2][0] = 2147483647
        try:
            prepare(inputs, attributes)
        except (TypeError, ValueError) as error:
            records.append({"id": name, "status": "PASS", "exception": type(error).__name__})
        else:
            raise AssertionError(name)
    return records


def _mutable(golden: Any) -> list[dict]:
    attrs = {"tile_size": 3, "max_tiles_per_seq": 3}
    values = [list(range(18)), [0, 9, 18], [9, 8], [0, 3, 6]]
    inputs = _inputs(values)
    prepared = prepare(inputs, attrs)
    first = prepared.run()
    original = first.cpu().clone()
    records = []
    updates = [
        [[3, 4, 5, 6, 7, 9, 9, 10, 11] + list(range(9)), [0, 9, 18], [8, 9], [0, 3, 6]],
        [list(range(18)), [0, 8, 18], [7, 9], [0, 3, 6]],
    ]
    for index, current in enumerate(updates):
        replacements = _inputs(current, offset=True)
        for x, y in zip(inputs, replacements):
            if index == 0:
                x.copy_(y)
            else:
                x.set_(y)
        checks = _check(prepared.run(), golden.calculate(current, attrs), inputs)
        assert torch.equal(first.cpu(), original)
        records.append({"id": f"mutable_{index}", "status": "PASS", "checks": checks})
    return records


def _engineering(case: dict, generator: Any, golden: Any) -> dict[str, Any]:
    values, attrs = generator.generate(case)
    inputs = _inputs(values)
    expected = golden.calculate(values, attrs)
    candidate = prepare(inputs, attrs).run
    baseline = golden.reference("baseline").prepare(inputs, attrs, {"rows": len(values[2])})
    records = {}
    for name, call in [("candidate", candidate), ("baseline", baseline)]:
        streams = [torch.npu.Stream() for _ in range(2)]
        torch.npu.synchronize()
        retained = []
        for stream in streams:
            with torch.npu.stream(stream):
                retained.extend(call() for _ in range(3))
        torch.npu.synchronize()
        retained_checks = []
        for index, output in enumerate(retained):
            retained_checks.append(
                {"stream": index // 3, "output": index % 3, "checks": _check(output, expected, inputs)}
            )
        assert len({x.untyped_storage().data_ptr() for x in retained}) == 6
        del output, retained
        for _ in range(3):
            output = call()
        del output
        torch.npu.synchronize()
        before = torch.npu.memory_allocated()
        torch.npu.reset_peak_memory_stats()
        output = call()
        torch.npu.synchronize()
        peak = torch.npu.max_memory_allocated() - before
        peak_checks = _check(output, expected, inputs)
        del output
        torch.npu.synchronize()
        release_checks = []
        for index in range(20):
            output = call()
            checks = _check(output, expected, inputs)
            del output
            torch.npu.synchronize()
            active = torch.npu.memory_allocated()
            assert active == before
            release_checks.append(
                {"call": index + 1, "checks": checks, "active_after_bytes": active, "residual_bytes": active - before}
            )
        after = torch.npu.memory_allocated()
        assert before == after, (name, before, after)
        records[name] = {
            "two_streams": "PASS",
            "retained_outputs": 6,
            "repeat20_released": "PASS",
            "active_before_bytes": before,
            "active_after_bytes": after,
            "peak_increment_bytes": peak,
            "retained_output_checks": retained_checks,
            "single_call_checks": peak_checks,
            "release_checks": release_checks,
            "residual_bytes": after - before,
            "output_bytes": len(expected),
            "status": "PASS",
        }
    return {"id": case["id"], "records": records}


def _measure(call: Callable[[], torch.Tensor]) -> float:
    torch.npu.synchronize()
    start = time.perf_counter_ns()
    for _ in range(200):
        output = call()
    torch.npu.synchronize()
    elapsed = (time.perf_counter_ns() - start) / 200000
    del output
    return elapsed


def _benchmark(case: dict, generator: Any, golden: Any) -> dict[str, Any]:
    values, attrs = generator.generate(case)
    inputs = _inputs(values)
    candidate = prepare(inputs, attrs).run
    baseline = golden.reference("baseline").prepare(inputs, attrs, {"rows": len(values[2])})
    for call in [baseline, candidate]:
        for _ in range(50):
            output = call()
        del output
        torch.npu.synchronize()
    rounds = []
    for index in range(6):
        order = [("baseline_us", baseline), ("candidate_us", candidate)]
        if index % 2:
            order.reverse()
        pair = {name: _measure(call) for name, call in order}
        pair["round"] = index + 1
        pair["order"] = [name for name, _ in order]
        rounds.append(pair)
    base = np.array([x["baseline_us"] for x in rounds])
    cand = np.array([x["candidate_us"] for x in rounds])
    diff = base - cand
    indices = np.random.default_rng(20261031).integers(0, 6, size=(10000, 6))
    ci = np.quantile(diff[indices].mean(axis=1), [0.025, 0.975], method="linear").tolist()
    gate = bool(base.mean() > cand.mean() and np.median(base) > np.median(cand) and ci[0] > 0 and cand.mean() <= 85)
    required = case.get("performance", False)
    return {
        "id": case["id"],
        "baseline_mean_us": float(base.mean()),
        "candidate_mean_us": float(cand.mean()),
        "saved_mean_us": float(diff.mean()),
        "baseline_median_us": float(np.median(base)),
        "candidate_median_us": float(np.median(cand)),
        "saved_mean_ci95_us": ci,
        "rounds": rounds,
        "warmup": 50,
        "iterations": 200,
        "timing": "complete prepared.run wall time and final synchronize",
        "required": required,
        "development_gate": ("PASS" if gate else "FAIL") if required else "REPORT_ONLY",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-package", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=["functional", "benchmark", "engineering", "all"], default="all")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    sys.path.insert(0, str(args.task_package))
    import generate_data
    import golden

    cases = json.loads((args.task_package / "case.json").read_text())["input_cases"]
    performance = [c for c in cases if c.get("performance")]
    result = {
        "platform_scope": "Ascend910B3 development validation; Atlas A3 acceptance pending",
        "python": platform.python_version(),
        "device": torch.npu.get_device_name(0),
        "versions": {x: importlib.metadata.version(x) for x in ["torch", "torch-npu", "triton-ascend", "numpy"]},
        "source_sha256": hashlib.sha256(Path(inspect.getfile(prepare)).read_bytes()).hexdigest(),
    }
    if args.mode in ("all", "functional"):
        result.update(_functional(cases, generate_data, golden))
        result["negative"] = _negative(performance[0], generate_data)
        result["mutable"] = _mutable(golden)
        result["engineering"] = [_engineering(c, generate_data, golden) for c in performance]
        (args.output_dir / "functional.json").write_text(json.dumps(result, indent=2) + "\n")
    if args.mode == "engineering":
        result["engineering"] = [_engineering(c, generate_data, golden) for c in performance]
        (args.output_dir / "engineering-supplement.json").write_text(json.dumps(result, indent=2) + "\n")
    if args.mode in ("all", "benchmark"):
        measured = performance + [c for c in cases if c.get("generalization", {}).get("measure")]
        result["benchmarks"] = [_benchmark(c, generate_data, golden) for c in measured]
        (args.output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
        assert all(x["development_gate"] == "PASS" for x in result["benchmarks"] if x["required"]), result["benchmarks"]
    subset = [x for x in result.get("cases", []) if x["id"] in {c["id"] for c in cases if "generalization" in c}]
    if subset:
        (args.output_dir / "generalization.json").write_text(json.dumps({"cases": subset}, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
