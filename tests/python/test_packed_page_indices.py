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

"""NPU regression tests for ragged prefix translation and mutable inputs."""

from __future__ import annotations

import pytest
import torch
import torch_npu

from xllm.python.kernels_npu.triton.packed_page_indices import prepare


def _oracle(x: torch.Tensor, mapping: torch.Tensor, cu: torch.Tensor) -> torch.Tensor:
    rows = mapping.cpu().reshape(-1, mapping.shape[-1]).tolist()
    logical, boundaries = x.cpu().tolist(), cu.cpu().tolist()
    result = [
        [
            row[value] if value >= 0 else -1
            for r, request in enumerate(logical)
            for value in request[: boundaries[r + 1] - boundaries[r]]
        ]
        for row in rows
    ]
    shape = (boundaries[-1],) if mapping.ndim == 1 else (len(rows), boundaries[-1])
    return torch.tensor(result, dtype=torch.int32).reshape(shape)


@pytest.mark.parametrize("large", [False, True, "native", "mapped_budget"])
def test_updates_rebinding_and_streams(large: bool | str) -> None:
    device = torch.device("npu:0")
    r, c, layers, blocks = (64, 256, 32, 4096) if large else (3, 9, 2, 17)
    if large == "native":
        r, c, layers, blocks = 3, 9, 2, 32769
    if large == "mapped_budget":
        r, c, layers, blocks = 8, 256, 2, 8192
    generator = torch.Generator().manual_seed(117)
    x = torch.randint(-3, blocks, (r, c), dtype=torch.int32, generator=generator).to(device)
    mapping = torch.randint(0, 2147483647, (layers, blocks), dtype=torch.int32, generator=generator).to(device)
    counts = [c // 2] * r
    cu = torch.tensor([0, *torch.tensor(counts).cumsum(0).tolist()], dtype=torch.int32, device=device)
    prepared = prepare([x, mapping, cu], {})
    expected = _oracle(x, mapping, cu)
    first = prepared.run()
    assert torch.equal(first.cpu(), expected)
    old = [tensor.clone() for tensor in (x, mapping, cu)]
    replacements = [old[0].flip(-1), torch.full_like(mapping, 2147483647), cu.clone()]
    new_cu = cu.cpu().clone()
    new_cu[1] -= 1
    replacements[2] = new_cu.to(device)
    for tensor, replacement in zip((x, mapping, cu), replacements):
        storage = torch.empty(tensor.numel() + 1, dtype=torch.int32, device=device)
        view = storage[1:].reshape(tensor.shape)
        view.copy_(replacement)
        tensor.set_(view)
        assert torch.equal(prepared.run().cpu(), _oracle(x, mapping, cu))
    assert torch.equal(first.cpu(), expected)
    streams = [torch.npu.Stream(device=device) for _ in range(2)]
    torch.npu.synchronize()
    outputs = []
    for stream in streams:
        with torch.npu.stream(stream):
            outputs.extend(prepared.run() for _ in range(3))
    torch.npu.synchronize()
    expected = _oracle(x, mapping, cu)
    assert len({output.data_ptr() for output in outputs}) == 6
    assert all(torch.equal(output.cpu(), expected) for output in outputs)


@pytest.mark.parametrize("r,c,counts", [(0, 3, []), (3, 0, [0, 0, 0]), (2, 3, [0, 0]), (3, 3, [2, 0, 1])])
def test_empty_and_ignored_tails(r: int, c: int, counts: list[int]) -> None:
    x = torch.full((r, c), 999999, dtype=torch.int32, device="npu:0")
    for row, count in enumerate(counts):
        x[row, :count] = -7
    mapping = torch.tensor([2147483647], dtype=torch.int32, device="npu:0")
    cu = torch.tensor([0, *torch.tensor(counts).cumsum(0).tolist()], dtype=torch.int32, device="npu:0")
    result = prepare([x, mapping, cu], {}).run()
    assert torch.equal(result.cpu(), _oracle(x, mapping, cu))


def test_invalid_inputs() -> None:
    x = torch.zeros((2, 3), dtype=torch.int32, device="npu:0")
    mapping = torch.zeros((2, 7), dtype=torch.int32, device="npu:0")
    cu = torch.tensor([0, 2, 3], dtype=torch.int32, device="npu:0")
    bad_mapping = mapping.clone()
    bad_mapping[0, 0] = -1
    bad_x = x.clone()
    bad_x[0, 0] = 2147483647
    bad_cu = cu.clone()
    bad_cu[1] = -1
    cases = [
        ([x, mapping], {}),
        ([x, mapping, cu], {"unknown": 1}),
        ([x.cpu(), mapping, cu], {}),
        ([x.long(), mapping, cu], {}),
        ([torch.zeros((2, 6), dtype=torch.int32, device="npu:0")[:, ::2], mapping, cu], {}),
        ([x, bad_mapping, cu], {}),
        ([bad_x, mapping, cu], {}),
        ([x, mapping, bad_cu], {}),
    ]
    for inputs, attributes in cases:
        with pytest.raises((TypeError, ValueError)):
            prepare(inputs, attributes)
