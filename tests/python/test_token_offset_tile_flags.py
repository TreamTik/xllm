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

"""Tile completeness, padding, mutation and storage regression tests."""

from __future__ import annotations

import pytest
import torch
import torch_npu
import triton
import triton.language as tl

from xllm.python.kernels_npu.triton.token_offset_tile_flags import _divide_nonnegative, prepare


@pytest.mark.parametrize("tile", [1, 3, 5, 127, 128, 129, 255, 257, 4096])
def test_tile_boundaries_and_rebinding(tile: int) -> None:
    capacity = 17
    data = list(range(tile * 16)) + list(range(tile * 16))
    values = [data, [0, tile * 16, tile * 32], [tile * 16, tile * 16 - 1], [0, capacity, capacity * 2]]
    inputs = [torch.tensor(x, dtype=torch.int32, device="npu:0") for x in values]
    prepared = prepare(inputs, {"tile_size": tile, "max_tiles_per_seq": capacity})
    expected = torch.tensor([1] * 16 + [0] + [1] * 15 + [0, 0], dtype=torch.uint8)
    first = prepared.run()
    assert torch.equal(first.cpu(), expected)
    for tensor in inputs:
        storage = torch.empty(tensor.numel() + 1, dtype=torch.int32, device=tensor.device)
        view = storage[1:]
        view.copy_(tensor)
        tensor.set_(view)
    rebound = prepared.run()
    assert torch.equal(rebound.cpu(), expected)
    assert first.untyped_storage().data_ptr() != rebound.untyped_storage().data_ptr()
    inputs[2].zero_()
    assert torch.equal(prepared.run().cpu(), torch.zeros_like(expected))
    assert torch.equal(first.cpu(), expected)


def test_empty_offsets_and_int32_limit() -> None:
    inputs = [
        torch.empty(0, dtype=torch.int32, device="npu:0"),
        torch.tensor([0, 0, 0], dtype=torch.int32, device="npu:0"),
        torch.tensor([0, 0], dtype=torch.int32, device="npu:0"),
        torch.tensor([0, 2, 4], dtype=torch.int32, device="npu:0"),
    ]
    assert prepare(inputs, {"tile_size": 4, "max_tiles_per_seq": 2}).run().cpu().tolist() == [0] * 4
    values = [[2147483643, 2147483644, 2147483645, 2147483646, 2147483647, 0], [0, 6], [6], [0, 2]]
    inputs = [torch.tensor(x, dtype=torch.int32, device="npu:0") for x in values]
    assert prepare(inputs, {"tile_size": 3, "max_tiles_per_seq": 2}).run().cpu().tolist() == [1, 0]


@pytest.mark.parametrize("tile", [3, 7, 17, 31, 96, 127, 128, 129, 255, 257, 4096])
def test_high_offset_alignment(tile: int) -> None:
    start = (2147483647 - 2 * tile) // tile * tile
    data = list(range(start, start + tile)) + list(range(start + 1, start + tile + 1))
    values = [data, [0, tile, 2 * tile], [tile, tile], [0, 1, 2]]
    inputs = [torch.tensor(x, dtype=torch.int32, device="npu:0") for x in values]
    assert prepare(inputs, {"tile_size": tile, "max_tiles_per_seq": 1}).run().cpu().tolist() == [1, 0]


@triton.jit
def _division_test_kernel(inputs, output, N: tl.constexpr, DIVISOR: tl.constexpr):
    lane = tl.program_id(0) * 256 + tl.arange(0, 256)
    value = tl.load(inputs + lane, lane < N, 0)
    tl.store(output + lane, _divide_nonnegative(value, DIVISOR), lane < N)


@pytest.mark.parametrize("divisor", [1, 3, 7, 17, 31, 65, 96, 128, 255, 257, 4096, 1073741823, 1073741824, 2147483647])
def test_exact_high_integer_division(divisor: int) -> None:
    generator = torch.Generator().manual_seed(19)
    values = torch.randint(0, 2147483647, (2048,), generator=generator, dtype=torch.int32)
    values[:6] = torch.tensor([0, divisor - 1, divisor, 2147483645, 2147483646, 2147483647], dtype=torch.int32)
    inputs = values.to("npu:0")
    output = torch.empty_like(inputs)
    _division_test_kernel[(8,)](inputs, output, values.numel(), divisor)
    assert torch.equal(output.cpu(), values // divisor)
