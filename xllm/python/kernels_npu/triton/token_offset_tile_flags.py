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

"""Generate complete, aligned and consecutive token-offset tile flags."""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch_npu
import triton
import triton.language as tl
from triton import knobs
from triton.runtime import driver


@triton.jit(do_not_specialize_on_alignment=["offsets", "indptr", "lengths"])
def _flags_kernel(
    offsets,
    indptr,
    lengths,
    output,
    TOTAL: tl.constexpr,
    K: tl.constexpr,
    T: tl.constexpr,
    LANES: tl.constexpr,
    GROUP: tl.constexpr,
):
    # One program owns an entire 32-byte output region, including row boundaries.
    # Different programs never update separate bytes of the same DMA block.
    base = tl.program_id(0) * 32
    lane = tl.arange(0, LANES)
    for group in range(32 // GROUP):
        flag = base + group * GROUP + tl.arange(0, GROUP)
        row = flag // K
        tile = flag % K
        begin = tl.load(indptr + row, flag < TOTAL, 0).to(tl.int64)
        length = tl.load(lengths + row, flag < TOTAL, 0)
        full = (flag < TOTAL) & (tile < length // T)
        address = begin + tl.where(full, tile, 0) * T
        first = tl.load(offsets + address, full, 0)
        data = tl.load(offsets + address[:, None] + lane[None, :], full[:, None] & (lane[None, :] < T), 0)
        # Nonnegative INT32 offsets make data-first fit signed INT32 exactly.
        equal = (lane[None, :] >= T) | ((data - first[:, None]) == lane[None, :])
        consecutive = tl.min(equal.to(tl.int32), axis=1) != 0
        result = full & (first % T == 0) & consecutive
        tl.store(output + flag, result.to(tl.uint8), flag < TOTAL)


def _validate(inputs: Sequence[torch.Tensor], attributes: dict[str, int]) -> tuple[int, int, int]:
    if not isinstance(inputs, Sequence) or len(inputs) != 4:
        raise ValueError("expected offsets, indptr, lengths and tileptr")
    if not isinstance(attributes, dict) or set(attributes) != {"tile_size", "max_tiles_per_seq"}:
        raise ValueError("tile_size and max_tiles_per_seq are required")
    tile, capacity = attributes["tile_size"], attributes["max_tiles_per_seq"]
    if type(tile) is not int or not 1 <= tile <= 4096:
        raise ValueError("tile_size must be an integer in [1,4096]")
    if type(capacity) is not int or capacity < 1:
        raise ValueError("max_tiles_per_seq must be a positive integer")
    for tensor in inputs:
        if not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.int32:
            raise TypeError("inputs must be int32 tensors")
        if tensor.device.type != "npu" or tensor.device != inputs[0].device:
            raise ValueError("inputs must be on the same NPU")
        if tensor.ndim != 1 or not tensor.is_contiguous() or tensor.numel() > 2147483647:
            raise ValueError("inputs must be contiguous int32-indexable vectors")
    offsets, indptr, lengths, tileptr = inputs
    rows = lengths.numel()
    if indptr.numel() != rows + 1 or tileptr.numel() != rows + 1 or rows * capacity > 2147483647:
        raise ValueError("metadata lengths or output size are invalid")
    # One-time boundary validation is excluded from prepared.run timing.
    values, boundaries, sizes, tiles = [tensor.cpu().tolist() for tensor in inputs]
    if boundaries[0] != 0 or boundaries[-1] != offsets.numel():
        raise ValueError("indptr must span offsets")
    if any(a > b for a, b in zip(boundaries, boundaries[1:])) or any(v < 0 for v in values):
        raise ValueError("indptr must be nondecreasing and offsets nonnegative")
    if any(
        v < 0 or v > b - a or (v + tile - 1) // tile > capacity for v, a, b in zip(sizes, boundaries, boundaries[1:])
    ):
        raise ValueError("lengths exceed their segment or tile capacity")
    if tiles != [r * capacity for r in range(rows + 1)]:
        raise ValueError("tileptr must contain dense fixed-capacity offsets")
    return rows, tile, capacity


class TokenOffsetTileFlags:
    """Prepared fixed-shape call; read current tensor contents on the current stream."""

    def __init__(self, inputs: Sequence[torch.Tensor], attributes: dict[str, int]) -> None:
        rows, self._tile, self._capacity = _validate(inputs, attributes)
        self._inputs = tuple(inputs)
        self._size = rows * self._capacity
        self._lanes = triton.next_power_of_2(self._tile)
        self._group = min(32, 4096 // self._lanes)
        self._grid = (triton.cdiv(self._size, 32), 1, 1)
        self._constants = (self._size, self._capacity, self._tile, self._lanes, self._group)
        if self._size and inputs[0].numel():
            output = torch.empty((self._size,), dtype=torch.uint8, device=inputs[0].device)
            self._kernel = _flags_kernel.warmup(*inputs[:3], output, *self._constants, grid=self._grid)

    def run(self) -> torch.Tensor:
        """Allocate independent output and evaluate the current inputs."""
        offsets, indptr, lengths, _ = self._inputs
        if offsets.numel() == 0:
            return torch.zeros((self._size,), dtype=torch.uint8, device=offsets.device)
        output = torch.empty((self._size,), dtype=torch.uint8, device=offsets.device)
        if self._size:
            stream = driver.active.get_current_stream(offsets.device.index)
            kernel = self._kernel
            enter_hook = knobs.runtime.launch_enter_hook
            exit_hook = knobs.runtime.launch_exit_hook
            metadata = None
            if enter_hook is not None:
                metadata = kernel.launch_metadata(
                    self._grid, stream, offsets, indptr, lengths, output, *self._constants
                )
            kernel.run.launch(
                *self._grid,
                stream,
                kernel.function,
                kernel.packed_metadata,
                metadata,
                enter_hook,
                exit_hook,
                offsets,
                indptr,
                lengths,
                output,
                *self._constants,
            )
        return output


def prepare(inputs: Sequence[torch.Tensor], attributes: dict[str, int]) -> TokenOffsetTileFlags:
    """Validate once and prepare a fixed-shape tile flag operation."""
    return TokenOffsetTileFlags(inputs, attributes)


def run(inputs: Sequence[torch.Tensor], attributes: dict[str, int]) -> torch.Tensor:
    """Validate inputs and execute one operation."""
    return prepare(inputs, attributes).run()
