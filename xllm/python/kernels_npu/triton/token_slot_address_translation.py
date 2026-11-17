# Copyright 2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/xLLM-AI/xllm/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Translate logical token slots through per-layer physical page tables."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch_npu
import triton
import triton.language as tl
from triton import knobs
from triton.runtime import driver


@triton.jit(do_not_specialize_on_alignment=["slots_ptr", "mapping_ptr"])
def _translate_table_kernel(
    slots_ptr,
    mapping_ptr,
    output_ptr,
    N: tl.constexpr,
    B: tl.constexpr,
    L: tl.constexpr,
    S: tl.constexpr,
    TABLE: tl.constexpr,
    TILE: tl.constexpr,
):
    # Own an aligned flat output interval. A tile intersects at most two layers.
    first = tl.program_id(0).to(tl.int64) * TILE
    lane = tl.arange(0, TILE)
    out = first + lane
    layer = first // N
    pos = first % N + lane
    left = tl.load(slots_ptr + pos, (out < L * N) & (pos < N), 0)
    right = tl.load(slots_ptr + pos - N, (out < L * N) & (pos >= N), 0)
    slot = tl.where(pos < N, left, right)
    safe = tl.maximum(slot, 0)
    if S & (S - 1) == 0:
        page = safe >> (S.bit_length() - 1)
    else:
        # ceil(2^32/S) overestimates the quotient by at most one for x<2^31.
        # The 64-bit product is below 2^63; correction makes divmod exact.
        MAGIC: tl.constexpr = ((1 << 32) + S - 1) // S
        page = ((safe.to(tl.int64) * MAGIC) >> 32).to(tl.int32)
        page -= (page.to(tl.int64) * S > safe).to(tl.int32)
    offset = safe - page * S
    indices = tl.arange(0, TABLE)
    mapping = tl.load(mapping_ptr + layer * B + indices, (indices < 2 * B) & (layer * B + indices < L * B), 0)
    physical = tl.gather(mapping, (tl.where(pos < N, 0, B) + page).to(tl.int32), 0)
    value = physical * S + offset
    tl.store(output_ptr + out, tl.where(slot < 0, -1, value.to(tl.int32)), out < L * N)


@triton.jit(do_not_specialize_on_alignment=["slots_ptr", "mapping_ptr"])
def _translate_direct_kernel(
    slots_ptr,
    mapping_ptr,
    output_ptr,
    N: tl.constexpr,
    B: tl.constexpr,
    L: tl.constexpr,
    S: tl.constexpr,
    TILE: tl.constexpr,
):
    out = tl.program_id(0).to(tl.int64) * TILE + tl.arange(0, TILE)
    slot = tl.load(slots_ptr + out % N, out < L * N, 0)
    safe = tl.maximum(slot, 0)
    if S & (S - 1) == 0:
        page = safe >> (S.bit_length() - 1)
    else:
        # ceil(2^32/S) overestimates the quotient by at most one for x<2^31.
        # The 64-bit product is below 2^63; correction makes divmod exact.
        MAGIC: tl.constexpr = ((1 << 32) + S - 1) // S
        page = ((safe.to(tl.int64) * MAGIC) >> 32).to(tl.int32)
        page -= (page.to(tl.int64) * S > safe).to(tl.int32)
    offset = safe - page * S
    physical = tl.load(mapping_ptr + (out // N) * B + page, out < L * N, 0)
    value = physical * S + offset
    tl.store(output_ptr + out, tl.where(slot < 0, -1, value.to(tl.int32)), out < L * N)


class _KernelLaunch:
    """Prepared runtime stub; retain tensor arguments for stream lifetime tracking."""

    def __init__(self, compiled: Any, grid: tuple[int, int, int], constants: tuple[int, ...]) -> None:
        self._compiled = compiled
        self._launch = compiled.run.launch
        self._grid = grid
        self._constants = constants

    def __call__(self, stream: int, *tensors: torch.Tensor) -> None:
        kernel = self._compiled
        enter_hook = knobs.runtime.launch_enter_hook
        exit_hook = knobs.runtime.launch_exit_hook
        metadata = None
        if enter_hook is not None:
            metadata = kernel.launch_metadata(self._grid, stream, *tensors, *self._constants)
        self._launch(
            *self._grid,
            stream,
            kernel.function,
            kernel.packed_metadata,
            metadata,
            enter_hook,
            exit_hook,
            *tensors,
            *self._constants,
        )


def _validate(inputs: Sequence[torch.Tensor], attributes: dict[str, int]) -> tuple[int, int, int, int]:
    if not isinstance(inputs, Sequence) or len(inputs) != 2:
        raise ValueError("expected slots and M")
    if not isinstance(attributes, dict) or set(attributes) != {"block_size"}:
        raise ValueError("attributes must contain only block_size")
    size = attributes["block_size"]
    if type(size) is not int or not 1 <= size <= 2147483647:
        raise ValueError("block_size must be an integer in [1,2147483647]")
    for tensor in inputs:
        if not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.int32:
            raise TypeError("inputs must be int32 tensors")
        if tensor.device.type != "npu" or tensor.device != inputs[0].device:
            raise ValueError("inputs must be on the same NPU")
        if not tensor.is_contiguous() or tensor.numel() > 2147483647:
            raise ValueError("inputs must be contiguous and fit int32 indexing")
    slots, mapping = inputs
    if slots.ndim != 1 or mapping.ndim not in (1, 2):
        raise ValueError("expected slots[N] and M[B] or M[L,B]")
    layers = 1 if mapping.ndim == 1 else mapping.shape[0]
    blocks = mapping.shape[-1]
    count = slots.numel()
    if layers < 1 or blocks < 1 or layers * count > 2147483647:
        raise ValueError("mapping dimensions must be positive and output must fit int32 indexing")
    # One-time value readback belongs to prepare, outside the measured hot path.
    cpu_slots, cpu_mapping = slots.cpu(), mapping.cpu()
    if count and int(cpu_slots.max()) // size >= blocks:
        raise ValueError("nonnegative logical page is outside M")
    if int(cpu_mapping.min()) < 0 or int(cpu_mapping.max()) * size + size - 1 > 2147483647:
        raise ValueError("mapping must be nonnegative and every page must fit int32 output")
    return count, blocks, layers, size


class TokenSlotAddressTranslation:
    """Fixed metadata; read current values and storage on every invocation."""

    def __init__(self, inputs: Sequence[torch.Tensor], attributes: dict[str, int]) -> None:
        count, blocks, layers, size = _validate(inputs, attributes)
        self._inputs = tuple(inputs)
        self._shape = (count,) if inputs[1].ndim == 1 else (layers, count)
        self._count = count
        if count == 0:
            return
        output = torch.empty(self._shape, dtype=torch.int32, device=inputs[0].device)
        # A two-row table plus live vectors must fit the 910B/A3 vector UB.
        table = triton.next_power_of_2(2 * blocks)
        if count >= 8 and 4 * table + 64 * min(2048, 1 << (count.bit_length() - 1)) <= 160 * 1024:
            tile = min(2048, 1 << (count.bit_length() - 1))
            constants = (count, blocks, layers, size, table, tile)
            kernel = _translate_table_kernel
        else:
            tile = 128
            constants = (count, blocks, layers, size, tile)
            kernel = _translate_direct_kernel
        grid = (triton.cdiv(layers * count, tile), 1, 1)
        compiled = kernel.warmup(*inputs, output, *constants, grid=grid)
        self._launch = _KernelLaunch(compiled, grid, constants)

    def run(self) -> torch.Tensor:
        """Return independent output on the caller's current NPU stream."""
        slots, mapping = self._inputs
        output = torch.empty(self._shape, dtype=torch.int32, device=slots.device)
        if self._count:
            stream = driver.active.get_current_stream(slots.device.index)
            self._launch(stream, slots, mapping, output)
        return output


def prepare(inputs: Sequence[torch.Tensor], attributes: dict[str, int]) -> TokenSlotAddressTranslation:
    """Validate once and compile a fixed-shape call object."""
    return TokenSlotAddressTranslation(inputs, attributes)


def run(inputs: Sequence[torch.Tensor], attributes: dict[str, int]) -> torch.Tensor:
    """Prepare and execute one call."""
    return prepare(inputs, attributes).run()
