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

"""Pack ragged logical page prefixes and translate per-layer page mappings."""

from __future__ import annotations

from collections.abc import Sequence
from functools import cache
from pathlib import Path
from typing import Any

import torch
import torch_npu
import triton
import triton.language as tl
from torch.utils.cpp_extension import load
from triton import knobs
from triton.runtime import driver


@triton.jit(do_not_specialize_on_alignment=["x_ptr", "m_ptr", "cu_ptr"])
def _fused_kernel(
    x_ptr,
    m_ptr,
    cu_ptr,
    out_ptr,
    R: tl.constexpr,
    C: tl.constexpr,
    B: tl.constexpr,
    L: tl.constexpr,
    P: tl.constexpr,
    XR: tl.constexpr,
    MR: tl.constexpr,
    CR: tl.constexpr,
    TILE: tl.constexpr,
):
    cv = tl.load(cu_ptr + tl.arange(0, CR), tl.arange(0, CR) <= R, 2147483647)
    xi = tl.arange(0, XR)
    row = tl.minimum(xi // C, R - 1)
    begin = tl.gather(cv, row, 0)
    end = tl.gather(cv, row + 1, 0)
    xv = tl.load(x_ptr + xi, (xi < R * C) & (xi % C < end - begin), 0)
    mi = tl.arange(0, MR)
    mv = tl.load(m_ptr + mi, mi < L * B, 0)
    q = tl.program_id(0) * TILE + tl.arange(0, TILE)
    pos = q % P
    lo = tl.full((TILE,), 0, tl.int32)
    hi = tl.full((TILE,), R, tl.int32)
    for _ in range((R + 1).bit_length()):
        mid = (lo + hi + 1) // 2
        before = tl.gather(cv, mid, 0) <= pos
        lo = tl.where(before, mid, lo)
        hi = tl.where(before, hi, mid - 1)
    offset = tl.gather(cv, lo, 0)
    logical = tl.gather(xv, lo * C + pos - offset, 0)
    address = tl.minimum(q // P, L - 1) * B + tl.maximum(logical, 0)
    physical = tl.gather(mv, address, 0)
    tl.store(out_ptr + q, tl.where(logical < 0, -1, physical), q < L * P)


@triton.jit(do_not_specialize_on_alignment=["x_ptr", "cu_ptr"])
def _pack_kernel(
    x_ptr,
    cu_ptr,
    ids_ptr,
    neg_ptr,
    R: tl.constexpr,
    C: tl.constexpr,
    P: tl.constexpr,
    TILE: tl.constexpr,
):
    # A program owns an aligned packed interval, including request boundaries.
    # No two programs write separate halves of the same 32-byte block.
    first = tl.program_id(0).to(tl.int64) * TILE
    stop = tl.minimum(first + TILE, P)
    low = tl.full((), 0, tl.int64)
    high = tl.full((), R, tl.int64)
    while low < high:
        mid = (low + high + 1) // 2
        boundary = tl.load(cu_ptr + mid)
        if boundary <= first:
            low = mid
        else:
            high = mid - 1
    row = low
    cursor = first
    lanes = tl.arange(0, TILE)
    while cursor < stop:
        begin = tl.load(cu_ptr + row)
        end = tl.load(cu_ptr + row + 1)
        length = tl.minimum(end, stop) - cursor
        logical = tl.load(x_ptr + row * C + cursor - begin + lanes, lanes < length, 0)
        tl.store(ids_ptr + cursor + lanes, tl.maximum(logical, 0).to(tl.int64), lanes < length)
        tl.store(neg_ptr + cursor + lanes, (logical < 0).to(tl.int32), lanes < length)
        cursor += length
        row += 1


@triton.jit(do_not_specialize_on_alignment=["x_ptr", "cu_ptr"])
def _pack_logical_kernel(
    x_ptr,
    cu_ptr,
    packed_ptr,
    R: tl.constexpr,
    C: tl.constexpr,
    P: tl.constexpr,
    TILE: tl.constexpr,
):
    # A program owns an aligned packed interval, including request boundaries.
    # No two programs write separate halves of the same 32-byte block.
    first = tl.program_id(0).to(tl.int64) * TILE
    stop = tl.minimum(first + TILE, P)
    low = tl.full((), 0, tl.int64)
    high = tl.full((), R, tl.int64)
    while low < high:
        mid = (low + high + 1) // 2
        boundary = tl.load(cu_ptr + mid)
        if boundary <= first:
            low = mid
        else:
            high = mid - 1
    row = low
    cursor = first
    lanes = tl.arange(0, TILE)
    while cursor < stop:
        begin = tl.load(cu_ptr + row)
        end = tl.load(cu_ptr + row + 1)
        length = tl.minimum(end, stop) - cursor
        logical = tl.load(x_ptr + row * C + cursor - begin + lanes, lanes < length, 0)
        tl.store(packed_ptr + cursor + lanes, logical, lanes < length)
        cursor += length
        row += 1


@triton.jit(do_not_specialize_on_alignment=["m_ptr"])
def _map_kernel(
    packed_ptr,
    m_ptr,
    out_ptr,
    P: tl.constexpr,
    B: tl.constexpr,
    L: tl.constexpr,
    TABLE: tl.constexpr,
    TILE: tl.constexpr,
):
    lane = tl.arange(0, TILE)
    first = tl.program_id(0).to(tl.int64) * TILE
    q = first + lane
    layer = first // P
    pos = first % P + lane
    left = tl.load(packed_ptr + pos, (q < L * P) & (pos < P), 0)
    right = tl.load(packed_ptr + pos - P, (q < L * P) & (pos >= P), 0)
    logical = tl.where(pos < P, left, right)
    mi = tl.arange(0, TABLE)
    mapping = tl.load(m_ptr + layer * B + mi, (mi < 2 * B) & (layer * B + mi < L * B), 0)
    index = tl.where(pos < P, 0, B) + tl.maximum(logical, 0)
    result = tl.gather(mapping, index, 0)
    tl.store(out_ptr + q, tl.where(logical < 0, -1, result), q < L * P)


@triton.jit
def _finish_kernel(out_ptr, neg_ptr, P: tl.constexpr, N: tl.constexpr, TILE: tl.constexpr):
    lane = tl.arange(0, TILE)
    first = tl.program_id(0).to(tl.int64) * TILE
    q = first + lane
    if P < TILE:
        pos = tl.arange(0, triton.next_power_of_2(P))
        flags = tl.load(neg_ptr + pos, pos < P, 0)
        negative = tl.gather(flags, q % P, 0)
    else:
        # Split at the layer boundary: both GM reads are contiguous.
        # A vector q % P pointer would lower to expensive scalar GM gathers.
        pos = first % P + lane
        left = tl.load(neg_ptr + pos, (q < N) & (pos < P), 0)
        right = tl.load(neg_ptr + pos - P, (q < N) & (pos >= P), 0)
        negative = tl.where(pos < P, left, right)
    value = tl.load(out_ptr + q, q < N, 0)
    tl.store(out_ptr + q, tl.where(negative != 0, -1, value), q < N)


class _KernelLaunch:
    """Prepared Ascend launch stub; tensor objects retain allocator stream tracking."""

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


@cache
def _native_extension() -> Any:
    root = Path(__file__).resolve().parents[4]
    return load(
        name="xllm_packed_page_indices",
        sources=[str(root / "xllm/core/kernels/npu/xllm_ops/packed_page_indices.cpp")],
        extra_cflags=["-O3"],
    )


def _validate(inputs: Sequence[torch.Tensor], attributes: dict[str, int]) -> dict[str, Any]:
    if not isinstance(inputs, Sequence) or len(inputs) != 3:
        raise ValueError("expected X, M and cu")
    if not isinstance(attributes, dict) or attributes:
        raise ValueError("attributes must be an empty dict")
    for tensor in inputs:
        if not isinstance(tensor, torch.Tensor) or tensor.dtype != torch.int32:
            raise TypeError("inputs must be int32 tensors")
        if tensor.device.type != "npu" or tensor.device != inputs[0].device:
            raise ValueError("inputs must be on the same NPU")
        if not tensor.is_contiguous() or tensor.numel() > 2147483647:
            raise ValueError("inputs must be contiguous and fit int32 indexing")
    x, mapping, cu = inputs
    if x.ndim != 2 or mapping.ndim not in (1, 2):
        raise ValueError("expected X[R,C] and M[B] or M[L,B]")
    if mapping.shape[-1] < 1 or (mapping.ndim == 2 and mapping.shape[0] < 1):
        raise ValueError("mapping dimensions must be positive")
    if cu.ndim != 1 or cu.numel() != x.shape[0] + 1:
        raise ValueError("cu must have R+1 elements")
    # Validation readbacks are outside the prepared hot path. Do not inspect tails.
    cv = cu.cpu().tolist()
    if cv[0] != 0 or any(a > b or b - a > x.shape[1] for a, b in zip(cv, cv[1:])):
        raise ValueError("cu must start at zero with counts in [0,C]")
    layers = 1 if mapping.ndim == 1 else mapping.shape[0]
    if layers * cv[-1] > 2147483647:
        raise ValueError("output exceeds int32 indexing")
    mv = mapping.cpu().tolist()
    rows = [mv] if mapping.ndim == 1 else mv
    if any(value < 0 for row in rows for value in row):
        raise ValueError("mapping values must be nonnegative")
    active = [x[r, : b - a].cpu().tolist() for r, (a, b) in enumerate(zip(cv, cv[1:]))]
    if any(value >= mapping.shape[-1] for row in active for value in row):
        raise ValueError("valid logical pages must be less than B")
    return {"values": [active, mv, cv], "layers": layers, "blocks": mapping.shape[-1], "single": mapping.ndim == 1}


class PackedPageIndices:
    """Prepared fixed-shape call; tensor contents and same-shape storage may change."""

    def __init__(self, inputs: Sequence[torch.Tensor], attributes: dict[str, int]) -> None:
        self.info = _validate(inputs, attributes)
        self.inputs = tuple(inputs)
        self.attributes = dict(attributes)
        self.n = self.info["values"][2][-1]
        self.shape = (self.n,) if self.info["single"] else (self.info["layers"], self.n)
        self._x_elements = triton.next_power_of_2(inputs[0].numel())
        self._m_elements = triton.next_power_of_2(inputs[1].numel())
        # Bound the on-chip tables and live temporaries, independently of case IDs.
        self._fused = (
            self._x_elements <= 1024
            and 24 * self._x_elements
            + 8 * self._m_elements
            + 12 * triton.next_power_of_2(inputs[0].shape[0] + 1)
            + 16384
            <= 160 * 1024
        )
        self._mapped = self.n >= 256 and 16 * triton.next_power_of_2(self.info["blocks"]) + 16384 <= 160 * 1024
        self._native = None if self.n == 0 else _native_extension()
        if self.n == 0:
            return
        r, c = inputs[0].shape
        layers, blocks = self.info["layers"], self.info["blocks"]
        if self._fused:
            cores = driver.active.utils.get_device_properties(inputs[0].device.index)["num_vectorcore"]
            tile = max(256, min(1024, triton.next_power_of_2(triton.cdiv(layers * self.n, cores))))
            self._fused_args = (
                r,
                c,
                blocks,
                layers,
                self.n,
                self._x_elements,
                self._m_elements,
                triton.next_power_of_2(r + 1),
                tile,
            )
            grid = (triton.cdiv(layers * self.n, tile), 1, 1)
            output = torch.empty(self.shape, dtype=torch.int32, device=inputs[0].device)
            compiled = _fused_kernel.warmup(*inputs, output, *self._fused_args, grid=grid)
            self._launch_fused = _KernelLaunch(compiled, grid, self._fused_args)
        elif self._mapped:
            self._pack_args = (r, c, self.n, 256)
            map_tile = min(1024, 1 << (self.n.bit_length() - 1))
            self._map_args = (self.n, blocks, layers, triton.next_power_of_2(2 * blocks), map_tile)
            packed = torch.empty((self.n,), dtype=torch.int32, device=inputs[0].device)
            output = torch.empty(self.shape, dtype=torch.int32, device=inputs[0].device)
            grid = (triton.cdiv(self.n, 256), 1, 1)
            compiled = _pack_logical_kernel.warmup(inputs[0], inputs[2], packed, *self._pack_args, grid=grid)
            self._launch_pack = _KernelLaunch(compiled, grid, self._pack_args)
            grid = (triton.cdiv(layers * self.n, map_tile), 1, 1)
            compiled = _map_kernel.warmup(packed, inputs[1], output, *self._map_args, grid=grid)
            self._launch_map = _KernelLaunch(compiled, grid, self._map_args)
        else:
            self._pack_args = (r, c, self.n, 256)
            self._finish_args = (self.n, layers * self.n, 1024)
            ids = torch.empty((self.n,), dtype=torch.int64, device=inputs[0].device)
            negative = torch.empty((self.n,), dtype=torch.int32, device=inputs[0].device)
            output = torch.empty(self.shape, dtype=torch.int32, device=inputs[0].device)
            grid = (triton.cdiv(self.n, 256), 1, 1)
            compiled = _pack_kernel.warmup(inputs[0], inputs[2], ids, negative, *self._pack_args, grid=grid)
            self._launch_pack = _KernelLaunch(compiled, grid, self._pack_args)
            grid = (triton.cdiv(layers * self.n, 1024), 1, 1)
            compiled = _finish_kernel.warmup(output, negative, *self._finish_args, grid=grid)
            self._launch_finish = _KernelLaunch(compiled, grid, self._finish_args)

    def run(self) -> torch.Tensor:
        """Read current inputs on the current stream and return independent storage."""
        x, mapping, cu = self.inputs
        if self.n == 0:
            return torch.empty(self.shape, dtype=torch.int32, device=x.device)
        stream = driver.active.get_current_stream(x.device.index)
        layers, blocks = self.info["layers"], self.info["blocks"]
        if self._fused:
            return self._native.run_fused(
                x, mapping, cu, self._launch_fused, stream, self.n, layers, self.info["single"]
            )
        if self._mapped:
            return self._native.run_mapped(
                x, mapping, cu, self._launch_pack, self._launch_map, stream, self.n, layers, self.info["single"]
            )
        return self._native.run(
            x, mapping, cu, self._launch_pack, self._launch_finish, stream, self.n, layers, blocks, self.info["single"]
        )


def prepare(inputs: Sequence[torch.Tensor], attributes: dict[str, int]) -> PackedPageIndices:
    """Validate once and prepare the fixed-shape operator."""
    return PackedPageIndices(inputs, attributes)


def run(inputs: Sequence[torch.Tensor], attributes: dict[str, int]) -> torch.Tensor:
    """Validate and execute one call."""
    return prepare(inputs, attributes).run()
