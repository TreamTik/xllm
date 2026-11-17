# TokenOffsetTileFlags

Generate UINT8 flags for complete, aligned and consecutive token-offset tiles.
The implementation uses Triton Ascend and runs independently of the xLLM server.

## Environment and execution

Use Python >= 3.10, PyTorch, torch_npu and triton-ascend compatible with CANN 9.1.0.
The development environment uses Python 3.12.13, torch 2.7.1+cpu,
torch_npu 2.7.1.post8 and triton-ascend 3.6.0.post2.

```bash
source /usr/local/Ascend/cann-9.1.0/set_env.sh
source <venv>/bin/activate
export ASCEND_RT_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MAX_JOBS=2
export PYTHONPATH="$PWD"
export TRITON_CACHE_DIR=<independent-cache-directory>
python -m pytest --noconftest tests/python/test_token_offset_tile_flags.py -q
python examples/kernels/token_offset_tile_flags/verify.py \
  --task-package <extracted-task-package> --output-dir <results>
```

The first nonempty prepare compiles only the required Triton kernel into TRITON_CACHE_DIR.
There is no server build, C++ extension, ACLNN package or CANN installation change.
The verification driver imports the task package's generate_data and frozen golden/baseline.
Generated results must stay outside this source checkout.

```python
import torch
import torch_npu
from xllm.python.kernels_npu.triton.token_offset_tile_flags import prepare, run

inputs = [torch.tensor(x, dtype=torch.int32, device="npu:0")
          for x in [[8, 9, 10, 11, 9, 10, 11, 12], [0, 8], [8], [0, 2]]]
prepared = prepare(inputs, {"tile_size": 4, "max_tiles_per_seq": 2})
output = prepared.run()  # uint8 [1, 0]
```

## Contract

Inputs are contiguous same-device NPU INT32 vectors: offsets[N], indptr[R+1],
lengths[R] and tileptr[R+1]. Nonzero storage offsets are supported.
Attributes are exactly tile_size=T (integer 1..4096) and max_tiles_per_seq=K
(positive integer). Boolean attributes are rejected. indptr starts at zero,
ends at N and is nondecreasing. lengths are nonnegative and do not exceed their
request segment or K tiles. tileptr equals [0,K,...,R*K]. Offsets are nonnegative.
Every input and output element count fits INT32_MAX.

Output is contiguous UINT8[R*K]. A tile produces 1 only when it is complete,
its first offset is divisible by T, and every offset equals first+lane.
Padding, partial tiles and unused capacity produce 0. Empty offsets use a device
zero allocation without launching a kernel with empty input pointers.

prepare validates once, including CPU readback of input values. The hot path reads
current NPU inputs and the calling thread's current stream. Keep shape, dtype,
device and attributes fixed after prepare; updated values must remain valid.
Every call allocates independent output storage and retains no per-call workspace.
The prepared launch stub receives current Tensor objects and preserves Triton launch hooks
and allocator stream tracking. No input pointer or stream is cached.

## Validation scope

The driver covers all 14 task cases, 9 illegal inputs, current-value updates,
same-shape set_ rebinding, and both performance shapes' two-stream retained outputs,
peak active allocations and 20-call release checks for candidate and frozen baseline.
Performance includes complete prepared.run, allocation and final synchronization:
50 warmups, six alternating paired rounds, 200 calls per round, 10000 paired bootstrap
resamples with seed 20261031 and linear percentiles. Only the two main cases have
an 85 us target and benefit gates. T=255 and T=257 are report-only timing cases.

The formal target is Atlas A3 / CANN 9.1.0. Current measurements are Ascend 910B3
development validation; Atlas A3 functional, performance and memory acceptance is pending.
