# TokenSlotAddressTranslation

Translate logical token slots to per-layer physical KV-cache slots with Triton Ascend.
For S=128, slot=130 and M[1]=5, the result is 642. Negative slots produce -1.

## Run

Use Python >=3.10, PyTorch, torch_npu and triton-ascend compatible with CANN 9.1.0.
No model, ACLNN installation, C++ build or server wheel is required. The first prepare
compiles only this operator's Triton kernel into the caller's Triton cache.

```bash
source /usr/local/Ascend/cann-9.1.0/set_env.sh
source <venv>/bin/activate
export ASCEND_RT_VISIBLE_DEVICES=0 OMP_NUM_THREADS=1 MAX_JOBS=2
export PYTHONPATH="$PWD"
python examples/kernels/token_slot_address_translation/verify.py \
  --mode all --output-dir <results>
python -m pytest --noconftest tests/python/test_token_slot_address_translation.py -q
```

Use a results directory outside the source tree. `verify.py` returns nonzero for any
functional error or failed main performance gate. It saves result.json, generalization.json
and error details. Redirect stdout/stderr to preserve complete logs.
The fixture generator and embedded scalar oracle/NPU baseline come from task 5;
wrapper formatting and annotations were normalized without changing the reference bodies.
case.json retains the original expected checks and explicitly labeled historical reference data.
Only newly generated result files represent this implementation's measurements.

```python
import torch
import torch_npu
from solution import prepare  # from this example directory

slots = torch.tensor([-1, 130], dtype=torch.int32, device="npu:0")
mapping = torch.tensor([3, 5], dtype=torch.int32, device="npu:0")
prepared = prepare([slots, mapping], {"block_size": 128})
output = prepared.run()  # [-1, 642]
```

## Contract and implementation

Inputs are contiguous INT32 tensors on one NPU: slots[N] and M[B] or M[L,B].
The output is a fresh contiguous INT32 tensor, [N] for one-dimensional M or [L,N] otherwise.
N may be zero; L and B must be positive. `block_size` is an integer in [1,INT32_MAX], excluding bool.
prepare validates logical pages and requires every entire physical page to fit INT32 output.
All input and output element counts fit INT32_MAX. Nonzero storage offsets are supported.

Keep shape, dtype, device and block_size fixed after prepare. Legal in-place value updates
and same-shape set_ storage rebinding are supported. prepared.run reads current values
on the current stream; it never caches an input pointer, stream or output.
The fast launch stub preserves Tensor lifetime tracking and Triton launch hooks.
It uses the Triton Ascend compiled runtime launch API; the tested version is recorded in reports.

The table path loads at most two mapping rows into on-chip memory and fuses exact integer
page division, remainder, gather and address synthesis. The output is split into aligned
flat intervals, including layer boundaries. Tiny inputs or mappings outside the on-chip
budget use a direct lookup kernel. Empty output launches no kernel. No per-call workspace
is allocated beyond the output.

## Validation protocol

The full driver checks 14 supplied positive cases, 10 negative cases, high integer boundaries,
non-power-of-two page sizes, mutation, offset-1 rebinding, independent streams and allocation release.
Both main performance inputs retain three independent outputs on each of two streams and
check release after 20 calls. Peak allocation is measured after warmup using PyTorch's
active tensor allocator; it is not total HBM or all runtime scratch.

Performance uses complete prepared.run wall time, fresh output allocation and final device
synchronization. Each implementation warms up 50 times; six rounds alternate baseline/candidate
order, with 200 calls per round. The paired 95% bootstrap CI uses 10,000 common resamples,
seed 20261031 and linear percentiles. Main gates are positive mean/median savings,
a positive CI lower bound, and candidate mean <=85/155 us. The S=64/96 diagnostics report
paired timings without main-case gates.

Target acceptance is Atlas A3 with CANN 9.1.0. Ascend910B3 results are development evidence;
A3 acceptance is pending. The task-required materials are placed in task_submission/ and task_submission.zip.
