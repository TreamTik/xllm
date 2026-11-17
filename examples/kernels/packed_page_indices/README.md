# PackedPageIndices

Pack valid ragged page-table prefixes and translate logical pages to per-layer physical pages.
The operator uses Triton Ascend kernels and a small host C++ extension. It does not load a model.

## Build and run

Use a source checkout with Python >= 3.10, a C++17 compiler, Ninja, PyTorch,
torch_npu and a matching triton-ascend installation for CANN 9.1.0.
The development environment uses Python 3.12, PyTorch 2.7.1, torch_npu 2.7.1.post8,
and triton-ascend 3.6.0.post2 (Triton runtime 3.6.0).

```bash
source /usr/local/Ascend/cann-9.1.0/set_env.sh
source <venv>/bin/activate
export ASCEND_RT_VISIBLE_DEVICES=0
export PYTHONPATH="$PWD"
export MAX_JOBS=2
export TORCH_EXTENSIONS_DIR=<build-directory>
python -m pip install ninja
python -m pytest --noconftest tests/python/test_packed_page_indices.py -q
```

The standalone kernel tests use --noconftest to avoid the server extension initialization
in tests/python/conftest.py. They do not use shared server fixtures.

The first nonempty prepare compiles this operator's C++ extension in
TORCH_EXTENSIONS_DIR and compiles the required Triton kernels in the Triton cache.
It does not build the xLLM server or modify the CANN installation.
Run from this source checkout; the JIT build needs the accompanying C++ source.

```python
import torch
import torch_npu
from xllm.python.kernels_npu.triton.packed_page_indices import prepare, run

x = torch.tensor([[0, -7, 999999], [1, 2, 999999]], dtype=torch.int32, device="npu:0")
m = torch.tensor([[3, 5, 7]], dtype=torch.int32, device="npu:0")
cu = torch.tensor([0, 2, 4], dtype=torch.int32, device="npu:0")
prepared = prepare([x, m, cu], {})
output = prepared.run()  # [[3, -1, 5, 7]]
```

Inputs must be contiguous INT32 tensors on the same NPU; nonzero storage offsets are supported.
X is [R,C], M is [B] or [L,B], and cu is [R+1].
Only each row's valid prefix is checked and read. Mapping values are nonnegative.
Negative logical pages produce -1, including negative values other than -1.
All input and output element counts must fit INT32_MAX.

After prepare, keep shape, dtype, device and total packed count P fixed.
X, M and cu may change contents or bind to new same-shape storage using set_.
Each call reads current data on the calling thread's current stream and allocates an independent output.
The fast launch wrapper uses the compiled Ascend launch stub, passes Tensor objects for stream tracking,
and preserves Triton launch hooks. No input pointers or streams are cached.

## Paths and validation

Small logical tables use one fused kernel. Larger ragged inputs first produce packed logical pages,
then translate with an on-chip table covering at most two mapping rows.
Mappings exceeding the on-chip budget use native NPU index_select with negative-page finalization.
C++ owns per-call allocation and dispatch. The paths depend on geometry and memory budget.

The task package supplies the independent scalar oracle and the full functional driver:

```bash
python <task-package>/golden.py --verify-functional \
  --implementation examples/kernels/packed_page_indices \
  --output <results>/functional.json
```

The adjacent solution.py is the standalone task entry. The repository tests additionally exercise
all three dispatch paths, mutable counts, INT32_MAX, storage rebinding and independent streams.
Performance uses complete prepared.run calls, fresh output allocation and final synchronization.
Keep logs, raw paired measurements and self-validation reports outside the source tree.

Target acceptance is Atlas A3 with CANN 9.1.0. Current 910B development measurements do not replace A3 acceptance.
