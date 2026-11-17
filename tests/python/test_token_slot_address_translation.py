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
"""Task-contract regression for token slot translation; requires one NPU."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch
import torch_npu

_EXAMPLE = Path(__file__).resolve().parents[2] / "examples/kernels/token_slot_address_translation"
sys.path.insert(0, str(_EXAMPLE))
_SPEC = importlib.util.spec_from_file_location("_token_slot_verify", _EXAMPLE / "verify.py")
_VERIFY = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_VERIFY)


def test_task_contract() -> None:
    """Check exact independent golden, errors, mutation, streams and allocation release."""
    result = _VERIFY.functional()
    assert len(result["cases"]) == 14
    assert len(result["negative"]) == 10
    assert all(case["status"] == "PASS" for case in result["cases"])
    assert set(result["engineering_by_case"]) == {"slot_perf_small", "slot_perf_large"}
