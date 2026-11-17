/* Copyright 2026 The xLLM Authors.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    https://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/
#include <torch/extension.h>

#include <cstdint>

namespace {
torch::Tensor gather(const torch::Tensor& mapping,
                     const torch::Tensor& indices,
                     int64_t layers,
                     int64_t blocks) {
  return mapping.reshape({layers, blocks}).index_select(/*dim=*/1, indices);
}
torch::Tensor run(const torch::Tensor& x,
                  const torch::Tensor& mapping,
                  const torch::Tensor& cu,
                  const pybind11::function& pack,
                  const pybind11::function& finish,
                  int64_t stream,
                  int64_t count,
                  int64_t layers,
                  int64_t blocks,
                  bool single) {
  torch::Tensor indices =
      torch::empty({count}, x.options().dtype(torch::kInt64));
  torch::Tensor negative = torch::empty({count}, x.options());
  pack(stream, x, cu, indices, negative);
  torch::Tensor output = gather(mapping, indices, layers, blocks);
  finish(stream, output, negative);
  return single ? output.reshape({count}) : output;
}
torch::Tensor run_mapped(const torch::Tensor& x,
                         const torch::Tensor& mapping,
                         const torch::Tensor& cu,
                         const pybind11::function& pack,
                         const pybind11::function& map,
                         int64_t stream,
                         int64_t count,
                         int64_t layers,
                         bool single) {
  torch::Tensor packed = torch::empty({count}, x.options());
  torch::Tensor output = torch::empty({layers, count}, x.options());
  pack(stream, x, cu, packed);
  map(stream, packed, mapping, output);
  return single ? output.reshape({count}) : output;
}
torch::Tensor run_fused(const torch::Tensor& x,
                        const torch::Tensor& mapping,
                        const torch::Tensor& cu,
                        const pybind11::function& launch,
                        int64_t stream,
                        int64_t count,
                        int64_t layers,
                        bool single) {
  torch::Tensor output = torch::empty({layers, count}, x.options());
  launch(stream, x, mapping, cu, output);
  return single ? output.reshape({count}) : output;
}
}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def(
      "gather", &gather, pybind11::call_guard<pybind11::gil_scoped_release>());
  module.def("run", &run);
  module.def("run_mapped", &run_mapped);
  module.def("run_fused", &run_fused);
}
