#include <string>

#include <torch/extension.h>

#include "src/ops_gpu.h"

namespace py = pybind11;

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m){
  py::class_<QDQ<float>>(m, "QDQ")
      .def(py::init<at::Tensor>())
      .def("qdqGPU", &QDQ<float>::qdqGPU);
}
