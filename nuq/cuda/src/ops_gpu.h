#include <ATen/cuda/CUDAContext.h>
#include <torch/extension.h>

#include <curand_kernel.h>
#include "src/ops_gpu.cuh"

template <typename Dtype>
class QDQ {
private:
  at::Tensor levels;
public:
  QDQ(at::Tensor levels);
  void qdqGPU(at::Tensor in_vector, at::Tensor norm, at::Tensor out_vector, at::Tensor rand_vector);
};

template class QDQ<float>;
