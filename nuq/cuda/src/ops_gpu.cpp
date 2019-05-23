#include "src/ops_gpu.h"

template <typename Dtype>
QDQ<Dtype>::QDQ(at::Tensor levels) : levels(levels){
}

template <typename Dtype>
void QDQ<Dtype>::qdqGPU(at::Tensor in_vector, at::Tensor norm, at::Tensor out_vector, at::Tensor rand_vector) {
  int N = in_vector.numel();
  int num_levels = levels.numel();

  qdqGPUKernel(
          in_vector.data<Dtype>(),
          norm.data<Dtype>(),
          out_vector.data<Dtype>(),
          N,
          levels.data<Dtype>(), num_levels,
          rand_vector.data<long>(),
          at::cuda::getCurrentCUDAStream());
}
