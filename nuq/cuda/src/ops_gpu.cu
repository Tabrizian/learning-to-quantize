
#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <curand.h>
#include "gpu.cuh"


//for QSGD: we should pass norm2 of the bucket and levels_uni,
// for NUQSGD: we should pass norm2 of the bucket and levels_exp,
// for QSGD-inf: we should pass maximum value of the bucket and levels_uni,
constexpr float EPS = 1e-7;

__global__ void _qdq(const float *in_vector, const float *norm, float
    *out_vector, const int n, const float *levels, const int num_levels, long
    *rand_vector)
{
    CUDA_KERNEL_LOOP(i, n) {
        int j = 0;
        float level_up, diff;
        while (j+1 < num_levels) 
        { 
            level_up =  levels[j+1];
            if (in_vector[i]/(norm[i]+EPS)<=level_up)
            {
                diff = level_up - levels[j];	
                if (in_vector[i]/(norm[i]+EPS)+diff*(rand_vector[i]%1000001 / 1000000.0)>level_up)
                {
                    j = j+1;
                }
                break;
            }
            j = j+1;			
        }
        out_vector[i] = norm[i]*levels[j];	        
    }
}


void qdqGPUKernel(float *in_vector, float *norm, float *out_vector, int n,
        float *levels, int num_levels, long * rand_vector, cudaStream_t
        stream)
{
    _qdq<<<GET_BLOCKS(n), CUDA_NUM_THREADS, 0, stream>>>(in_vector, norm,
            out_vector, n, levels, num_levels, rand_vector);
    cudaStreamSynchronize(stream);
    
}
