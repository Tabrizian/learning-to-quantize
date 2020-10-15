# Adaptive Gradient Quantization for Data-Parallel SGD 

Code for "Adaptive Gradient Quantization for Data-Parallel SGD"

## Dependencies
We recommend using Anaconda to install the following packages,

* Python 3.7.1
* [PyTorch](http://pytorch.org/) **(=1.0.1)**
* TensorboardX
* Pyyaml
* Scipy

```
conda install pytorch==1.0.1 torchvision==0.2.2 -c pytorch
pip install pyyaml scipy tensorboardx
```

## Cuda kernel installation

It is also required to install the CUDA kernel below for the quantization.

```
cd nuq/cuda/;
python setup.py install
cd ../../
```

## Usage

There are examples for training `CIFAR10` and `ImageNet` datasets in the `pjobs` folder. For a description about what each of the flags do, please refer to the [args.py](./args.py).

Each file in the pjobs folder contains all of the experiments for a single task. This will create tensorboard files for the experiments in the paper. You can use the [figs_nuq](./notebooks/figs_nuq.ipynb) notebook to generate the graphs from the paper.

## Generating Different Set of Experiments

By changing the [grid/nuq.py](./grid/nuq.py) you can create a different set of experiments.

```bash
grid_run.py --prefix exp --cluster slurm --run_name cifar10_full_resnet32 --grid nuq --cluster_args 38,1,p100
```

This generates a set of experiments prefixed with `exp` and generates the required `sbatch` file for submitting to SLURM in the `jobs/exp_slurm.sbatch`. It may require minor changes to suit your environment. About the `cluster_args`, the first parameter shows the total number of bash scripts to generate that may contain multiple experiments. Generated bash scripts will be located in `jobs/exp_{0-37}`.

`--grid nuq` will use the file [grid/nuq.py](./grid/nuq.py) to generate the experiments and `--run_name cifar10_full_resnet32` will use the set of parameters described in the `cifar10_full_resnet32` function for experiments.

## Important Files and General Overview

The quantization is performed in three major steps. First, calculating the statistics for the gradient distribution by sampling the gradient. Second, instantiating the distribution object (e.g. for norm-based methods instantiating the `CondTruncNormalHist` class and for the norm-less methods instantiating the `TruncNorm`). Third, updating the levels according to the specified method.

This functionality is mainly implemented in three files:

### [estim/dist.py](estim/dist.py)

This file implements the necessary distribution classes. Two of the classes are used throughout the project:

1. `TruncNorm`

This creates a truncated normal distribution using the mean and sigma provided.

2. `CondTruncNormHist`

This class requires a list of norms, means, and sigmas which are the statistics for individual buckets of the gradient. Then
by creating buckets it approximates the distribution generated by the weighted sum of the Truncated Normal distributions.

### [nuq/quantize.py](nuq/quantize.py)

This is the file that implements different quantization schemes. ALQ is implemented in [alq](https://github.com/Tabrizian/learning-to-quantize/blob/04467ce2afd7ffb62624337c2068efbaf59da7ea/nuq/quantize.py#L179) function. The AMQ variations are implemented in [this](https://github.com/Tabrizian/learning-to-quantize/blob/04467ce2afd7ffb62624337c2068efbaf59da7ea/nuq/quantize.py#L307) and [this](https://github.com/Tabrizian/learning-to-quantize/blob/04467ce2afd7ffb62624337c2068efbaf59da7ea/nuq/quantize.py#L138) line.

Other important function is [update_levels](https://github.com/Tabrizian/learning-to-quantize/blob/04467ce2afd7ffb62624337c2068efbaf59da7ea/nuq/quantize.py#L307). This function updates the gradient levels at the specified interations using the appropriate quantization scheme. We try various initializations for AMQ and ALQ to make sure that we have found the best set of levels. The proxy metric that we use to estimate quantization error is the Variance error which is calculated on the given distribution. This is calculated in the [Distribution](https://github.com/Tabrizian/nuqsgd/blob/0cdc534b527e3de3780993b2f8a8609bf9f70520/estim/dist.py#L57) class.

### [estim/gestim.py](estim/gestim.py)

The important function in this file is [snap_online_mean](https://github.com/Tabrizian/learning-to-quantize/blob/04467ce2afd7ffb62624337c2068efbaf59da7ea/estim/gestim.py#L78) function. This is the function that calculates norms, means, and sigmas of the individual buckets of the gradient samples. At the end of the function it selects the stats with the largest norms that contribute most to the aggregated distribution.

This function also needs to calculate the mean variance and mean of the gradient samples for the norm-less methods.

[Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0)
