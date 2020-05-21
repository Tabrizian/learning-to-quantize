# Learning to Quantize



## Dependencies
We recommend using Anaconda to install the following packages,

* Python 3.7.1
* [PyTorch](http://pytorch.org/) (=1.3.1)
* TensorboardX
* Pyyaml

```
conda install pytorch==1.3.1 torchvision= cudatoolkit=10.1 -c pytorch
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

Each file in the pjobs folder contains all of the experiments for a single job.

## Generating Different Set of Experiments

By changing the [grid/nuq.py](./grid/nuq.py) you can create a different set of experiments.

```bash
grid_run.py --prefix exp --cluster slurm --run_name cifar10_full_resnet32 --grid nuq --cluster_args 38,1,p100```
```

This generates a set of experiments prefixed with `exp` and generates the required `sbatch` file for submitting to SLURM in the `jobs/exp_slurm.sbatch`. It may require minor changes to suit your environment. About the `cluster_args`, the first parameter shows the total number of bash scripts to generate thay may contain multiple experiments. Generated bash scripts will be located in `jobs/exp_{0-37}`.

`--grid nuq` suggests to use the experiments described in [grid/nuq.py](./grid/nuq.py) and `--run_name cifar10_full_resnet32` will use the set of parameters described in the `cifar10_full_resnet32` function.


[Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0)
