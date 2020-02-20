# Learning to Quantize

Code for the ICML submission 6132.



## Dependencies
We recommend using Anaconda to install the following packages,

* Python 3.7.1
* [PyTorch](http://pytorch.org/) (>1.1.0)
* TensorboardX
* Pyyaml

```
conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
```

## Cuda kernel installation

```
cd nuq/cuda/;
python setup.py install
cd ../../
```

## Running experiments in the paper
The commands used to run the experiments can be found in `pjobs/` directory.
These commands are generated using the `grid_run.py` script. Each experiment is 
described using a function in the `grid/nuq.py` file.

Each of the experiments can be run using `bash pjobs/supp_{i}.sh` where `i` is
the experiment number being used. The small configuration changes in each
experiment corresponds to variations in the quantization method used for
generating the plots.

After running the experiments, the results will be stored in a directory
specified in the configuration of the job that can be served using `tensorboard`
to view the results. 



## Quantization
Quantization methods are implemented in NumPy (`nuq/quantize.py`) as well is in 
Cuda (`nuq/cuda/src/ops_gpu.cu`).

[Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0)
