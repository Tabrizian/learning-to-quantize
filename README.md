# NUQSGD: Improved Communication Efficiency for Data-parallel SGD via Nonuniform Quantization

Code for quantization methods from
**[NUQSGD: Improved Communication Efficiency for Data-parallel SGD via Nonuniform Quantization](https://arxiv.org/abs/1908.06077)**
*, A. Ramezani-Kebrya, F. Faghri, Roy D. M., arXiv preprint arXiv:1908.06077, 2019*



## Dependencies
We recommend using Anaconda to install the following packages,

* Python 3.7.1
* [PyTorch](http://pytorch.org/) (>1.1.0)

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

## Running using SLURM

```
python3 -m grid.cluster slurm 
```

## Reference

If you found this code useful, please cite the following paper:

    @misc{ramezanikebrya2019nuqsgd,
      title={{NUQSGD}: Improved Communication Efficiency for Data-parallel SGD 
      via Nonuniform Quantization},
      author={Ramezani-Kebrya, Ali and Faghri, Fartash and Roy, Daniel M.},
      url={https://github.com/fartashf/nuqsgd},
      archivePrefix={arXiv},
      eprint={1908.06077},
      primaryClass={cs},
      year={2019}
    }

## License

[Apache License 2.0](http://www.apache.org/licenses/LICENSE-2.0)
