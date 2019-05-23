import os
from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

os.system('make -j%d' % os.cpu_count())

# Python interface
setup(
    name='CuQuantize',
    version='0.1.0',
    install_requires=['torch'],
    packages=['cuquant'],
    package_dir={'cuquant': './'},
    ext_modules=[
        CUDAExtension(
            name='cuquant_back',
            include_dirs=['./'],
            sources=[
                'pybind/bind.cpp',
            ],
            libraries=['cuquant'],
            library_dirs=['objs'],
            # extra_compile_args=['-g']
        )
    ],
    cmdclass={'build_ext': BuildExtension},
    author='Fartash Faghri',
    author_email='faghri@cs.toronto.edu',
    description='Quantize-Dequantize cuda kernel',
    zip_safe=False,
)
